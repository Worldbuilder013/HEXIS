"""Sealed tests for env (.env discovery/parsing) and llm_client (OpenAI-compatible endpoint).

**No network at all**: every HTTP request goes through :class:`httpx.MockTransport`, and the backoff sleep is
replaced by a recording function, so the whole file runs in milliseconds. What is pinned here is behavior that
real endpoints taught us the hard way -- inline ``<think>`` must be stripped, a budget eaten up by reasoning must
be rescued exactly once, which status codes are worth retrying, and **the API key must never appear anywhere**.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import httpx
import pytest

from hexis.llm import env as envmod
from hexis.llm.llm_client import (
    Completion,
    LLMError,
    LLMHTTPError,
    ModelAdapter,
    OpenAIClient,
    _split_inline_thinking,
    extract_json,
)
from hexis.llm.model_iface import Model, ScriptedModel

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-fake-secret-abcd1234"


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #
def chat_response(content: str, *, finish_reason: str = "stop",
                  reasoning_content: str | None = None,
                  usage: dict | None = None) -> dict:
    """A /chat/completions response body with a realistic shape."""
    msg: dict = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    body: dict = {"id": "chatcmpl-abc123", "model": "MiniMax-M2.5-highspeed",
                  "choices": [{"index": 0, "message": msg,
                               "finish_reason": finish_reason}]}
    if usage is not None:
        body["usage"] = usage
    return body


def make_client(responses, **kw):
    """A client that answers with ``responses`` in order. Returns ``(client, seen, sleeps)``.

    Each item of ``responses`` is a ``(status, body)``, an ``httpx.Response``, or an exception to raise; once
    exhausted, the last item repeats. ``seen`` collects each request's payload, ``sleeps`` each backoff duration.
    """
    seen: list[dict] = []
    sleeps: list[float] = []
    seq = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode("utf-8")))
        item = seq[min(len(seen) - 1, len(seq) - 1)]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item
        status, body = item
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    client = OpenAIClient("MiniMax-M2.5-highspeed", "https://api.example/v1", FAKE_KEY,
                          transport=httpx.MockTransport(handler), **kw)
    client._sleep = sleeps.append          # backoff does not really sleep: tests must run in milliseconds
    return client, seen, sleeps


# --------------------------------------------------------------------------- #
# env: discovering and parsing .env
# --------------------------------------------------------------------------- #
def test_find_env_stops_at_the_git_holder(tmp_path):
    """The search stops at the directory holding .git -- a .env outside the checkout must never be picked up."""
    (tmp_path / "stray.env").write_text("X=1", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=outside-the-checkout", encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    deep = repo / "pkg" / "sub"
    deep.mkdir(parents=True)

    assert envmod.find_env_file(deep) is None          # no .env in the repository -> do not look outside

    (repo / ".env").write_text("MODEL=m", encoding="utf-8")
    assert envmod.find_env_file(deep) == repo / ".env"  # within one directory .env is checked before .git


def test_find_env_accepts_a_file_as_start(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".env").write_text("MODEL=m", encoding="utf-8")
    f = repo / "pkg" / "mod.py"
    f.parent.mkdir(parents=True)
    f.write_text("# when the start is a file, the search begins from its directory", encoding="utf-8")
    assert envmod.find_env_file(f) == repo / ".env"


def test_find_env_default_start_is_this_checkout():
    """The default start is where the package itself lives; if this repository root has a .env it is found (its values are not read, let alone printed)."""
    found = envmod.find_env_file()
    if (ROOT / ".env").exists():
        assert found == ROOT / ".env"
    else:                                              # on CI, injecting environment variables is also legitimate
        assert found is None


def test_load_env_parses_and_does_not_override(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(["# comment line", "", "MODEL=MiniMax-M2.5-highspeed",
                   'BASE_URL="https://api.example/v1"', "API_KEY='sk-quoted'",
                   "a line without an equals sign"]),
        encoding="utf-8")
    fake_env = {"MODEL": "already-set"}
    parsed = envmod.load_env(path=p, environ=fake_env)

    assert parsed == {"MODEL": "MiniMax-M2.5-highspeed",
                      "BASE_URL": "https://api.example/v1",
                      "API_KEY": "sk-quoted"}
    assert fake_env["MODEL"] == "already-set"          # no override by default
    assert fake_env["BASE_URL"] == "https://api.example/v1"

    envmod.load_env(path=p, environ=fake_env, override=True)
    assert fake_env["MODEL"] == "MiniMax-M2.5-highspeed"


def test_load_env_missing_file_is_not_an_error(tmp_path):
    assert envmod.load_env(path=tmp_path / "nope.env", environ={}) == {}


def test_llm_config_names_the_missing_key_without_printing_values():
    with pytest.raises(envmod.EnvError) as ei:
        envmod.llm_config(environ={"MODEL": "m", "BASE_URL": "u", "API_KEY": "  "})
    msg = str(ei.value)
    assert "API_KEY" in msg and "MODEL" not in msg     # names only the missing one

    cfg = envmod.llm_config(environ={"MODEL": "m", "BASE_URL": "u",
                                     "API_KEY": FAKE_KEY})
    assert (cfg.model, cfg.base_url, cfg.api_key) == ("m", "u", FAKE_KEY)
    assert cfg.configured


def test_config_never_shows_the_key():
    cfg = envmod.LLMConfig(model="m", base_url="u", api_key=FAKE_KEY)
    for s in (repr(cfg), str(cfg), cfg.redacted()):
        assert FAKE_KEY not in s
    assert cfg.redacted().endswith("1234")             # the last 4 characters are all that may show


# --------------------------------------------------------------------------- #
# Inline chain of thought
# --------------------------------------------------------------------------- #
def test_split_inline_thinking_on_a_minimax_shaped_reply():
    """MiniMax inlines <think> in content: text must be the clean answer, and reasoning holds the reasoning."""
    content = ("<think>\nLet me compute 6*7. 6*7 = 42. The answer is 42.\n</think>\n\n"
               "The answer is $\\boxed{42}$.")
    client, seen, _ = make_client([(200, chat_response(
        content, usage={"prompt_tokens": 31, "completion_tokens": 57}))])

    comp = client.complete([{"role": "user", "content": "6*7?"}], max_tokens=256)

    assert comp.text == "The answer is $\\boxed{42}$."
    assert "<think>" not in comp.text and "6*7 = 42" not in comp.text
    assert "6*7 = 42" in comp.reasoning
    assert (comp.prompt_tokens, comp.completion_tokens) == (31, 57)
    assert comp.finish_reason == "stop" and comp.retried_for_length is False
    assert comp.request_id == "chatcmpl-abc123"
    assert comp.model == "MiniMax-M2.5-highspeed"
    assert seen[0]["max_tokens"] == 256 and seen[0]["temperature"] == 0.0


def test_split_inline_thinking_unclosed_block_yields_no_answer():
    """An unclosed think block counts entirely as reasoning: there is no answer in it, and handing over half the reasoning would be dishonest."""
    text, reasoning = _split_inline_thinking("<think>still thinking about it")
    assert text == ""
    assert reasoning == "<think>still thinking about it"
    assert _split_inline_thinking("plain answer") == ("plain answer", "")


def test_separate_reasoning_content_field_is_used():
    """A DeepSeek-style separate field is accepted too, and takes precedence over the inline block."""
    client, _, _ = make_client([(200, chat_response(
        "the answer", reasoning_content="internal chain of thought"))])
    comp = client.complete("q")
    assert comp.text == "the answer"
    assert comp.reasoning == "internal chain of thought"


def test_usage_absent_means_none_not_zero():
    """No usage reported means None -- an estimate posing as a measurement is worse than nothing."""
    client, _, _ = make_client([(200, chat_response("hi"))])
    comp = client.complete("q")
    assert comp.prompt_tokens is None and comp.completion_tokens is None


# --------------------------------------------------------------------------- #
# Truncation rescue
# --------------------------------------------------------------------------- #
def test_length_truncation_retries_once_with_a_bigger_budget():
    """A budget eaten up by reasoning (finish_reason=length + empty answer) -> retry once with 4x the budget and leave a mark."""
    starved = chat_response("<think>reasoning that never finishes",
                            finish_reason="length")
    rescued = chat_response("<think>short</think>\n\\boxed{42}")
    client, seen, _ = make_client([(200, starved), (200, rescued)])

    comp = client.complete("hard question", max_tokens=512)

    assert client.n_requests == 2
    assert [p["max_tokens"] for p in seen] == [512, 2048]     # 4x
    assert comp.text == "\\boxed{42}"
    assert comp.retried_for_length is True


def test_length_retry_fires_exactly_once_and_never_hides_the_empty_answer():
    """Still empty means no further retry: hand over length + empty answer together with the flag, so the caller can see it."""
    starved = chat_response("<think>never finishes", finish_reason="length")
    client, seen, _ = make_client([(200, starved)])

    comp = client.complete("q", max_tokens=256)

    assert client.n_requests == 2                              # rescued only once
    assert comp.text == "" and comp.finish_reason == "length"
    assert comp.retried_for_length is True and comp.truncated_empty is True


def test_no_length_retry_when_the_answer_is_present():
    client, _, _ = make_client([(200, chat_response("done", finish_reason="length"))])
    comp = client.complete("q", max_tokens=256)
    assert client.n_requests == 1 and comp.retried_for_length is False


def test_no_length_retry_at_the_ceiling():
    """Already at the ceiling means no more multiplying -- otherwise every call would burn one extra budget for nothing."""
    starved = chat_response("<think>...", finish_reason="length")
    client, _, _ = make_client([(200, starved)], max_tokens_ceiling=256)
    comp = client.complete("q", max_tokens=256)
    assert client.n_requests == 1 and comp.retried_for_length is False


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #
def test_429_then_200_succeeds_after_a_backoff():
    client, _, sleeps = make_client([(429, "rate limited"), (200, chat_response("ok"))])
    comp = client.complete("q")
    assert comp.text == "ok"
    assert client.n_requests == 2
    assert len(sleeps) == 1 and sleeps[0] > 0                  # backed off once, with a positive delay


def test_400_raises_immediately_without_retrying():
    client, _, sleeps = make_client([(400, "bad request: max_tokens too large")])
    with pytest.raises(LLMHTTPError) as ei:
        client.complete("q")
    assert ei.value.status_code == 400
    assert client.n_requests == 1 and sleeps == []             # no retry at all


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_other_client_errors_are_not_retried(status):
    client, _, sleeps = make_client([(status, "nope")])
    with pytest.raises(LLMHTTPError):
        client.complete("q")
    assert client.n_requests == 1 and sleeps == []


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_are_retried(status):
    client, _, sleeps = make_client([(status, "boom"), (200, chat_response("ok"))])
    assert client.complete("q").text == "ok"
    assert client.n_requests == 2 and len(sleeps) == 1


def test_transport_errors_are_retried_then_give_up():
    client, _, sleeps = make_client([httpx.ConnectError("connection reset")],
                                    max_retries=3)
    with pytest.raises(LLMError) as ei:
        client.complete("q")
    assert client.n_requests == 3                              # all attempts used
    assert len(sleeps) == 2                                    # no sleep after the last attempt
    assert "connection reset" in str(ei.value)


def test_backoff_uses_a_private_rng(monkeypatch):
    """Jitter must not consume the global random stream: the calibration test (tests/legacy/test_10_calibrate.py) pins seeded error-rate injection."""
    import random as _random

    monkeypatch.setattr(_random, "random",
                        lambda: pytest.fail("touched the global random stream"))
    client, _, _ = make_client([(503, "boom"), (200, chat_response("ok"))])
    assert client.complete("q").text == "ok"


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
def test_extract_json_from_a_fence():
    text = 'Sure thing:\n```json\n{"label": "yes", "why": "because"}\n```\nHope it helps.'
    assert extract_json(text) == {"label": "yes", "why": "because"}


def test_extract_json_from_prose_and_nested_braces():
    text = ('Here is the plan {not json} really: '
            '{"kind": "tool", "input": {"path": "a/b", "opts": {"n": 1}}, '
            '"note": "a } inside a string"} — done.')
    assert extract_json(text) == {"kind": "tool",
                                  "input": {"path": "a/b", "opts": {"n": 1}},
                                  "note": "a } inside a string"}


def test_extract_json_ignores_braces_inside_the_think_block():
    """Braces on the scratch paper appear before the answer -- without stripping the chain of thought we would fish that one out."""
    text = '<think>maybe {"label": "no"} is right</think>\n{"label": "yes"}'
    assert extract_json(text) == {"label": "yes"}


def test_extract_json_returns_none_when_there_is_none():
    assert extract_json("no object here at all") is None
    assert extract_json("") is None


# --------------------------------------------------------------------------- #
# ModelAdapter
# --------------------------------------------------------------------------- #
def test_adapter_matches_the_model_protocol_signatures():
    """The signatures must be identical to ScriptedModel's so runtime.run_task can swap one for the other in place."""
    client, _, _ = make_client([(200, chat_response("x"))])
    adapter = ModelAdapter(client)
    assert isinstance(adapter, Model)
    for name in ("classify", "generate"):
        assert inspect.signature(getattr(adapter, name)) == \
            inspect.signature(getattr(ScriptedModel(), name))


def test_classify_returns_the_label_from_json():
    client, seen, _ = make_client([(200, chat_response(
        '<think>looks messy</think>\n{"label": "dirty"}'))])
    adapter = ModelAdapter(client, seed=7)
    got = adapter.classify(prompt="Is the header clean?", values={"n": 3},
                           labels=["clean", "dirty", "abstain"])
    assert got == "dirty"
    assert seen[0]["seed"] == 7 and seen[0]["temperature"] == 0.0


@pytest.mark.parametrize("reply", [
    "purple",                                  # outside the label set
    "```json\n{\"label\": \"maybe\"}\n```",    # parses, but the label is not in the set
    "…………",                                    # pure noise
    "",                                        # empty reply
    "could be clean or dirty, hard to say",     # bare text mentions two labels: do not guess
])
def test_classify_falls_back_to_abstain_on_offlist_or_garbage(reply):
    """Unparseable, or something outside the set: always abstain -- never invent a label that sends the machine down an ungrounded edge."""
    client, _, _ = make_client([(200, chat_response(reply))])
    adapter = ModelAdapter(client)
    labels = ["clean", "dirty", "abstain"]
    assert adapter.classify(prompt="q", values={}, labels=labels) == "abstain"


def test_classify_never_raises_when_the_endpoint_fails():
    client, _, _ = make_client([(400, "bad request")])
    adapter = ModelAdapter(client)
    assert adapter.classify(prompt="q", values={},
                            labels=["a", "b", "abstain"]) == "abstain"
    assert adapter.usage()["abstained"] == 1


def test_generate_returns_the_object_and_accounts_tokens():
    client, seen, _ = make_client([(200, chat_response(
        '<think>step</think>\n{"kind": "tool", "name": "read_table"}',
        usage={"prompt_tokens": 10, "completion_tokens": 4}))])
    adapter = ModelAdapter(client)
    out = adapter.generate(prompt="full text of SKILL.md", values={"path": "a.csv"},
                           history=({"step": 1},))
    assert out == {"kind": "tool", "name": "read_table"}
    assert adapter.usage() == {"llm_calls": 1, "prompt_tokens": 10,
                               "completion_tokens": 4, "unmeasured_calls": 0,
                               "length_retries": 0, "abstained": 0}
    assert "a.csv" in seen[0]["messages"][-1]["content"]


def test_generate_raises_when_there_is_no_json_object():
    client, _, _ = make_client([(200, chat_response("I could not comply."))])
    with pytest.raises(LLMError):
        ModelAdapter(client).generate(prompt="p", values={})


def test_usage_counts_unmeasured_calls_honestly():
    client, _, _ = make_client([(200, chat_response('{"a": 1}'))])
    adapter = ModelAdapter(client)
    adapter.generate(prompt="p", values={})
    u = adapter.usage()
    assert u["unmeasured_calls"] == 1 and u["prompt_tokens"] == 0


# --------------------------------------------------------------------------- #
# The API key never leaks
# --------------------------------------------------------------------------- #
def test_the_api_key_never_appears_in_reprs_or_errors():
    """The key must not appear in repr/str/exception messages -- even when the server echoes it, it is erased before leaving."""
    client, _, _ = make_client([(401, f"invalid api key: {FAKE_KEY}")])
    for s in (repr(client), str(client)):
        assert FAKE_KEY not in s and "***1234" in s            # only the last 4 characters show

    with pytest.raises(LLMHTTPError) as ei:
        client.complete("q")
    msg = str(ei.value)
    assert FAKE_KEY not in msg and "***1234" in msg

    comp = Completion(text="hi")
    assert FAKE_KEY not in repr(comp)


def test_giving_up_after_retries_also_scrubs_the_key():
    client, _, _ = make_client([(503, f"upstream rejected {FAKE_KEY}")],
                               max_retries=2)
    with pytest.raises(LLMError) as ei:
        client.complete("q")
    assert FAKE_KEY not in str(ei.value)


def test_the_key_only_travels_in_the_authorization_header():
    """The key only goes into the header, never the body -- the body is the half that gets written to logs and traces."""
    seen_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("authorization", ""))
        assert FAKE_KEY not in request.content.decode("utf-8")
        return httpx.Response(200, json=chat_response("ok"))

    client = OpenAIClient("m", "https://api.example/v1/", FAKE_KEY,
                          transport=httpx.MockTransport(handler))
    assert client.complete("q").text == "ok"
    assert seen_headers == [f"Bearer {FAKE_KEY}"]
    assert client.url == "https://api.example/v1/chat/completions"


def test_llm_config_profile_reads_prefixed_keys_with_defaults():
    from hexis.llm.env import EnvError, llm_config
    cfg = llm_config(environ={"MINIMAX_API_KEY": "k-1234"}, profile="minimax")
    assert cfg.model == "MiniMax-M2.5-highspeed" and cfg.base_url.endswith("/v1")
    cfg = llm_config(environ={"MINIMAX_API_KEY": "k", "MINIMAX_MODEL": "MiniMax-M3"}, profile="minimax")
    assert cfg.model == "MiniMax-M3"
    import pytest
    with pytest.raises(EnvError) as ei:
        llm_config(environ={"MODEL": "x", "BASE_URL": "y", "API_KEY": "z"}, profile="minimax")
    assert "MINIMAX_API_KEY" in str(ei.value) and "k-1234" not in str(ei.value)
