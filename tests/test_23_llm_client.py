"""env（.env 定位/解析）与 llm_client（OpenAI 兼容端点）的密闭测试。

**全程无网络**：所有 HTTP 都走 :class:`httpx.MockTransport`，退避 sleep 被换成记账函数，
所以整份文件是毫秒级的。钉住的都是被真实端点教训过的行为——内联 ``<think>`` 必须剥、
预算被推理吃光要救援且只救一次、哪些状态码值得重试、以及**任何地方都不许出现 API key**。
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
# 脚手架
# --------------------------------------------------------------------------- #
def chat_response(content: str, *, finish_reason: str = "stop",
                  reasoning_content: str | None = None,
                  usage: dict | None = None) -> dict:
    """一份真实形状的 /chat/completions 响应体。"""
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
    """按 ``responses`` 依次应答的客户端。返回 ``(client, seen, sleeps)``。

    ``responses`` 的每一项是 ``(status, body)``、``httpx.Response``，或一个要抛的异常；
    用完之后重复最后一项。``seen`` 收下每次请求的 payload，``sleeps`` 收下每次退避时长。
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
    client._sleep = sleeps.append          # 退避不真睡：测试要毫秒级
    return client, seen, sleeps


# --------------------------------------------------------------------------- #
# env：.env 的定位与解析
# --------------------------------------------------------------------------- #
def test_find_env_stops_at_the_git_holder(tmp_path):
    """走到拿着 .git 的那一层就收手——检出之外的 .env 永远不该被捡进来。"""
    (tmp_path / "stray.env").write_text("X=1", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=outside-the-checkout", encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    deep = repo / "pkg" / "sub"
    deep.mkdir(parents=True)

    assert envmod.find_env_file(deep) is None          # 仓库里没有 .env → 不外捡

    (repo / ".env").write_text("MODEL=m", encoding="utf-8")
    assert envmod.find_env_file(deep) == repo / ".env"  # 同层 .env 先于 .git 判定


def test_find_env_accepts_a_file_as_start(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".env").write_text("MODEL=m", encoding="utf-8")
    f = repo / "pkg" / "mod.py"
    f.parent.mkdir(parents=True)
    f.write_text("# 起点给文件时应从它所在目录起步", encoding="utf-8")
    assert envmod.find_env_file(f) == repo / ".env"


def test_find_env_default_start_is_this_checkout():
    """默认起点是包自己所在处；本仓库根上确实有一份 .env（值不看、更不打印）。"""
    found = envmod.find_env_file()
    if (ROOT / ".env").exists():
        assert found == ROOT / ".env"
    else:                                              # CI 上靠环境变量注入也算合法
        assert found is None


def test_load_env_parses_and_does_not_override(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(["# 注释行", "", "MODEL=MiniMax-M2.5-highspeed",
                   'BASE_URL="https://api.example/v1"', "API_KEY='sk-quoted'",
                   "没有等号的一行"]),
        encoding="utf-8")
    fake_env = {"MODEL": "already-set"}
    parsed = envmod.load_env(path=p, environ=fake_env)

    assert parsed == {"MODEL": "MiniMax-M2.5-highspeed",
                      "BASE_URL": "https://api.example/v1",
                      "API_KEY": "sk-quoted"}
    assert fake_env["MODEL"] == "already-set"          # 默认不覆盖
    assert fake_env["BASE_URL"] == "https://api.example/v1"

    envmod.load_env(path=p, environ=fake_env, override=True)
    assert fake_env["MODEL"] == "MiniMax-M2.5-highspeed"


def test_load_env_missing_file_is_not_an_error(tmp_path):
    assert envmod.load_env(path=tmp_path / "nope.env", environ={}) == {}


def test_llm_config_names_the_missing_key_without_printing_values():
    with pytest.raises(envmod.EnvError) as ei:
        envmod.llm_config(environ={"MODEL": "m", "BASE_URL": "u", "API_KEY": "  "})
    msg = str(ei.value)
    assert "API_KEY" in msg and "MODEL" not in msg     # 只点名缺的那个

    cfg = envmod.llm_config(environ={"MODEL": "m", "BASE_URL": "u",
                                     "API_KEY": FAKE_KEY})
    assert (cfg.model, cfg.base_url, cfg.api_key) == ("m", "u", FAKE_KEY)
    assert cfg.configured


def test_config_never_shows_the_key():
    cfg = envmod.LLMConfig(model="m", base_url="u", api_key=FAKE_KEY)
    for s in (repr(cfg), str(cfg), cfg.redacted()):
        assert FAKE_KEY not in s
    assert cfg.redacted().endswith("1234")             # 末 4 位是允许露的全部


# --------------------------------------------------------------------------- #
# 内联思维链
# --------------------------------------------------------------------------- #
def test_split_inline_thinking_on_a_minimax_shaped_reply():
    """MiniMax 把 <think> 内联在 content 里：text 必须是干净答案，reasoning 装推理。"""
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
    """未闭合的思考块整段算推理：里面没有答案可捞，交半段推理是不诚实的。"""
    text, reasoning = _split_inline_thinking("<think>still thinking about it")
    assert text == ""
    assert reasoning == "<think>still thinking about it"
    assert _split_inline_thinking("plain answer") == ("plain answer", "")


def test_separate_reasoning_content_field_is_used():
    """DeepSeek 式的单独字段也要认，且优先于内联块。"""
    client, _, _ = make_client([(200, chat_response(
        "the answer", reasoning_content="internal chain of thought"))])
    comp = client.complete("q")
    assert comp.text == "the answer"
    assert comp.reasoning == "internal chain of thought"


def test_usage_absent_means_none_not_zero():
    """没报 usage 就是 None——估算冒充测量比不报还糟。"""
    client, _, _ = make_client([(200, chat_response("hi"))])
    comp = client.complete("q")
    assert comp.prompt_tokens is None and comp.completion_tokens is None


# --------------------------------------------------------------------------- #
# 截断救援
# --------------------------------------------------------------------------- #
def test_length_truncation_retries_once_with_a_bigger_budget():
    """预算被推理吃光（finish_reason=length + 空答案）→ 加 4 倍预算重试一次并留痕。"""
    starved = chat_response("<think>reasoning that never finishes",
                            finish_reason="length")
    rescued = chat_response("<think>short</think>\n\\boxed{42}")
    client, seen, _ = make_client([(200, starved), (200, rescued)])

    comp = client.complete("hard question", max_tokens=512)

    assert client.n_requests == 2
    assert [p["max_tokens"] for p in seen] == [512, 2048]     # 4 倍
    assert comp.text == "\\boxed{42}"
    assert comp.retried_for_length is True


def test_length_retry_fires_exactly_once_and_never_hides_the_empty_answer():
    """再空也不再重试：把 length + 空答案连同旗子交出去，让调用方看得见。"""
    starved = chat_response("<think>never finishes", finish_reason="length")
    client, seen, _ = make_client([(200, starved)])

    comp = client.complete("q", max_tokens=256)

    assert client.n_requests == 2                              # 只救一次
    assert comp.text == "" and comp.finish_reason == "length"
    assert comp.retried_for_length is True and comp.truncated_empty is True


def test_no_length_retry_when_the_answer_is_present():
    client, _, _ = make_client([(200, chat_response("done", finish_reason="length"))])
    comp = client.complete("q", max_tokens=256)
    assert client.n_requests == 1 and comp.retried_for_length is False


def test_no_length_retry_at_the_ceiling():
    """已经顶到天花板就不再加倍——否则每次调用都白烧一次预算。"""
    starved = chat_response("<think>...", finish_reason="length")
    client, _, _ = make_client([(200, starved)], max_tokens_ceiling=256)
    comp = client.complete("q", max_tokens=256)
    assert client.n_requests == 1 and comp.retried_for_length is False


# --------------------------------------------------------------------------- #
# 重试策略
# --------------------------------------------------------------------------- #
def test_429_then_200_succeeds_after_a_backoff():
    client, _, sleeps = make_client([(429, "rate limited"), (200, chat_response("ok"))])
    comp = client.complete("q")
    assert comp.text == "ok"
    assert client.n_requests == 2
    assert len(sleeps) == 1 and sleeps[0] > 0                  # 退了一次，退避为正


def test_400_raises_immediately_without_retrying():
    client, _, sleeps = make_client([(400, "bad request: max_tokens too large")])
    with pytest.raises(LLMHTTPError) as ei:
        client.complete("q")
    assert ei.value.status_code == 400
    assert client.n_requests == 1 and sleeps == []             # 一次都不重试


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
    assert client.n_requests == 3                              # 用满次数
    assert len(sleeps) == 2                                    # 最后一次不再睡
    assert "connection reset" in str(ei.value)


def test_backoff_uses_a_private_rng(monkeypatch):
    """抖动不许消费全局随机流：test_10 那边钉着按种子注入的误差率。"""
    import random as _random

    monkeypatch.setattr(_random, "random",
                        lambda: pytest.fail("动了全局随机流"))
    client, _, _ = make_client([(503, "boom"), (200, chat_response("ok"))])
    assert client.complete("q").text == "ok"


# --------------------------------------------------------------------------- #
# JSON 抽取
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
    """草稿纸上的花括号比答案先出现——不剥思维链就会捞到它。"""
    text = '<think>maybe {"label": "no"} is right</think>\n{"label": "yes"}'
    assert extract_json(text) == {"label": "yes"}


def test_extract_json_returns_none_when_there_is_none():
    assert extract_json("no object here at all") is None
    assert extract_json("") is None


# --------------------------------------------------------------------------- #
# ModelAdapter
# --------------------------------------------------------------------------- #
def test_adapter_matches_the_model_protocol_signatures():
    """签名必须和 ScriptedModel 一模一样，runtime.run_task 才能原地换人。"""
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
    got = adapter.classify(prompt="表头干净吗？", values={"n": 3},
                           labels=["clean", "dirty", "弃权"])
    assert got == "dirty"
    assert seen[0]["seed"] == 7 and seen[0]["temperature"] == 0.0


@pytest.mark.parametrize("reply", [
    "purple",                                  # 标签集之外
    "```json\n{\"label\": \"maybe\"}\n```",    # 会解析，但标签不在集里
    "…………",                                    # 纯噪声
    "",                                        # 空回复
    "could be clean or dirty, hard to say",     # 裸文本提到两个，不猜
])
def test_classify_falls_back_to_abstain_on_offlist_or_garbage(reply):
    """解析不出、或回了集外的东西，一律弃权——绝不发明一个标签让机器沿无依据的边走。"""
    client, _, _ = make_client([(200, chat_response(reply))])
    adapter = ModelAdapter(client)
    labels = ["clean", "dirty", "弃权"]
    assert adapter.classify(prompt="q", values={}, labels=labels) == "弃权"


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
    out = adapter.generate(prompt="SKILL.md 全文", values={"path": "a.csv"},
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
# 密钥不外泄
# --------------------------------------------------------------------------- #
def test_the_api_key_never_appears_in_reprs_or_errors():
    """repr/str/异常消息里都不许出现 key——服务端回显它也要在出门前抹掉。"""
    client, _, _ = make_client([(401, f"invalid api key: {FAKE_KEY}")])
    for s in (repr(client), str(client)):
        assert FAKE_KEY not in s and "***1234" in s            # 只露末 4 位

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
    """key 只进 header，绝不进 body——body 是会被记进日志和轨迹的那一半。"""
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
