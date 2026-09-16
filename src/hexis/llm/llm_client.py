"""An OpenAI-compatible `/chat/completions` client, plus the adapter that plugs it into the
:class:`~hexis.llm.model_iface.Model` protocol.

Native tool calling, logprobs
and logit_bias probing were cut -- the machine's actions are decided by the state machine, and the model does only
two things: **judge** (pick one label from a given label set) and **generate** (reply with one JSON object). What
remains is the parts that real endpoints taught us the hard way:

**Inline chain of thought must be split off here.** DeepSeek puts it in a separate ``reasoning_content`` field, so
the contract holds naturally; MiniMax's M2/M3 inline ``<think>...</think>`` directly in ``content``. Not splitting it
means sending the scratch paper downstream as the answer: the judge's number-extraction regex reads the numbers in
the reasoning first, and JSON extraction finds the braces in the reasoning first. The observed consequence: a reply
**whose content was actually correct** got task_success 0 because the judge saw the whole think block, and the
comparison between the three arms lost its signal on the spot. So :attr:`Completion.text` is always the cleanly
stripped answer, and :attr:`Completion.reasoning` holds the reasoning separately.

**A budget eaten up by reasoning must be rescued.** This endpoint has been observed to return
``finish_reason="length"`` + an empty answer: the think block used up ``max_tokens``. An empty answer looks exactly
like "answered wrong" downstream, disguising "did not run" as "ran but was wrong". So retry **once** with 4x the
budget and leave a mark on :attr:`Completion.retried_for_length`; if it is still empty, hand over
``finish_reason='length'`` + the empty text as they are (together with that flag) and let the caller decide -- no
silence, no pretending.

**Token counts only report measurements.** Read them from ``usage`` when present; otherwise ``None``, never an
estimate passed off as a measurement.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import httpx

# keep only one brace-balancing scanner: the one in runtime is already used by the FALLBACK interpretation path; reuse
# it here so the two places never give different answers about "the model replied with half a JSON".
from hexis.execution.runtime import _first_json_object
from hexis.llm.env import redact
from hexis.llm.model_iface import ModelUnavailable, _abstain

#: inline chain of thought: non-greedy up to the first ``</think>``, across lines, case-insensitive.
_THINK = re.compile(r"^.*?</think\s*>", re.DOTALL | re.IGNORECASE)

#: status codes worth retrying: rate limiting and gateway hiccups. For the rest (400/401/403/404/422 etc.) retrying just
#: repeats the same error max_retries times and buries the real cause under the backoff.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: backoff caps: a rate-limit window needs longer to cool down than a 5xx.
_BACKOFF_CAP_429 = 30.0
_BACKOFF_CAP_5XX = 8.0


class LLMError(RuntimeError):
    """A model call failed. The message never contains the API key (see :meth:`OpenAIClient._scrub`)."""


class LLMUnavailable(LLMError, ModelUnavailable):
    """The endpoint kept failing (server errors, rate limits or network errors) until the retries were used up."""


class LLMHTTPError(LLMError, ModelUnavailable):
    """The endpoint returned a non-2xx status. ``status_code`` decides whether it is retryable."""

    def __init__(self, status_code: int, body: str, url: str = "") -> None:
        super().__init__(f"HTTP {status_code} from {url}: {body}")
        self.status_code = int(status_code)
        self.body = body


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def _split_inline_thinking(content: str) -> tuple[str, str]:
    """``(answer, inline chain of thought)``. The chain of thought is an empty string when there is no inline block.

    An unclosed block (the model ran out of budget mid-thought) counts entirely as chain of thought with an empty
    answer: there is no answer to salvage in it, and handing it to the caller's empty-reply path is more honest than
    returning half of the reasoning.
    """
    s = content or ""
    low = s.lower()
    if "</think" not in low:
        if "<think" in low:
            return "", s
        return s, ""
    answer = _THINK.sub("", s, count=1).strip()
    return answer, s[:len(s) - len(answer)].strip()


def strip_thinking(text: str) -> str:
    """Only the answer half."""
    return _split_inline_thinking(text)[0]


def extract_json(text: str) -> Optional[dict]:
    """Take the first balanced JSON object from a reply that may be wrapped in prose or a ```json fence.

    Strip the chain of thought before scanning: braces in the reasoning appear before the answer, and without
    stripping we would fish out the object from the scratch paper. The scanner reuses
    :func:`hexis.execution.runtime._first_json_object`.
    """
    answer = strip_thinking(text or "")
    obj = _first_json_object(answer)
    if obj is None and answer != (text or ""):
        # no object on the answer side: look at the whole text (e.g. the model wrote the JSON inside an unclosed think block)
        obj = _first_json_object(text or "")
    return obj


def _to_chat_messages(messages: Any) -> list[dict]:
    """Normalize messages to ``[{"role","content"}, ...]``. A bare string becomes one user message."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: list[dict] = []
    for m in messages or []:
        if isinstance(m, str):
            out.append({"role": "user", "content": m})
        elif isinstance(m, Mapping):
            out.append({"role": str(m.get("role") or "user"),
                        "content": str(m.get("content") or "")})
        else:                                   # an object with .role/.content
            out.append({"role": str(getattr(m, "role", "user")),
                        "content": str(getattr(m, "content", ""))})
    return out


def _int_or_none(v: Any) -> Optional[int]:
    """Read an integer if possible, otherwise ``None`` -- no estimates, no 0 passed off as a measurement."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# One completion
# --------------------------------------------------------------------------- #
@dataclass
class Completion:
    """Everything observable about one ``/chat/completions`` call.

    ``text`` is **the answer with the chain of thought stripped** (the only thing downstream should parse/score),
    ``reasoning`` is the internal reasoning. ``prompt_tokens``/``completion_tokens`` being ``None`` means the
    endpoint reported no usage.
    """

    text: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    ms: float = 0.0
    model: str = ""
    request_id: str = ""
    #: retried once with a larger budget because of "finish_reason=length and an empty answer"
    retried_for_length: bool = False

    @property
    def truncated_empty(self) -> bool:
        """The budget ran out and the answer is still empty -- the caller should treat it as "this did not run", not as "answered wrong"."""
        return self.finish_reason == "length" and not self.text.strip()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """One OpenAI-compatible endpoint. Synchronous calls within a thread, with backoff retries and truncation rescue.

    ``transport`` exists only for tests: pass an :class:`httpx.MockTransport` to run entirely without network.
    ``jitter_seed`` derives a private :class:`random.Random` and **does not touch the global random stream** (tests
    pin the global seed for other purposes).
    """

    def __init__(self, model: str, base_url: str, api_key: str, *,
                 timeout: float = 180.0, max_retries: int = 6,
                 max_tokens_ceiling: int = 32768,
                 jitter_seed: int = 0,
                 transport: Optional[httpx.BaseTransport] = None) -> None:
        if not api_key:
            raise LLMError("no API key (see hexis.llm.env.llm_config)")
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.url = self.base_url if self.base_url.endswith("/chat/completions") \
            else self.base_url + "/chat/completions"
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.max_tokens_ceiling = int(max_tokens_ceiling)
        self.n_requests = 0                      # HTTP requests actually sent (including retries)
        #: streaming receive: long chain-of-thought models generate for minutes at a time, and non-streaming responses get cut
        #: off by a gateway 504 (observed on MiniMax's ALB), while streaming replies as it thinks. Once complete, the pieces are
        #: assembled into a response of the same shape as a non-streaming one, invisible to the layers above.
        self.stream = False
        self._api_key = api_key
        self._rng = random.Random(jitter_seed)   # jitter uses its own stream
        self._sleep: Callable[[float], None] = time.sleep   # replaceable in tests
        self._transport = transport
        self._http: Optional[httpx.Client] = None

    # ---- lifecycle ---- #
    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout, transport=self._transport)
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> "OpenAIClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:                   # the API key only shows its last 4 characters
        return (f"OpenAIClient(model={self.model!r}, url={self.url!r}, "
                f"api_key={redact(self._api_key)!r})")

    def _scrub(self, text: str) -> str:
        """Erase the API key, which the server may echo back, from any string about to leave this object."""
        s = str(text or "")
        return s.replace(self._api_key, redact(self._api_key)) if self._api_key else s

    # ---- main entry point ---- #
    def complete(self, messages: Any, *, temperature: float = 0.0,
                 max_tokens: int = 8192, seed: Optional[int] = None,
                 stop: Optional[Sequence[str]] = None,
                 response_format: Optional[dict] = None,
                 extra_body: Optional[Mapping[str, Any]] = None) -> Completion:
        """Request one completion. ``<think>`` blocks in the answer are stripped, and a truncated empty answer has been rescued once with 4x the budget.

        ``extra_body`` is merged into the request body as is: endpoint-specific parameters go here (DeepSeek turns
        thinking off with ``{"thinking": {"type": "disabled"}}``, the OpenAI family uses ``{"reasoning_effort": "low"}``).
        An endpoint that does not accept them returns 400; that means the caller picked the wrong switch, and it is
        not papered over here.
        """
        payload: dict = {
            "model": self.model,
            "messages": _to_chat_messages(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if extra_body:
            payload.update({str(k): v for k, v in dict(extra_body).items()})
        if seed is not None:
            payload["seed"] = int(seed)
        if stop:
            payload["stop"] = [stop] if isinstance(stop, str) else list(stop)
        if response_format:
            payload["response_format"] = response_format

        t0 = time.perf_counter()
        data, headers = self._post_with_retry(payload)
        comp = self._to_completion(data, headers, t0)

        # the reasoning used up the budget: ask once more with a larger one. Only fire when the budget really is to blame --
        # finish_reason says length, and the answer is empty after stripping the chain of thought.
        if (comp.finish_reason == "length" and not comp.text.strip()
                and payload["max_tokens"] < self.max_tokens_ceiling):
            payload["max_tokens"] = min(payload["max_tokens"] * 4, self.max_tokens_ceiling)
            data, headers = self._post_with_retry(payload)
            comp = self._to_completion(data, headers, t0)
            comp.retried_for_length = True       # retry only once: leave a mark, no recursion
        return comp

    # ---- internals ---- #
    def _to_completion(self, data: dict, headers: Mapping[str, str],
                       t0: float) -> Completion:
        choice = ((data.get("choices") or [{}])[0]) or {}
        msg = choice.get("message") or {}
        text, inline = _split_inline_thinking(msg.get("content") or "")
        separate = (msg.get("reasoning_content") or "").strip()
        usage = data.get("usage") or {}
        rid = data.get("id") or headers.get("x-request-id") or headers.get("trace-id")
        return Completion(
            text=text,
            reasoning=separate or inline,        # the separate field first, then the inline block
            finish_reason=str(choice.get("finish_reason") or ""),
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            ms=(time.perf_counter() - t0) * 1000.0,
            model=str(data.get("model") or self.model),
            request_id=str(rid or ""),
        )

    def _post_stream(self, payload: dict, headers: dict) -> tuple[dict, Mapping[str, str]]:
        """Streaming POST: assemble the SSE deltas back into a response shaped like a non-streaming one."""
        body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        content: list[str] = []; reasoning: list[str] = []
        finish = ""; usage: dict = {}; rid = ""; model = ""
        with self._client().stream("POST", self.url, json=body, headers=headers,
                                   timeout=self.timeout) as resp:
            if resp.status_code >= 400:
                raw = resp.read()
                raise LLMHTTPError(resp.status_code, self._scrub(raw.decode("utf-8", "replace"))[:300], self.url)
            resp_headers = resp.headers
            for line in resp.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(j, dict):
                    continue
                rid = rid or str(j.get("id") or ""); model = model or str(j.get("model") or "")
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content.append(str(d["content"]))
                    if d.get("reasoning_content"):
                        reasoning.append(str(d["reasoning_content"]))
                    if ch.get("finish_reason"):
                        finish = str(ch["finish_reason"])
        data = {"id": rid, "model": model,
                "choices": [{"message": {"content": "".join(content), "reasoning_content": "".join(reasoning)},
                             "finish_reason": finish}],
                "usage": usage}
        return data, resp_headers

    def _post(self, payload: dict) -> tuple[dict, Mapping[str, str]]:
        headers = {"Authorization": f"Bearer {self._api_key}",
                   "Content-Type": "application/json"}
        self.n_requests += 1
        if self.stream:
            return self._post_stream(payload, headers)
        resp = self._client().post(self.url, json=payload, headers=headers,
                                   timeout=self.timeout)
        if resp.status_code >= 400:
            # include the status code and a snippet of the body: a "rate limited, back off and retry" 429 and a
            # "quota exhausted" 429 can only be told apart by the body, and the latter should not be retried at all.
            raise LLMHTTPError(resp.status_code, self._scrub(resp.text)[:300], self.url)
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"{self.url} returned non-JSON: "
                           f"{self._scrub(resp.text)[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMError(f"{self.url} returned JSON that is not an object: {type(data).__name__}")
        return data, resp.headers

    def _post_with_retry(self, payload: dict) -> tuple[dict, Mapping[str, str]]:
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._post(payload)
            except LLMHTTPError as exc:
                if exc.status_code not in RETRYABLE_STATUS:
                    raise                        # 400/401/403/404/422: retrying is pointless
                last = exc
            except httpx.HTTPError as exc:       # transport layer: timeouts, connection resets, broken reads
                last = exc
            if attempt + 1 < self.max_retries:
                self._sleep(self._backoff(attempt, last))
        raise LLMUnavailable(f"{self.url} failed {self.max_retries} times in a row: "
                             f"{self._scrub(str(last))[:300]}") from last

    def _backoff(self, attempt: int, exc: Optional[Exception]) -> float:
        """Exponential backoff + jitter. The cap for 429 is higher than for 5xx -- a paid quota hitting a rate limit needs time to cool down."""
        is_429 = isinstance(exc, LLMHTTPError) and exc.status_code == 429
        cap = _BACKOFF_CAP_429 if is_429 else _BACKOFF_CAP_5XX
        base = min(2.0 ** attempt, cap)
        return base + self._rng.random() * base * 0.25   # jitter spreads out synchronized retries of parallel experiments


# --------------------------------------------------------------------------- #
# Plugging into the Model protocol
# --------------------------------------------------------------------------- #
_CLASSIFY_SYS = (
    "You are a deterministic classifier inside a state machine. "
    "Pick exactly one label from the given list. Reply with one JSON object "
    'like {"label": "<label>"} and nothing else.'
)
_GENERATE_SYS = (
    "You execute one step of a documented procedure. "
    "Reply with exactly one JSON object and nothing else."
)


class ModelAdapter:
    """Wraps an :class:`OpenAIClient` as a :class:`~hexis.llm.model_iface.Model`.

    ``classify`` **never raises and never invents a label**: if the reply cannot be parsed, or names something
    outside the label set, it falls back to the abstain label. Misjudgments by judge actions are already on the
    machine's account (abstaining hands control to FALLBACK), whereas an invented label would send the machine
    down an edge **with no grounds** -- the hardest kind of failure to track down.

    ``generate`` raises :class:`LLMError` when no JSON object comes back; ``runtime`` records it as this step's
    state_error, which is easier to diagnose than returning an empty dict and letting downstream blow up on an
    "undefined variable".
    """

    def __init__(self, client: OpenAIClient, *, seed: Optional[int] = None,
                 temperature: float = 0.0, max_tokens: int = 8192,
                 history_limit: int = 12,
                 extra_body: Optional[Mapping[str, Any]] = None,
                 classify_extra_body: Optional[Mapping[str, Any]] = None,
                 repair_chars: Optional[int] = 6000) -> None:
        self.client = client
        #: how much of an unparseable reply is sent back when asking the model to repair it (None = all of it)
        self.repair_chars = repair_chars
        self.seed = seed
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.extra_body = dict(extra_body or {})
        #: extra request fields used only by classify (judge actions), e.g. Qwen's {"enable_thinking": False}:
        #: a three-way classification does not need hundreds of tokens of chain of thought; turning it off is faster, and was observed to be less overly conservative.
        self.classify_extra_body = dict(classify_extra_body or {})
        self.history_limit = int(history_limit)
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unmeasured_calls = 0                # number of times the endpoint reported no usage
        self.length_retries = 0
        self.json_repairs = 0                    # number of repair retries because generate's reply was not valid JSON
        self.abstained = 0
        self.last: Optional[Completion] = None

    # ---- Model protocol ---- #
    def classify(self, *, prompt: str, values: dict, labels: list[str],
                 examples: tuple = ()) -> str:
        fallback = _abstain(list(labels)) or (labels[-1] if labels else "")
        parts = [f"QUESTION: {prompt}",
                 f"VARIABLES: {_safe_json(values)}"]
        if examples:
            parts.append("EXAMPLES: " + _safe_json(list(examples))[:2000])
        parts.append("LABELS: " + _safe_json(list(labels)))
        parts.append('Answer with one JSON object: {"label": "<one of LABELS>"}.')
        try:
            comp = self._complete([{"role": "system", "content": _CLASSIFY_SYS},
                                   {"role": "user", "content": "\n\n".join(parts)}],
                                  extra=self.classify_extra_body or None)
        except Exception:                                       # noqa: BLE001
            self.abstained += 1
            return fallback                      # even if the endpoint fails, never invent a label
        label = _pick_label(comp.text, list(labels))
        if label is None:
            self.abstained += 1
            return fallback
        return label

    def generate(self, *, prompt: str, values: dict, history: tuple = ()) -> dict:
        parts = [prompt, f"VARIABLES: {_safe_json(values)}"]
        if history:
            parts.append("HISTORY (oldest first): " + _safe_json(
                [_trim(h) for h in list(history)[-self.history_limit:]]))
        parts.append("Answer with exactly one JSON object.")
        messages = [{"role": "system", "content": _GENERATE_SYS},
                    {"role": "user", "content": "\n\n".join(parts)}]
        comp = self._complete(messages)
        obj = extract_json(comp.text)
        if obj is None and not comp.truncated_empty:
            # something came back but it is not valid JSON (common when a multi-line program is stuffed into a string without
            # proper escaping): give it one chance to repair. This only remedies a parse failure; it changes neither the prompt nor what to produce.
            self.json_repairs += 1
            comp = self._complete(messages + [
                {"role": "assistant", "content": comp.text if self.repair_chars is None else comp.text[:self.repair_chars]},
                {"role": "user", "content": "That was not a valid JSON object. Return the same content as exactly "
                                            "one JSON object: escape every double quote, backslash and newline "
                                            "inside string values. No prose, no code fences."}])
            obj = extract_json(comp.text)
        if obj is None:
            hint = " (the reasoning used up the token budget; the answer is empty)" if comp.truncated_empty else ""
            raise LLMError(f"the model did not return a parseable JSON object{hint}: {comp.text[:200]!r}")
        return obj

    # ---- accounting ---- #
    def _complete(self, messages: list[dict], *, extra: Optional[Mapping[str, Any]] = None) -> Completion:
        # only pass extra_body when it is really set: fake clients in tests and older adapters' complete lack this parameter
        merged = {**self.extra_body, **dict(extra or {})}
        kw = {"extra_body": merged} if merged else {}
        comp = self.client.complete(messages, temperature=self.temperature,
                                    max_tokens=self.max_tokens, seed=self.seed, **kw)
        self.llm_calls += 1
        self.last = comp
        if comp.retried_for_length:
            self.length_retries += 1
        if comp.prompt_tokens is None and comp.completion_tokens is None:
            self.unmeasured_calls += 1           # not measured is recorded as not measured
        self.prompt_tokens += comp.prompt_tokens or 0
        self.completion_tokens += comp.completion_tokens or 0
        return comp

    def usage(self) -> dict:
        """Usage as actually measured. ``unmeasured_calls`` states how many times the endpoint reported no usage."""
        return {"llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "unmeasured_calls": self.unmeasured_calls,
                "length_retries": self.length_retries,
                "abstained": self.abstained}


def _pick_label(text: str, labels: list[str]) -> Optional[str]:
    """Take a label **that belongs to labels** from the reply; ``None`` if there is none (the caller abstains)."""
    obj = extract_json(text)
    cand = ""
    if isinstance(obj, dict):
        for k in ("label", "answer", "value", "choice"):
            if isinstance(obj.get(k), str):
                cand = obj[k].strip()
                break
    if not cand:
        cand = (text or "").strip().strip('"').strip()
    for l in labels:                             # exact first, then case-insensitive
        if cand == l:
            return l
    low = cand.lower()
    for l in labels:
        if low == l.lower():
            return l
    # accept it when the bare text mentions exactly one label; when it mentions several, do not guess
    hit = [l for l in labels if l and l.lower() in (text or "").lower()]
    return hit[0] if len(hit) == 1 else None


def _safe_json(obj: Any) -> str:
    """Best-effort JSON serialization; unserializable parts fall back to repr, so a single object never blows up the whole call."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(obj)


def _trim(item: Any, limit: int = 800) -> Any:
    """Flatten one history item into a string of bounded length -- FALLBACK history keeps growing."""
    s = _safe_json(item)
    return s if len(s) <= limit else s[:limit] + "…"


def client_from_env(*, timeout: float = 180.0, max_retries: int = 6,
                    jitter_seed: int = 0, profile: str = "", model: str = "", base_url: str = "",
                    api_key_env: str = "") -> OpenAIClient:
    """Build a client from the three settings in the repository root ``.env``. Raises :class:`~hexis.llm.env.EnvError` when configuration is missing.

    ``profile`` selects an endpoint profile (``"minimax"`` => the ``MINIMAX_*`` settings); ``model``, ``base_url``
    and ``api_key_env`` override the environment. See :func:`hexis.llm.env.llm_config`.
    """
    from hexis.llm.env import llm_config

    cfg = llm_config(profile=profile, model=model, base_url=base_url, api_key_env=api_key_env)
    return OpenAIClient(cfg.model, cfg.base_url, cfg.api_key, timeout=timeout,
                        max_retries=max_retries, jitter_seed=jitter_seed)
