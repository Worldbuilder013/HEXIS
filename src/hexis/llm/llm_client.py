"""OpenAI 兼容 `/chat/completions` 客户端，以及把它接进 :class:`~hexis.llm.model_iface.Model`
协议的适配器。

裁掉了原生工具调用、
logprobs、logit_bias 探测——机器的动作由状态机决定，模型只做两件事：**判断**（在给定标签
集里选一个）与**生成**（回一个 JSON 对象）。留下的都是被真实端点教训过的部分：

**内联思维链要在这里拆。** DeepSeek 走 ``reasoning_content`` 单独一个字段，契约自然成立；
MiniMax 的 M2/M3 把 ``<think>…</think>`` 直接内联在 ``content`` 里。不拆就等于把草稿纸当
答案往下游送：判官的取数正则先读到推理里的数字，JSON 抽取先读到推理里的花括号。原代码
记着实测后果——一条**内容其实是对的**回复，因为判官看到的是整个 think 块，task_success
被判成 0，三条臂的比较当场失去信号。所以 :attr:`Completion.text` 一定是剥干净的答案，
:attr:`Completion.reasoning` 单独装推理。

**预算被推理吃光要救回来。** 实测这个端点会返回 ``finish_reason="length"`` + 空答案：
think 块把 ``max_tokens`` 用完了。空答案在下游长得跟「答错了」一模一样，会把「没跑成」
伪装成「跑了但不对」。所以按 4 倍预算重试**一次**，并在 :attr:`Completion.retried_for_length`
上留痕；再空就把 ``finish_reason='length'`` + 空文本原样交出去（连同那面旗），让调用方
自己决定——不静默、不假装。

**token 计数只报实测的。** ``usage`` 里有就读，没有就是 ``None``，绝不用估算冒充测量。
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import httpx

from hexis.llm.env import redact
from hexis.llm.model_iface import _abstain
# 花括号配平扫描器只留一份：runtime 里那个已经被 FALLBACK 解释路径用着，这里复用同一个，
# 免得两处对「模型回了半个 JSON」给出不同答案。
from hexis.execution.runtime import _first_json_object

#: 内联思维链：非贪婪吃到第一个 ``</think>``，跨行、大小写不敏感。
_THINK = re.compile(r"^.*?</think\s*>", re.DOTALL | re.IGNORECASE)

#: 值得重试的状态码：限流与网关抖动。其余（400/401/403/404/422 等）重试只是把同一个
#: 错误再犯 max_retries 遍，还把真正的原因埋在退避里。
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: 退避封顶：限流窗口要冷却得比 5xx 久。
_BACKOFF_CAP_429 = 30.0
_BACKOFF_CAP_5XX = 8.0


class LLMError(RuntimeError):
    """调用模型失败。消息里绝不出现 API key（见 :meth:`OpenAIClient._scrub`）。"""


class LLMHTTPError(LLMError):
    """端点返回了非 2xx。``status_code`` 决定它可不可重试。"""

    def __init__(self, status_code: int, body: str, url: str = "") -> None:
        super().__init__(f"HTTP {status_code} from {url}: {body}")
        self.status_code = int(status_code)
        self.body = body


# --------------------------------------------------------------------------- #
# 文本工具
# --------------------------------------------------------------------------- #
def _split_inline_thinking(content: str) -> tuple[str, str]:
    """``(答案, 内联思维链)``。没有内联块时思维链为空串。

    未闭合的块（模型在思考中途用尽预算）整段算作思维链、答案为空：那里面没有答案可以捞，
    交给调用方的空回复路径处理，比交还半段推理诚实。
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
    """只要答案的那一半。"""
    return _split_inline_thinking(text)[0]


def extract_json(text: str) -> Optional[dict]:
    """从一段可能裹着散文或 ```json 围栏的回复里取第一个配平的 JSON 对象。

    先剥思维链再扫：推理里的花括号比答案先出现，不剥就会捞到草稿纸上的那个对象。
    扫描器复用 :func:`hexis.execution.runtime._first_json_object`。
    """
    answer = strip_thinking(text or "")
    obj = _first_json_object(answer)
    if obj is None and answer != (text or ""):
        # 答案侧没有对象时再看整段（例如模型把 JSON 写在了未闭合的思考里）
        obj = _first_json_object(text or "")
    return obj


def _to_chat_messages(messages: Any) -> list[dict]:
    """把消息归一成 ``[{"role","content"}, ...]``。裸字符串当一条 user 消息。"""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: list[dict] = []
    for m in messages or []:
        if isinstance(m, str):
            out.append({"role": "user", "content": m})
        elif isinstance(m, Mapping):
            out.append({"role": str(m.get("role") or "user"),
                        "content": str(m.get("content") or "")})
        else:                                   # 带 .role/.content 的对象
            out.append({"role": str(getattr(m, "role", "user")),
                        "content": str(getattr(m, "content", ""))})
    return out


def _int_or_none(v: Any) -> Optional[int]:
    """能读出整数就读，读不出就 ``None``——不估算、不填 0 冒充测量。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 一次完成
# --------------------------------------------------------------------------- #
@dataclass
class Completion:
    """一次 ``/chat/completions`` 的全部可观察结果。

    ``text`` 是**剥掉思维链后的答案**（唯一该被下游解析/打分的东西），``reasoning`` 是内部
    推理。``prompt_tokens``/``completion_tokens`` 为 ``None`` 表示端点没报 usage。
    """

    text: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    ms: float = 0.0
    model: str = ""
    request_id: str = ""
    #: 因为「finish_reason=length 且答案为空」加大预算重试过一次
    retried_for_length: bool = False

    @property
    def truncated_empty(self) -> bool:
        """预算耗尽且答案仍为空——调用方该按「这次没跑成」处理，而不是按「答错了」。"""
        return self.finish_reason == "length" and not self.text.strip()


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """一个 OpenAI 兼容端点。线程内同步调用，带退避重试与截断救援。

    ``transport`` 只为测试留门：传 :class:`httpx.MockTransport` 就能全程无网络。
    ``jitter_seed`` 派生自己的 :class:`random.Random`，**不碰全局随机流**（测试里全局种子
    被钉死用于别的用途）。
    """

    def __init__(self, model: str, base_url: str, api_key: str, *,
                 timeout: float = 180.0, max_retries: int = 6,
                 max_tokens_ceiling: int = 32768,
                 jitter_seed: int = 0,
                 transport: Optional[httpx.BaseTransport] = None) -> None:
        if not api_key:
            raise LLMError("没有 API key（见 hexis.llm.env.llm_config）")
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.url = self.base_url if self.base_url.endswith("/chat/completions") \
            else self.base_url + "/chat/completions"
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.max_tokens_ceiling = int(max_tokens_ceiling)
        self.n_requests = 0                      # 实际发出的 HTTP 请求数（含重试）
        #: 流式收包：长思维链模型一次生成几分钟，非流式响应会被网关 504 掐断（MiniMax 的 alb 实测），
        #: 流式则边想边回。收齐后拼成与非流式同形的响应，上层不感知。
        self.stream = False
        self._api_key = api_key
        self._rng = random.Random(jitter_seed)   # 抖动用自己的流
        self._sleep: Callable[[float], None] = time.sleep   # 测试可替换
        self._transport = transport
        self._http: Optional[httpx.Client] = None

    # ---- 生命周期 ---- #
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

    def __repr__(self) -> str:                   # 密钥只以末 4 位露面
        return (f"OpenAIClient(model={self.model!r}, url={self.url!r}, "
                f"api_key={redact(self._api_key)!r})")

    def _scrub(self, text: str) -> str:
        """把可能被服务端回显的 API key 从任何要外传的字符串里抹掉。"""
        s = str(text or "")
        return s.replace(self._api_key, redact(self._api_key)) if self._api_key else s

    # ---- 主入口 ---- #
    def complete(self, messages: Any, *, temperature: float = 0.0,
                 max_tokens: int = 8192, seed: Optional[int] = None,
                 stop: Optional[Sequence[str]] = None,
                 response_format: Optional[dict] = None,
                 extra_body: Optional[Mapping[str, Any]] = None) -> Completion:
        """要一次完成。答案里的 ``<think>`` 块已剥离，截断的空答案已按 4 倍预算救援过一次。

        ``extra_body`` 原样并进请求体：端点私有的参数走这里（DeepSeek 关思考是
        ``{"thinking": {"type": "disabled"}}``，OpenAI 系是 ``{"reasoning_effort": "low"}``）。
        不认的端点会 400，那是调用方选错了开关，不在这里兜。
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

        # 预算被推理吃光：加大一次再要。只在「确实是预算的锅」时才开火——finish_reason
        # 说是 length，且剥完思维链之后答案是空的。
        if (comp.finish_reason == "length" and not comp.text.strip()
                and payload["max_tokens"] < self.max_tokens_ceiling):
            payload["max_tokens"] = min(payload["max_tokens"] * 4, self.max_tokens_ceiling)
            data, headers = self._post_with_retry(payload)
            comp = self._to_completion(data, headers, t0)
            comp.retried_for_length = True       # 只重试一次：留痕，不递归
        return comp

    # ---- 内部 ---- #
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
            reasoning=separate or inline,        # 单独字段优先，其次内联块
            finish_reason=str(choice.get("finish_reason") or ""),
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            ms=(time.perf_counter() - t0) * 1000.0,
            model=str(data.get("model") or self.model),
            request_id=str(rid or ""),
        )

    def _post_stream(self, payload: dict, headers: dict) -> tuple[dict, Mapping[str, str]]:
        """流式 POST：把 SSE 增量拼回一份非流式同形的响应。"""
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
            # 带上状态码与正文片段：一个「限流、退避重试」的 429 和一个「配额用光」的
            # 429 只有正文能区分，而后者根本不该重试。
            raise LLMHTTPError(resp.status_code, self._scrub(resp.text)[:300], self.url)
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"{self.url} 回了非 JSON: "
                           f"{self._scrub(resp.text)[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMError(f"{self.url} 回了非对象 JSON: {type(data).__name__}")
        return data, resp.headers

    def _post_with_retry(self, payload: dict) -> tuple[dict, Mapping[str, str]]:
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._post(payload)
            except LLMHTTPError as exc:
                if exc.status_code not in RETRYABLE_STATUS:
                    raise                        # 400/401/403/404/422：重试没有意义
                last = exc
            except httpx.HTTPError as exc:       # 传输层：超时、连接重置、读断
                last = exc
            if attempt + 1 < self.max_retries:
                self._sleep(self._backoff(attempt, last))
        raise LLMError(f"{self.url} 连续 {self.max_retries} 次失败: "
                       f"{self._scrub(str(last))[:300]}") from last

    def _backoff(self, attempt: int, exc: Optional[Exception]) -> float:
        """指数退避 + 抖动。429 的封顶比 5xx 高——付费配额撞限流要给它冷却时间。"""
        is_429 = isinstance(exc, LLMHTTPError) and exc.status_code == 429
        cap = _BACKOFF_CAP_429 if is_429 else _BACKOFF_CAP_5XX
        base = min(2.0 ** attempt, cap)
        return base + self._rng.random() * base * 0.25   # 抖动打散并行实验的同步重试


# --------------------------------------------------------------------------- #
# 接进 Model 协议
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
    """把 :class:`OpenAIClient` 接成 :class:`~hexis.llm.model_iface.Model`。

    ``classify`` **永不抛异常、永不发明标签**：解析不出、或回了标签集之外的东西，就落到
    弃权标签。判断动作的误判本来就在机器的账上（弃权会把控制权交给 FALLBACK），而一个
    编造的标签会让机器沿着一条**没有依据**的边走下去——那是最难查的失败。

    ``generate`` 回不出 JSON 对象时抛 :class:`LLMError`；``runtime`` 会把它记成这一步的
    state_error，比返回空 dict 让下游在「未定义变量」上炸要好查。
    """

    def __init__(self, client: OpenAIClient, *, seed: Optional[int] = None,
                 temperature: float = 0.0, max_tokens: int = 8192,
                 history_limit: int = 12,
                 extra_body: Optional[Mapping[str, Any]] = None,
                 classify_extra_body: Optional[Mapping[str, Any]] = None) -> None:
        self.client = client
        self.seed = seed
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.extra_body = dict(extra_body or {})
        #: 只给 classify（判断动作）用的请求附加字段，例如 Qwen 的 {"enable_thinking": False}：
        #: 三选一的分类不需要几百 token 的思维链，关掉更快，实测也更不容易过度保守。
        self.classify_extra_body = dict(classify_extra_body or {})
        self.history_limit = int(history_limit)
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unmeasured_calls = 0                # 端点没报 usage 的次数
        self.length_retries = 0
        self.json_repairs = 0                    # generate 回的不是合法 JSON、补救重试的次数
        self.abstained = 0
        self.last: Optional[Completion] = None

    # ---- Model 协议 ---- #
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
            return fallback                      # 端点炸了也不能编标签
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
            # 回了东西但不是合法 JSON（常见于把多行程序塞进字符串没转义好）：给一次修复机会。
            # 这是解析失败的补救，不改提示词、不改要产出什么。
            self.json_repairs += 1
            comp = self._complete(messages + [
                {"role": "assistant", "content": comp.text[:6000]},
                {"role": "user", "content": "That was not a valid JSON object. Return the same content as exactly "
                                            "one JSON object: escape every double quote, backslash and newline "
                                            "inside string values. No prose, no code fences."}])
            obj = extract_json(comp.text)
        if obj is None:
            hint = "（预算被推理吃光，答案为空）" if comp.truncated_empty else ""
            raise LLMError(f"模型没有回出可解析的 JSON 对象{hint}: {comp.text[:200]!r}")
        return obj

    # ---- 记账 ---- #
    def _complete(self, messages: list[dict], *, extra: Optional[Mapping[str, Any]] = None) -> Completion:
        # extra_body 只在真设了才传：测试里的假客户端与旧适配器的 complete 没有这个参数
        merged = {**self.extra_body, **dict(extra or {})}
        kw = {"extra_body": merged} if merged else {}
        comp = self.client.complete(messages, temperature=self.temperature,
                                    max_tokens=self.max_tokens, seed=self.seed, **kw)
        self.llm_calls += 1
        self.last = comp
        if comp.retried_for_length:
            self.length_retries += 1
        if comp.prompt_tokens is None and comp.completion_tokens is None:
            self.unmeasured_calls += 1           # 没测到就记成没测到
        self.prompt_tokens += comp.prompt_tokens or 0
        self.completion_tokens += comp.completion_tokens or 0
        return comp

    def usage(self) -> dict:
        """实测口径的用量。``unmeasured_calls`` 明说有几次端点没报 usage。"""
        return {"llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "unmeasured_calls": self.unmeasured_calls,
                "length_retries": self.length_retries,
                "abstained": self.abstained}


def _pick_label(text: str, labels: list[str]) -> Optional[str]:
    """从回复里取一个**属于 labels** 的标签；取不到就 ``None``（调用方落弃权）。"""
    obj = extract_json(text)
    cand = ""
    if isinstance(obj, dict):
        for k in ("label", "answer", "value", "choice"):
            if isinstance(obj.get(k), str):
                cand = obj[k].strip()
                break
    if not cand:
        cand = (text or "").strip().strip('"').strip()
    for l in labels:                             # 先精确、再大小写不敏感
        if cand == l:
            return l
    low = cand.lower()
    for l in labels:
        if low == l.lower():
            return l
    # 裸文本里恰好只提到一个标签时认它；提到多个就不猜
    hit = [l for l in labels if l and l.lower() in (text or "").lower()]
    return hit[0] if len(hit) == 1 else None


def _safe_json(obj: Any) -> str:
    """尽力 JSON 化；不可序列化的部分退成 repr，绝不因为一个对象把整次调用炸掉。"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(obj)


def _trim(item: Any, limit: int = 800) -> Any:
    """历史里的单条压扁成有限长度的字符串——FALLBACK 的历史会越滚越长。"""
    s = _safe_json(item)
    return s if len(s) <= limit else s[:limit] + "…"


def client_from_env(*, timeout: float = 180.0, max_retries: int = 6,
                    jitter_seed: int = 0, profile: str = "") -> OpenAIClient:
    """按仓库根 ``.env`` 里的三件套建客户端。缺配置时 :class:`~hexis.llm.env.EnvError`。

    ``profile`` 选端点档案（``"minimax"`` ⇒ ``MINIMAX_*`` 三件套），见 :func:`hexis.llm.env.llm_config`。
    """
    from hexis.llm.env import llm_config

    cfg = llm_config(profile=profile)
    return OpenAIClient(cfg.model, cfg.base_url, cfg.api_key, timeout=timeout,
                        max_retries=max_retries, jitter_seed=jitter_seed)
