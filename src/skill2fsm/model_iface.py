"""模型与工具的接口，以及一个密闭、可复现的测试桩。

运行一台机器只需要两种外部能力：**调工具**（``Tool``，一次纯函数式的副作用）和**问模型**
（``Model``，判断或生成）。两者都封装成窄接口，好换实现、好在测试里替身。

模型有两个触点：``classify`` 给判断动作（在固定标签集里选一个，含弃权），``generate`` 给
生成型动作与 FALLBACK 解释执行（读一段私有 prompt + 变量，回一个 JSON 对象）。真实实现
在这两个方法内部拼提示、调 LLM；:class:`ScriptedModel` 则全查表——无网络、同输入同输出，
错误率按「种子 + 输入指纹」确定性注入，好让判断动作的误差率既能被标定又能复现。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
@runtime_checkable
class Tool(Protocol):
    """一次工具调用：``input`` 一个 dict 进，``output`` 一个 dict 出。确定性、无网络。"""

    name: str

    def run(self, inp: dict) -> dict: ...


class FnTool:
    """把一个纯函数包成 :class:`Tool`。"""

    def __init__(self, name: str, fn: Callable[[dict], dict]):
        self.name = name
        self._fn = fn

    def run(self, inp: dict) -> dict:
        return self._fn(inp)


class ToolRegistry:
    """名字 → 工具。进程内直接调，不进沙箱（玩具环境的工具是自带可信代码）。"""

    def __init__(self, tools: Optional[dict[str, Tool]] = None):
        self._tools: dict[str, Tool] = dict(tools or {})

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def add(self, name: str, fn: Callable[[dict], dict]) -> None:
        self.register(FnTool(name, fn))

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def call(self, name: str, inp: dict) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"没有注册工具 {name!r}")
        return tool.run(inp)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
@runtime_checkable
class Model(Protocol):
    """判断与生成两个触点。"""

    def classify(self, *, prompt: str, values: dict, labels: list[str],
                 examples: tuple = ()) -> str:
        """在 ``labels`` 里选一个（判断动作）。返回值必属于 labels。"""
        ...

    def generate(self, *, prompt: str, values: dict, history: tuple = ()) -> dict:
        """读私有 prompt + 变量（+ FALLBACK 时的历史），回一个 JSON 对象。

        生成型 ``model`` 动作用不到 ``history``；FALLBACK 逐步解释执行时靠它看「已经做过
        什么」（例如修了几次表头）。
        """
        ...


class ScriptedModel:
    """密闭测试桩：判断与生成都查表，错误率按种子+指纹确定性注入。

    * ``judge`` —— ``Callable[[prompt, values], label]`` 或 ``dict[fingerprint, label]``。
      给出「本该回什么」；错误率再在它之上确定性地翻错。
    * ``gen`` —— ``Callable[[prompt, values], dict]`` 或 ``dict[fingerprint, dict]``。
      生成型动作/FALLBACK 一步的回复。
    * ``error_rate`` + ``seed`` —— 对每次判断，按 ``sha256(seed:指纹)`` 派生一个 [0,1)
      的数，小于 ``error_rate`` 就把标签翻成同标签集里的另一个**非弃权**标签。同一输入
      永远同一结果（不推进全局 RNG），所以标定可复现。

    ``calls`` 记录每次调用的 (kind, prompt, values)，供测试断言调用形状。
    """

    def __init__(self, *, judge: Any = None, gen: Any = None,
                 error_rate: float = 0.0, seed: int = 0):
        self._judge = judge
        self._gen = gen
        self.error_rate = float(error_rate)
        self.seed = int(seed)
        self.calls: list[dict] = []

    # ---- Model 协议 ---- #
    def classify(self, *, prompt: str, values: dict, labels: list[str],
                 examples: tuple = ()) -> str:
        self.calls.append({"kind": "classify", "prompt": prompt,
                           "values": dict(values)})
        truth = self._lookup_judge(prompt, values, labels)
        if truth not in labels:
            raise ValueError(f"脚本给的标签 {truth!r} 不在 labels {labels} 里")
        if self.error_rate > 0 and self._should_err(prompt, values):
            wrong = [l for l in labels if l != truth and l != _abstain(labels)]
            if wrong:
                return self._pick_wrong(prompt, values, wrong)
        return truth

    def generate(self, *, prompt: str, values: dict, history: tuple = ()) -> dict:
        self.calls.append({"kind": "generate", "prompt": prompt,
                           "values": dict(values)})
        if callable(self._gen):
            return dict(self._gen(prompt, values, history))
        if isinstance(self._gen, dict):
            fp = _fingerprint(prompt, values)
            if fp in self._gen:
                return dict(self._gen[fp])
        raise KeyError(f"ScriptedModel.generate 没有脚本覆盖这次调用: {prompt[:60]!r}")

    # ---- 内部 ---- #
    def _lookup_judge(self, prompt: str, values: dict, labels: list[str]) -> str:
        if callable(self._judge):
            return self._judge(prompt, values)
        if isinstance(self._judge, dict):
            fp = _fingerprint(prompt, values)
            if fp in self._judge:
                return self._judge[fp]
        raise KeyError(f"ScriptedModel.classify 没有脚本覆盖这次判断: {prompt[:60]!r}")

    def _should_err(self, prompt: str, values: dict) -> bool:
        fp = _fingerprint(prompt, values)
        h = hashlib.sha256(f"{self.seed}:{fp}".encode("utf-8")).digest()
        r = int.from_bytes(h[:8], "big") / 2 ** 64
        return r < self.error_rate

    def _pick_wrong(self, prompt: str, values: dict, wrong: list[str]) -> str:
        fp = _fingerprint(prompt, values)
        h = hashlib.sha256(f"{self.seed}:wrong:{fp}".encode("utf-8")).digest()
        return wrong[int.from_bytes(h[:4], "big") % len(wrong)]


def _abstain(labels: list[str]) -> str:
    for cand in ("弃权", "abstain", "unknown"):
        if cand in labels:
            return cand
    return ""


def _fingerprint(head: str, values: dict) -> str:
    """(问题/提示, 变量取值) 的规范化指纹，用于查表与确定性错误注入。"""
    items = sorted((str(k), repr(v)) for k, v in values.items())
    return head.strip() + "|" + "|".join(f"{k}={v}" for k, v in items)
