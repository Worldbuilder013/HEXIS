"""Model and tool interfaces, plus a sealed, reproducible test stub.

Running a machine needs only two external capabilities: **calling tools** (``Tool``, a single functional-style
side effect) and **asking the model** (``Model``, judge or generate). Both are wrapped in narrow interfaces so
implementations are easy to swap and to stand in for in tests.

The model has two touch points: ``classify`` for judge actions (pick one from a fixed label set, abstain
included) and ``generate`` for generative actions and FALLBACK interpreted execution (read a private prompt +
variables, reply with one JSON object). Real implementations build the prompt and call the LLM inside these
two methods; :class:`ScriptedModel` looks everything up in tables -- no network, same input same output, with
errors injected deterministically from "seed + input fingerprint", so that the error rate of judge actions can
be both calibrated and reproduced.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from hexis.machine.schema import ABSTAIN, LEGACY_ABSTAIN


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class ModelUnavailable(RuntimeError):
    """The model endpoint could not be reached or refused the request (as opposed to an unusable answer)."""


@runtime_checkable
class Tool(Protocol):
    """One tool call: a dict ``input`` in, a dict ``output`` out. Deterministic, no network."""

    name: str

    def run(self, inp: dict) -> dict: ...


class FnTool:
    """Wraps a pure function as a :class:`Tool`."""

    def __init__(self, name: str, fn: Callable[[dict], dict]):
        self.name = name
        self._fn = fn

    def run(self, inp: dict) -> dict:
        return self._fn(inp)


class ToolRegistry:
    """Name -> tool. Called directly in-process, not in a sandbox (the tools of a toy environment are bundled trusted code)."""

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
            raise KeyError(f"no tool registered as {name!r}")
        return tool.run(inp)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@runtime_checkable
class Model(Protocol):
    """The two touch points: judge and generate."""

    def classify(self, *, prompt: str, values: dict, labels: list[str],
                 examples: tuple = ()) -> str:
        """Pick one of ``labels`` (judge action). The return value always belongs to labels."""
        ...

    def generate(self, *, prompt: str, values: dict, history: tuple = ()) -> dict:
        """Read a private prompt + variables (+ history, for FALLBACK) and reply with one JSON object.

        Generative ``model`` actions do not use ``history``; FALLBACK step-by-step interpreted execution relies on
        it to see "what has already been done" (e.g. how many times the header was fixed).
        """
        ...


class ScriptedModel:
    """Sealed test stub: both judge and generate are table lookups, with errors injected deterministically from seed + fingerprint.

    * ``judge`` -- ``Callable[[prompt, values], label]`` or ``dict[fingerprint, label]``.
      Gives "what it should answer"; the error rate then deterministically flips it on top of that.
    * ``gen`` -- ``Callable[[prompt, values], dict]`` or ``dict[fingerprint, dict]``.
      The reply for a generative action / one FALLBACK step.
    * ``error_rate`` + ``seed`` -- for each judgment, derive a number in [0,1) from ``sha256(seed:fingerprint)``;
      if it is below ``error_rate``, flip the label to another **non-abstain** label from the same label set.
      The same input always gives the same result (the global RNG is not advanced), so calibration is reproducible.

    ``calls`` records each call's (kind, prompt, values) so tests can assert on the call shape.
    """

    def __init__(self, *, judge: Any = None, gen: Any = None,
                 error_rate: float = 0.0, seed: int = 0):
        self._judge = judge
        self._gen = gen
        self.error_rate = float(error_rate)
        self.seed = int(seed)
        self.calls: list[dict] = []

    # ---- Model protocol ---- #
    def classify(self, *, prompt: str, values: dict, labels: list[str],
                 examples: tuple = ()) -> str:
        self.calls.append({"kind": "classify", "prompt": prompt,
                           "values": dict(values)})
        truth = self._lookup_judge(prompt, values, labels)
        if truth not in labels:
            raise ValueError(f"scripted label {truth!r} is not in labels {labels}")
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
        raise KeyError(f"ScriptedModel.generate: no script covers this call: {prompt[:60]!r}")

    # ---- internals ---- #
    def _lookup_judge(self, prompt: str, values: dict, labels: list[str]) -> str:
        if callable(self._judge):
            return self._judge(prompt, values)
        if isinstance(self._judge, dict):
            fp = _fingerprint(prompt, values)
            if fp in self._judge:
                return self._judge[fp]
        raise KeyError(f"ScriptedModel.classify: no script covers this judgment: {prompt[:60]!r}")

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
    for cand in (LEGACY_ABSTAIN, ABSTAIN, "unknown"):
        if cand in labels:
            return cand
    return ""


def _fingerprint(head: str, values: dict) -> str:
    """Canonical fingerprint of (question/prompt, variable values), used for table lookup and deterministic error injection."""
    items = sorted((str(k), repr(v)) for k, v in values.items())
    return head.strip() + "|" + "|".join(f"{k}={v}" for k, v in items)
