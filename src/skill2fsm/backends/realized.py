"""模型实现的工具：机器里有、后端没有的工具，用模型把「这次调用」翻成一条 shell 命令去跑。

手写参考机器里的 ``inspect_workbook`` / ``apply_edits`` / ``audit_workbook`` 是假想工具，OpenCode 没有。
要把这台机器跑起来，就得有人实现它们。:class:`RealizedTools` 把这件事交给模型：拿工具定义（描述、
参数字段）和这一次的实参，让模型写一条能在工作目录里完成这次操作的 shell 命令，交给后端的 ``bash``
执行，结果按采集器封装（ok / returncode / stdout / stderr）原样返回。后端本来就有的工具直接透传。

这不是把假想工具「换成」bash：机器看到的仍是它自己的工具名和参数，只是执行方式是模型现场实现。
每次实现都是一次模型调用，记在同一个模型适配器的账上；命令原文写进结果的 ``realized_command``。
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from ..toolspec import ToolSpec

REALIZE_PROMPT = """You implement a tool call for an automated procedure. The procedure believes a tool named '{name}' exists; it does not, so you write ONE shell command that performs exactly this call in the current working directory and prints its result to stdout.

Tool description: {description}
Parameter fields: {fields}
This call's arguments: {args}

Rules: use python3 with a heredoc (python3 - <<'EOF' ... EOF) for anything longer than one line; exit with status 0 only if the operation succeeded, non-zero otherwise; print what the caller needs to read (reports, changed cells, check results); never modify files other than the ones named in the arguments; use the given paths verbatim.
Return exactly one JSON object {{"command": "<the shell command>"}}."""


class RealizedTools:
    """后端 + 模型 → 能执行机器里全部工具名的后端。``call(name, args) -> dict``。"""

    def __init__(self, native: Any, specs: Mapping[str, ToolSpec], model: Any, *,
                 shell_tool: str = "bash", shell_key: str = "command") -> None:
        self.native = native
        self.specs = dict(specs)
        self.model = model
        self.shell_tool = shell_tool
        self.shell_key = shell_key
        self.calls: list[dict] = []
        self.realized: list[dict] = []

    @property
    def available_tools(self) -> set:
        base = set(getattr(self.native, "available_tools", None) or getattr(self.native, "describe_tools", lambda: {})().keys())
        return base | {n for n, s in self.specs.items() if s.source in ("registry", "backend")}

    def describe_tools(self) -> dict:
        out = dict(getattr(self.native, "describe_tools", lambda: {})())
        out.update({n: s for n, s in self.specs.items() if n not in out})
        return out

    def call(self, name: str, inp: dict) -> dict:
        native_names = set(getattr(self.native, "available_tools", None) or ())
        if name in native_names or name not in self.specs:
            out = self.native.call(name, inp)
            self.calls.append({"name": name, "input": inp, "realized": False})
            return out
        spec = self.specs[name]
        fields = {k: (v if isinstance(v, dict) else {"type": str(v)}) for k, v in (spec.input_schema or {}).items()}
        prompt = REALIZE_PROMPT.format(name=name, description=spec.description or "(no description)",
                                       fields=json.dumps(fields, ensure_ascii=False),
                                       args=json.dumps(inp, ensure_ascii=False, default=str)[:6000])
        try:
            obj = self.model.generate(prompt=prompt, values={"tool": name, "arguments": inp})
        except Exception as exc:                                     # noqa: BLE001
            return {"ok": False, "returncode": 1, "stdout": "", "stderr": f"realize failed: {exc}"[:2000],
                    "error_kind": "realize"}
        cmd = str((obj or {}).get(self.shell_key) or "").strip()
        if not cmd:
            return {"ok": False, "returncode": 1, "stdout": "", "stderr": "realize returned no command",
                    "error_kind": "realize"}
        out = dict(self.native.call(self.shell_tool, {self.shell_key: cmd}))
        out["realized_command"] = cmd[:4000]
        self.realized.append({"name": name, "input": inp, "command": cmd[:4000],
                              "returncode": out.get("returncode")})
        self.calls.append({"name": name, "input": inp, "realized": True, "command": cmd[:400]})
        return out

    # 让 with 语句直接套在原后端上
    def __enter__(self) -> "RealizedTools":
        if hasattr(self.native, "__enter__"):
            self.native.__enter__()
        return self

    def __exit__(self, *exc) -> Optional[bool]:
        if hasattr(self.native, "__exit__"):
            return self.native.__exit__(*exc)
        return None


__all__ = ["REALIZE_PROMPT", "RealizedTools"]
