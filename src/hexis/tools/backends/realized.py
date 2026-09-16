"""Model-realized tools: for tools the machine has but the backend does not, the model translates "this call" into one shell command to run.

``inspect_workbook`` / ``apply_edits`` / ``audit_workbook`` in the hand-written reference machine are hypothetical
tools that OpenCode does not have. To run that machine, someone has to implement them. :class:`RealizedTools`
hands this to the model: given the tool definition (description, parameter fields) and this call's arguments, the
model writes one shell command that performs the operation in the working directory; it is executed by the
backend's ``bash``, and the result is returned as is, in the collector's wrapper (ok / returncode / stdout /
stderr). Tools the backend already has are passed straight through.

This does not "replace" the hypothetical tool with bash: the machine still sees its own tool name and arguments;
only the way it is executed is implemented by the model on the spot. Each realization is one model call, counted
on the same model adapter's account; the command text is written to ``realized_command`` in the result.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from hexis.tools.toolspec import ToolSpec

REALIZE_PROMPT = """You implement a tool call for an automated procedure. The procedure believes a tool named '{name}' exists; it does not, so you write ONE shell command that performs exactly this call in the current working directory and prints its result to stdout.

Tool description: {description}
Parameter fields: {fields}
This call's arguments: {args}

Rules: use python3 with a heredoc (python3 - <<'EOF' ... EOF) for anything longer than one line; exit with status 0 only if the operation succeeded, non-zero otherwise; print what the caller needs to read (reports, changed cells, check results); never modify files other than the ones named in the arguments; use the given paths verbatim.
Return exactly one JSON object {{"command": "<the shell command>"}}."""


class RealizedTools:
    """Backend + model -> a backend that can execute every tool name in the machine. ``call(name, args) -> dict``."""

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

    # let a with statement wrap the underlying backend directly
    def __enter__(self) -> "RealizedTools":
        if hasattr(self.native, "__enter__"):
            self.native.__enter__()
        return self

    def __exit__(self, *exc) -> Optional[bool]:
        if hasattr(self.native, "__exit__"):
            return self.native.__exit__(*exc)
        return None


__all__ = ["REALIZE_PROMPT", "RealizedTools"]
