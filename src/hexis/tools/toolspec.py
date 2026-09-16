"""Tool definitions: the interface description shared by the compiler and the execution backends.

The compiler treats tool names as opaque identifiers and learns about a tool only through :class:`ToolSpec`:
parameter fields, output fields, success criterion and primary output. Definitions come from three sources,
recorded in ``source`` by trustworthiness:

* ``registry``  an explicitly provided tool registry (JSON file); an interface guarantee;
* ``backend``   reported by an execution backend's ``describe_tools()``;
* ``inferred``  only observed in traces: it says "what has appeared", not an interface guarantee.

There is only one inference rule, the **status field convention** (:data:`STATUS_CONVENTIONS`): if the output has
``returncode``, success is ``returncode == 0``; if it has ``ok``, success is ``ok == True``; if it has neither,
success is not judged. This is a convention about field names, kept here rather than in the compiler; the
compiler only reads the ``success`` expression.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

#: status field conventions: (field name, success criterion, success value, failure value). Order is priority.
STATUS_CONVENTIONS: tuple[tuple[str, str, Any, Any], ...] = (
    ("returncode", "returncode == 0", 0, 1),
    ("ok", "ok == True", True, False),
)
#: these fields describe the call status rather than content; they are not semantic variables and not read by judge states.
STATUS_KEYS = frozenset({"ok", "returncode", "stderr", "error", "error_text", "exists",
                         "metadata", "opencode_part"})


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)      # field -> {type, required?, constant?}
    output_schema: dict = field(default_factory=dict)     # field -> type
    success: str = ""                                     # success criterion over output fields (guard expression)
    primary: Optional[str] = None                         # primary output field
    status_keys: list = field(default_factory=lambda: sorted(STATUS_KEYS))
    label: str = ""                                       # fixed phase label given by the registry (may be empty)
    source: str = "inferred"
    observed_inputs: Counter = field(default_factory=Counter)
    observed_outputs: Counter = field(default_factory=Counter)
    calls: int = 0

    # ---- derived ---- #
    def input_keys(self) -> list[str]:
        keys = list(self.input_schema) + [k for k in self.observed_inputs if k not in self.input_schema]
        return list(dict.fromkeys(keys))

    def output_keys(self) -> list[str]:
        keys = list(self.output_schema) + [k for k in self.observed_outputs if k not in self.output_schema]
        return list(dict.fromkeys(keys))

    def sure_outputs(self) -> frozenset:
        """Output fields guaranteed to appear: all fields declared by the registry; when inferred, the fields present in every call."""
        if self.source in ("registry", "backend") and self.output_schema:
            return frozenset(self.output_schema)
        if self.calls:
            return frozenset(k for k, n in self.observed_outputs.items() if n >= self.calls)
        return frozenset()

    def constant_keys(self) -> frozenset:
        return frozenset(k for k, v in self.input_schema.items()
                         if isinstance(v, Mapping) and v.get("constant"))

    def status_values(self, ok: Optional[bool]) -> dict:
        """When the call's success is known, the value of each field in the success criterion (for three-valued guard evaluation). Empty when unknown."""
        if ok is None or not self.success:
            return {}
        out: dict = {}
        try:
            tree = ast.parse(self.success, mode="eval")
        except SyntaxError:
            return {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
                    and isinstance(node.left, ast.Name) and isinstance(node.comparators[0], ast.Constant)):
                c = node.comparators[0].value
                if ok:
                    out[node.left.id] = c
                elif isinstance(c, bool):
                    out[node.left.id] = not c
                elif isinstance(c, (int, float)):
                    out[node.left.id] = c + 1
                else:
                    out[node.left.id] = f"not-{c}"
        return out

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema,
                "output_schema": self.output_schema, "success": self.success, "primary": self.primary,
                "status_keys": list(self.status_keys), "label": self.label, "source": self.source,
                "observed_inputs": dict(self.observed_inputs),
                "observed_outputs": dict(self.observed_outputs), "calls": self.calls}

    @classmethod
    def from_dict(cls, name: str, d: Mapping, *, source: str = "registry") -> "ToolSpec":
        return cls(name=name, description=str(d.get("description") or ""),
                   input_schema=dict(d.get("input_schema") or d.get("input") or {}),
                   output_schema=dict(d.get("output_schema") or d.get("output") or {}),
                   success=str(d.get("success") or ""), primary=d.get("primary"),
                   status_keys=list(d.get("status_keys") or sorted(STATUS_KEYS)),
                   label=str(d.get("label") or ""), source=source)


def infer_success(output_keys: Iterable[str]) -> str:
    keys = set(output_keys)
    for key, expr, _s, _f in STATUS_CONVENTIONS:
        if key in keys:
            return expr
    return ""


def infer_primary(observed_outputs: Mapping[str, int]) -> Optional[str]:
    """Primary output: the most frequent non-status field."""
    cands = [(n, k) for k, n in observed_outputs.items() if k not in STATUS_KEYS]
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[0], x[1]))
    return cands[0][1]


def observe(spec: ToolSpec, inp: Mapping, out: Mapping) -> None:
    """Record one call in the definition's observation counts; inferred definitions fill in the success criterion and primary output from them."""
    spec.calls += 1
    for k in (inp or {}):
        spec.observed_inputs[str(k)] += 1
    for k in (out or {}):
        spec.observed_outputs[str(k)] += 1
    if spec.source == "inferred":
        if not spec.success:
            spec.success = infer_success(spec.observed_outputs)
        spec.primary = infer_primary(spec.observed_outputs)


def registry_dict(registry: Mapping[str, ToolSpec]) -> dict:
    """A registry as JSON-ready data that :func:`load_registry` reads back unchanged."""
    return {"tools": {name: {"description": s.description, "input_schema": dict(s.input_schema),
                             "output_schema": dict(s.output_schema), "success": s.success, "primary": s.primary,
                             "status_keys": list(s.status_keys), "label": s.label}
                      for name, s in registry.items()}}


def load_registry(path: Any) -> dict[str, ToolSpec]:
    """Read a tool registry JSON: ``{"tools": {name: {...}}}`` or ``{name: {...}}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tools = data.get("tools", data) if isinstance(data, dict) else {}
    return {name: ToolSpec.from_dict(name, d) for name, d in tools.items()}


__all__ = ["STATUS_CONVENTIONS", "STATUS_KEYS", "ToolSpec", "infer_primary", "infer_success",
           "load_registry", "observe", "registry_dict"]
