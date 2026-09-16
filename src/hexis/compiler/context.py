"""Compile context: the compile inputs of one skill, all taken from the document, tool definitions, traces and
explicit rules.

The compiler does not assume that a skill processes files, makes modifications or uses some library. All it knows is:

* task input fields               ← ``task.input`` in trace headers (only that a field appeared; ``present_in``
                                    counts occurrences)
* tools and their interfaces      ← the tool registry / backend definitions (guaranteed by the interface),
                                    supplemented by trace observations (inferred, not guaranteed)
* label rules                     ← skill rules: give events derived labels (such as "read the output after a
                                    modification"), with document quotes
* steps that must be performed    ← skill rules: must_occur / before / forbid, with document quotes
* completion and acceptance       ← skill rules: which evidence a terminal needs and which events invalidate it

Rules come from ``compile.json`` in the skill directory, a ``--rules`` file, or are extracted from the document by
the model during initialization and verified against their quotes (see :mod:`.init`). The update algorithm itself
only runs these checks and makes no decisions based on skill names.

Event patterns (:class:`EventPattern`) are the atoms of rules: they match on event kind, tool name, label, argument
text and outcome. ``${field}`` in ``args_contain`` is substituted with the current task's input value (for path-like
values the last segment, the file name, also matches).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from hexis.tools.toolspec import STATUS_KEYS, ToolSpec, observe
from hexis.traces.trace_adapter import tool_output

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class FieldSpec:
    name: str
    present_in: int = 0
    total: int = 0
    sample: Any = None
    source: str = "traces"

    @property
    def always_present(self) -> bool:
        return self.total > 0 and self.present_in == self.total


@dataclass
class EventPattern:
    kind: Optional[str] = None            # tool / model / judge / user / end
    tool: Optional[str] = None
    label: Optional[str] = None           # base or derived label
    role: Optional[str] = None            # model events: narration / output
    args_contain: Optional[str] = None    # the argument text must contain this (may contain ${field})
    arg_regex: Optional[str] = None
    success: Optional[bool] = None
    after: Optional["EventPattern"] = None    # an event matching it must have occurred earlier

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None and k != "after"}
        if self.after is not None:
            d["after"] = self.after.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "EventPattern":
        after = d.get("after")
        return cls(kind=d.get("kind"), tool=d.get("tool"), label=d.get("label"), role=d.get("role"),
                   args_contain=d.get("args_contain"), arg_regex=d.get("arg_regex"),
                   success=d.get("success"),
                   after=cls.from_dict(after) if isinstance(after, Mapping) else None)

    def describe(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self.to_dict().items() if k != "after"]
        if self.after is not None:
            parts.append(f"after=({self.after.describe()})")
        return " ".join(parts) or "any event"


@dataclass
class LabelRule:
    label: str
    when: EventPattern
    quote: str = ""
    source: str = "rules"


@dataclass
class Requirement:
    id: str
    kind: str                              # must_occur / before / forbid
    a: EventPattern
    b: Optional[EventPattern] = None
    quote: str = ""
    source: str = "rules"

    def describe(self) -> str:
        if self.kind == "before":
            return f"{self.id}: ({self.a.describe()}) must precede ({self.b.describe() if self.b else ''})"
        return f"{self.id}: {self.kind} ({self.a.describe()})"


@dataclass
class TerminalCondition:
    terminal: str
    required_evidence: list = field(default_factory=list)      # list[EventPattern]
    invalidating_events: list = field(default_factory=list)    # list[EventPattern]
    quote: str = ""
    source: str = "rules"


@dataclass
class CompileContext:
    skill_id: str
    skill_text: str
    clauses: list = field(default_factory=list)                # [(id, text, locator)]
    task_inputs: dict = field(default_factory=dict)            # name → FieldSpec
    tools: dict = field(default_factory=dict)                  # name → ToolSpec
    terminals: list = field(default_factory=list)              # [{"id", "kind"}]
    label_rules: list = field(default_factory=list)
    requirements: list = field(default_factory=list)
    terminal_conditions: list = field(default_factory=list)
    event_kinds: Counter = field(default_factory=Counter)
    n_traces: int = 0
    notes: list = field(default_factory=list)

    # ---- conveniences ---- #
    def spec(self, tool: str) -> ToolSpec:
        s = self.tools.get(tool)
        if s is None:
            s = ToolSpec(name=tool, source="inferred")
            self.tools[tool] = s
        return s

    def input_keys(self) -> list[str]:
        return list(self.task_inputs)

    def default_terminal(self) -> str:
        """The first non-fallback terminal without evidence conditions."""
        conditioned = {tc.terminal for tc in self.terminal_conditions if tc.required_evidence}
        for t in self.terminals:
            if t["id"] not in conditioned and t.get("kind") != "fallback":
                return t["id"]
        return self.terminals[0]["id"] if self.terminals else "done"

    def conditioned_terminals(self) -> list[str]:
        return [tc.terminal for tc in self.terminal_conditions if tc.required_evidence]

    def patterns(self) -> list[EventPattern]:
        """All patterns that appear in the rules (used to decide the "required states")."""
        out: list[EventPattern] = []
        for r in self.requirements:
            out.append(r.a)
            if r.b is not None:
                out.append(r.b)
        for tc in self.terminal_conditions:
            out.extend(tc.required_evidence)
            out.extend(tc.invalidating_events)
        return out

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id, "n_traces": self.n_traces,
            "task_inputs": {k: {"present_in": v.present_in, "total": v.total, "source": v.source,
                                "sample": _clip(v.sample)} for k, v in self.task_inputs.items()},
            "tools": {k: v.to_dict() for k, v in self.tools.items()},
            "terminals": list(self.terminals),
            "label_rules": [{"label": r.label, "when": r.when.to_dict(), "quote": r.quote,
                             "source": r.source} for r in self.label_rules],
            "requirements": [{"id": r.id, "kind": r.kind, "a": r.a.to_dict(),
                              "b": r.b.to_dict() if r.b else None, "quote": r.quote,
                              "source": r.source} for r in self.requirements],
            "terminal_conditions": [{"terminal": tc.terminal,
                                     "required_evidence": [p.to_dict() for p in tc.required_evidence],
                                     "invalidating_events": [p.to_dict() for p in tc.invalidating_events],
                                     "quote": tc.quote, "source": tc.source}
                                    for tc in self.terminal_conditions],
            "event_kinds": dict(self.event_kinds), "notes": list(self.notes),
        }


def _clip(v: Any, n: int = 120) -> Any:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


# --------------------------------------------------------------------------- #
# Rule files
# --------------------------------------------------------------------------- #
def parse_rules(data: Mapping, *, source: str = "rules") -> dict:
    """Rules JSON → (terminals, label_rules, requirements, terminal_conditions)."""
    terminals = [dict(t) if isinstance(t, Mapping) else {"id": str(t), "kind": ""}
                 for t in (data.get("terminals") or [])]
    labels = [LabelRule(label=str(r["label"]), when=EventPattern.from_dict(r.get("when") or {}),
                        quote=str(r.get("quote") or ""), source=source)
              for r in (data.get("labels") or data.get("label_rules") or []) if r.get("label")]
    reqs = []
    for i, r in enumerate(data.get("requirements") or [], 1):
        kind = str(r.get("kind") or "must_occur")
        if kind not in ("must_occur", "before", "forbid"):
            continue
        reqs.append(Requirement(id=str(r.get("id") or f"R{i}"), kind=kind,
                                a=EventPattern.from_dict(r.get("a") or r.get("when") or {}),
                                b=EventPattern.from_dict(r["b"]) if isinstance(r.get("b"), Mapping) else None,
                                quote=str(r.get("quote") or ""), source=source))
    tcs = [TerminalCondition(terminal=str(t["terminal"]),
                             required_evidence=[EventPattern.from_dict(p) for p in (t.get("required_evidence") or [])],
                             invalidating_events=[EventPattern.from_dict(p) for p in (t.get("invalidating_events") or [])],
                             quote=str(t.get("quote") or ""), source=source)
           for t in (data.get("terminal_conditions") or []) if t.get("terminal")]
    return {"terminals": terminals, "label_rules": labels, "requirements": reqs,
            "terminal_conditions": tcs}


def _is_parsed(rules: Any) -> bool:
    """True for the output of :func:`parse_rules`, False for rules JSON (which may also use the key label_rules)."""
    if not isinstance(rules, dict) or "label_rules" not in rules:
        return False
    items = list(rules.get("label_rules") or []) + list(rules.get("requirements") or []) \
        + list(rules.get("terminal_conditions") or [])
    return all(isinstance(x, (LabelRule, Requirement, TerminalCondition)) for x in items)


def rules_dict(ctx: "CompileContext") -> dict:
    """The context's skill rules in the rules JSON format (``compile.json``) that :func:`load_rules` reads."""
    d = ctx.to_dict()
    return {"terminals": d["terminals"], "labels": d["label_rules"], "requirements": d["requirements"],
            "terminal_conditions": d["terminal_conditions"]}


def load_rules(path: Any) -> dict:
    return parse_rules(json.loads(Path(path).read_text(encoding="utf-8")))


def quote_in_document(quote: str, doc: str) -> bool:
    """Quote verification: a substring of the document once whitespace differences are ignored. An empty quote does
    not count."""
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()
    q = norm(quote)
    return bool(q) and q in norm(doc)


# --------------------------------------------------------------------------- #
# Event matching (dynamic: on trace events)
# --------------------------------------------------------------------------- #
def substitute(template: str, task_input: Mapping) -> list[str]:
    """Substitute task inputs for ``${field}``. Path-like values also get a variant with only the last segment.
    Returns the list of possible texts."""
    if not template:
        return []
    outs = [template]
    for m in _VAR.findall(template):
        val = task_input.get(m)
        if val is None:
            return []
        s = str(val)
        alts = [s]
        if "/" in s and s.rsplit("/", 1)[-1]:
            alts.append(s.rsplit("/", 1)[-1])
        outs = [o.replace("${" + m + "}", a) for o in outs for a in alts]
    return outs


def event_matches(pat: EventPattern, ev: Any, task_input: Mapping, earlier: Sequence[Any] = ()) -> bool:
    """Whether event ``ev`` matches the pattern. ``ev`` must have kind / tool / label / labels / role / ok / args_text."""
    if pat.kind is not None and getattr(ev, "kind", None) != pat.kind:
        return False
    if pat.tool is not None and getattr(ev, "tool", None) != pat.tool:
        return False
    if pat.label is not None:
        labels = set(getattr(ev, "labels", ()) or ()) | {getattr(ev, "label", "") or ""}
        if pat.label not in labels:
            return False
    if pat.role is not None and getattr(ev, "role", None) != pat.role:
        return False
    if pat.success is not None and bool(getattr(ev, "ok", True)) != pat.success:
        return False
    text = getattr(ev, "args_text", "") or ""
    if pat.args_contain is not None:
        alts = substitute(pat.args_contain, task_input)
        if not alts or not any(a in text for a in alts):
            return False
    if pat.arg_regex is not None:
        try:
            if not re.search(pat.arg_regex, text):
                return False
        except re.error:
            return False
    if pat.after is not None:
        if not any(event_matches(pat.after, e, task_input, ()) for e in earlier):
            return False
    return True


def apply_labels(events: Sequence[Any], ctx: CompileContext, task_input: Mapping) -> None:
    """Add derived labels to events according to the label rules (in place, in rule order; later rules see the labels
    added by earlier ones)."""
    for i, ev in enumerate(events):
        if not hasattr(ev, "labels"):
            continue
        for rule in ctx.label_rules:
            if event_matches(rule.when, ev, task_input, events[:i]):
                ev.labels.add(rule.label)


def check_requirements(events: Sequence[Any], ctx: CompileContext, task_input: Mapping) -> list[str]:
    """Which requirements a trace violates. Returns a list of explanations (empty = all satisfied)."""
    out: list[str] = []
    for r in ctx.requirements:
        hits_a = [i for i, e in enumerate(events) if event_matches(r.a, e, task_input, events[:i])]
        if r.kind == "must_occur" and not hits_a:
            out.append(f"{r.id}: missing required event ({r.a.describe()})" + (f" \"{r.quote}\"" if r.quote else ""))
        elif r.kind == "forbid" and hits_a:
            out.append(f"{r.id}: forbidden event occurred ({r.a.describe()})" + (f" \"{r.quote}\"" if r.quote else ""))
        elif r.kind == "before" and r.b is not None:
            for j, e in enumerate(events):
                if event_matches(r.b, e, task_input, events[:j]) and not any(i < j for i in hits_a):
                    out.append(f"{r.id}: event {j + 1} ({r.b.describe()}) is not preceded by ({r.a.describe()})"
                               + (f" \"{r.quote}\"" if r.quote else ""))
                    break
    return out


def evidence_held(events: Sequence[Any], ctx: CompileContext, task_input: Mapping) -> set:
    """Evidence {(terminal, i)} still held after all events. Successful evidence events add it; invalidating events and
    failed evidence events remove it."""
    held: set = set()
    for j, ev in enumerate(events):
        earlier = events[:j]
        for tc in ctx.terminal_conditions:
            for inv in tc.invalidating_events:
                if event_matches(inv, ev, task_input, earlier):
                    held -= {(tc.terminal, i) for i in range(len(tc.required_evidence))}
            for i, pat in enumerate(tc.required_evidence):
                relaxed = EventPattern(**{**pat.__dict__, "success": None})
                if not event_matches(relaxed, ev, task_input, earlier):
                    continue
                ok = bool(getattr(ev, "ok", True))
                if pat.success is None or ok == pat.success:
                    held.add((tc.terminal, i))
                elif pat.success and not ok:
                    held.discard((tc.terminal, i))
    return held


def terminal_for(events: Sequence[Any], ctx: CompileContext, task_input: Mapping) -> tuple[str, set]:
    """τ*(T): the first conditioned terminal whose evidence is complete; otherwise the default terminal."""
    held = evidence_held(events, ctx, task_input)
    for tc in ctx.terminal_conditions:
        if tc.required_evidence and all((tc.terminal, i) in held for i in range(len(tc.required_evidence))):
            return tc.terminal, held
    return ctx.default_terminal(), held


# --------------------------------------------------------------------------- #
# Building the context
# --------------------------------------------------------------------------- #
def _args_text(inp: Any) -> str:
    try:
        return json.dumps(inp, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(inp)


def build_context(skill_id: str, skill_text: str, clauses: Sequence[tuple], traces: Iterable[Any], *,
                  registry: Optional[Mapping[str, ToolSpec]] = None,
                  rules: Optional[Mapping] = None) -> CompileContext:
    """Build the compile context from the skill document, the tool registry, traces and rules. ``traces`` is a
    sequence of Trace objects."""
    ctx = CompileContext(skill_id=skill_id, skill_text=skill_text, clauses=list(clauses))
    for name, spec in (registry or {}).items():
        ctx.tools[name] = ToolSpec(**{**spec.__dict__, "observed_inputs": Counter(),
                                      "observed_outputs": Counter(), "calls": 0})
    traces = list(traces)
    ctx.n_traces = len(traces)
    for tr in traces:
        task_in = tr.task.get("input") if isinstance(tr.task, dict) else None
        for k, v in (task_in or {}).items():
            fs = ctx.task_inputs.setdefault(str(k), FieldSpec(name=str(k), sample=v))
            fs.present_in += 1
        for r in tr.records:
            act = r.action or {}
            kind = str(act.get("kind") or "")
            ctx.event_kinds[kind] += 1
            if kind == "tool":
                name = str(act.get("name") or "")
                inp = act.get("input") or act.get("args") or {}
                observe(ctx.spec(name), inp if isinstance(inp, Mapping) else {}, tool_output(r))
    for fs in ctx.task_inputs.values():
        fs.total = len(traces)
    if rules:
        parsed = rules if _is_parsed(rules) else parse_rules(rules)
        ctx.terminals = list(parsed["terminals"])
        ctx.label_rules = list(parsed["label_rules"])
        ctx.requirements = list(parsed["requirements"])
        ctx.terminal_conditions = list(parsed["terminal_conditions"])
    if not ctx.terminals:
        ctx.terminals = [{"id": "done", "kind": "done"}]
        ctx.notes.append("the rules give no terminal table, using the single terminal done")
    known = {t["id"] for t in ctx.terminals}
    for tc in ctx.terminal_conditions:
        if tc.terminal not in known:
            ctx.terminals.append({"id": tc.terminal, "kind": tc.terminal.lower()})
            known.add(tc.terminal)
    inferred = sorted(n for n, s in ctx.tools.items() if s.source == "inferred")
    if inferred:
        ctx.notes.append(f"tools {inferred} have no registry definition: their interface comes only from trace observations and is not guaranteed")
    for k, v in ctx.task_inputs.items():
        if not v.always_present:
            ctx.notes.append(f"task input {k} appears in only {v.present_in}/{v.total} traces")
    return ctx


__all__ = ["CompileContext", "EventPattern", "FieldSpec", "LabelRule", "Requirement",
           "STATUS_KEYS", "TerminalCondition", "apply_labels", "build_context", "check_requirements",
           "event_matches", "evidence_held", "load_rules", "parse_rules", "quote_in_document", "rules_dict",
           "substitute", "terminal_for"]
