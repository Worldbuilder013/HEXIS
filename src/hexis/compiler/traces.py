"""Trace preprocessing (turning records into events).

Traces are read in file name order. Every record folds into an **event** of one of five kinds: model generation,
tool call, judge result, user input, task end. Every event keeps its own input, output and dependencies; tool names
are kept verbatim, never merged or renamed. Records of an unrecognized kind are not dropped: the whole trace is
marked as "containing unrecognized events" and does not take part in the update.

* Consecutive calls of the same tool with the same base label merge into one event (several calls = a loop need).
* Base labels split only generic command tools into apply / probe, by "whether there is a write operation"; other
  tools use the label given by the registry; otherwise the label is empty. **Derived labels** (such as "read the
  output after a modification") all come from skill rules (see :mod:`.context`).
* Model events are narration (text between two operations, attached to the intent of the next event) or output
  (structured output, or the last generation before the end). The latter are observable events.
* The end class τ*(T) is determined by the skill's terminal conditions; a trace whose claimed terminal disagrees
  with the evidence is a violation.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hexis.compiler.context import CompileContext, apply_labels, check_requirements, terminal_for
from hexis.machine import cond as _cond
from hexis.machine.schema import Machine, Trace
from hexis.tools.toolspec import ToolSpec
from hexis.traces import judge as _judge
from hexis.traces import phases as _phases
from hexis.traces.trace_adapter import load_any_trace, tool_output

EVENT_KINDS = ("tool", "model", "judge", "user", "end")


@dataclass
class Call:
    step: int
    input: dict
    output: dict
    ok: bool
    narration: str = ""


@dataclass
class Event:
    kind: str                                   # tool / model / judge / user / end
    index: int = 0
    tool: str = ""                              # tool events
    label: str = ""                             # base label
    labels: set = field(default_factory=set)    # derived labels
    calls: list = field(default_factory=list)   # all calls of a tool event
    role: str = ""                              # model events: narration / output
    text: str = ""                              # body of a model event / input of a user event
    output: dict = field(default_factory=dict)  # output of model / judge events
    step: int = 0                               # record step number of non-tool events
    terminal: str = ""                          # terminal claimed by an end event
    intent: str = ""                            # attached narration

    @property
    def ok(self) -> bool:
        if self.kind == "tool":
            return bool(self.calls) and bool(self.calls[-1].ok)
        return True

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def observable(self) -> bool:
        return self.kind in ("tool", "user", "end") or (self.kind == "model" and self.role == "output")

    @property
    def args_text(self) -> str:
        if self.kind == "tool":
            return "\n".join(_dumps(c.input) for c in self.calls)
        if self.kind == "model":
            return self.text
        return _dumps(self.output)

    def describe(self) -> str:
        if self.kind == "tool":
            labs = "+".join(sorted(self.labels))
            return f"{self.tool}{'/' + self.label if self.label else ''}{'[' + labs + ']' if labs else ''}" \
                   f"×{self.n_calls}{'✓' if self.ok else '✗'}"
        if self.kind == "model":
            return f"model:{self.role}"
        if self.kind == "end":
            return f"end{':' + self.terminal if self.terminal else ''}"
        return self.kind


Segment = Event


@dataclass
class Prepared:
    trace_id: str
    source: str
    verdict: str
    task_input: dict
    events: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)      # [(step, kind)]
    claims: str = ""                                     # terminal claimed by the trace itself
    tau: str = ""                                        # end class computed from the terminal conditions
    evidence: set = field(default_factory=set)
    violation: Optional[str] = None
    requirement_violations: list = field(default_factory=list)
    n_records: int = 0

    @property
    def observable(self) -> list:
        return [e for e in self.events if e.observable]

    @property
    def tool_events(self) -> list:
        return [e for e in self.events if e.kind == "tool"]

    def obs(self) -> list[tuple]:
        """Obs(T): the observable event sequence as (kind, name, outcome), ending with (end, τ*)."""
        out: list[tuple] = []
        for e in self.observable:
            if e.kind == "tool":
                out.append(("tool", e.tool, e.ok))
            elif e.kind == "model":
                out.append(("model", "", True))
            elif e.kind == "user":
                out.append(("user", "", True))
            elif e.kind == "end":
                out.append(("end", self.tau, True))
        return out


# --------------------------------------------------------------------------- #
def _dumps(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(v)


def _restore_code(inp: dict, meta: dict) -> dict:
    """Put the code bodies extracted during collection (meta.code_bodies) back into the arguments; the arguments are
    only for the record, not for comparison."""
    bodies = meta.get("code_bodies") if isinstance(meta, dict) else None
    if not isinstance(bodies, dict):
        return inp
    out = dict(inp)
    for k in list(out):
        if k.endswith("_sha256") and out[k] in bodies:
            out[k[:-7]] = bodies[out[k]]
            del out[k]
    return out


def base_label(tool: str, inp: Mapping, spec: ToolSpec) -> str:
    """Base label. Generic command tools are split into apply / probe by write operations; others follow the
    registry; otherwise empty."""
    if tool.lower() in _phases.GENERIC_TOOLS:
        text = _phases.command_text(inp)
        if text.strip():
            return _phases.PHASE_APPLY if _phases.writes_something(text) else _phases.PHASE_PROBE
    return spec.label or ""


def call_ok(spec: ToolSpec, out: Mapping) -> bool:
    """Outcome of one call: evaluate the tool's success condition on the output; a missing or unevaluable condition
    counts as success."""
    if not spec.success:
        return True
    try:
        return bool(_cond.evaluate(spec.success, dict(out)))
    except _cond.CondError:
        return True


def segment_trace(trace: Trace, ctx: CompileContext) -> tuple[list, list, list]:
    """Trace → event sequence. Returns (events, notes, unrecognized records)."""
    events: list[Event] = []
    notes: list[str] = []
    unsupported: list = []
    pending: list[str] = []
    for r in trace.records:
        act = dict(r.action or {})
        kind = str(act.get("kind") or "")
        if kind == "model":
            out = dict(r.output or {})
            said = str(out.get("reply") or act.get("text") or "").strip()
            structured = {k: v for k, v in out.items() if k not in ("reply", "text")}
            ev = Event(kind="model", step=r.step, text=said[:2000], output=structured,
                       role="output" if structured else "narration", intent=" ".join(pending))
            events.append(ev)
            if said:
                pending.append(said[:400])
            continue
        if kind == "end":
            events.append(Event(kind="end", step=r.step, terminal=str(act.get("terminal") or ""),
                                intent=" ".join(pending)))
            pending = []
            continue
        if kind == "judge":
            events.append(Event(kind="judge", step=r.step, output=dict(r.output or {}),
                                text=str(act.get("prompt") or ""), intent=" ".join(pending)))
            pending = []
            continue
        if kind == "user":
            out = dict(r.output or {})
            events.append(Event(kind="user", step=r.step, output=out,
                                text=str(out.get("answer") or act.get("prompt") or ""),
                                intent=" ".join(pending)))
            pending = []
            continue
        if kind != "tool":
            unsupported.append((r.step, kind or "(empty)"))
            continue
        name = str(act.get("name") or "")
        spec = ctx.spec(name)
        inp = act.get("input") or act.get("args") or {}
        inp = _restore_code(deepcopy(inp) if isinstance(inp, dict) else {}, r.meta or {})
        out = deepcopy(tool_output(r))
        ok = call_ok(spec, out)
        label = base_label(name, inp, spec)
        call = Call(step=r.step, input=inp, output=out, ok=ok, narration=" ".join(pending))
        # merge consecutive calls with the same tool and label; calls separated only by narration (not output) count as
        # consecutive
        prev = None
        for cand in reversed(events):
            if cand.kind == "model" and cand.role == "narration":
                continue
            prev = cand
            break
        if prev is not None and prev.kind == "tool" and prev.tool == name and prev.label == label:
            prev.calls.append(call)
            if pending:
                prev.intent = (prev.intent + " " + " ".join(pending)).strip()
        else:
            events.append(Event(kind="tool", tool=name, label=label, calls=[call],
                                intent=" ".join(pending)))
        pending = []
    if pending and not (events and events[-1].kind == "end"):
        notes.append(f"{len(pending)} trailing narration passages are not followed by any event")
    # the last model generation before the end is the deliverable (output); earlier ones are narration
    last_model = None
    for i, ev in enumerate(events):
        if ev.kind == "model":
            last_model = i
        elif ev.kind in ("tool", "user"):
            last_model = None
    if last_model is not None and events and events[-1].kind == "end":
        events[last_model].role = "output"
    for i, ev in enumerate(events):
        ev.index = i
    return events, notes, unsupported


def prepare(trace: Trace, ctx: CompileContext, *, source: str = "") -> Prepared:
    """All preprocessing results for one trace: events, derived labels, requirement checks, end class, violation."""
    tid = str(trace.task.get("task_id") or Path(source).stem) if isinstance(trace.task, dict) else source
    task_in = dict((trace.task.get("input") if isinstance(trace.task, dict) else None) or {})
    events, notes, unsupported = segment_trace(trace, ctx)
    p = Prepared(trace_id=tid, source=source, verdict=trace.verdict, task_input=task_in,
                 events=events, notes=notes, unsupported=unsupported, n_records=len(trace.records))
    if not events:
        return p
    if events[-1].kind != "end":
        events.append(Event(kind="end", index=len(events), terminal=""))
        p.notes.append("trace has no end record, appended an end event")
    apply_labels(events, ctx, task_in)
    p.requirement_violations = check_requirements(events, ctx, task_in)
    p.tau, p.evidence = terminal_for(events, ctx, task_in)
    p.claims = events[-1].terminal
    conditioned = set(ctx.conditioned_terminals())
    if p.claims in conditioned and p.claims != p.tau:
        p.violation = f"claimed to reach {p.claims}, but the evidence only supports {p.tau}"
    return p


def load_traces(tdir: Path, *, phase_rules: str = "") -> list[tuple[Path, Optional[Trace], str]]:
    """Read the *.jsonl files of a directory in file name order. Unreadable ones become (path, None, error). No phase
    classification: labels are decided by this module."""
    out: list = []
    for f in sorted(Path(tdir).glob("*.jsonl")):
        try:
            out.append((f, load_any_trace(f, phase_rules=phase_rules), ""))
        except Exception as exc:                               # noqa: BLE001
            out.append((f, None, f"{type(exc).__name__}: {exc}"[:300]))
    return out


def violates_prohibitions(trace: Trace, machine: Machine) -> Optional[str]:
    """External prohibition rules (the machine's prohibitions). Returns the reason when one is violated."""
    if not machine.prohibitions:
        return None
    v = _judge.evaluate(trace, None, list(machine.prohibitions))
    return v.reason if v.verdict == "rejected" else None


def end_state_for(machine: Machine, terminal: str) -> str:
    """End class (terminal id) → end state: the state of that terminal; if missing, a non-fallback terminal; failing
    that, the fallback state."""
    kinds = terminal_kinds(machine)
    for sid, st in machine.states.items():
        if st.action.kind == "end" and st.action.terminal == terminal:
            return sid
    for sid, st in machine.states.items():
        if st.action.kind == "end" and sid != machine.fallback and kinds.get(st.action.terminal) != "fallback":
            return sid
    return machine.fallback


def terminal_kinds(m: Machine) -> dict[str, str]:
    return {t.id: t.kind for t in m.terminals}


__all__ = ["Call", "EVENT_KINDS", "Event", "Prepared", "Segment", "base_label", "call_ok",
           "end_state_for", "load_traces", "prepare", "segment_trace", "violates_prohibitions"]
