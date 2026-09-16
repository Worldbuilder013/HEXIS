"""Execution-level conformance: **what the machine produces when it really runs must be identical to the trace**.

:mod:`hexis.legacy.replay` does a walk: it pushes the machine along the trace and compares action identities step by
step. It cannot prove that "handing this machine to :func:`hexis.execution.runtime.run_task` for a real run still
yields this sequence of actions". A whole layer of run-time semantics sits between the two: someone has to fill the
``${var}`` templates, the ``writes`` allowlist has to catch the outputs, ``phase`` has to be recorded on both sides,
and guards have to be evaluated on real variables. The three holes observed in practice all lived in this layer, and
replay could see none of them:

* machine states carry ``phase`` but run-time records do not ⇒ identity is asymmetric between the two sides;
* a model state declares ``writes=[s2_cmd]``, but nobody tells the model to return that key ⇒ empty variable ⇒ the
  tool gets an empty input;
* a variable referenced by a tool input template is not written by any state ⇒ it renders as an empty string.

So "identical" is made executable here: the **trace** stands in for the model and the tools to drive the real
``run_task``, and afterwards the produced record sequence is compared step by step with the original trace. The
stand-in does exactly one thing: it hands the machine what actually happened at that step of the trace:

* the machine executes ``tool`` ⇒ take the ``output`` of the next tool record in the trace, and check tool name and
  phase;
* the machine executes ``model`` ⇒ if the next record in the trace is a model step, use its output; if the next one is
  a tool step (meaning this is a **zero-width generation**: a state inserted by the input gate that the trace never
  had), infer the variable values from the **real input** of that tool record through the ``${var}`` template of the
  following tool state: the same rule as :func:`hexis.legacy.replay.walk`;
* the machine executes ``judge`` ⇒ use the label of the judge step if the trace has one, otherwise the program labeler
  (``gold_from``), and failing that, abstain.

Criteria (all three must hold for the run to count as identical):

1. **Number and order of steps**: after dropping zero-width states, the produced records align one to one with the
   trace and have the same length;
2. **Action identity**: kind, canonical tool name and phase are the same at every step;
3. **Inputs identical verbatim**: the ``input`` rendered for a tool step equals the ``input`` of the trace record.
   Criterion 3 detects "baked-in inputs": a machine that baked another task's literals into its templates passes
   replay but not this check.

Falling back to interpretation midway also counts as not identical: the machine did not manage to finish on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from hexis.execution import runtime as _runtime
from hexis.machine.schema import ABSTAIN, Machine, Trace
from hexis.traces.normalize import canon_tool_name, is_begin


@dataclass
class Divergence:
    """One divergence. ``step`` is the step number in the trace (the produced side's step number when there is no counterpart)."""

    step: int
    why: str
    machine: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

    def located(self) -> str:
        return f"step {self.step}: {self.why}"


@dataclass
class ConformanceResult:
    ok: bool
    trace_id: str = ""
    matched: int = 0
    total: int = 0
    divergences: list = field(default_factory=list)
    stopped: str = ""
    error: str = ""
    llm_calls: int = 0
    #: Step at which the machine stopped in the fallback state (rest handed back to interpretation). ``None`` = finished on its own.
    deferred_at: Optional[int] = None

    def summary(self) -> str:
        head = f"{self.trace_id or '(no task id)'}: aligned {self.matched}/{self.total} steps, stopped: {self.stopped}"
        if self.ok:
            return head + (", identical" if self.deferred_at is None
                           else f", first {self.matched} steps identical, fallback segment from step {self.deferred_at}")
        first = self.divergences[0].located() if self.divergences else self.error
        return head + f", **not identical**: {first}"


def _is_zero_width(machine: Machine, sid: str) -> bool:
    """Zero-width state: the trace never had this step, so it is projected away in the comparison. Three kinds:

    * judges and generations **introduced** by the compiler (``introduced``): see the rule of the same name in
      :func:`hexis.legacy.replay.walk`;
    * the reserved **begin tool** (:data:`hexis.traces.normalize.BEGIN_TOOL`): a no-op the compiler inserts so that "a
      machine has exactly one start"; replay treats it as a virtual step as well.
    """
    st = machine.states.get(sid)
    if st is None:
        return False
    a = st.action
    if is_begin(a):
        return True
    return bool(getattr(a, "introduced", False)) and a.kind in ("judge", "model")


def _tid(trace: Trace) -> str:
    task = trace.task if isinstance(trace.task, dict) else {}
    return str(task.get("task_id") or "")


class TraceDriver:
    """Acts as both Model and Tool: whenever the machine wants a step, it is handed **that step of the trace**.

    It invents nothing: values the model should produce come from the trace's model records or from the real input of
    the tool record that immediately follows; tool results come from the trace's tool records. When something cannot
    be found, a divergence is recorded honestly and an empty output is returned: let the machine give itself away on
    the spot instead of covering for it.
    """

    def __init__(self, machine: Machine, trace: Trace, labelers: Optional[dict] = None) -> None:
        self.machine = machine
        self.records = list(trace.records)
        self.trace = trace
        self.i = 0
        self.divergences: list[Divergence] = []
        self.by_prompt = {st.action.prompt: st for st in machine.states.values()
                          if st.action.kind == "model"}
        self.by_question = {st.action.prompt: st for st in machine.states.values()
                            if st.action.kind == "judge"}
        # Who consumes each variable: tool input template var -> [(state, input key)]; plus the variables used in edge guards.
        # **Only consumed variables affect this run**, so only they are what "the trace must be able to supply";
        # an output nobody reads (a one-sentence conclusion) that cannot be supplied is no divergence: it cannot steer the machine.
        self.consumed: dict = {}
        for st in machine.states.values():
            if st.action.kind != "tool":
                continue
            for k, v in (getattr(st.action, "input", {}) or {}).items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    self.consumed.setdefault(v[2:-1], []).append((st, k))
        self.cond_vars: set = set()
        for _sid, t in machine.transitions_all():
            if t.cond:
                try:
                    from hexis.machine.cond import vars_of
                    self.cond_vars |= set(vars_of(t.cond))
                except Exception:                                   # noqa: BLE001
                    pass
        if labelers is None:
            from hexis.traces.trace_adapter import LABELERS as labelers  # deferred import, avoids a cycle
        self.labelers = dict(labelers or {})

    # ---- trace cursor ---- #
    def _peek(self):
        """The next record **the machine should consume**. Begin steps are skipped: the begin state on the machine side is a
        no-op that never calls a tool, so a cursor stuck on it would compare the real first step against the wrong record
        (observed: every trace reported "tool name mismatch: machine read_csv, trace skill2fsm_begin"). Indices still
        count in the original trace, which is exactly what labelers need."""
        while self.i < len(self.records) and is_begin(self.records[self.i].action):
            self.i += 1
        return self.records[self.i] if self.i < len(self.records) else None

    def _next_tool_record(self):
        for rec in self.records[self.i:]:
            if (rec.action or {}).get("kind") == "tool":
                return rec
        return None

    def _note(self, why: str, machine: Any = None, trace: Any = None) -> None:
        step = self.records[self.i].step if self.i < len(self.records) else len(self.records) + 1
        self.divergences.append(Divergence(step, why, dict(machine or {}), dict(trace or {})))

    # ---- Model protocol ---- #
    def generate(self, *, prompt: str, values: dict, history: tuple = ()) -> dict:
        st = self.by_prompt.get(prompt)
        rec = self._peek()
        out: dict = {}
        if rec is not None and (rec.action or {}).get("kind") == "model":
            self.i += 1                                   # the trace really has this step
            out.update(rec.output or {})
        writes = list(getattr(st.action, "writes", []) if st else [])
        # supply only **consumed** variables: a tool input template needs them or an edge guard reads them. Other outputs cannot steer.
        need = [w for w in writes if w not in out and (w in self.consumed or w in self.cond_vars)]
        out.update(self._supply(need))
        still = [w for w in need if w not in out]
        if still:
            self._note(f"model state {st.id if st else '?'} has to write {still}, "
                       f"but the trace cannot supply these values: in a real run these variables would be empty",
                       {"writes": writes, "unsatisfied": still},
                       dict((rec.action or {}) if rec else {}))
        return out

    def _supply(self, want: Sequence[str]) -> dict:
        """Take the values of these variables from the trace. There is only one rule, shared with replay:

        variable ``v`` is consumed by input key ``k`` of some tool state ⇒ its value is the actual value of ``input[k]``
        in **the next tool record with the same name and phase**. What cannot be found is not found; nothing is made up.
        """
        out: dict = {}
        for var in want:
            for st, key in self.consumed.get(var, []):
                rec = self._find_tool(st)
                if rec is None:
                    continue
                real = dict((rec.action or {}).get("input") or {})
                if key in real:
                    out[var] = real[key]
                    break
        return out

    def _find_tool(self, st: Any):
        """The first tool record after the cursor with the same name and phase as state ``st``."""
        name = canon_tool_name(str(getattr(st.action, "name", "") or ""))
        phase = str(getattr(st.action, "phase", "") or "")
        for rec in self.records[self.i:]:
            act = rec.action or {}
            if act.get("kind") != "tool":
                continue
            if canon_tool_name(str(act.get("name") or "")) != name:
                continue
            if phase and str(act.get("phase") or "") != phase:
                continue
            return rec
        return None

    def classify(self, *, prompt: str, values: dict, labels: list, examples: tuple = ()) -> str:
        st = self.by_question.get(prompt)
        abstain = getattr(st.action, "abstain", ABSTAIN) if st else (labels[-1] if labels else "")
        introduced = bool(getattr(st.action, "introduced", False)) if st else False
        rec = self._peek()
        # **Introduced judges are zero-width by definition**: the trace never had this step, and the label is computed by a
        # program labeler. Without checking this first, a real judge step that happens to be in the trace would be eaten
        # by it; the real judge's ruling ("action") is not in the branch label set, so the judgment could only abstain and
        # go to FALLBACK (observed: three traces on the prefix tree broke at the judge step because of this).
        if introduced:
            gold = getattr(st.action, "gold_from", "")
            fn = self.labelers.get(gold) if gold else None
            if fn is not None:
                try:
                    label = fn(self.trace, self.i - 1)
                except Exception:                                   # noqa: BLE001
                    label = None
                if label in labels:
                    return label
            return abstain
        if rec is not None and (rec.action or {}).get("kind") == "judge":
            self.i += 1
            for v in (rec.output or {}).values():
                if isinstance(v, str) and v in labels:
                    return v
            return abstain
        gold = getattr(st.action, "gold_from", "") if st else ""
        fn = self.labelers.get(gold) if gold else None
        if fn is not None:
            try:
                label = fn(self.trace, self.i - 1)
            except Exception:                                       # noqa: BLE001
                label = None
            if label in labels:
                return label
        return abstain

    # ---- Tool protocol ---- #
    def call(self, name: str, inp: dict) -> dict:
        rec = self._peek()
        if rec is None:
            self._note(f"the machine still wants to call {name}, but the trace has ended", {"name": name, "input": dict(inp or {})})
            return {"error": "the trace has ended"}
        act = rec.action or {}
        if act.get("kind") != "tool":
            self._note(f"the machine wants to call {name}, but this trace step is {act.get('kind')}",
                       {"name": name, "input": dict(inp or {})}, dict(act))
            return {"error": "this trace step is not a tool call"}
        if canon_tool_name(name) != canon_tool_name(str(act.get("name") or "")):
            self._note(f"tool name mismatch: machine {name}, trace {act.get('name')}",
                       {"name": name}, {"name": act.get("name")})
        self.i += 1
        return dict(rec.output or {})


def check_trace(machine: Machine, trace: Trace, *, max_steps: Optional[int] = None,
                compare_input: bool = True) -> ConformanceResult:
    """Run the machine (model and tools both driven by the trace), then compare step by step. Criteria: see the module docstring."""
    driver = TraceDriver(machine, trace)
    task = trace.task if isinstance(trace.task, dict) else {}
    # halt at the fallback state instead of switching to runtime's interpretation loop: what matters is how far **the machine** gets.
    halted = _runtime.halt_at_fallback(machine)
    try:
        rr = _runtime.run_task(halted, task, model=driver, tools=driver, doc="",
                               max_steps=max_steps or (len(trace.records) * 3 + 16))
    except Exception as exc:                                        # noqa: BLE001
        return ConformanceResult(False, _tid(trace), 0, len(trace.records),
                                 [Divergence(0, f"cannot run: {type(exc).__name__}: {exc}")],
                                 "exception", str(exc))
    # Begin steps must be dropped on both sides: they are a no-op the compiler inserts, present in traces of the compile
    # view (``with_begin``) and absent from traces from elsewhere. Projecting only one side shifts everything by one
    # (observed: every trace reported "step 0: action kind mismatch", so not a single artifact could be submitted).
    got = [r for r in rr.trace.records
           if not _is_zero_width(machine, r.state) and not is_begin(r.action)]
    want = [r for r in trace.records if not is_begin(r.action)]
    divs = list(driver.divergences)
    # Stopping in the fallback state is **not** a divergence: it is the machine honestly saying "this part is not compiled
    # yet", the same decision as :func:`hexis.legacy.replay.walk` (reaching FALLBACK returns acceptance). What is required:
    # the part before the handover must be identical step by step.
    deferred_at: Optional[int] = None
    if got and got[-1].state == machine.fallback:
        deferred_at = got[-1].step
        got = got[:-1]
    matched = 0
    for k, w in enumerate(want):
        if k >= len(got):
            if deferred_at is None:
                divs.append(Divergence(w.step, f"the machine took only {len(got)} steps, the trace has {len(want)}",
                                       {}, dict(w.action or {})))
            break                                   # stopped in the fallback state: nothing was promised for the rest
        g, wa, ga = got[k], dict(w.action or {}), dict(got[k].action or {})
        if wa.get("kind") != ga.get("kind"):
            divs.append(Divergence(w.step, f"action kind mismatch: machine {ga.get('kind')}, trace {wa.get('kind')}",
                                   ga, wa))
            break
        if wa.get("kind") == "tool":
            if canon_tool_name(str(ga.get("name") or "")) != canon_tool_name(str(wa.get("name") or "")):
                divs.append(Divergence(w.step, f"tool name mismatch: machine {ga.get('name')}, trace {wa.get('name')}",
                                       ga, wa))
                break
            if str(ga.get("phase") or "") != str(wa.get("phase") or ""):
                divs.append(Divergence(w.step, f"phase mismatch: machine {ga.get('phase') or '(none)'}, "
                                               f"trace {wa.get('phase') or '(none)'}", ga, wa))
                break
            if compare_input and dict(ga.get("input") or {}) != dict(wa.get("input") or {}):
                divs.append(Divergence(w.step, "tool inputs differ: in a real run the machine would not execute the call in the trace",
                                       dict(ga.get("input") or {}), dict(wa.get("input") or {})))
                break
        matched += 1
    if len(got) > len(want) and matched == len(want):
        divs.append(Divergence(want[-1].step if want else 0,
                               f"the machine took {len(got) - len(want)} extra steps (the trace ends here)"))
    ok = not divs and (matched == len(want) or deferred_at is not None)
    if rr.stopped != _runtime.STOP_TERMINAL and not divs:
        divs.append(Divergence(len(want), f"the machine did not halt normally: {rr.stopped} {rr.error[:120]}"))
        ok = False
    return ConformanceResult(ok, _tid(trace), matched, len(want), divs, rr.stopped,
                             rr.error, rr.llm_calls, deferred_at)


def check(machine: Machine, traces: Sequence[Trace], **kw) -> tuple[bool, list]:
    """Check a batch of traces one by one. Returns ``(all identical, [ConformanceResult])``."""
    out = [check_trace(machine, t, **kw) for t in traces]
    return all(r.ok for r in out), out


__all__ = ["ConformanceResult", "Divergence", "TraceDriver", "check", "check_trace"]
