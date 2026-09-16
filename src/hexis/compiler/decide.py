"""Collect step decisions for one trace from a decider, one step at a time.

The stepwise update (:mod:`hexis.compiler.stepwise`) applies a decision per trace step: the step is produced by
an existing state (``match``), needs a new state (``new``), or is not part of the skill's workflow (``ignore``);
a whole trace can also be excluded. :func:`decide_trace` walks the observable steps of a trace, computes the
candidates of each step exactly as the update does, and asks a callback for the decision. Its result is the
decision spec that :func:`hexis.compiler.stepwise.update_with_decisions` consumes.

The callback is where a model, a person or a script decides; this module never calls a model itself. Decisions
depend on the position reached by the earlier decisions, so the walk is sequential. When a step is ignored, the
trace is prepared again without it (neighbouring calls of the same tool can merge) and the walk restarts from the
first step; a caching decider answers the unchanged questions without asking again. The walk ends because every
restart ignores one more step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from hexis.compiler.align import NEW
from hexis.compiler.context import CompileContext
from hexis.compiler.stepwise import candidates, pre_status, prepare_steps, propose
from hexis.compiler.traces import Event, Prepared, end_state_for
from hexis.machine.schema import Machine, Trace

DECISIONS = ("match", "new", "ignore", "exclude")


@dataclass
class StepView:
    """Everything a decider needs to decide one step."""

    trace: str                               # trace key
    prep: Prepared                           # the trace as prepared with the steps ignored so far
    event: Event                             # the step to decide
    step: int                                # step number in the unmodified trace (``orig``)
    position: Optional[str]                  # state of the previous step; None before the first step
    previous_ok: Optional[bool]              # success of the previous tool step
    tier1: list = field(default_factory=list)    # candidates reachable from the position
    tier2: list = field(default_factory=list)    # other candidates (a new transition would be needed)
    proposal: dict = field(default_factory=dict)  # deterministic default decision (not an instruction)
    terminal_state: str = ""                 # end state implied by the trace's terminal conditions
    before: list = field(default_factory=list)   # [(step, event description, decision summary)]
    ignored: list = field(default_factory=list)  # step numbers ignored so far
    after: list = field(default_factory=list)    # descriptions of the following observable steps

    @property
    def is_new_position(self) -> bool:
        return bool(self.position) and str(self.position).startswith(NEW)


#: ``ask(view) -> {"decision", "state", "purpose", "clause"}``; the answer must already be valid for the view
Ask = Callable[[StepView], dict]


def _summary(decision: dict) -> str:
    d = decision.get("decision", "")
    return f"match {decision.get('state')}" if d == "match" else d


def decide_trace(machine: Machine, ctx: CompileContext, trace: Trace, *, ask: Ask, source: str = "",
                 key: str = "", ignore_calls: Sequence[int] = ()) -> dict:
    """Ask for a decision on every non-end observable step and return the decision spec for the trace."""
    ignore: list[int] = []
    while True:
        prep = prepare_steps(trace, ctx, source=source, ignore=ignore, ignore_calls=ignore_calls)
        spec_ignored = {str(i): {"d": "ignore"} for i in ignore}
        if pre_status(prep, trace, machine) is not None:
            return {"accept_proposals": False, "ignore_calls": list(ignore_calls), "steps": dict(spec_ignored)}
        term = end_state_for(machine, prep.tau)
        obs = prep.observable
        steps: dict = {}
        before: list = []
        pos: Optional[str] = None
        ok: Optional[bool] = None
        restart = False
        for k, ev in enumerate(obs):
            if ev.kind == "end":
                break
            orig = getattr(ev, "orig", ev.index)
            t1, t2 = candidates(machine, ctx, ev, pos, ok, term, allow_tier2=True)
            view = StepView(trace=key or prep.trace_id, prep=prep, event=ev, step=orig, position=pos,
                            previous_ok=ok, tier1=list(t1), tier2=list(t2),
                            proposal=propose(machine, ev, t1, t2, term), terminal_state=term,
                            before=list(before), ignored=sorted(ignore),
                            after=[e.describe() for e in obs[k + 1:]])
            answer = ask(view)
            d = answer.get("decision")
            if d == "exclude":
                return {"exclude": str(answer.get("purpose") or "excluded by the decider")}
            if d == "ignore":
                ignore.append(orig)
                restart = True
                break
            entry = {"d": d, "purpose": str(answer.get("purpose") or ""), "clause": str(answer.get("clause") or "")}
            if d == "match":
                entry["state"] = str(answer.get("state"))
                pos = entry["state"]
            else:
                pos = f"{NEW}{ev.index}"
            steps[str(orig)] = entry
            ok = ev.ok if ev.kind == "tool" else None
            before.append((orig, ev.describe(), _summary(answer)))
        if not restart:
            steps.update(spec_ignored)
            return {"accept_proposals": False, "ignore_calls": list(ignore_calls), "steps": steps}


__all__ = ["Ask", "DECISIONS", "StepView", "decide_trace"]
