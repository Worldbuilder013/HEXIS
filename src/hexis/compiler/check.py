"""Candidate machine checks (rule driven).

Variables and evidence: a fixed point over incoming edges. A_q is an intersection (the variables available before
    entering q); Z_q is a union whose elements are "sets of evidence that may be held". Evidence comes from the
    skill's terminal conditions: for a state that statically matches required_evidence, the success branch
    establishes the evidence, the failure branch invalidates it, and when the branch cannot be told both are
    possible; a state matching invalidating_events invalidates all evidence of that terminal. A conditioned terminal
    requires every set in Z_q to contain all of its evidence; the fallback terminal must not be conditioned. The
    skill's must_occur / before / forbid requirements are also checked statically on the graph. Together with the
    structural checks this forms Check(M).
Trace path check Replay(M,T,π̃): advance variable values along the path, using placeholder values for task inputs
    and model outputs and the real result of the event's last call for tool outputs; judge states pick the label that
    leads to the next state; the transition taken at every step must agree with the path; variables a tool needs
    must be assigned and non-empty; the path must finish in an end state; the observable event sequences are compared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hexis.compiler.common import (
    branch_class,
    guaranteed,
    is_observable,
    need,
    reachable,
    seed_vars,
    static_match,
    terminal_kinds,
)
from hexis.compiler.context import CompileContext, EventPattern
from hexis.compiler.traces import Prepared
from hexis.execution.runtime import bind_outputs, rebuild
from hexis.machine import checks as _checks
from hexis.machine import cond as _cond
from hexis.machine.schema import Machine, State

ZSet = frozenset  # frozenset[frozenset[tuple[str, int]]]


# --------------------------------------------------------------------------- #
# Static analysis
# --------------------------------------------------------------------------- #
def _relaxed(pat: EventPattern) -> EventPattern:
    return EventPattern(**{**pat.__dict__, "success": None, "after": None})


def post(z: ZSet, st: State, guard: str, ctx: CompileContext, m: Machine) -> ZSet:
    """Post(Z, a, g): the possible values of the evidence sets after state st and guard g."""
    if st.action.kind not in ("tool", "model", "user"):
        return z
    cls = branch_class(guard, ctx, st) if st.action.kind == "tool" else "success"
    out: set = set()
    for held in z:
        variants = [set(held)]
        for tc in ctx.terminal_conditions:
            ids = {(tc.terminal, i) for i in range(len(tc.required_evidence))}
            if any(static_match(_relaxed(inv), st, m) for inv in tc.invalidating_events):
                variants = [v - ids for v in variants]
            for i, pat in enumerate(tc.required_evidence):
                if not static_match(_relaxed(pat), st, m):
                    continue
                eid = (tc.terminal, i)
                if pat.success is None:
                    variants = [v | {eid} for v in variants]
                elif cls == "success":
                    variants = [(v | {eid}) if pat.success else (v - {eid}) for v in variants]
                elif cls == "failure":
                    variants = [(v - {eid}) if pat.success else (v | {eid}) for v in variants]
                else:
                    variants = [v | {eid} for v in variants] + [v - {eid} for v in variants]
        out.update(frozenset(v) for v in variants)
    return frozenset(out)


@dataclass
class Analysis:
    avail: dict            # A_q
    zstate: dict           # Z_q
    reach: set


def analyze(m: Machine, ctx: CompileContext) -> Analysis:
    reach = reachable(m)
    universe = set(seed_vars(m))
    for st in m.states.values():
        universe |= set(guaranteed(st, ctx))
    avail = {sid: frozenset(universe) for sid in reach}
    zst: dict = {sid: frozenset() for sid in reach}
    avail[m.initial] = seed_vars(m)
    zst[m.initial] = frozenset({frozenset()})
    incoming: dict[str, list] = {sid: [] for sid in reach}
    for src in reach:
        for t in m.states[src].transitions:
            if t.to in incoming:
                incoming[t.to].append((src, t))
    for _ in range(len(m.states) + 2):
        changed = False
        for sid in reach:
            a_new = set(seed_vars(m)) if sid == m.initial else set(universe)
            z_new: set = {frozenset()} if sid == m.initial else set()
            for src, t in incoming[sid]:
                a_new &= set(avail[src]) | set(guaranteed(m.states[src], ctx))
                z_new |= set(post(zst[src], m.states[src], t.cond, ctx, m))
            if not incoming[sid] and sid != m.initial:
                a_new = set()
            if frozenset(a_new) != avail[sid] or frozenset(z_new) != zst[sid]:
                avail[sid], zst[sid] = frozenset(a_new), frozenset(z_new)
                changed = True
        if not changed:
            break
    return Analysis(avail=avail, zstate=zst, reach=reach)


def _avoiding_reach(m: Machine, blocked: set) -> set:
    if m.initial in blocked or m.initial not in m.states:
        return set()
    seen, stack = {m.initial}, [m.initial]
    while stack:
        u = stack.pop()
        for t in m.states[u].transitions:
            if t.to in seen or t.to in blocked or t.to not in m.states:
                continue
            seen.add(t.to)
            stack.append(t.to)
    return seen


def requirement_findings(m: Machine, ctx: CompileContext) -> list[str]:
    """Static checks of skill requirements: must_occur (every path to a non-fallback terminal passes it), before,
    forbid."""
    out: list[str] = []
    kinds = terminal_kinds(m)
    ends = {sid for sid, st in m.states.items()
            if st.action.kind == "end" and sid != m.fallback and kinds.get(st.action.terminal) != "fallback"}
    for r in ctx.requirements:
        a_states = {sid for sid, st in m.states.items() if static_match(_relaxed(r.a), st, m)}
        if r.kind == "forbid":
            if a_states:
                out.append(f"requirement {r.id}: states {sorted(a_states)} match a forbidden event")
            continue
        if r.kind == "must_occur":
            if not a_states:
                out.append(f"requirement {r.id}: no state can produce the required event ({r.a.describe()})")
                continue
            leaked = ends & _avoiding_reach(m, a_states)
            if leaked:
                out.append(f"requirement {r.id}: terminals {sorted(leaked)} are reachable without passing {sorted(a_states)}")
        elif r.kind == "before" and r.b is not None:
            b_states = {sid for sid, st in m.states.items() if static_match(_relaxed(r.b), st, m)}
            bad = sorted(b_states & _avoiding_reach(m, a_states))
            if bad:
                out.append(f"requirement {r.id}: {bad} reachable before ({r.a.describe()})")
    return out


def check(m: Machine, ctx: CompileContext) -> list[str]:
    """Check(M) = G_structure ∧ G_variables ∧ G_evidence ∧ G_requirements. An empty list = passed."""
    out: list[str] = list(_checks.structural_findings(m))
    if m.initial not in m.states:
        return out
    an = analyze(m, ctx)
    conditions = {tc.terminal: tc for tc in ctx.terminal_conditions if tc.required_evidence}
    for sid in sorted(an.reach):
        st = m.states[sid]
        a_q, z_q = an.avail[sid], an.zstate[sid]
        after = set(a_q) | set(guaranteed(st, ctx))
        miss = sorted(need(st) - a_q)
        if miss:
            out.append(f"variable: {sid} needs {miss}, not guaranteed to be defined on entry")
        for t in st.transitions:
            if t.cond:
                try:
                    gv = _cond.vars_of(t.cond)
                except _cond.CondError as exc:
                    out.append(f"guard: {sid}→{t.to} {t.cond!r} failed to parse: {exc}")
                    continue
                bad = sorted(set(gv) - after)
                if bad:
                    out.append(f"variable: guard of {sid}→{t.to} reads {bad}, still undefined after the action")
            if t.to in an.reach:
                nxt = an.avail[t.to]
                if not nxt <= after:
                    out.append(f"variable: {sid}→{t.to} cannot provide {sorted(nxt - after)}")
                zp = post(z_q, st, t.cond, ctx, m)
                if not zp <= an.zstate[t.to]:
                    out.append(f"evidence: evidence sets of {sid}→{t.to} are not within the invariant of {t.to}")
        if st.action.kind == "end":
            tc = conditions.get(st.action.terminal)
            if tc is not None:
                ids = {(tc.terminal, i) for i in range(len(tc.required_evidence))}
                lacking = sorted({i for held in z_q for (_t, i) in (ids - held)})
                if lacking or not z_q:
                    names = [tc.required_evidence[i].describe() for i in lacking] or ["all"]
                    out.append(f"evidence: terminal {sid}({tc.terminal}) may lack evidence {names}")
    fb = m.states.get(m.fallback)
    if fb is not None and fb.action.kind == "end" and fb.action.terminal in conditions:
        out.append(f"evidence: fallback state {m.fallback} must not use a terminal with evidence conditions")
    out += requirement_findings(m, ctx)
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- #
# Trace path check
# --------------------------------------------------------------------------- #
@dataclass
class ReplayResult:
    ok: bool
    why: str = ""
    path: list = field(default_factory=list)
    events: list = field(default_factory=list)
    visits: dict = field(default_factory=dict)


def _pick(m: Machine, st: State, values: dict):
    """Pick an edge in run-time order: the first transition that holds. Returns (edge, error)."""
    for t in st.ordered_transitions():
        if not t.cond:
            return t, ""
        try:
            if _cond.evaluate(t.cond, values):
                return t, ""
        except _cond.CondError as exc:
            return None, f"guard {t.cond!r} of {st.id} failed to evaluate: {exc}"
    return None, ""


def _candidates(st: State, var: str, values: dict) -> list:
    """Constants that a variable written by this state is compared with in its outgoing guards: the values replay may
    choose from. The placeholder comes first."""
    opts: list = [f"<{var}>"]
    for t in st.transitions:
        if not t.cond:
            continue
        try:
            atoms = _cond.atoms_of(t.cond)
        except _cond.CondError:
            continue
        for a in atoms:
            if a.var != var:
                continue
            consts = list(a.const) if isinstance(a.const, tuple) else [a.const]
            for c in consts:
                if c not in opts and c is not None:
                    opts.append(c)
    return opts


def _assignments(st: State, values: dict) -> list[dict]:
    """Candidate value combinations for the variables a state writes. Judge states go by label; model states by the
    constants that appear in outgoing guards, with placeholders for the rest."""
    a = st.action
    if a.kind == "judge":
        return [{a.writes[0]: label} for label in list(a.labels)]
    if a.kind in ("model", "user"):
        decisive = [w for w in a.writes if any(w in _cond.vars_of(t.cond) for t in st.transitions if t.cond)]
        base = {w: f"<{w}>" for w in a.writes}
        if not decisive:
            return [base]
        out: list[dict] = [dict(base)]
        for w in decisive:
            out = [{**d, w: c} for d in out for c in _candidates(st, w, values)]
        return out
    return [{}]


def _simulate_zero(m: Machine, sid: str, values: dict, target: str, *,
                   ignore_counters: bool, depth: int = 0) -> Optional[dict]:
    """Whether target can be reached from sid through zero-width states only (simulated with the run-time edge
    selection rule).

    Returns the values each zero-width state should write {state: {variable: value}}: the chosen label for judge
    states, and for model states the constants in outgoing guards that lead to the next state (placeholder values for
    the other variables)."""
    if depth > len(m.states) + 2:
        return None
    st = m.states.get(sid)
    if st is None:
        return None
    if is_observable(st):
        return {} if sid == target else None
    for assign in _assignments(st, values):
        vals = dict(values)
        vals.update(assign)
        t, err = _pick(m, st, vals)
        if err or t is None:
            continue
        if t.inc and not ignore_counters:
            vals[t.inc] = int(vals.get(t.inc) or 0) + 1
        r = _simulate_zero(m, t.to, vals, target, ignore_counters=ignore_counters, depth=depth + 1)
        if r is not None:
            return {sid: assign, **r}
    return None


def _choose_outputs(m: Machine, st: State, values: dict, target: Optional[str], *,
                    ignore_counters: bool) -> dict:
    """Observable model / user states: write values that make the next hop lead to target; with no target, or when
    nothing works, use placeholder values."""
    options = _assignments(st, values)
    if target is None:
        return options[0]
    for assign in options:
        vals = dict(values)
        vals.update(assign)
        t, err = _pick(m, st, vals)
        if err or t is None:
            continue
        if t.inc and not ignore_counters:
            vals[t.inc] = int(vals.get(t.inc) or 0) + 1
        if _simulate_zero(m, t.to, vals, target, ignore_counters=ignore_counters) is not None:
            return assign
    return options[0]


def placeholders(m: Machine) -> dict:
    vals: dict = {}
    for v in m.variables:
        if v.init_from:
            vals[v.name] = f"<{v.init_from}>"
        elif v.init is not None and v.init != "":
            vals[v.name] = v.init
    return vals


def replay(m: Machine, prep: Prepared, anchors: list[str], *, max_steps: int = 0,
           ignore_counters: bool = False) -> ReplayResult:
    """Replay(M, T, π̃). ``anchors`` holds the state id of every observable event (the last one is the end state).

    With ``ignore_counters`` counter variables are not incremented (bound exits never fire); this is used only to
    count visits along the candidate path."""
    evs = prep.observable
    if len(anchors) != len(evs):
        return ReplayResult(False, f"path length {len(anchors)} does not match the number of observable events {len(evs)}")
    values = placeholders(m)
    expect = prep.obs()
    events: list = []
    path: list = []
    visits: dict = {}
    cur = m.initial
    i = 0
    limit = max_steps or (len(m.states) + 2) * (len(anchors) + 2) + 8
    for _ in range(limit):
        st = m.states.get(cur)
        if st is None:
            return ReplayResult(False, f"path hit a nonexistent state {cur}", path, events)
        path.append(cur)
        visits[cur] = visits.get(cur, 0) + 1
        a = st.action
        if is_observable(st):
            if i >= len(anchors) or anchors[i] != cur:
                want = anchors[i] if i < len(anchors) else "(none)"
                return ReplayResult(False, f"reached {cur}, the path requires {want}", path, events)
            ev = evs[i]
            if a.kind == "tool":
                miss = [x for x in sorted(need(st)) if values.get(x) in (None, "")]
                if miss:
                    return ReplayResult(False, f"{cur} needs {miss}, unassigned or empty at this step", path, events)
                out = dict(ev.calls[-1].output) if ev.calls else {}
                got = rebuild(bind_outputs(out, getattr(a, "binds", {}) or {}), list(a.writes))
                for w in a.writes:
                    got.setdefault(w, "")
                values.update(got)
                events.append(("tool", a.name, bool(ev.ok)))
            elif a.kind in ("model", "user"):
                nxt = anchors[i + 1] if i + 1 < len(anchors) else None
                values.update(_choose_outputs(m, st, values, nxt, ignore_counters=ignore_counters))
                events.append((a.kind, "", True))
            elif a.kind == "end":
                events.append(("end", a.terminal, True))
                if i != len(anchors) - 1:
                    return ReplayResult(False, f"ended at {cur}, but the path still has {len(anchors) - 1 - i} events", path, events)
                if events != expect:
                    return ReplayResult(False, f"event sequences differ: machine {events} vs trace {expect}", path, events)
                return ReplayResult(True, "", path, events, visits)
            i += 1
        elif a.kind in ("model", "judge"):
            if i >= len(anchors):
                return ReplayResult(False, f"zero-width state {cur} appears after the end of the path", path, events)
            sim = _simulate_zero(m, cur, values, anchors[i], ignore_counters=ignore_counters)
            if sim is None:
                what = "label" if a.kind == "judge" else "value"
                return ReplayResult(False, f"{'judge' if a.kind == 'judge' else 'model'} state {cur} has no {what} leading to {anchors[i]}",
                                    path, events)
            values.update(sim[cur])
        else:
            return ReplayResult(False, f"action kind {a.kind} of {cur} cannot be replayed", path, events)
        chosen, err = _pick(m, st, values)
        if err:
            return ReplayResult(False, err, path, events)
        if chosen is None:
            return ReplayResult(False, f"{cur} has no transition to take", path, events)
        if chosen.inc and not ignore_counters:
            values[chosen.inc] = int(values.get(chosen.inc) or 0) + 1
        cur = chosen.to
    return ReplayResult(False, f"no end state reached within {limit} steps", path, events)


__all__ = ["Analysis", "ReplayResult", "analyze", "check", "placeholders", "post", "replay",
           "requirement_findings"]
