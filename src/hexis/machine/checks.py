"""Structural checks: deterministic and free, run after every compile edit. Empty list = all pass.

Graph-level problems ("part of the artifact cannot be reached / cannot stop / has nowhere to
go") are three pure graph algorithms on the machine: reachability, termination and transition
completeness. Data-level problems are two declaration checks: **mutual exclusion and
completeness** (Theorem 2: the guarded outgoing edges of a state are pairwise never true at the
same time, and together they cover every configuration) and **write before read** (every
variable a state reads has already been written on every path leading to it).

Mutual exclusion and completeness work by compressing "infinitely many variable values" into
"finitely many configurations" and enumerating them; the compression is based on the atomic
predicates in the guards (see :mod:`hexis.machine.cond`). A numeric variable can only flip an
edge's truth value at the thresholds it is compared against, so one representative value
between thresholds suffices; a variable written by a judge action only takes values from its
finite label set.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Optional

from hexis.machine import cond
from hexis.machine.schema import Machine


def structural_findings(machine: Machine) -> list[str]:
    """Graph and data-flow problems decidable deterministically. Empty list = all pass."""
    out: list[str] = []
    if machine.initial not in machine.states:
        return [f"initial points to a missing state {machine.initial!r}"]
    for t in machine.transitions_all():
        for side, sid in (("from", t[0]), ("to", t[1].to)):
            if sid not in machine.states:
                out.append(f"transition {t[0]}→{t[1].to}: {side} points to a missing state "
                           f"{sid!r}")
    if out:
        return out

    reachable = _reachable(machine)
    out += _reachability(machine, reachable)
    out += _termination(machine, reachable)
    out += _completeness(machine, reachable)
    out += _determinism(machine, reachable)          # mutual exclusion and completeness (Theorem 2)
    out += _loop_bounds(machine, reachable)
    out += _write_before_read(machine, reachable)
    out += _output_completeness(machine, reachable)
    return out


# --------------------------------------------------------------------------- #
def _reachable(machine: Machine) -> set[str]:
    seen, stack = {machine.initial}, [machine.initial]
    while stack:
        cur = stack.pop()
        for t in machine.out_edges(cur):
            if t.to not in seen:
                seen.add(t.to)
                stack.append(t.to)
    return seen


def _reachability(machine: Machine, reachable: set[str]) -> list[str]:
    """FALLBACK is a reserved state: having no edge into it yet is not dead code (the machine
    connects to it as it grows)."""
    return [f"state {sid} is unreachable from {machine.initial}: dead code, delete it or "
            "connect it with an edge"
            for sid in sorted(set(machine.states) - reachable)
            if sid != machine.fallback]


def _termination(machine: Machine, reachable: set[str]) -> list[str]:
    terminals = {sid for sid, s in machine.states.items() if s.action.kind == "end"}
    if not terminals:
        return ["no end state (kind end): this machine can never halt normally"]
    can_stop, changed = set(terminals), True
    while changed:
        changed = False
        for src, t in machine.transitions_all():
            if t.to in can_stop and src not in can_stop:
                can_stop.add(src)
                changed = True
    return [f"state {sid} cannot reach any end state: once entered it never leaves and "
            "will hit max_steps"
            for sid in sorted(reachable - can_stop)]


def _completeness(machine: Machine, reachable: set[str]) -> list[str]:
    out: list[str] = []
    for sid in sorted(reachable):
        st = machine.states[sid]
        if st.action.kind == "end":
            continue
        edges = st.transitions
        if not edges:
            out.append(f"state {sid} has no outgoing transitions: it gets stuck once its "
                       "action finishes")
        elif all(e.cond for e in edges):
            out.append(f"state {sid} has only guarded transitions and no default edge: it gets "
                       "stuck when all guards are false, and being stuck is indistinguishable "
                       "from a wrong result")
    return out


# --------------------------------------------------------------------------- #
# Mutual exclusion and completeness (Theorem 2): finite configuration enumeration
# --------------------------------------------------------------------------- #
_ATOM_CAP = 24            # per-state atom cap; above it, no enumeration (avoids product blowup)


def _judge_var_labels(machine: Machine) -> dict[str, list]:
    """Variable written by a judge action → its label set (the enumeration domain)."""
    out: dict[str, list] = {}
    for st in machine.states.values():
        if st.action.kind == "judge":
            for w in st.action.writes:
                out[w] = list(st.action.labels)
    return out


def _determinism(machine: Machine, reachable: set[str]) -> list[str]:
    out: list[str] = []
    judge_vars = _judge_var_labels(machine)
    for sid in sorted(reachable):
        st = machine.states[sid]
        if st.action.kind == "end":
            continue
        guarded = [e for e in st.transitions if e.cond]
        if len(guarded) < 2:
            continue                          # 0/1 guarded edge + default: exclusive by design
        atoms = [a for e in guarded for a in cond.atoms_of(e.cond)]
        if len(atoms) > _ATOM_CAP:
            out.append(f"state {sid} has too many guard atoms ({len(atoms)}>{_ATOM_CAP}), "
                       "cannot enumerate configurations to check mutual exclusion; move the "
                       "decision into a judge action or route to FALLBACK")
            continue
        allvars: set[str] = set()
        for e in guarded:
            allvars |= cond.vars_of(e.cond)
        domains = {}
        undecidable = []
        for v in sorted(allvars):
            dom = _domain(v, machine, judge_vars, atoms)
            if dom is None:
                undecidable.append(v)
            else:
                domains[v] = dom
        if undecidable:
            out.append(f"state {sid} uses variables without a finite domain in its guards "
                       f"{undecidable} (variable-to-variable comparison?), so mutual exclusion "
                       "and completeness cannot be decided; use a judge action or narrow the "
                       "guards")
            continue
        names = list(domains)
        has_fallback = any(not e.cond for e in st.transitions)
        for combo in itertools.product(*(domains[n] for n in names)):
            env = dict(zip(names, combo))
            fired = []
            for e in guarded:
                try:
                    if cond.evaluate(e.cond, env):
                        fired.append(e.to)
                except cond.CondError:
                    pass
            if len(fired) >= 2:
                out.append(f"state {sid} has overlapping guards: under configuration {env}, "
                           f"{fired} hold at the same time (violates mutual exclusion)")
                break
            if len(fired) == 0 and not has_fallback:
                out.append(f"state {sid} has a gap in its guards: under configuration {env} no "
                           "edge applies and there is no default edge (violates completeness)")
                break
    return out


def _domain(var: str, machine: Machine, judge_vars: dict, atoms: list) -> Optional[list]:
    """Give a variable a finite enumeration domain. Returns None if that is not possible."""
    va = [a for a in atoms if a.var == var]
    if any(a.op in ("empty", "nonempty") for a in va):
        return [[], [1]]                      # representative values for empty / nonempty
    if var in judge_vars:
        return list(judge_vars[var])
    v = machine.var(var)
    vtype = v.type if v else None
    if vtype == "boolean":
        return [True, False]
    nums = sorted({a.const for a in va
                   if a.op in ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")
                   and isinstance(a.const, (int, float)) and not isinstance(a.const, bool)})
    if vtype in ("integer", "number") or nums:
        if not nums:
            return [0, 1]
        reps = [nums[0] - 1]
        for t in nums:
            reps.append(t)
        reps.append(nums[-1] + 1)
        return sorted(set(reps))
    eqs = {a.const for a in va if a.op in ("Eq", "NotEq")
           and isinstance(a.const, str)}
    if eqs:
        return list(eqs) + ["__other__"]
    return None


# --------------------------------------------------------------------------- #
def _loop_bounds(machine: Machine, reachable: set[str]) -> list[str]:
    """A back edge (one that closes a cycle) must carry a counter inc and have a bound exit."""
    out: list[str] = []
    back_edges = _back_edges(machine, reachable)
    for src, e in back_edges:
        if not e.inc:
            out.append(f"back edge {src}→{e.to} has no counter (inc), may never halt")
            continue
        cnt = e.inc
        target = machine.states.get(e.to)
        exits = [g for g in (target.transitions if target else [])
                 if g.cond and cnt in cond.vars_of(g.cond)]
        if not exits:
            out.append(f"back edge {src}→{e.to}: counter {cnt} has no bound exit"
                       f" (no guard on {e.to} reads it to break out)")
    return out


def _write_before_read(machine: Machine, reachable: set[str]) -> list[str]:
    """Variables a state reads (action.reads + outgoing guard variables) are written on every path.

    Paths are combined by **intersection** (not union): union asks "is there some path on which
    it exists", intersection asks "does it exist on every path". Only the latter guarantees we
    never hit an undefined variable on some occasional path.
    """
    seed = {v.name for v in machine.variables
            if v.init is not None or v.init_from is not None}
    universe = set(seed)
    for s in machine.states.values():
        universe |= _writes_of(s)

    avail: dict[str, set] = {sid: set(universe) for sid in machine.states}
    avail[machine.initial] = set(seed)
    for _ in range(len(machine.states) + 2):
        changed = False
        for sid in machine.states:
            if sid == machine.initial:
                continue
            incoming = [(src, t) for src, t in machine.transitions_all() if t.to == sid]
            if not incoming:
                new = set() if sid in reachable else set(universe)
            else:
                new = set(universe)
                for src, _t in incoming:
                    new &= avail[src] | _writes_of(machine.states[src])
            if new != avail[sid]:
                avail[sid] = new
                changed = True
        if not changed:
            break

    out: list[str] = []
    for sid in sorted(reachable):
        st = machine.states[sid]
        in_avail = avail[sid]                        # available on entry (before the action runs)
        after = in_avail | _writes_of(st)            # after the action runs (guards evaluated here)
        missing = [k for k in sorted(_reads_of(st)) if k not in in_avail]
        cond_need: set[str] = set()
        for e in st.transitions:
            cond_need |= cond.vars_of(e.cond)
        missing += [k for k in sorted(cond_need) if k not in after and k not in missing]
        if missing:
            out.append(f"state {sid} reads {sorted(set(missing))}, but these variables have not "
                       "been written when it is reached (no upstream path writes all of them)")
    return out


def _output_completeness(machine: Machine, reachable: set[str]) -> list[str]:
    out: list[str] = []
    term_out = {t.id: t.output for t in machine.terminals}
    seed = {v.name for v in machine.variables
            if v.init is not None or v.init_from is not None}
    universe = set(seed)
    for s in machine.states.values():
        universe |= _writes_of(s)
    # simplification: when an end state is reached, its output keys should be in universe
    # (the stricter per-path intersection is what write-before-read does)
    for sid in sorted(reachable):
        st = machine.states[sid]
        if st.action.kind != "end":
            continue
        keys = term_out.get(st.action.terminal, [])
        missing = [k for k in keys if k not in universe]
        if missing:
            out.append(f"end state {sid} declares outputs {missing}, but no state produces them")
    return out


# --------------------------------------------------------------------------- #
#: Placeholders in an input template: the ``cmd`` in ``{"command": "${cmd}"}``.
_TEMPLATE_RE = re.compile(r"\$\{(\w+)\}")


def template_vars(action) -> list[str]:
    """Variables referenced by a tool input template. ``{"path": "${p}"}`` ⇒ ``["p"]``; nested
    lists/dicts are scanned too."""
    out: list[str] = []

    def walk(v) -> None:
        if isinstance(v, str):
            out.extend(_TEMPLATE_RE.findall(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(getattr(action, "input", None) or {})
    return list(dict.fromkeys(out))


def _reads_of(state) -> list[str]:
    """Variables this state reads = declared ``reads`` + **input template variables**.

    Template variables used not to count as reads, so a hole like "nobody ever writes
    ``${apply_cmd}``" could quietly pass every structural check: when the machine actually ran,
    that slot rendered empty and the tool received an empty input. This was observed in practice:
    after the model step that produced the variable was cut from the document skeleton, the
    remaining tool state still referenced it, and replay was still all green. A template is a
    read and must be checked by write-before-read.
    """
    reads = list(getattr(state.action, "reads", []) or [])
    return list(dict.fromkeys(reads + template_vars(state.action)))


def _writes_of(state) -> set[str]:
    return set(getattr(state.action, "writes", []) or [])


def _back_edges(machine: Machine, reachable: set[str]):
    """Back edge = an edge to an ancestor (or itself) on the current DFS stack. Standard
    three-color DFS starting from initial."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s: WHITE for s in machine.states}
    out: list = []

    def dfs(u: str):
        color[u] = GRAY
        for t in machine.out_edges(u):
            c = color.get(t.to, WHITE)
            if c == GRAY:
                out.append((u, t))              # points to an ancestor on the stack → back edge
            elif c == WHITE:
                dfs(t.to)
        color[u] = BLACK

    if machine.initial in color:
        dfs(machine.initial)
    for s in reachable:
        if color.get(s) == WHITE:
            dfs(s)
    return out
