"""Alignment of a trace with the machine: dynamic programming over a fixed cost table.

What gets aligned are **observable events** and **observable states** (Q_obs: tool, end, user and observable model
states). Narration and judge records take no slot and are attached to the next event. Between adjacent events,
p ⇝_b q (connected through zero-width states only, with the first transition consistent with the outcome of the
previous event) decides "keep the existing path". Every event also has a new-state candidate.

Cost table::

    match with the same kind, tool and label                        0
    keep the existing path                                          0
    change the tool, same label                                     1
    different label set (extra or missing derived labels)           1
    add a loop                                                      1
    add a transition or change the entry                            3
    add a state                                                     4
    add a transition that bypasses a document state the rules name  ∞
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hexis.compiler.common import INF, anchors, entry_anchors, required_states, skip_blocked, status_known, zero_paths
from hexis.compiler.context import CompileContext
from hexis.compiler.traces import Event, Prepared, end_state_for
from hexis.machine.schema import Machine

C_MATCH, C_KEEP = 0, 0
C_REALIZE, C_LABEL, C_LOOP = 1, 1, 1
C_EDGE, C_START = 3, 3
C_NEW = 4

NEW = "new:"


@dataclass
class Slot:
    """One slot of the alignment path: which state the i-th observable event lands on, and how."""
    index: int                      # index of the event in prep.events
    state: str                      # state id, or new:<i>
    how: str                        # match / realize / label / new / end
    c_match: int = 0
    c_loop: int = 0
    c_edge: int = 0
    edge: str = ""                  # start / keep / add

    @property
    def is_new(self) -> bool:
        return self.state.startswith(NEW)

    @property
    def loop(self) -> bool:
        return self.c_loop > 0


@dataclass
class Alignment:
    slots: list = field(default_factory=list)
    cost: int = 0
    end_state: str = ""
    notes: list = field(default_factory=list)

    def anchors(self) -> list[str]:
        return [s.state for s in self.slots]


def match_cost(m: Machine, ev: Event, q: str, term: str, *, allow_realize: bool) -> tuple[int, str]:
    """c_m(s_i, q): the match cost of event s_i on state q."""
    st = m.states[q]
    a = st.action
    if ev.kind == "end":
        return (C_MATCH, "end") if (a.kind == "end" and q == term) else (INF, "")
    if ev.kind == "model":
        return (C_MATCH, "match") if (a.kind == "model" and getattr(a, "observable", False)) else (INF, "")
    if ev.kind == "user":
        return (C_MATCH, "match") if a.kind == "user" else (INF, "")
    if ev.kind != "tool" or a.kind != "tool":
        return INF, ""
    if (getattr(a, "phase", "") or "") != ev.label:
        return INF, ""
    cost, how = 0, "match"
    if a.name != ev.tool:
        if not allow_realize:
            return INF, ""
        cost, how = C_REALIZE, "realize"
    if set(getattr(a, "labels", []) or []) != set(ev.labels):
        cost, how = cost + C_LABEL, ("realize" if how == "realize" else "label")
    return cost, how


def loop_cost(ev: Event, has_loop: bool) -> int:
    """c_l: one point when the event has several calls, the last one succeeds and the original graph lacks a loop."""
    if ev.kind == "tool" and ev.n_calls > 1 and ev.ok and not has_loop:
        return C_LOOP
    return 0


def align(m: Machine, prep: Prepared, ctx: CompileContext, *,
          allow_realize: bool = True) -> tuple[Optional[Alignment], str]:
    """Minimum cost alignment path for the whole trace. Returns (alignment, rejection reason)."""
    evs = [e for e in prep.events if e.observable]
    if not evs:
        return None, "trace has no observable events"
    term = end_state_for(m, prep.tau)
    anc = anchors(m)
    req = required_states(m, ctx)
    entry = entry_anchors(m)
    n = len(evs)
    comp: dict = {}

    def paths(p: str, b: Optional[bool]) -> dict:
        key = (p, b)
        if key not in comp:
            comp[key] = zero_paths(m, p, status_known(ctx, m.states[p], b))
        return comp[key]

    def edge_cost(p: str, q: str, b: Optional[bool]) -> tuple[int, str]:
        if p.startswith(NEW) or q.startswith(NEW):
            return C_KEEP, "add"              # connections of new states are not charged again
        if q in paths(p, b):
            return C_KEEP, "keep"
        if skip_blocked(m, p, q, req):
            return INF, ""
        return C_EDGE, "add"

    def has_self_loop(q: str) -> bool:
        return (not q.startswith(NEW)) and q in paths(q, True)

    def nodes_for(i: int) -> list[str]:
        if evs[i].kind == "end":
            return [term] if term in m.states else []
        return anc + [f"{NEW}{evs[i].index}"]

    def cm(i: int, q: str) -> tuple[int, str]:
        if q.startswith(NEW):
            return C_NEW, "new"
        return match_cost(m, evs[i], q, term, allow_realize=allow_realize)

    dp: list[dict] = [dict() for _ in range(n)]
    for q in nodes_for(0):
        c, how = cm(0, q)
        if c >= INF:
            continue
        cl = loop_cost(evs[0], has_self_loop(q))
        if q.startswith(NEW):
            c0, ek = C_KEEP, "start"
        elif q in entry:
            c0, ek = 0, "keep"
        else:
            c0, ek = C_START, "start"
        dp[0][q] = (c0 + c + cl, None, how, c, cl, c0, ek)
    for i in range(1, n):
        prev = evs[i - 1]
        b_prev = prev.ok if prev.kind == "tool" else None
        for q in nodes_for(i):
            c, how = cm(i, q)
            if c >= INF:
                continue
            cl = loop_cost(evs[i], has_self_loop(q))
            best = None
            for p, (pc, *_rest) in dp[i - 1].items():
                ce, ek = edge_cost(p, q, b_prev)
                if ce >= INF:
                    continue
                tot = pc + ce + c + cl
                if best is None or tot < best[0] or (tot == best[0] and p < best[1]):
                    best = (tot, p, how, c, cl, ce, ek)
            if best is not None:
                dp[i][q] = best
    if not dp[n - 1]:
        return None, "no alignment path available"
    last = min(dp[n - 1].items(), key=lambda kv: (kv[1][0], kv[0]))
    slots: list[Slot] = []
    q = last[0]
    for i in range(n - 1, -1, -1):
        tot, prv, how, c, cl, ce, ek = dp[i][q]
        slots.append(Slot(index=evs[i].index, state=q, how=how, c_match=c, c_loop=cl, c_edge=ce, edge=ek))
        q = prv
    slots.reverse()
    return Alignment(slots=slots, cost=last[1][0], end_state=term), ""


__all__ = ["Alignment", "NEW", "Slot", "align", "loop_cost", "match_cost"]
