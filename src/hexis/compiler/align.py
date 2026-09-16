"""轨迹与状态机对齐（算法文档第 4 节）：固定代价表上的动态规划。

对齐对象是**可观察事件**与**可观察状态**（Q_obs：工具、结束、用户、可观察的模型状态）。
叙述与判断记录不占格，随附到下一个事件。相邻事件之间用 p ⇝_b q（只经零宽状态相连，首条
转移与前一事件的成败相容）判「保留已有路径」。每个事件还带一个新增状态候选。

代价表::

    同类型、同工具、同标签匹配          0
    保留已有路径                        0
    更换同标签的工具                    1
    标签集不同（多出或缺少派生标签）      1
    增加循环                            1
    新增转移或修改入口                  3
    新增状态                            4
    新增转移绕过规则提到的文档状态       ∞
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hexis.machine.schema import Machine
from hexis.compiler.common import (INF, anchors, entry_anchors, required_states, skip_blocked, status_known,
                     zero_paths)
from hexis.compiler.context import CompileContext
from hexis.compiler.traces import Event, Prepared, end_state_for

C_MATCH, C_KEEP = 0, 0
C_REALIZE, C_LABEL, C_LOOP = 1, 1, 1
C_EDGE, C_START = 3, 3
C_NEW = 4

NEW = "new:"


@dataclass
class Slot:
    """对齐路径上的一格：第 i 个可观察事件落到哪个状态、怎么落的。"""
    index: int                      # 事件在 prep.events 里的下标
    state: str                      # 状态 id，或 new:<i>
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
    """c_m(s_i, q)。"""
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
    """c_l：事件多次调用、末次成功、原图缺少循环时计一分。"""
    if ev.kind == "tool" and ev.n_calls > 1 and ev.ok and not has_loop:
        return C_LOOP
    return 0


def align(m: Machine, prep: Prepared, ctx: CompileContext, *,
          allow_realize: bool = True) -> tuple[Optional[Alignment], str]:
    """整条轨迹的最小代价对齐路径。返回 (对齐, 拒绝原因)。"""
    evs = [e for e in prep.events if e.observable]
    if not evs:
        return None, "轨迹没有可观察事件"
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
            return C_KEEP, "add"              # 新增状态的连接不再重复计费
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
        return None, "没有可选的对齐路径"
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
