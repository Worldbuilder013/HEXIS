"""候选机器检查（算法文档第 6 节，规则驱动）。

6.1 变量与证据：按入边做不动点。A_q 取交集（进入 q 前可用的变量）；Z_q 取并集，元素是
    「可能持有的证据集合」——证据来自技能的终点条件：某个状态静态匹配 required_evidence 且
    成功分支 → 证据成立，失败分支 → 证据失效，分不出 → 两种可能；匹配 invalidating_events 的
    状态使该终点的证据全部失效。带条件的终点要求 Z_q 里每个集合都含齐它的证据；回退终点不得
    带条件。技能的 must_occur / before / forbid 要求也在图上静态检查。与结构检查一起组成 Check(M)。
6.2 轨迹路径检查 Replay(M,T,π̃)：沿路径推进变量取值，任务输入与模型输出用占位值，工具输出用
    事件末次调用的真实结果；判断状态选通向下一状态的标签；每步选中的转移须符合路径；工具所需
    变量须已赋值且非空；末端须为结束状态；比较可观察事件序列。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .. import checks as _checks
from .. import cond as _cond
from ..runtime import bind_outputs, rebuild
from ..schema import Machine, State
from .common import (branch_class, guaranteed, is_observable, need, reachable, seed_vars,
                     static_match, terminal_kinds)
from .context import CompileContext, EventPattern
from .traces import Prepared

ZSet = frozenset  # frozenset[frozenset[tuple[str, int]]]


# --------------------------------------------------------------------------- #
# 6.1 静态分析
# --------------------------------------------------------------------------- #
def _relaxed(pat: EventPattern) -> EventPattern:
    return EventPattern(**{**pat.__dict__, "success": None, "after": None})


def post(z: ZSet, st: State, guard: str, ctx: CompileContext, m: Machine) -> ZSet:
    """Post(Z, a, g)：证据集合经状态 st 与护卫 g 后的可能取值。"""
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
    """技能要求的静态检查：must_occur（每条到非回退终点的路径都经过）、before、forbid。"""
    out: list[str] = []
    kinds = terminal_kinds(m)
    ends = {sid for sid, st in m.states.items()
            if st.action.kind == "end" and sid != m.fallback and kinds.get(st.action.terminal) != "fallback"}
    for r in ctx.requirements:
        a_states = {sid for sid, st in m.states.items() if static_match(_relaxed(r.a), st, m)}
        if r.kind == "forbid":
            if a_states:
                out.append(f"要求 {r.id}: 状态 {sorted(a_states)} 匹配了禁止的事件")
            continue
        if r.kind == "must_occur":
            if not a_states:
                out.append(f"要求 {r.id}: 没有状态能产生必须出现的事件 ({r.a.describe()})")
                continue
            leaked = ends & _avoiding_reach(m, a_states)
            if leaked:
                out.append(f"要求 {r.id}: 不经 {sorted(a_states)} 也能到达终点 {sorted(leaked)}")
        elif r.kind == "before" and r.b is not None:
            b_states = {sid for sid, st in m.states.items() if static_match(_relaxed(r.b), st, m)}
            bad = sorted(b_states & _avoiding_reach(m, a_states))
            if bad:
                out.append(f"要求 {r.id}: {bad} 可在 ({r.a.describe()}) 之前到达")
    return out


def check(m: Machine, ctx: CompileContext) -> list[str]:
    """Check(M) = G_structure ∧ G_variables ∧ G_evidence ∧ G_requirements。空列表 = 通过。"""
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
            out.append(f"变量: {sid} 需要 {miss}，进入时不保证已定义")
        for t in st.transitions:
            if t.cond:
                try:
                    gv = _cond.vars_of(t.cond)
                except _cond.CondError as exc:
                    out.append(f"条件: {sid}→{t.to} {t.cond!r} 解析失败：{exc}")
                    continue
                bad = sorted(set(gv) - after)
                if bad:
                    out.append(f"变量: {sid}→{t.to} 的条件读 {bad}，动作后仍未定义")
            if t.to in an.reach:
                nxt = an.avail[t.to]
                if not nxt <= after:
                    out.append(f"变量: {sid}→{t.to} 给不出 {sorted(nxt - after)}")
                zp = post(z_q, st, t.cond, ctx, m)
                if not zp <= an.zstate[t.to]:
                    out.append(f"证据: {sid}→{t.to} 的证据集合不在 {t.to} 的不变式内")
        if st.action.kind == "end":
            tc = conditions.get(st.action.terminal)
            if tc is not None:
                ids = {(tc.terminal, i) for i in range(len(tc.required_evidence))}
                lacking = sorted({i for held in z_q for (_t, i) in (ids - held)})
                if lacking or not z_q:
                    names = [tc.required_evidence[i].describe() for i in lacking] or ["全部"]
                    out.append(f"证据: 终点 {sid}({tc.terminal}) 可能缺少证据 {names}")
    fb = m.states.get(m.fallback)
    if fb is not None and fb.action.kind == "end" and fb.action.terminal in conditions:
        out.append(f"证据: 回退状态 {m.fallback} 不得使用带证据条件的终点")
    out += requirement_findings(m, ctx)
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- #
# 6.2 轨迹路径检查
# --------------------------------------------------------------------------- #
@dataclass
class ReplayResult:
    ok: bool
    why: str = ""
    path: list = field(default_factory=list)
    events: list = field(default_factory=list)
    visits: dict = field(default_factory=dict)


def _pick(m: Machine, st: State, values: dict):
    """按运行时顺序选边：首条成立的转移。返回 (边, 错误)。"""
    for t in st.ordered_transitions():
        if not t.cond:
            return t, ""
        try:
            if _cond.evaluate(t.cond, values):
                return t, ""
        except _cond.CondError as exc:
            return None, f"{st.id} 的条件 {t.cond!r} 求值失败：{exc}"
    return None, ""


def _candidates(st: State, var: str, values: dict) -> list:
    """一个由本状态写出的变量在它出边条件里被拿来比较的常量：回放时可选的取值。占位值排第一。"""
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
    """状态写出的变量的候选取值组合。判断状态按标签；模型状态按出边条件里出现的常量，其余占位。"""
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
    """从 sid 出发只经零宽状态能否走到 target（按运行时选边规则推演）。

    返回各零宽状态该写出的取值 {状态: {变量: 值}}：判断状态是选中的标签，模型状态是出边条件里
    能通向下一状态的常量（其余变量用占位值）。"""
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
    """可观察的模型 / 用户状态：写出能让下一跳通向 target 的取值；没有目标或都不行就用占位值。"""
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
    """Replay(M, T, π̃)。``anchors`` 是每个可观察事件对应的状态 id（末项是结束状态）。

    ``ignore_counters`` 时计数变量不递增（上限出口永不触发），只用来数候选路径上的访问次数。"""
    evs = prep.observable
    if len(anchors) != len(evs):
        return ReplayResult(False, f"路径长度 {len(anchors)} 与可观察事件数 {len(evs)} 不符")
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
            return ReplayResult(False, f"路径撞到不存在的状态 {cur}", path, events)
        path.append(cur)
        visits[cur] = visits.get(cur, 0) + 1
        a = st.action
        if is_observable(st):
            if i >= len(anchors) or anchors[i] != cur:
                want = anchors[i] if i < len(anchors) else "(无)"
                return ReplayResult(False, f"走到 {cur}，路径要求 {want}", path, events)
            ev = evs[i]
            if a.kind == "tool":
                miss = [x for x in sorted(need(st)) if values.get(x) in (None, "")]
                if miss:
                    return ReplayResult(False, f"{cur} 需要 {miss}，到这一步未赋值或为空", path, events)
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
                    return ReplayResult(False, f"在 {cur} 结束，但路径还有 {len(anchors) - 1 - i} 个事件", path, events)
                if events != expect:
                    return ReplayResult(False, f"事件序列不同：机器 {events} vs 轨迹 {expect}", path, events)
                return ReplayResult(True, "", path, events, visits)
            i += 1
        elif a.kind in ("model", "judge"):
            if i >= len(anchors):
                return ReplayResult(False, f"零宽状态 {cur} 出现在路径末端之后", path, events)
            sim = _simulate_zero(m, cur, values, anchors[i], ignore_counters=ignore_counters)
            if sim is None:
                what = "标签" if a.kind == "judge" else "取值"
                return ReplayResult(False, f"{'判断' if a.kind == 'judge' else '模型'}状态 {cur} 没有{what}能通向 {anchors[i]}",
                                    path, events)
            values.update(sim[cur])
        else:
            return ReplayResult(False, f"{cur} 的动作类型 {a.kind} 不能回放", path, events)
        chosen, err = _pick(m, st, values)
        if err:
            return ReplayResult(False, err, path, events)
        if chosen is None:
            return ReplayResult(False, f"{cur} 没有可走的转移", path, events)
        if chosen.inc and not ignore_counters:
            values[chosen.inc] = int(values.get(chosen.inc) or 0) + 1
        cur = chosen.to
    return ReplayResult(False, f"超过 {limit} 步仍未到结束状态", path, events)


__all__ = ["Analysis", "ReplayResult", "analyze", "check", "placeholders", "post", "replay",
           "requirement_findings"]
