"""结构检查：确定性、免费，编译每动一次就跑一遍。空列表 = 全过。

图那一层的问题——「产物里有一部分够不着 / 停不下来 / 走投无路」——在状态机上是可达性、
可终止、转移完备三条纯图算法。数据那一层是两条声明比对：**互斥完备**（定理2：同一状态各
带条件出边两两不同时成立、且合起来盖满所有格局）与**先写后读**（一个状态要读的变量，在
通往它的每条路径上都已被写过）。

互斥完备靠把「无穷的变量取值」压成「有限的格局」再逐个枚举——压缩依据是条件里出现的原子
谓词（见 :mod:`hexis.machine.cond`）。一个数值变量只在它被比较的那几个阈值处才可能翻转某条边
的真假，阈值之间取一个代表值即可；一个判断动作写入的变量只在它那有限个标签里取值。
"""

from __future__ import annotations

import re

import itertools
from typing import Any, Optional

from hexis.machine import cond
from hexis.machine.schema import Machine


def structural_findings(machine: Machine) -> list[str]:
    """图与数据流上能确定性判掉的问题。空列表表示全过。"""
    out: list[str] = []
    if machine.initial not in machine.states:
        return [f"initial 指向不存在的状态 {machine.initial!r}"]
    for t in machine.transitions_all():
        for side, sid in (("from", t[0]), ("to", t[1].to)):
            if sid not in machine.states:
                out.append(f"转移 {t[0]}→{t[1].to} 的 {side} 指向不存在的状态 {sid!r}")
    if out:
        return out

    reachable = _reachable(machine)
    out += _reachability(machine, reachable)
    out += _termination(machine, reachable)
    out += _completeness(machine, reachable)
    out += _determinism(machine, reachable)          # 互斥完备（定理2）
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
    """FALLBACK 是保留状态：暂时没有边指向它不算死代码——机器长着长着就会接过去。"""
    return [f"状态 {sid} 从 {machine.initial} 走不到：死代码，删掉或接上一条边"
            for sid in sorted(set(machine.states) - reachable)
            if sid != machine.fallback]


def _termination(machine: Machine, reachable: set[str]) -> list[str]:
    terminals = {sid for sid, s in machine.states.items() if s.action.kind == "end"}
    if not terminals:
        return ["没有任何终止（end）状态：这台机器不可能正常停机"]
    can_stop, changed = set(terminals), True
    while changed:
        changed = False
        for src, t in machine.transitions_all():
            if t.to in can_stop and src not in can_stop:
                can_stop.add(src)
                changed = True
    return [f"状态 {sid} 走不到任何终止：进去就出不来，必然撞 max_steps"
            for sid in sorted(reachable - can_stop)]


def _completeness(machine: Machine, reachable: set[str]) -> list[str]:
    out: list[str] = []
    for sid in sorted(reachable):
        st = machine.states[sid]
        if st.action.kind == "end":
            continue
        edges = st.transitions
        if not edges:
            out.append(f"状态 {sid} 没有任何出边：跑完就卡住了")
        elif all(e.cond for e in edges):
            out.append(f"状态 {sid} 的出边都带条件，没有兜底边：全部为假时会 stuck，"
                       "而 stuck 与算错了在结果上分不出来")
    return out


# --------------------------------------------------------------------------- #
# 互斥完备（定理2）：有限格局枚举
# --------------------------------------------------------------------------- #
_ATOM_CAP = 24            # 单状态原子数上限，超了不枚举（防笛卡尔积爆炸）


def _judge_var_labels(machine: Machine) -> dict[str, list]:
    """判断动作写入的变量 → 它的标签集（枚举域）。"""
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
            continue                          # 0/1 条条件边 + 兜底：天然互斥
        atoms = [a for e in guarded for a in cond.atoms_of(e.cond)]
        if len(atoms) > _ATOM_CAP:
            out.append(f"状态 {sid} 的条件原子过多（{len(atoms)}>{_ATOM_CAP}），"
                       "无法枚举格局判互斥——把判断收进一个 judge 动作或接 FALLBACK")
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
            out.append(f"状态 {sid} 的条件用到无法定域的变量 {undecidable}"
                       "（变量对变量比较？），互斥完备判不了——改判断动作或收窄条件")
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
                out.append(f"状态 {sid} 条件重叠：格局 {env} 下 {fired} 同时成立（违互斥）")
                break
            if len(fired) == 0 and not has_fallback:
                out.append(f"状态 {sid} 条件有空隙：格局 {env} 下无边可走且无兜底（违完备）")
                break
    return out


def _domain(var: str, machine: Machine, judge_vars: dict, atoms: list) -> Optional[list]:
    """给一个变量定有限枚举域。定不了返回 None。"""
    va = [a for a in atoms if a.var == var]
    if any(a.op in ("empty", "nonempty") for a in va):
        return [[], [1]]                      # 空 / 非空 的代表值
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
    """回边（成环的边）必须带计数 inc、且有上限出口。"""
    out: list[str] = []
    back_edges = _back_edges(machine, reachable)
    for src, e in back_edges:
        if not e.inc:
            out.append(f"回边 {src}→{e.to} 没有计数变量（inc），可能不停机")
            continue
        cnt = e.inc
        target = machine.states.get(e.to)
        exits = [g for g in (target.transitions if target else [])
                 if g.cond and cnt in cond.vars_of(g.cond)]
        if not exits:
            out.append(f"回边 {src}→{e.to} 的计数变量 {cnt} 没有上限出口"
                       f"（{e.to} 没有一条条件读它来跳出）")
    return out


def _write_before_read(machine: Machine, reachable: set[str]) -> list[str]:
    """状态要读的变量（action.reads + 出边条件的变量），在到它的每条路径上都已被写。

    按路径**取交集**（不是并集）：并集问「有没有一条路让它存在」，交集问「是不是每条路都
    让它存在」。只有后者能保证不在某条偶发路径上撞未定义变量。
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
        in_avail = avail[sid]                        # 进入状态时可用（动作执行前）
        after = in_avail | _writes_of(st)            # 动作执行后（出边条件在此时求值）
        missing = [k for k in sorted(_reads_of(st)) if k not in in_avail]
        cond_need: set[str] = set()
        for e in st.transitions:
            cond_need |= cond.vars_of(e.cond)
        missing += [k for k in sorted(cond_need) if k not in after and k not in missing]
        if missing:
            out.append(f"状态 {sid} 要读 {sorted(set(missing))}，但走到它时这些变量还没被写过"
                       "（上游没有一条路径把它们都写上）")
    return out


def _output_completeness(machine: Machine, reachable: set[str]) -> list[str]:
    out: list[str] = []
    term_out = {t.id: t.output for t in machine.terminals}
    seed = {v.name for v in machine.variables
            if v.init is not None or v.init_from is not None}
    universe = set(seed)
    for s in machine.states.values():
        universe |= _writes_of(s)
    # 简化：终止态到达时，其 output 键应在 universe 里（更严的按路径交集见先写后读）
    for sid in sorted(reachable):
        st = machine.states[sid]
        if st.action.kind != "end":
            continue
        keys = term_out.get(st.action.terminal, [])
        missing = [k for k in keys if k not in universe]
        if missing:
            out.append(f"终止态 {sid} 声明输出 {missing}，但没有状态产出它们")
    return out


# --------------------------------------------------------------------------- #
#: 入参模板里的占位符：``{"command": "${cmd}"}`` 里的 ``cmd``。
_TEMPLATE_RE = re.compile(r"\$\{(\w+)\}")


def template_vars(action) -> list[str]:
    """工具入参模板引用的变量。``{"path": "${p}"}`` ⇒ ``["p"]``；嵌套的 list/dict 也扫。"""
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
    """这个状态要读的变量 = 声明的 ``reads`` + **入参模板引用的变量**。

    模板变量原来不算读，于是「``${apply_cmd}`` 谁都没写过」这种洞能安静地通过全部结构检查
    ——机器真跑时那一格渲染成空，工具拿到空入参。实测过：文档骨架里产这个变量的模型步被
    裁掉之后，剩下的工具状态照样引用它，复述还全绿。模板是读，必须按先写后读查。
    """
    reads = list(getattr(state.action, "reads", []) or [])
    return list(dict.fromkeys(reads + template_vars(state.action)))


def _writes_of(state) -> set[str]:
    return set(getattr(state.action, "writes", []) or [])


def _back_edges(machine: Machine, reachable: set[str]):
    """回边 = 指向当前 DFS 递归栈上祖先（含自身）的边。标准三色 DFS，从 initial 出发。"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s: WHITE for s in machine.states}
    out: list = []

    def dfs(u: str):
        color[u] = GRAY
        for t in machine.out_edges(u):
            c = color.get(t.to, WHITE)
            if c == GRAY:
                out.append((u, t))              # 指向栈上祖先 → 回边
            elif c == WHITE:
                dfs(t.to)
        color[u] = BLACK

    if machine.initial in color:
        dfs(machine.initial)
    for s in reachable:
        if color.get(s) == WHITE:
            dfs(s)
    return out
