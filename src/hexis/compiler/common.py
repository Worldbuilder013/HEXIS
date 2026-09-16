"""公共定义：变量需求、保证写出、三值护卫判定、零宽路径、模式的静态匹配。

对应算法文档第 1 节的记号：状态动作的读集 R_q、写集 W_q、参数模板 P_q、输出映射 β_q，
Q_obs（可观察状态：工具、结束、用户、可观察的模型状态）和 p ⇝_b q（只经零宽状态相连）。

这里没有任何工具名、字段名或技能名。工具的成功判据、产出字段来自
:class:`~hexis.tools.toolspec.ToolSpec`；模式来自 :mod:`.context`。
"""
from __future__ import annotations

import ast
import json
import re
from collections import deque
from typing import Any, Mapping, Optional

from hexis.machine import cond as _cond
from hexis.machine.schema import Machine, State, Transition
from hexis.tools.toolspec import ToolSpec
from hexis.compiler.context import CompileContext, EventPattern

ABSTAIN = "弃权"
INF = 10 ** 9

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_COUNTER_EXIT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*(\d+)\s*$")


# --------------------------------------------------------------------------- #
# 变量
# --------------------------------------------------------------------------- #
def template_vars(inp: Any) -> list[str]:
    """参数模板引用的变量 Vars(P_q)。"""
    out: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            out.extend(_VAR.findall(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(inp)
    return list(dict.fromkeys(out))


def need(st: State) -> frozenset:
    """Need(q) = R_q ∪ Vars(P_q)。同时读写的变量仍保留读取需求。"""
    a = st.action
    out = set(getattr(a, "reads", []) or [])
    if a.kind == "tool":
        out |= set(template_vars(a.input))
    return frozenset(out)


def spec_of(ctx: CompileContext, st: State) -> Optional[ToolSpec]:
    if st.action.kind != "tool":
        return None
    return ctx.spec(st.action.name)


def guaranteed(st: State, ctx: CompileContext) -> frozenset:
    """G_q：动作保证写出的变量。工具按定义的保证字段加输出映射；模型、判断、用户按写入声明。"""
    a = st.action
    if a.kind == "tool":
        sure = ctx.spec(a.name).sure_outputs()
        out = {w for w in a.writes if w in sure}
        for src, dst in (getattr(a, "binds", {}) or {}).items():
            if src in sure and dst in a.writes:
                out.add(dst)
        return frozenset(out)
    if a.kind in ("model", "judge", "user"):
        return frozenset(a.writes)
    return frozenset()


def task_inputs(m: Machine) -> list[str]:
    return [v.name for v in m.variables if v.init_from]


def seed_vars(m: Machine) -> frozenset:
    """A_0：任务输入与有效初值。空字符串不计为有效初值。"""
    return frozenset(v.name for v in m.variables
                     if v.init_from is not None or (v.init is not None and v.init != ""))


def is_observable(st: State) -> bool:
    """可观察状态：工具、结束、用户，以及声明为可观察的模型状态（写出交付内容的那些）。"""
    a = st.action
    if a.kind in ("tool", "end", "user"):
        return True
    if a.kind == "model":
        return bool(getattr(a, "observable", False))
    return False


def is_zero_width(st: State) -> bool:
    return not is_observable(st)


def anchors(m: Machine) -> list[str]:
    """Q_obs。"""
    return [sid for sid, st in m.states.items() if is_observable(st)]


def terminal_kinds(m: Machine) -> dict[str, str]:
    return {t.id: t.kind for t in m.terminals}


def end_terminal(m: Machine, sid: str) -> str:
    st = m.states.get(sid)
    if st is None or st.action.kind != "end":
        return ""
    return st.action.terminal


# --------------------------------------------------------------------------- #
# 三值护卫判定：只有已知取值的变量能定真假，其余一律「可能成立」
# --------------------------------------------------------------------------- #
_UNKNOWN = object()

_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}


def _tv(node: ast.AST, known: Mapping) -> Any:
    if isinstance(node, ast.BoolOp):
        vals = [_tv(v, known) for v in node.values]
        if isinstance(node.op, ast.And):
            if any(v is False for v in vals):
                return False
            return True if all(v is True for v in vals) else _UNKNOWN
        if any(v is True for v in vals):
            return True
        return False if all(v is False for v in vals) else _UNKNOWN
    if isinstance(node, ast.UnaryOp):
        v = _tv(node.operand, known)
        return _UNKNOWN if v is _UNKNOWN else (not v)
    if isinstance(node, ast.Compare):
        if (len(node.ops) == 1 and isinstance(node.left, ast.Name) and node.left.id in known
                and isinstance(node.comparators[0], ast.Constant) and type(node.ops[0]) in _CMP):
            try:
                return bool(_CMP[type(node.ops[0])](known[node.left.id], node.comparators[0].value))
            except TypeError:
                return _UNKNOWN
        return _UNKNOWN
    return _UNKNOWN


def truth(expr: str, known: Optional[Mapping] = None) -> Optional[bool]:
    """护卫在已知取值 ``known`` 下的三值真值：True / False / None。空条件恒成立。"""
    if not expr:
        return True
    try:
        tree = _cond.parse(expr)
    except _cond.CondError:
        return None
    v = _tv(tree.body, known or {})
    return None if v is _UNKNOWN else bool(v)


def maybe_true(expr: str, known: Optional[Mapping] = None) -> bool:
    return truth(expr, known) is not False


def takeable(st: State, known: Optional[Mapping] = None) -> list[Transition]:
    """按运行时顺序，在已知取值下可能被选中的转移：确定成立的边之后的边不再可达。"""
    out: list[Transition] = []
    for t in st.ordered_transitions():
        v = truth(t.cond, known)
        if v is False:
            continue
        out.append(t)
        if v is True:
            break
    return out


def status_known(ctx: CompileContext, st: State, ok: Optional[bool]) -> dict:
    """工具状态 st 的调用成败为 ok 时，护卫里已知的变量取值。"""
    if st.action.kind != "tool" or ok is None:
        return {}
    return ctx.spec(st.action.name).status_values(ok)


def branch_class(expr: str, ctx: CompileContext, st: State) -> str:
    """分支类别：success / failure / unknown（由工具成功判据的简单条件识别）。"""
    if st.action.kind != "tool":
        return "unknown"
    ok, fail = maybe_true(expr, status_known(ctx, st, True)), maybe_true(expr, status_known(ctx, st, False))
    if ok and not fail:
        return "success"
    if fail and not ok:
        return "failure"
    return "unknown"


# --------------------------------------------------------------------------- #
# 图算法
# --------------------------------------------------------------------------- #
def zero_paths(m: Machine, src: str, known: Optional[Mapping] = None, *,
               include_src_zero: bool = False) -> dict[str, list[tuple[str, Transition]]]:
    """从 src 出发、只经零宽状态可达的可观察状态 → 最短边序列 [(状态, 边)…]。

    首条转移须与已知取值相容；零宽状态的出边全部允许。
    ``include_src_zero``：src 自身是零宽状态（入口的情形），它的出边不受约束。
    """
    out: dict[str, list] = {}
    if src not in m.states:
        return out
    first_known = {} if include_src_zero else (known or {})
    q: deque = deque()
    seen = {src}
    for t in takeable(m.states[src], first_known):
        q.append((t.to, [(src, t)]))
    while q:
        sid, path = q.popleft()
        if sid not in m.states:
            continue
        st = m.states[sid]
        if is_observable(st):
            out.setdefault(sid, path)
            continue
        if sid in seen:
            continue
        seen.add(sid)
        for t in takeable(st, {}):
            q.append((t.to, path + [(sid, t)]))
    return out


def entry_anchors(m: Machine) -> dict[str, list[tuple[str, Transition]]]:
    """入口只经零宽状态可达的可观察状态（入口本身可观察时即它自己）。"""
    st = m.states.get(m.initial)
    if st is None:
        return {}
    if is_observable(st):
        return {m.initial: []}
    return zero_paths(m, m.initial, {}, include_src_zero=True)


def reaches(m: Machine, a: str, b: str, *, without: Optional[str] = None) -> bool:
    """全图上 a 能否到 b（可绕开 without）。a == b 时要求存在真正的环。"""
    if a not in m.states or b not in m.states or without == a:
        return False
    stack, seen = [a], set()
    while stack:
        u = stack.pop()
        for t in m.states[u].transitions:
            if t.to == b:
                return True
            if t.to == without or t.to in seen or t.to not in m.states:
                continue
            seen.add(t.to)
            stack.append(t.to)
    return False


def reachable(m: Machine) -> set[str]:
    seen, stack = {m.initial}, [m.initial]
    while stack:
        u = stack.pop()
        st = m.states.get(u)
        if st is None:
            continue
        for t in st.transitions:
            if t.to not in seen and t.to in m.states:
                seen.add(t.to)
                stack.append(t.to)
    return seen


def back_edges(m: Machine) -> list[tuple[str, Transition]]:
    """回边：指向 DFS 栈上祖先（含自身）的边。"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s: WHITE for s in m.states}
    out: list = []

    def dfs(u: str) -> None:
        color[u] = GRAY
        for t in m.states[u].ordered_transitions():
            c = color.get(t.to, BLACK)
            if c == GRAY:
                out.append((u, t))
            elif c == WHITE:
                dfs(t.to)
        color[u] = BLACK

    if m.initial in color:
        dfs(m.initial)
    for s in list(m.states):
        if color.get(s) == WHITE:
            dfs(s)
    return out


def counter_exit(expr: str) -> Optional[tuple[str, int]]:
    """``cnt >= K`` 形式的上限出口 → (cnt, K)。"""
    mm = _COUNTER_EXIT.match(expr or "")
    return (mm.group(1), int(mm.group(2))) if mm else None


# --------------------------------------------------------------------------- #
# 模式的静态匹配（在机器状态上）：只看状态自己声明的属性
# --------------------------------------------------------------------------- #
def state_uses(m: Machine, st: State) -> frozenset:
    """状态用到的变量：自己的需求，加上专门为它生成参数的模型状态读的变量。"""
    out = set(need(st))
    for g in m.states.values():
        if g.action.kind == "model" and [t.to for t in g.transitions] == [st.id]:
            out |= set(g.action.reads)
    return frozenset(out)


def static_match(pat: EventPattern, st: State, m: Machine) -> bool:
    """状态是否可能产生匹配模式的事件。参数文本条件按模板引用的变量判；成败与先后不看。"""
    a = st.action
    if pat.kind is not None and a.kind != pat.kind:
        return False
    if pat.tool is not None and (a.kind != "tool" or a.name != pat.tool):
        return False
    if pat.label is not None:
        labels = set(getattr(a, "labels", ()) or ()) | {getattr(a, "phase", "") or ""}
        if pat.label not in labels:
            return False
    if pat.role is not None and a.kind == "model":
        role = "output" if getattr(a, "observable", False) else "narration"
        if role != pat.role:
            return False
    if pat.args_contain is not None:
        fields = set(_VAR.findall(pat.args_contain))
        text = json.dumps(getattr(a, "input", {}) or {}, ensure_ascii=False)
        if fields and not fields <= state_uses(m, st):
            return False
        if not fields and pat.args_contain not in text:
            return False
    if pat.arg_regex is not None:
        text = json.dumps(getattr(a, "input", {}) or {}, ensure_ascii=False)
        try:
            if not re.search(pat.arg_regex, text):
                return False
        except re.error:
            return False
    return True


def required_states(m: Machine, ctx: CompileContext) -> set[str]:
    """Q_req：文档来源、且被某条规则模式提到的状态。绕过它们就是削弱约束。"""
    pats = ctx.patterns()
    return {sid for sid, st in m.states.items()
            if st.origin == "document" and st.action.kind != "end"
            and any(static_match(p, st, m) for p in pats)}


def skip_blocked(m: Machine, p: str, q: str, req: set[str]) -> Optional[str]:
    """Skip_M(p,q)：p→q 已有路径，且某个必经的 Q_req 状态会被直达转移绕过。返回该状态。"""
    if not reaches(m, p, q):
        return None
    for r in sorted(req - {p, q}):
        if not reaches(m, p, q, without=r):
            return r
    return None


def summary_of(st: State) -> str:
    """一个状态的一句话身份，给判断状态的提示词用。"""
    a = st.action
    if a.kind == "tool":
        lab = getattr(a, "phase", "") or ""
        return f"{st.id}: {a.name}{'/' + lab if lab else ''} {json.dumps(a.input, ensure_ascii=False)[:80]}"
    if a.kind == "model":
        return f"{st.id}: model → {a.writes} ({a.prompt[:80]!r})"
    if a.kind == "judge":
        return f"{st.id}: judge {a.labels} ({a.prompt[:60]!r})"
    if a.kind == "user":
        return f"{st.id}: user → {a.writes}"
    if a.kind == "end":
        return f"{st.id}: end {a.terminal}"
    return f"{st.id}: {a.kind}"


__all__ = ["ABSTAIN", "INF", "anchors", "back_edges", "branch_class", "counter_exit", "end_terminal",
           "entry_anchors", "guaranteed", "is_observable", "is_zero_width", "maybe_true", "need",
           "reachable", "reaches", "required_states", "seed_vars", "skip_blocked", "spec_of",
           "state_uses", "static_match", "status_known", "summary_of", "takeable", "task_inputs",
           "template_vars", "terminal_kinds", "truth", "zero_paths"]
