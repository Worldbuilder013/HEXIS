"""拟合：编译器里**真正可复用**的那几段算法，从守门程序里抽出来独立成篇。

:mod:`hexis.legacy.compiler` 是「沿轨迹走、建状态、接边」的流程骨架，它跟 table_clean 那套
记录格式绑得很紧；但骨架里嵌着三段与流程无关、任何一台机器都用得上的算法，抽到这里：

* :func:`learn_cond` —— **在变量快照上学一个分岔条件**。给定「每个后继各自被观测到时的
  变量快照」，找一组两两互斥的谓词，把「该走哪一支」从模型的临场判断变成机器上的确定性
  跳转。这是整套编译最值钱的一步：学得出，这个分岔就不再需要模型。
* :func:`install_counter` / :func:`install_counters` —— **给回边配计数与上限出口**。循环
  是 EFSM 比普通自动机多出来的那点表达力，也是唯一可能不停机的地方；上限出口把「学出来的
  环」变成「一定停得下来的环」。
* :func:`calibrate` —— **标定判断动作的误差率**。编译期唯一还要碰模型的一步：把一个判断
  动作放在带正确标签的快照上重跑，量出它错多少，写进机器供审阅（εᵢ 的来源）。

三关（learn_cond 收条件前必须过的三道闸）
-----------------------------------------
1. **支持度**：每一支的观测快照数 ≥ ``min_support``。只见过一次的分岔不是分岔，是巧合。
2. **留出正确率**：条件在**拟合份**上挑，再在**留出份**上算正确率，低于 ``acc_thr`` 不收。
   没有这一关，学出来的往往是「把训练快照背下来」的谓词——例如拿计数变量的一个中点把两支
   切开，在观测上完美、换一个任务立刻错。
3. **互斥可证**：各支的条件在**符号层**两两不同时成立（把变量压成有限格局逐个枚举，与
   :func:`hexis.machine.checks.structural_findings` 判互斥同一套办法），不是只在观测到的那几个
   快照上碰巧不重叠。定理2 要的是前者。

三关任一不过就返回 ``None``——调用方（``compiler._solve_branches``）据此把这个分岔整体接到
FALLBACK：**不会算就明说不会算**，比学一个半对的条件安全。

循环上限 K 的来源
-----------------
:func:`loop_bound` 把 K 的取法摆到台面上：文档若写了圈数上限，**文档说了算**；文档没写，
才用 ``ceil(margin × 观测最大圈数)`` 由编译器补一个，并在台账里标成 ``"compiler"``。这条
区分要留痕：对 math-skill，解题主干（S1..S4）通篇没有任何圈数上限——唯一的数字是「2-3 个
可能的方法族」，那是方法数不是重试次数——所以它的 K 全部是编译器引入的，覆盖报告必须照实
说「这条上限不是文档要求，是编译器为了停机补的」。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence

from hexis.machine import cond
from hexis.machine.checks import _back_edges, _reachable
from hexis.machine.schema import Machine, Transition, Variable

__all__ = [
    "LoopBound", "calibrate", "candidate_atoms", "install_counter",
    "install_counters", "learn_cond", "loop_bound", "loop_bound_detail",
    "mutually_exclusive", "separating",
]

#: 互斥枚举的格局数上限：超了就判「证不了」（宁可不收条件，也不假装证过）。
_CONFIG_CAP = 4096

#: 无法定域的字符串变量用的代表值：表示「除列出的那几个字面量之外的任何值」。
_OTHER = "__other__"

#: 关三回退时每支最多留几个备选条件、整组最多试几种搭配（防组合爆炸）。
_MAX_ALT = 6
_ASSIGN_CAP = 256


# --------------------------------------------------------------------------- #
# 候选原子谓词
# --------------------------------------------------------------------------- #
def _flatten(snaps: Any) -> list[dict]:
    """``[快照, ...]`` 或 ``{目标: [快照, ...]}`` 一律摊平成快照列表。"""
    if isinstance(snaps, Mapping):
        return [s for group in snaps.values() for s in group]
    return list(snaps or [])


def _task_input_names(variables: Sequence) -> set:
    """变量表里由任务输入初始化的那些名字（``init_from`` 非空）。它们不参与分岔条件。"""
    return {v.name for v in variables if getattr(v, "init_from", None)}


def candidate_atoms(snaps: Any, variables: Sequence) -> list[str]:
    """枚举候选原子谓词：数值取相邻观测的中点与端点、字符串取等值/不等值。

    ``snaps`` 可以是快照列表，也可以是 ``{目标: [快照]}``（会摊平）。``variables`` 是机器
    的变量表，只用来读类型：类型说是数就按数处理，没登记类型就看实测值。集合/数组这类不
    直接产原子——它们要么进不了有限格局，要么该由 ``empty()``/``nonempty()`` 表达。

    **任务输入不产原子**（``init_from`` 非空的变量，见 :data:`_task_input_names`）。它们是
    「这次是哪道题」的身份——输入簿路径、题面原文，每次运行都不同。拿它们当分岔谓词，学出来
    的必然是「如果输入簿是 /var/folders/…/s2f_sandbox_hbs1e473/input.xlsx 就走这边」：在训练
    的那几条轨迹上百分之百可分，换一次运行、换一个沙箱就永远为假。实测产物里真出现过这条。
    分岔条件要读的是**执行中发生的事**（工具的退出码、判断的标签、计数器），不是这道题叫什么。

    返回**保序去重**的表达式串列表：顺序即搜索顺序，同一批快照两次调用给出同一个列表。
    """
    allsnaps = _flatten(snaps)
    vtypes = {v.name: v.type for v in variables}
    skip = _task_input_names(variables)
    out: list[str] = []
    keys: set = set()
    for s in allsnaps:
        keys |= set(s.keys())
    for k in sorted(keys - skip):
        vals = [s.get(k) for s in allsnaps if k in s]
        t = vtypes.get(k)
        if t in ("integer", "number") or all(isinstance(v, (int, float))
                                             and not isinstance(v, bool) for v in vals):
            nums = sorted({float(v) for v in vals
                           if isinstance(v, (int, float)) and not isinstance(v, bool)})
            for a, b in zip(nums, nums[1:]):
                mid = (a + b) / 2
                out += [f"{k} < {mid}", f"{k} >= {mid}"]
            for n in nums:
                out += [f"{k} < {n}", f"{k} >= {n}", f"{k} == {int(n) if n.is_integer() else n}"]
        else:
            for v in {v for v in vals if isinstance(v, str)}:
                out += [f"{k} == {v!r}", f"{k} != {v!r}"]
    seen, uniq = set(), []
    for e in out:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


# --------------------------------------------------------------------------- #
# 分离谓词搜索
# --------------------------------------------------------------------------- #
def _truth(expr: str, snap: dict) -> Optional[bool]:
    """在一个快照上求值。条件非法或撞未定义变量 → ``None``（既不算真也不算假）。"""
    try:
        return bool(cond.evaluate(expr, snap))
    except cond.CondError:
        return None


def _iter_exprs(atoms: Sequence[str], max_atoms: int) -> Iterator[str]:
    """搜索顺序：先逐个单原子，再两两合取，依次到 ``max_atoms`` 元合取。"""
    for r in range(1, max(1, int(max_atoms)) + 1):
        for combo in itertools.combinations(atoms, r):
            yield " and ".join(combo)


def _separates(expr: str, mine: Sequence[dict], others: Sequence[dict]) -> bool:
    """在 ``mine`` 上恒真、在 ``others`` 上恒假（求值出错一律不算分开）。"""
    return (all(_truth(expr, s) is True for s in mine)
            and all(_truth(expr, s) is False for s in others))


def separating(atoms: Sequence[str], mine: Sequence[dict], others: Sequence[dict],
               *, max_atoms: int = 2) -> Optional[str]:
    """找一个把 ``mine`` 与 ``others`` 完全分开的谓词。找不到返回 ``None``。"""
    for expr in _iter_exprs(atoms, max_atoms):
        if _separates(expr, mine, others):
            return expr
    return None


# --------------------------------------------------------------------------- #
# 关二：留出份
# --------------------------------------------------------------------------- #
def _holdout_split(snaps: Sequence[dict], ratio: float) -> tuple[list[dict], list[dict]]:
    """确定性地把一支的快照切成（拟合份, 留出份）。**不碰随机数**。

    按等距抽样取留出份（``ratio=0.2`` ⇒ 每 5 条抽 1 条），而不是切尾巴：轨迹是按长度排过
    序的，切尾巴会让留出份系统性地全是长轨迹。少于 2 条的支留不出，留出份为空。
    """
    items = list(snaps)
    if ratio <= 0 or len(items) < 2:
        return items, []
    stride = max(2, int(round(1.0 / ratio)))
    hold = items[::stride]
    holdset = set(range(0, len(items), stride))
    fit = [s for i, s in enumerate(items) if i not in holdset]
    if not fit:                                  # 极端比例下别把拟合份抽空
        return items, []
    return fit, hold


def _holdout_rate(expr: str, hold_mine: Sequence[dict],
                  hold_others: Sequence[dict]) -> float:
    """条件在留出份上的正确率：自家的要判真、别家的要判假。留不出份时按 1.0 记。"""
    total = len(hold_mine) + len(hold_others)
    if not total:
        return 1.0
    ok = sum(1 for s in hold_mine if _truth(expr, s) is True)
    ok += sum(1 for s in hold_others if _truth(expr, s) is False)
    return ok / total


# --------------------------------------------------------------------------- #
# 关三：互斥可证（符号层，不是「观测上碰巧不重叠」）
# --------------------------------------------------------------------------- #
def _domain(var: str, atoms: Sequence[cond.Atom], observed: Sequence,
            vtypes: Mapping[str, str]) -> Optional[list]:
    """给一个变量定有限枚举域：条件里的阈值/字面量 + 实测值 + 一个「其它」代表。"""
    va = [a for a in atoms if a.var == var]
    if any(a.op in ("empty", "nonempty") for a in va):
        return [[], [1]]                                  # 空 / 非空 两个代表值
    nums = sorted({float(a.const) for a in va
                   if a.op in ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")
                   and isinstance(a.const, (int, float)) and not isinstance(a.const, bool)})
    obs_nums = sorted({float(v) for v in observed
                       if isinstance(v, (int, float)) and not isinstance(v, bool)})
    if vtypes.get(var) == "boolean":
        return [True, False]
    if nums or obs_nums or vtypes.get(var) in ("integer", "number"):
        reps = set(nums) | set(obs_nums)
        if nums:
            reps |= {min(nums) - 1, max(nums) + 1}
        elif not reps:
            reps = {0, 1}
        return sorted(reps)
    strs = {a.const for a in va if a.op in ("Eq", "NotEq") and isinstance(a.const, str)}
    strs |= {v for v in observed if isinstance(v, str)}
    if strs:
        return sorted(strs) + [_OTHER]
    return None


def mutually_exclusive(exprs: Sequence[str], snaps: Any = (),
                       variables: Sequence = ()) -> bool:
    """证明一组条件两两不同时成立：把变量压成有限格局，逐个格局枚举。

    枚举域由条件里出现的原子（阈值、字面量）加上实测值撑起来——数值取阈值本身与两侧各一个
    代表值，字符串取出现过的字面量再加一个「其它」。定不了域、或格局数超过
    :data:`_CONFIG_CAP`，一律**判证不了**（返回假）：证不了就不收，宁可接 FALLBACK。
    """
    exprs = [e for e in exprs if e]
    if len(exprs) < 2:
        return True                                       # 0/1 条条件天然互斥
    atoms: list[cond.Atom] = []
    allvars: set[str] = set()
    for e in exprs:
        try:
            atoms += cond.atoms_of(e)
            allvars |= cond.vars_of(e)
        except cond.CondError:
            return False
    allsnaps = _flatten(snaps)
    vtypes = {v.name: v.type for v in variables}
    domains: dict[str, list] = {}
    size = 1
    for v in sorted(allvars):
        observed = [s[v] for s in allsnaps if v in s]
        dom = _domain(v, atoms, observed, vtypes)
        if not dom:
            return False
        domains[v] = dom
        size *= len(dom)
        if size > _CONFIG_CAP:
            return False
    names = list(domains)
    for combo in itertools.product(*(domains[n] for n in names)):
        env = dict(zip(names, combo))
        fired = sum(1 for e in exprs if _truth(e, env) is True)
        if fired >= 2:
            return False
    return True


# --------------------------------------------------------------------------- #
# 学分岔条件（过三关）
# --------------------------------------------------------------------------- #
def learn_cond(snaps_by_target: Mapping[str, Sequence[dict]], variables: Sequence,
               *, max_atoms: int = 2, min_support: int = 2,
               holdout_ratio: float = 0.2, acc_thr: float = 0.9) -> Optional[dict]:
    """在各目标的变量快照上找一组两两互斥的区分谓词。学不出返回 ``None``。

    ``snaps_by_target`` = ``{目标状态: [变量快照, ...]}``。返回 ``{目标状态: 条件串}``，
    兜底目标的条件由调用方决定留不留（``compiler._solve_branches`` 把支持度最大的那一支
    留空当兜底边）。

    三关缺一不可（见模块文档）：支持度 ``min_support``、留出正确率 ``acc_thr``、互斥可证。
    ``max_atoms`` 是合取的元数上限（默认 2：先试单原子，再试两两合取）。
    """
    targets = list(snaps_by_target)
    if len(targets) < 2:
        return None
    # 关一：支持度——每一支都得有足够多的观测
    for tgt in targets:
        if len(snaps_by_target[tgt]) < min_support:
            return None

    atoms = candidate_atoms(snaps_by_target, variables)
    split = {t: _holdout_split(snaps_by_target[t], holdout_ratio) for t in targets}

    def passing(tgt: str, limit: int) -> list[str]:
        """这一支过了关一关二的候选（按搜索顺序，最多 ``limit`` 个）。"""
        fit_mine, hold_mine = split[tgt]
        fit_others = [s for o in targets if o != tgt for s in split[o][0]]
        hold_others = [s for o in targets if o != tgt for s in split[o][1]]
        out: list[str] = []
        for cand in _iter_exprs(atoms, max_atoms):
            # 关二：在拟合份上分得开，且在留出份上仍然对——背下训练快照的谓词在这里被筛掉
            if not _separates(cand, fit_mine, fit_others):
                continue
            if _holdout_rate(cand, hold_mine, hold_others) < acc_thr:
                continue
            out.append(cand)
            if len(out) >= limit:
                break
        return out

    # 快路：每支取第一个过关的候选，直接过关三
    first = {t: passing(t, 1) for t in targets}
    if any(not v for v in first.values()):
        return None
    chosen = {t: v[0] for t, v in first.items()}
    if mutually_exclusive(list(chosen.values()), snaps_by_target, variables):
        return chosen

    # 关三：互斥可证（符号层枚举，不只是观测上不重叠）。各支的首选凑不成互斥的一组时，
    # 退一步在每支的前几个候选里找一组能互斥的——「这一支单独看最好」不等于「合起来最好」。
    alts = {t: passing(t, _MAX_ALT) for t in targets}
    if any(not v for v in alts.values()):
        return None
    tried = 0
    for combo in itertools.product(*(alts[t] for t in targets)):
        tried += 1
        if tried > _ASSIGN_CAP:
            break
        cand = dict(zip(targets, combo))
        if mutually_exclusive(list(cand.values()), snaps_by_target, variables):
            return cand
    return None


# --------------------------------------------------------------------------- #
# 循环上限 K：文档说了算，文档没说才由编译器补
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoopBound:
    """一次 K 取值的台账。``source`` ∈ ``"document"`` / ``"compiler"``。"""

    k: int
    source: str
    observed_max: int
    margin: float
    doc_bound: Optional[int] = None


def loop_bound_detail(observed_max: int, *, margin: float = 1.5,
                      doc_bound: Optional[int] = None) -> LoopBound:
    """算 K 并**留下它是谁定的**。文档给了上限就用文档的，否则编译器按余量补一个。"""
    obs = max(int(observed_max or 0), 1)
    if doc_bound is not None:
        return LoopBound(k=max(1, int(doc_bound)), source="document",
                         observed_max=obs, margin=float(margin), doc_bound=int(doc_bound))
    k = max(1, math.ceil(float(margin) * obs))
    return LoopBound(k=k, source="compiler", observed_max=obs, margin=float(margin))


def loop_bound(observed_max: int, *, margin: float = 1.5,
               doc_bound: Optional[int] = None) -> int:
    """K = ``ceil(margin × 观测最大圈数)``；**文档写了上限则文档赢**。

    要连同来源一起拿（覆盖报告要照实说这条上限是不是编译器自己加的），用
    :func:`loop_bound_detail`。
    """
    return loop_bound_detail(observed_max, margin=margin, doc_bound=doc_bound).k


# --------------------------------------------------------------------------- #
# 回边计数变量 + 上限出口
# --------------------------------------------------------------------------- #
def install_counter(machine: Machine, src: str, dst: str, *, k: int,
                    name: str) -> Transition:
    """给回边 ``src→dst`` 配一个计数变量 ``name`` 与上限 ``k`` 的出口，返回那条出口边。

    做三件事，缺一条就不成立：给回边挂 ``inc``；把目标状态**原有的条件出边**统统 and 上
    ``name < k``；在目标状态出边表的**最前面**插一条 ``name >= k → FALLBACK``。第二件是
    互斥（定理2）的要害——不 and 上去，「计满走 FALLBACK」与「继续循环」会在计满那一格同时
    成立，结构检查当场报重叠。

    幂等：目标状态上已经有一条读 ``name`` 的兜底出口时，原样返回那条边、不重复装。

    **文档已经把这个环封住时，不再自己封一遍。** 文档骨架常常自带上限出口
    （``repair_count >= 3 → s_done``：转数用尽就交一份未验证的结果）。再插一条同名同阈值的
    ``repair_count >= 3 → FALLBACK``，两条在计满那一格同时成立，结构检查当场判条件重叠
    （实测：不摘文档边之后整批回边因此被驳）。所以先认文档那条出口，只补 ``inc`` 与其余条件
    上的 ``count < k``。
    """
    st = machine.states[src]
    edge = next((t for t in st.transitions if t.to == dst), None)
    if edge is None:
        raise KeyError(f"{src} 没有一条通向 {dst} 的边，装不了计数")
    if not machine.var(name):
        machine.variables.append(Variable(name=name, type="integer", init=0))
    edge.inc = name
    tgt = machine.states[dst]
    exist = next((g for g in tgt.transitions
                  if g.cond and name in cond.vars_of(g.cond) and g.to == machine.fallback),
                 None)
    if exist is not None:
        return exist
    # 文档自带的上限出口（去哪儿都算，不必是 FALLBACK）：认它，不再插一条同名的
    doc_exit = next((g for g in tgt.transitions
                     if g.cond and name in cond.vars_of(g.cond)), None)
    for g in tgt.transitions:                    # 原有条件 and 上 count<k，保互斥
        if g.cond and name not in cond.vars_of(g.cond):
            g.cond = f"({g.cond}) and {name} < {k}"
    if doc_exit is not None:
        return doc_exit
    exit_edge = Transition(cond=f"{name} >= {k}", to=machine.fallback)
    tgt.transitions.insert(0, exit_edge)
    return exit_edge


def install_counters(machine: Machine, max_visits: Mapping[str, int], *,
                     margin: float = 1.5,
                     doc_bounds: Optional[Mapping[str, int]] = None) -> list[dict]:
    """给机器里每条还没配计数的回边装上计数与上限出口，返回 K 的取值台账。

    ``max_visits`` = ``{状态: 该状态在单条轨迹里被进入的最大次数}``；``doc_bounds`` 是文档
    明写的圈数上限（``{状态: K}``），给了就压过观测推断。台账每项形如
    ``{"back_edge","var","k","source","observed_max","margin"}``，``source`` 说明这条 K 是
    文档要求还是编译器补的——覆盖报告直接照抄。
    """
    ledger: list[dict] = []
    reachable = _reachable(machine)
    for src, edge in _back_edges(machine, reachable):
        if edge.inc:
            continue
        tgt_id = edge.to
        name = f"{tgt_id}_count"
        lb = loop_bound_detail(max_visits.get(tgt_id, 1), margin=margin,
                               doc_bound=(doc_bounds or {}).get(tgt_id))
        install_counter(machine, src, tgt_id, k=lb.k, name=name)
        ledger.append({"back_edge": f"{src}->{tgt_id}", "var": name, "k": lb.k,
                       "source": lb.source, "observed_max": lb.observed_max,
                       "margin": lb.margin})
    return ledger


# --------------------------------------------------------------------------- #
# 标定判断动作的误差率（编译期唯一的模型触点）
# --------------------------------------------------------------------------- #
def calibrate(judge, samples: Sequence[tuple], *, model) -> float:
    """在带正确标签的快照上跑判断，量出误差率。

    ``samples`` = ``[(变量快照, 正确标签), ...]``。误差率 = **非弃权**里判错的比例；弃权
    单独计、不算错（弃权是「我不知道」，代价是走 FALLBACK，不是走错）。全弃权时返回 0.0。
    """
    errors = nonabstain = 0
    for values, gold in samples:
        vread = {k: values.get(k) for k in judge.reads}
        pred = model.classify(prompt=judge.prompt, values=vread,
                              labels=judge.labels,
                              examples=tuple(e.model_dump() for e in judge.examples))
        if pred == judge.abstain:
            continue
        nonabstain += 1
        if pred != gold:
            errors += 1
    return errors / nonabstain if nonabstain else 0.0
