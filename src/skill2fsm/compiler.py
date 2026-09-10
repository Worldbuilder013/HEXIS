"""顺序转向编译：沿真实执行轨迹逐动作走，把技能归纳成一台状态机。

编译器是「编译智能体 + 确定性守门程序」的合体。这份文件是**守门程序**那一半：对齐、建状态、
接边、成环、学分岔条件、标定误差率、结构检查——全是确定性计算，可复算、可撤销。语义判断
（这一步属哪个条款、读哪些变量、分岔处该问什么问题）在真实系统里由编译智能体给，这里对
玩具技能用确定性启发式代理，好让整条编译在无网络、秒级下自测。**模型只在两处出现**：
:func:`make_judge`（分岔学不出确定条件时，起草一个判断动作）与 :func:`calibrate`（在快照上
标定那个判断动作的误差率）。其余一律不碰模型。

**这份文件只剩流程骨架**：学分岔条件、装回边计数与上限出口、标定误差率这三段与流程无关、
任何一台机器都用得上的算法，住在 :mod:`skill2fsm.fit`；判「同一步」的动作签名住在
:mod:`skill2fsm.normalize`。本模块从那两处 import 并按原样对外转出（``learn_cond`` /
``calibrate`` 的调用方无需改动）。

算法沿轨迹走（短的优先），逐动作与当前状态对齐：对得上就前移并给这条边记一次支持；当前
状态还没装动作就装上；下一步的动作若和某个已有状态相同，就接回那个状态成环（合并
≈-等价的历史，对应 Myhill-Nerode 的状态最小性）；一个状态长出第二条通向不同后继的边，
就是分岔，在两侧的变量快照上学一个区分条件，学不出才起判断动作。最后统一做结构检查，
违规就撤销最近的构造。
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from . import fit
from .checks import structural_findings
from .fit import learn_cond          # noqa: F401  （对外转出：调用方一直从这里拿）
from .normalize import canon_action
from .replay import excludes, reproduces
from .schema import (
    EndAction, Example, JudgeAction, Machine, State, Terminal, ToolAction,
    Transition, Variable,
)

_ABSTAIN = "弃权"


# --------------------------------------------------------------------------- #
# 文档条款切分（对应算法1 L1「给条款编号」）
# --------------------------------------------------------------------------- #
_CLAUSE_RE = re.compile(r"^#+\s*(S\d+(?:\.\d+)?|P\d+)\b")


def partition(doc: str) -> list[tuple[str, str]]:
    """把 SKILL.md 正文按 ``## Sx`` / ``### Sx.y`` / ``## Px`` 标题切成带编号的条款块。"""
    clauses: list[tuple[str, str]] = []
    cur_id: Optional[str] = None
    buf: list[str] = []
    for line in doc.splitlines():
        m = _CLAUSE_RE.match(line)
        if m:
            if cur_id:
                clauses.append((cur_id, "\n".join(buf)))
            cur_id, buf = m.group(1), [line]
        elif cur_id:
            buf.append(line)
    if cur_id:
        clauses.append((cur_id, "\n".join(buf)))
    return clauses


def _infer_clause(action: dict, clauses: list[tuple[str, str]]) -> str:
    """把一个动作归到最贴的条款。工具按名字在正文里找，判断优先带「判据」的最细条款。"""
    kind = action.get("kind")
    if kind == "tool":
        name = action.get("name", "")
        hits = [cid for cid, text in clauses if name and name in text]
        return hits[0] if hits else ""
    if kind == "judge":
        specific = [cid for cid, text in clauses if "判据" in text]
        if specific:
            return max(specific, key=len)       # S2.1 比 S2 更细
        hits = [cid for cid, text in clauses if "规范" in text]
        return max(hits, key=len) if hits else ""
    return ""


# --------------------------------------------------------------------------- #
# 动作签名：判「同一步」（真实系统里由编译智能体判新/重复，玩具用签名代理）
# --------------------------------------------------------------------------- #
def _sig(action: dict) -> tuple:
    """编译档的动作 KEY，委托给 :func:`skill2fsm.normalize.canon_action`（``strict=True``）。

    传的是**裸 action dict**（编译期手上只有 ``rec.action``），所以 judge/model 的 writes
    反推不出来、一律为空——严档因此恰好退回本函数原先「工具比名字、判断比提问、终止比
    terminal」的分组。test_14 在 table_clean 的记录表上逐对钉死了这一点。
    """
    return canon_action(action, strict=True)


# --------------------------------------------------------------------------- #
# 从轨迹记录推断动作的 reads/writes（确定性启发式，代理编译智能体）
# --------------------------------------------------------------------------- #
def _infer_writes(rec_action: dict, output: dict) -> list[str]:
    if rec_action.get("kind") == "judge":
        return list(output.keys())              # judge 的 output 就是 {写入变量: 标签}
    return [k for k in output.keys() if k != "ok"]


def _infer_reads(rec_action: dict, prev_vars: dict) -> list[str]:
    if "reads" in rec_action:                   # judge 动作运行时已记 reads
        return list(rec_action["reads"])
    reads: list[str] = []
    for _pk, pv in (rec_action.get("input") or {}).items():
        for var, val in prev_vars.items():
            if val == pv and var not in reads:
                reads.append(var)
    return reads


def _templatize(inp: dict, prev_vars: dict) -> dict:
    """把具体 input 值反写成 ``${var}`` 模板（值等于某变量当前值时）。"""
    out: dict = {}
    for k, v in (inp or {}).items():
        hit = next((var for var, val in prev_vars.items() if val == v), None)
        out[k] = f"${{{hit}}}" if hit else v
    return out


# --------------------------------------------------------------------------- #
# 编译状态：机器 + 增量台账
# --------------------------------------------------------------------------- #
@dataclass
class _Build:
    machine: Machine
    sig2sid: dict = field(default_factory=dict)      # 动作签名 → 状态 id
    #: 每个状态执行后走向哪个后继 + 当时的变量快照（供学分岔条件 / 标定）
    branch_obs: dict = field(default_factory=lambda: defaultdict(list))
    #: judge 状态观测到的标签集合（定 labels）
    judge_labels: dict = field(default_factory=lambda: defaultdict(set))
    #: 每个状态在**单条轨迹**里被进入的最大次数（定循环上限）
    max_visits: dict = field(default_factory=lambda: defaultdict(int))
    #: 任务输入的字段名（这些变量 init_from task.input）
    input_keys: set = field(default_factory=set)
    counter: int = 0

    def new_sid(self) -> str:
        self.counter += 1
        return f"s{self.counter}"


def _seed(skill_id: str) -> Machine:
    """初始机器：一个占位起点 + FALLBACK + done 终止。起点的动作沿第一条轨迹装上。"""
    return Machine(
        skill_id=skill_id,
        initial="s0",
        states={
            "s0": State(id="s0", action=EndAction(terminal="__placeholder__")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
            "end": State(id="end", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done", output=[])],
    )


def _is_placeholder(state: State) -> bool:
    return state.action.kind == "end" and getattr(state.action, "terminal", "") == "__placeholder__"


def _install(build: _Build, sid: str, rec, prev_vars: dict, clauses) -> None:
    """把一条记录的动作装进状态。"""
    ra = rec.action
    kind = ra.get("kind")
    clause = _infer_clause(ra, clauses)
    writes = _infer_writes(ra, rec.output)
    reads = _infer_reads(ra, prev_vars)
    st = build.machine.states[sid]
    st.clause = clause
    if kind == "tool":
        st.action = ToolAction(name=ra["name"],
                               input=_templatize(ra.get("input", {}), prev_vars),
                               reads=reads, writes=writes)
    elif kind == "judge":
        wr = writes or ["header_ok"]
        for w in writes:
            build.judge_labels[sid].add(rec.output.get(w))
        first = rec.output.get(wr[0])
        lbls = ([first, _ABSTAIN] if first and first != _ABSTAIN else [_ABSTAIN])
        st.action = JudgeAction(prompt=ra.get("prompt", ""),
                                reads=reads or ["header_row"],
                                writes=wr,
                                labels=lbls)        # _finalize 再补全全部观测标签
    st.transitions = []
    build.sig2sid[_sig(ra)] = sid


# --------------------------------------------------------------------------- #
# 沿一条轨迹走，建状态与边，累积分支观测
# --------------------------------------------------------------------------- #
def _walk(build: _Build, records, clauses, initial_vars=None) -> None:
    m = build.machine
    p = m.initial
    prev_vars: dict = dict(initial_vars or {})       # 起点动作要能把 task.input 反写成 ${var}
    visits: dict = defaultdict(int)
    try:
        for i, rec in enumerate(records):
            visits[p] += 1
            if rec.action.get("kind") == "end":
                _add_edge(m, p, "end")
                return
            # 装动作到 p（若占位或签名匹配）
            st = m.states[p]
            if _is_placeholder(st):
                _install(build, p, rec, prev_vars, clauses)
            elif _sig(rec.action) != _sig({"kind": st.action.kind,
                                            "name": getattr(st.action, "name", None),
                                            "prompt": getattr(st.action, "prompt", None)}):
                # p 的动作与本记录不符：接一条边到 FALLBACK（编译尚浅，交解释兜底）
                _add_edge(m, p, m.fallback)
                return
            # 累积 judge 观测标签
            if m.states[p].action.kind == "judge":
                for w in m.states[p].action.writes:
                    if w in rec.output:
                        build.judge_labels[p].add(rec.output[w])
            # 决定下一状态
            nxt = records[i + 1] if i + 1 < len(records) else None
            if nxt is None:
                _add_edge(m, p, "end")
                return
            if nxt.action.get("kind") == "end":
                _add_edge(m, p, "end")
                build.branch_obs[p].append(("end", dict(rec.vars)))
                return
            nsig = _sig(nxt.action)
            if nsig in build.sig2sid:
                tgt = build.sig2sid[nsig]           # 重复：接回已有状态（可能成环）
            else:
                tgt = build.new_sid()
                m.states[tgt] = State(id=tgt, action=EndAction(terminal="__placeholder__"))
            _add_edge(m, p, tgt)
            build.branch_obs[p].append((tgt, dict(rec.vars)))
            prev_vars = dict(rec.vars)
            p = tgt
    finally:
        for sid, c in visits.items():
            build.max_visits[sid] = max(build.max_visits[sid], c)


def _add_edge(m: Machine, src: str, dst: str) -> None:
    """加一条（暂无条件的）边，或给已存在的同目标边 +1 支持。"""
    for t in m.states[src].transitions:
        if t.to == dst:
            t.support += 1
            return
    m.states[src].transitions.append(Transition(to=dst, support=1))


# --------------------------------------------------------------------------- #
# 起草判断动作（make_judge）
#
# 学分岔条件（learn_cond / candidate_atoms / separating）已搬进 :mod:`skill2fsm.fit`，
# 本模块顶上按原名转出，调用方照旧 ``compiler.learn_cond``。
# --------------------------------------------------------------------------- #
# ⚠️ **已弃用（dead code）**：本函数从不使用它的 ``model`` 形参——起草完全是确定性的；
# 而它唯一的调用点（``_solve_branches``）喂进来的 labels 取自快照里的 ``__lbl__`` 键，那个
# 键**全仓没有任何地方写过**，所以真实编译里 labels 恒为 ``[""]``。原样留着不动（改它等于
# 改一条从未跑通的路径），别在它上面接新东西：要起判断动作，重写一条带真标签来源的路。
def make_judge(prompt: str, reads: list[str], snaps_by_target: dict,
               labels: list[str], model) -> tuple[JudgeAction, dict]:
    """分岔学不出确定条件时，起草一个判断动作。**已弃用，见上方注释。**

    给每个目标分配一个标签，样例取自各目标的变量快照。返回 (judge, {目标: 条件串})，条件
    形如 ``verdict == '<标签>'``，读的是 judge 写入的裁决变量。
    """
    from .schema import Example
    targets = list(snaps_by_target)
    verdict_var = "verdict"
    tgt_label = {tgt: (labels[i] if i < len(labels) else f"L{i}")
                 for i, tgt in enumerate(targets)}
    examples = []
    for tgt, snaps in snaps_by_target.items():
        for s in snaps[:2]:
            ex = {k: s.get(k) for k in reads}
            ex["label"] = tgt_label[tgt]
            examples.append(Example(**ex))
    lbls = list(tgt_label.values()) + [_ABSTAIN]
    judge = JudgeAction(prompt=prompt or "该走哪一支", reads=reads,
                        writes=[verdict_var], labels=lbls, examples=examples)
    conds = {tgt: f"{verdict_var} == {lab!r}" for tgt, lab in tgt_label.items()}
    return judge, conds


def calibrate(judge: JudgeAction, labeled_snaps: list[tuple], model) -> tuple[float, int]:
    """在带正确标签的快照上跑判断，标出误差率与支持度。**编译期第二个模型触点。**

    ``labeled_snaps`` = ``[(变量快照, 正确标签), ...]``。误差率 = 非弃权里判错的比例；弃权
    单独计不算错。支持度 = 快照数。算法本体在 :func:`skill2fsm.fit.calibrate`。
    """
    rate = fit.calibrate(judge, labeled_snaps, model=model)
    return rate, len(labeled_snaps)


# --------------------------------------------------------------------------- #
# 分支求解：给一个多出边状态定条件
# --------------------------------------------------------------------------- #
def _solve_branches(build: _Build, sid: str, thresholds, model) -> None:
    m = build.machine
    st = m.states[sid]
    obs = build.branch_obs.get(sid, [])
    targets = []
    for t in st.transitions:
        if t.to not in targets:
            targets.append(t.to)
    if len(targets) < 2:
        return                                     # 单出边：无需条件
    snaps_by_target: dict = defaultdict(list)
    for tgt, snap in obs:
        snaps_by_target[tgt].append(snap)
    # 支持度不足的目标 → 该分岔整体接 FALLBACK
    if any(len(snaps_by_target.get(t, [])) < thresholds.min_support for t in targets):
        st.transitions = [Transition(to=m.fallback)]
        return
    learned = fit.learn_cond(snaps_by_target, m.variables,
                             min_support=thresholds.min_support,
                             holdout_ratio=thresholds.holdout_ratio,
                             acc_thr=thresholds.acc_thr)
    if learned is None and model is not None:
        # 学不出确定条件 → 起判断动作（模型触点），改状态为 judge 前的裁决
        judge, conds = make_judge(getattr(st.action, "prompt", ""),
                                  getattr(st.action, "reads", []) or ["header_row"],
                                  snaps_by_target,
                                  sorted({s.get("__lbl__", "") for s in obs}), model)
        st.action = judge
        learned = conds
    if learned is None:
        st.transitions = [Transition(to=m.fallback)]
        return
    # 落条件：按目标写回，最大支持的目标做兜底（留空条件）
    fallback_tgt = max(targets, key=lambda t: len(snaps_by_target.get(t, [])))
    new_edges = []
    for tgt in targets:
        if tgt == fallback_tgt:
            continue
        new_edges.append(Transition(cond=learned[tgt], to=tgt,
                                     support=len(snaps_by_target.get(tgt, []))))
    new_edges.append(Transition(to=fallback_tgt,
                                support=len(snaps_by_target.get(fallback_tgt, []))))
    st.transitions = new_edges


# --------------------------------------------------------------------------- #
# 回边计数变量 + 上限出口
# --------------------------------------------------------------------------- #
def _install_counters(build: _Build, thresholds) -> list[dict]:
    """给每条回边配一个计数变量与上限出口，并让回边目标的已有条件与出口互斥。

    上限 K 与「谁定的 K」都由 :func:`skill2fsm.fit.loop_bound_detail` 算：文档写了圈数上限
    就照文档，没写才由编译器按 ``ceil(loop_margin × 单条轨迹里该目标被访问的最大次数)``
    补一个。返回 K 的取值台账，编译结果照抄进 report——覆盖报告据此说明哪些上限是编译器
    自己加的（math-skill 的解题主干通篇没写过圈数上限，所以全是编译器加的）。
    """
    return fit.install_counters(build.machine, build.max_visits,
                                margin=thresholds.loop_margin)


# --------------------------------------------------------------------------- #
# 落 judge 的 labels / 变量表补全 / 清理占位
# --------------------------------------------------------------------------- #
def _finalize(build: _Build) -> None:
    m = build.machine
    for sid, st in m.states.items():
        if st.action.kind == "judge":
            observed = sorted(x for x in build.judge_labels.get(sid, set()) if x)
            if _ABSTAIN not in observed:
                observed = observed + [_ABSTAIN]
            st.action.labels = observed
            # 样例取自轨迹：每个观测标签留一个变量快照作代表
            wkey = st.action.writes[0]
            exs, seen = [], set()
            for _tgt, snap in build.branch_obs.get(sid, []):
                lbl = snap.get(wkey)
                if lbl and lbl not in seen:
                    ex = {k: snap.get(k) for k in st.action.reads}
                    ex["label"] = lbl
                    exs.append(Example(**ex))
                    seen.add(lbl)
            st.action.examples = exs
    # 变量表：把出现过的变量补进去；任务输入的字段标 init_from。
    known = {v.name for v in m.variables}
    used: dict[str, str] = {}
    for st in m.states.values():
        for w in getattr(st.action, "writes", []) or []:
            used.setdefault(w, "array" if w == "rows" else "string")
        for r in getattr(st.action, "reads", []) or []:
            used.setdefault(r, "array" if r == "rows" else "string")
    for name, t in used.items():
        if name not in known:
            ifrom = f"task.input.{name}" if name in build.input_keys else None
            m.variables.append(Variable(name=name, type=t, init_from=ifrom))
            known.add(name)
    for v in m.variables:                            # 已有变量若是输入字段，补 init_from
        if v.name in build.input_keys and v.init is None and v.init_from is None:
            v.init_from = f"task.input.{v.name}"


# --------------------------------------------------------------------------- #
# 顶层：一轮编译
# --------------------------------------------------------------------------- #
@dataclass
class CompileResult:
    machine: Machine
    calibration: dict
    report: dict
    findings: list[str]


def compile(doc: str, t_plus: list, t_minus: Optional[list] = None,
            thresholds=None, *, skill_id: str = "compiled", model=None,
            prohibitions: Optional[list] = None) -> CompileResult:
    """从接受轨迹（+可选拒绝轨迹）顺序转向编译出一台机器。

    ``prohibitions`` 是人工标出的禁止性要求，直接写进机器——它们编不进图（禁止性违规在
    结构上和正常执行一样），靠评判层拦。
    """
    from .schema import Thresholds
    thresholds = thresholds or Thresholds()
    t_minus = t_minus or []
    clauses = partition(doc)
    build = _Build(machine=_seed(skill_id))
    build.machine.prohibitions = list(prohibitions or [])
    for trace in t_plus:
        build.input_keys |= set(trace.task.get("input", {}))

    for trace in sorted(t_plus, key=lambda t: len(t.records)):
        _walk(build, trace.records, clauses, trace.task.get("input", {}))

    for sid in list(build.machine.states):
        _solve_branches(build, sid, thresholds, model)

    loop_bounds = _install_counters(build, thresholds)
    _finalize(build)

    # 标定判断动作误差率（若给了模型）：在观测快照上重跑判断、与实际走向对照
    calibration: dict = {}
    if model is not None:
        for sid, st in build.machine.states.items():
            if st.action.kind != "judge":
                continue
            wkey = st.action.writes[0]
            labeled = [(snap, snap.get(wkey))
                       for _t, snap in build.branch_obs.get(sid, [])
                       if snap.get(wkey)]
            if labeled:
                rate, sup = calibrate(st.action, labeled, model)
                st.action.error_rate = rate
                st.action.support = sup
                calibration[sid] = {"error_rate": rate, "support": sup}

    # 拒绝轨迹：确认被排除（阶段 C 再做主动修复；这里先记录）
    unexcluded = [i for i, neg in enumerate(t_minus) if not excludes(build.machine, neg)]

    findings = structural_findings(build.machine)
    report = {
        "n_states": build.machine.n_states(),
        "clauses_seen": sorted({s.clause for s in build.machine.states.values() if s.clause}),
        "t_plus": len(t_plus),
        "t_plus_reproduced": sum(1 for t in t_plus if reproduces(build.machine, t)),
        "t_minus": len(t_minus),
        "t_minus_excluded": len(t_minus) - len(unexcluded),
        "loop_bounds": loop_bounds,          # 每条回边的 K 与它的来源（文档 / 编译器）
    }
    return CompileResult(machine=build.machine, calibration=calibration,
                         report=report, findings=findings)


# --------------------------------------------------------------------------- #
# 增量一轮编译（带整轮撤销）
# --------------------------------------------------------------------------- #
def compile_round(base: Optional[Machine], doc: str, traces: list, *,
                  thresholds=None, skill_id: str = "compiled", model=None,
                  prohibitions: Optional[list] = None) -> tuple[CompileResult, bool]:
    """编译一轮并做整轮校验。结构检查不过就**撤销**，退回 ``base``（机器文件不变）。

    返回 ``(result, applied)``：``applied`` 为假表示这一轮被回滚，``result.machine`` 就是
    ``base``（没有 base 时是一台空机器）。这是「扩展只在校验通过时才落」的实现——注入一条
    让校验失败的轨迹，整轮不落地。
    """
    from .schema import Thresholds, empty_machine
    cr = compile(doc, traces, thresholds=thresholds or Thresholds(),
                 skill_id=skill_id, model=model, prohibitions=prohibitions)
    if cr.findings:
        fallback_machine = base if base is not None else empty_machine(skill_id)
        rolled = CompileResult(machine=fallback_machine, calibration={},
                               report={"rolled_back": True, "findings": cr.findings},
                               findings=cr.findings)
        return rolled, False
    return cr, True


# --------------------------------------------------------------------------- #
# 分裂：同一动作签名撞成一个状态，但两处需要不同的后继且本状态变量分不开
# --------------------------------------------------------------------------- #
def split_groups(preds: Sequence[str], outs: Sequence[str],
                 contingency: Optional[Mapping[str, Mapping[str, int]]] = None,
                 ) -> Optional[dict[str, list[str]]]:
    """按前驱给出「哪些前驱该合成一份克隆」的分组——分裂的**纯配对核**，不碰机器。

    ``contingency[pred][out]`` 是观测计数：从前驱 ``pred`` 进来之后走向出口 ``out`` 的次数。
    给了它就按证据分：每个前驱只往**一个**出口走（列联表每行恰好一个非零格）时，按出口把
    前驱归组，返回 ``{出口: [前驱...]}``；任何一个前驱往两个以上出口走过，前驱就分不开它们，
    返回 ``None``（那是判断动作的活，不是分裂的活）。

    不给列联表就退回**位置法**：第 i 个前驱配第 i 条出边（编译器按轨迹先后建边，顺序即
    对应）——前驱数与出边数不等或少于 2 时返回 ``None``。这是 :func:`split_by_predecessor`
    原来的规则，原样保留给旧路径。
    """
    preds, outs = list(preds), list(outs)
    if len(preds) < 2 or len(outs) < 2:
        return None
    if contingency is None:
        if len(preds) != len(outs):
            return None
        return {o: [p] for p, o in zip(preds, outs)}
    groups: dict[str, list[str]] = {}
    for p in preds:
        row = contingency.get(p) or {}
        hit = [o for o in outs if int(row.get(o, 0) or 0) > 0]
        if len(hit) != 1:
            return None                     # 这个前驱去过 0 个或 ≥2 个出口：前驱分不开
        groups.setdefault(hit[0], []).append(p)
    if len(groups) < 2:
        return None                         # 全部前驱都只去同一个出口：没什么可分
    return groups


def split_by_predecessor(machine: Machine, sid: str) -> bool:
    """把一个「按签名合并、却行为矛盾」的状态按**前驱**拆开。

    当一个状态的多条出边无法用它自己的变量区分（分岔学不出条件），但矛盾与「从哪个状态
    进来的」一一对应时，按前驱把它克隆成几份，各自只保留对应的那条出边——这是 Myhill-Nerode
    意义上「这两段历史其实不等价」的迟到修正。返回是否发生了分裂。

    配对交给 :func:`split_groups` 的位置法：第 i 个前驱配第 i 条出边。受票的版本是
    :meth:`skill2fsm.checker.Checker.split_state`（显式分组、克隆 id 由 harness 分配）。
    """
    st = machine.states.get(sid)
    if st is None:
        return False
    preds = [(src, t) for src, t in machine.transitions_all() if t.to == sid]
    outs = list(st.transitions)
    out_ids = [f"{i}" for i in range(len(outs))]
    grouping = split_groups([f"{i}" for i in range(len(preds))], out_ids)
    if grouping is None:
        return False
    for oi, members in grouping.items():
        i = int(members[0])
        out_edge = outs[int(oi)]
        _psrc, pedge = preds[i]
        clone_id = f"{sid}_{i}"
        machine.states[clone_id] = State(
            id=clone_id, clause=st.clause,
            action=st.action.model_copy(deep=True),
            transitions=[Transition(cond=out_edge.cond, to=out_edge.to,
                                    inc=out_edge.inc, support=out_edge.support)])
        pedge.to = clone_id                     # 前驱重定向到它那份克隆
    if machine.initial == sid:
        machine.initial = f"{sid}_0"
    del machine.states[sid]
    return True
