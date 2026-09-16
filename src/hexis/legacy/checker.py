"""确定性守门程序（上）：受票的机器改写接口。**本模块不调模型。**

从头到尾没有一个 ``model`` 参数、不 import 任何模型客户端、不读网络——一台机器该不该被
改成某个样子，是纯确定性的图与数据流问题，不该由一次采样来裁决。编译智能体（后面那一波）
只能通过这里的八个受票接口动机器；它自己**不许**直接改 ``Machine``，也不许写机器文件。

**为什么要有这一层。** :func:`hexis.legacy.compiler.compile_round` 已经能做「整轮编译，结构
检查不过就整轮撤销」。但一个智能体式的编译过程不是「一轮」——它是一串小提议：加个状态、
接条边、把这个分岔收成判断动作、给这个环配个计数器。整轮撤销在这种粒度上是灾难：第 7 条
提议写错了，前面 6 条正确的一起没了，智能体只能从头再来一遍，而且它不知道究竟是哪一条错
的。所以这里的规矩反过来：

* **每条提议单独裁决**——先在**候选副本**上改，改完跑一遍检查，过了副本才转正，没过则当前
  机器一个字节都没动，只多出一张写着理由的回执（:class:`Receipt`）。
* **被拒的提议不牵连它之前被接受的提议。** 这正是 ``compile_round`` 表达不了的那件事。
* 只有 :meth:`Checker.commit` 是全有全无的：它跑一遍验收（:mod:`hexis.legacy.verify`），
  不过就整批回滚到上一次 commit（或 open）时的机器。

**回执必须可据以重试。** 拒绝理由要说清是**哪一条检查**在**哪个状态**上失败的，否则智能体
只能瞎猜着改。所以结构检查返回的中文句子在这里被解析成带 ``code``/``state_id`` 的
:class:`Finding`，理由串里带上 ``[E_OVERLAP@j]`` 这种定位前缀。

检查本身**不重写**：图与数据流那八条在 :mod:`hexis.machine.checks` 里已经是对的（可达、可终止、
转移完备、互斥完备的有限格局枚举、回边上限、先写后读、输出完备），本模块只包装它，再补两
条它管不着的门槛（转移支持度、判断动作误差率）与一条它漏掉的（一个状态至多一条兜底边）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from pydantic import ValidationError

#: commit 时与 machine.json 并排写出的溯源表文件名。溯源**不进** machine.json：
#: 可执行产物要字节稳定、要私有；溯源是给人和台账读的。
PROVENANCE_FILE = "provenance.json"

from hexis.machine import cond as _cond
from hexis.legacy import fit as _fit
from hexis.machine.checks import _back_edges as _checks_back_edges_raw
from hexis.machine.checks import _reachable as _checks_reachable
from hexis.machine.checks import structural_findings
from hexis.machine.schema import (
    EndAction, JudgeAction, Machine, Prohibition, State, Terminal, Thresholds,
    Transition, Variable, empty_machine, save_machine,
)

#: 受票接口的名字。:class:`Proposal` 的 ``op`` 只能是其中之一。
#: 前八个是原始的八个；``split_state``/``mark``/``rewind`` 是多智能体编译加的三个
#: （按前驱分裂、打检查点、回退到检查点），同样走候选副本裁决、同样出回执。
OPS: tuple[str, ...] = (
    "open_machine", "add_state", "add_transition", "close_loop",
    "add_judge", "set_terminal", "demote_to_fallback", "commit",
    "split_state", "close_loops", "bound_loop", "mark", "rewind",
)

#: 溯源行的 ``origin`` 取值域。台账按它分栏：哪些结构来自文档、哪些是轨迹学出来的、哪些
#: 是编译器自作主张、哪些枚举域还没在真实执行器上标定过。
ORIGINS: tuple[str, ...] = (
    "document", "document(弱)", "trace", "compiler",
    "harness(待标定)", "harness(已标定)",
)


def prov_key(kind: str, *parts: str) -> str:
    """溯源表的键：``state:s1`` / ``edge:s1->s2#cond`` / ``judge:s3`` / ``var:x`` /
    ``terminal:END_VERIFIED`` / ``prohibition:P1`` / ``domain:audit_status``。"""
    if kind == "edge":
        src, dst, cond = parts
        return f"edge:{src}->{dst}#{cond}"
    return f"{kind}:{parts[0]}"


# --------------------------------------------------------------------------- #
# 发现与回执
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Finding:
    """一条检查结论。``severity`` ∈ ``{"error", "warn"}``。

    ``error`` 表示**机器坏了**（走不到、停不下来、条件重叠、读未写变量……），受票接口见到
    它就拒绝提议；``warn`` 表示**证据不够**（这条边只被一条轨迹走过、这个判断误差率超标），
    机器结构本身没坏，改写照样能落，但它过不了 :mod:`hexis.legacy.verify` 的验收。

    ``state_id`` 是问题所在的状态（定位不到时是空串），``code`` 供程序分派、``message``
    是原始的中文诊断句（结构检查那八条一字不改地转过来）。
    """

    code: str
    severity: str
    state_id: str
    message: str

    def located(self) -> str:
        """``[E_OVERLAP@j] 状态 j 条件重叠……``——带定位前缀的一行。"""
        where = f"@{self.state_id}" if self.state_id else ""
        return f"[{self.code}{where}] {self.message}"


@dataclass(frozen=True)
class Receipt:
    """一张回执：某次改写提议的审计痕迹。接受与拒绝都出回执，**回滚也不擦回执**。

    ``reason`` 在接受时说清改了什么、在拒绝时说清哪条检查在哪里挂了；``detail`` 装机读的
    附件（提议参数、findings 的字典形式、验收报告）。
    """

    op: str
    accepted: bool
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    """一条待裁决的改写提议：``op`` 是受票接口名，``args`` 是它的关键字参数。"""

    op: str
    args: dict = field(default_factory=dict)


class _Reject(Exception):
    """前置条件不满足（状态不存在、重复的边……）。带上可据以重试的理由。"""

    def __init__(self, reason: str, detail: Optional[dict] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


# --------------------------------------------------------------------------- #
# 结构检查的中文诊断句 → 带 code/state 的 Finding
# --------------------------------------------------------------------------- #
#: ``(正则, code)``。正则里的 ``sid`` 组给出问题所在的状态。顺序即匹配优先级。
_STRUCT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^initial 指向不存在的状态", "E_INITIAL_MISSING"),
    (r"^转移 (?P<sid>[^\s→]+)→[^\s]+ 的 \S+ 指向不存在的状态", "E_DANGLING_EDGE"),
    (r"^状态 (?P<sid>\S+) 从 \S+ 走不到", "E_UNREACHABLE"),
    (r"^没有任何终止", "E_NO_TERMINAL"),
    (r"^状态 (?P<sid>\S+) 走不到任何终止", "E_NO_STOP"),
    (r"^状态 (?P<sid>\S+) 没有任何出边", "E_NO_EDGE"),
    (r"^状态 (?P<sid>\S+) 的出边都带条件", "E_NO_DEFAULT"),
    (r"^状态 (?P<sid>\S+) 的条件原子过多", "E_ATOM_CAP"),
    (r"^状态 (?P<sid>\S+) 的条件用到无法定域", "E_UNDECIDABLE_VAR"),
    (r"^状态 (?P<sid>\S+) 条件重叠", "E_OVERLAP"),
    (r"^状态 (?P<sid>\S+) 条件有空隙", "E_GAP"),
    (r"^回边 (?P<sid>[^\s→]+)→[^\s]+ 没有计数变量", "E_LOOP_UNCOUNTED"),
    (r"^回边 (?P<sid>[^\s→]+)→[^\s]+ 的计数变量", "E_LOOP_UNBOUNDED"),
    (r"^状态 (?P<sid>\S+) 要读 ", "E_READ_BEFORE_WRITE"),
    (r"^终止态 (?P<sid>\S+) 声明输出", "E_OUTPUT_MISSING"),
)


def classify(message: str) -> tuple[str, str]:
    """把一条结构诊断句认成 ``(code, state_id)``。认不出来落 ``E_STRUCT`` + 空状态。"""
    for pat, code in _STRUCT_PATTERNS:
        m = re.match(pat, message)
        if m:
            sid = (m.groupdict().get("sid") or "") if m.groupdict() else ""
            return code, sid
    return "E_STRUCT", ""


# --------------------------------------------------------------------------- #
# 结构检查管不着的三条门槛
# --------------------------------------------------------------------------- #
def _default_edges(state: State) -> list[Transition]:
    return [t for t in state.transitions if not t.cond]


def _multi_default_findings(m: Machine) -> list[Finding]:
    """一个状态至多一条兜底边。

    :func:`hexis.execution.runtime.pick_edge` 按声明顺序取第一条为真的边，兜底边永远为真——第二
    条兜底边因此**永远走不到**。结构检查那八条不管这个（它只问「有没有兜底」），但一台机器
    里出现两条兜底边，几乎总是「本该改写已有的那条、结果又加了一条」，静默地把新接的分支
    埋掉。所以在这里判错。
    """
    out: list[Finding] = []
    for sid in sorted(m.states):
        st = m.states[sid]
        if st.action.kind == "end":
            continue
        dflt = _default_edges(st)
        if len(dflt) > 1:
            out.append(Finding(
                "E_MULTI_DEFAULT", "error", sid,
                f"状态 {sid} 有 {len(dflt)} 条兜底边（无条件出边），去向 "
                f"{[t.to for t in dflt]}：兜底边只能有一条，第二条起永远走不到——"
                "改写已有的那条，或给新边加条件"))
    return out


def _support_rows(m: Machine, thr: Thresholds) -> list[tuple[str, Transition]]:
    """支持度不足的边。三类不算。

    * **通往 FALLBACK 的边与 FALLBACK 自己的出边**：去 FALLBACK 是「这里我没学会，交回解释
      执行」——它按定义没有轨迹支持，拿支持度去要求它等于要求编译器不许认怂；
    * ``origin="document"`` 的边：那是**文档骨架**的主张，不是从轨迹编下来的。拿轨迹支持度
      要求它，等于说「文档写了而这批轨迹恰好没走到的分支不许存在」——错误处理、边界情形会
      因此整段消失。它没有证据是事实，记在「文档有、轨迹没见」那一栏里，不是失败。

    要求有证据的是**从轨迹编下来的那条路**：说「我见过」的才需要拿出见过的次数。
    """
    out: list[tuple[str, Transition]] = []
    for sid, t in m.transitions_all():
        if sid == m.fallback or t.to == m.fallback:
            continue
        if str(getattr(t, "origin", "") or "") == "document":
            continue
        if t.support < thr.min_support:
            out.append((sid, t))
    return out


def weak_edges(m: Machine, thresholds: Optional[Thresholds] = None) -> list[tuple[str, str]]:
    """支持度低于 ``min_support`` 的边，作 ``(状态 id, 条件)``（兜底边条件是空串）。"""
    thr = thresholds or m.thresholds
    return [(sid, t.cond) for sid, t in _support_rows(m, thr)]


def hot_judges(m: Machine, thresholds: Optional[Thresholds] = None) -> list[tuple[str, float]]:
    """标定误差率高于 ``judge_err_max`` 的判断动作，作 ``(状态 id, 误差率)``。

    「一条路径至少错一次的概率 ≤ Σεᵢ」这条不等式里，每个 εᵢ 都得有天花板，否则右边随便
    一个判断就能把上界顶穿。"""
    thr = thresholds or m.thresholds
    out: list[tuple[str, float]] = []
    for sid in sorted(m.states):
        act = m.states[sid].action
        if act.kind == "judge" and act.error_rate > thr.judge_err_max:
            out.append((sid, act.error_rate))
    return out


def _uncalibrated_judge_findings(m: Machine) -> list[Finding]:
    """**从文档引入**的判断动作没有程序金标 / 没标定过，就只能去 FALLBACK。

    只管 ``introduced=True`` 的判断：手写参考机里的判断是**靶子**，按定义 ``support=0``
    却带标签边（test_19 钉着这一点），它们不是编译器的产物，不受这条约束。编译器引入的
    判断则不同——``gold_from`` 空着或 ``support==0`` 说明**没有任何程序证据**说它判得准，
    让它分岔等于让一个未量误差的 εᵢ 进入「路径至少错一次的概率 ≤ Σεᵢ」的求和。

    * 带非 FALLBACK 出边 ⇒ ``E_JUDGE_UNCALIBRATED_BRANCH``（error，接口拒收）；
    * 只有去 FALLBACK 的边 ⇒ ``W_JUDGE_UNCALIBRATED``（warn，台账点名「文档要求在此判定，
      我们量不出」）。
    """
    out: list[Finding] = []
    for sid in sorted(m.states):
        act = m.states[sid].action
        if act.kind != "judge" or not act.introduced:
            continue
        uncal = (not act.gold_from) or act.support <= 0
        if not uncal:
            continue
        branching = [t for t in m.states[sid].transitions if t.to != m.fallback]
        why = ("没有登记程序打标器（gold_from 为空）" if not act.gold_from
               else f"support={act.support}，还没在任何轨迹快照上标定过")
        if branching:
            out.append(Finding(
                "E_JUDGE_UNCALIBRATED_BRANCH", "error", sid,
                f"引入的判断动作 {sid} {why}，却带 {len(branching)} 条非 FALLBACK 出边 "
                f"{[t.to for t in branching]}：未标定的判断只能是 FALLBACK 边界，"
                "不能是决策——先给它一个程序打标器并标定，或只留一条兜底边"))
        else:
            out.append(Finding(
                "W_JUDGE_UNCALIBRATED", "warn", sid,
                f"引入的判断动作 {sid} {why}：留在图里作 FALLBACK 边界，台账会点名"))
    return out


def _verified_ends(m: Machine) -> set[str]:
    """声称「已验证」地结束的状态（不含 FALLBACK 自己：解释执行不是机器的声明）。"""
    kinds = {t.id: t.kind for t in m.terminals}
    return {sid for sid, s in m.states.items()
            if s.action.kind == "end" and kinds.get(s.action.terminal) == "verified"
            and sid != m.fallback}


def _verified_terminal_findings(m: Machine) -> list[Finding]:
    """``verified`` 终点只能经**程序执行的审计工具**到达。

    实证来自 xlsx：让模型顶替审计工具，机器在两道题上的自审与金标**全部反相关**
    （金标 PASS 的自审 ERROR、金标 FAIL 的自审 PASS 且到了 END_VERIFIED）。「已验证」这个
    终点类别只有在验证者是程序时才有含金量，所以：

    * 没声明 ``audit_tools`` 的机器**不在此检查**：手写的玩具机器、单智能体旧路径的产物都
      没有这个声明，突然对它们发警告只会把每个「findings 为空」的断言变成噪声。声明缺失由
      :func:`audit_declaration_findings` 单独判，多智能体编译在 open_machine 时强制它；
    * 声明了 ``audit_tools`` 之后三条 error：
      - ``E_VERIFIED_UNAUDITED``：从 initial 出发**绕开所有审计工具状态**仍能走到 verified
        终点（test_19 对参考机的证法，变成对任何机器的规则）；
      - ``E_VERIFIED_NOT_VIA_STATUS``：审计工具状态直接通向 verified 终点（或其提交状态）
        的边，条件没读审计工具写出的任何变量——验证跑了但分岔没看它的结果；
      - ``E_P1_AUDIT_MISMATCH``：管 verified 终点的 ``require_before`` 禁止项，其 requires
        与 ``audit_tools`` 没有交集——评判侧与结构侧认的「验证」不是同一件事。
    """
    ends = _verified_ends(m)
    if not ends or not m.audit_tools:
        return []
    out: list[Finding] = []
    audit_states = {sid for sid, s in m.states.items()
                    if s.action.kind == "tool" and s.action.name in m.audit_tools}
    # ① 绕开审计工具状态的可达性
    seen, stack = set(), [m.initial]
    if m.initial not in audit_states:
        seen.add(m.initial)
    while stack:
        cur = stack.pop()
        if cur in audit_states:
            continue
        for t in m.out_edges(cur):
            if t.to in audit_states or t.to in seen:
                continue
            seen.add(t.to)
            stack.append(t.to)
    for sid in sorted(ends & seen):
        out.append(Finding(
            "E_VERIFIED_UNAUDITED", "error", sid,
            f"verified 终点 {sid} 不经过任何审计工具状态 {sorted(audit_states) or '（无）'} "
            f"（audit_tools={m.audit_tools}）也走得到：「已验证」的结束必须构造上绕不开程序审计"))
    # ② 审计状态 → (verified 终点 | 其提交状态) 的边必须读审计写出的变量
    feeders = set(ends)
    for src, t in m.transitions_all():
        if t.to in ends and src not in audit_states:
            feeders.add(src)
    for a in sorted(audit_states):
        writes = set(m.states[a].action.writes or [])
        for t in m.states[a].transitions:
            if t.to in feeders and t.to != m.fallback:
                used = _cond.vars_of(t.cond) if t.cond else set()
                if not (used & writes):
                    out.append(Finding(
                        "E_VERIFIED_NOT_VIA_STATUS", "error", a,
                        f"审计工具状态 {a} 通向 verified 路径的边 → {t.to}（条件 "
                        f"{t.cond or '（兜底）'}）没读它自己写出的变量 {sorted(writes)}："
                        "验证跑了、分岔却不看结果，等于没验证"))
    # ③ P1 与 audit_tools 要认同一批工具
    for p in m.prohibitions:
        pat = p.pattern if isinstance(p.pattern, dict) else {}
        if p.check != "require_before":
            continue
        only = pat.get("only_when") or {}
        if only.get("terminal_kind") != "verified":
            continue
        req = set(pat.get("requires") or [])
        if req and not (req & set(m.audit_tools)):
            out.append(Finding(
                "E_P1_AUDIT_MISMATCH", "error", "",
                f"禁止项 {p.id} 要求 verified 结束前先跑 {sorted(req)}，但机器的 audit_tools="
                f"{m.audit_tools} 与之无交集：评判侧与结构侧认的「验证」不是同一批工具"))
    return out


def audit_declaration_findings(m: Machine) -> list[Finding]:
    """有 verified 终点却没声明 ``audit_tools`` ⇒ ``E_AUDIT_TOOLS_MISSING``。

    不进 :func:`check_machine`（见 :func:`_verified_terminal_findings` 的说明）；多智能体
    编译的 orchestrator 在 open_machine 之后调它，缺声明就不开工——一台声称「已验证」却说
    不出验证者是谁的机器，编出来也是空头支票。
    """
    ends = _verified_ends(m)
    if not ends or m.audit_tools:
        return []
    return [Finding(
        "E_AUDIT_TOOLS_MISSING", "error", sorted(ends)[0],
        f"有 verified 终点 {sorted(ends)} 但机器没声明 audit_tools：「已验证」没有程序作证，"
        "是空头声明——在 open_machine 里声明审计工具的规范名")]


# --------------------------------------------------------------------------- #
# 一台机器的全部检查
# --------------------------------------------------------------------------- #
def check_machine(m: Machine, *, thresholds: Optional[Thresholds] = None) -> list[Finding]:
    """一台机器上能**不看轨迹**就判掉的全部问题。空列表 = 全过。

    = :func:`hexis.machine.checks.structural_findings` 的八条（一律 ``error``）
      + 「至多一条兜底边」（``error``）
      + 转移支持度、判断误差率两条门槛（``warn``——结构没坏，但过不了验收）。

    ``thresholds`` 不给就用机器自带的 ``m.thresholds``。
    """
    thr = thresholds or m.thresholds
    out: list[Finding] = []
    for msg in structural_findings(m):
        code, sid = classify(msg)
        out.append(Finding(code, "error", sid, msg))
    out += _multi_default_findings(m)
    out += _uncalibrated_judge_findings(m)
    out += _verified_terminal_findings(m)
    for sid, t in _support_rows(m, thr):
        out.append(Finding(
            "W_LOW_SUPPORT", "warn", sid,
            f"转移 {sid}→{t.to}（条件 {t.cond or '（兜底）'}）支持度 {t.support} "
            f"< 下限 {thr.min_support}：只凭这么少的轨迹就把它编译下来，是在把偶然当规律"))
    for sid, rate in hot_judges(m, thr):
        out.append(Finding(
            "W_JUDGE_ERR", "warn", sid,
            f"判断动作 {sid} 的标定误差率 {rate} > 上限 {thr.judge_err_max}："
            "改写提问、补样例重标定，或 demote_to_fallback 把这个分岔退回解释执行"))
    return out


def _reason_from(findings: Sequence[Finding], head: str) -> str:
    errs = [f for f in findings if f.severity == "error"]
    warns = [f for f in findings if f.severity == "warn"]
    body = "；".join(f.located() for f in errs) or "（无）"
    return f"{head}：{len(errs)} 处错误、{len(warns)} 处警告 —— {body}"


# --------------------------------------------------------------------------- #
# 小工具：把 dict / 模型统一成模型
# --------------------------------------------------------------------------- #
def _mk_transition(x: Any) -> Transition:
    if isinstance(x, Transition):
        return x.model_copy(deep=True)
    return Transition.model_validate(x)


def _mk_variable(x: Any) -> Variable:
    if isinstance(x, Variable):
        return x.model_copy(deep=True)
    return Variable.model_validate(x)


def _mk_terminal(x: Any) -> Terminal:
    if isinstance(x, Terminal):
        return x.model_copy(deep=True)
    return Terminal.model_validate(x)


def _mk_prohibition(x: Any) -> Prohibition:
    if isinstance(x, Prohibition):
        return x.model_copy(deep=True)
    return Prohibition.model_validate(x)


def _checks_back_edges(m: Machine):
    """机器上 DFS 意义的回边（转调 checks，口径与结构检查完全一致）。"""
    return _checks_back_edges_raw(m, _checks_reachable(m))


def _reachable(m: Machine) -> set[str]:
    seen, stack = {m.initial}, [m.initial]
    while stack:
        cur = stack.pop()
        for t in m.out_edges(cur):
            if t.to not in seen:
                seen.add(t.to)
                stack.append(t.to)
    return seen


# --------------------------------------------------------------------------- #
# 守门程序
# --------------------------------------------------------------------------- #
class Checker:
    """一台机器的受票改写台。八个接口，每个接口出一张 :class:`Receipt`。

    用法：:meth:`open_machine` 打开（空机器或接管一台已有的），然后用其余六个接口一条一条
    地改，最后 :meth:`commit` 跑验收并（给了 ``root`` 的话）落盘。任何一条改写都是
    **候选副本先行**：

    1. 深拷贝当前机器 → 候选；
    2. 在候选上做这条改写（前置条件不满足就直接拒，机器没动过）；
    3. :func:`check_machine` 跑候选，出现 ``error`` 就拒（机器仍然没动过）；
    4. 都过了，候选转正。

    ``doc`` 只作溯源留存（哪份技能文档编出来的），检查不读它。
    """

    def __init__(self, skill_id: str, doc: str = "", *,
                 thresholds: Optional[Thresholds] = None,
                 require_provenance: bool = False) -> None:
        self.skill_id = skill_id
        self.doc = doc
        self.thresholds = thresholds or Thresholds()
        self._thr_given = thresholds is not None
        self._machine: Optional[Machine] = None
        self._baseline: Optional[Machine] = None      # commit 的回滚点
        self._receipts: list[Receipt] = []
        #: 溯源表：prov_key → 溯源行（dict）。与机器同进退：回滚/回退一起恢复。
        self._prov: dict[str, dict] = {}
        self._baseline_prov: dict[str, dict] = {}
        #: 检查点：label → (机器深拷贝, 溯源表深拷贝, 当时的回执数, 当时的 commit 序号)。
        self._marks: dict[str, tuple[Machine, dict, int, int]] = {}
        self._commits = 0                             # 成功 commit 的次数（rewind 不能越过）
        #: 多智能体模式下打开：commit 时每个活着的结构都必须有溯源行，缺一条就拒。
        #: 单智能体旧路径默认关着——它的提议不带 prov，不该因此提交不了。
        self.require_provenance = bool(require_provenance)

    # ---- 观察面 ---- #
    @property
    def machine(self) -> Machine:
        """当前机器的**深拷贝**。

        故意不交出内部对象：交出去就等于给了一条绕过受票接口的就地修改通道，那样这一层
        写下的每张回执都不再能证明机器是怎么长成现在这样的。
        """
        if self._machine is None:
            raise ValueError("尚未 open_machine：先打开一台机器再取用")
        return self._machine.model_copy(deep=True)

    @property
    def opened(self) -> bool:
        return self._machine is not None

    def receipts(self) -> list[Receipt]:
        """完整审计痕迹，按发生顺序。回滚不擦它——回滚本身也是一条记录。"""
        return list(self._receipts)

    def provenance(self) -> dict[str, dict]:
        """溯源表的深拷贝：``prov_key`` → 行。行由各受票接口的 ``prov=`` 参数写入。"""
        return {k: dict(v) for k, v in self._prov.items()}

    def missing_provenance(self) -> list[str]:
        """当前机器里**没有**溯源行的活结构的键。多智能体模式下 commit 前必须为空。"""
        return [k for k in _live_keys(self._machine) if k not in self._prov] \
            if self._machine is not None else []

    # ---- 内部：候选副本 + 裁决 ---- #
    def _record(self, receipt: Receipt) -> Receipt:
        self._receipts.append(receipt)
        return receipt

    def _attempt(self, op: str, mutate: Callable[[Machine], str], detail: dict,
                 *, prov: Optional[dict] = None) -> Receipt:
        """在候选副本上做一次改写并裁决。``mutate`` 返回一句「改了什么」。

        ``mutate`` 可以往 ``detail["prov_keys"]`` 里放这次改写触及的溯源键；提议带了
        ``prov`` 且被接受时，这些键各记一行（同一份 ``prov``）。``prov`` 的 ``origin``
        必须在 :data:`ORIGINS` 里——写错分栏的溯源比没有溯源更糟。
        """
        if self._machine is None:
            return self._record(Receipt(
                op, False, "尚未 open_machine：先打开一台机器，再提改写", dict(detail)))
        prov = _norm_prov(prov)
        if prov is not None:
            bad = _prov_problem(prov)
            if bad:
                return self._record(Receipt(op, False, f"溯源行不合法：{bad}",
                                            {**detail, "prov": dict(prov)}))
        cand = self._machine.model_copy(deep=True)
        # 不在这里拷 detail：各接口的 mutate 闭包往**这同一个** dict 里放 prov_keys，
        # 拷了副本就读不到它们。回执里放的是 {**detail, ...} 的新 dict，原件不外泄。
        try:
            note = mutate(cand)
        except _Reject as rej:
            return self._record(Receipt(op, False, rej.reason,
                                        {**detail, **rej.detail}))
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            return self._record(Receipt(
                op, False, f"提议本身不合法（{type(exc).__name__}）：{exc}", dict(detail)))
        findings = check_machine(cand, thresholds=self.thresholds)
        errs = [f for f in findings if f.severity == "error"]
        fdicts = [vars(f) for f in findings]
        keys = list(detail.pop("prov_keys", []) or [])
        if errs:
            return self._record(Receipt(
                op, False, _reason_from(findings, "改完过不了结构检查，已丢弃候选"),
                {**detail, "findings": fdicts}))
        old, self._machine = self._machine, cand
        self._migrate_edge_prov(old, cand)
        if prov is not None:
            for k in keys:
                self._prov[k] = dict(prov)
        return self._record(Receipt(op, True, note,
                                    {**detail, "findings": fdicts,
                                     **({"prov_keys": keys} if prov is not None else {})}))

    @staticmethod
    def _edge_keys_by_pair(m: Optional[Machine]) -> dict:
        out: dict = {}
        if m is None:
            return out
        for sid, st in m.states.items():
            for t in st.transitions:
                out.setdefault((sid, t.to), []).append(prov_key("edge", sid, t.to, t.cond))
        return out

    def _migrate_edge_prov(self, old: Optional[Machine], new: Machine) -> None:
        """边的**条件被改写**时，把它的溯源行跟着搬过去。

        :func:`hexis.legacy.fit.install_counter` 收环时会把目标状态**已有的**每条条件出边都
        ``and`` 上 ``count < K``。那还是同一条边——同一个源、同一个去处、同一份支持度、同一
        条轨迹作证——但溯源键里含条件，于是旧键失效、新键无主。不搬的话 commit 会以
        ``E_PROVENANCE_MISSING`` 拒收一台完全合法的机器，然后整批回滚成空机器（实测：
        table_clean 上 ``s3→s4`` 与 ``s3→s6`` 两条边就是这么把整台机器带走的）。

        判据严格：同一个 ``(源, 去处)`` 下，旧键**已经不在图上**、新键**还没有行**，才搬。
        """
        before = self._edge_keys_by_pair(old)
        after = self._edge_keys_by_pair(new)
        live = {k for ks in after.values() for k in ks}
        for pair, new_keys in after.items():
            olds = [k for k in before.get(pair, ())
                    if k in self._prov and k not in live]
            for nk in new_keys:
                if nk in self._prov or not olds:
                    continue
                self._prov[nk] = self._prov.pop(olds.pop(0))

    # ---- ①  打开 ---- #
    def open_machine(self, *, base: Optional[Machine] = None,
                     variables: Sequence[Any] = (), terminals: Sequence[Any] = (),
                     prohibitions: Sequence[Any] = (),
                     max_steps: Optional[int] = None,
                     version: Optional[str] = None,
                     audit_tools: Sequence[str] = (),
                     prov: Optional[dict] = None) -> Receipt:
        """打开一台机器：``base`` 给了就接管它的深拷贝，不给就是一台全回退的空机器。

        空机器（``initial = FALLBACK``）是合法的地基：它复述任何轨迹、排除不了任何反例，
        对应「什么都还没学到，一切交给解释执行」。

        ``audit_tools`` 是程序执行的审计工具规范名（见 :class:`Machine.audit_tools`）；
        ``prov`` 给了就给这次声明的变量/终点/禁止项各记一行溯源。
        """
        detail = {"base": bool(base), "skill_id": self.skill_id}
        prov = _norm_prov(prov)
        if prov is not None:
            bad = _prov_problem(prov)
            if bad:
                return self._record(Receipt("open_machine", False, f"溯源行不合法：{bad}",
                                            {**detail, "prov": prov if isinstance(prov, dict)
                                             else str(prov)}))
        if self._machine is not None:
            return self._record(Receipt(
                "open_machine", False,
                "这台守门程序已经打开过机器了：一个 Checker 只管一台机器，"
                "换一台请另起一个 Checker", detail))
        if base is not None and base.skill_id != self.skill_id:
            return self._record(Receipt(
                "open_machine", False,
                f"base 的 skill_id 是 {base.skill_id!r}，与本守门程序的 "
                f"{self.skill_id!r} 对不上：机器与技能必须一一对应", detail))
        cand = base.model_copy(deep=True) if base is not None else empty_machine(self.skill_id)
        if self._thr_given:
            cand.thresholds = self.thresholds
        else:
            self.thresholds = cand.thresholds
        try:
            for v in variables:
                _upsert_variable(cand, _mk_variable(v))
            for t in terminals:
                _upsert_terminal(cand, _mk_terminal(t))
            for p in prohibitions:
                cand.prohibitions.append(_mk_prohibition(p))
            if max_steps is not None:
                cand.max_steps = int(max_steps)
            if version is not None:
                cand.version = str(version)
            if audit_tools:
                cand.audit_tools = sorted(set(cand.audit_tools) | {str(a) for a in audit_tools})
            _ensure_fallback(cand)
        except (ValidationError, ValueError, TypeError) as exc:
            return self._record(Receipt(
                "open_machine", False,
                f"打开参数不合法（{type(exc).__name__}）：{exc}", detail))
        findings = check_machine(cand, thresholds=self.thresholds)
        errs = [f for f in findings if f.severity == "error"]
        if errs:
            return self._record(Receipt(
                "open_machine", False,
                _reason_from(findings, "base 本身就过不了结构检查，拒绝接管"),
                {**detail, "findings": [vars(f) for f in findings]}))
        self._machine = cand
        self._baseline = cand.model_copy(deep=True)
        if prov is not None:
            for v in variables:
                self._prov[prov_key("var", _mk_variable(v).name)] = dict(prov)
            for t in terminals:
                self._prov[prov_key("terminal", _mk_terminal(t).id)] = dict(prov)
            for p in prohibitions:
                obj = _mk_prohibition(p)
                pat = obj.pattern if isinstance(obj.pattern, dict) else {}
                row = dict(prov)
                if pat.get("clause") or pat.get("quote"):
                    # 引了条款、带着原句的禁止项是**文档**来的，不是编译器自作主张的。
                    # 这一栏是台账「编译器自作主张清单」的分栏依据，混了就说不清了。
                    row.update({"origin": "document", "clause": str(pat.get("clause") or ""),
                                "locator": (f"{pat.get('source')}:{pat.get('line')}"
                                            if pat.get("source") and pat.get("line")
                                            else row.get("locator", "")),
                                "note": "禁止项引用文档原句（quote），由采集侧人工标出"})
                self._prov[prov_key("prohibition", obj.id)] = row
        self._baseline_prov = {k: dict(v) for k, v in self._prov.items()}
        return self._record(Receipt(
            "open_machine", True,
            f"打开机器 {cand.skill_id}（起点 {cand.initial}，"
            f"{len(cand.states)} 个状态）", {**detail,
                                        "findings": [vars(f) for f in findings]}))

    # ---- ②  加状态 ---- #
    def add_state(self, state_id: str, action: Any, *, clause: str = "",
                  transitions: Sequence[Any] = (), initial: bool = False,
                  from_state: Optional[str] = None, from_cond: str = "",
                  from_inc: Optional[str] = None, from_support: int = 0,
                  variables: Sequence[Any] = (),
                  origin: str = "", locator: str = "",
                  prov: Optional[dict] = None) -> Receipt:
        """加一个新状态，并**在同一次提议里**把它接进图。

        单独加一个状态必然不合法（要么走不到、要么没有出边），所以入边与出边是这个接口的
        一部分，而不是「回头再补一条」：

        * 出边不给 ``transitions`` 时默认一条通往 FALLBACK 的兜底边——新状态天然是「这一步
          学会了，下一步还没学会」。
        * 入边由 ``from_state`` (+ ``from_cond``) 给。**无条件入边接的是主干**：源状态已有
          的那条兜底边会被**改写**指向新状态（而不是多长出一条第二兜底边），新状态则继承了
          「继续往下还是回退」这件事。带条件的入边只是新增一条边，不动已有的边。
        """
        detail = {"state_id": state_id, "from_state": from_state,
                  "from_cond": from_cond, "initial": initial}

        def mutate(cand: Machine) -> str:
            if state_id in cand.states:
                raise _Reject(f"状态 {state_id!r} 已存在：换个 id，或用 add_transition / "
                              "demote_to_fallback 改它，别重复创建")
            _ensure_fallback(cand)
            outs = [_mk_transition(x) for x in transitions] or \
                [Transition(to=cand.fallback)]
            st = State(id=state_id, clause=clause, action=action, transitions=outs,
                       origin=origin, locator=locator)
            cand.states[state_id] = st
            for v in variables:
                _upsert_variable(cand, _mk_variable(v))
            _autodeclare(cand, st)
            if initial:
                cand.initial = state_id
            if from_state is not None:
                _attach(cand, from_state, state_id, from_cond, from_inc, from_support,
                        origin=origin)
            elif not initial and cand.initial != state_id:
                raise _Reject(
                    f"状态 {state_id!r} 没有入边也不是起点：给 from_state 把它接上，"
                    "或 initial=True 让它当起点，否则它是死代码")
            keys = [prov_key("state", state_id)]
            keys += [prov_key("edge", state_id, t.to, t.cond) for t in st.transitions]
            if from_state is not None:
                keys.append(prov_key("edge", from_state, state_id, from_cond))
            keys += [prov_key("var", _mk_variable(v).name) for v in variables]
            # 这个状态写出的变量（含 _autodeclare 自动补声明的）归它溯源
            keys += [prov_key("var", w) for w in (getattr(st.action, "writes", []) or [])]
            detail["prov_keys"] = keys
            return (f"加状态 {state_id}（{st.action.kind}，条款 {clause or '（未归属）'}），"
                    f"入边 {from_state or '（无，作起点）'}，出边 "
                    f"{[t.to for t in st.transitions]}")

        return self._attempt("add_state", mutate, detail, prov=prov)

    # ---- ③  加转移 ---- #
    def add_transition(self, from_state: str, to: str, *, cond: str = "",
                       inc: Optional[str] = None, support: int = 0,
                       origin: str = "", prov: Optional[dict] = None) -> Receipt:
        """在两个**已有**状态之间加一条边。只加边，不动任何已有的边。

        ``cond`` 为空是兜底边——源状态已经有兜底边时直接拒（一个状态只能有一条），这跟
        :meth:`add_state` 的「无条件入边改写主干」是刻意的不对称：加状态是在长主干，加边是
        在补分支，后者不该悄悄改掉已有的走向。
        """
        detail = {"from_state": from_state, "to": to, "cond": cond,
                  "inc": inc, "support": support,
                  "prov_keys": [prov_key("edge", from_state, to, cond)]}

        def mutate(cand: Machine) -> str:
            src = cand.states.get(from_state)
            if src is None:
                raise _Reject(f"没有状态 {from_state!r}：先 add_state 把它建出来，"
                              "再往它上面接边")
            if to not in cand.states:
                raise _Reject(f"边的目标 {to!r} 不存在：先 add_state 建出目标状态"
                              f"（现有状态：{sorted(cand.states)}）")
            if src.action.kind == "end":
                raise _Reject(f"状态 {from_state!r} 是终止态（terminal="
                              f"{src.action.terminal!r}），终止态不能再有出边")
            if cond:
                _check_cond(cand, from_state, cond)
                dup = [t for t in src.transitions if t.cond == cond]
                if dup:
                    raise _Reject(
                        f"状态 {from_state} 上已有一条条件完全相同的边 {cond!r} → "
                        f"{dup[0].to}：两条同条件的边必然重叠（违互斥），改条件或改目标")
            else:
                dflt = _default_edges(src)
                if dflt:
                    raise _Reject(
                        f"状态 {from_state} 已有兜底边 → {dflt[0].to}：一个状态只能有一条"
                        "兜底边，给这条新边加个条件，或先 demote_to_fallback 重排它的出边")
            if inc is not None:
                _require_counter(cand, inc)
            src.transitions.append(
                Transition(cond=cond, to=to, inc=inc, support=support, origin=origin))
            return (f"加转移 {from_state}→{to}"
                    f"（条件 {cond or '（兜底）'}，支持度 {support}"
                    f"{'，计数 ' + inc if inc else ''}）")

        return self._attempt("add_transition", mutate, detail, prov=prov)

    # ---- ④  收环 ---- #
    def close_loop(self, from_state: str, to: str, *, cond: str = "",
                   counter: Optional[str] = None, bound: Optional[int] = None,
                   support: int = 0, origin: str = "",
                   prov: Optional[dict] = None) -> Receipt:
        """成一个**有上限**的环：回边 + 计数变量 + 上限出口，三件一次做完。

        单独加一条回边一定过不了检查（``checks._loop_bounds``：回边没有 inc / 计数变量没有
        上限出口），所以这三件事必须是同一次提议。装计数这一步直接调
        :func:`hexis.legacy.fit.install_counter`——算法只该有一份：在环的**目标**状态上插一条
        ``counter >= K → FALLBACK`` 的出口，并把它原有的条件出边都 ``and`` 上
        ``counter < K``，保证「计满回退」与「继续绕」在任何格局下至多一条成立（定理2 的
        互斥）。上限 ``bound`` 不给时取 ``thresholds.retry_budget``。

        ``install_counter`` 按**目标**认那条回边，所以本接口拒绝「``from_state`` 上已经有
        一条通向 ``to`` 的边」——那时哪条才是要配计数的回边有歧义，守门程序不替调用方猜。
        同一个目标的第二条回边用 :meth:`add_transition` 带上已有的计数变量即可（上限出口
        已经装好了）。
        """
        detail = {"from_state": from_state, "to": to, "cond": cond,
                  "counter": counter, "bound": bound}

        def mutate(cand: Machine) -> str:
            src = cand.states.get(from_state)
            if src is None:
                raise _Reject(f"没有状态 {from_state!r}：回边的起点得先存在")
            if to not in cand.states:
                raise _Reject(f"没有状态 {to!r}：回边的目标得先存在")
            if src.action.kind == "end":
                raise _Reject(f"状态 {from_state!r} 是终止态，不能从它拉回边")
            if from_state not in _forward_closure(cand, to):
                raise _Reject(
                    f"{to} 走不到 {from_state}，{from_state}→{to} 不是回边（不成环）："
                    "不成环的边用 add_transition 加，别用 close_loop")
            dup_target = [t for t in src.transitions if t.to == to]
            if dup_target:
                raise _Reject(
                    f"状态 {from_state} 上已有一条通向 {to} 的边（条件 "
                    f"{dup_target[0].cond or '（兜底）'}）：一个目标的上限只设一次。"
                    f"再往同一个目标拉回边，用 add_transition 带上已有的计数变量 "
                    f"{dup_target[0].inc or (counter or to + '_count')!r}")
            if cond:
                _check_cond(cand, from_state, cond)
                if any(t.cond == cond for t in src.transitions):
                    raise _Reject(f"状态 {from_state} 上已有条件 {cond!r} 的边：会重叠")
            elif _default_edges(src):
                raise _Reject(
                    f"状态 {from_state} 已有兜底边 → {_default_edges(src)[0].to}："
                    "回边要么带条件，要么先 demote_to_fallback 腾出兜底位")
            cname = counter or f"{to}_count"
            K = int(bound) if bound is not None else int(self.thresholds.retry_budget)
            if K < 1:
                raise _Reject(f"上限 {K} < 1：环至少要允许绕一次，否则这条回边永远走不到")
            existing = cand.var(cname)
            if existing is not None and existing.type != "integer":
                raise _Reject(f"变量 {cname!r} 已存在且类型是 {existing.type}，"
                              "不能当计数变量（要 integer）")
            src.transitions.append(Transition(cond=cond, to=to, support=support,
                                              origin=origin))
            exit_edge = _fit.install_counter(cand, from_state, to, k=K, name=cname)
            detail["prov_keys"] = [prov_key("edge", from_state, to, cond),
                                   prov_key("var", cname),
                                   prov_key("edge", to, exit_edge.to, exit_edge.cond)]
            return (f"收环 {from_state}→{to}（条件 {cond or '（兜底）'}），计数变量 "
                    f"{cname} 上限 {K}，已在 {to} 装好上限出口 → {cand.fallback}")

        return self._attempt("close_loop", mutate, detail, prov=prov)

    # ---- ④''  一次收一组互相牵连的环 ---- #
    def close_loops(self, loops: Sequence[Any], *, prov: Optional[dict] = None) -> Receipt:
        """把**一组**回边连同它们的计数与上限一次收完，收完再检查一次。

        为什么不能一条一条收：``checks._back_edges`` 认的是 **DFS 意义上指向栈上祖先的边**，
        谁是回边取决于遍历顺序。一组互相牵连的环里，闭掉第一条会让第二条原本的前向边变成
        回边——那一瞬间它没有计数，结构检查当场判 ``E_LOOP_UNCOUNTED``，于是**每一条**单独提
        都过不去，而它们**合起来**是完全合法的一台机器。实测：s9→s5、s9→s6、s10→s6 三条
        逐条提全灭，理由分别指着 s10→s2 与 s11→s5 这两条没人动过的边。

        所以这是一次提议、一张回执：先把这一组边全加上、各自装好计数与上限，**再**把图上剩
        下的、补边之后才被 DFS 认出来的回边也一并配上上限（同 :meth:`bound_loop`），最后统
        一检查。收不住就整组丢弃，机器一个字节没动。

        ``loops`` 的元素是 dict：``{from_state, to, cond?, counter?, bound?, support?}``。
        """
        rows = [dict(x) for x in loops]
        detail = {"loops": rows, "n": len(rows)}

        def mutate(cand: Machine) -> str:
            if not rows:
                raise _Reject("空的环组：没什么可收的")
            keys: list[str] = []
            done: list[str] = []
            for row in rows:
                fs, to = str(row.get("from_state") or ""), str(row.get("to") or "")
                cond = str(row.get("cond") or "")
                src = cand.states.get(fs)
                if src is None:
                    raise _Reject(f"没有状态 {fs!r}：回边的起点得先存在")
                if to not in cand.states:
                    raise _Reject(f"没有状态 {to!r}：回边的目标得先存在")
                if src.action.kind == "end":
                    raise _Reject(f"状态 {fs!r} 是终止态，不能从它拉回边")
                if any(t.to == to for t in src.transitions):
                    raise _Reject(f"状态 {fs} 上已有一条通向 {to} 的边：一个目标的上限只设一次")
                if cond:
                    _check_cond(cand, fs, cond)
                    if any(t.cond == cond for t in src.transitions):
                        raise _Reject(f"状态 {fs} 上已有条件 {cond!r} 的边：会重叠")
                elif _default_edges(src):
                    raise _Reject(f"状态 {fs} 已有兜底边：回边要么带条件，要么先腾出兜底位")
                cname = str(row.get("counter") or f"{to}_count")
                K = int(row["bound"]) if row.get("bound") else int(self.thresholds.retry_budget)
                if K < 1:
                    raise _Reject(f"上限 {K} < 1：环至少要允许绕一次")
                ex = cand.var(cname)
                if ex is not None and ex.type != "integer":
                    raise _Reject(f"变量 {cname!r} 已存在且类型是 {ex.type}，不能当计数变量")
                src.transitions.append(Transition(cond=cond, to=to,
                                                  support=int(row.get("support") or 0),
                                                  origin=str(row.get("origin") or "")))
                exit_edge = _fit.install_counter(cand, fs, to, k=K, name=cname)
                keys += [prov_key("edge", fs, to, cond), prov_key("var", cname),
                         prov_key("edge", to, exit_edge.to, exit_edge.cond)]
                done.append(f"{fs}→{to}(K={K})")
            # 补边之后 DFS 才认出来的回边：一并配上限，否则这一组合起来仍然不合法
            extra: list[str] = []
            for _ in range(len(cand.states) + 4):
                todo = [(s, t) for s, t in _checks_back_edges(cand)
                        if not t.inc and t.to != cand.fallback]
                if not todo:
                    break
                s, t = todo[0]
                cname = f"{t.to}_count"
                ex = cand.var(cname)
                if ex is not None and ex.type != "integer":
                    raise _Reject(f"变量 {cname!r} 类型是 {ex.type}，不能当计数变量")
                exit_edge = _fit.install_counter(cand, s, t.to, k=int(self.thresholds.retry_budget),
                                                 name=cname)
                keys += [prov_key("var", cname),
                         prov_key("edge", t.to, exit_edge.to, exit_edge.cond)]
                extra.append(f"{s}→{t.to}")
            detail["prov_keys"] = keys
            return (f"一次收 {len(done)} 条环：{'、'.join(done)}"
                    + (f"；另给补边后才认出的回边配上限：{'、'.join(extra)}" if extra else ""))

        return self._attempt("close_loops", mutate, detail, prov=prov)

    # ---- ④'  给已有的回边配上限 ---- #
    def bound_loop(self, from_state: str, to: str, *, counter: Optional[str] = None,
                   bound: Optional[int] = None, prov: Optional[dict] = None) -> Receipt:
        """给一条**已经存在**的边配计数变量与上限出口——:meth:`close_loop` 少掉建边那一半。

        为什么需要它：``checks._back_edges`` 认的回边是 **DFS 意义上指向栈上祖先的边**，
        而这取决于遍历顺序；台账在转写期按「目标能不能走回来」标的那一批，只是其中一部分。
        实测过：闭掉 ``s9→s5`` 之后，原本是前向边的 ``s10→s2`` 变成了 DFS 回边，于是
        ``E_LOOP_UNCOUNTED`` 指着一条**没人再动过**的边报错——用 ``close_loop`` 补不了它
        （那条边已经在了，会被判「已有一条通向 X 的边」）。

        所以补边之后需要一次「按机器实际的回边补上限」的收尾，而它必须同样受票、同样出回执。
        算法与 close_loop 共用 :func:`hexis.legacy.fit.install_counter`：在目标上插
        ``counter >= K → FALLBACK``，并把目标原有的条件出边都 ``and`` 上 ``counter < K``。
        """
        detail = {"from_state": from_state, "to": to, "counter": counter, "bound": bound}

        def mutate(cand: Machine) -> str:
            src = cand.states.get(from_state)
            if src is None:
                raise _Reject(f"没有状态 {from_state!r}")
            if to not in cand.states:
                raise _Reject(f"没有状态 {to!r}")
            edges = [t for t in src.transitions if t.to == to]
            if not edges:
                raise _Reject(f"{from_state}→{to} 这条边不存在：要新建回边请用 close_loop")
            if len(edges) > 1:
                raise _Reject(f"{from_state} 上有 {len(edges)} 条通向 {to} 的边，"
                              "配计数会有歧义：守门程序不替调用方猜")
            if edges[0].inc:
                raise _Reject(f"{from_state}→{to} 已经配了计数变量 {edges[0].inc!r}")
            if from_state not in _forward_closure(cand, to):
                raise _Reject(f"{to} 走不到 {from_state}：这条边不成环，不需要上限")
            cname = counter or f"{to}_count"
            K = int(bound) if bound is not None else int(self.thresholds.retry_budget)
            if K < 1:
                raise _Reject(f"上限 {K} < 1：环至少要允许绕一次")
            existing = cand.var(cname)
            if existing is not None and existing.type != "integer":
                raise _Reject(f"变量 {cname!r} 已存在且类型是 {existing.type}，不能当计数变量")
            exit_edge = _fit.install_counter(cand, from_state, to, k=K, name=cname)
            detail["prov_keys"] = [prov_key("var", cname),
                                   prov_key("edge", to, exit_edge.to, exit_edge.cond)]
            return (f"给已有回边 {from_state}→{to} 配计数变量 {cname} 上限 {K}，"
                    f"已在 {to} 装好上限出口 → {cand.fallback}")

        return self._attempt("bound_loop", mutate, detail, prov=prov)

    # ---- ⑤  加判断动作 ---- #
    def add_judge(self, state_id: str, prompt: str, reads: Sequence[str],
                  writes: Sequence[str], labels: Sequence[str], *,
                  clause: str = "", abstain: str = "弃权",
                  examples: Sequence[Any] = (), error_rate: float = 0.0,
                  support: int = 0, transitions: Sequence[Any] = (),
                  from_state: Optional[str] = None, from_cond: str = "",
                  from_support: int = 0, variables: Sequence[Any] = (),
                  introduced: bool = False, gold_from: str = "",
                  origin: str = "", locator: str = "",
                  prov: Optional[dict] = None) -> Receipt:
        """装一个判断动作：编不成确定条件的语义判断落在这里，是机器里唯一的模型触点。

        ``introduced=True`` 表示它是编译器**从文档引入**的（轨迹里本没有这一步），
        ``gold_from`` 是给它打金标的程序打标器名——没有打标器或 ``support==0`` 的引入判断
        只能带一条去 FALLBACK 的边（:func:`_uncalibrated_judge_findings`）。

        ``state_id`` 已存在时**就地改写**它的动作（分岔学不出条件、于是把这一步改成判断，
        出边照旧）；不存在时新建一个状态，接法与 :meth:`add_state` 相同。

        ``error_rate`` 已经超过 ``judge_err_max`` 的判断当场拒收——明知它比允许的还吵还往
        机器里装，等于主动把「路径至少错一次的概率 ≤ Σεᵢ」这条不等式的上界顶穿。
        """
        detail = {"state_id": state_id, "reads": list(reads), "writes": list(writes),
                  "labels": list(labels), "error_rate": error_rate}

        def mutate(cand: Machine) -> str:
            if error_rate > self.thresholds.judge_err_max:
                raise _Reject(
                    f"判断动作 {state_id} 的误差率 {error_rate} > 上限 "
                    f"{self.thresholds.judge_err_max}：先改写提问 / 补样例重标定，"
                    "拿不准就 demote_to_fallback 把这个分岔退回解释执行")
            act = JudgeAction(prompt=prompt, reads=list(reads), writes=list(writes),
                              labels=list(labels), abstain=abstain,
                              examples=list(examples), error_rate=error_rate,
                              support=support, introduced=introduced,
                              gold_from=gold_from)
            _ensure_fallback(cand)
            for v in variables:
                _upsert_variable(cand, _mk_variable(v))
            st = cand.states.get(state_id)
            if st is not None:
                if st.action.kind == "end":
                    raise _Reject(f"状态 {state_id!r} 是终止态，不能改成判断动作")
                if transitions:
                    st.transitions = [_mk_transition(x) for x in transitions]
                st.action = act
                if clause:
                    st.clause = clause
                if origin:
                    st.origin = origin
                if locator:
                    st.locator = locator
                _autodeclare(cand, st)
                where = "就地改写"
            else:
                outs = [_mk_transition(x) for x in transitions] or \
                    [Transition(to=cand.fallback)]
                st = State(id=state_id, clause=clause, action=act, transitions=outs,
                           origin=origin, locator=locator)
                cand.states[state_id] = st
                _autodeclare(cand, st)
                if from_state is not None:
                    _attach(cand, from_state, state_id, from_cond, None, from_support,
                            origin=origin)
                elif cand.initial != state_id:
                    raise _Reject(f"新判断状态 {state_id!r} 没有入边：给 from_state 接上")
                where = "新建"
            keys = [prov_key("state", state_id), prov_key("judge", state_id)]
            keys += [prov_key("edge", state_id, t.to, t.cond) for t in st.transitions]
            if from_state is not None:
                keys.append(prov_key("edge", from_state, state_id, from_cond))
            keys += [prov_key("var", w) for w in writes]
            keys += [prov_key("var", _mk_variable(v).name) for v in variables]
            detail["prov_keys"] = keys
            return (f"{where}判断动作 {state_id}：问「{prompt}」，读 {list(reads)} → 写 "
                    f"{list(writes)}，标签 {list(labels)}（弃权 {abstain!r}），"
                    f"误差率 {error_rate}")

        return self._attempt("add_judge", mutate, detail, prov=prov)

    # ---- ⑥  定终止 ---- #
    def set_terminal(self, state_id: str, terminal: str, *, kind: str = "",
                     output: Sequence[str] = (), clause: str = "",
                     from_state: Optional[str] = None, from_cond: str = "",
                     from_support: int = 0, origin: str = "", locator: str = "",
                     prov: Optional[dict] = None) -> Receipt:
        """把一个状态定成终止态，并登记这种结束方式。

        ``kind`` 是这个终点的**类别**（``verified`` / ``unverified``），禁止项的 ``only_when``
        读的就是它——「提交前必须先跑核验」只该管声称已核验的那种结束。已存在的状态会被
        改成 ``end`` 动作并**清空出边**（终止态没有出边）。
        """
        detail = {"state_id": state_id, "terminal": terminal, "kind": kind,
                  "output": list(output)}

        def mutate(cand: Machine) -> str:
            _upsert_terminal(cand, Terminal(id=terminal, kind=kind,
                                            output=list(output)))
            st = cand.states.get(state_id)
            if st is not None:
                st.action = EndAction(terminal=terminal)
                st.transitions = []
                if clause:
                    st.clause = clause
                if origin:
                    st.origin = origin
                if locator:
                    st.locator = locator
                where = "就地改写"
            else:
                cand.states[state_id] = State(id=state_id, clause=clause,
                                              action=EndAction(terminal=terminal),
                                              origin=origin, locator=locator)
                if from_state is not None:
                    _attach(cand, from_state, state_id, from_cond, None, from_support,
                            origin=origin)
                elif cand.initial != state_id:
                    raise _Reject(f"新终止态 {state_id!r} 没有入边：给 from_state 接上，"
                                  "否则它是死代码")
                where = "新建"
            keys = [prov_key("state", state_id), prov_key("terminal", terminal)]
            if from_state is not None:
                keys.append(prov_key("edge", from_state, state_id, from_cond))
            detail["prov_keys"] = keys
            return (f"{where}终止态 {state_id} → 结束方式 {terminal}"
                    f"（类别 {kind or '（未表态）'}，输出 {list(output)}）")

        return self._attempt("set_terminal", mutate, detail, prov=prov)

    # ---- ⑦  退回兜底 ---- #
    def demote_to_fallback(self, state_id: str, *, note: str = "",
                           prov: Optional[dict] = None) -> Receipt:
        """把一个状态的出边全部换成一条通往 FALLBACK 的兜底边：**逃生口**。

        条件学不出来、判断太吵、环收不住——任何「这里我编不下去了」都可以退到这里。它在任何
        结构良好的机器上恒被接受：单条无条件出边不可能重叠、不可能有空隙，FALLBACK 是终止
        态所以停得下来，删条件只会让「先写后读」更松。因此它是智能体永远可用的那条退路。

        被它切下去的那一整段（只能经过这个状态到达的状态）会一并**删掉**——退回解释执行就
        是放弃那一段编译产物，把它们留着只会变成走不到的死代码。
        """
        detail = {"state_id": state_id, "note": note}

        def mutate(cand: Machine) -> str:
            st = cand.states.get(state_id)
            if st is None:
                raise _Reject(f"没有状态 {state_id!r}：无从退起"
                              f"（现有状态：{sorted(cand.states)}）")
            if st.action.kind == "end":
                raise _Reject(f"状态 {state_id!r} 是终止态，本来就没有出边可退；"
                              "要放弃它请改它的入边")
            _ensure_fallback(cand)
            if state_id == cand.fallback:
                raise _Reject("FALLBACK 自己就是兜底，退无可退")
            st.transitions = [Transition(to=cand.fallback, origin="compiler")]
            keep = _reachable(cand) | {cand.fallback, cand.initial}
            dropped = sorted(set(cand.states) - keep)
            for sid in dropped:
                del cand.states[sid]
            detail["prov_keys"] = [prov_key("edge", state_id, cand.fallback, "")]
            return (f"状态 {state_id} 退回解释执行（出边只剩 → {cand.fallback}）"
                    f"{'；' + note if note else ''}"
                    f"；连带删掉因此走不到的状态 {dropped or '（无）'}")

        return self._attempt("demote_to_fallback", mutate, detail,
                             prov=prov if prov is not None else
                             {"origin": "compiler", "agent_id": "checker",
                              "touchpoint_id": "", "note": note or "demote_to_fallback"})

    # ---- ⑧  提交 ---- #
    def commit(self, *, t_plus: Sequence[Any] = (), t_minus: Sequence[Any] = (),
               holdout: Sequence[Any] = (), root: Any = None) -> Receipt:
        """跑一遍验收，过了才落地。**这一步是全有全无的。**

        与前七个接口相反：单条改写被拒不牵连别人，但 commit 一旦不过，自上次 commit
        （或 open_machine）以来的**全部**改写一起回滚——一批改写要么整体成为新的地基，要么
        整体不算数，中间态不该被别人读到。回执照留：回滚也是审计痕迹的一部分。

        ``root`` 给了且验收通过时把 machine.json 写进去。**这是全仓库唯一写机器文件的地
        方**——机器文件只能是一次通过验收的提交的产物。
        """
        detail: dict = {"t_plus": len(t_plus), "t_minus": len(t_minus),
                        "holdout": len(holdout), "root": str(root) if root else ""}
        if self._machine is None:
            return self._record(Receipt("commit", False, "尚未 open_machine：没有可提交的机器",
                                        detail))
        from hexis.legacy import verify as _verify         # 延迟导入：verify 反过来要用本模块的 Finding

        rep = _verify.verify_machine(self._machine, t_plus, t_minus,
                                     thresholds=self.thresholds, holdout=holdout)
        detail = {**detail, "report": _verify.report_dict(rep),
                  "findings": [vars(f) for f in rep.findings]}
        missing = self.missing_provenance() if self.require_provenance else []
        if not rep.ok or missing:
            self._machine = self._baseline.model_copy(deep=True)
            self._prov = {k: dict(v) for k, v in self._baseline_prov.items()}
            why = ("验收未过" if not rep.ok else
                   f"[E_PROVENANCE_MISSING] {len(missing)} 个活结构没有溯源行 "
                   f"{missing[:8]}{'…' if len(missing) > 8 else ''}")
            return self._record(Receipt(
                "commit", False,
                f"{why}，整批已回滚到上一次提交时的机器：" + _verify.summary(rep),
                {**detail, "rolled_back": True, "missing_provenance": missing}))
        self._baseline = self._machine.model_copy(deep=True)
        self._baseline_prov = {k: dict(v) for k, v in self._prov.items()}
        self._commits += 1
        path = ""
        if root is not None:
            path = str(save_machine(self._machine, root))
            if self._prov:
                Path(root).mkdir(parents=True, exist_ok=True)
                (Path(root) / PROVENANCE_FILE).write_text(
                    json.dumps(self._prov, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
        return self._record(Receipt(
            "commit", True,
            "验收通过：" + _verify.summary(rep) + (f"；已写入 {path}" if path else ""),
            {**detail, "path": path}))

    # ---- ⑨  按前驱分裂 ---- #
    def split_state(self, state_id: str, groups: dict, *, outs: Optional[dict] = None,
                    prov: Optional[dict] = None) -> Receipt:
        """把一个状态按**前驱分组**克隆成几份（Myhill-Nerode：这几段历史其实不等价）。

        ``groups = {clone_id: [前驱状态 id, ...]}``：``state_id`` 的每个前驱必须恰好落在
        一组里；克隆 id 必须是新的（由 harness 分配，不由智能体取名）；每份克隆继承原状态
        的动作与条款，出边取 ``outs[clone_id]``（``[{"if","to"}]`` 的子集，按 (cond,to) 在
        原状态的出边里找）——不给就整份继承。原状态删掉。

        拒绝：分起点（起点没有前驱可分）；原状态有自环或带计数的回边指向它（计数上限装在
        目标上，分裂会让上限出口错位——先拆环再分再收）。回放对分裂**不敏感**：克隆的动作
        是深拷贝，``replay._action_matches`` 只比动作不比身份（test_08 证过这一点）。
        """
        detail = {"state_id": state_id, "groups": {k: list(v) for k, v in groups.items()},
                  "outs": {k: list(v) for k, v in (outs or {}).items()}}

        def mutate(cand: Machine) -> str:
            st = cand.states.get(state_id)
            if st is None:
                raise _Reject(f"没有状态 {state_id!r}：无从分起")
            if st.action.kind == "end":
                raise _Reject(f"状态 {state_id!r} 是终止态：终止态没有出边可分")
            if state_id == cand.initial:
                raise _Reject(f"状态 {state_id!r} 是起点：起点没有前驱，按前驱分不了")
            if state_id == cand.fallback:
                raise _Reject("FALLBACK 不能分裂")
            preds = [(src, t) for src, t in cand.transitions_all() if t.to == state_id]
            if any(src == state_id for src, _t in preds):
                raise _Reject(f"状态 {state_id!r} 有自环：先 demote 拆环，再分，再 close_loop")
            if any(t.inc for _s, t in preds):
                raise _Reject(f"状态 {state_id!r} 是带计数回边的目标：分裂会让上限出口错位，"
                              "先拆环再分")
            pred_ids = sorted({src for src, _t in preds})
            if len(groups) < 2:
                raise _Reject("至少要分成两组，否则不叫分裂")
            assigned: dict[str, str] = {}
            for cid, members in groups.items():
                if cid in cand.states:
                    raise _Reject(f"克隆 id {cid!r} 已存在：克隆 id 必须是新分配的")
                for p in members:
                    if p not in pred_ids:
                        raise _Reject(f"{p!r} 不是 {state_id} 的前驱（前驱：{pred_ids}）")
                    if p in assigned:
                        raise _Reject(f"前驱 {p!r} 同时在 {assigned[p]} 与 {cid} 两组里")
                    assigned[p] = cid
            left = [p for p in pred_ids if p not in assigned]
            if left:
                raise _Reject(f"前驱 {left} 没有分到任何组：每个前驱都得有去处")
            keys: list[str] = []
            for cid, members in groups.items():
                if outs and cid in outs:
                    picked = []
                    for spec in outs[cid]:
                        t = _mk_transition(spec)
                        match = [o for o in st.transitions if o.cond == t.cond and o.to == t.to]
                        if not match:
                            raise _Reject(f"克隆 {cid} 要的出边 {t.cond!r}→{t.to} 不在 "
                                          f"{state_id} 的出边里")
                        picked.append(match[0].model_copy(deep=True))
                    if not any(not o.cond for o in picked):
                        picked.append(Transition(to=cand.fallback, origin="compiler"))
                else:
                    picked = [o.model_copy(deep=True) for o in st.transitions]
                clone = State(id=cid, clause=st.clause, action=st.action.model_copy(deep=True),
                              transitions=picked, origin=st.origin or "trace",
                              locator=st.locator)
                cand.states[cid] = clone
                keys.append(prov_key("state", cid))
                keys += [prov_key("edge", cid, o.to, o.cond) for o in picked]
                for src, t in preds:
                    if src in members:
                        t.to = cid
                        keys.append(prov_key("edge", src, cid, t.cond))
            del cand.states[state_id]
            detail["prov_keys"] = keys
            return (f"状态 {state_id} 按前驱分裂为 {sorted(groups)}："
                    + "；".join(f"{cid} ← {sorted(m)}" for cid, m in groups.items()))

        return self._attempt("split_state", mutate, detail, prov=prov)

    # ---- ⑩ ⑪  检查点与回退 ---- #
    def mark(self, label: str) -> Receipt:
        """在当前机器上打一个检查点。受票、出回执，但不改机器。"""
        detail = {"label": label}
        if self._machine is None:
            return self._record(Receipt("mark", False, "尚未 open_machine", detail))
        self._marks[label] = (self._machine.model_copy(deep=True),
                              {k: dict(v) for k, v in self._prov.items()},
                              len(self._receipts), self._commits)
        return self._record(Receipt(
            "mark", True, f"检查点 {label!r}：{len(self._machine.states)} 个状态，"
            f"{len(self._prov)} 行溯源，回执 #{len(self._receipts)}", detail))

    def rewind(self, label: str) -> Receipt:
        """回到一个检查点：机器与溯源表一起恢复，**回执不擦**（回退本身也是一条记录）。

        不能越过上一次成功的 commit：提交已经落盘，往前回退等于让磁盘上的机器与守门程序
        手里的机器分家。
        """
        detail = {"label": label}
        if self._machine is None:
            return self._record(Receipt("rewind", False, "尚未 open_machine", detail))
        got = self._marks.get(label)
        if got is None:
            return self._record(Receipt(
                "rewind", False, f"没有检查点 {label!r}（现有：{sorted(self._marks)}）", detail))
        m, prov, n_receipts, n_commits = got
        if n_commits < self._commits:
            return self._record(Receipt(
                "rewind", False,
                f"[E_REWIND_PAST_COMMIT] 检查点 {label!r} 早于上一次 commit：提交已落盘，"
                "不能回退到它之前", detail))
        self._machine = m.model_copy(deep=True)
        self._prov = {k: dict(v) for k, v in prov.items()}
        return self._record(Receipt(
            "rewind", True,
            f"回到检查点 {label!r}（回执 #{n_receipts} 时的机器，{len(m.states)} 个状态）",
            {**detail, "receipts_then": n_receipts, "receipts_now": len(self._receipts)}))

    # ---- 提议分派 ---- #
    def apply(self, proposal: Proposal) -> Receipt:
        """按 ``proposal.op`` 分派到对应的受票接口。未知操作 / 参数不对都出拒绝回执。"""
        op = proposal.op
        if op not in OPS:
            return self._record(Receipt(
                op, False, f"未知操作 {op!r}：只认 {list(OPS)}", dict(proposal.args)))
        meth = getattr(self, op)
        try:
            return meth(**dict(proposal.args))
        except TypeError as exc:                       # 参数名/个数不对
            return self._record(Receipt(
                op, False, f"参数不对：{exc}", dict(proposal.args)))


# --------------------------------------------------------------------------- #
# 批量：一串提议，逐条裁决
# --------------------------------------------------------------------------- #
def batch_check(base: Machine, proposals: Sequence[Proposal]) -> tuple[Machine, list[Receipt]]:
    """在 ``base`` 上逐条裁决一串提议，返回 ``(改完的机器, 每条提议一张回执)``。

    **被拒的提议不牵连它之前被接受的提议**——这是本模块存在的理由，也是它与
    :func:`hexis.legacy.compiler.compile_round`（整轮撤销）唯一但要命的区别。``base`` 本身
    永不被就地修改。

    回执与提议一一对应；唯一的例外是 ``base`` 连接管都过不了检查——那时返回
    ``(base 的副本, [那张 open_machine 的拒绝回执])``。
    """
    ck = Checker(base.skill_id, thresholds=base.thresholds)
    opened = ck.open_machine(base=base)
    if not opened.accepted:
        return base.model_copy(deep=True), [opened]
    out = [ck.apply(p) for p in proposals]
    return ck.machine, out


# --------------------------------------------------------------------------- #
# 改写的原子操作（都在候选副本上做）
# --------------------------------------------------------------------------- #
def _ensure_fallback(m: Machine) -> None:
    """保证 FALLBACK 状态存在。它是保留状态：进去 = 放弃编译路径，改回解释执行。"""
    if m.fallback in m.states:
        return
    term = m.terminals[0].id if m.terminals else "done"
    if not m.terminals:
        m.terminals.append(Terminal(id=term))
    m.states[m.fallback] = State(id=m.fallback, action=EndAction(terminal=term))


def _attach(m: Machine, src_id: str, dst_id: str, cond: str,
            inc: Optional[str], support: int, *, origin: str = "") -> None:
    """把 ``dst`` 接到 ``src`` 后面。无条件入边**改写**源状态已有的兜底边（接主干）。

    只改写**指向 FALLBACK** 的兜底边（「下一步还没学会」的那个空位）。兜底边已经指向
    某个真实状态时拒绝——那是别人接好的主干，再无条件挂一个上去就是静默抢主干
    （``E_SPINE_TAKEN``）：单智能体时它是一个悄悄改走向的 bug，多智能体时它是两个智能体
    的提案互相覆盖。要挂第二个后继，给它一个条件。
    """
    src = m.states.get(src_id)
    if src is None:
        raise _Reject(f"没有状态 {src_id!r}：入边的源得先存在"
                      f"（现有状态：{sorted(m.states)}）")
    if src.action.kind == "end":
        raise _Reject(f"状态 {src_id!r} 是终止态，不能从它接出边")
    if inc is not None:
        _require_counter(m, inc)
    if cond:
        _check_cond(m, src_id, cond)
        if any(t.cond == cond for t in src.transitions):
            raise _Reject(f"状态 {src_id} 上已有条件 {cond!r} 的边：两条同条件的边必然重叠")
        src.transitions.append(Transition(cond=cond, to=dst_id, inc=inc, support=support,
                                          origin=origin))
        return
    dflt = _default_edges(src)
    if dflt:                                   # 接主干：改写原来的兜底去向
        if dflt[0].to != m.fallback and dflt[0].to != dst_id:
            raise _Reject(
                f"[E_SPINE_TAKEN] 状态 {src_id} 的兜底边已经指向 {dflt[0].to}，不是空位："
                f"再无条件挂 {dst_id} 上去等于抢走已接好的主干——给这条入边一个条件，"
                "或先 demote_to_fallback 腾出兜底位",
                {"code": "E_SPINE_TAKEN", "state_id": src_id, "taken_by": dflt[0].to})
        dflt[0].to = dst_id
        dflt[0].inc = inc
        dflt[0].support = support
        if origin:
            dflt[0].origin = origin
    else:
        src.transitions.append(Transition(to=dst_id, inc=inc, support=support,
                                          origin=origin))


def _upsert_variable(m: Machine, v: Variable) -> None:
    old = m.var(v.name)
    if old is None:
        m.variables.append(v)
        return
    if old.type != v.type:
        raise _Reject(f"变量 {v.name!r} 已声明为 {old.type}，与新声明的 {v.type} 冲突："
                      "同名变量的类型必须一致")
    old.init, old.init_from = v.init, v.init_from


def _upsert_terminal(m: Machine, t: Terminal) -> None:
    old = next((x for x in m.terminals if x.id == t.id), None)
    if old is None:
        m.terminals.append(t)
        return
    if t.kind:
        old.kind = t.kind
    if t.output:
        old.output = list(t.output)


def _autodeclare(m: Machine, st: State) -> None:
    """状态写入却没声明过的变量自动补进变量表（无 init：它由这个状态产出）。

    类型给不出更好的猜测时落 ``string``；要 ``array``/``integer`` 这类精确类型（或要
    ``init_from``），在 ``variables=`` 里显式声明，:func:`_upsert_variable` 会以显式声明为准。
    """
    for w in getattr(st.action, "writes", []) or []:
        if m.var(w) is None:
            m.variables.append(Variable(name=w, type="string"))


def _require_counter(m: Machine, name: str) -> None:
    v = m.var(name)
    if v is None:
        raise _Reject(f"计数变量 {name!r} 没有声明：用 close_loop 收环（它会连计数变量与"
                      "上限出口一起装好），别手工加 inc")
    if v.type != "integer":
        raise _Reject(f"计数变量 {name!r} 的类型是 {v.type}，不是 integer")


def _check_cond(m: Machine, sid: str, expr: str) -> None:
    """条件必须能解析（白名单 AST），且只用已声明的变量。"""
    try:
        used = _cond.vars_of(expr)
    except _cond.CondError as exc:
        raise _Reject(f"状态 {sid} 上的条件 {expr!r} 不合法：{exc}") from exc
    unknown = sorted(v for v in used if m.var(v) is None)
    if unknown:
        raise _Reject(
            f"状态 {sid} 上的条件 {expr!r} 用到未声明的变量 {unknown}："
            "先让某个状态把它写出来（或在 open_machine/add_state 的 variables 里声明）")


# --------------------------------------------------------------------------- #
# 溯源：活结构的键、溯源行的合法性
# --------------------------------------------------------------------------- #
def _live_keys(m: Machine) -> list[str]:
    """一台机器里**每个该有溯源行的结构**的键：状态、边、判断、变量、终点、禁止项。

    FALLBACK 状态与它的终止方式是保留结构，不要求溯源；去 FALLBACK 的兜底边要（它是
    「这里我没学会」的记录，谁在何时决定放弃是要能查的）。
    """
    keys: list[str] = []
    for sid in sorted(m.states):
        st = m.states[sid]
        if sid == m.fallback:
            continue
        keys.append(prov_key("state", sid))
        if st.action.kind == "judge":
            keys.append(prov_key("judge", sid))
        for t in st.transitions:
            keys.append(prov_key("edge", sid, t.to, t.cond))
    keys += [prov_key("var", v.name) for v in m.variables]
    fb_term = m.states[m.fallback].action.terminal if m.fallback in m.states \
        and m.states[m.fallback].action.kind == "end" else None
    keys += [prov_key("terminal", t.id) for t in m.terminals if t.id != fb_term]
    keys += [prov_key("prohibition", p.id) for p in m.prohibitions]
    return keys


def _norm_prov(prov: Any) -> Any:
    """接受 dict 或带 ``to_dict()`` 的对象（batch.Provenance）；其余原样交给 _prov_problem 去拒。"""
    if prov is not None and not isinstance(prov, dict) and hasattr(prov, "to_dict"):
        return prov.to_dict()
    return prov


def _prov_problem(prov: Any) -> str:
    """溯源行不合法的原因；合法返回空串。只查形状与分栏，不查内容真伪。"""
    if not isinstance(prov, dict):
        return f"要一个 dict，给的是 {type(prov).__name__}"
    origin = prov.get("origin")
    if origin not in ORIGINS:
        return f"origin={origin!r} 不在 {list(ORIGINS)} 里"
    if not str(prov.get("agent_id") or ""):
        return "缺 agent_id（谁提的这条）"
    return ""


# --------------------------------------------------------------------------- #
def _forward_closure(m: Machine, start: str) -> set[str]:
    """从 ``start`` 顺着边能到达的全部状态（含自身）。供「这条边是不是回边」判定。"""
    seen, stack = {start}, [start]
    while stack:
        cur = stack.pop()
        for t in m.out_edges(cur):
            if t.to not in seen:
                seen.add(t.to)
                stack.append(t.to)
    return seen


__all__ = [
    "Checker", "Finding", "OPS", "ORIGINS", "PROVENANCE_FILE", "Proposal", "Receipt",
    "audit_declaration_findings", "batch_check", "check_machine", "classify",
    "hot_judges", "prov_key", "weak_edges",
]
