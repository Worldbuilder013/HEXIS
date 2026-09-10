"""确定性守门程序（下）：一台**做完了的**机器的验收门槛。**本模块不调模型。**

没有一个 ``model`` 参数，不 import 任何模型客户端。验收全部是可重放的确定性核对：拿轨迹
驱动机器走一遍（:mod:`skill2fsm.replay`，定理2 保证给定结果序列时状态与变量序列是它的
函数），再对机器自己的账面数字设门槛。

四道门，前两道是这一层新加的——**仓库里此前没有任何地方检查它们**：

1. **每条转移的支持度 ≥ ``min_support``。** 只被一条轨迹走过的边不是规律，是巧合；把它
   编进机器，等于把一次偶然的执行顺序固化成技能的一部分。（通往 FALLBACK 的边豁免：那是
   「这里我没学会」，按定义没有轨迹支持。）
2. **每个判断动作的标定误差率 ≤ ``judge_err_max``。** 「一条路径至少错一次的概率 ≤ Σεᵢ」
   这条不等式里每个 εᵢ 都要有天花板，否则一个太吵的判断就能把上界顶穿。
3. **每条接受轨迹（T+）都被复述**（``replay.reproduces``）。
4. **每条拒绝轨迹（T-）都被排除**（``replay.excludes``）——但只对**落在已编译区段**里的
   反例这样要求。

第 4 条那个例外是这份验收里最容易做错的地方，单独说明：一条反例的 ``error_step`` 若落在
机器已经进了 FALLBACK 之后的那一段，机器在那里根本没有结构可言（解释模式不承诺具体路径，
:func:`skill2fsm.replay.replay` 走到 FALLBACK 就无条件接受剩余），于是它**不可能**偏离，
也就**排除不了**。这不是失败，是「这一段还没编译到」——把它算成失败，会逼着编译过程去
「修」一个不存在的缺陷，或者更糟：为了让数字好看而把 FALLBACK 砍掉。所以这类反例被单独
报成 :attr:`VerifyReport.fallback_deferred`（一条 ``W_FALLBACK_DEFERRED`` 警告），不计入
失败，也不算已排除。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import conformance as _conformance
from . import replay as _replay
from .checker import Finding, check_machine, hot_judges, weak_edges
from .runtime import pick_edge
from .schema import Machine, Thresholds, Trace


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VerifyReport:
    """一次验收的全部账面。``ok`` 是四道门的合取。

    ``total_negative`` 是反例**总数**（不是「该被排除的数」）；因此负例那道门的形式是
    ``excluded + len(fallback_deferred) == total_negative``——落在回退段里的反例既不算已
    排除、也不算失败。

    ``holdout_acc`` 只在给了 ``holdout`` 时有值（留出集里被复述的比例），有值时按
    ``thresholds.acc_thr`` 一起判。
    """

    reproduced: int
    total_positive: int
    excluded: int
    total_negative: int
    judge_err_ok: bool
    min_support_ok: bool
    findings: list[Finding]
    holdout_acc: Optional[float]
    weak_edges: list[tuple[str, str]]
    hot_judges: list[tuple[str, float]]
    ok: bool
    #: 复述得出、但真跑不等同的接受轨迹下标（见 skill2fsm.conformance）。空 = 全部等同。
    nonconforming: list = field(default_factory=list)
    #: 出错位置落在 FALLBACK 段里、**尚不可排除**的反例下标（不计失败）
    fallback_deferred: list[int] = field(default_factory=list)
    #: 复述失败的接受轨迹下标
    unreproduced: list[int] = field(default_factory=list)
    #: 落在已编译区段却没被排除的反例下标（真失败）
    unexcluded: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
def fallback_entry(machine: Machine, trace: Trace) -> Optional[int]:
    """机器沿这条轨迹走到第几步（``Record.step``）进了 FALLBACK。没进过 → ``None``。

    驱动逻辑与 :func:`skill2fsm.replay.replay` 一致（用轨迹记录的 output 按 writes 白名单
    推变量、回边自己 inc、``pick_edge`` 选边），区别只在于 replay 走到 FALLBACK 就返回
    「接受」，这里要的是它**在哪一步**进去的——「这条反例还没到能排除的时候」正是靠这个
    位置与 ``error_step`` 比出来的。

    机器在中途就与轨迹对不上（结构上已经偏离）时返回 ``None``：那已经是排除逻辑的地盘，
    与回退段无关。
    """
    return _replay.walk(machine, trace).fallback_at


def _cutoff(neg: Trace) -> int:
    if neg.error_step is not None:
        return neg.error_step
    return neg.records[-1].step if neg.records else 0


def _tid(trace: Trace) -> str:
    task = trace.task if isinstance(trace.task, dict) else {}
    return str(task.get("task_id", "") or "（无 task_id）")


# --------------------------------------------------------------------------- #
def verify_machine(m: Machine, t_plus: Sequence[Trace] = (),
                   t_minus: Sequence[Trace] = (), *,
                   thresholds: Optional[Thresholds] = None,
                   holdout: Sequence[Trace] = ()) -> VerifyReport:
    """跑完四道门，出一份 :class:`VerifyReport`。不调模型、不写文件。"""
    thr = thresholds or m.thresholds
    findings: list[Finding] = list(check_machine(m, thresholds=thr))
    weak = weak_edges(m, thr)
    hot = hot_judges(m, thr)

    reproduced, unrep = 0, []
    for i, t in enumerate(t_plus):
        r = _replay.replay(m, t)
        if r.ok:
            reproduced += 1
            continue
        unrep.append(i)
        findings.append(Finding(
            "E_NOT_REPRODUCED", "error", "",
            f"第 {i} 条接受轨迹（{_tid(t)}）复述不出来："
            f"{r.reason or '未给出原因'}（第 {r.diverged_at} 条记录处偏离）——"
            "机器学到的路径与真实执行不一致"))

    excluded, deferred, unexc = 0, [], []
    for i, neg in enumerate(t_minus):
        if _replay.excludes(m, neg):
            excluded += 1
            continue
        entry = fallback_entry(m, neg)
        cut = _cutoff(neg)
        if entry is not None and cut >= entry:
            deferred.append(i)
            findings.append(Finding(
                "W_FALLBACK_DEFERRED", "warn", m.fallback,
                f"第 {i} 条拒绝轨迹（{_tid(neg)}）的出错位置 step={cut} 落在 FALLBACK 段里"
                f"（机器第 {entry} 步就进了解释执行）：这一段还没编译到，机器在那里没有结构"
                "可偏离，**尚不可排除**——这是预期之内的，不算失败；要排除它，先把这一段"
                "编译出来"))
            continue
        unexc.append(i)
        findings.append(Finding(
            "E_NOT_EXCLUDED", "error", "",
            f"第 {i} 条拒绝轨迹（{_tid(neg)}）的出错位置 step={cut} 落在已编译区段里，"
            "机器却照样把它走完了：状态合并过头，或缺一条禁止项——正例只能证实不能证伪，"
            "这里就是 T- 该起作用的地方"))

    # ---- 第五道门：**执行级一致性** ---- #
    # 前面那道「复述」是推演：沿轨迹推机器、逐步比动作身份，看不见运行时那一层
    # （``${var}`` 谁来填、``writes`` 白名单收不收得住、``phase`` 两侧记不记、条件在真实变量
    # 上怎么求值）。实测的三个洞全长在那一层而复述全绿。所以交付前再跑一遍**真的**
    # ``run_task``，模型与工具都由轨迹驱动，产出的动作序列必须与轨迹逐步相同、工具入参逐字
    # 相同。见 :mod:`skill2fsm.conformance`。
    nonconforming: list[int] = []
    for i, t in enumerate(t_plus):
        if i in unrep:
            continue                      # 复述都没过，一致性不再重复报一遍
        cr = _conformance.check_trace(m, t)
        if cr.ok:
            continue
        nonconforming.append(i)
        findings.append(Finding(
            "E_NOT_CONFORMANT", "error", "",
            f"第 {i} 条接受轨迹（{_tid(t)}）复述得出、**真跑却不等同**："
            f"{cr.divergences[0].located() if cr.divergences else cr.error}——"
            "机器交付出去执行的东西与它凭以学出来的轨迹不是同一串动作"))

    holdout_acc: Optional[float] = None
    if holdout:
        holdout_acc = round(
            sum(1 for t in holdout if _replay.reproduces(m, t)) / len(holdout), 4)

    errs = [f for f in findings if f.severity == "error"]
    ok = (not errs
          and not weak
          and not hot
          and reproduced == len(t_plus)
          and excluded + len(deferred) == len(t_minus)
          and (holdout_acc is None or holdout_acc >= thr.acc_thr))
    return VerifyReport(
        reproduced=reproduced, total_positive=len(t_plus),
        excluded=excluded, total_negative=len(t_minus),
        judge_err_ok=not hot, min_support_ok=not weak,
        findings=findings, holdout_acc=holdout_acc,
        weak_edges=weak, hot_judges=hot, ok=ok,
        fallback_deferred=deferred, unreproduced=unrep, unexcluded=unexc,
        nonconforming=nonconforming)


def batch_check(m: Machine, t_plus: Sequence[Trace] = (),
                t_minus: Sequence[Trace] = (), *,
                thresholds: Optional[Thresholds] = None) -> list[Finding]:
    """只要 findings 的那个口子。等价于 ``verify_machine(...).findings``。"""
    return verify_machine(m, t_plus, t_minus, thresholds=thresholds).findings


# --------------------------------------------------------------------------- #
def report_dict(rep: VerifyReport) -> dict:
    """机读的报告字典（进回执的 ``detail``）。"""
    return {
        "ok": rep.ok,
        "reproduced": rep.reproduced, "total_positive": rep.total_positive,
        "excluded": rep.excluded, "total_negative": rep.total_negative,
        "fallback_deferred": list(rep.fallback_deferred),
        "unreproduced": list(rep.unreproduced), "unexcluded": list(rep.unexcluded),
        "nonconforming": list(rep.nonconforming),
        "min_support_ok": rep.min_support_ok, "judge_err_ok": rep.judge_err_ok,
        "weak_edges": [list(x) for x in rep.weak_edges],
        "hot_judges": [list(x) for x in rep.hot_judges],
        "holdout_acc": rep.holdout_acc,
        "errors": sum(1 for f in rep.findings if f.severity == "error"),
        "warnings": sum(1 for f in rep.findings if f.severity == "warn"),
    }


def summary(rep: VerifyReport) -> str:
    """人读的一行小结，进回执的 ``reason``——拒绝理由要能直接据以修。"""
    parts = [f"T+ 复述 {rep.reproduced}/{rep.total_positive}",
             f"T- 排除 {rep.excluded}/{rep.total_negative}"]
    if rep.fallback_deferred:
        parts.append(f"其中 {len(rep.fallback_deferred)} 条出错位置在 FALLBACK 段、"
                     "尚不可排除（不算失败）")
    if rep.weak_edges:
        parts.append(f"支持度不足的边 {len(rep.weak_edges)} 条 {rep.weak_edges}")
    if rep.hot_judges:
        parts.append(f"误差率超限的判断 {rep.hot_judges}")
    if rep.holdout_acc is not None:
        parts.append(f"留出集复述率 {rep.holdout_acc}")
    errs = [f for f in rep.findings if f.severity == "error"]
    if errs:
        parts.append("错误 " + "；".join(f.located() for f in errs[:3])
                     + ("……" if len(errs) > 3 else ""))
    return "；".join(parts)


def render(rep: VerifyReport) -> str:
    """把验收报告渲染成一段人读的文本。"""
    lines = ["机器验收报告", "=" * 32,
             f"结论: {'通过' if rep.ok else '未通过'}",
             f"T+ 复述: {rep.reproduced}/{rep.total_positive}",
             f"T- 排除: {rep.excluded}/{rep.total_negative}"
             f"（尚不可排除 {len(rep.fallback_deferred)} 条在 FALLBACK 段）",
             f"转移支持度: {'过' if rep.min_support_ok else '不过'}"
             f" {rep.weak_edges or ''}",
             f"判断误差率: {'过' if rep.judge_err_ok else '不过'}"
             f" {rep.hot_judges or ''}",
             f"留出集复述率: {rep.holdout_acc if rep.holdout_acc is not None else '（未给）'}"]
    for f in rep.findings:
        lines.append(f"  [{f.severity}] {f.located()}")
    return "\n".join(lines)


__all__ = [
    "VerifyReport", "batch_check", "fallback_entry", "render", "report_dict",
    "summary", "verify_machine",
]
