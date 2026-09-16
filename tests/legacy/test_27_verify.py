"""㉗ 确定性守门程序（下）：一台做完了的机器的验收门槛。

四道门里有两道是这一层新加的（仓库里此前没有任何地方检查）：**每条转移的支持度**与**每个
判断动作的标定误差率**。另外两道是复述接受轨迹、排除拒绝轨迹。

这套测试最要紧的一条是负例那道门的例外：一条反例的出错位置若落在机器已经进了 FALLBACK
之后的那一段，机器在那里**没有结构可偏离**（解释模式不承诺具体路径），因此排除不了。这不
是失败，是「这一段还没编译到」。把它算成失败，会逼着编译过程去修一个不存在的缺陷，或者
更糟：为了让数字好看而把 FALLBACK 砍掉。

密闭：机器与轨迹全部手工构造，**不调模型、不碰网络、不读文件**。
"""

from hexis.legacy import verify
from hexis.legacy.checker import Finding
from hexis.machine.schema import (
    EndAction, JudgeAction, Machine, Record, State, Terminal, Thresholds,
    ToolAction, Trace, Transition, Variable,
)

JUDGE_Q = "这一步的核验结果算通过吗"
LABELS = ["通过", "弃权"]


# --------------------------------------------------------------------------- #
# 机器
# --------------------------------------------------------------------------- #
def _clean_machine(*, support: int = 3) -> Machine:
    """读题 → 跑代码 → 提交。三条门槛都过得去的一台小机器。"""
    return Machine(
        skill_id="math", initial="s1",
        variables=[Variable(name="problem", type="string"),
                   Variable(name="ok", type="string")],
        states={
            "s1": State(id="s1", clause="S1",
                        action=ToolAction(name="read_problem", writes=["problem"]),
                        transitions=[Transition(to="s2", support=support)]),
            "s2": State(id="s2", clause="RV.0.1",
                        action=ToolAction(name="run_python", reads=["problem"],
                                          writes=["ok"]),
                        transitions=[Transition(to="END", support=support)]),
            "END": State(id="END", action=EndAction(terminal="done")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done", kind="verified", output=["ok"])],
    )


def _judge_machine(error_rate: float) -> Machine:
    """读题 → 判断 → 提交。判断动作的标定误差率由参数给。"""
    m = _clean_machine()
    m.states["s2"] = State(
        id="s2", clause="RV.0.1",
        action=JudgeAction(prompt=JUDGE_Q, reads=["problem"], writes=["ok"],
                           labels=list(LABELS), error_rate=error_rate, support=8),
        transitions=[Transition(to="END", support=3)])
    return m


def _stub_machine() -> Machine:
    """只学会了第一步、之后整段交回解释执行的机器。反例落在它的回退段里。"""
    return Machine(
        skill_id="math", initial="s1",
        variables=[Variable(name="problem", type="string")],
        states={
            "s1": State(id="s1", clause="S1",
                        action=ToolAction(name="read_problem", writes=["problem"]),
                        transitions=[Transition(to="FALLBACK")]),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done")])


# --------------------------------------------------------------------------- #
# 轨迹
# --------------------------------------------------------------------------- #
def _positive(task_id="t+", judge=False) -> Trace:
    mid = ({"kind": "judge"} if judge else {"kind": "tool", "name": "run_python"})
    return Trace(task={"task_id": task_id, "input": {}}, verdict="accepted", records=[
        Record(step=1, action={"kind": "tool", "name": "read_problem"},
               output={"problem": "1+1"}),
        Record(step=2, action=mid, output={"ok": "2"}),
        Record(step=3, action={"kind": "end", "terminal": "done"}),
    ])


def _negative_in_compiled_region(task_id="t-compiled") -> Trace:
    """走的路跟机器一模一样、结果却错了：出错位置在已编译区段，机器该偏离却没偏离。"""
    t = _positive(task_id)
    return Trace(task=t.task, verdict="rejected", error_step=2, records=t.records)


def _negative_in_fallback_segment(task_id="t-fallback") -> Trace:
    """第 2 步就跑到机器还没编译的地方去了（机器第 2 步已经在 FALLBACK 里）。"""
    return Trace(task={"task_id": task_id, "input": {}}, verdict="rejected",
                 error_step=2, records=[
                     Record(step=1, action={"kind": "tool", "name": "read_problem"},
                            output={"problem": "1+1"}),
                     Record(step=2, action={"kind": "tool", "name": "submit_answer",
                                            "input": {"answer": "41"}}),
                 ])


# --------------------------------------------------------------------------- #
# 门 ①：支持度
# --------------------------------------------------------------------------- #
def test_an_edge_with_support_one_fails_min_support_and_shows_up_in_weak_edges():
    """只被一条轨迹走过的边不是规律，是巧合。"""
    m = _clean_machine()
    m.states["s2"].transitions[0].support = 1
    rep = verify.verify_machine(m, [_positive()])
    assert rep.min_support_ok is False
    assert rep.weak_edges == [("s2", "")]
    assert rep.ok is False
    assert rep.reproduced == 1 and rep.total_positive == 1     # 复述那道门是过的
    hit = [f for f in rep.findings if f.code == "W_LOW_SUPPORT"]
    assert len(hit) == 1 and hit[0].state_id == "s2" and hit[0].severity == "warn"
    assert "支持度 1" in hit[0].message and "下限 2" in hit[0].message


def test_the_min_support_bar_is_configurable_and_exempts_edges_into_fallback():
    """去 FALLBACK 的边按定义没有轨迹支持——拿支持度要求它，等于要求编译器不许认怂。"""
    m = _clean_machine(support=1)
    assert verify.verify_machine(
        m, thresholds=Thresholds(min_support=1)).min_support_ok is True
    stub = _stub_machine()                                     # 唯一的边就是去 FALLBACK
    assert verify.verify_machine(stub).weak_edges == []


# --------------------------------------------------------------------------- #
# 门 ②：判断动作误差率
# --------------------------------------------------------------------------- #
def test_a_judge_with_error_rate_over_the_cap_fails_and_shows_up_in_hot_judges():
    """Σεᵢ 那条不等式里每个 εᵢ 都要有天花板，否则一个太吵的判断就把上界顶穿。"""
    m = _judge_machine(0.3)
    rep = verify.verify_machine(m, [_positive(judge=True)])
    assert rep.judge_err_ok is False
    assert rep.hot_judges == [("s2", 0.3)]
    assert rep.ok is False
    hit = [f for f in rep.findings if f.code == "W_JUDGE_ERR"]
    assert len(hit) == 1 and hit[0].state_id == "s2"
    assert "0.3" in hit[0].message and "0.2" in hit[0].message


def test_a_judge_right_at_the_cap_passes():
    """上限是 ≤，不是 <：正好压线的判断留得住。"""
    rep = verify.verify_machine(_judge_machine(0.2), [_positive(judge=True)])
    assert rep.judge_err_ok is True and rep.hot_judges == [] and rep.ok is True


# --------------------------------------------------------------------------- #
# 门 ③：复述接受轨迹
# --------------------------------------------------------------------------- #
def test_a_positive_trace_that_does_not_replay_is_an_error():
    alien = Trace(task={"task_id": "t-alien", "input": {}}, verdict="accepted",
                  records=[Record(step=1, action={"kind": "tool", "name": "browse"})])
    rep = verify.verify_machine(_clean_machine(), [_positive(), alien])
    assert rep.reproduced == 1 and rep.total_positive == 2
    assert rep.unreproduced == [1]
    assert rep.ok is False
    err = [f for f in rep.findings if f.code == "E_NOT_REPRODUCED"]
    assert len(err) == 1 and err[0].severity == "error"
    assert "t-alien" in err[0].message


# --------------------------------------------------------------------------- #
# 门 ④：排除拒绝轨迹 —— 以及它那个必须分清的例外
# --------------------------------------------------------------------------- #
def test_a_negative_in_the_compiled_region_that_is_not_excluded_is_a_real_failure():
    """反例走的路机器照样走得通：状态合并过头，或缺一条禁止项。这是真失败。"""
    neg = _negative_in_compiled_region()
    rep = verify.verify_machine(_clean_machine(), [], [neg])
    assert rep.excluded == 0 and rep.total_negative == 1
    assert rep.unexcluded == [0] and rep.fallback_deferred == []
    assert rep.ok is False
    err = [f for f in rep.findings if f.code == "E_NOT_EXCLUDED"]
    assert len(err) == 1 and err[0].severity == "error"


def test_a_negative_whose_error_step_is_in_the_fallback_segment_is_deferred_not_failed():
    """出错位置落在回退段：机器在那里没有结构可偏离，**尚不可排除**——不算失败。"""
    m, neg = _stub_machine(), _negative_in_fallback_segment()
    rep = verify.verify_machine(m, [], [neg])
    assert rep.excluded == 0 and rep.total_negative == 1
    assert rep.fallback_deferred == [0] and rep.unexcluded == []
    assert not [f for f in rep.findings if f.severity == "error"]
    assert rep.ok is True                                   # ← 不计失败
    warn = [f for f in rep.findings if f.code == "W_FALLBACK_DEFERRED"]
    assert len(warn) == 1 and warn[0].severity == "warn"
    assert "尚不可排除" in warn[0].message and "不算失败" in warn[0].message
    assert "尚不可排除" in verify.summary(rep)


def test_fallback_entry_reports_the_step_where_the_machine_gives_up():
    """回退段的边界是可算的：机器第几步进的 FALLBACK。"""
    assert verify.fallback_entry(_stub_machine(),
                                 _negative_in_fallback_segment()) == 2
    # 全程编译好的机器上没有回退段
    assert verify.fallback_entry(_clean_machine(), _positive()) is None


def test_the_deferral_disappears_once_that_step_is_actually_compiled():
    """例外是暂时的，不是免死金牌：同一条反例，在把那一步编译出来的机器上真的被排除了。

    它第 2 步走的是 ``submit_answer``，而编译好的机器在那一步要 ``run_python``——机器就在
    出错位置偏离，这正是 ``replay.excludes`` 要的。回退段那条豁免只是「还没轮到检查它」。
    """
    neg = _negative_in_fallback_segment()
    rep = verify.verify_machine(_clean_machine(), [_positive()], [neg])
    assert rep.fallback_deferred == []          # 不再享受豁免
    assert rep.excluded == 1 and rep.unexcluded == []
    assert rep.ok is True


def test_an_excluded_negative_counts_as_excluded():
    """机器在出错位置或更早偏离 = 排除。"""
    neg = Trace(task={"task_id": "t-div", "input": {}}, verdict="rejected",
                error_step=2, records=[
                    Record(step=1, action={"kind": "tool", "name": "read_problem"},
                           output={"problem": "1+1"}),
                    Record(step=2, action={"kind": "tool", "name": "guess"}),
                ])
    rep = verify.verify_machine(_clean_machine(), [_positive()], [neg])
    assert rep.excluded == 1 and rep.unexcluded == [] and rep.fallback_deferred == []
    assert rep.ok is True


# --------------------------------------------------------------------------- #
# 全过 / 留出集 / 口子
# --------------------------------------------------------------------------- #
def test_a_machine_that_passes_everything_is_ok():
    rep = verify.verify_machine(_clean_machine(), [_positive("a"), _positive("b")])
    assert rep.ok is True
    assert (rep.reproduced, rep.total_positive) == (2, 2)
    assert (rep.excluded, rep.total_negative) == (0, 0)
    assert rep.min_support_ok and rep.judge_err_ok
    assert rep.weak_edges == [] and rep.hot_judges == []
    assert rep.findings == [] and rep.holdout_acc is None


def test_the_holdout_gate_is_independent_of_the_training_traces():
    """留出集复述率低于 acc_thr 也不过——路径对了但泛化不了，同样不该落地。"""
    held = Trace(task={"task_id": "h", "input": {}}, verdict="accepted",
                 records=[Record(step=1, action={"kind": "tool", "name": "browse"})])
    rep = verify.verify_machine(_clean_machine(), [_positive()], holdout=[held])
    assert rep.holdout_acc == 0.0 and rep.ok is False
    assert not [f for f in rep.findings if f.severity == "error"]   # 训练集全过
    good = verify.verify_machine(_clean_machine(), [_positive()],
                                 holdout=[_positive("h2")])
    assert good.holdout_acc == 1.0 and good.ok is True


def test_batch_check_is_just_the_findings_of_verify_machine():
    m = _judge_machine(0.3)
    fs = verify.batch_check(m, [_positive(judge=True)])
    assert all(isinstance(f, Finding) for f in fs)
    assert fs == verify.verify_machine(m, [_positive(judge=True)]).findings
    assert {f.code for f in fs} == {"W_JUDGE_ERR"}


def test_structural_breakage_shows_up_as_an_error_finding_and_blocks_ok():
    """结构检查那八条照样在这道验收里：它们是 error，直接把 ok 判掉。"""
    m = _clean_machine()
    m.states["s2"].transitions = []                    # 跑完就卡住
    rep = verify.verify_machine(m, [])
    assert rep.ok is False
    codes = {f.code for f in rep.findings if f.severity == "error"}
    assert "E_NO_EDGE" in codes


def test_report_and_render_round_out_the_receipt_surface():
    rep = verify.verify_machine(_clean_machine(), [_positive()])
    d = verify.report_dict(rep)
    assert d["ok"] is True and d["reproduced"] == 1 and d["errors"] == 0
    text = verify.render(rep)
    assert "机器验收报告" in text and "T+ 复述: 1/1" in text
