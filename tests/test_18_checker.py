"""⑱ 确定性守门程序（上）：受票的机器改写接口。

这套测试盯的是 :mod:`skill2fsm.checker` 与 :func:`skill2fsm.compiler.compile_round` 之间那
条唯一但要命的分界线——**被拒的提议不牵连它之前被接受的提议**。整轮撤销（compile_round）
在智能体式的编译里是灾难：第 7 条提议写错了，前 6 条正确的一起没了。所以这里逐条裁决，
只有 :meth:`~skill2fsm.checker.Checker.commit` 是全有全无的。

密闭：全部机器手工用受票接口搭出来，轨迹手工构造，**不调模型、不碰网络、不读文件**
（只有落盘那一条用 pytest 的 tmp_path）。
"""

import inspect

from skill2fsm import checker, verify
from skill2fsm.checker import Checker, Proposal, Receipt, check_machine
from skill2fsm.schema import Record, Trace, load_machine

JUDGE_Q = "这一步的核验结果算通过吗"
LABELS = ["通过", "不通过", "弃权"]

#: 这条条件与 ``verdict == '弃权'`` 在格局 ``verdict='弃权'`` 下同时成立——互斥完备（定理2）
#: 的反例，但它跟已有的边**文本不同**，所以过得了「重复的边」这道前置检查，非得靠
#: checks._determinism 的有限格局枚举才判得出来。坏提议就用它。
BAD_COND = "verdict != '通过'"


# --------------------------------------------------------------------------- #
# 一条正常的提议链：读题 → 跑代码 → 判断 →（不通过就绕回去修，有上限）→ 提交
# --------------------------------------------------------------------------- #
def _chain(ck: Checker, *, bad: bool = False) -> Checker:
    """把一台小机器一条提议一条提议地搭出来。``bad`` 在中间插一条会被拒的加边。"""
    ck.open_machine()
    ck.add_state("s1", {"kind": "tool", "name": "read_problem", "writes": ["problem"]},
                 clause="S1", initial=True)
    ck.add_state("s2", {"kind": "tool", "name": "run_python",
                        "reads": ["problem"], "writes": ["ok"]},
                 clause="S2", from_state="s1", from_support=2)
    ck.add_judge("j", prompt=JUDGE_Q, reads=["ok"], writes=["verdict"],
                 labels=LABELS, clause="RV.0.1", from_state="s2", from_support=2)
    ck.set_terminal("END", "done", kind="verified", output=["ok"],
                    from_state="j", from_support=2)
    ck.add_transition("j", "FALLBACK", cond="verdict == '弃权'")
    if bad:
        ck.add_transition("j", "s1", cond=BAD_COND, support=2)      # ← 会被拒
    ck.close_loop("j", "s2", cond="verdict == '不通过'",
                  counter="fix_count", bound=3, support=2)
    return ck


def _built(**kw) -> Checker:
    return _chain(Checker("math", doc="（技能文档，只作溯源）"), **kw)


def _good_trace(task_id="t1") -> Trace:
    """这台机器复述得出来的一条接受轨迹：读题、跑代码、判「通过」、提交。"""
    return Trace(task={"task_id": task_id, "input": {}}, verdict="accepted", records=[
        Record(step=1, action={"kind": "tool", "name": "read_problem"},
               output={"problem": "1+1"}),
        Record(step=2, action={"kind": "tool", "name": "run_python"},
               output={"ok": "2"}),
        Record(step=3, action={"kind": "judge"}, output={"verdict": "通过"}),
        Record(step=4, action={"kind": "end", "terminal": "done"}),
    ])


def _alien_trace() -> Trace:
    """第一步就跟机器对不上的接受轨迹——验收必然不过。"""
    return Trace(task={"task_id": "t-alien", "input": {}}, verdict="accepted", records=[
        Record(step=1, action={"kind": "tool", "name": "browse_the_web"}),
    ])


# --------------------------------------------------------------------------- #
# 正常链
# --------------------------------------------------------------------------- #
def test_a_valid_proposal_chain_is_accepted_end_to_end():
    ck = _built()
    assert all(r.accepted for r in ck.receipts()), \
        [r.reason for r in ck.receipts() if not r.accepted]
    m = ck.machine
    assert m.initial == "s1"
    assert set(m.states) == {"s1", "s2", "j", "END", "FALLBACK"}
    assert check_machine(m) == []                     # 结构 + 支持度 + 误差率 全过


def test_close_loop_installs_counter_and_bound_in_one_proposal():
    """回边、计数变量、上限出口是同一次提议——分开做的话中间态必然过不了检查。"""
    m = _built().machine
    back = [t for t in m.states["j"].transitions if t.to == "s2"]
    assert len(back) == 1 and back[0].inc == "fix_count"
    counter = m.var("fix_count")
    assert counter is not None and counter.type == "integer" and counter.init == 0
    exits = [t for t in m.states["s2"].transitions
             if t.cond == "fix_count >= 3" and t.to == "FALLBACK"]
    assert exits, m.states["s2"].transitions


def test_a_new_state_joins_the_spine_instead_of_growing_a_second_default_edge():
    """无条件入边**改写**源状态的兜底去向，而不是再长一条永远走不到的兜底边。"""
    m = _built().machine
    for sid in ("s1", "s2", "j"):
        assert len([t for t in m.states[sid].transitions if not t.cond]) == 1


# --------------------------------------------------------------------------- #
# 核心：一条坏提议被拒，而它之前被接受的那些**活着**
# --------------------------------------------------------------------------- #
def test_one_bad_transition_is_rejected_while_the_accepted_prefix_survives():
    ck = _built(bad=True)
    rs = ck.receipts()
    bad = [r for r in rs if not r.accepted]
    assert len(bad) == 1 and bad[0].op == "add_transition"

    m = ck.machine
    # 坏提议之前的那些改写一个不少
    assert set(m.states) == {"s1", "s2", "j", "END", "FALLBACK"}
    assert m.states["s1"].clause == "S1"
    assert m.states["j"].action.kind == "judge"
    assert [t for t in m.states["j"].transitions if t.cond == "verdict == '弃权'"]
    # 坏提议那条边一个字节都没落进来
    assert not [t for t in m.states["j"].transitions if t.cond == BAD_COND]
    # 坏提议之后的那条 close_loop 照常生效
    assert [t for t in m.states["j"].transitions if t.inc == "fix_count"]
    assert check_machine(m) == []


def test_the_rejection_receipt_names_the_failing_check_and_where():
    """理由要可据以重试：说清是哪条检查、在哪个状态上挂的。"""
    ck = _built(bad=True)
    bad = next(r for r in ck.receipts() if not r.accepted)
    assert "重叠" in bad.reason                 # 哪条检查（互斥完备）
    assert "E_OVERLAP" in bad.reason and "@j" in bad.reason      # 哪个状态
    codes = {f["code"] for f in bad.detail["findings"]}
    assert "E_OVERLAP" in codes
    assert all(f["severity"] in ("error", "warn") for f in bad.detail["findings"])


def test_preconditions_are_refused_before_anything_is_touched():
    """状态不存在、重复的边、第二条兜底边——都在候选副本上就地拒掉，理由带现有状态。"""
    ck = _built()
    before = ck.machine.model_dump()

    r1 = ck.add_transition("j", "ghost")
    assert not r1.accepted and "ghost" in r1.reason

    r2 = ck.add_transition("j", "END", cond="verdict == '弃权'")
    assert not r2.accepted and "已有一条条件完全相同的边" in r2.reason

    r3 = ck.add_transition("s1", "END")
    assert not r3.accepted and "兜底边" in r3.reason

    r4 = ck.add_transition("END", "s1")
    assert not r4.accepted and "终止态" in r4.reason

    r5 = ck.add_transition("j", "END", cond="nowhere == 1")
    assert not r5.accepted and "未声明的变量" in r5.reason

    assert ck.machine.model_dump() == before      # 五次拒绝，机器一动没动


def test_close_loop_refuses_an_edge_that_does_not_close_a_loop():
    ck = _built()
    r = ck.close_loop("s1", "END", counter="k", bound=2)
    assert not r.accepted and "不是回边" in r.reason


def test_close_loop_sets_a_targets_bound_once_and_says_how_to_add_the_second_back_edge():
    """``fit.install_counter`` 按目标认回边，所以同一个目标的上限只设一次——歧义不替人猜。"""
    ck = _built()
    r = ck.close_loop("j", "s2", cond="verdict == '弃权'", counter="fix_count")
    assert not r.accepted
    assert "已有一条通向 s2 的边" in r.reason and "add_transition" in r.reason
    # 上限出口已经装好了，第二条回边带上同一个计数变量走 add_transition 就行
    ok = ck.add_transition("j", "s2", cond="verdict == '通过'",
                           inc="fix_count", support=2)
    assert ok.accepted, ok.reason
    assert check_machine(ck.machine) == []


def test_add_transition_refuses_an_inc_on_an_undeclared_counter():
    """计数变量只能由 close_loop 连着上限出口一起装，不能手工挂个 inc 了事。"""
    ck = _built()
    r = ck.add_transition("j", "s2", cond="verdict == '通过'", inc="nope")
    assert not r.accepted and "close_loop" in r.reason


def test_add_judge_refuses_a_judge_that_is_already_too_noisy():
    """明知误差率超上限还往机器里装，等于主动顶穿 Σεᵢ 那条不等式。"""
    ck = _built()
    r = ck.add_judge("j", prompt=JUDGE_Q, reads=["ok"], writes=["verdict"],
                     labels=LABELS, error_rate=0.35)
    assert not r.accepted
    assert "0.35" in r.reason and "0.2" in r.reason
    assert ck.machine.states["j"].action.error_rate == 0.0


# --------------------------------------------------------------------------- #
# commit：全有全无
# --------------------------------------------------------------------------- #
def test_commit_rolls_back_the_whole_batch_when_verification_fails():
    ck = _built()
    r = ck.commit(t_plus=[_good_trace(), _alien_trace()])
    assert not r.accepted
    assert "回滚" in r.reason and "复述 1/2" in r.reason
    assert r.detail["rolled_back"] is True
    # 整批回到 open_machine 时的样子：一台全回退的空机器
    m = ck.machine
    assert m.initial == "FALLBACK" and set(m.states) == {"FALLBACK"}


def test_commit_passes_and_is_the_only_thing_that_writes_the_machine_file(tmp_path):
    ck = _built()
    r = ck.commit(t_plus=[_good_trace()], root=tmp_path)
    assert r.accepted, r.reason
    assert (tmp_path / "machine.json").is_file()
    assert load_machine(tmp_path).initial == "s1"
    assert r.detail["report"]["ok"] is True


def test_a_failed_commit_rolls_back_only_to_the_last_successful_commit():
    """上一次成功的 commit 是新的地基：回滚回到它，不是回到 open。"""
    ck = _built()
    assert ck.commit(t_plus=[_good_trace()]).accepted
    ck.add_state("s9", {"kind": "tool", "name": "cleanup"},
                 from_state="s2", from_cond="fix_count < 3 and ok == 'X'",
                 from_support=2)
    assert ck.commit(t_plus=[_good_trace(), _alien_trace()]).accepted is False
    m = ck.machine
    assert set(m.states) == {"s1", "s2", "j", "END", "FALLBACK"}    # s9 没了
    assert m.initial == "s1"                                       # 但地基还在


def test_rollback_does_not_erase_the_receipts():
    ck = _built()
    n = len(ck.receipts())
    ck.commit(t_plus=[_alien_trace()])
    assert len(ck.receipts()) == n + 1
    assert ck.receipts()[-1].op == "commit" and not ck.receipts()[-1].accepted


# --------------------------------------------------------------------------- #
# 逃生口
# --------------------------------------------------------------------------- #
def test_demote_to_fallback_is_the_escape_hatch_that_always_works():
    """条件学不出来 / 判断太吵 / 环收不住——退回解释执行永远是可用的那条路。"""
    ck = _built(bad=True)
    assert not ck.receipts()[-2].accepted          # 刚被拒过一条
    r = ck.demote_to_fallback("j", note="这个分岔的条件学不出来")
    assert r.accepted, r.reason
    m = ck.machine
    assert [(t.cond, t.to) for t in m.states["j"].transitions] == [("", "FALLBACK")]
    assert check_machine(m) == []


def test_demote_drops_the_segment_that_only_hung_off_that_state():
    """退回解释执行 = 放弃那一段编译产物；留着它们只会变成走不到的死代码。"""
    ck = _built()
    r = ck.demote_to_fallback("s2")
    assert r.accepted, r.reason
    m = ck.machine
    assert set(m.states) == {"s1", "s2", "FALLBACK"}      # j / END 一并下线
    assert "END" in r.reason and "j" in r.reason


def test_demote_refuses_only_where_there_is_nothing_to_demote():
    ck = _built()
    assert not ck.demote_to_fallback("nope").accepted
    assert not ck.demote_to_fallback("END").accepted      # 终止态本就没有出边
    assert not ck.demote_to_fallback("FALLBACK").accepted


# --------------------------------------------------------------------------- #
# 审计痕迹
# --------------------------------------------------------------------------- #
def test_receipts_form_a_complete_audit_trail():
    ck = _built(bad=True)
    ck.demote_to_fallback("j")
    ck.commit(t_plus=[_good_trace()])
    rs = ck.receipts()
    assert [r.op for r in rs] == [
        "open_machine", "add_state", "add_state", "add_judge", "set_terminal",
        "add_transition", "add_transition", "close_loop",
        "demote_to_fallback", "commit"]
    assert all(isinstance(r, Receipt) for r in rs)
    assert all(r.reason.strip() for r in rs)              # 每一张都写了理由
    assert [r.accepted for r in rs].count(False) == 1     # 只有那条坏提议被拒
    # 每张回执都带得回它的提议参数，能照着重放
    assert rs[1].detail["state_id"] == "s1"
    assert rs[-1].detail["t_plus"] == 1


def test_ops_before_open_machine_are_refused_with_an_actionable_reason():
    ck = Checker("math")
    r = ck.add_state("s1", {"kind": "tool", "name": "x"}, initial=True)
    assert not r.accepted and "open_machine" in r.reason
    assert not ck.opened
    assert not ck.commit().accepted


def test_open_machine_refuses_a_second_open_and_a_mismatched_base():
    ck = _built()
    assert not ck.open_machine().accepted
    other = Checker("table-clean")
    r = other.open_machine(base=_built().machine)
    assert not r.accepted and "skill_id" in r.reason


def test_the_machine_property_hands_out_a_copy_not_the_live_object():
    """交出内部对象就等于开了一条绕过受票接口的就地修改通道。"""
    ck = _built()
    stolen = ck.machine
    stolen.states["s1"].transitions = []
    assert ck.machine.states["s1"].transitions            # 内部没被动到
    assert check_machine(ck.machine) == []


# --------------------------------------------------------------------------- #
# 批量
# --------------------------------------------------------------------------- #
def test_batch_check_keeps_the_accepted_prefix_and_returns_one_receipt_per_proposal():
    ck = Checker("math")
    ck.open_machine()
    ck.add_state("s1", {"kind": "tool", "name": "read_problem", "writes": ["problem"]},
                 initial=True)
    base = ck.machine

    props = [
        Proposal("add_state", {"state_id": "s2",
                               "action": {"kind": "tool", "name": "run_python",
                                          "reads": ["problem"], "writes": ["ok"]},
                               "from_state": "s1", "from_support": 2}),
        Proposal("add_transition", {"from_state": "s2", "to": "ghost"}),   # 坏
        Proposal("set_terminal", {"state_id": "END", "terminal": "done",
                                  "output": ["ok"], "from_state": "s2",
                                  "from_support": 2}),
        Proposal("no_such_op", {}),                                        # 坏
    ]
    m, rs = checker.batch_check(base, props)
    assert [r.accepted for r in rs] == [True, False, True, False]
    assert len(rs) == len(props)
    assert set(m.states) == {"s1", "s2", "END", "FALLBACK"}
    assert "只认" in rs[3].reason
    assert set(base.states) == {"s1", "FALLBACK"}          # base 没被就地改


def test_batch_check_reports_a_base_it_cannot_even_take_over():
    ck = _built()
    broken = ck.machine
    broken.states["s2"].transitions = []                   # 跑完就卡住
    m, rs = checker.batch_check(broken, [Proposal("demote_to_fallback",
                                                  {"state_id": "s2"})])
    assert len(rs) == 1 and rs[0].op == "open_machine" and not rs[0].accepted
    assert "拒绝接管" in rs[0].reason
    assert m.states["s2"].transitions == []                # 原样交还


# --------------------------------------------------------------------------- #
# 「不调模型」是结构性的，不是承诺
# --------------------------------------------------------------------------- #
def test_neither_gate_module_can_call_a_model():
    """没有一个 model 参数、不 import 任何模型客户端——判机器好坏不该由一次采样裁决。"""
    for mod in (checker, verify):
        src = inspect.getsource(mod)
        for banned in ("llm_client", "model_iface", "OpenAIClient", "ModelAdapter"):
            assert banned not in src, (mod.__name__, banned)
        for name, obj in vars(mod).items():
            fns = []
            if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                fns.append(obj)
            elif inspect.isclass(obj) and obj.__module__ == mod.__name__:
                fns += [f for _n, f in inspect.getmembers(obj, inspect.isfunction)
                        if f.__module__ == mod.__name__]
            for fn in fns:
                params = set(inspect.signature(fn).parameters)
                assert "model" not in params, (mod.__name__, name, fn.__name__)
