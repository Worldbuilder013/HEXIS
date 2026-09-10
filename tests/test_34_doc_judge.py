"""从文档引入的判断动作：零宽回放、程序打标器、金标不经模型（不变量 I7）。

参考机 12 个变量里 6 个在任何轨迹里零快照（reference_machine.py 自述），所以结构②④那种
判断**不可能从轨迹拟合**，只能由编译器从文档引入。引入之后轨迹里没有这一步——回放时它是
**零宽**的：不消费记录，标签由 ``gold_from`` 指名的程序打标器现算；标定与回放用同一张
:data:`trace_adapter.LABELERS`，两边构造上一致。这里钉住：

* 有打标器的引入判断在回放里按打标器的标签选边，T+ 复述、错路排除；
* 打标器缺失 / 打不出 ⇒ 弃权 ⇒ FALLBACK，``fallback_at`` 指在判断**之前**那条记录之后；
* 臂三跑出来、真带一条匹配判断记录的轨迹按普通判断步消费，不重复零宽；
* 内置打标器 ``fail_attr_from_trace`` 的三种裁决；
* 打标器表里只有纯函数，没有任何东西持有模型句柄。
"""
from __future__ import annotations

import inspect

import pytest

from skill2fsm import replay, trace_adapter
from skill2fsm.schema import (
    EndAction, JudgeAction, Machine, Record, State, Terminal, ToolAction, Trace,
    Transition, Variable,
)


def _tool(step, name, **vars_):
    return Record(step=step, action={"kind": "tool", "name": name, "input": {}},
                  output={}, vars=dict(vars_))


def _end(step, **vars_):
    return Record(step=step, action={"kind": "end", "terminal": "done"}, vars=dict(vars_))


def _judge_machine(gold_from: str = "x_is_big", support: int = 5) -> Machine:
    """s1 read(x) → j（引入判断，写 big）→ 是: s2 hi ；否: s3 lo ；默认 FALLBACK。"""
    return Machine(
        skill_id="toy", initial="s1",
        variables=[Variable(name="x"), Variable(name="big")],
        states={
            "s1": State(id="s1", action=ToolAction(name="read", writes=["x"]),
                        transitions=[Transition(to="j", support=5)]),
            "j": State(id="j", action=JudgeAction(
                prompt="x 大吗？", reads=["x"], writes=["big"],
                labels=["是", "否", "弃权"], introduced=True, gold_from=gold_from,
                support=support),
                transitions=[Transition(cond="big == '是'", to="s2", support=3),
                             Transition(cond="big == '否'", to="s3", support=2),
                             Transition(to="FALLBACK")]),
            "s2": State(id="s2", action=ToolAction(name="hi"),
                        transitions=[Transition(to="E", support=3)]),
            "s3": State(id="s3", action=ToolAction(name="lo"),
                        transitions=[Transition(to="E", support=2)]),
            "E": State(id="E", action=EndAction(terminal="done")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done")],
    )


@pytest.fixture
def labeler():
    def x_is_big(trace: Trace, i: int):
        return "是" if str(trace.records[i].vars.get("x")) == "9" else "否"
    trace_adapter.LABELERS["x_is_big"] = x_is_big
    yield x_is_big
    trace_adapter.LABELERS.pop("x_is_big", None)


T_HI = Trace(records=[_tool(1, "read", x="9"), _tool(2, "hi", x="9"), _end(3, x="9")])
T_LO = Trace(records=[_tool(1, "read", x="1"), _tool(2, "lo", x="1"), _end(3, x="1")])
T_BAD = Trace(records=[_tool(1, "read", x="9"), _tool(2, "lo", x="9"), _end(3, x="9")])


# --------------------------------------------------------------------------- #
def test_labeler_drives_the_zero_width_judge_in_replay(labeler):
    m = _judge_machine()
    assert replay.replay(m, T_HI).ok
    assert replay.replay(m, T_LO).ok
    r = replay.walk(m, T_BAD)
    assert not r.ok and r.diverged_at == 1               # 判断说「是」，机器要 hi，轨迹是 lo
    # 零宽判断记在它之前那条记录的下标上，没有消费记录
    assert [(i, s) for i, s in replay.walk(m, T_HI).seq] == [(0, "s1"), (0, "j"), (1, "s2"), (2, "E")]


def test_missing_labeler_abstains_into_fallback():
    m = _judge_machine(gold_from="no_such_labeler")
    r = replay.walk(m, T_HI)
    assert r.ok and r.fallback_at == 2                   # 判断之后、hi 那一步（step 2）起是回退段
    assert r.values["big"] == "弃权"


def test_a_real_judge_record_is_consumed_instead_of_zero_width(labeler):
    m = _judge_machine()
    with_rec = Trace(records=[
        _tool(1, "read", x="9"),
        Record(step=2, action={"kind": "judge", "prompt": "x 大吗？", "reads": ["x"]},
               output={"big": "是"}, vars={"x": "9", "big": "是"}),
        _tool(3, "hi", x="9", big="是"), _end(4, x="9", big="是")])
    r = replay.walk(m, with_rec)
    assert r.ok
    assert [(i, s) for i, s in r.seq] == [(0, "s1"), (1, "j"), (2, "s2"), (3, "E")]


def test_replay_and_calibration_share_the_labeler_table(labeler):
    """回放里的标签就是打标器给的标签——标定样本从同一张表取，两边构造上一致。"""
    m = _judge_machine()
    got = replay.walk(m, T_HI).values["big"]
    assert got == trace_adapter.LABELERS["x_is_big"](T_HI, 0) == "是"


def test_labelers_are_plain_functions_without_a_model_handle():
    for name, fn in trace_adapter.LABELERS.items():
        assert callable(fn), name
        params = list(inspect.signature(fn).parameters)
        assert params[:2] == ["trace", "i"], name
        assert "model" not in params and "client" not in params, name
    assert "fail_attr_from_trace" in trace_adapter.LABELERS


# --------------------------------------------------------------------------- #
def _verify(step, status, candidate):
    return Record(step=step, action={"kind": "tool", "name": "math_verify", "input": {}},
                  output={"verify_status": status},
                  vars={"verify_status": status, "candidate": candidate})


def test_fail_attr_from_trace_three_rulings():
    fn = trace_adapter.fail_attr_from_trace
    # 下一次核验 candidate 没变且 PASS ⇒ 核验写错了
    t = Trace(records=[_verify(1, "FAIL", "4"), _verify(2, "PASS", "4")])
    assert fn(t, 0) == "核验写错了"
    # 下一次核验 candidate 变了 ⇒ 解答有错
    t = Trace(records=[_verify(1, "FAIL", "3"), _verify(2, "PASS", "4")])
    assert fn(t, 0) == "解答有错"
    # 没有下一次核验 / 这一步本来就 PASS / 不是核验步 ⇒ None
    assert fn(Trace(records=[_verify(1, "FAIL", "3")]), 0) is None
    assert fn(Trace(records=[_verify(1, "PASS", "3"), _verify(2, "PASS", "3")]), 0) is None
    assert fn(Trace(records=[_tool(1, "read"), _verify(2, "PASS", "3")]), 0) is None
