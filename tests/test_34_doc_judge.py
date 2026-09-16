"""Judge actions introduced from the document: zero-width replay, programmatic labelers, gold labels never produced by a model (invariant I7).

Some variables of a reference machine have no snapshot in any trace, so judges that read them
**cannot be fitted from traces** and can only be introduced by the compiler from the document.
Once introduced, the trace has no such step -- during replay the judge is **zero-width**: it
consumes no record, and its label is computed on the spot by the programmatic labeler named by
``gold_from``; calibration and replay use the same :data:`trace_adapter.LABELERS`, so the two
sides agree by construction. Pinned here:

* an introduced judge with a labeler picks its edge in replay by the labeler's label; T+ is
  reproduced, wrong paths are excluded;
* labeler missing / no label => abstain => FALLBACK, and ``fallback_at`` points just
  after the record **before** the judge;
* a trace produced by arm three that really carries a matching judge record is consumed as an
  ordinary judge step, not zero-width a second time;
* the three rulings of the built-in labeler ``fail_attr_from_trace``;
* the labeler table holds only pure functions; nothing holds a model handle.
"""
from __future__ import annotations

import inspect

import pytest

from hexis.legacy import replay
from hexis.machine.schema import (
    EndAction,
    JudgeAction,
    Machine,
    Record,
    State,
    Terminal,
    ToolAction,
    Trace,
    Transition,
    Variable,
)
from hexis.traces import trace_adapter


def _tool(step, name, **vars_):
    return Record(step=step, action={"kind": "tool", "name": name, "input": {}},
                  output={}, vars=dict(vars_))


def _end(step, **vars_):
    return Record(step=step, action={"kind": "end", "terminal": "done"}, vars=dict(vars_))


def _judge_machine(gold_from: str = "x_is_big", support: int = 5) -> Machine:
    """s1 read(x) -> j (introduced judge, writes big) -> yes: s2 hi; no: s3 lo; default FALLBACK."""
    return Machine(
        skill_id="toy", initial="s1",
        variables=[Variable(name="x"), Variable(name="big")],
        states={
            "s1": State(id="s1", action=ToolAction(name="read", writes=["x"]),
                        transitions=[Transition(to="j", support=5)]),
            "j": State(id="j", action=JudgeAction(
                prompt="is x big?", reads=["x"], writes=["big"],
                labels=["yes", "no", "abstain"], introduced=True, gold_from=gold_from,
                support=support),
                transitions=[Transition(cond="big == 'yes'", to="s2", support=3),
                             Transition(cond="big == 'no'", to="s3", support=2),
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
        return "yes" if str(trace.records[i].vars.get("x")) == "9" else "no"
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
    assert not r.ok and r.diverged_at == 1               # the judge says "yes", the machine wants hi, the trace has lo
    # the zero-width judge is recorded at the index of the record before it and consumes no record
    assert [(i, s) for i, s in replay.walk(m, T_HI).seq] == [(0, "s1"), (0, "j"), (1, "s2"), (2, "E")]


def test_missing_labeler_abstains_into_fallback():
    m = _judge_machine(gold_from="no_such_labeler")
    r = replay.walk(m, T_HI)
    assert r.ok and r.fallback_at == 2                   # after the judge, the fallback segment starts at the hi step (step 2)
    assert r.values["big"] == "abstain"


def test_a_real_judge_record_is_consumed_instead_of_zero_width(labeler):
    m = _judge_machine()
    with_rec = Trace(records=[
        _tool(1, "read", x="9"),
        Record(step=2, action={"kind": "judge", "prompt": "is x big?", "reads": ["x"]},
               output={"big": "yes"}, vars={"x": "9", "big": "yes"}),
        _tool(3, "hi", x="9", big="yes"), _end(4, x="9", big="yes")])
    r = replay.walk(m, with_rec)
    assert r.ok
    assert [(i, s) for i, s in r.seq] == [(0, "s1"), (1, "j"), (2, "s2"), (3, "E")]


def test_replay_and_calibration_share_the_labeler_table(labeler):
    """The label in replay is the label given by the labeler -- calibration samples come from the same table, so the two sides agree by construction."""
    m = _judge_machine()
    got = replay.walk(m, T_HI).values["big"]
    assert got == trace_adapter.LABELERS["x_is_big"](T_HI, 0) == "yes"


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
    # at the next check candidate is unchanged and PASS => the check was written wrong (核验写错了)
    t = Trace(records=[_verify(1, "FAIL", "4"), _verify(2, "PASS", "4")])
    assert fn(t, 0) == "核验写错了"
    # at the next check candidate has changed => the solution was wrong (解答有错)
    t = Trace(records=[_verify(1, "FAIL", "3"), _verify(2, "PASS", "4")])
    assert fn(t, 0) == "解答有错"
    # no next check / this step already PASSed / not a check step => None
    assert fn(Trace(records=[_verify(1, "FAIL", "3")]), 0) is None
    assert fn(Trace(records=[_verify(1, "PASS", "3"), _verify(2, "PASS", "3")]), 0) is None
    assert fn(Trace(records=[_tool(1, "read"), _verify(2, "PASS", "3")]), 0) is None
