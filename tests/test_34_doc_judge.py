"""Programmatic labelers: pure functions of a trace position, never a model.

Judges introduced from the document get their gold labels from the labelers registered in
:data:`hexis.traces.trace_adapter.LABELERS`. Pinned here:

* the labeler table holds only pure functions; nothing holds a model handle;
* the three rulings of the built-in labeler ``fail_attr_from_trace``.
"""
from __future__ import annotations

import inspect

from hexis.machine.schema import Record, Trace
from hexis.traces import trace_adapter


def _tool(step, name, **vars_):
    return Record(step=step, action={"kind": "tool", "name": name, "input": {}},
                  output={}, vars=dict(vars_))


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
