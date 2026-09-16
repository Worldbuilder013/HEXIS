"""Judging is correct: the objective acceptance check is right, and violating a prohibition (overwriting the source file) is rejected even when the result is correct."""

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.traces import judge


def _run(task):
    fs = tc.MemFS(task["files"])
    return runtime.run_task(tc.reference_machine(), task, model=tc.build_model(),
                            tools=tc.build_registry(fs), doc=tc.skill_doc())


def test_clean_task_is_accepted():
    task = tc.gen_tasks(1, seed=5)[0]
    res = _run(task)
    v = judge.evaluate(res.trace, lambda t: tc.verify(task, t),
                       tc.reference_machine().prohibitions)
    assert v.verdict == "accepted", (v.reason, res.path())


def test_overwriting_source_is_rejected_even_when_result_is_correct():
    """output_path == path: the resulting header is still well-formed (acceptance passes), but P1 is violated, so the run must be rejected."""
    task = {"task_id": "p1",
            "input": {"path": "x.csv", "output_path": "x.csv", "request": "r"},
            "files": {"x.csv": {"header": ["name", "quantity"], "rows": [["a", "1"]]}}}
    res = _run(task)
    assert tc.verify(task, res.trace), "precondition: the result itself is correct"
    v = judge.evaluate(res.trace, lambda t: tc.verify(task, t),
                       tc.reference_machine().prohibitions)
    assert v.verdict == "rejected"
    assert v.reason.endswith("P1")
    assert v.error_step is not None                # a rejected trace must carry the error location


def test_judged_stamps_verdict_onto_a_new_trace():
    task = tc.gen_tasks(1, seed=5)[0]
    res = _run(task)
    stamped = judge.judged(res.trace, lambda t: tc.verify(task, t),
                           tc.reference_machine().prohibitions)
    assert stamped.verdict == "accepted"
    assert res.trace.verdict == "unknown"          # the original trace is not modified
