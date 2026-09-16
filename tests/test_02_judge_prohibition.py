"""② 评判正确：客观验收对，且触犯禁止性要求（覆盖原文件）即使结果对也判拒。"""

from hexis.traces import judge
from hexis.execution import runtime
from hexis.examples import table_clean as tc


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
    """output_path == path：结果表头照样规范（验收会过），但触犯 P1，必须判拒。"""
    task = {"task_id": "p1",
            "input": {"path": "x.csv", "output_path": "x.csv", "request": "r"},
            "files": {"x.csv": {"header": ["名称", "数量"], "rows": [["a", "1"]]}}}
    res = _run(task)
    assert tc.verify(task, res.trace), "前提：结果本身是对的"
    v = judge.evaluate(res.trace, lambda t: tc.verify(task, t),
                       tc.reference_machine().prohibitions)
    assert v.verdict == "rejected"
    assert v.reason.endswith("P1")
    assert v.error_step is not None                # 拒绝轨迹必带出错位置


def test_judged_stamps_verdict_onto_a_new_trace():
    task = tc.gen_tasks(1, seed=5)[0]
    res = _run(task)
    stamped = judge.judged(res.trace, lambda t: tc.verify(task, t),
                           tc.reference_machine().prohibitions)
    assert stamped.verdict == "accepted"
    assert res.trace.verdict == "unknown"          # 原 trace 不被改
