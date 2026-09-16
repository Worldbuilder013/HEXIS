"""⑤ 反例成立：全接受轨迹被复述、全拒绝轨迹被排除；held-out 也复述（泛化）。"""

from hexis.legacy import compiler, replay
from hexis.traces import judge
from hexis.execution import runtime
from hexis.examples import table_clean as tc


def _compile(accepted):
    return compiler.compile(tc.skill_doc(), accepted(24, seed=2),
                            skill_id="table-clean",
                            prohibitions=tc.reference_machine().prohibitions)


def test_reproduces_all_positive(accepted):
    cr = _compile(accepted)
    train = accepted(24, seed=2)
    assert all(replay.reproduces(cr.machine, t) for t in train)


def test_holdout_generalizes(accepted):
    cr = _compile(accepted)
    held = accepted(16, seed=99)               # 全新任务跑出的接受轨迹
    assert held
    assert all(replay.reproduces(cr.machine, t) for t in held)


def test_excludes_prohibition_violating_trace(accepted):
    cr = _compile(accepted)
    # 覆盖原文件的拒绝轨迹：结构上和正常一样，靠机器带的 P1 拦
    task = {"task_id": "neg",
            "input": {"path": "x.csv", "output_path": "x.csv", "request": "r"},
            "files": {"x.csv": {"header": ["名称", "数量"], "rows": [["a", "1"]]}}}
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(tc.reference_machine(), task, model=tc.build_model(),
                           tools=tc.build_registry(fs), doc=tc.skill_doc())
    neg = judge.judged(res.trace, (lambda t: tc.verify(task, t)),
                       tc.reference_machine().prohibitions)
    assert neg.verdict == "rejected"
    assert replay.excludes(cr.machine, neg)


def test_empty_machine_excludes_nothing(accepted):
    """全回退机器复述一切、排除不了任何反例——正是 T- 存在的理由。"""
    from hexis.machine.schema import empty_machine
    em = empty_machine("table-clean")
    task = {"task_id": "neg",
            "input": {"path": "x.csv", "output_path": "x.csv", "request": "r"},
            "files": {"x.csv": {"header": ["名称", "数量"], "rows": [["a", "1"]]}}}
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(tc.reference_machine(), task, model=tc.build_model(),
                           tools=tc.build_registry(fs), doc=tc.skill_doc())
    neg = judge.judged(res.trace, (lambda t: tc.verify(task, t)),
                       tc.reference_machine().prohibitions)
    assert not replay.excludes(em, neg)        # 空机器无 prohibitions、结构不偏离
