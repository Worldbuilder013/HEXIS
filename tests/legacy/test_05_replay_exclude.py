"""Counterexamples hold: all accepted traces are reproduced, all rejected traces are excluded; held-out traces are reproduced too (generalization)."""

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.legacy import compiler, replay
from hexis.traces import judge


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
    held = accepted(16, seed=99)               # accepted traces from brand-new tasks
    assert held
    assert all(replay.reproduces(cr.machine, t) for t in held)


def test_excludes_prohibition_violating_trace(accepted):
    cr = _compile(accepted)
    # a rejected trace that overwrites the source file: structurally identical to a normal run, stopped by the machine's P1
    task = {"task_id": "neg",
            "input": {"path": "x.csv", "output_path": "x.csv", "request": "r"},
            "files": {"x.csv": {"header": ["name", "quantity"], "rows": [["a", "1"]]}}}
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(tc.reference_machine(), task, model=tc.build_model(),
                           tools=tc.build_registry(fs), doc=tc.skill_doc())
    neg = judge.judged(res.trace, (lambda t: tc.verify(task, t)),
                       tc.reference_machine().prohibitions)
    assert neg.verdict == "rejected"
    assert replay.excludes(cr.machine, neg)


def test_empty_machine_excludes_nothing(accepted):
    """An all-fallback machine reproduces everything and excludes no counterexample, which is exactly why T- exists."""
    from hexis.machine.schema import empty_machine
    em = empty_machine("table-clean")
    task = {"task_id": "neg",
            "input": {"path": "x.csv", "output_path": "x.csv", "request": "r"},
            "files": {"x.csv": {"header": ["name", "quantity"], "rows": [["a", "1"]]}}}
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(tc.reference_machine(), task, model=tc.build_model(),
                           tools=tc.build_registry(fs), doc=tc.skill_doc())
    neg = judge.judged(res.trace, (lambda t: tc.verify(task, t)),
                       tc.reference_machine().prohibitions)
    assert not replay.excludes(em, neg)        # the empty machine has no prohibitions and never deviates structurally
