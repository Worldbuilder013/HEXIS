"""The initial machine is all fallback: the start goes straight into FALLBACK, and any accepted trace is trivially reproduced."""

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.legacy import replay
from hexis.machine.schema import FALLBACK, empty_machine


def test_empty_machine_starts_at_fallback():
    m = empty_machine("table-clean")
    assert m.initial == FALLBACK
    assert FALLBACK in m.states


def test_empty_machine_reproduces_any_trace():
    """The empty machine is in interpretation mode as soon as it enters FALLBACK, so replaying any trace on it trivially passes."""
    ref = tc.reference_machine()
    em = empty_machine("table-clean")
    for task in tc.gen_tasks(6, seed=7):
        fs = tc.MemFS(task["files"])
        # run a real machine to produce a real trace, then replay it on the empty machine
        res = runtime.run_task(ref, task, model=tc.build_model(),
                               tools=tc.build_registry(fs), doc=tc.skill_doc())
        assert replay.reproduces(em, res.trace)


def test_empty_machine_run_is_terminal_and_correct():
    task = tc.gen_tasks(1, seed=7)[0]
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(empty_machine("table-clean"), task,
                           model=tc.build_model(), tools=tc.build_registry(fs),
                           doc=tc.skill_doc())
    assert res.stopped == "terminal"
    assert tc.verify(task, res.trace)
