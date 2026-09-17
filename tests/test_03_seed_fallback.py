"""The initial machine is all fallback: the start goes straight into FALLBACK, and any accepted trace is trivially reproduced."""

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.machine.schema import FALLBACK, empty_machine


def test_empty_machine_starts_at_fallback():
    m = empty_machine("table-clean")
    assert m.initial == FALLBACK
    assert FALLBACK in m.states


def test_empty_machine_run_is_terminal_and_correct():
    task = tc.gen_tasks(1, seed=7)[0]
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(empty_machine("table-clean"), task,
                           model=tc.build_model(), tools=tc.build_registry(fs),
                           doc=tc.skill_doc())
    assert res.stopped == "terminal"
    assert tc.verify(task, res.trace)
