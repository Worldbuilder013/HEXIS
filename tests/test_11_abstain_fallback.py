"""When the judge abstains, execution is routed to FALLBACK and the task still completes.

When a judge action is unsure it outputs the abstain label; the machine then takes the default edge
into FALLBACK (interpretation) instead of forcing a label that may be wrong. This is the runtime
form of abstention acting as the control valve in "the probability that a path makes at least one
error is bounded by the sum of the judges' error rates".
"""

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.machine.schema import FALLBACK


def _run_with_abstaining_judge(task):
    """A model whose judge always abstains."""
    fs = tc.MemFS(task["files"])
    model = tc.build_model(abstain_on=lambda values: True)
    return runtime.run_task(tc.reference_machine(), task, model=model,
                            tools=tc.build_registry(fs), doc=tc.skill_doc())


def test_abstain_routes_into_fallback():
    # use a task with a well-formed header: the judge could answer "well_formed" directly, but abstention is forced here
    tasks = [t for t in tc.gen_tasks(12, seed=9)
             if tc.is_canonical(",".join(t["files"][t["input"]["path"]]["header"]))]
    task = tasks[0]
    res = _run_with_abstaining_judge(task)
    assert FALLBACK in res.path(), f"should enter FALLBACK after abstaining, got {res.path()}"


def test_task_still_completes_after_abstain():
    tasks = [t for t in tc.gen_tasks(12, seed=9)
             if tc.is_canonical(",".join(t["files"][t["input"]["path"]]["header"]))]
    task = tasks[0]
    res = _run_with_abstaining_judge(task)
    assert res.stopped == "terminal", res.error
    assert tc.verify(task, res.trace)              # abstention falls back to interpretation, and the task still completes
