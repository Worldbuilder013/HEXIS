"""Under a simulated task stream, the fallback rate falls monotonically as observations accumulate.

The machine learns the loop bound K = ceil(loop_margin × max observed iterations) from traces. Early
on it has only seen tasks that need one repair round, so K is small, and a task that needs three
rounds fills the counter and drops into FALLBACK; the more it has seen, the larger K gets, and the
share of the same evaluation stream that ends up in FALLBACK goes down. This is what it looks like
when the first term of "violation rate = mass of not-yet-learned forms + abstention rate in the
learned region" shrinks along the task stream.
"""

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.legacy import compiler
from hexis.machine.schema import FALLBACK


def _fallback_rate(machine, tasks) -> float:
    hits = 0
    for task in tasks:
        fs = tc.MemFS(task["files"])
        res = runtime.run_task(machine, task, model=tc.build_model(),
                               tools=tc.build_registry(fs), doc=tc.skill_doc())
        if any(r.state == FALLBACK for r in res.trace.records):
            hits += 1
    return hits / len(tasks)


def test_fallback_rate_decreases_with_more_observed_traces(accepted, max_fix):
    pool = accepted(60, seed=2)
    eval_tasks = [t for t in tc.gen_tasks(30, seed=77)]      # fixed evaluation stream (includes tasks needing three repair rounds)
    prohibitions = tc.reference_machine().prohibitions

    rates = []
    for cap in (1, 2, 3):                                    # widen observations round by round: at most cap repair rounds seen
        train = [t for t in pool if max_fix(t) <= cap]
        cr = compiler.compile(tc.skill_doc(), train, skill_id="tc",
                              prohibitions=prohibitions)
        assert cr.findings == []
        rates.append(_fallback_rate(cr.machine, eval_tasks))

    assert rates == sorted(rates, reverse=True), rates       # monotonically non-increasing
    assert rates[-1] < rates[0], rates                       # and it really goes down
    assert rates[-1] == 0.0                                  # once fully learned, the evaluation stream never falls back
