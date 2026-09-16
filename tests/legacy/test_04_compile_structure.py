"""Compilation learns the structure: read / check / split / repair loop / export, with guards learned at branches."""

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.legacy import compiler


def _compile(accepted):
    traces = accepted(24, seed=2)
    return compiler.compile(tc.skill_doc(), traces, skill_id="table-clean",
                            prohibitions=tc.reference_machine().prohibitions)


def test_learns_the_four_phases_with_clauses(accepted):
    cr = _compile(accepted)
    assert cr.findings == [], cr.findings
    by_clause = {(s.action.kind, s.clause) for s in cr.machine.states.values()}
    assert ("tool", "S1") in by_clause          # read
    assert ("judge", "S2.1") in by_clause        # header check
    assert ("tool", "S3") in by_clause           # repair
    assert ("tool", "S4") in by_clause           # export


def test_repair_forms_a_counted_loop(accepted):
    cr = _compile(accepted)
    back = [(src, t) for src, t in cr.machine.transitions_all() if t.inc]
    assert back, "repair should form a back edge with a counter"
    # the back edge's counter variable should have a bound exit
    for _src, t in back:
        tgt = cr.machine.states[t.to]
        assert any(g.cond and t.inc in __import__("hexis.machine.cond", fromlist=["vars_of"]).vars_of(g.cond)
                   for g in tgt.transitions)


def test_split_learns_a_condition_on_the_judge_output(accepted):
    cr = _compile(accepted)
    judge_state = next(s for s in cr.machine.states.values()
                       if s.action.kind == "judge")
    guarded = [t for t in judge_state.transitions if t.cond]
    assert len(guarded) >= 1
    conds = " ".join(t.cond for t in guarded)
    assert "header_ok" in conds                  # the branch guard is built on the judge output


def test_compiled_machine_runs_fresh_tasks(accepted):
    cr = _compile(accepted)
    ok = 0
    for task in tc.gen_tasks(20, seed=99):
        fs = tc.MemFS(task["files"])
        res = runtime.run_task(cr.machine, task, model=tc.build_model(),
                               tools=tc.build_registry(fs), doc=tc.skill_doc())
        if res.stopped == "terminal" and tc.verify(task, res.trace):
            ok += 1
    assert ok == 20                              # the learned machine generalizes to unseen tasks
