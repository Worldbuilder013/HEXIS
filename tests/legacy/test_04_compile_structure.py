"""④ 编译学出结构：读入/检查/分裂/修复成环/导出，分岔处学出条件。"""

from hexis.legacy import compiler
from hexis.execution import runtime
from hexis.examples import table_clean as tc


def _compile(accepted):
    traces = accepted(24, seed=2)
    return compiler.compile(tc.skill_doc(), traces, skill_id="table-clean",
                            prohibitions=tc.reference_machine().prohibitions)


def test_learns_the_four_phases_with_clauses(accepted):
    cr = _compile(accepted)
    assert cr.findings == [], cr.findings
    by_clause = {(s.action.kind, s.clause) for s in cr.machine.states.values()}
    assert ("tool", "S1") in by_clause          # 读取
    assert ("judge", "S2.1") in by_clause        # 表头检查
    assert ("tool", "S3") in by_clause           # 修复
    assert ("tool", "S4") in by_clause           # 导出


def test_repair_forms_a_counted_loop(accepted):
    cr = _compile(accepted)
    back = [(src, t) for src, t in cr.machine.transitions_all() if t.inc]
    assert back, "修复应形成一条带计数的回边"
    # 回边的计数变量应有上限出口
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
    assert "header_ok" in conds                  # 分岔条件建立在判断输出上


def test_compiled_machine_runs_fresh_tasks(accepted):
    cr = _compile(accepted)
    ok = 0
    for task in tc.gen_tasks(20, seed=99):
        fs = tc.MemFS(task["files"])
        res = runtime.run_task(cr.machine, task, model=tc.build_model(),
                               tools=tc.build_registry(fs), doc=tc.skill_doc())
        if res.stopped == "terminal" and tc.verify(task, res.trace):
            ok += 1
    assert ok == 20                              # 学出的机器泛化到没见过的任务
