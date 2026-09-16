"""⑫ 条款作废：失去轨迹支持的状态从机器里退掉，后续任务流让它被重新学出。"""

from hexis.legacy import compiler
from hexis.examples import table_clean as tc


def _clauses(machine) -> set[str]:
    return {s.clause for s in machine.states.values() if s.clause}


def test_unsupported_clause_has_no_state(accepted, max_fix):
    """只喂规范表头的轨迹（S3 修复从未被触发）——机器里不该有 S3 的状态。"""
    pool = accepted(60, seed=2)
    no_fix = [t for t in pool if max_fix(t) == 0]
    assert len(no_fix) >= 5
    cr = compiler.compile(tc.skill_doc(), no_fix, skill_id="tc",
                          prohibitions=tc.reference_machine().prohibitions)
    assert cr.findings == []
    assert "S3" not in _clauses(cr.machine)


def test_the_clause_is_relearned_when_traces_return(accepted, max_fix):
    """后续任务流带来修复轨迹，重编译后 S3 的状态与成环结构回来了。"""
    pool = accepted(60, seed=2)
    no_fix = [t for t in pool if max_fix(t) == 0]
    with_fix = no_fix + [t for t in pool if max_fix(t) > 0]
    cr = compiler.compile(tc.skill_doc(), with_fix, skill_id="tc",
                          prohibitions=tc.reference_machine().prohibitions)
    assert cr.findings == []
    assert "S3" in _clauses(cr.machine)
    assert any(t.inc for _s, t in cr.machine.transitions_all())   # 修复环也回来了
