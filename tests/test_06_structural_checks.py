"""⑥ 结构检查：条件互斥完备、修复环带上限、变量先写后读，坏机器逐条报出。"""

from skill2fsm import checks
from skill2fsm.examples import table_clean as tc
from skill2fsm.schema import EndAction, State, Transition, empty_machine


def test_reference_and_empty_machines_are_clean():
    assert checks.structural_findings(tc.reference_machine()) == []
    assert checks.structural_findings(empty_machine("table-clean")) == []


def test_missing_fallback_edge_is_flagged():
    m = tc.reference_machine()
    m.states["s2"].transitions = [t for t in m.states["s2"].transitions if t.cond]
    f = checks.structural_findings(m)
    assert any("兜底" in x or "空隙" in x for x in f)


def test_overlapping_conditions_are_flagged():
    m = tc.reference_machine()
    m.states["s2"].transitions = [
        Transition(**{"if": "header_ok == '规范'", "to": "s4"}),
        Transition(**{"if": "header_ok == '规范'", "to": "s3"}),  # 与上条重叠
        Transition(to="FALLBACK"),
    ]
    assert any("重叠" in x for x in checks.structural_findings(m))


def test_uncounted_loop_is_flagged():
    m = tc.reference_machine()
    m.states["s3"].transitions = [Transition(to="s2")]           # 回边丢了 inc
    assert any("inc" in x for x in checks.structural_findings(m))


def test_read_before_write_is_flagged():
    m = tc.reference_machine()
    m.states["s4"].action.reads = m.states["s4"].action.reads + ["ghost"]
    assert any("ghost" in x for x in checks.structural_findings(m))


def test_unreachable_state_is_flagged():
    m = tc.reference_machine()
    m.states["orphan"] = State(id="orphan", action=EndAction(terminal="done"))
    assert any("走不到" in x for x in checks.structural_findings(m))


def test_mutual_exclusion_holds_on_reference_judge():
    """参考机器 s2 的三条出边在所有 (header_ok, fix_count) 格局下至多一条成立。"""
    m = tc.reference_machine()
    det = [x for x in checks.structural_findings(m) if "重叠" in x or "空隙" in x]
    assert det == []
