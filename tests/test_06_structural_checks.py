"""⑥ Structural checks: guards are mutually exclusive and complete, repair loops are bounded, variables are written before read, and broken machines are reported one by one."""

from hexis.examples import table_clean as tc
from hexis.machine import checks
from hexis.machine.schema import EndAction, State, Transition, empty_machine


def test_reference_and_empty_machines_are_clean():
    assert checks.structural_findings(tc.reference_machine()) == []
    assert checks.structural_findings(empty_machine("table-clean")) == []


def test_missing_fallback_edge_is_flagged():
    m = tc.reference_machine()
    m.states["s2"].transitions = [t for t in m.states["s2"].transitions if t.cond]
    f = checks.structural_findings(m)
    assert any("default edge" in x or "gap" in x for x in f)


def test_overlapping_conditions_are_flagged():
    m = tc.reference_machine()
    m.states["s2"].transitions = [
        Transition(**{"if": "header_ok == 'well_formed'", "to": "s4"}),
        Transition(**{"if": "header_ok == 'well_formed'", "to": "s3"}),  # overlaps the previous one
        Transition(to="FALLBACK"),
    ]
    assert any("overlapping" in x for x in checks.structural_findings(m))


def test_uncounted_loop_is_flagged():
    m = tc.reference_machine()
    m.states["s3"].transitions = [Transition(to="s2")]           # back edge lost its inc
    assert any("inc" in x for x in checks.structural_findings(m))


def test_read_before_write_is_flagged():
    m = tc.reference_machine()
    m.states["s4"].action.reads = m.states["s4"].action.reads + ["ghost"]
    assert any("ghost" in x for x in checks.structural_findings(m))


def test_unreachable_state_is_flagged():
    m = tc.reference_machine()
    m.states["orphan"] = State(id="orphan", action=EndAction(terminal="done"))
    assert any("unreachable" in x for x in checks.structural_findings(m))


def test_mutual_exclusion_holds_on_reference_judge():
    """On the reference machine, at most one of s2's three outgoing edges holds under every (header_ok, fix_count) configuration."""
    m = tc.reference_machine()
    det = [x for x in checks.structural_findings(m) if "overlapping" in x or "gap" in x]
    assert det == []
