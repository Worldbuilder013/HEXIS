"""Clause retirement: a state that loses trace support is retired from the machine, and a later task stream lets it be learned again."""

from hexis.examples import table_clean as tc
from hexis.legacy import compiler


def _clauses(machine) -> set[str]:
    return {s.clause for s in machine.states.values() if s.clause}


def test_unsupported_clause_has_no_state(accepted, max_fix):
    """Feed only traces with well-formed headers (S3 repair is never triggered): the machine should have no S3 state."""
    pool = accepted(60, seed=2)
    no_fix = [t for t in pool if max_fix(t) == 0]
    assert len(no_fix) >= 5
    cr = compiler.compile(tc.skill_doc(), no_fix, skill_id="tc",
                          prohibitions=tc.reference_machine().prohibitions)
    assert cr.findings == []
    assert "S3" not in _clauses(cr.machine)


def test_the_clause_is_relearned_when_traces_return(accepted, max_fix):
    """A later task stream brings repair traces; after recompiling, the S3 state and the loop structure come back."""
    pool = accepted(60, seed=2)
    no_fix = [t for t in pool if max_fix(t) == 0]
    with_fix = no_fix + [t for t in pool if max_fix(t) > 0]
    cr = compiler.compile(tc.skill_doc(), with_fix, skill_id="tc",
                          prohibitions=tc.reference_machine().prohibitions)
    assert cr.findings == []
    assert "S3" in _clauses(cr.machine)
    assert any(t.inc for _s, t in cr.machine.transitions_all())   # the repair loop is back too
