"""Split takes effect: a conflict from colliding with an existing state is taken apart, and each clone keeps only its own outgoing edge.

Merging by signature can squeeze two steps that "share a name but differ in meaning" into one
state, which then grows two outgoing edges that cannot be told apart. A split takes it back apart
into two states by predecessor (the two histories were never Myhill-Nerode equivalent), and the
conflict disappears.
"""

from hexis.legacy import compiler
from hexis.machine.schema import (
    EndAction,
    Machine,
    State,
    Terminal,
    ToolAction,
    Transition,
    Variable,
)


def _conflict_machine() -> Machine:
    """start routes to P1/P2 by x, both enter the same A; yet A requires two different continuations (-> B / -> C)."""
    return Machine(
        skill_id="conflict", initial="start",
        variables=[Variable(name="x", type="integer", init_from="task.input.x")],
        states={
            "start": State(id="start", action=ToolAction(name="route", writes=[]),
                           transitions=[Transition(cond="x == 1", to="P1"),
                                        Transition(to="P2")]),
            "P1": State(id="P1", action=ToolAction(name="p1", writes=[]),
                        transitions=[Transition(to="A")]),
            "P2": State(id="P2", action=ToolAction(name="p2", writes=[]),
                        transitions=[Transition(to="A")]),
            "A": State(id="A", action=ToolAction(name="shared", writes=[]),
                       transitions=[Transition(to="B"), Transition(to="C")]),
            "B": State(id="B", action=EndAction(terminal="done")),
            "C": State(id="C", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done")])


def test_split_takes_the_state_apart():
    m = _conflict_machine()
    assert len([1 for _s, t in m.transitions_all() if t.to == "A"]) == 2
    assert compiler.split_by_predecessor(m, "A")
    assert "A" not in m.states                               # the original state is taken apart
    assert "A_0" in m.states and "A_1" in m.states


def test_each_clone_keeps_exactly_one_outgoing_edge():
    m = _conflict_machine()
    compiler.split_by_predecessor(m, "A")
    outs0 = [t.to for t in m.states["A_0"].transitions]
    outs1 = [t.to for t in m.states["A_1"].transitions]
    assert len(outs0) == 1 and len(outs1) == 1
    assert set(outs0 + outs1) == {"B", "C"}                  # each continuation goes to its own clone


def test_predecessors_are_redirected_to_their_clone():
    m = _conflict_machine()
    compiler.split_by_predecessor(m, "A")
    targets = {src: t.to for src, t in m.transitions_all() if src in ("P1", "P2")}
    assert set(targets.values()) == {"A_0", "A_1"}           # one clone per predecessor


def test_split_resolves_the_nondeterminism():
    """Before the split, A's two unguarded outgoing edges violate mutual exclusion; after it, each clone has a single outgoing edge and the conflict is gone."""
    from hexis.machine.checks import structural_findings
    m = _conflict_machine()
    compiler.split_by_predecessor(m, "A")
    findings = structural_findings(m)
    assert not any("A_0" in f or "A_1" in f for f in findings)
