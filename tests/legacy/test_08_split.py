"""⑧ 分裂生效：撞既有状态的冲突被拆开，各克隆只保留自己那条出边。

按签名合并会把「名字相同、语义不同」的两步压进一个状态——它随后长出两条分不开的出边。
分裂按前驱把它拆回两个状态（这两段历史本就不 Myhill-Nerode 等价），冲突消除。
"""

from hexis.legacy import compiler
from hexis.machine.schema import (
    EndAction, Machine, State, Terminal, ToolAction, Transition, Variable,
)


def _conflict_machine() -> Machine:
    """start 按 x 分流到 P1/P2，两者都进同一个 A；A 却要求两种后续（→B / →C）。"""
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
    assert "A" not in m.states                               # 原状态被拆掉
    assert "A_0" in m.states and "A_1" in m.states


def test_each_clone_keeps_exactly_one_outgoing_edge():
    m = _conflict_machine()
    compiler.split_by_predecessor(m, "A")
    outs0 = [t.to for t in m.states["A_0"].transitions]
    outs1 = [t.to for t in m.states["A_1"].transitions]
    assert len(outs0) == 1 and len(outs1) == 1
    assert set(outs0 + outs1) == {"B", "C"}                  # 两条后续各归各


def test_predecessors_are_redirected_to_their_clone():
    m = _conflict_machine()
    compiler.split_by_predecessor(m, "A")
    targets = {src: t.to for src, t in m.transitions_all() if src in ("P1", "P2")}
    assert set(targets.values()) == {"A_0", "A_1"}           # 一前驱配一克隆


def test_split_resolves_the_nondeterminism():
    """拆开前 A 的两条无条件出边违互斥；拆开后每个克隆单出边，冲突消失。"""
    from hexis.machine.checks import structural_findings
    m = _conflict_machine()
    compiler.split_by_predecessor(m, "A")
    findings = structural_findings(m)
    assert not any("A_0" in f or "A_1" in f for f in findings)
