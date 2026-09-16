"""Provenance, checkpoints, splitting, spine stealing: the four things multi-agent compilation adds to the gatekeeper.

* **Provenance** (invariant I8): in multi-agent mode every live structure must have a provenance row
  at commit, otherwise commit is refused and rolled back; provenance moves with the machine
  (restored together on rollback and rewind); an ``origin`` in the wrong column is rejected at once.
* **Checkpoints**: ``mark``/``rewind`` are receipted, never erase receipts, and cannot cross the last commit.
* **Splitting**: ``split_state`` clones by predecessor group, each predecessor in exactly one group, clone ids must be new.
* **Spine stealing**: attaching unconditionally to the same source twice ⇒ ``E_SPINE_TAKEN``; attaching into the empty slot pointing to FALLBACK is still spine growth.
* **Uncalibrated introduced judges** may only go to FALLBACK (invariant I7).
"""
from __future__ import annotations

import json

from hexis.legacy.checker import PROVENANCE_FILE, Checker, prov_key
from hexis.machine.schema import ToolAction, Transition, Variable

PROV = {"origin": "trace", "agent_id": "A1", "touchpoint_id": "", "clause": "S1",
        "locator": "SKILL.md:1", "support": 3}
COMP = {"origin": "compiler", "agent_id": "checker", "touchpoint_id": ""}


def _tool(name, **kw):
    return ToolAction(name=name, **kw)


def _open(require=True) -> Checker:
    ck = Checker("toy", require_provenance=require)
    r = ck.open_machine(variables=[Variable(name="x")],
                        terminals=[{"id": "END", "kind": ""}],
                        prov=PROV if require else None)
    assert r.accepted, r.reason
    return ck


def _chain(ck: Checker, *, with_prov=True) -> None:
    p = PROV if with_prov else None
    assert ck.add_state("s1", _tool("read", writes=["x"]), initial=True, prov=p).accepted
    assert ck.add_state("s2", _tool("work", reads=["x"], writes=["y"]),
                        from_state="s1", from_support=3, prov=p).accepted
    assert ck.set_terminal("s3", "END", from_state="s2", from_support=3, prov=p).accepted


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_every_live_key_has_a_row_at_commit(tmp_path):
    ck = _open()
    _chain(ck)
    assert ck.missing_provenance() == []
    r = ck.commit(root=tmp_path)
    assert r.accepted, r.reason
    rows = json.loads((tmp_path / PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert prov_key("state", "s2") in rows and rows[prov_key("state", "s2")]["agent_id"] == "A1"
    assert prov_key("edge", "s1", "s2", "") in rows
    assert prov_key("var", "x") in rows
    assert prov_key("terminal", "END") in rows


def test_commit_refuses_and_rolls_back_when_a_row_is_missing(tmp_path):
    ck = _open()
    _chain(ck)
    # an edge without prov: the machine accepts it, but commit does not accept live structures without provenance
    assert ck.add_transition("s1", "s3", cond="x == 'skip'", support=3).accepted
    before = ck.machine.model_dump_json(by_alias=True)
    r = ck.commit(root=tmp_path)
    assert not r.accepted and "E_PROVENANCE_MISSING" in r.reason
    assert r.detail["rolled_back"] and r.detail["missing_provenance"] == \
        [prov_key("edge", "s1", "s3", "x == 'skip'")]
    assert ck.machine.model_dump_json(by_alias=True) != before      # rolled back to the open state
    assert not (tmp_path / PROVENANCE_FILE).exists()


def test_single_agent_path_is_not_required_to_carry_provenance(tmp_path):
    ck = _open(require=False)
    _chain(ck, with_prov=False)
    assert ck.commit(root=tmp_path).accepted
    assert not (tmp_path / PROVENANCE_FILE).exists()               # no rows, so no empty file is written


def test_a_row_with_a_bad_origin_is_refused_before_touching_the_machine():
    ck = _open()
    r = ck.add_state("s1", _tool("read", writes=["x"]), initial=True,
                     prov={"origin": "guess", "agent_id": "A1"})
    assert not r.accepted and "origin" in r.reason
    assert "s1" not in ck.machine.states


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def test_rewind_restores_machine_and_provenance_but_keeps_receipts():
    ck = _open()
    _chain(ck)
    assert ck.mark("inc").accepted
    snap = ck.machine.model_dump_json(by_alias=True)
    prov_snap = ck.provenance()
    n = len(ck.receipts())
    assert ck.add_transition("s1", "s3", cond="x == 'skip'", support=3, prov=PROV).accepted
    assert ck.machine.model_dump_json(by_alias=True) != snap
    r = ck.rewind("inc")
    assert r.accepted
    assert ck.machine.model_dump_json(by_alias=True) == snap
    assert ck.provenance() == prov_snap
    assert len(ck.receipts()) == n + 2                              # add edge + rewind, both recorded


def test_rewind_cannot_cross_a_commit(tmp_path):
    ck = _open()
    _chain(ck)
    assert ck.mark("early").accepted
    assert ck.commit(root=tmp_path).accepted
    r = ck.rewind("early")
    assert not r.accepted and "E_REWIND_PAST_COMMIT" in r.reason


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _fork_into_x(ck: Checker) -> None:
    """s1 →(x=='a') p1 → X; s1 →(default) p2 → X; X → END. X has two predecessors."""
    assert ck.add_state("s1", _tool("read", writes=["x"]), initial=True, prov=PROV).accepted
    assert ck.add_state("p1", _tool("left"), from_state="s1", from_cond="x == 'a'",
                        from_support=3, prov=PROV).accepted
    assert ck.add_state("p2", _tool("right"), from_state="s1", from_support=3,
                        prov=PROV).accepted
    assert ck.add_state("X", _tool("verify", writes=["v"]), from_state="p1",
                        from_support=3, prov=PROV).accepted
    # add_transition does not rewrite an existing default edge (that is add_state's privilege when extending the spine), so p2→X has a guard
    assert ck.add_transition("p2", "X", cond="x == 'b'", support=3, prov=PROV).accepted
    assert ck.set_terminal("s3", "END", from_state="X", from_support=3, prov=PROV).accepted


def test_split_state_clones_by_predecessor_group():
    ck = _open()
    _fork_into_x(ck)
    r = ck.split_state("X", {"X1": ["p1"], "X2": ["p2"]}, prov=PROV)
    assert r.accepted, r.reason
    m = ck.machine
    assert "X" not in m.states and {"X1", "X2"} <= set(m.states)
    assert m.states["p1"].transitions[0].to == "X1"
    assert [t.to for t in m.states["p2"].transitions if t.cond] == ["X2"]
    assert m.states["X1"].action == m.states["X2"].action
    assert prov_key("state", "X1") in ck.provenance()


def test_split_state_refuses_unassigned_and_unknown_predecessors():
    ck = _open()
    _fork_into_x(ck)
    assert not ck.split_state("X", {"X1": ["p1"], "X2": []}, prov=PROV).accepted
    assert not ck.split_state("X", {"X1": ["p1"], "X2": ["nope"]}, prov=PROV).accepted
    assert not ck.split_state("s1", {"a": [], "b": []}, prov=PROV).accepted   # initial state
    assert "X" in ck.machine.states


# --------------------------------------------------------------------------- #
# Spine stealing
# --------------------------------------------------------------------------- #
def test_spine_steal_is_refused_but_growing_into_the_fallback_slot_is_not():
    ck = _open()
    assert ck.add_state("s1", _tool("read", writes=["x"]), initial=True, prov=PROV).accepted
    # empty slot (s1's default → FALLBACK): extend the spine
    assert ck.add_state("s2", _tool("a"), from_state="s1", from_support=3, prov=PROV).accepted
    # the slot is taken by s2: attaching another one unconditionally ⇒ rejected
    r = ck.add_state("s2b", _tool("b"), from_state="s1", from_support=3, prov=PROV)
    assert not r.accepted and "E_SPINE_TAKEN" in r.reason
    assert ck.machine.states["s1"].transitions[0].to == "s2"
    # with a guard it is fine
    assert ck.add_state("s2b", _tool("b"), from_state="s1", from_cond="x == 'b'",
                        from_support=3, prov=PROV).accepted


# --------------------------------------------------------------------------- #
# Uncalibrated introduced judges
# --------------------------------------------------------------------------- #
def test_uncalibrated_introduced_judge_routes_only_to_fallback():
    ck = _open()
    assert ck.add_state("s1", _tool("read", writes=["x"]), initial=True, prov=PROV).accepted
    assert ck.set_terminal("s9", "END", from_state="s1", from_cond="x == 'done'",
                           from_support=3, prov=PROV).accepted
    # introduced judge without a labeler but with labeled edges ⇒ rejected
    r = ck.add_judge("j", "Is it OK?", ["x"], ["ok"], ["yes", "no", "abstain"],
                     from_state="s1", from_support=3, introduced=True,
                     transitions=[Transition(cond="ok == 'yes'", to="s9"),
                                  Transition(to="FALLBACK")], prov=PROV)
    assert not r.accepted
    assert any(f["code"] == "E_JUDGE_UNCALIBRATED_BRANCH" for f in r.detail["findings"])
    # only a default edge ⇒ accepted, but the ledger calls it out
    r = ck.add_judge("j", "Is it OK?", ["x"], ["ok"], ["yes", "no", "abstain"],
                     from_state="s1", from_support=3, introduced=True, prov=PROV)
    assert r.accepted, r.reason
    assert any(f["code"] == "W_JUDGE_UNCALIBRATED" for f in r.detail["findings"])
    # with a labeler and calibrated (support>0) ⇒ labeled edges are allowed
    r = ck.add_judge("j", "Is it OK?", ["x"], ["ok"], ["yes", "no", "abstain"],
                     introduced=True, gold_from="ok_from_trace", support=4,
                     transitions=[Transition(cond="ok == 'yes'", to="s9", support=3),
                                  Transition(to="FALLBACK")], prov=PROV)
    assert r.accepted, r.reason
