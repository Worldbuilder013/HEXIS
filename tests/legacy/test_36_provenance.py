"""溯源、检查点、分裂、抢主干：多智能体编译加在守门程序上的四样东西。

* **溯源**（I8）：多智能体模式下 commit 时每个活结构都得有溯源行，缺一条就拒且回滚；
  溯源与机器同进退（回滚、回退一起恢复）；``origin`` 写错分栏当场拒。
* **检查点**：``mark``/``rewind`` 受票、出回执、不擦回执，不能越过上一次 commit。
* **分裂**：``split_state`` 按前驱分组克隆，每个前驱恰在一组，克隆 id 必须是新的。
* **抢主干**：两次无条件挂到同一源 ⇒ ``E_SPINE_TAKEN``；挂到指向 FALLBACK 的空位仍是主干生长。
* **未标定的引入判断**只能去 FALLBACK（I7）。
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
# 溯源
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
    # 一条没带 prov 的边：机器接受它，但 commit 不接受没有溯源的活结构
    assert ck.add_transition("s1", "s3", cond="x == 'skip'", support=3).accepted
    before = ck.machine.model_dump_json(by_alias=True)
    r = ck.commit(root=tmp_path)
    assert not r.accepted and "E_PROVENANCE_MISSING" in r.reason
    assert r.detail["rolled_back"] and r.detail["missing_provenance"] == \
        [prov_key("edge", "s1", "s3", "x == 'skip'")]
    assert ck.machine.model_dump_json(by_alias=True) != before      # 回滚到 open 时
    assert not (tmp_path / PROVENANCE_FILE).exists()


def test_single_agent_path_is_not_required_to_carry_provenance(tmp_path):
    ck = _open(require=False)
    _chain(ck, with_prov=False)
    assert ck.commit(root=tmp_path).accepted
    assert not (tmp_path / PROVENANCE_FILE).exists()               # 没有行就不写空文件


def test_a_row_with_a_bad_origin_is_refused_before_touching_the_machine():
    ck = _open()
    r = ck.add_state("s1", _tool("read", writes=["x"]), initial=True,
                     prov={"origin": "guess", "agent_id": "A1"})
    assert not r.accepted and "origin" in r.reason
    assert "s1" not in ck.machine.states


# --------------------------------------------------------------------------- #
# 检查点
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
    assert len(ck.receipts()) == n + 2                              # 加边 + 回退，都留痕


def test_rewind_cannot_cross_a_commit(tmp_path):
    ck = _open()
    _chain(ck)
    assert ck.mark("early").accepted
    assert ck.commit(root=tmp_path).accepted
    r = ck.rewind("early")
    assert not r.accepted and "E_REWIND_PAST_COMMIT" in r.reason


# --------------------------------------------------------------------------- #
# 分裂
# --------------------------------------------------------------------------- #
def _fork_into_x(ck: Checker) -> None:
    """s1 →(x=='a') p1 → X ；s1 →(默认) p2 → X ；X → END。X 有两个前驱。"""
    assert ck.add_state("s1", _tool("read", writes=["x"]), initial=True, prov=PROV).accepted
    assert ck.add_state("p1", _tool("left"), from_state="s1", from_cond="x == 'a'",
                        from_support=3, prov=PROV).accepted
    assert ck.add_state("p2", _tool("right"), from_state="s1", from_support=3,
                        prov=PROV).accepted
    assert ck.add_state("X", _tool("verify", writes=["v"]), from_state="p1",
                        from_support=3, prov=PROV).accepted
    # add_transition 不改写已有的兜底边（那是 add_state 接主干的特权），所以 p2→X 带条件
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
    assert not ck.split_state("s1", {"a": [], "b": []}, prov=PROV).accepted   # 起点
    assert "X" in ck.machine.states


# --------------------------------------------------------------------------- #
# 抢主干
# --------------------------------------------------------------------------- #
def test_spine_steal_is_refused_but_growing_into_the_fallback_slot_is_not():
    ck = _open()
    assert ck.add_state("s1", _tool("read", writes=["x"]), initial=True, prov=PROV).accepted
    # 空位（s1 的兜底 → FALLBACK）：接主干
    assert ck.add_state("s2", _tool("a"), from_state="s1", from_support=3, prov=PROV).accepted
    # 位子已被 s2 占了：再无条件挂一个 ⇒ 拒
    r = ck.add_state("s2b", _tool("b"), from_state="s1", from_support=3, prov=PROV)
    assert not r.accepted and "E_SPINE_TAKEN" in r.reason
    assert ck.machine.states["s1"].transitions[0].to == "s2"
    # 带条件就行
    assert ck.add_state("s2b", _tool("b"), from_state="s1", from_cond="x == 'b'",
                        from_support=3, prov=PROV).accepted


# --------------------------------------------------------------------------- #
# 未标定的引入判断
# --------------------------------------------------------------------------- #
def test_uncalibrated_introduced_judge_routes_only_to_fallback():
    ck = _open()
    assert ck.add_state("s1", _tool("read", writes=["x"]), initial=True, prov=PROV).accepted
    assert ck.set_terminal("s9", "END", from_state="s1", from_cond="x == 'done'",
                           from_support=3, prov=PROV).accepted
    # 引入的判断，没有打标器，却带标签边 ⇒ 拒
    r = ck.add_judge("j", "好吗？", ["x"], ["ok"], ["是", "否", "弃权"],
                     from_state="s1", from_support=3, introduced=True,
                     transitions=[Transition(cond="ok == '是'", to="s9"),
                                  Transition(to="FALLBACK")], prov=PROV)
    assert not r.accepted
    assert any(f["code"] == "E_JUDGE_UNCALIBRATED_BRANCH" for f in r.detail["findings"])
    # 只留兜底边 ⇒ 接受，但台账会点名
    r = ck.add_judge("j", "好吗？", ["x"], ["ok"], ["是", "否", "弃权"],
                     from_state="s1", from_support=3, introduced=True, prov=PROV)
    assert r.accepted, r.reason
    assert any(f["code"] == "W_JUDGE_UNCALIBRATED" for f in r.detail["findings"])
    # 有打标器且标定过（support>0）⇒ 标签边放行
    r = ck.add_judge("j", "好吗？", ["x"], ["ok"], ["是", "否", "弃权"],
                     introduced=True, gold_from="ok_from_trace", support=4,
                     transitions=[Transition(cond="ok == '是'", to="s9", support=3),
                                  Transition(to="FALLBACK")], prov=PROV)
    assert r.accepted, r.reason
