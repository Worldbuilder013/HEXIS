"""状态身份可以往回看、回放看不见它；开局工具让机器只有一个起点。

* ``context_key`` 把严格档 KEY 合并掉的「同一动作、不同前驱」分开，``k=0`` 退化成严格档；
* ``Checker.split_state`` 分裂前后，所有 T+ 的回放结果**逐条相同**——克隆的动作是深拷贝，
  ``replay._action_matches`` 只比动作不比身份（test_08 的结论，这里对受票版本再钉一次）；
* ``split_groups`` 按列联表分组：每个前驱只去过一个出口才分得开；
* ``compile_skill(begin=True)`` 让开局动作不同的轨迹不再 ``start_mismatch``，机器起点恒为
  BEGIN_TOOL；回放对没垫开局步的原轨迹透明，且报的下标按**原轨迹**计。
"""
from __future__ import annotations

from hexis.legacy import compile_agent, compiler, replay
from hexis.legacy.checker import Checker
from hexis.traces.normalize import BEGIN_TOOL, canon_action, context_key, is_begin
from hexis.machine.schema import Record, ToolAction, Trace, Transition, Variable
from hexis.traces.trace_adapter import with_begin

PROV = {"origin": "trace", "agent_id": "A1"}


def _tool(step, name, **vars_):
    # 回放按各状态的 writes 白名单从 output 取值，所以 output 也带上这一步的变量
    return Record(step=step, action={"kind": "tool", "name": name, "input": {}},
                  output=dict(vars_), vars=dict(vars_))


def _end(step, **vars_):
    return Record(step=step, action={"kind": "end", "terminal": "done"}, vars=dict(vars_))


# --------------------------------------------------------------------------- #
def test_context_key_separates_what_strict_key_merged():
    verify = {"kind": "tool", "name": "math_verify", "input": {"argv": ["a"]}}
    after_python = {"kind": "tool", "name": "run_python"}
    after_model = {"kind": "model", "prompt": "think"}
    assert context_key(verify, [after_python], k=0) == canon_action(verify, strict=True)
    assert context_key(verify, [after_python]) != context_key(verify, [after_model])
    assert context_key(verify, [after_python])[:2] == canon_action(verify, strict=True)
    # 同一前驱、参数不同的同名工具：身份仍相同（参数活在变量里）
    verify2 = {"kind": "tool", "name": "math_verify", "input": {"argv": ["b"]}}
    assert context_key(verify, [after_python]) == context_key(verify2, [after_python])


def test_split_groups_by_contingency_and_by_position():
    c = {"p1": {"A": 3}, "p2": {"B": 2}, "p3": {"A": 1}}
    assert compiler.split_groups(["p1", "p2", "p3"], ["A", "B"], c) == \
        {"A": ["p1", "p3"], "B": ["p2"]}
    mixed = {"p1": {"A": 3, "B": 1}, "p2": {"B": 2}}
    assert compiler.split_groups(["p1", "p2"], ["A", "B"], mixed) is None     # p1 两头都去过
    same = {"p1": {"A": 3}, "p2": {"A": 2}}
    assert compiler.split_groups(["p1", "p2"], ["A", "B"], same) is None      # 没什么可分
    assert compiler.split_groups(["p1", "p2"], ["A", "B"]) == {"A": ["p1"], "B": ["p2"]}
    assert compiler.split_groups(["p1", "p2", "p3"], ["A", "B"]) is None


# --------------------------------------------------------------------------- #
def _fork_machine() -> Checker:
    """s1 read(x) →(x=='a') p1 left → X verify ；→(默认) p2 right → X ；X → END。"""
    ck = Checker("toy")
    assert ck.open_machine(variables=[Variable(name="x")],
                           terminals=[{"id": "done"}]).accepted
    assert ck.add_state("s1", ToolAction(name="read", writes=["x"]), initial=True).accepted
    assert ck.add_state("p1", ToolAction(name="left"), from_state="s1",
                        from_cond="x == 'a'", from_support=3).accepted
    assert ck.add_state("p2", ToolAction(name="right"), from_state="s1",
                        from_support=3).accepted
    assert ck.add_state("X", ToolAction(name="verify", writes=["v"]), from_state="p1",
                        from_support=3).accepted
    assert ck.add_transition("p2", "X", cond="x == 'b'", support=3).accepted
    assert ck.set_terminal("E", "done", from_state="X", from_support=3).accepted
    return ck


def _fork_traces() -> list[Trace]:
    ta = Trace(records=[_tool(1, "read", x="a"), _tool(2, "left", x="a"),
                        _tool(3, "verify", x="a", v="ok"), _end(4, x="a", v="ok")])
    tb = Trace(records=[_tool(1, "read", x="b"), _tool(2, "right", x="b"),
                        _tool(3, "verify", x="b", v="ok"), _end(4, x="b", v="ok")])
    bad = Trace(records=[_tool(1, "read", x="a"), _tool(2, "right", x="a")])
    return [ta, tb, bad]


def test_split_state_preserves_replay_of_all_traces():
    ck = _fork_machine()
    before = ck.machine
    traces = _fork_traces()
    res_before = [replay.replay(before, t) for t in traces]
    assert [r.ok for r in res_before] == [True, True, False]

    r = ck.split_state("X", {"X1": ["p1"], "X2": ["p2"]}, prov=PROV)
    assert r.accepted, r.reason
    after = ck.machine
    res_after = [replay.replay(after, t) for t in traces]
    assert [(r.ok, r.diverged_at) for r in res_after] == \
        [(r.ok, r.diverged_at) for r in res_before]
    # 分裂真的发生了：两份克隆各自只从自己的前驱进
    assert "X" not in after.states and {"X1", "X2"} <= set(after.states)


# --------------------------------------------------------------------------- #
DOC = "# toy\n\n## S1 Do\nRead, then act, then verify, then finish.\n"


def _open_traces() -> list[Trace]:
    """三种开局动作各两条（支持度过 min_support），之后都 verify → end。"""
    out = []
    for first in ("run_python", "math_verify", "read_reference"):
        for k in range(2):
            out.append(Trace(
                task={"task_id": f"{first}-{k}", "input": {"problem": "p"}},
                verdict="accepted",
                records=[_tool(1, first, verify_status=""),
                         _tool(2, "math_verify", verify_status="PASS"),
                         _tool(3, "submit_answer", verify_status="PASS", answer="4"),
                         _end(4, verify_status="PASS", answer="4")]))
    return out


def test_begin_gives_one_initial_state_and_no_start_mismatch():
    traces = _open_traces()
    plain = compile_agent.compile_skill(DOC, traces, model=None)
    assert any(n.get("kind") == "start_mismatch" for n in plain.coverage["notes"])

    res = compile_agent.compile_skill(DOC, traces, model=None, begin=True)
    assert not any(n.get("kind") == "start_mismatch" for n in res.coverage["notes"])
    m = res.machine
    assert is_begin(m.states[m.initial].action)
    assert m.states[m.initial].action.name == BEGIN_TOOL
    # 回放对没垫开局步的原轨迹透明
    assert all(replay.replay(m, t).ok for t in traces)


def test_with_begin_is_idempotent_and_keeps_real_steps():
    t = _open_traces()[0]
    v = with_begin(t)
    assert is_begin(v.records[0].action) and v.records[0].step == 0
    assert [r.step for r in v.records[1:]] == [r.step for r in t.records]
    assert with_begin(v).records == v.records
    assert t.records[0].action["name"] == "run_python"           # 原轨迹一个字节不动


def test_walk_reports_indices_of_the_original_trace():
    """机器起点是开局工具、轨迹没垫：偏离下标按原轨迹计，不多 1。"""
    ck = Checker("toy")
    assert ck.open_machine(terminals=[{"id": "done"}]).accepted
    assert ck.add_state("b", ToolAction(name=BEGIN_TOOL), initial=True).accepted
    assert ck.add_state("s1", ToolAction(name="read", writes=["x"]),
                        from_state="b", from_support=3).accepted
    assert ck.add_state("s2", ToolAction(name="left"), from_state="s1",
                        from_support=3).accepted
    m = ck.machine
    good = Trace(records=[_tool(1, "read", x="a"), _tool(2, "left", x="a")])
    bad = Trace(records=[_tool(1, "read", x="a"), _tool(2, "right", x="a")])
    assert replay.replay(m, good).ok
    r = replay.walk(m, bad)
    assert not r.ok and r.diverged_at == 1
    assert [i for i, _s in r.seq] == [-1, 0]                      # 虚拟开局步记 -1
