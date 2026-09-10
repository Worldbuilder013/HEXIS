"""skill2fsm.fsm：初始化与更新算法的密闭回归（不调模型、不联网）。

用一份文件修改类的小技能做夹具：两个注册表工具（``run`` 跑命令、``put`` 写文件），一份
规则（修改后读产出 = verify 标签；已验证终点要求它成功）。这里检查算法本身：三值护卫、
事件化、对齐代价、候选构造、检查、回放、接受规则。技能无关性的回归在 test_46。
"""
from __future__ import annotations

import json

import pytest

from skill2fsm.fsm import check as C
from skill2fsm.fsm.align import align
from skill2fsm.fsm.common import maybe_true, takeable, truth
from skill2fsm.fsm.context import build_context, parse_rules
from skill2fsm.fsm.init import g_init, normalize
from skill2fsm.fsm.modify import Builder, build_candidate
from skill2fsm.fsm.traces import end_state_for, prepare, segment_trace
from skill2fsm.fsm.update import update
from skill2fsm.schema import Machine, Record, Trace
from skill2fsm.toolspec import ToolSpec

INPUTS = {"request": "r", "src": "/w/in.dat", "dst": "/w/out.dat"}
RULES = {
    "terminals": [{"id": "END_VERIFIED", "kind": "verified"}, {"id": "END_UNVERIFIED", "kind": "unverified"}],
    "labels": [{"label": "verify", "when": {"kind": "tool", "label": "probe", "args_contain": "${dst}",
                                             "after": {"label": "apply"}}, "quote": "reopen the output"}],
    "terminal_conditions": [{"terminal": "END_VERIFIED", "required_evidence": [{"label": "verify", "success": True}],
                             "invalidating_events": [{"label": "apply"}], "quote": "reopen the output"}],
}
DOC = "# Edit\nInspect first. Then reopen the output."

REGISTRY = {
    "run": ToolSpec(name="run", input_schema={"cmd": {"type": "string", "required": True},
                                              "note": {"type": "string", "constant": True}},
                    output_schema={"ok": "boolean", "code": "integer", "out": "string"},
                    success="code == 0", primary="out", source="registry"),
    "put": ToolSpec(name="put", input_schema={"path": {"type": "string", "required": True},
                                              "content": {"type": "string", "required": True}},
                    output_schema={"ok": "boolean", "code": "integer", "out": "string"},
                    success="code == 0", primary="out", label="apply", source="registry"),
}

MACHINE = {
    "format": "efsm-v1", "skill_id": "edit", "initial": "s1", "fallback": "FALLBACK", "max_steps": 48,
    "variables": [
        {"name": "request", "init_from": "task.input.request"}, {"name": "src", "init_from": "task.input.src"},
        {"name": "dst", "init_from": "task.input.dst"}, {"name": "inspect_cmd"}, {"name": "report"},
        {"name": "new_content"}, {"name": "check_cmd"}, {"name": "code", "type": "integer", "init": 0},
        {"name": "out"}, {"name": "repair_count", "type": "integer", "init": 0}],
    "terminals": [{"id": "END_VERIFIED", "kind": "verified"}, {"id": "END_UNVERIFIED", "kind": "unverified"},
                  {"id": "END_FALLBACK", "kind": "fallback"}],
    "states": {
        "s1": {"id": "s1", "origin": "document", "action": {"kind": "model", "prompt": "inspect cmd → {inspect_cmd}",
               "reads": ["request", "src"], "writes": ["inspect_cmd"]}, "transitions": [{"if": "", "to": "s2"}]},
        "s2": {"id": "s2", "origin": "document", "action": {"kind": "tool", "name": "run", "phase": "probe",
               "input": {"cmd": "${inspect_cmd}"}, "writes": ["code", "out", "report"], "binds": {"out": "report"}},
               "transitions": [{"if": "code == 0", "to": "s3"}, {"if": "", "to": "FALLBACK"}]},
        "s3": {"id": "s3", "origin": "document", "action": {"kind": "model", "prompt": "plan → {new_content, check_cmd}",
               "reads": ["request", "report", "dst"], "writes": ["new_content", "check_cmd"]},
               "transitions": [{"if": "repair_count >= 3", "to": "END_UNVERIFIED"}, {"if": "", "to": "s4"}]},
        "s4": {"id": "s4", "origin": "document", "action": {"kind": "tool", "name": "put", "phase": "apply",
               "input": {"path": "${dst}", "content": "${new_content}"}, "writes": ["code", "out"]},
               "transitions": [{"if": "code == 0", "to": "s5"}, {"if": "", "to": "s3", "inc": "repair_count"}]},
        "s5": {"id": "s5", "origin": "document", "action": {"kind": "tool", "name": "run", "phase": "probe",
               "labels": ["verify"], "input": {"cmd": "${check_cmd}"}, "writes": ["code", "out"]},
               "transitions": [{"if": "code == 0", "to": "END_VERIFIED"}, {"if": "", "to": "s3", "inc": "repair_count"}]},
        "END_VERIFIED": {"id": "END_VERIFIED", "origin": "document", "action": {"kind": "end", "terminal": "END_VERIFIED"}},
        "END_UNVERIFIED": {"id": "END_UNVERIFIED", "origin": "document", "action": {"kind": "end", "terminal": "END_UNVERIFIED"}},
        "FALLBACK": {"id": "FALLBACK", "origin": "compiler", "action": {"kind": "end", "terminal": "END_FALLBACK"}},
    },
}


def _run(step, cmd, code=0, out="x"):
    return Record(step=step, action={"kind": "tool", "name": "run", "input": {"cmd": cmd, "note": "n"}},
                  output={"ok": code == 0, "code": code, "out": out})


def _put(step, path="/w/out.dat", content="c", code=0):
    return Record(step=step, action={"kind": "tool", "name": "put", "input": {"path": path, "content": content}},
                  output={"ok": code == 0, "code": code, "out": ""})


def _model(step, text="thinking"):
    return Record(step=step, action={"kind": "model"}, output={"reply": text})


def _end(step, terminal="done"):
    return Record(step=step, action={"kind": "end", "terminal": terminal})


def make_trace(records, tid="t1", verdict="accepted", inputs=INPUTS):
    return Trace(task={"task_id": tid, "input": dict(inputs)}, verdict=verdict, records=records)


def happy_trace(tid="t1"):
    return make_trace([_run(1, "cat /w/in.dat"), _model(2), _put(3), _run(4, "cat /w/out.dat"), _end(5)], tid)


def context(*traces):
    return build_context("edit", DOC, [], list(traces), registry=REGISTRY, rules=parse_rules(RULES))


def machine():
    m = Machine.model_validate(json.loads(json.dumps(MACHINE)))
    normalize(m, context(happy_trace()))
    return m


# --------------------------------------------------------------------------- #
def test_fixture_passes_init_and_update_checks():
    ctx = context(happy_trace())
    m = machine()
    assert g_init(m, ctx) == []
    assert C.check(m, ctx) == []


def test_three_valued_guards_use_known_values_only():
    assert truth("code == 0", {"code": 0}) is True
    assert truth("code == 0", {"code": 1}) is False
    assert truth("cnt < 3 and code == 0", {"code": 0}) is None
    assert maybe_true("cnt >= 3", {"code": 0})
    assert truth("", {}) is True
    assert truth("ok == True", {"ok": False}) is False


def test_takeable_blocks_after_definite_edge():
    m = machine()
    assert [t.to for t in takeable(m.states["s5"], {"code": 0})] == ["END_VERIFIED"]
    assert [t.to for t in takeable(m.states["s5"], {"code": 1})] == ["s3"]


def test_events_merge_same_tool_and_label_and_keep_narration():
    ctx = context(happy_trace())
    tr = make_trace([_run(1, "ls"), _model(2, "look"), _run(3, "cat /w/in.dat"), _put(4), _run(5, "cat /w/out.dat"),
                     _model(6, "all good"), _end(7)])
    events, _notes, unsupported = segment_trace(tr, ctx)
    assert not unsupported
    obs = [e.describe() for e in events if e.observable]
    assert obs == ["run/probe×2✓", "put/apply×1✓", "run/probe×1✓", "model:output", "end:done"]
    assert "look" in events[0].intent


def test_derived_label_and_terminal_selection():
    ctx = context(happy_trace())
    p = prepare(happy_trace(), ctx)
    assert p.tool_events[-1].labels == {"verify"}
    assert p.tau == "END_VERIFIED"
    p2 = prepare(make_trace([_run(1, "cat /w/in.dat"), _put(2), _end(3)]), ctx)
    assert p2.tau == "END_UNVERIFIED" and p2.violation is None
    p3 = prepare(make_trace([_put(1), _end(2, "END_VERIFIED")]), ctx)
    assert p3.violation                                               # 声明已验证却没有证据


def test_end_state_selection_falls_back():
    m = machine()
    assert end_state_for(m, "END_VERIFIED") == "END_VERIFIED"
    del m.states["END_UNVERIFIED"]
    assert end_state_for(m, "END_UNVERIFIED") == "END_VERIFIED"


def test_alignment_matches_document_path_at_zero_cost():
    ctx = context(happy_trace())
    m = machine()
    al, why = align(m, prepare(happy_trace(), ctx), ctx)
    assert al is not None, why
    assert al.anchors() == ["s2", "s4", "s5", "END_VERIFIED"] and al.cost == 0


def test_alignment_charges_loop_only_when_graph_lacks_one():
    ctx = context(happy_trace())
    m = machine()
    tr = make_trace([_run(1, "cat /w/in.dat"), _put(2), _put(3), _run(4, "cat /w/out.dat"), _end(5)])
    al, _ = align(m, prepare(tr, ctx), ctx)
    assert al.cost == 1 and al.slots[1].loop and al.slots[1].state == "s4"


def test_alignment_prefers_tool_change_over_new_state():
    ctx = context(happy_trace())
    m = machine()
    tr = make_trace([_run(1, "cat /w/in.dat"),
                     Record(step=2, action={"kind": "tool", "name": "run", "input": {"cmd": "cp a b"}},
                            output={"ok": True, "code": 0, "out": ""}),
                     _run(3, "cat /w/out.dat"), _end(4)])
    p = prepare(tr, ctx)
    assert p.tool_events[1].label == "apply"                             # 通用命令按写操作分标签
    al, _ = align(m, p, ctx, allow_realize=True)
    assert al.slots[1].how == "realize" and al.slots[1].state == "s4" and al.cost == 1
    al2, _ = align(m, p, ctx, allow_realize=False)
    assert al2.slots[1].is_new and al2.cost == 4


def test_update_accepts_and_replays_protected_traces():
    a, b = happy_trace("a"), happy_trace("b")
    ctx = context(a, b)
    res = update(machine(), [("a.jsonl", a, ""), ("b.jsonl", b, "")], ctx)
    assert res.counts() == {"accepted": 2}
    assert C.check(res.machine, ctx) == []
    assert res.machine.states["s4"].transitions[0].support == 2


def test_update_adds_judge_for_conflicting_target_and_installs_counter():
    tr = make_trace([_run(1, "cat /w/in.dat"), _put(2), _run(3, "cat /w/out.dat"), _put(4, content="v2"),
                     _run(5, "cat /w/out.dat"), _end(6)])
    ctx = context(tr)
    res = update(machine(), [("t.jsonl", tr, "")], ctx)
    assert res.counts() == {"accepted": 1}, res.entries[0]
    m2 = res.machine
    assert any(s.action.kind == "judge" for s in m2.states.values())
    assert any(t.inc for _s, t in m2.transitions_all())
    assert C.check(m2, ctx) == []
    assert res.entries[0]["path"][-1] == "END_VERIFIED"


def test_trace_without_verification_is_accepted_into_unverified_terminal():
    tr = make_trace([_run(1, "cat /w/in.dat"), _put(2), _end(3)])
    ctx = context(tr)
    res = update(machine(), [("t.jsonl", tr, "")], ctx)
    assert res.counts() == {"accepted": 1}, res.entries[0]
    assert res.entries[0]["anchors"][-1] == "END_UNVERIFIED"


def test_read_only_trace_is_not_skipped():
    tr = make_trace([_run(1, "cat /w/in.dat"), _model(2, "nothing to change"), _end(3)])
    ctx = context(tr)
    res = update(machine(), [("t.jsonl", tr, "")], ctx)
    assert res.counts() == {"accepted": 1}, res.entries[0]


def test_loop_judge_for_repeated_calls():
    tr = make_trace([_run(1, "cat /w/in.dat"), _put(2), _put(3, content="v2"), _run(4, "cat /w/out.dat"), _end(5)])
    ctx = context(tr)
    res = update(machine(), [("t.jsonl", tr, "")], ctx)
    assert res.counts() == {"accepted": 1}, res.entries[0]
    assert "s4_loop" in res.machine.states
    assert C.check(res.machine, ctx) == []


def test_claimed_verified_without_evidence_is_violation():
    tr = make_trace([_run(1, "cat /w/in.dat"), _put(2), _end(3, "END_VERIFIED")])
    ctx = context(tr)
    res = update(machine(), [("t.jsonl", tr, "")], ctx)
    assert res.counts() == {"violation": 1}


def test_replay_rejects_wrong_event_sequence():
    ctx = context(happy_trace())
    m = machine()
    p = prepare(happy_trace(), ctx)
    assert not C.replay(m, p, ["s2", "s4", "s5", "END_UNVERIFIED"]).ok
    good = C.replay(m, p, ["s2", "s4", "s5", "END_VERIFIED"])
    assert good.ok and good.events[-1] == ("end", "END_VERIFIED", True)


def test_build_candidate_keeps_original_untouched():
    ctx = context(happy_trace())
    m = machine()
    before = m.model_dump_json()
    p = prepare(happy_trace(), ctx)
    al, _ = align(m, p, ctx)
    bd = build_candidate(m, p, al, ctx)
    assert bd.machine is not m and m.model_dump_json() == before


@pytest.mark.parametrize("name, args, generated", [
    ("run", {"cmd": "python x.py", "note": "constant text"}, {"cmd"}),
    ("put", {"path": "/w/out.dat", "content": "new\ntext"}, {"content"}),
    ("put", {"path": "out.dat", "content": "c"}, {"content"}),
])
def test_tool_template_from_registry_and_task_inputs(name, args, generated):
    from skill2fsm import runtime
    from skill2fsm.model_iface import ScriptedModel
    tr = make_trace([Record(step=1, action={"kind": "tool", "name": name, "input": args},
                            output={"ok": True, "code": 0, "out": "o"}), _end(2)])
    ctx = context(tr)
    prep = prepare(tr, ctx)
    ev = prep.tool_events[0]
    bd = Builder(machine(), prep, ctx)
    sid = bd.add_state(ev)
    st = bd.m.states[sid]
    assert st.action.name == name
    gate = bd.m.states.get(f"{sid}_gate")
    assert set(gate.action.writes) == {f"{sid}_{k}" for k in generated}
    values = dict(INPUTS)
    values.update({f"{sid}_{k}": args[k] for k in generated})
    filled = runtime.fill_template(st.action.input, values)
    assert filled == {**args, **({"path": "/w/out.dat"} if "path" in args else {})}

    class Tools:
        def __init__(self):
            self.calls = []

        def call(self, tool, inp):
            self.calls.append((tool, inp))
            return {"ok": True, "code": 0, "out": "native"}

    tools = Tools()
    _, _, err, _ = runtime._run_action(st, values, model=ScriptedModel(), tools=tools)
    assert not err and tools.calls[0][0] == name
    assert values["out"] == "native"


def test_unknown_tool_in_machine_is_an_init_error():
    ctx = context(happy_trace())
    m = machine()
    m.states["s2"].action.name = "mystery"
    assert any("mystery" in e for e in g_init(m, ctx))
