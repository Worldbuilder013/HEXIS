"""阶段分类：给「同一个通用工具、不同用途」的步骤一个可判定的身份。**技能无关。**

不加这一层，``bash``/``run_python`` 的三步（查表 / 写公式 / 回头核对）全折成一个
``('tool','name=run_python')``，整条轨迹变成一个状态自环三次——信息全在命令文本里，
而命令文本按设计不进 KEY。

这些测试钉住四件事：判据本身、判据的**顺序**（存盘要先于产出路径判）、阶段在采集时算出
并随记录走（编译时正文已被抽到 artifacts）、以及两侧对称——机器状态也带 ``phase``，
所以回放不需要任何特殊处理，没有阶段的动作行为逐字照旧。
"""
from __future__ import annotations

from skill2fsm import phases
from skill2fsm.normalize import canon_action
from skill2fsm.replay import replay
from skill2fsm.schema import (
    EndAction, Machine, State, ToolAction, Trace, Record, Transition, Variable,
)
from skill2fsm.trace_adapter import RawRun, RawStep, to_trace


def _py(code: str) -> RawStep:
    return RawStep(kind="tool", name="run_python", args={"code": code})


# --------------------------------------------------------------------------- #
# 判据
# --------------------------------------------------------------------------- #
OUT = ("output_path", "/tmp/out.xlsx")


def test_three_phases_on_generic_tools():
    f = phases.default_phase
    assert f("run_python", {"code": "wb = load_workbook(src); print(wb.active.max_row)"},
             outputs=OUT) == phases.PHASE_PROBE
    assert f("run_python", {"code": 'ws["C1"] = "=A1&B1"; wb.save(output_path)'},
             outputs=OUT) == phases.PHASE_APPLY
    assert f("run_python", {"code": "w = load_workbook(output_path); print(w['C1'].value)"},
             outputs=OUT) == phases.PHASE_VERIFY
    assert f("run_python", {"code": "   "}, outputs=OUT) == phases.PHASE_OTHER


def test_same_rules_work_on_a_completely_different_skill():
    """三阶段是技能无关的：写 markdown 报告、跑 shell，同一套判据。"""
    f = phases.default_phase
    out = ("output_path", "report.md")
    assert f("bash", {"command": "cat notes/a.md"}, outputs=out) == phases.PHASE_PROBE
    assert f("bash", {"command": "cat notes/*.md > report.md"}, outputs=out) == phases.PHASE_APPLY
    assert f("bash", {"command": "wc -l report.md"}, outputs=out) == phases.PHASE_VERIFY
    # shell 的写操作按词边界认，cpu 不该命中 cp
    assert f("bash", {"command": "nproc && echo cpu"}, outputs=out) == phases.PHASE_PROBE


def test_outputs_come_from_the_task_not_from_guessing_filenames():
    assert phases.outputs_of({"workbook_path": "/i.xlsx", "output_path": "/o.xlsx"}) == \
        ("output_path", "/o.xlsx")
    assert phases.outputs_of({"src": "a", "dst": "b"}) == ("dst", "b")
    assert phases.outputs_of({"problem": "solve x"}) == ()


def test_no_declared_output_degrades_to_write_read_but_stays_decidable():
    """技能不产出文件（数学题只交答案字符串）：verify 那一分自然失效，仍可判定。"""
    f = phases.default_phase
    assert f("run_python", {"code": "print(2+2)"}) == phases.PHASE_PROBE
    assert f("run_python", {"code": "open('x.txt','w').write('1')"}) == phases.PHASE_APPLY


def test_save_is_judged_before_output_path():
    """应用那一步往往也提到产出路径。先判产出路径会把它错分成校验，
    整张图塌成「只有校验没有应用」——这条顺序是判据的一部分，不是实现细节。"""
    both = {"code": "ws['C1']='=A1'; wb.save(output_path)"}
    assert phases.default_phase("run_python", both, outputs=OUT) == phases.PHASE_APPLY


def test_specialised_tools_are_not_refined():
    """名字即用途的工具不参与细化——它们的 name 已经是身份了。"""
    assert phases.default_phase("math_verify", {"argv": ["equiv", "1", "1"]}) == ""
    assert phases.default_phase("apply_formula", {"formula_map": {"C1": "=A1"}}) == ""


def test_unknown_rules_and_bad_classifier_degrade_to_no_refinement():
    assert phases.classify("run_python", {"code": "x"}, "") == ""
    assert phases.classify("run_python", {"code": "x"}, "no_such_rules") == ""


def test_command_text_never_reads_the_hash_left_after_spilling():
    """正文被抽走之后只剩 code_sha256 —— 拿哈希判阶段等于拿指纹猜内容。"""
    spilled = {"code_sha256": "deadbeef", "code_path": "artifacts/deadbeef.py",
               "code_bytes": 12}
    assert phases.command_text(spilled) == ""
    assert phases.default_phase("run_python", spilled, outputs=OUT) == phases.PHASE_OTHER


# --------------------------------------------------------------------------- #
# KEY：两侧对称
# --------------------------------------------------------------------------- #
def test_phase_splits_the_key_and_absence_keeps_old_behaviour():
    a = {"kind": "tool", "name": "run_python", "phase": "probe"}
    b = {"kind": "tool", "name": "run_python", "phase": "apply"}
    plain = {"kind": "tool", "name": "run_python"}
    assert canon_action(a, strict=True) != canon_action(b, strict=True)
    assert canon_action(plain, strict=True) == ("tool", "name=run_python")
    assert canon_action(plain, strict=False) == ("tool", "name=run_python")
    # 机器侧的 Action 模型折出同一个 KEY（对称性）
    assert canon_action(ToolAction(name="run_python", phase="probe")) == canon_action(a)


# --------------------------------------------------------------------------- #
# 采集时判、随记录走
# --------------------------------------------------------------------------- #
def test_phase_is_computed_at_collection_before_the_body_is_spilled():
    raw = RawRun(task={"task_id": "t"}, steps=[
        _py("wb = load_workbook(src); print(wb.active.max_row)"),
        _py("ws['C1']='=A1&B1'; wb.save(output_path)"),
    ])
    t = to_trace(raw, phase_rules="default")
    assert [r.action["phase"] for r in t.records] == ["probe", "apply"]
    # 正文确实已经被抽走了，但阶段留下来了
    assert "code" not in t.records[0].action["input"]
    assert t.records[0].action["input"]["code_sha256"]
    # 这两步因此是两个不同的状态
    assert canon_action(t.records[0], strict=True) != canon_action(t.records[1], strict=True)


def test_without_rules_traces_are_byte_identical_to_before():
    raw = RawRun(task={"task_id": "t"}, steps=[_py("print(1)"), _py("wb.save(o)")])
    plain = to_trace(raw)
    assert all("phase" not in r.action for r in plain.records)
    assert canon_action(plain.records[0], strict=True) == \
        canon_action(plain.records[1], strict=True)      # 照旧折成同一个


# --------------------------------------------------------------------------- #
# 回放：不需要任何特殊处理
# --------------------------------------------------------------------------- #
def _phased_machine() -> Machine:
    """probe → apply → END，两个状态同名工具、不同阶段。"""
    return Machine(
        skill_id="toy", initial="s1", phase_rules="default",
        variables=[Variable(name="x")],
        states={
            "s1": State(id="s1", action=ToolAction(name="run_python", phase="probe",
                                                   writes=["x"]),
                        transitions=[Transition(to="s2", support=3)]),
            "s2": State(id="s2", action=ToolAction(name="run_python", phase="apply",
                                                   reads=["x"]),
                        transitions=[Transition(to="END", support=3)]),
            "END": State(id="END", action=EndAction(terminal="done")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
        },
        terminals=[{"id": "done"}],
    )


def test_replay_matches_phases_without_special_casing():
    m = _phased_machine()
    good = to_trace(RawRun(task={"task_id": "g"}, steps=[
        _py("print(load_workbook(src).active.max_row)"),
        _py("ws['C1']='=A1'; wb.save(out)"),
        RawStep(kind="end", args={"terminal": "done"}),
    ]), phase_rules="default")
    assert replay(m, good).ok

    # 顺序反了：先应用后探查 —— 同一个工具名，但阶段对不上，回放必须判偏离
    swapped = to_trace(RawRun(task={"task_id": "b"}, steps=[
        _py("ws['C1']='=A1'; wb.save(out)"),
        _py("print(load_workbook(src).active.max_row)"),
        RawStep(kind="end", args={"terminal": "done"}),
    ]), phase_rules="default")
    r = replay(m, swapped)
    assert not r.ok and r.diverged_at == 0


def test_machine_records_which_rules_it_was_compiled_with():
    """自描述：换了规则就是另一台机器，不会「拿 A 规则编、用 B 规则跑」还没人发现。"""
    assert _phased_machine().phase_rules == "default"
    assert Machine(skill_id="t", initial="FALLBACK").phase_rules == ""
