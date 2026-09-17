"""Phase classification: give steps that use "the same generic tool for different purposes" a decidable identity. **Skill-agnostic.**

Without this layer, the three steps of ``bash``/``run_python`` (look up the table / write a
formula / check it back) all fold into one ``('tool','name=run_python')``, and the whole trace
becomes one state looping on itself three times -- all the information is in the command text,
and by design the command text does not enter the KEY.

These tests pin four things: the criteria themselves, the **order** of the criteria (saving is
checked before the output path), that the phase is computed at collection time and travels with
the record (by compile time the body has been moved out to artifacts), and symmetry between the
two sides -- machine states carry ``phase`` too, so replay needs no special handling, and actions
without a phase behave exactly as before.
"""
from __future__ import annotations

from hexis.machine.schema import (
    EndAction,
    Machine,
    Record,
    State,
    ToolAction,
    Trace,
    Transition,
    Variable,
)
from hexis.traces import phases
from hexis.traces.normalize import canon_action
from hexis.traces.trace_adapter import RawRun, RawStep, to_trace


def _py(code: str) -> RawStep:
    return RawStep(kind="tool", name="run_python", args={"code": code})


# --------------------------------------------------------------------------- #
# Criteria
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
    """The three phases are skill-agnostic: writing a markdown report, running shell -- the same criteria."""
    f = phases.default_phase
    out = ("output_path", "report.md")
    assert f("bash", {"command": "cat notes/a.md"}, outputs=out) == phases.PHASE_PROBE
    assert f("bash", {"command": "cat notes/*.md > report.md"}, outputs=out) == phases.PHASE_APPLY
    assert f("bash", {"command": "wc -l report.md"}, outputs=out) == phases.PHASE_VERIFY
    # shell write operations are matched on word boundaries; cpu must not match cp
    assert f("bash", {"command": "nproc && echo cpu"}, outputs=out) == phases.PHASE_PROBE


def test_outputs_come_from_the_task_not_from_guessing_filenames():
    assert phases.outputs_of({"workbook_path": "/i.xlsx", "output_path": "/o.xlsx"}) == \
        ("output_path", "/o.xlsx")
    assert phases.outputs_of({"src": "a", "dst": "b"}) == ("dst", "b")
    assert phases.outputs_of({"problem": "solve x"}) == ()


def test_no_declared_output_degrades_to_write_read_but_stays_decidable():
    """The skill produces no file (a math problem only submits an answer string): the verify split naturally drops out, still decidable."""
    f = phases.default_phase
    assert f("run_python", {"code": "print(2+2)"}) == phases.PHASE_PROBE
    assert f("run_python", {"code": "open('x.txt','w').write('1')"}) == phases.PHASE_APPLY


def test_save_is_judged_before_output_path():
    """The apply step often mentions the output path too. Checking the output path first would misclassify it as verify,
    and the whole graph would collapse into "only verify, no apply" -- this order is part of the criteria, not an implementation detail."""
    both = {"code": "ws['C1']='=A1'; wb.save(output_path)"}
    assert phases.default_phase("run_python", both, outputs=OUT) == phases.PHASE_APPLY


def test_specialised_tools_are_not_refined():
    """Tools whose name is their purpose take no part in refinement -- their name already is the identity."""
    assert phases.default_phase("math_verify", {"argv": ["equiv", "1", "1"]}) == ""
    assert phases.default_phase("apply_formula", {"formula_map": {"C1": "=A1"}}) == ""


def test_unknown_rules_and_bad_classifier_degrade_to_no_refinement():
    assert phases.classify("run_python", {"code": "x"}, "") == ""
    assert phases.classify("run_python", {"code": "x"}, "no_such_rules") == ""


def test_command_text_never_reads_the_hash_left_after_spilling():
    """After the body is moved out only code_sha256 remains -- classifying the phase from a hash is guessing content from a fingerprint."""
    spilled = {"code_sha256": "deadbeef", "code_path": "artifacts/deadbeef.py",
               "code_bytes": 12}
    assert phases.command_text(spilled) == ""
    assert phases.default_phase("run_python", spilled, outputs=OUT) == phases.PHASE_OTHER


# --------------------------------------------------------------------------- #
# KEY: symmetric on both sides
# --------------------------------------------------------------------------- #
def test_phase_splits_the_key_and_absence_keeps_old_behaviour():
    a = {"kind": "tool", "name": "run_python", "phase": "probe"}
    b = {"kind": "tool", "name": "run_python", "phase": "apply"}
    plain = {"kind": "tool", "name": "run_python"}
    assert canon_action(a, strict=True) != canon_action(b, strict=True)
    assert canon_action(plain, strict=True) == ("tool", "name=run_python")
    assert canon_action(plain, strict=False) == ("tool", "name=run_python")
    # the machine side's Action model folds into the same KEY (symmetry)
    assert canon_action(ToolAction(name="run_python", phase="probe")) == canon_action(a)


# --------------------------------------------------------------------------- #
# Classified at collection time, travels with the record
# --------------------------------------------------------------------------- #
def test_phase_is_computed_at_collection_before_the_body_is_spilled():
    raw = RawRun(task={"task_id": "t"}, steps=[
        _py("wb = load_workbook(src); print(wb.active.max_row)"),
        _py("ws['C1']='=A1&B1'; wb.save(output_path)"),
    ])
    t = to_trace(raw, phase_rules="default")
    assert [r.action["phase"] for r in t.records] == ["probe", "apply"]
    # the body really has been moved out, but the phase stayed
    assert "code" not in t.records[0].action["input"]
    assert t.records[0].action["input"]["code_sha256"]
    # so these two steps are two different states
    assert canon_action(t.records[0], strict=True) != canon_action(t.records[1], strict=True)


def test_without_rules_traces_are_byte_identical_to_before():
    raw = RawRun(task={"task_id": "t"}, steps=[_py("print(1)"), _py("wb.save(o)")])
    plain = to_trace(raw)
    assert all("phase" not in r.action for r in plain.records)
    assert canon_action(plain.records[0], strict=True) == \
        canon_action(plain.records[1], strict=True)      # fold into the same one, as before


# --------------------------------------------------------------------------- #
# Replay: no special handling needed
# --------------------------------------------------------------------------- #
def _phased_machine() -> Machine:
    """probe -> apply -> END: two states with the same tool name and different phases."""
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


def test_machine_records_which_rules_it_was_compiled_with():
    """Self-describing: different rules make a different machine, so "compiled with rules A, run with rules B" cannot go unnoticed."""
    assert _phased_machine().phase_rules == "default"
    assert Machine(skill_id="t", initial="FALLBACK").phase_rules == ""
