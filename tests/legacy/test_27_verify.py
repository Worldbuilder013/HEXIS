"""㉗ Deterministic gatekeeper (part 2): acceptance gates for a finished machine.

Two of the four gates are new in this layer (nothing else in the repository checked them before):
**support of every transition** and **calibrated error rate of every judge action**. The other two
are replaying accepted traces and excluding rejected traces.

The most important test here is the exception to the negative gate: if a counterexample's error
position falls in the segment after the machine has already entered FALLBACK, the machine **has no
structure there to diverge from** (interpreted mode does not commit to a specific path), so it
cannot exclude it. That is not a failure but "this segment has not been compiled yet". Counting it
as a failure would force the compile process to fix a nonexistent defect or, worse, to cut FALLBACK
just to make the numbers look good.

Hermetic: machines and traces are all constructed by hand; **no model calls, no network, no file reads**.
"""

from hexis.legacy import verify
from hexis.legacy.checker import Finding
from hexis.machine.schema import (
    EndAction,
    JudgeAction,
    Machine,
    Record,
    State,
    Terminal,
    Thresholds,
    ToolAction,
    Trace,
    Transition,
    Variable,
)

JUDGE_Q = "Did this step's verification pass?"
LABELS = ["pass", "abstain"]


# --------------------------------------------------------------------------- #
# Machines
# --------------------------------------------------------------------------- #
def _clean_machine(*, support: int = 3) -> Machine:
    """Read problem → run code → submit. A small machine that passes all three gates."""
    return Machine(
        skill_id="math", initial="s1",
        variables=[Variable(name="problem", type="string"),
                   Variable(name="ok", type="string")],
        states={
            "s1": State(id="s1", clause="S1",
                        action=ToolAction(name="read_problem", writes=["problem"]),
                        transitions=[Transition(to="s2", support=support)]),
            "s2": State(id="s2", clause="RV.0.1",
                        action=ToolAction(name="run_python", reads=["problem"],
                                          writes=["ok"]),
                        transitions=[Transition(to="END", support=support)]),
            "END": State(id="END", action=EndAction(terminal="done")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done", kind="verified", output=["ok"])],
    )


def _judge_machine(error_rate: float) -> Machine:
    """Read problem → judge → submit. The judge action's calibrated error rate is a parameter."""
    m = _clean_machine()
    m.states["s2"] = State(
        id="s2", clause="RV.0.1",
        action=JudgeAction(prompt=JUDGE_Q, reads=["problem"], writes=["ok"],
                           labels=list(LABELS), error_rate=error_rate, support=8),
        transitions=[Transition(to="END", support=3)])
    return m


def _stub_machine() -> Machine:
    """A machine that learned only the first step and hands everything after it back to interpreted execution. Counterexamples fall in its fallback segment."""
    return Machine(
        skill_id="math", initial="s1",
        variables=[Variable(name="problem", type="string")],
        states={
            "s1": State(id="s1", clause="S1",
                        action=ToolAction(name="read_problem", writes=["problem"]),
                        transitions=[Transition(to="FALLBACK")]),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done")])


# --------------------------------------------------------------------------- #
# Traces
# --------------------------------------------------------------------------- #
def _positive(task_id="t+", judge=False) -> Trace:
    mid = ({"kind": "judge"} if judge else {"kind": "tool", "name": "run_python"})
    return Trace(task={"task_id": task_id, "input": {}}, verdict="accepted", records=[
        Record(step=1, action={"kind": "tool", "name": "read_problem"},
               output={"problem": "1+1"}),
        Record(step=2, action=mid, output={"ok": "2"}),
        Record(step=3, action={"kind": "end", "terminal": "done"}),
    ])


def _negative_in_compiled_region(task_id="t-compiled") -> Trace:
    """Takes exactly the machine's path but the result is wrong: the error position is in the compiled region, and the machine should have diverged but did not."""
    t = _positive(task_id)
    return Trace(task=t.task, verdict="rejected", error_step=2, records=t.records)


def _negative_in_fallback_segment(task_id="t-fallback") -> Trace:
    """At step 2 it already runs into territory the machine has not compiled (the machine is in FALLBACK by step 2)."""
    return Trace(task={"task_id": task_id, "input": {}}, verdict="rejected",
                 error_step=2, records=[
                     Record(step=1, action={"kind": "tool", "name": "read_problem"},
                            output={"problem": "1+1"}),
                     Record(step=2, action={"kind": "tool", "name": "submit_answer",
                                            "input": {"answer": "41"}}),
                 ])


# --------------------------------------------------------------------------- #
# Gate ①: support
# --------------------------------------------------------------------------- #
def test_an_edge_with_support_one_fails_min_support_and_shows_up_in_weak_edges():
    """An edge taken by only one trace is not a rule but a coincidence."""
    m = _clean_machine()
    m.states["s2"].transitions[0].support = 1
    rep = verify.verify_machine(m, [_positive()])
    assert rep.min_support_ok is False
    assert rep.weak_edges == [("s2", "")]
    assert rep.ok is False
    assert rep.reproduced == 1 and rep.total_positive == 1     # the replay gate passes
    hit = [f for f in rep.findings if f.code == "W_LOW_SUPPORT"]
    assert len(hit) == 1 and hit[0].state_id == "s2" and hit[0].severity == "warn"
    assert "support 1" in hit[0].message and "minimum 2" in hit[0].message


def test_the_min_support_bar_is_configurable_and_exempts_edges_into_fallback():
    """Edges into FALLBACK have no trace support by definition: demanding support for them amounts to forbidding the compiler from admitting it is stuck."""
    m = _clean_machine(support=1)
    assert verify.verify_machine(
        m, thresholds=Thresholds(min_support=1)).min_support_ok is True
    stub = _stub_machine()                                     # its only edge goes to FALLBACK
    assert verify.verify_machine(stub).weak_edges == []


# --------------------------------------------------------------------------- #
# Gate ②: judge action error rate
# --------------------------------------------------------------------------- #
def test_a_judge_with_error_rate_over_the_cap_fails_and_shows_up_in_hot_judges():
    """Every εᵢ in the Σεᵢ inequality needs a ceiling, otherwise one noisy judge blows through the bound."""
    m = _judge_machine(0.3)
    rep = verify.verify_machine(m, [_positive(judge=True)])
    assert rep.judge_err_ok is False
    assert rep.hot_judges == [("s2", 0.3)]
    assert rep.ok is False
    hit = [f for f in rep.findings if f.code == "W_JUDGE_ERR"]
    assert len(hit) == 1 and hit[0].state_id == "s2"
    assert "0.3" in hit[0].message and "0.2" in hit[0].message


def test_a_judge_right_at_the_cap_passes():
    """The cap is ≤, not <: a judge exactly at the cap stays."""
    rep = verify.verify_machine(_judge_machine(0.2), [_positive(judge=True)])
    assert rep.judge_err_ok is True and rep.hot_judges == [] and rep.ok is True


# --------------------------------------------------------------------------- #
# Gate ③: replay accepted traces
# --------------------------------------------------------------------------- #
def test_a_positive_trace_that_does_not_replay_is_an_error():
    alien = Trace(task={"task_id": "t-alien", "input": {}}, verdict="accepted",
                  records=[Record(step=1, action={"kind": "tool", "name": "browse"})])
    rep = verify.verify_machine(_clean_machine(), [_positive(), alien])
    assert rep.reproduced == 1 and rep.total_positive == 2
    assert rep.unreproduced == [1]
    assert rep.ok is False
    err = [f for f in rep.findings if f.code == "E_NOT_REPRODUCED"]
    assert len(err) == 1 and err[0].severity == "error"
    assert "t-alien" in err[0].message


# --------------------------------------------------------------------------- #
# Gate ④: exclude rejected traces, and the exception that must be kept apart
# --------------------------------------------------------------------------- #
def test_a_negative_in_the_compiled_region_that_is_not_excluded_is_a_real_failure():
    """The machine can still walk the counterexample's path: states were over-merged, or a prohibition is missing. This is a real failure."""
    neg = _negative_in_compiled_region()
    rep = verify.verify_machine(_clean_machine(), [], [neg])
    assert rep.excluded == 0 and rep.total_negative == 1
    assert rep.unexcluded == [0] and rep.fallback_deferred == []
    assert rep.ok is False
    err = [f for f in rep.findings if f.code == "E_NOT_EXCLUDED"]
    assert len(err) == 1 and err[0].severity == "error"


def test_a_negative_whose_error_step_is_in_the_fallback_segment_is_deferred_not_failed():
    """The error position is in the fallback segment: the machine has no structure there to diverge from, so it is **not yet excludable**, and that is not a failure."""
    m, neg = _stub_machine(), _negative_in_fallback_segment()
    rep = verify.verify_machine(m, [], [neg])
    assert rep.excluded == 0 and rep.total_negative == 1
    assert rep.fallback_deferred == [0] and rep.unexcluded == []
    assert not [f for f in rep.findings if f.severity == "error"]
    assert rep.ok is True                                   # ← not counted as a failure
    warn = [f for f in rep.findings if f.code == "W_FALLBACK_DEFERRED"]
    assert len(warn) == 1 and warn[0].severity == "warn"
    assert "not yet excludable" in warn[0].message and "not counted as a failure" in warn[0].message
    assert "not yet excludable" in verify.summary(rep)


def test_fallback_entry_reports_the_step_where_the_machine_gives_up():
    """The boundary of the fallback segment is computable: the step at which the machine entered FALLBACK."""
    assert verify.fallback_entry(_stub_machine(),
                                 _negative_in_fallback_segment()) == 2
    # a fully compiled machine has no fallback segment
    assert verify.fallback_entry(_clean_machine(), _positive()) is None


def test_the_deferral_disappears_once_that_step_is_actually_compiled():
    """The exception is temporary, not a free pass: the same counterexample is really excluded on a
    machine that has that step compiled.

    At step 2 it takes ``submit_answer`` while the compiled machine expects ``run_python`` there, so
    the machine diverges right at the error position, which is exactly what ``replay.excludes``
    needs. The fallback-segment exemption only means "its turn to be checked has not come yet".
    """
    neg = _negative_in_fallback_segment()
    rep = verify.verify_machine(_clean_machine(), [_positive()], [neg])
    assert rep.fallback_deferred == []          # no longer exempt
    assert rep.excluded == 1 and rep.unexcluded == []
    assert rep.ok is True


def test_an_excluded_negative_counts_as_excluded():
    """The machine diverging at or before the error position = excluded."""
    neg = Trace(task={"task_id": "t-div", "input": {}}, verdict="rejected",
                error_step=2, records=[
                    Record(step=1, action={"kind": "tool", "name": "read_problem"},
                           output={"problem": "1+1"}),
                    Record(step=2, action={"kind": "tool", "name": "guess"}),
                ])
    rep = verify.verify_machine(_clean_machine(), [_positive()], [neg])
    assert rep.excluded == 1 and rep.unexcluded == [] and rep.fallback_deferred == []
    assert rep.ok is True


# --------------------------------------------------------------------------- #
# All pass / holdout / findings shortcut
# --------------------------------------------------------------------------- #
def test_a_machine_that_passes_everything_is_ok():
    rep = verify.verify_machine(_clean_machine(), [_positive("a"), _positive("b")])
    assert rep.ok is True
    assert (rep.reproduced, rep.total_positive) == (2, 2)
    assert (rep.excluded, rep.total_negative) == (0, 0)
    assert rep.min_support_ok and rep.judge_err_ok
    assert rep.weak_edges == [] and rep.hot_judges == []
    assert rep.findings == [] and rep.holdout_acc is None


def test_the_holdout_gate_is_independent_of_the_training_traces():
    """A holdout replay rate below acc_thr also fails: the path is right but does not generalize, so it should not land either."""
    held = Trace(task={"task_id": "h", "input": {}}, verdict="accepted",
                 records=[Record(step=1, action={"kind": "tool", "name": "browse"})])
    rep = verify.verify_machine(_clean_machine(), [_positive()], holdout=[held])
    assert rep.holdout_acc == 0.0 and rep.ok is False
    assert not [f for f in rep.findings if f.severity == "error"]   # the training traces all pass
    good = verify.verify_machine(_clean_machine(), [_positive()],
                                 holdout=[_positive("h2")])
    assert good.holdout_acc == 1.0 and good.ok is True


def test_batch_check_is_just_the_findings_of_verify_machine():
    m = _judge_machine(0.3)
    fs = verify.batch_check(m, [_positive(judge=True)])
    assert all(isinstance(f, Finding) for f in fs)
    assert fs == verify.verify_machine(m, [_positive(judge=True)]).findings
    assert {f.code for f in fs} == {"W_JUDGE_ERR"}


def test_structural_breakage_shows_up_as_an_error_finding_and_blocks_ok():
    """The eight structural checks are still part of this acceptance: they are errors and fail ok directly."""
    m = _clean_machine()
    m.states["s2"].transitions = []                    # gets stuck once its action finishes
    rep = verify.verify_machine(m, [])
    assert rep.ok is False
    codes = {f.code for f in rep.findings if f.severity == "error"}
    assert "E_NO_EDGE" in codes


def test_report_and_render_round_out_the_receipt_surface():
    rep = verify.verify_machine(_clean_machine(), [_positive()])
    d = verify.report_dict(rep)
    assert d["ok"] is True and d["reproduced"] == 1 and d["errors"] == 0
    text = verify.render(rep)
    assert "Machine acceptance report" in text and "T+ replayed: 1/1" in text
