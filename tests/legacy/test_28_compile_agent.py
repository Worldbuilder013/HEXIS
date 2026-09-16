"""Algorithm 1 "sequential transcription compilation": compile agent + deterministic gatekeeper.

Focus: the division of labour between :mod:`hexis.legacy.compile_agent` and
:mod:`hexis.legacy.checker`: **the agent only proposes; only the gatekeeper writes the machine**. So
these tests ask four things --

1. with ``model=None`` (deterministic heuristics) the compiled machine has no structural issue at all,
   and every accepted trace replays (**"a correct machine must be compilable even without a model" is
   the floor of the method**);
2. a proposal that would break mutual exclusion is **rejected** by the gatekeeper, while **none of the
   proposals accepted before it are lost**;
3. two rejections in a row at the same point => ``demote_to_fallback``, falling back to interpreted
   execution, **not a crash**;
4. the coverage report can say "no trace ever touched this clause" and "this loop bound was added by
   the compiler".

Plus one baseline: compiling the same input twice yields a **byte-for-byte identical** machine.

Hermetic: ``model=None``, no network, no file writes; traces are generated on the fly by conftest's
``accepted`` fixture from a hand-written reference machine + scripted model stub.
"""

import json

import pytest

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.legacy import compile_agent, replay, report
from hexis.legacy.checker import Checker
from hexis.legacy.compile_agent import Proposal, apply_plan, compile_skill, draft_judge
from hexis.llm.model_iface import ScriptedModel
from hexis.machine import checks
from hexis.machine.schema import FALLBACK, Record, Trace

#: Traces and compile results are deterministic; cache one per (count, seed) so each test does not
#: rerun all 24 tasks.
_TRACES: dict = {}
_COMPILED: dict = {}


def _traces(accepted, n=24, seed=2):
    if (n, seed) not in _TRACES:
        _TRACES[(n, seed)] = accepted(n, seed=seed)
    return _TRACES[(n, seed)]


def _compiled(accepted, n=24, seed=2):
    if (n, seed) not in _COMPILED:
        _COMPILED[(n, seed)] = compile_skill(tc, _traces(accepted, n, seed), (),
                                             model=None)
    return _COMPILED[(n, seed)]


def _judge_state(machine):
    return next(s for s in machine.states.values() if s.action.kind == "judge")


def _neg_trace() -> Trace:
    """A rejected trace: export right after reading, **skipping the header check**; the error is at the
    export step."""
    inp = {"path": "in.csv", "output_path": "out.csv", "request": "Clean this table and export it"}
    bad = ",unit_price,stock"
    rows = [["v0", "v1", "v2"]]
    v1 = {**inp, "header_row": bad, "rows": rows}
    v2 = {**v1, "output_path": "out.csv"}
    return Trace(
        task={"task_id": "neg-skip-check", "input": dict(inp)},
        verdict="rejected", error_step=2,
        records=[
            Record(step=1, state="", action={"kind": "tool", "name": "read_csv",
                                             "input": {"path": "in.csv"}},
                   output={"ok": True, "header_row": bad, "rows": rows}, vars=v1),
            Record(step=2, state="",
                   action={"kind": "tool", "name": "export",
                           "input": {"header_row": bad, "rows": rows,
                                     "output_path": "out.csv",
                                     "source_path": "in.csv"}},
                   output={"ok": True, "output_path": "out.csv"}, vars=v2),
            Record(step=3, state="", action={"kind": "end", "terminal": "done"},
                   vars=v2),
        ])


# --------------------------------------------------------------------------- #
# (1) model=None: deterministic heuristics must also compile a correct machine
# --------------------------------------------------------------------------- #
def test_compiles_the_toy_with_no_model(accepted):
    res = _compiled(accepted)
    assert checks.structural_findings(res.machine) == [], \
        checks.structural_findings(res.machine)
    assert res.stats["model_calls"] == 0            # not a single model call
    assert res.stats["committed"] is True, res.stats["commit"]


def test_every_accepted_trace_replays(accepted):
    res = _compiled(accepted)
    traces = _traces(accepted)
    bad = [t.task.get("task_id") for t in traces if not replay.reproduces(res.machine, t)]
    assert bad == [], f"these accepted traces do not replay: {bad}"


def test_learns_read_judge_repair_export(accepted):
    """All four main-path segments are learned, and repair becomes a loop **with a counter and a bound
    exit**."""
    m = _compiled(accepted).machine
    names = {s.action.name for s in m.states.values() if s.action.kind == "tool"}
    assert {"read_csv", "fix_header", "export"} <= names
    assert any(s.action.kind == "judge" for s in m.states.values())

    back = [(src, t) for src, t in m.transitions_all() if t.inc]
    assert back, "repair should form a back edge with a counter"
    for _src, t in back:
        tgt = m.states[t.to]
        assert any(g.cond and t.inc in g.cond for g in tgt.transitions), \
            "the back edge's counter variable must have a bound exit on the target state"


def test_branch_condition_is_built_on_the_judge_output(accepted):
    m = _compiled(accepted).machine
    j = _judge_state(m)
    guarded = [t for t in j.transitions if t.cond and t.to != FALLBACK]
    assert len(guarded) >= 2, "there should be two guarded branches after the judge"
    assert all(j.action.writes[0] in t.cond for t in guarded)


def test_compiled_machine_runs_fresh_tasks(accepted):
    """The learned machine must generalize to unseen tasks rather than memorize the training traces."""
    m = _compiled(accepted).machine
    ok = 0
    for task in tc.gen_tasks(20, seed=99):
        fs = tc.MemFS(task["files"])
        r = runtime.run_task(m, task, model=tc.build_model(),
                             tools=tc.build_registry(fs), doc=tc.skill_doc())
        if r.stopped == "terminal" and tc.verify(task, r.trace):
            ok += 1
    assert ok == 20


def _neg_overwrite() -> Trace:
    """A rejected trace **structurally identical to the positive ones**: the export target is the source
    file (violates P1).

    Such a negative example has no point of divergence on the graph -- it has to be stopped by a
    prohibition, and the compiled artifact has no prohibitions (the ``table_clean`` skill object itself
    carries no ``prohibitions``, and the compiler does not pick them up from the hand-written reference
    machine). So it forces exactly L12's last resort: if it cannot be repaired, fall back to interpreted
    execution.
    """
    inp = {"path": "in.csv", "output_path": "in.csv", "request": "Clean this table and export it"}
    good, rows = "name,quantity,date", [["v0", "v1", "v2"]]
    v1 = {**inp, "header_row": good, "rows": rows}
    v2 = {**v1, "header_ok": "well_formed"}
    return Trace(
        task={"task_id": "neg-overwrite", "input": dict(inp)},
        verdict="rejected", error_step=3,
        records=[
            Record(step=1, action={"kind": "tool", "name": "read_csv",
                                   "input": {"path": "in.csv"}},
                   output={"ok": True, "header_row": good, "rows": rows}, vars=v1),
            Record(step=2, action={"kind": "judge", "prompt": tc.JUDGE_Q,
                                   "reads": ["header_row"]},
                   output={"header_ok": "well_formed"}, vars=v2),
            Record(step=3, action={"kind": "tool", "name": "export",
                                   "input": {"header_row": good, "rows": rows,
                                             "output_path": "in.csv",
                                             "source_path": "in.csv"}},
                   output={"ok": True, "output_path": "in.csv"}, vars=v2),
            Record(step=4, action={"kind": "end", "terminal": "done"}, vars=v2),
        ])


def test_negative_trace_is_excluded(accepted):
    """L12: for the negative example that skips the header check, the machine must diverge at its error
    position or earlier."""
    res = compile_skill(tc, _traces(accepted), [_neg_trace()], model=None)
    assert replay.excludes(res.machine, _neg_trace())
    assert res.coverage["verify"]["excluded"] == 1
    assert res.stats["committed"] is True, res.stats["commit"]


def test_unexcludable_negative_forces_a_demotion(accepted):
    """L12's last resort: a negative example that cannot be excluded => that segment falls back to
    interpreted execution instead of being forced through.

    After demotion the machine is still valid, the positive traces still replay, and the negative
    example lands in "not yet excludable" rather than "missed" -- a bit less was compiled, but nothing
    was compiled wrong.
    """
    neg = _neg_overwrite()
    res = compile_skill(tc, _traces(accepted), [neg], model=None)

    assert res.stats["fallback_demotions"] >= 1
    assert res.coverage["fallback_surface"]["demoted"], "the demoted point should be recorded"
    assert checks.structural_findings(res.machine) == []
    assert all(replay.reproduces(res.machine, t) for t in _traces(accepted))
    v = res.coverage["verify"]
    assert v["unexcluded"] == [] and len(v["fallback_deferred"]) == 1
    assert res.stats["committed"] is True, res.stats["commit"]


def test_one_trace_is_not_enough_to_compile_anything(accepted):
    """L14: edges with insufficient support are removed entirely. Every edge of a single trace has
    support 1, so **nothing should be compiled**."""
    one = [t for t in _traces(accepted) if len(t.records) == 4][:1]
    res = compile_skill(tc, one, (), model=None)
    assert res.stats["thin_edges_dropped"] >= 1
    assert res.machine.n_states() <= 1                 # at most the start remains; everything after goes to interpreted execution
    assert checks.structural_findings(res.machine) == []
    assert replay.reproduces(res.machine, one[0])


def test_no_traces_yields_a_valid_all_fallback_machine():
    """No traces means nothing to learn: deliver a valid, all-fallback empty machine instead of
    crashing."""
    res = compile_skill(tc, (), (), model=None)
    assert res.machine.initial == FALLBACK
    assert checks.structural_findings(res.machine) == []
    assert res.stats["committed"] is True
    assert report.render(res.coverage).startswith("Skill compilation coverage report")


@pytest.mark.parametrize("form,want", [
    ("module", "table-clean"),                          # the examples.table_clean module
    ("dict", "x"),                                      # {"skill_id": ..., "doc": ...}
    ("dir", "table-clean"),                             # an Agent Skill directory
    ("text", "compiled"),                               # just a body text
])
def test_skill_argument_forms(accepted, form, want):
    skill = {"module": tc, "dict": {"skill_id": "x", "doc": tc.skill_doc()},
             "dir": str(tc.SKILL_PATH.parent), "text": tc.skill_doc()}[form]
    res = compile_skill(skill, _traces(accepted)[:6], (), model=None)
    assert res.machine.skill_id == want
    assert len(res.coverage["clause_table"]) == 6       # SKILL.md splits into 6 clauses


# --------------------------------------------------------------------------- #
# (2) bad proposals are rejected; the accepted prefix loses nothing
# --------------------------------------------------------------------------- #
def _open_on(machine) -> Checker:
    ck = Checker(machine.skill_id, thresholds=machine.thresholds)
    assert ck.open_machine(base=machine).accepted
    return ck


def _overlapping(sid: str, cond: str) -> Proposal:
    """An add-edge proposal overlapping an existing guard on ``sid`` -- violates mutual exclusion
    (Theorem 2) and must be rejected."""
    return Proposal("add_transition",
                    {"from_state": sid, "to": FALLBACK, "cond": cond, "support": 4},
                    rationale="bad proposal deliberately overlapping an existing branch guard")


def test_mutual_exclusion_breaking_proposal_is_rejected_and_prefix_survives(accepted):
    m = _compiled(accepted).machine
    sid = _judge_state(m).id
    ck = _open_on(m)
    before = ck.machine.model_dump_json(by_alias=True)

    res = apply_plan(ck, [_overlapping(sid, "header_ok != 'abstain'")])

    assert res.rejected == 1 and res.accepted == 0
    receipt = res.receipts[0]
    assert not receipt.accepted
    assert "overlapping" in receipt.reason or "E_OVERLAP" in receipt.reason, receipt.reason
    # the rejected proposal left **not a single byte on the machine**; the previously accepted batch is intact
    assert ck.machine.model_dump_json(by_alias=True) == before
    assert res.demoted == []


def test_two_consecutive_rejections_demote_to_fallback(accepted):
    """Two rejections in a row at the same point => fall back to interpreted execution. **Neither a
    crash nor forcing it in.**"""
    m = _compiled(accepted).machine
    sid = _judge_state(m).id
    ck = _open_on(m)

    res = apply_plan(ck, [_overlapping(sid, "header_ok != 'abstain'"),
                          _overlapping(sid, "header_ok != 'well_formed'")])

    assert res.rejected == 2
    assert res.demoted == [sid], res.demoted
    after = ck.machine
    assert [t.to for t in after.states[sid].transitions] == [FALLBACK]
    assert after.states[sid].action.kind == "judge"      # the action is still there, just not compiled further
    assert checks.structural_findings(after) == []       # the machine is still valid after demotion


def test_one_rejection_alone_does_not_demote(accepted):
    """A single rejection is only recorded and must not change the machine -- demotion takes two."""
    m = _compiled(accepted).machine
    sid = _judge_state(m).id
    ck = _open_on(m)
    res = apply_plan(ck, [_overlapping(sid, "header_ok != 'abstain'")])
    assert res.demoted == []
    assert len(ck.machine.states[sid].transitions) > 1


def test_receipts_cover_every_proposal(accepted):
    """One receipt per proposal -- not erased even on rollback; this is the audit trail."""
    res = _compiled(accepted)
    ops = [r.op for r in res.receipts]
    assert ops[0] == "open_machine"
    assert ops[-1] == "commit"
    assert all(r.accepted for r in res.receipts), \
        [r.reason for r in res.receipts if not r.accepted]


# --------------------------------------------------------------------------- #
# (3) coverage report
# --------------------------------------------------------------------------- #
def test_coverage_lists_a_clause_no_trace_touched(accepted):
    """P1 is a prohibition: it cannot be compiled into the graph and no trace "walks" to it -- the
    report must name it."""
    cov = _compiled(accepted).coverage
    assert "P1" in cov["untouched"], cov["untouched"]
    row = next(r for r in cov["clause_table"] if r["id"] == "P1")
    assert row["status"] == "untouched"
    assert row["states"] == [] and row["traces"] == []
    # model=None => clause attribution is left entirely empty, and the report must say so honestly
    assert cov["clause_attribution"] == "none"
    assert set(cov["untouched"]) == {r["id"] for r in cov["clause_table"]}


def test_coverage_says_the_loop_bound_is_compiler_introduced(accepted):
    """SKILL.md never states any iteration bound; K is added by the compiler to guarantee halting --
    the report must say so."""
    cov = _compiled(accepted).coverage
    assert cov["loop_bounds"], "with back edges there should be a K ledger"
    for lb in cov["loop_bounds"]:
        assert lb["source"] == "compiler"
        assert "document" in lb["why"]
    kinds = {x["kind"] for x in cov["structures"]["compiler_introduced"]}
    assert {"loop_bound", "counter_variable", "branch_guard", "fallback_surface"} <= kinds


def test_coverage_names_what_needs_a_model(accepted):
    cov = _compiled(accepted).coverage
    dep = cov["model_dependence"]
    assert dep["model_used"] is False
    assert dep["touchpoints"] == list(compile_agent.MODEL_TOUCHPOINTS)
    assert any("learn_cond" in s for s in dep["model_free"])
    assert any("clause attribution" in s for s in dep["needs_model"])


def test_coverage_reports_the_fallback_surface(accepted):
    cov = _compiled(accepted).coverage
    fb = cov["fallback_surface"]
    assert fb["n_edges_to_fallback"] >= 1
    assert fb["demoted"] == [] and fb["blocked_branches"] == []
    assert cov["next_traces"], "the report should say which additional traces are most valuable"


def test_report_renders_the_coverage(accepted):
    text = report.render(_compiled(accepted).coverage)
    assert "coverage report" in text and "T+ replayed" in text


def test_diff_vs_reference_matches_the_target_shape(accepted):
    d = _compiled(accepted).diff_vs_reference
    assert d is not None
    assert d["actions_missing"] == [] and d["actions_extra"] == []
    assert d["n_states_compiled"] == d["n_states_reference"]


# --------------------------------------------------------------------------- #
# (4) no model means no drafted judge actions; drafting itself must pass the schema check
# --------------------------------------------------------------------------- #
def test_no_judge_is_drafted_without_a_model(accepted):
    res = _compiled(accepted)
    assert res.stats["judges_drafted"] == 0
    assert [j["source"] for j in res.judges] == ["trace"]   # a judge step already in the traces


def test_draft_judge_needs_a_model():
    assert draft_judge({"reads": ["x"], "targets": {"a": [{"x": 1}]}}, model=None) is None


@pytest.mark.parametrize("reply", [
    {"labels": ["A", "B"]},                        # missing question
    {"prompt": "  ", "labels": ["A", "B"]},       # question is blank
    {"prompt": "which branch", "labels": ["A"]},   # fewer than two labels
    {"prompt": "which branch"},                   # no labels
    "not a JSON object",                            # wrong shape altogether
])
def test_off_schema_reply_is_a_reject_not_a_guess(reply):
    model = ScriptedModel(gen=lambda p, v, h: reply)
    assert draft_judge({"state": "s2", "reads": ["x"],
                        "targets": {"a": [{"x": 1}]}}, model=model) is None


def test_draft_judge_accepts_a_well_formed_reply():
    model = ScriptedModel(gen=lambda p, v, h: {
        "prompt": "which way should this branch go", "labels": ["A", "B"]})
    j = draft_judge({"state": "s2", "reads": ["x", "y"], "writes": ["verdict"],
                     "targets": {"a": [{"x": 1}], "b": [{"x": 2}]}}, model=model)
    assert j is not None
    assert j.writes == ["verdict"] and j.reads == ["x", "y"]
    assert compile_agent.ABSTAIN in j.labels        # abstain is mandatory; added even if the model forgets it


def test_draft_judge_refuses_reads_outside_the_whitelist():
    """A model wanting to read one more variable is sneaking extra context into the judge action --
    only whitelisted variables are accepted."""
    model = ScriptedModel(gen=lambda p, v, h: {
        "prompt": "which branch", "labels": ["A", "B"], "reads": ["x", "sneaked_in"]})
    j = draft_judge({"state": "s2", "reads": ["x"], "writes": ["verdict"],
                     "targets": {"a": [{"x": 1}]}}, model=model)
    assert j is not None and j.reads == ["x"]


# --------------------------------------------------------------------------- #
# (5) the two touchpoints with a model (scripted stub, still hermetic and free)
# --------------------------------------------------------------------------- #
_CLAUSE_BY_STEP = {"tool:read_csv": "S1", "tool:fix_header": "S3", "tool:export": "S4"}


def _scripted_agent(prompt: str, values: dict) -> str:
    """A model stub that answers both decisions correctly: new step/repeat by KEY, clause attribution
    by looking up the action."""
    if prompt.startswith("Is this step a new step in the procedure"):
        cands = values.get("existing states of the same kind", "(none)")
        return "new step" if cands == "(none)" else "repeat:" + cands.split(",")[0]
    if prompt.startswith("Which clause of the skill document"):
        step = values.get("this step", "")
        if step.startswith("judge:"):
            return "S2.1"
        return _CLAUSE_BY_STEP.get(step, compile_agent.ABSTAIN)
    return compile_agent.ABSTAIN


def test_model_fills_in_clause_attribution(accepted):
    """Clause attribution is the half that **needs a model**: only with a model can the coverage report
    say which sentence each state exists because of."""
    res = compile_skill(tc, _traces(accepted), (), model=ScriptedModel(judge=_scripted_agent))
    assert checks.structural_findings(res.machine) == []
    got = {sid: s.clause for sid, s in res.machine.states.items() if s.clause}
    assert set(got.values()) == {"S1", "S2.1", "S3", "S4"}
    assert res.coverage["clause_attribution"] == "model"
    assert set(res.coverage["supported"]) == {"S1", "S2.1", "S3", "S4"}
    assert "P1" in res.coverage["untouched"]     # no trace ever "walks" to a prohibition
    assert res.stats["model_calls"] > 0 and res.stats["model_rejects"] == 0
    assert res.stats["committed"] is True, res.stats["commit"]


def test_model_only_changes_attribution_not_the_shape(accepted):
    """When touchpoints (a)(b) answer correctly, the machine's **shape** must be identical to the
    model=None one -- the model only adds semantics, it does not change structure."""
    a = _compiled(accepted).machine
    b = compile_skill(tc, _traces(accepted), (),
                      model=ScriptedModel(judge=_scripted_agent)).machine
    assert sorted(a.states) == sorted(b.states)
    for sid in a.states:
        assert [(t.cond, t.to, t.inc, t.support) for t in a.states[sid].transitions] == \
            [(t.cond, t.to, t.inc, t.support) for t in b.states[sid].transitions]


def test_off_schema_new_or_repeat_reply_is_rejected_then_demoted(accepted):
    """(a)'s reply never fits the schema => two rejections in a row at the same point => that segment
    falls back to interpreted execution, and the machine is still valid."""
    def always_abstain(prompt, values):
        return compile_agent.ABSTAIN if prompt.startswith("Is this step a new step in the procedure") \
            else compile_agent.ABSTAIN

    res = compile_skill(tc, _traces(accepted), (),
                        model=ScriptedModel(judge=always_abstain))
    assert res.stats["model_rejects"] >= 2
    assert res.stats["blocked_branches"], "the point that hit the rejection limit should be recorded"
    assert checks.structural_findings(res.machine) == []
    assert all(replay.reproduces(res.machine, t) for t in _traces(accepted))


# --------------------------------------------------------------------------- #
# (6) determinism
# --------------------------------------------------------------------------- #
def test_compile_skill_is_deterministic(accepted):
    traces = _traces(accepted)
    a = compile_skill(tc, traces, (), model=None)
    b = compile_skill(tc, traces, (), model=None)
    assert a.machine.model_dump_json(by_alias=True) == \
        b.machine.model_dump_json(by_alias=True)
    assert json.dumps(a.coverage["clause_table"], ensure_ascii=False, sort_keys=True) == \
        json.dumps(b.coverage["clause_table"], ensure_ascii=False, sort_keys=True)
    assert [r.op for r in a.receipts] == [r.op for r in b.receipts]


def test_trace_order_does_not_change_the_machine(accepted):
    """L2 says "fewest steps first" -- so the input order must not affect the artifact."""
    traces = _traces(accepted)
    a = compile_skill(tc, traces, (), model=None)
    b = compile_skill(tc, list(reversed(traces)), (), model=None)
    assert a.machine.model_dump_json(by_alias=True) == \
        b.machine.model_dump_json(by_alias=True)
