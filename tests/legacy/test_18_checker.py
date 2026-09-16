"""⑱ Deterministic gatekeeper (part 1): the receipted machine-edit interface.

These tests guard the single but crucial dividing line between :mod:`hexis.legacy.checker` and
:func:`hexis.legacy.compiler.compile_round`: **a rejected proposal does not affect proposals
accepted before it**. Undoing a whole round (compile_round) is a disaster in agentic compilation:
if proposal 7 is wrong, the 6 correct ones before it are lost too. So here every proposal is ruled
on separately, and only :meth:`~hexis.legacy.checker.Checker.commit` is all-or-nothing.

Hermetic: every machine is built by hand through the receipted interfaces and traces are
constructed by hand; **no model calls, no network, no file reads** (only the persistence test
uses pytest's tmp_path).
"""

import inspect

from hexis.legacy import checker, verify
from hexis.legacy.checker import Checker, Proposal, Receipt, check_machine
from hexis.machine.schema import Record, Trace, load_machine

JUDGE_Q = "Did this step's verification pass?"
LABELS = ["pass", "fail", "abstain"]

#: This guard and ``verdict == 'abstain'`` both hold under the configuration ``verdict='abstain'``: a
#: counterexample to mutual exclusion and completeness (Theorem 2). But its **text differs** from
#: the existing edge, so it passes the "duplicate edge" precondition and only the finite
#: configuration enumeration in checks._determinism can catch it. The bad proposal uses it.
BAD_COND = "verdict != 'pass'"


# --------------------------------------------------------------------------- #
# A normal proposal chain: read problem → run code → judge → (on fail, loop back to fix, bounded) → submit
# --------------------------------------------------------------------------- #
def _chain(ck: Checker, *, bad: bool = False) -> Checker:
    """Build a small machine one proposal at a time. ``bad`` inserts an add-edge proposal that will be rejected."""
    ck.open_machine()
    ck.add_state("s1", {"kind": "tool", "name": "read_problem", "writes": ["problem"]},
                 clause="S1", initial=True)
    ck.add_state("s2", {"kind": "tool", "name": "run_python",
                        "reads": ["problem"], "writes": ["ok"]},
                 clause="S2", from_state="s1", from_support=2)
    ck.add_judge("j", prompt=JUDGE_Q, reads=["ok"], writes=["verdict"],
                 labels=LABELS, clause="RV.0.1", from_state="s2", from_support=2)
    ck.set_terminal("END", "done", kind="verified", output=["ok"],
                    from_state="j", from_support=2)
    ck.add_transition("j", "FALLBACK", cond="verdict == 'abstain'")
    if bad:
        ck.add_transition("j", "s1", cond=BAD_COND, support=2)      # ← will be rejected
    ck.close_loop("j", "s2", cond="verdict == 'fail'",
                  counter="fix_count", bound=3, support=2)
    return ck


def _built(**kw) -> Checker:
    return _chain(Checker("math", doc="(skill document, provenance only)"), **kw)


def _good_trace(task_id="t1") -> Trace:
    """An accepted trace this machine can replay: read problem, run code, judge "pass", submit."""
    return Trace(task={"task_id": task_id, "input": {}}, verdict="accepted", records=[
        Record(step=1, action={"kind": "tool", "name": "read_problem"},
               output={"problem": "1+1"}),
        Record(step=2, action={"kind": "tool", "name": "run_python"},
               output={"ok": "2"}),
        Record(step=3, action={"kind": "judge"}, output={"verdict": "pass"}),
        Record(step=4, action={"kind": "end", "terminal": "done"}),
    ])


def _alien_trace() -> Trace:
    """An accepted trace that mismatches the machine at the very first step: acceptance must fail."""
    return Trace(task={"task_id": "t-alien", "input": {}}, verdict="accepted", records=[
        Record(step=1, action={"kind": "tool", "name": "browse_the_web"}),
    ])


# --------------------------------------------------------------------------- #
# Normal chain
# --------------------------------------------------------------------------- #
def test_a_valid_proposal_chain_is_accepted_end_to_end():
    ck = _built()
    assert all(r.accepted for r in ck.receipts()), \
        [r.reason for r in ck.receipts() if not r.accepted]
    m = ck.machine
    assert m.initial == "s1"
    assert set(m.states) == {"s1", "s2", "j", "END", "FALLBACK"}
    assert check_machine(m) == []                     # structure + support + error rate all pass


def test_close_loop_installs_counter_and_bound_in_one_proposal():
    """Back edge, counter variable and bound exit are one proposal: done separately, the intermediate state could never pass the checks."""
    m = _built().machine
    back = [t for t in m.states["j"].transitions if t.to == "s2"]
    assert len(back) == 1 and back[0].inc == "fix_count"
    counter = m.var("fix_count")
    assert counter is not None and counter.type == "integer" and counter.init == 0
    exits = [t for t in m.states["s2"].transitions
             if t.cond == "fix_count >= 3" and t.to == "FALLBACK"]
    assert exits, m.states["s2"].transitions


def test_a_new_state_joins_the_spine_instead_of_growing_a_second_default_edge():
    """An unguarded incoming edge **rewrites** the source state's default target instead of growing another, never-reachable default edge."""
    m = _built().machine
    for sid in ("s1", "s2", "j"):
        assert len([t for t in m.states[sid].transitions if not t.cond]) == 1


# --------------------------------------------------------------------------- #
# Core: a bad proposal is rejected while the ones accepted before it **survive**
# --------------------------------------------------------------------------- #
def test_one_bad_transition_is_rejected_while_the_accepted_prefix_survives():
    ck = _built(bad=True)
    rs = ck.receipts()
    bad = [r for r in rs if not r.accepted]
    assert len(bad) == 1 and bad[0].op == "add_transition"

    m = ck.machine
    # every edit before the bad proposal is still there
    assert set(m.states) == {"s1", "s2", "j", "END", "FALLBACK"}
    assert m.states["s1"].clause == "S1"
    assert m.states["j"].action.kind == "judge"
    assert [t for t in m.states["j"].transitions if t.cond == "verdict == 'abstain'"]
    # not a single byte of the bad proposal's edge landed
    assert not [t for t in m.states["j"].transitions if t.cond == BAD_COND]
    # the close_loop after the bad proposal takes effect as usual
    assert [t for t in m.states["j"].transitions if t.inc == "fix_count"]
    assert check_machine(m) == []


def test_the_rejection_receipt_names_the_failing_check_and_where():
    """The reason must be actionable for a retry: say which check failed and on which state."""
    ck = _built(bad=True)
    bad = next(r for r in ck.receipts() if not r.accepted)
    assert "overlapping" in bad.reason                 # which check (mutual exclusion and completeness)
    assert "E_OVERLAP" in bad.reason and "@j" in bad.reason      # which state
    codes = {f["code"] for f in bad.detail["findings"]}
    assert "E_OVERLAP" in codes
    assert all(f["severity"] in ("error", "warn") for f in bad.detail["findings"])


def test_preconditions_are_refused_before_anything_is_touched():
    """Missing state, duplicate edge, second default edge: all refused on the candidate copy, with the existing states in the reason."""
    ck = _built()
    before = ck.machine.model_dump()

    r1 = ck.add_transition("j", "ghost")
    assert not r1.accepted and "ghost" in r1.reason

    r2 = ck.add_transition("j", "END", cond="verdict == 'abstain'")
    assert not r2.accepted and "already has an edge with the identical guard" in r2.reason

    r3 = ck.add_transition("s1", "END")
    assert not r3.accepted and "default edge" in r3.reason

    r4 = ck.add_transition("END", "s1")
    assert not r4.accepted and "end state" in r4.reason

    r5 = ck.add_transition("j", "END", cond="nowhere == 1")
    assert not r5.accepted and "undeclared variables" in r5.reason

    assert ck.machine.model_dump() == before      # five rejections, machine untouched


def test_close_loop_refuses_an_edge_that_does_not_close_a_loop():
    ck = _built()
    r = ck.close_loop("s1", "END", counter="k", bound=2)
    assert not r.accepted and "is not a back edge" in r.reason


def test_close_loop_sets_a_targets_bound_once_and_says_how_to_add_the_second_back_edge():
    """``fit.install_counter`` identifies back edges by target, so a target's bound is set only once: ambiguity is not guessed away."""
    ck = _built()
    r = ck.close_loop("j", "s2", cond="verdict == 'abstain'", counter="fix_count")
    assert not r.accepted
    assert "already has an edge to s2" in r.reason and "add_transition" in r.reason
    # the bound exit is already installed; a second back edge just uses add_transition with the same counter
    ok = ck.add_transition("j", "s2", cond="verdict == 'pass'",
                           inc="fix_count", support=2)
    assert ok.accepted, ok.reason
    assert check_machine(ck.machine) == []


def test_add_transition_refuses_an_inc_on_an_undeclared_counter():
    """A counter can only be installed by close_loop together with its bound exit, not by hand-attaching an inc."""
    ck = _built()
    r = ck.add_transition("j", "s2", cond="verdict == 'pass'", inc="nope")
    assert not r.accepted and "close_loop" in r.reason


def test_add_judge_refuses_a_judge_that_is_already_too_noisy():
    """Knowingly installing a judge whose error rate exceeds the cap actively breaks the Σεᵢ inequality."""
    ck = _built()
    r = ck.add_judge("j", prompt=JUDGE_Q, reads=["ok"], writes=["verdict"],
                     labels=LABELS, error_rate=0.35)
    assert not r.accepted
    assert "0.35" in r.reason and "0.2" in r.reason
    assert ck.machine.states["j"].action.error_rate == 0.0


# --------------------------------------------------------------------------- #
# commit: all-or-nothing
# --------------------------------------------------------------------------- #
def test_commit_rolls_back_the_whole_batch_when_verification_fails():
    ck = _built()
    r = ck.commit(t_plus=[_good_trace(), _alien_trace()])
    assert not r.accepted
    assert "rolled back" in r.reason and "replayed 1/2" in r.reason
    assert r.detail["rolled_back"] is True
    # the whole batch is back to its open_machine state: an empty all-fallback machine
    m = ck.machine
    assert m.initial == "FALLBACK" and set(m.states) == {"FALLBACK"}


def test_commit_passes_and_is_the_only_thing_that_writes_the_machine_file(tmp_path):
    ck = _built()
    r = ck.commit(t_plus=[_good_trace()], root=tmp_path)
    assert r.accepted, r.reason
    assert (tmp_path / "machine.json").is_file()
    assert load_machine(tmp_path).initial == "s1"
    assert r.detail["report"]["ok"] is True


def test_a_failed_commit_rolls_back_only_to_the_last_successful_commit():
    """The last successful commit is the new foundation: rollback returns to it, not to open."""
    ck = _built()
    assert ck.commit(t_plus=[_good_trace()]).accepted
    ck.add_state("s9", {"kind": "tool", "name": "cleanup"},
                 from_state="s2", from_cond="fix_count < 3 and ok == 'X'",
                 from_support=2)
    assert ck.commit(t_plus=[_good_trace(), _alien_trace()]).accepted is False
    m = ck.machine
    assert set(m.states) == {"s1", "s2", "j", "END", "FALLBACK"}    # s9 is gone
    assert m.initial == "s1"                                       # but the foundation is still there


def test_rollback_does_not_erase_the_receipts():
    ck = _built()
    n = len(ck.receipts())
    ck.commit(t_plus=[_alien_trace()])
    assert len(ck.receipts()) == n + 1
    assert ck.receipts()[-1].op == "commit" and not ck.receipts()[-1].accepted


# --------------------------------------------------------------------------- #
# Escape hatch
# --------------------------------------------------------------------------- #
def test_demote_to_fallback_is_the_escape_hatch_that_always_works():
    """Guard cannot be learned / judge too noisy / loop cannot be bounded: falling back to interpreted execution always works."""
    ck = _built(bad=True)
    assert not ck.receipts()[-2].accepted          # one proposal was just rejected
    r = ck.demote_to_fallback("j", note="the guard for this branch cannot be learned")
    assert r.accepted, r.reason
    m = ck.machine
    assert [(t.cond, t.to) for t in m.states["j"].transitions] == [("", "FALLBACK")]
    assert check_machine(m) == []


def test_demote_drops_the_segment_that_only_hung_off_that_state():
    """Falling back to interpreted execution = giving up that part of the compiled artifact; keeping it would only leave unreachable dead code."""
    ck = _built()
    r = ck.demote_to_fallback("s2")
    assert r.accepted, r.reason
    m = ck.machine
    assert set(m.states) == {"s1", "s2", "FALLBACK"}      # j / END are removed as well
    assert "END" in r.reason and "j" in r.reason


def test_demote_refuses_only_where_there_is_nothing_to_demote():
    ck = _built()
    assert not ck.demote_to_fallback("nope").accepted
    assert not ck.demote_to_fallback("END").accepted      # an end state has no outgoing edges to begin with
    assert not ck.demote_to_fallback("FALLBACK").accepted


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
def test_receipts_form_a_complete_audit_trail():
    ck = _built(bad=True)
    ck.demote_to_fallback("j")
    ck.commit(t_plus=[_good_trace()])
    rs = ck.receipts()
    assert [r.op for r in rs] == [
        "open_machine", "add_state", "add_state", "add_judge", "set_terminal",
        "add_transition", "add_transition", "close_loop",
        "demote_to_fallback", "commit"]
    assert all(isinstance(r, Receipt) for r in rs)
    assert all(r.reason.strip() for r in rs)              # every receipt states a reason
    assert [r.accepted for r in rs].count(False) == 1     # only the bad proposal was rejected
    # every receipt carries its proposal arguments, so it can be replayed from them
    assert rs[1].detail["state_id"] == "s1"
    assert rs[-1].detail["t_plus"] == 1


def test_ops_before_open_machine_are_refused_with_an_actionable_reason():
    ck = Checker("math")
    r = ck.add_state("s1", {"kind": "tool", "name": "x"}, initial=True)
    assert not r.accepted and "open_machine" in r.reason
    assert not ck.opened
    assert not ck.commit().accepted


def test_open_machine_refuses_a_second_open_and_a_mismatched_base():
    ck = _built()
    assert not ck.open_machine().accepted
    other = Checker("table-clean")
    r = other.open_machine(base=_built().machine)
    assert not r.accepted and "skill_id" in r.reason


def test_the_machine_property_hands_out_a_copy_not_the_live_object():
    """Handing out the internal object would open an in-place modification channel that bypasses the receipted interfaces."""
    ck = _built()
    stolen = ck.machine
    stolen.states["s1"].transitions = []
    assert ck.machine.states["s1"].transitions            # the internal machine was not touched
    assert check_machine(ck.machine) == []


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
def test_batch_check_keeps_the_accepted_prefix_and_returns_one_receipt_per_proposal():
    ck = Checker("math")
    ck.open_machine()
    ck.add_state("s1", {"kind": "tool", "name": "read_problem", "writes": ["problem"]},
                 initial=True)
    base = ck.machine

    props = [
        Proposal("add_state", {"state_id": "s2",
                               "action": {"kind": "tool", "name": "run_python",
                                          "reads": ["problem"], "writes": ["ok"]},
                               "from_state": "s1", "from_support": 2}),
        Proposal("add_transition", {"from_state": "s2", "to": "ghost"}),   # bad
        Proposal("set_terminal", {"state_id": "END", "terminal": "done",
                                  "output": ["ok"], "from_state": "s2",
                                  "from_support": 2}),
        Proposal("no_such_op", {}),                                        # bad
    ]
    m, rs = checker.batch_check(base, props)
    assert [r.accepted for r in rs] == [True, False, True, False]
    assert len(rs) == len(props)
    assert set(m.states) == {"s1", "s2", "END", "FALLBACK"}
    assert "unknown op" in rs[3].reason
    assert set(base.states) == {"s1", "FALLBACK"}          # base was not modified in place


def test_batch_check_reports_a_base_it_cannot_even_take_over():
    ck = _built()
    broken = ck.machine
    broken.states["s2"].transitions = []                   # gets stuck once its action finishes
    m, rs = checker.batch_check(broken, [Proposal("demote_to_fallback",
                                                  {"state_id": "s2"})])
    assert len(rs) == 1 and rs[0].op == "open_machine" and not rs[0].accepted
    assert "refusing to take it over" in rs[0].reason
    assert m.states["s2"].transitions == []                # returned unchanged


# --------------------------------------------------------------------------- #
# "Does not call a model" is structural, not a promise
# --------------------------------------------------------------------------- #
def test_neither_gate_module_can_call_a_model():
    """No model parameter, no model client import: whether a machine is good should not be settled by a single sample."""
    for mod in (checker, verify):
        src = inspect.getsource(mod)
        for banned in ("llm_client", "model_iface", "OpenAIClient", "ModelAdapter"):
            assert banned not in src, (mod.__name__, banned)
        for name, obj in vars(mod).items():
            fns = []
            if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                fns.append(obj)
            elif inspect.isclass(obj) and obj.__module__ == mod.__name__:
                fns += [f for _n, f in inspect.getmembers(obj, inspect.isfunction)
                        if f.__module__ == mod.__name__]
            for fn in fns:
                params = set(inspect.signature(fn).parameters)
                assert "model" not in params, (mod.__name__, name, fn.__name__)
