"""Deterministic gatekeeper (part 2): acceptance gates for a **finished** machine.
**This module does not call a model.**

There is not a single ``model`` parameter and no model client is imported. Acceptance consists
entirely of replayable deterministic checks: drive the machine along the traces
(:mod:`hexis.legacy.replay`; Theorem 2 guarantees that, given the sequence of results, the state and
variable sequences are a function of it), then apply gates to the machine's own bookkeeping.

Four gates; the first two are new in this layer (**nothing else in the repository checked them
before**):

1. **Every transition's support ≥ ``min_support``.** An edge taken by only one trace is not a rule
   but a coincidence; compiling it into the machine freezes one accidental execution order into
   the skill. (Edges into FALLBACK are exempt: they mean "I did not learn this part" and by
   definition have no trace support.)
2. **Every judge action's calibrated error rate ≤ ``judge_err_max``.** In the inequality
   "P(a path errs at least once) ≤ Σεᵢ" every εᵢ needs a ceiling, otherwise a single noisy judge
   can blow through the bound.
3. **Every accepted trace (T+) is replayed** (``replay.reproduces``).
4. **Every rejected trace (T-) is excluded** (``replay.excludes``), but this is only required for
   counterexamples that **fall within the compiled region**.

The exception in gate 4 is the easiest thing to get wrong in this acceptance, so it is explained
separately: if a counterexample's ``error_step`` falls in the segment after the machine has already
entered FALLBACK, the machine has no structure there at all (interpreted mode does not commit to a
specific path, and :func:`hexis.legacy.replay.replay` unconditionally accepts the rest once it
reaches FALLBACK), so it **cannot** diverge and therefore **cannot exclude** the counterexample.
That is not a failure but "this segment has not been compiled yet". Counting it as a failure would
force the compile process to "fix" a nonexistent defect or, worse, to cut FALLBACK just to make the
numbers look good. So such counterexamples are reported separately as
:attr:`VerifyReport.fallback_deferred` (a ``W_FALLBACK_DEFERRED`` warning), counted neither as
failures nor as excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from hexis.execution.runtime import pick_edge
from hexis.legacy import conformance as _conformance
from hexis.legacy import replay as _replay
from hexis.legacy.checker import Finding, check_machine, hot_judges, weak_edges
from hexis.machine.schema import Machine, Thresholds, Trace


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VerifyReport:
    """The full bookkeeping of one acceptance run. ``ok`` is the conjunction of the four gates.

    ``total_negative`` is the **total number** of counterexamples (not "the number that should be
    excluded"); so the negative gate takes the form
    ``excluded + len(fallback_deferred) == total_negative``: counterexamples in the fallback
    segment count neither as excluded nor as failures.

    ``holdout_acc`` only has a value when ``holdout`` is given (the fraction of the holdout set
    that is replayed), and is then also judged against ``thresholds.acc_thr``.
    """

    reproduced: int
    total_positive: int
    excluded: int
    total_negative: int
    judge_err_ok: bool
    min_support_ok: bool
    findings: list[Finding]
    holdout_acc: Optional[float]
    weak_edges: list[tuple[str, str]]
    hot_judges: list[tuple[str, float]]
    ok: bool
    #: Indices of accepted traces that replay but whose real run is not equivalent
    #: (see hexis.legacy.conformance). Empty = all equivalent.
    nonconforming: list = field(default_factory=list)
    #: Indices of counterexamples whose error position lies in the FALLBACK segment,
    #: **not yet excludable** (not counted as failures)
    fallback_deferred: list[int] = field(default_factory=list)
    #: Indices of accepted traces that failed to replay
    unreproduced: list[int] = field(default_factory=list)
    #: Indices of counterexamples in the compiled region that were not excluded (real failures)
    unexcluded: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
def fallback_entry(machine: Machine, trace: Trace) -> Optional[int]:
    """At which step (``Record.step``) the machine entered FALLBACK along this trace. Never entered → ``None``.

    The driving logic matches :func:`hexis.legacy.replay.replay` (derive variables from the recorded
    outputs through the writes whitelist, back edges increment their own counters, ``pick_edge``
    selects edges); the only difference is that replay returns "accept" once it reaches FALLBACK,
    whereas here we want **at which step** it entered. Whether "this counterexample cannot be
    excluded yet" is decided by comparing exactly this position with ``error_step``.

    If the machine stops matching the trace midway (it has already diverged structurally), returns
    ``None``: that is the exclusion logic's territory and unrelated to the fallback segment.
    """
    return _replay.walk(machine, trace).fallback_at


def _cutoff(neg: Trace) -> int:
    if neg.error_step is not None:
        return neg.error_step
    return neg.records[-1].step if neg.records else 0


def _tid(trace: Trace) -> str:
    task = trace.task if isinstance(trace.task, dict) else {}
    return str(task.get("task_id", "") or "(no task_id)")


# --------------------------------------------------------------------------- #
def verify_machine(m: Machine, t_plus: Sequence[Trace] = (),
                   t_minus: Sequence[Trace] = (), *,
                   thresholds: Optional[Thresholds] = None,
                   holdout: Sequence[Trace] = ()) -> VerifyReport:
    """Run all four gates and produce a :class:`VerifyReport`. Does not call a model or write files."""
    thr = thresholds or m.thresholds
    findings: list[Finding] = list(check_machine(m, thresholds=thr))
    weak = weak_edges(m, thr)
    hot = hot_judges(m, thr)

    reproduced, unrep = 0, []
    for i, t in enumerate(t_plus):
        r = _replay.replay(m, t)
        if r.ok:
            reproduced += 1
            continue
        unrep.append(i)
        findings.append(Finding(
            "E_NOT_REPRODUCED", "error", "",
            f"accepted trace #{i} ({_tid(t)}) cannot be replayed: "
            f"{r.reason or 'no reason given'} (diverged at record {r.diverged_at}); "
            "the path the machine learned does not match the real execution"))

    excluded, deferred, unexc = 0, [], []
    for i, neg in enumerate(t_minus):
        if _replay.excludes(m, neg):
            excluded += 1
            continue
        entry = fallback_entry(m, neg)
        cut = _cutoff(neg)
        if entry is not None and cut >= entry:
            deferred.append(i)
            findings.append(Finding(
                "W_FALLBACK_DEFERRED", "warn", m.fallback,
                f"rejected trace #{i} ({_tid(neg)}) has its error position step={cut} in the FALLBACK segment"
                f" (the machine entered interpreted execution at step {entry}): this segment is not compiled yet "
                "and the machine has no structure there to diverge from, so it is **not yet excludable**; this is "
                "expected and not counted as a failure. To exclude it, compile this "
                "segment first"))
            continue
        unexc.append(i)
        findings.append(Finding(
            "E_NOT_EXCLUDED", "error", "",
            f"rejected trace #{i} ({_tid(neg)}) has its error position step={cut} in the compiled region, "
            "yet the machine still ran it to completion: states were over-merged, or a prohibition is missing. "
            "Positive examples can only confirm, never refute; this is exactly where T- has to do its job"))

    # ---- fifth gate: **execution-level conformance** ---- #
    # The "replay" gate above is a simulation: it drives the machine along the trace and compares
    # action identities step by step, so it cannot see the runtime layer (who fills ``${var}``,
    # whether the ``writes`` whitelist holds, whether ``phase`` is recorded on both sides, how guards
    # evaluate on real variables). The three holes observed in practice all lived in that layer
    # while replay was all green. So before delivery the **real** ``run_task`` is run once more,
    # with model and tools driven by the trace; the produced action sequence must match the trace
    # step by step and tool inputs must match verbatim. See :mod:`hexis.legacy.conformance`.
    nonconforming: list[int] = []
    for i, t in enumerate(t_plus):
        if i in unrep:
            continue                      # replay already failed; do not report conformance again
        cr = _conformance.check_trace(m, t)
        if cr.ok:
            continue
        nonconforming.append(i)
        findings.append(Finding(
            "E_NOT_CONFORMANT", "error", "",
            f"accepted trace #{i} ({_tid(t)}) replays, but **the real run is not equivalent**: "
            f"{cr.divergences[0].located() if cr.divergences else cr.error}; "
            "what the machine executes once delivered is not the same action sequence as the "
            "traces it was learned from"))

    holdout_acc: Optional[float] = None
    if holdout:
        holdout_acc = round(
            sum(1 for t in holdout if _replay.reproduces(m, t)) / len(holdout), 4)

    errs = [f for f in findings if f.severity == "error"]
    ok = (not errs
          and not weak
          and not hot
          and reproduced == len(t_plus)
          and excluded + len(deferred) == len(t_minus)
          and (holdout_acc is None or holdout_acc >= thr.acc_thr))
    return VerifyReport(
        reproduced=reproduced, total_positive=len(t_plus),
        excluded=excluded, total_negative=len(t_minus),
        judge_err_ok=not hot, min_support_ok=not weak,
        findings=findings, holdout_acc=holdout_acc,
        weak_edges=weak, hot_judges=hot, ok=ok,
        fallback_deferred=deferred, unreproduced=unrep, unexcluded=unexc,
        nonconforming=nonconforming)


def batch_check(m: Machine, t_plus: Sequence[Trace] = (),
                t_minus: Sequence[Trace] = (), *,
                thresholds: Optional[Thresholds] = None) -> list[Finding]:
    """Just the findings. Equivalent to ``verify_machine(...).findings``."""
    return verify_machine(m, t_plus, t_minus, thresholds=thresholds).findings


# --------------------------------------------------------------------------- #
def report_dict(rep: VerifyReport) -> dict:
    """Machine-readable report dict (goes into the receipt's ``detail``)."""
    return {
        "ok": rep.ok,
        "reproduced": rep.reproduced, "total_positive": rep.total_positive,
        "excluded": rep.excluded, "total_negative": rep.total_negative,
        "fallback_deferred": list(rep.fallback_deferred),
        "unreproduced": list(rep.unreproduced), "unexcluded": list(rep.unexcluded),
        "nonconforming": list(rep.nonconforming),
        "min_support_ok": rep.min_support_ok, "judge_err_ok": rep.judge_err_ok,
        "weak_edges": [list(x) for x in rep.weak_edges],
        "hot_judges": [list(x) for x in rep.hot_judges],
        "holdout_acc": rep.holdout_acc,
        "errors": sum(1 for f in rep.findings if f.severity == "error"),
        "warnings": sum(1 for f in rep.findings if f.severity == "warn"),
    }


def summary(rep: VerifyReport) -> str:
    """One-line human-readable summary for the receipt's ``reason``: a rejection reason must be directly actionable."""
    parts = [f"T+ replayed {rep.reproduced}/{rep.total_positive}",
             f"T- excluded {rep.excluded}/{rep.total_negative}"]
    if rep.fallback_deferred:
        parts.append(f"of which {len(rep.fallback_deferred)} have their error position in the FALLBACK segment and "
                     "are not yet excludable (not counted as failures)")
    if rep.weak_edges:
        parts.append(f"{len(rep.weak_edges)} edges with insufficient support {rep.weak_edges}")
    if rep.hot_judges:
        parts.append(f"judges over the error rate cap {rep.hot_judges}")
    if rep.holdout_acc is not None:
        parts.append(f"holdout replay rate {rep.holdout_acc}")
    errs = [f for f in rep.findings if f.severity == "error"]
    if errs:
        parts.append("errors: " + "; ".join(f.located() for f in errs[:3])
                     + ("……" if len(errs) > 3 else ""))
    return "; ".join(parts)


def render(rep: VerifyReport) -> str:
    """Render the acceptance report as human-readable text."""
    lines = ["Machine acceptance report", "=" * 32,
             f"verdict: {'passed' if rep.ok else 'failed'}",
             f"T+ replayed: {rep.reproduced}/{rep.total_positive}",
             f"T- excluded: {rep.excluded}/{rep.total_negative}"
             f" ({len(rep.fallback_deferred)} not yet excludable, in the FALLBACK segment)",
             f"transition support: {'pass' if rep.min_support_ok else 'fail'}"
             f" {rep.weak_edges or ''}",
             f"judge error rate: {'pass' if rep.judge_err_ok else 'fail'}"
             f" {rep.hot_judges or ''}",
             f"holdout replay rate: {rep.holdout_acc if rep.holdout_acc is not None else '(not given)'}"]
    for f in rep.findings:
        lines.append(f"  [{f.severity}] {f.located()}")
    return "\n".join(lines)


__all__ = [
    "VerifyReport", "batch_check", "fallback_entry", "render", "report_dict",
    "summary", "verify_machine",
]
