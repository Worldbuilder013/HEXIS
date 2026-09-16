"""Replay and exclusion: deterministically checking how a machine relates to a trace. **No model calls.**

Given the sequence of results, the machine's sequence of states and variables is a function of it. So replay does not
have to actually execute actions: the ``vars`` recorded at every step of the trace are enough to drive edge selection
and to check, record by record, whether the machine would take the same path.

* :func:`walk` -- the **single** implementation that pushes a machine along a trace. Replay, the fallback position and
  locating "which state to demote" at compile time used to be three look-alike loops (this module,
  verify.fallback_entry, compile_agent._machine_walk); each of the three drifted once, which was enough; now they are
  all thin wrappers around it.
* :func:`replay` -- whether the machine can replay this trace. Reaching ``FALLBACK`` accepts the rest unconditionally
  (interpretation mode promises no specific path, which is why "an all-fallback machine trivially replays any trace").
* :func:`reproduces` -- an accepted trace (T+) should be replayed.
* :func:`excludes` -- the machine should **diverge** from a rejected trace (T-) at its error position or earlier. This
  is "positive examples can only confirm, never refute" in code: without T-, over-merged states pass silently.

Two things are handled only in :func:`walk` and not reinvented anywhere else:

* The **begin tool** :data:`~hexis.traces.normalize.BEGIN_TOOL`: when the machine starts with it but the first record
  of the trace does not (traces from arms 1 and 2 never record it), a virtual record is inserted, and indices are still
  reported against the **original trace**, so callers that look up ``trace.records`` with ``diverged_at`` are not off
  by one.
* **Judge actions introduced from the document** (``JudgeAction.introduced``): the trace has no such step, so they are
  **zero-width**: they consume no record, their label is computed on the spot by the program labeler named in
  ``gold_from`` (:data:`trace_adapter.LABELERS`), they abstain when no label can be computed, and then pick an edge as
  usual. Calibration (``fit.calibrate``) and replay use the same labeler, so the two sides agree by construction. If a
  trace produced by arm 3 really does contain a matching judge record, it is consumed as an ordinary judge step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from hexis.execution.runtime import pick_edge
from hexis.machine.schema import Machine, Record, Trace
from hexis.traces.normalize import BEGIN_TOOL, canon_action, is_begin


@dataclass
class ReplayResult:
    ok: bool
    diverged_at: Optional[int] = None      # index (0-based) of the trace record where the divergence starts
    reason: str = ""


@dataclass
class WalkResult:
    """Everything observable about one walk. Indices always count in the ``records`` of the **original trace**.

    ``seq`` is ``[(record index, state id)]``: a state that consumed a record records that record; a zero-width judge
    records the index of the record **before** it; the virtual begin step records ``-1``. ``fallback_at`` is the
    ``Record.step`` at which FALLBACK was entered (the last step + 1 if the records ran out while still in the fallback
    state), or ``None`` if it was never entered.
    """

    ok: bool
    seq: list = field(default_factory=list)
    diverged_at: Optional[int] = None
    reason: str = ""
    fallback_at: Optional[int] = None
    values: dict = field(default_factory=dict)
    ended: bool = False


def _is_introduced_judge(action: Any) -> bool:
    """Zero-width state: an introduced judge, or a generation introduced by the input gate; neither consumes a record in replay."""
    return action.kind in ("judge", "model") and bool(getattr(action, "introduced", False))


def _begin_record(step: int, vars_: dict) -> Record:
    return Record(step=step, action={"kind": "tool", "name": BEGIN_TOOL, "input": {}},
                  output={}, vars=dict(vars_))


def walk(machine: Machine, trace: Trace, *,
         labelers: Optional[Mapping[str, Callable[[Trace, int], Optional[str]]]] = None
         ) -> WalkResult:
    """Push the machine along the trace. Variables are computed by the machine itself (trace outputs update them
    through each state's writes allowlist, back edges apply their own inc), so counter variables the machine created
    itself take part in edge selection correctly."""
    task = trace.task if isinstance(trace.task, dict) else {}
    inp = task.get("input", {}) if isinstance(task.get("input", {}), dict) else {}
    values: dict = dict(inp)
    values.update(machine.initial_values(inp))

    records = list(trace.records)
    offset = 0
    init = machine.states.get(machine.initial)
    if (init is not None and is_begin(init.action)
            and not (records and is_begin(records[0].action))):
        first_step = records[0].step - 1 if records else 0
        records = [_begin_record(first_step, values)] + records
        offset = 1                                  # subtracted when reporting indices

    if labelers is None:
        from hexis.traces.trace_adapter import (
            LABELERS as labelers,  # deferred: trace_adapter depends on modules around this one
        )

    def idx(i: int) -> int:
        return i - offset

    cur = machine.initial
    i = 0
    seq: list = []
    guard = 0
    limit = int(machine.max_steps or 24) + len(records) + 8
    while True:
        if cur == machine.fallback:
            if i < len(records):
                fb = records[i].step
            else:
                fb = (records[-1].step + 1) if records else 0
            return WalkResult(True, seq, None, "", fb, values, False)
        st = machine.states.get(cur)
        if st is None:
            return WalkResult(False, seq, idx(i), f"machine has no state {cur!r}", None, values)
        rec = records[i] if i < len(records) else None

        # ---- zero-width judge: no such step in the trace, the label is computed by a program labeler ---- #
        if st.action.kind == "judge" and _is_introduced_judge(st.action) and (
                rec is None or not _action_matches(st.action, rec.action)):
            guard += 1
            if guard > limit:
                return WalkResult(False, seq, idx(i), f"zero-width judge {cur} cannot get out ({limit} attempts)",
                                  None, values)
            fn = labelers.get(st.action.gold_from) if st.action.gold_from else None
            label: Optional[str] = None
            if fn is not None:
                try:
                    # The labeler contract is "``i`` is the index of the record **before** this judge". A zero-width
                    # judge sits at the position "about to consume records[idx(i)]", so the one before is idx(i)-1.
                    # Passing idx(i) makes every judge predict the step **after next**: measured, only 13 of 54 T+ replayed.
                    label = fn(trace, idx(i) - 1)
                except Exception:                                   # noqa: BLE001
                    label = None
            if label not in st.action.labels:
                label = st.action.abstain
            values[st.action.writes[0]] = label
            seq.append((idx(i) - 1, cur))
            edge, eerr = pick_edge(machine, cur, values)
            if eerr:
                return WalkResult(False, seq, idx(i), eerr, None, values)
            if edge is None:
                return WalkResult(False, seq, idx(i), f"{cur} has no edge to take at this step", None, values)
            if edge.inc:
                values[edge.inc] = (values.get(edge.inc) or 0) + 1
            cur = edge.to
            continue

        # ---- zero-width generation: a model state inserted by the input gate, with no step in the trace; the
        #      variables it writes are the real inputs of the very next tool record (matched back through the
        #      ${var} template of the following tool state) ---- #
        if st.action.kind == "model" and getattr(st.action, "introduced", False) and (
                rec is None or not _action_matches(st.action, rec.action)):
            guard += 1
            if guard > limit:
                return WalkResult(False, seq, idx(i), f"zero-width generation {cur} cannot get out ({limit} attempts)",
                                  None, values)
            dflt = next((t for t in st.transitions if not t.cond), None)
            nxt_state = machine.states.get(dflt.to) if dflt else None
            if (rec is not None and nxt_state is not None and nxt_state.action.kind == "tool"
                    and (rec.action or {}).get("kind") == "tool"):
                tmpl = getattr(nxt_state.action, "input", {}) or {}
                real = (rec.action or {}).get("input") or {}
                for k, v in tmpl.items():
                    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                        var = v[2:-1]
                        if var in st.action.writes and k in real:
                            values[var] = real[k]
            for w in st.action.writes:
                values.setdefault(w, None)
            seq.append((idx(i) - 1, cur))
            edge, eerr = pick_edge(machine, cur, values)
            if eerr:
                return WalkResult(False, seq, idx(i), eerr, None, values)
            if edge is None:
                return WalkResult(False, seq, idx(i), f"{cur} has no edge to take at this step", None, values)
            if edge.inc:
                values[edge.inc] = (values.get(edge.inc) or 0) + 1
            cur = edge.to
            continue

        if rec is None:
            return WalkResult(True, seq, None, "", None, values, False)   # records exhausted
        if not _action_matches(st.action, rec.action):
            return WalkResult(
                False, seq, idx(i),
                f"step {idx(i)} action mismatch: the machine at {cur} wants {st.action.kind}"
                f"/{getattr(st.action, 'name', '')}, the trace has "
                f"{rec.action.get('kind')}/{rec.action.get('name', '')}", None, values)
        seq.append((idx(i), cur))
        if st.action.kind == "end":
            return WalkResult(True, seq, None, "", None, values, True)
        for w in getattr(st.action, "writes", []) or []:
            if w in rec.output:
                values[w] = rec.output[w]
        edge, eerr = pick_edge(machine, cur, values)
        if eerr:
            return WalkResult(False, seq, idx(i), eerr, None, values)
        if edge is None:
            return WalkResult(False, seq, idx(i), f"{cur} has no edge to take at this step", None, values)
        if edge.inc:
            values[edge.inc] = (values.get(edge.inc) or 0) + 1
        nxt = edge.to
        if i + 1 < len(records) and nxt != machine.fallback:
            ns = machine.states.get(nxt)
            if ns is None or (not _is_introduced_judge(ns.action)
                              and not _action_matches(ns.action, records[i + 1].action)):
                return WalkResult(
                    False, seq, idx(i + 1),
                    f"step {idx(i + 1)}: the machine moves to {nxt}, whose action does not match the next trace step", None, values)
        cur = nxt
        i += 1


def replay(machine: Machine, trace: Trace) -> ReplayResult:
    """Check record by record whether the machine can take this trace.

    State names are not compared (the trace may come from another machine); **actions** are."""
    r = walk(machine, trace)
    return ReplayResult(r.ok, r.diverged_at, r.reason)


def reproduces(machine: Machine, trace: Trace) -> bool:
    return replay(machine, trace).ok


def excludes(machine: Machine, neg: Trace, *,
             evaluate: Optional[Callable[..., Any]] = None) -> bool:
    """Whether the machine excludes this rejected trace: it **diverges** at or before the error, or its prohibitions **stop** it.

    Two kinds of counterexample need different treatment: for one that takes a wrong order or misjudges a guard, the
    machine's structure diverges at the error; a prohibition violation (such as overwriting the source file) is
    structurally identical to a normal execution and is stopped by the prohibitions the machine carries. Either one
    counts as exclusion.

    ``evaluate`` is an **injected grader** with the same signature as :func:`hexis.traces.judge.evaluate`
    ``(trace, acceptance, prohibitions) -> Verdict``; when it is not given, that function is used. It is injectable so
    that the objective acceptance check can be swapped, for example for one that passes when the answer is equivalent
    to the reference:

    .. code-block:: python

        from hexis.traces import judge
        acc = my_acceptance(gold)                       # passes when the answer is equivalent to the reference
        excludes(machine, neg, evaluate=lambda t, _a, p: judge.evaluate(t, acc, p))

    Replay itself still **does not import a grader** (sympy is a heavy dependency), nor does it import judge at module
    level: the default implementation is fetched on demand, so the import graph stays one-directional.
    """
    cutoff = neg.error_step if neg.error_step is not None else len(neg.records)

    r = replay(machine, neg)
    if not r.ok and r.diverged_at is not None:
        d = r.diverged_at
        if d < 0:
            diverge_step = neg.records[0].step - 1 if neg.records else 0
        elif d < len(neg.records):
            diverge_step = neg.records[d].step
        else:
            diverge_step = 10 ** 9
        if diverge_step <= cutoff:
            return True

    if machine.prohibitions:
        if evaluate is None:
            from hexis.traces.judge import evaluate as evaluate  # default implementation: judge.evaluate
        v = evaluate(neg, None, machine.prohibitions)
        if v.verdict == "rejected" and (v.error_step is None or v.error_step <= cutoff):
            return True
    return False


def _action_matches(m_act, r_act: dict) -> bool:
    """Whether a machine state's action and a trace record's action are "the same step": the **replay mode**; folding
    is delegated to :func:`hexis.traces.normalize.canon_action` (``strict=False``).

    The comparison is **cross-source**: one side is an Action model from the machine (input is a ``${var}`` template,
    the prompt belongs to the private side), the other a **bare action dict** from the trace (all concrete rendered
    values). The lenient mode exists for exactly this comparison: it groups only by kind + tool name / written
    variables, and neither parameters nor prompts enter the KEY.

    Two components of the lenient KEY are **deliberately left out** of the comparison here: they are not criteria for
    "the same step" but things the trace side cannot provide, or that cannot be compared across sources:

    * ``writes``: judge/model/user actions recorded in traces have **no** writes field; it can only be inferred from
      the keys of ``output``, but replay gets ``rec.action`` (a bare dict without output), so one side is necessarily
      empty. The normalize module documentation states this degradation openly, and this function follows it: when one
      side is unknown, this component is not compared.
    * ``terminal``: how an end is named is the private business of **the machine that produced the trace**; judging
      divergence by it would mistake "finishing under a different name" for taking the wrong path. test_14's
      ``test_end_terminal_participates_in_both_modes`` pins this difference down as an assertion (the lenient mode is
      finer than this function), so it is kept explicit instead of quietly disappearing through the delegation.

    Apart from these two relaxations, the folding rules (tool name normalization, kinds never merged) all go through
    normalize and are not decided here.
    """
    a = canon_action(m_act, strict=False)
    b = canon_action(r_act, strict=False)
    if len(a) != len(b) or a[0] != b[0]:
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        if x.startswith("writes=") and y.startswith("writes=") and "writes=" in (x, y):
            continue                    # one side cannot infer writes (a bare action dict has no output)
        if x.startswith("terminal=") and y.startswith("terminal="):
            continue                    # end naming is each machine's own business, not compared across sources
        return False
    return True


__all__ = ["ReplayResult", "WalkResult", "excludes", "replay", "reproduces", "walk"]
