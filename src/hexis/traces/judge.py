"""Judging: decide accepted / rejected for a trace.

Judging has two independent axes; failing either one rejects:

* **Objective acceptance** (``acceptance``) -- is the result right. A deterministic gate that
  ships with the skill and looks only at the trace.
* **Prohibitions** (``prohibitions``) -- did the run do something it must not do. **Violating one
  rejects the trace even if the result is right**, and the step where the violation happened is
  marked as ``error_step`` (the anchor for the rejection-set exclusion check).

Judging is deterministic and never calls a model: it is one half of the "deterministic
gatekeeper", which the compiler uses to split traces into an accepted set and a rejected set.

Prohibitions come in two shapes. ``absent``/``present``/``regex`` look at **text** (whether some
string ever appeared), while ``forbid_action``/``require_before`` look at the **event stream**
(what came before what, which two arguments are equal). A math skill's P1, "every non-trivial
result must be independently checked at least once", belongs to the latter: it is not "don't say
a certain sentence" but "a check must have run before submitting", which can only be decided by
scanning the trace as a sequence of events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from hexis.machine.schema import Prohibition, Trace
from hexis.traces.normalize import canon_action, canon_tool_name


@dataclass
class Verdict:
    verdict: str                       # "accepted" / "rejected"
    error_step: Optional[int] = None
    reason: str = ""


def evaluate(trace: Trace, acceptance: Optional[Callable[[Trace], bool]],
             prohibitions: list[Prohibition]) -> Verdict:
    """Decide the verdict. Prohibitions come first (a violation rejects even if acceptance passes), then objective acceptance."""
    for p in prohibitions or []:
        step = _violation_step(p, trace)
        if step is not None:
            return Verdict("rejected", error_step=step,
                           reason=f"violates prohibition {p.id}")
    ok = acceptance(trace) if acceptance else True
    if ok:
        return Verdict("accepted")
    return Verdict("rejected", error_step=_last_step(trace), reason="failed objective acceptance")


def judged(trace: Trace, acceptance, prohibitions) -> Trace:
    """Return a **new** trace with verdict/error_step filled in (the original trace is untouched), for building the accepted/rejected sets.

    Run-level provenance (arm/run/model/harness) is carried over as is: judging should not erase
    "who ran this trace".
    """
    v = evaluate(trace, acceptance, prohibitions)
    return Trace(task=trace.task, arm=trace.arm, run=trace.run, model=trace.model,
                 harness=trace.harness, verdict=v.verdict, error_step=v.error_step,
                 records=trace.records)


# --------------------------------------------------------------------------- #
# Prohibition checks
# --------------------------------------------------------------------------- #
def _violation_step(p: Prohibition, trace: Trace) -> Optional[int]:
    """Return the step where the violation happened, or None if there is no violation."""
    if p.check == "forbid_action":
        return _forbid_action(p.pattern, trace)
    if p.check == "require_before":
        return _require_before(p.pattern, trace)
    if p.check == "regex":
        pat = re.compile(str(p.pattern))
        for r in trace.records:
            if pat.search(_record_text(r)):
                return r.step
        return None
    if p.check == "absent":
        for r in trace.records:
            if str(p.pattern) in _record_text(r):
                return r.step
        return None
    if p.check == "present":
        if not any(str(p.pattern) in _record_text(r) for r in trace.records):
            return _last_step(trace)
        return None
    return None


def _forbid_action(pattern: Any, trace: Trace) -> Optional[int]:
    """Structured prohibition: a violation when an action + a variable relation both hold.

    ``pattern`` = ``{"name": tool name, "equal": ["input.a", "input.b"]}``: a violation when the
    values at the two paths in that tool action are equal (e.g. export target == source file ->
    overwrites the original file).
    """
    if not isinstance(pattern, dict):
        return None
    name = pattern.get("name")
    equal = pattern.get("equal")
    for r in trace.records:
        if name and (r.action or {}).get("name") != name:
            continue
        if equal:
            vals = [_resolve(r.action, path) for path in equal]
            if all(v is not None for v in vals) and len({str(v) for v in vals}) == 1:
                return r.step
    return None


def _resolve(action: dict, path: str) -> Any:
    """Take a value from an action dict by a dotted path such as ``input.output_path``."""
    cur: Any = action
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


# --------------------------------------------------------------------------- #
# require_before: ordering requirements on the event stream (a math skill's P1)
# --------------------------------------------------------------------------- #
#: Generic terminal ids that **carry no category**. ``done`` only says "it ended", not which way
#: it ended; using it as a category would turn only_when into a filter that happens to never match.
_GENERIC_TERMINALS = frozenset({"done", "end", "stop", "ok"})


def _require_before(pattern: Any, trace: Trace) -> Optional[int]:
    """One of ``requires`` must have appeared before ``action`` appears; otherwise it is a violation.

    ``pattern``::

        {"action": "submit_answer",                  # the guarded action
         "requires": ["math_verify", "run_python"],  # any one suffices (compared by canonical tool name)
         "only_when": {"terminal_kind": "verified"}, # optional; omitted = applies to every trace
         "clause": "RV.0.1", "quote": "<original sentence from the skill doc>"}

    Scans ``trace.records`` **in order**, looking only at the first hit: hitting an action from
    ``requires`` first => this run really did check first, no violation; hitting the guarded
    action first => a violation, and ``error_step`` is that record's ``step``. Neither appears
    (e.g. budget exhausted, never submitted) => no violation -- P1 is about "was it checked when
    submitting", not "must submit".

    All action comparisons go through :func:`~hexis.traces.normalize.canon_action`, so tool names
    pass through :func:`~hexis.traces.normalize.canon_tool_name`: ``scripts/math_verify.py``,
    ``math-verify`` and ``MATH_VERIFY`` fold into the same name. **No second set of name-matching
    rules is written here** -- one would drift away from the folding used by compilation and
    replay. Elements of ``action``/``requires`` may be a bare tool name (a string) or a complete
    action dict (which must have ``kind``, e.g. ``{"kind": "end", "terminal": "END_VERIFIED"}``).

    ``only_when.terminal_kind`` is decided by :func:`terminal_kind`: a run that ended **explicitly
    marked as unverified** claims nothing, and P1 should not fire on it. The category value may be
    a string or a collection of strings.
    """
    if not isinstance(pattern, Mapping):
        return None
    guarded = pattern.get("action")
    if guarded is None or guarded == "":
        return None
    only = pattern.get("only_when")
    if isinstance(only, Mapping) and "terminal_kind" in only:
        if not _kind_matches(terminal_kind(trace), only["terminal_kind"]):
            return None
    want = _action_key(guarded)
    required = {_action_key(x) for x in _as_list(pattern.get("requires"))}
    for r in trace.records:
        key = canon_action(r)
        if key in required:
            return None                 # the check ran first
        if key == want:
            return r.step               # guarded action came first: no check at all before it
    return None


def _action_key(spec: Any) -> tuple[str, ...]:
    """Fold one entry of the pattern into an action KEY. A bare string is taken as a tool name (both sides of P1 are tools)."""
    if isinstance(spec, Mapping):
        return canon_action(spec)
    return canon_action({"kind": "tool", "name": str(spec)})


def _as_list(v: Any) -> list:
    """None -> []; a single string -> one element; other sequences are flattened one level as is."""
    if v is None:
        return []
    if isinstance(v, str) or isinstance(v, Mapping):
        return [v]
    if isinstance(v, Sequence):
        return list(v)
    return [v]


def _fold_kind(value: Any) -> str:
    """Canonical form of a category name. Reuses the tool-name folding (lower-case, ``-``/whitespace -> ``_``), then strips the ``end_`` prefix,
    so that the terminal id ``END_UNVERIFIED`` and the category ``unverified`` are the same thing."""
    s = canon_tool_name(str(value or ""))
    return s[4:] if s.startswith("end_") else s


def _kind_matches(actual: str, want: Any) -> bool:
    """Whether the trace's actual terminal category is one of ``want``.

    **When the category cannot be determined (``actual`` is empty) it counts as a match**, i.e. the
    only_when filter does not apply to traces that take no position, and the prohibition is still
    checked. This is the deliberately conservative direction: P1 is meant to catch "submitted
    without checking", and an exemption is granted only when the run is **explicitly** marked (the
    machine's ``END_UNVERIFIED`` terminal, or ``verified: false`` self-reported by the submit
    action); letting every undeterminable trace through instead would let a trace with a truncated
    head quietly bypass the check, and the violation rate would be systematically underestimated.
    """
    wants = {_fold_kind(w) for w in _as_list(want)}
    wants.discard("")
    if not wants:
        return True
    return not actual or actual in wants


def terminal_kind(trace: Trace) -> str:
    """Determine from the trace **which way** this run ended; empty string if it cannot be determined.

    Judging only has the trace, not the machine, so the ``kind`` declared on
    :class:`~hexis.machine.schema.Terminal` must be written into the trace by the execution side.
    Read in this priority order (earlier wins):

    1. an explicit ``terminal_kind`` in the ending record's
       ``action``/``action.input``/``output``/``vars`` -- the most direct; an executor that wants
       to be clear writes this key;
    2. the terminal id of the ending record (``{"kind": "end", "terminal": ...}``) folded into a
       category: ``END_UNVERIFIED`` -> ``unverified``. The generic ids ``done``/``end``/``stop``/
       ``ok`` are not categories and are skipped;
    3. the boolean ``verified`` self-reported by the submit action (the marker forced to ``False``
       when the budget is exhausted);
    4. none of these => empty string (no position; see how :func:`_kind_matches` treats the empty
       string).

    "The ending record" = the last record with ``kind == "end"``; without an end record
    (interpretive traces often finish with a ``submit_answer``) the last record is used. When the
    two are different records, both are examined.
    """
    recs = list(trace.records or ())
    if not recs:
        return ""
    last = recs[-1]
    end = next((r for r in reversed(recs) if _rec_kind(r) == "end"), None)
    tails = [end, last] if (end is not None and end is not last) else [end or last]

    for r in tails:                                     # (1) explicit marker
        for src in _rec_sources(r):
            if "terminal_kind" in src:
                return _fold_kind(src["terminal_kind"])
    if end is not None:                                 # (2) terminal id
        k = _fold_kind(_rec_action(end).get("terminal"))
        if k and k not in _GENERIC_TERMINALS:
            return k
    for r in tails:                                     # (3) self-reported verified marker
        for src in _rec_sources(r):
            v = src.get("verified")
            if isinstance(v, bool):
                return "verified" if v else "unverified"
    return ""


def _rec_action(r) -> Mapping:
    a = getattr(r, "action", None)
    return a if isinstance(a, Mapping) else {}


def _rec_kind(r) -> str:
    return str(_rec_action(r).get("kind") or "")


def _rec_sources(r) -> list[Mapping]:
    """The places in a record where an ending marker may be, in lookup order."""
    act = _rec_action(r)
    out = [act]
    for m in (act.get("input"), getattr(r, "output", None), getattr(r, "vars", None)):
        if isinstance(m, Mapping):
            out.append(m)
    return out


def _record_text(r) -> str:
    import json
    return json.dumps({"action": r.action, "output": r.output},
                      ensure_ascii=False, default=str)


def _last_step(trace: Trace) -> Optional[int]:
    return trace.records[-1].step if trace.records else None
