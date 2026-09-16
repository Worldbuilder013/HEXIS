"""Fold one real agent loop into a :class:`~hexis.machine.schema.Trace`, and parse the output of check tools.

This module sits between **collection** and **compilation**, and is the only place in the
three-arm experiment that converts "raw run -> training data". It does three things, and getting
any of them wrong quietly makes everything downstream meaningless:

1. **Parse check results** (:func:`parse_verify`). Almost every branch condition the compiler
   learns reads ``verify_status``: submit if the check passed, go back and repair if it did not.
   Reading ``INCONCLUSIVE`` as ``FAIL``, a crash as "wrong answer", or flattening the inverted
   polarity of ``counterexample`` into ``ok = (rc == 0)`` -- in each case the machine learns a
   wrong edge from it, and **runs without any visible anomaly**: it simply repairs at the wrong
   time and submits when it should repair.
2. **Maintain variable snapshots** (:func:`to_trace`). Each record's ``vars`` is a sample point
   the compiler fits conditions on. The variable update rules are the generating rules of this
   training data, so they are spelled out one by one in the "variable table" below.
3. **Write to and read back from disk** (:func:`write_jsonl` / :func:`read_jsonl`).

Cross-arm comparability is a hard constraint
--------------------------------------------
Comparing the three arms presumes that the same thing has the same name in all three arms. So:

* every tool name goes through :func:`~hexis.traces.normalize.canon_tool_name` --
  ``scripts/math_verify.py``, ``math-verify`` and ``MATH_VERIFY`` fold into the same
  ``math_verify``;
* the final submit step is called ``submit_answer`` (:data:`SUBMIT_TOOL`) in **every arm**. The P1
  prohibition is ``require_before: a check must have run before submit_answer``; once the guarded
  action is named differently in some arm, that arm's violation rate is always 0 -- not because it
  follows the rules, but because the check never fired.

Both are checked one by one by :func:`check_canonical` at the end of :func:`to_trace`; a mismatch
raises :class:`TraceAdapterError`, so a trace with inconsistent naming never quietly enters the
dataset.

Variable table (every record carries all of them in ``Record.vars``; initial values in parentheses)
-----------------------------------------------------------------------------------------------------
============== ================================================================
variable        update rule
============== ================================================================
``repair_count`` (0) **when a check starts executing**, if the previous check's ``verify_status``
                 is neither empty nor ``PASS``, first add 1. I.e. "checked again after a check
                 that did not pass" = one round of repair. The first check never adds 1 (there
                 is no failure before it).
``verify_status``(``""``) status of the most recent check, taken from :func:`parse_verify`:
                 ``PASS``/``FAIL``/``INCONCLUSIVE``/``ERROR``/``TIMEOUT``. ``""`` = not checked yet.
``verify_exit``  (-1) process exit code of the most recent check. -1 is a sentinel (not checked
                 yet); real exit codes are >= 0, and the sandbox's timeout/spawn failure use
                 negative codes (see :data:`TIMEOUT_RC`/:data:`ERROR_RC`).
``verify_stdout``(``""``) stdout of the most recent check, clipped to :data:`VERIFY_STDOUT_MAX`
                 characters. Clipping is necessary: vars records a snapshot per record, and not
                 clipping means copying the same stdout a dozen times. The status line is on the
                 first line, so clipping does not hurt the part that needs to be read.
``candidate``    (``""``) the candidate answer currently on the table. Every step with an answer
                 argument updates it (check steps and submit steps both count). **This goes one
                 step beyond the spec**: if it were filled only on submit, every record before
                 the submit would have an empty candidate, and the compiler could not fit the
                 edge "a candidate answer exists => go check it" -- which is exactly the main loop
                 of the math skill.
``answer``       (``""``) the final answer, written **only** by the submit step. It is kept apart
                 from ``candidate`` so that "a submit happened" leaves a trace in the variables.
============== ================================================================

Deliberately not done
---------------------
**The task input is not spread into ``vars``.** ``runtime.run_task`` does
``values = dict(task["input"])``, which is harmless on toy skills; but MATH-500 task dicts **carry
the reference answer**. Spreading it in would put the reference answer into the variable snapshot
of every record, and the compiler could well fit a branch condition that reads the gold standard
-- the machine would then have "learned" to look at the answer, and the three-arm comparison would
be void immediately. The task input is still written into the trace header (needed for
reproduction), but not into the variables.

Code bodies are stored offline
------------------------------
Python written on the fly by the model (the ``code`` argument of ``run_python``) is not kept in
the record: the record keeps only ``code_sha256`` and ``code_path``, and the body is written to
``artifacts/<sha256>.py``. If ``artifacts_dir`` is given it is written to disk immediately;
otherwise it is parked in ``Record.meta`` first (``meta`` takes no part in normalization or
judging) and moved out by :func:`write_jsonl` when writing to disk -- both paths produce
byte-identical JSONL, and **judging always happens after the move**, so re-judging a trace read
back from JSONL always yields the same verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from hexis.machine.schema import FALLBACK, Prohibition, Record, Trace
from hexis.traces import judge as _judge
from hexis.traces import phases as _phases
from hexis.traces.normalize import BEGIN_TOOL, canon_action, canon_tool_name, is_begin

#: Killed by timeout / never started: two return codes that cannot collide with real exit codes
#: keep them apart. The old harness recorded -1 for both, so "crashed" and "timed out" looked
#: exactly the same downstream.
TIMEOUT_RC = -100
ERROR_RC = -101

__all__ = [
    "branch_key", "branch_label", "next_action_label",
    "ensure_phases", "load_any_trace", "read_raw_jsonl",
    "ANSWER_ARG_KEYS", "ARTIFACTS_DIRNAME", "CODE_KEYS", "ERROR", "FAIL",
    "INCONCLUSIVE", "INVERTED_SUBCOMMANDS", "PASS", "REPLY_MAX", "RawRun",
    "RawStep", "SUBMIT_ALIASES", "SUBMIT_TOOL", "TIMEOUT", "TOOL_STDOUT_MAX",
    "TraceAdapterError", "VERIFY_STDOUT_MAX", "VERIFY_SUBCOMMANDS",
    "VERIFY_TOOLS", "VerifyOutcome", "check_canonical", "initial_vars",
    "parse_verify", "read_jsonl", "to_trace", "write_jsonl",
]


class TraceAdapterError(RuntimeError):
    """Errors in trace conversion that **cannot be tolerated**: inconsistent naming, non-consecutive step numbers. Better to fail loudly than to produce dirty data."""


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
ERROR = "ERROR"
TIMEOUT = "TIMEOUT"

#: All five statuses. The first three are reported by the tool **itself**; the last two are this
#: module's classification of "no trustworthy ruling".
STATUSES = (PASS, FAIL, INCONCLUSIVE, ERROR, TIMEOUT)

#: All subcommands of ``math_verify.py`` (taken verbatim from the vendored script's ``build_parser``).
VERIFY_SUBCOMMANDS = frozenset({
    "equiv", "derivative", "antiderivative", "definite-integral", "substitute",
    "satisfies", "limit", "solve", "system", "counterexample",
})

#: Subcommands with **inverted** polarity: what they look for is a counterexample, and only finding
#: one counts as "search succeeded". See :func:`parse_verify`.
INVERTED_SUBCOMMANDS = frozenset({"counterexample"})

#: Tools (canonical names) that count as "check code was run". ``run_python``/``run_script`` are
#: treated the same as the skill's own ``math_verify`` -- in calibration runs models used
#: ``run_python`` more often than ``math_verify.py``, and recognising only the latter would
#: misjudge runs that really did check as violations.
VERIFY_TOOLS = frozenset({"math_verify", "run_python", "run_script"})

#: The **only** canonical name of the final submit step.
SUBMIT_TOOL = "submit_answer"

#: Aliases rewritten to :data:`SUBMIT_TOOL` after folding. Models name the submit action in all
#: sorts of ways, and failing to align them across arms amounts to switching off the P1 check, so
#: they are funnelled into one name here. ``done``/``finish`` and the like are **not** included --
#: they mean "end", not "hand in the answer", and mixing them in would count an abstention as a
#: submission.
SUBMIT_ALIASES = frozenset({
    "submit_answer", "submit", "submit_final_answer", "final_answer",
    "finalize_answer", "give_answer", "answer",
})

#: Keys in step arguments that may hold the answer, in priority order.
ANSWER_ARG_KEYS = ("answer", "final_answer", "candidate", "result", "value")

#: Argument keys holding a "code body": these values move to ``artifacts/`` and the record keeps only the hash.
CODE_KEYS = ("code", "script", "source")

#: Directory name for offline bodies (relative to the trace directory).
ARTIFACTS_DIRNAME = "artifacts"

VERIFY_STDOUT_MAX = 400
TOOL_STDOUT_MAX = 2000
REPLY_MAX = 4000

#: Exit codes left by wall-clock guards on various platforms: 124 (coreutils ``timeout``),
#: 137/-9 (SIGKILL), 143/-15 (SIGTERM), and this repository's sandbox :data:`TIMEOUT_RC`.
_TIMEOUT_RCS = frozenset({124, 137, 143, -9, -15, TIMEOUT_RC})

_STATUS_LINE_RE = re.compile(r"^\s*status\s*:\s*([A-Za-z_]+)\s*$")


# --------------------------------------------------------------------------- #
# Raw runs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawStep:
    """One step of the agent loop, **unprocessed**.

    ``kind`` is one of four: ``model`` (one model reply) / ``tool`` (one tool call) / ``judge``
    (one semantic judgement) / ``end`` (halt). ``text`` is the model body with ``<think>`` already
    stripped -- endpoints always inline the reasoning block in ``content``, and without stripping
    it the draft would be taken for the answer. ``meta`` holds execution-side accounting such as
    token counts, durations and model id; ``meta["state"]`` can override the record's state name,
    and ``meta["timed_out"]`` tells the parser this run was killed by the wall clock.
    """

    kind: str
    name: str = ""
    args: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    text: str = ""
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RawRun:
    """One complete run: the task, the step sequence, plus the provenance fields saying "who ran it"."""

    task: dict
    steps: list[RawStep]
    arm: str = ""
    run: int = 0
    model: str = ""
    harness: str = ""


# --------------------------------------------------------------------------- #
# Parsing check output
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VerifyOutcome:
    """The ruling of one check.

    ``status`` is **the raw status reported by the tool** (or ``ERROR``/``TIMEOUT`` assigned by this
    module); it is the value the compiler reads when fitting conditions. ``ok`` is **the reading
    with subcommand semantics**; the two deliberately disagree on ``counterexample``, for the
    reasons given in :func:`parse_verify`. ``detail`` is supporting evidence for auditing and does
    not enter the variables.
    """

    status: str
    ok: bool
    detail: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """The status is exactly ``PASS`` (ignoring subcommand polarity). Use this to ask "is the answer right"."""
        return self.status == PASS


def parse_verify(stdout: str, returncode: Optional[int], *,
                 argv: Sequence[str] = (),
                 timed_out: Optional[bool] = None) -> VerifyOutcome:
    """Read one output of the check tool into a :class:`VerifyOutcome`.

    All rules come from measurements against the vendored ``scripts/math_verify.py`` and the
    cases of its validation suite:

    * ``--json`` mode: stdout is one JSON object whose keys come back in alphabetical order (the
      script uses ``sort_keys=True``); read ``["status"]``;
    * plain mode: the **first line** of stdout is always ``status: PASS|FAIL|INCONCLUSIVE``;
    * **exit code 0 <=> ``status: PASS``; ``FAIL`` and ``INCONCLUSIVE`` both exit with 1**. So the
      exit code alone can never distinguish "wrong answer" from "tool reached no conclusion", let
      alone "wrong answer" from "tool crashed" -- all three are 1. Hence this function **first
      requires a ``status``**, and judges ``ERROR`` if there is none;
    * empty stdout + a traceback on stderr + rc 1 => ``ERROR`` (feeding it LaTeX gives this shape);
    * rc 2 (argparse usage error, e.g. ``--json`` placed after the subcommand) => ``ERROR``;
    * rc 124 / killed by the wall-clock guard => ``TIMEOUT`` (``equiv "9^9^9" "1"`` never returns;
      this rule catches it);
    * ``rc not in {0, 1}`` is always ``ERROR``: an out-of-range exit code means this is not output
      of the check protocol, and even if stdout really contains a ``status:`` line it should not
      be trusted.

    **``counterexample`` has inverted polarity.** Its job is "find a counterexample":
    ``status: PASS`` means "the equality was proven symbolically, so no counterexample exists"
    (the search found nothing), and ``status: FAIL`` means **a witness was found** (the search
    succeeded, the claim is false). So on this subcommand ``ok`` equals ``status == FAIL``. ``ok``
    reads as "this call got the affirmative answer it was looking for", **not** "the answer is
    right" -- for the latter read ``status`` or :attr:`VerifyOutcome.passed`. The
    ``verify_status`` maintained by :func:`to_trace` uses ``status``, so this inversion does not
    contaminate the repair loop.

    When a status line is found, **the status line is trusted** even if it disagrees with the exit
    code (``detail["exit_agrees"]`` records this faithfully): a check script written by the model
    may well ``print("status: FAIL")`` and then exit normally, and judging that as ERROR would
    throw away a readable ruling. The genuinely untrustworthy cases (rc out of range, no status
    line) have already been filtered out above.
    """
    text = stdout or ""
    rc = returncode
    check = _subcommand_of(argv)
    inverted = check in INVERTED_SUBCOMMANDS
    detail: dict[str, Any] = {"rc": rc, "check": check, "inverted": inverted}

    label, mode, line_no, payload = _read_status(text)
    detail["mode"] = mode
    if payload is not None:
        detail["payload"] = payload
        if "witness" in payload:
            detail["witness"] = payload["witness"]
    if line_no is not None:
        detail["status_line"] = line_no
        detail["status_first_line"] = (line_no == 0)

    # ---- first, the cases where no trustworthy ruling was obtained at all ---- #
    if timed_out or (isinstance(rc, int) and rc in _TIMEOUT_RCS):
        detail["reason"] = "timeout"
        return VerifyOutcome(TIMEOUT, False, detail)
    if not isinstance(rc, int):
        detail["reason"] = "no_returncode"
        return VerifyOutcome(ERROR, False, detail)
    if rc not in (0, 1):
        detail["reason"] = ("usage_error" if rc == 2 else
                            "spawn_error" if rc == ERROR_RC else "bad_exit")
        return VerifyOutcome(ERROR, False, detail)
    if label is None:
        detail["reason"] = "crash" if not text.strip() else "no_status"
        return VerifyOutcome(ERROR, False, detail)
    if label not in (PASS, FAIL, INCONCLUSIVE):
        detail["reason"] = "unknown_status"
        detail["raw_status"] = label
        return VerifyOutcome(ERROR, False, detail)

    # ---- there is a ruling ---- #
    detail["exit_agrees"] = ((rc == 0) == (label == PASS))
    if inverted:
        detail["witness_found"] = (label == FAIL)
        ok = (label == FAIL)
    else:
        ok = (label == PASS)
    return VerifyOutcome(label, ok, detail)


def _subcommand_of(argv: Sequence[str]) -> str:
    """Recognise the ``math_verify.py`` subcommand in argv; empty string if none is recognised.

    Only words from the allow-list are recognised, so ``--var``, the script path, or an argument
    value that happens to share a name are never taken for a subcommand.
    """
    for tok in argv or ():
        s = str(tok)
        if s in VERIFY_SUBCOMMANDS:
            return s
    return ""


def _read_status(text: str) -> tuple[Optional[str], str, Optional[int], Optional[dict]]:
    """Take the status from stdout: returns ``(status label, mode, line number, JSON payload)``.

    First tries JSON (``--json`` mode, the whole stdout is one object); then looks line by line for
    a line that **as a whole** has the form ``status: XXX``. If none is found returns
    ``(None, ...)`` -- the caller judges ERROR from that.

    In plain mode the status is always on the first line, but all lines are still scanned and the
    line number recorded: a check script written by the model may print a couple of other things
    first. Requiring a whole-line match keeps wording like ``note: ... status: ...`` in body text
    out.
    """
    stripped = (text or "").strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            st = payload.get("status")
            label = str(st).strip().upper() if isinstance(st, str) else None
            return label, "json", None, payload
    for i, line in enumerate(stripped.splitlines()):
        m = _STATUS_LINE_RE.match(line)
        if m:
            return m.group(1).strip().upper(), "plain", i, None
    return None, ("json" if stripped.startswith("{") else "plain"), None, None


# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #
def initial_vars() -> dict:
    """The variable snapshot at the start of a run. See the "variable table" in the module docs for what the initial values mean."""
    return {
        "repair_count": 0,
        "verify_status": "",
        "verify_exit": -1,
        "verify_stdout": "",
        "candidate": "",
        "answer": "",
    }


#: **Model-side** variables: arms one and two expose only tool calls, so what is in the model's head
#: (what it said last, which tool it called last, whether that succeeded) originally has no
#: snapshot in the trace. Judge actions introduced from the document need to read exactly these --
#: without a snapshot, ``fit.calibrate`` has no samples to calibrate on. So each record's ``vars``
#: additionally records these three; they are not in :func:`initial_vars` (that is the fixed
#: tool-side variable table), and the compiler does not treat them as machine variables -- unless
#: some judge action declares that it reads them.
MODEL_VARS = ("last_reply", "last_tool", "last_tool_ok")


def model_vars() -> dict:
    return {"last_reply": "", "last_tool": "", "last_tool_ok": None}


# --------------------------------------------------------------------------- #
# Begin tool: the compile-time view of a trace
# --------------------------------------------------------------------------- #
def with_begin(trace: Trace) -> Trace:
    """Prepend a :data:`~hexis.traces.normalize.BEGIN_TOOL` record to the trace (step = first - 1,
    usually 0). Idempotent: an already-prepended trace is returned as is. **Not written back to
    disk** -- this is the view the compiler and replay look at; the collected trace is left
    byte-for-byte untouched (``records[0]`` there stays the first real tool step).

    ``error_step`` needs no shift: real records keep their step numbers.
    """
    recs = list(trace.records or ())
    if recs and is_begin(recs[0].action):
        return trace
    task = trace.task if isinstance(trace.task, dict) else {}
    inp = task.get("input", {}) if isinstance(task.get("input", {}), dict) else {}
    first = (recs[0].step - 1) if recs else 0
    begin = Record(step=first, state=FALLBACK, clause="",
                   action={"kind": "tool", "name": BEGIN_TOOL, "input": {}},
                   output={}, vars=dict(inp))
    return trace.model_copy(update={"records": [begin] + recs})


def with_input(trace: Trace, keys: Sequence[str]) -> Trace:
    """Mirror the top-level ``keys`` of the task dict into ``task["input"]`` (compile view, not written back to disk).

    Older traces have only ``task_id/problem/answer_ref/…`` in their header and no ``input`` -- so
    ``problem`` is not a declarable variable, and judge actions introduced from the document cannot
    read the problem statement. Reference answers (``answer_ref``/``answer``) are **never
    mirrored**: ``run_task`` spreads ``task["input"]`` into the variable table, and once the gold
    answer is in the variables it could be learned as a condition that reads the answer.
    Idempotent.
    """
    task = trace.task if isinstance(trace.task, dict) else {}
    inp = dict(task.get("input") or {}) if isinstance(task.get("input"), dict) else {}
    added = False
    for k in keys:
        if k in ("answer", "answer_ref", "solution", "golden"):
            continue
        if k not in inp and k in task:
            inp[k] = task[k]
            added = True
    if not added and "input" in task:
        return trace
    return trace.model_copy(update={"task": {**task, "input": inp}})


# --------------------------------------------------------------------------- #
# Programmatic labelers: gold labels for judge actions introduced from the document
# --------------------------------------------------------------------------- #
#: ``gold_from`` name -> ``(trace, i) -> label | None``. ``i`` is the index of the record
#: **before** the judge in the trace (a zero-width judge consumes no record). Labelers are pure
#: functions, reviewed like prohibitions; **a model never produces gold labels**. Replay
#: (replay.walk) and calibration (the samples of fit.calibrate) use the same table, so the two
#: sides agree by construction.
LABELERS: dict[str, Callable[[Trace, int], Optional[str]]] = {}


def register_labeler(name: str) -> Callable:
    """Decorator: register a ``(trace, i) -> label | None`` function as a labeler."""
    def deco(fn: Callable[[Trace, int], Optional[str]]):
        LABELERS[name] = fn
        return fn
    return deco


def _verify_records_after(trace: Trace, i: int) -> list:
    """Check records after ``i``. Programmatic labeling of judge actions looks at their results."""
    return [r for r in trace.records[i + 1:]
            if (r.action or {}).get("kind") == "tool"
            and (r.action or {}).get("name") in VERIFY_TOOLS]


@register_labeler("next_action")
def next_action_label(trace: Trace, i: int) -> Optional[str]:
    """The **action label** of the next step: ``tool:bash/apply`` / ``model`` / ``judge`` / ``end:done``.

    This is the cornerstone labeler of the "split first, merges must be verified" approach. For a
    judge action inserted at a branch point, its gold label is "what the trace does next" -- read
    directly from the trace by a program; a model never produces gold labels. With it, the
    outgoing conditions of a branch are all ``v == '<label>'``, pairwise exclusive by construction,
    so equivalence can be proven.

    Following the labelers' common convention, ``i`` is the index of the record **before** this
    judge; the step to label is ``i + 1``.
    """
    recs = list(trace.records or ())
    j = i + 1
    if j < 0 or j >= len(recs):
        return None
    return branch_label(recs[j])          # pass the **record**: a bare action has no output, writes would come out empty, and the fingerprint would not match the tree-building side


def branch_key(rec_or_action: Any) -> tuple:
    """The identity key of a step, **without prompt or question text**.

    The strict mode of normalization includes a model step's prompt and a judge step's question in
    the key. Those two pieces of text come from the machine that produced the trace, so the same
    step gets different keys under different machine versions. Identity should not depend on
    whoever produced the trace, so they are removed here. The remaining components are the action
    kind, tool name, phase and written variables, all properties of the step itself.
    """
    key = canon_action(rec_or_action, strict=True)
    return tuple(x for x in key if not (x.startswith("prompt=") or x.startswith("question=")))


def branch_label(rec_or_action: Any) -> str:
    """The branch label of a step, translated directly from :func:`branch_key` into readable form.

    A label must satisfy three conditions. It depends only on the step itself, because the program
    reading the label only sees the trace. It gives different values for different keys, otherwise
    two successors share a label and one of them becomes dead code. It must be understandable by a
    model, because at run time the model picks one of these labels.
    """
    key = branch_key(rec_or_action)
    parts = {k.split("=", 1)[0]: k.split("=", 1)[1] for k in key[1:] if "=" in k}
    kind = key[0] if key else "?"
    if kind == "tool":
        name, ph = parts.get("name", ""), parts.get("phase", "")
        return f"{name}/{ph}" if ph else name
    if kind == "end":
        return f"结束:{parts.get('terminal', 'done')}"
    writes = parts.get("writes", "")
    return f"{kind}→{writes}" if writes else kind


@register_labeler("next_action_after")
def next_action_after(trace: Trace, i: int) -> Optional[str]:
    """Which kind of action is the first step after ``i`` (:func:`branch_label`) -- the most general
    programmatic gold label for a judge of the "the document says to judge here" kind: the gold
    label is the trace's own future, and the reference answer is never consulted. No record after
    ``i`` => ``None``.
    """
    recs = trace.records
    if i + 1 >= len(recs) or i < -1:
        return None
    return branch_label(recs[i + 1])


@register_labeler("fail_attr_from_trace")
def fail_attr_from_trace(trace: Trace, i: int) -> Optional[str]:
    """Failure attribution after a check FAIL (the two failure-attribution labels), inferred backwards from **what happened afterwards**:

    * at the next check ``candidate`` is unchanged but the result is PASS => the answer was right
      all along, and the check itself was written wrong: label ``核验写错了``;
    * at the next check ``candidate`` has changed => the model changed its answer, so the solution
      was wrong: label ``解答有错``;
    * no next check, or nothing can be told => ``None`` (abstain).

    Only meaningful when the immediately preceding record is a check step with
    ``verify_status != PASS``; otherwise ``None``.
    """
    recs = trace.records
    if i < 0 or i >= len(recs):
        return None
    here = recs[i]
    if (here.action or {}).get("name") not in VERIFY_TOOLS:
        return None
    if (here.vars or {}).get("verify_status") in ("", PASS):
        return None
    cand_before = (here.vars or {}).get("candidate")
    later = _verify_records_after(trace, i)
    if not later:
        return None
    nxt = later[0]
    cand_after = (nxt.vars or {}).get("candidate")
    if cand_after != cand_before:
        return "解答有错"
    if (nxt.vars or {}).get("verify_status") == PASS:
        return "核验写错了"
    return None


def _clip(text: str, limit: int) -> str:
    """Clip and **leave a trace of the clipping**: the ellipsis is followed by the original length, so clipping is never later mistaken for the tool really having no output."""
    s = text or ""
    if len(s) <= limit:
        return s
    return s[:limit] + f"…[truncated, {len(s)} chars total]"


def _first_answer(args: dict) -> Optional[str]:
    """Take the answer text from a step's arguments; None if there is none."""
    for key in ANSWER_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, (str, int, float)) and str(v).strip():
            return str(v)
    return None


def _canon_step_name(raw_name: str) -> str:
    """Canonical form of a tool name: first :func:`canon_tool_name`, then submit aliases are funnelled into ``submit_answer``."""
    name = canon_tool_name(raw_name)
    return SUBMIT_TOOL if name in SUBMIT_ALIASES else name


# --------------------------------------------------------------------------- #
# Offline storage of code bodies
# --------------------------------------------------------------------------- #
def _extract_code(inp: dict) -> tuple[dict, dict]:
    """Replace code bodies in the arguments with hashes; returns ``(new arguments, {sha: body})``."""
    out = dict(inp)
    bodies: dict[str, str] = {}
    for key in CODE_KEYS:
        v = out.get(key)
        if not isinstance(v, str) or not v.strip():
            continue
        sha = hashlib.sha256(v.encode("utf-8")).hexdigest()
        out.pop(key)
        out[f"{key}_sha256"] = sha
        out[f"{key}_path"] = f"{ARTIFACTS_DIRNAME}/{sha}.py"
        out[f"{key}_bytes"] = len(v.encode("utf-8"))
        bodies[sha] = v
    return out, bodies


def _spill(bodies: dict, artifacts_dir: Any) -> None:
    """Write bodies as ``<artifacts_dir>/<sha>.py``. Same name means same content (the hash is the file name), so existing files are not rewritten."""
    if not bodies:
        return
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    for sha, body in bodies.items():
        p = root / f"{sha}.py"
        if not p.exists():
            p.write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Converting a single step
# --------------------------------------------------------------------------- #
def _record_for(step: RawStep, idx: int, values: dict, *,
                artifacts_dir: Any, phase_rules: str = "",
                outputs: tuple = ()) -> tuple[Record, Optional[VerifyOutcome]]:
    """Fold one :class:`RawStep` into a :class:`~hexis.machine.schema.Record`, updating ``values`` along the way.

    The keys of ``output`` **are the names of the variables this step writes**:
    ``normalize.action_writes`` infers writes from the output keys (except ``ok``/``error``), and
    the compiler labels the state's writes from that. So a check step's output uses the full names
    ``verify_status``/``verify_exit``/``verify_stdout`` rather than ``status`` -- otherwise the
    state declaration would say ``status`` while the condition reads ``verify_status``, and the
    two would not match.
    """
    kind = (step.kind or "").strip().lower()
    meta = dict(step.meta or {})
    state = str(meta.pop("state", "") or FALLBACK)
    outcome: Optional[VerifyOutcome] = None

    if kind == "tool":
        name = _canon_step_name(step.name)
        raw_args = dict(step.args or {})
        # The phase must be classified **before the body is moved out**: after _extract_code the
        # record only has code_sha256, and a hash cannot tell what this step is doing (see the
        # module docs of hexis.traces.phases).
        phase = (_phases.classify(name, raw_args, phase_rules, outputs=outputs)
                 if phase_rules else "")
        inp, bodies = _extract_code(raw_args)
        if bodies:
            if artifacts_dir is not None:
                _spill(bodies, artifacts_dir)
            else:
                meta["code_bodies"] = dict(bodies)   # moved out by write_jsonl when writing to disk
        action = {"kind": "tool", "name": name, "input": inp}
        if phase:
            action["phase"] = phase

        if name == SUBMIT_TOOL:
            output = _submit_output(step, values)
        elif name in VERIFY_TOOLS:
            output, outcome = _verify_output(step, values, meta)
        else:
            output = {"ok": (step.returncode in (None, 0)),
                      "stdout": _clip(step.stdout, TOOL_STDOUT_MAX)}
            # candidate answer on the table: tools other than checks may carry it too (e.g. solve's --expected)
            cand = _first_answer(inp)
            if cand is not None:
                values["candidate"] = cand
        if step.returncode is not None:
            meta.setdefault("returncode", step.returncode)
        if (step.stderr or "").strip():
            meta.setdefault("stderr", _clip(step.stderr, TOOL_STDOUT_MAX))
        if "last_tool" in values:                # only recorded when to_trace enabled model-side snapshots
            values["last_tool"] = name
            values["last_tool_ok"] = bool(output.get("ok", step.returncode in (None, 0)))
    elif kind == "model":
        # The model body goes into output rather than meta: text prohibitions like
        # ``absent``/``regex`` only scan action + output (see judge._record_text); hiding it in
        # meta would mean never finding out what the model said.
        action = {"kind": "model"}
        output = {"reply": _clip(step.text or step.stdout, REPLY_MAX)}
        if "last_reply" in values:
            values["last_reply"] = output["reply"]
    elif kind == "judge":
        args = dict(step.args or {})
        writes = [str(w) for w in (args.get("writes") or [])]
        label = args.get("label", step.text or "")
        action = {"kind": "judge",
                  "prompt": str(args.get("prompt", "")),
                  "reads": [str(r) for r in (args.get("reads") or [])]}
        output = {w: label for w in writes}
        values.update(output)
    elif kind == "end":
        terminal = str((step.args or {}).get("terminal") or step.name or "done")
        action = {"kind": "end", "terminal": terminal}
        output = {}
    elif kind == "user":
        # User input: what was asked is in action.prompt, what the user answered is in output.answer
        args = dict(step.args or {})
        action = {"kind": "user", "prompt": str(args.get("prompt") or step.text or "")}
        output = {"answer": _clip(str(args.get("answer") or step.stdout or ""), REPLY_MAX)}
    else:
        raise TraceAdapterError(
            f"unrecognized event: step {idx} kind={step.kind!r} (expected model / tool / judge / user / end); "
            f"this log needs its own adapter; events must not be dropped and the rest treated as compiled")

    return Record(step=idx, state=state, clause="", action=action,
                  output=output, vars=dict(values), meta=meta), outcome


def _verify_output(step: RawStep, values: dict, meta: dict
                   ) -> tuple[dict, VerifyOutcome]:
    """Check step: parse the ruling, update ``repair_count`` and the three ``verify_*`` per the variable table."""
    args = dict(step.args or {})
    argv = args.get("argv") or (step.meta or {}).get("argv") or ()
    if isinstance(argv, str):
        argv = argv.split()
    outcome = parse_verify(step.stdout, step.returncode, argv=argv,
                           timed_out=(step.meta or {}).get("timed_out"))
    prev = values.get("verify_status") or ""
    if prev and prev != PASS:
        # The previous check did not pass and another check ran => one round of repair in between.
        values["repair_count"] = int(values.get("repair_count") or 0) + 1
    cand = _first_answer(args)
    if cand is not None:
        values["candidate"] = cand
    output = {
        "ok": outcome.ok,
        "verify_status": outcome.status,
        "verify_exit": (step.returncode if isinstance(step.returncode, int) else -1),
        "verify_stdout": _clip(step.stdout, VERIFY_STDOUT_MAX),
    }
    values["verify_status"] = output["verify_status"]
    values["verify_exit"] = output["verify_exit"]
    values["verify_stdout"] = output["verify_stdout"]
    meta["verify"] = dict(outcome.detail)
    return output, outcome


def _submit_output(step: RawStep, values: dict) -> dict:
    """Submit step: write ``answer`` (and ``candidate``), and carry the self-reported ``verified`` marker into output.

    ``verified`` is one of the three clues recognised by :func:`hexis.traces.judge.terminal_kind`:
    a run forced to submit because its budget was exhausted is marked ``False``, so P1's
    ``only_when: {terminal_kind: verified}`` does not fire on it. **When the category cannot be
    determined, P1 is still checked** (the conservative direction of judge._kind_matches), so this
    marker is written only when the collection side really knows; it never guesses.
    """
    args = dict(step.args or {})
    text = _first_answer(args)
    if text is None:
        text = (step.text or "").strip()
    output: dict[str, Any] = {"ok": True, "answer": text}
    verified = args.get("verified", (step.meta or {}).get("verified"))
    if isinstance(verified, bool):
        output["verified"] = verified
    values["answer"] = text
    values["candidate"] = text
    return output


# --------------------------------------------------------------------------- #
# Whole runs
# --------------------------------------------------------------------------- #
def to_trace(raw: RawRun, *, acceptance: Optional[Callable[[Any], bool]] = None,
             prohibitions: Iterable[Prohibition] = (),
             artifacts_dir: Any = None,
             snapshot_model_vars: bool = False,
             phase_rules: str = "") -> Trace:
    """Fold one raw run into a judged :class:`~hexis.machine.schema.Trace`.

    * step numbers are consecutive from 1, consistent with ``runtime.run_task``;
    * ``state`` defaults to :data:`~hexis.machine.schema.FALLBACK` -- arms one and two are
      interpretive execution throughout, so every step is a fallback step; arm three has
      ``runtime`` write its own state names and never gets here. The collection side overrides it
      via ``step.meta["state"]``;
    * ``clause`` is always left empty: clause attribution is filled in by the compile agent during
      transcription, and guessing one at collection time would only fabricate provenance;
    * the update rules of ``vars`` are in the variable table in the module docs.

    **How the verdict is decided**:

    1. any prohibition violated => ``rejected``, ``error_step`` = the step where the violation
       happened (given by :func:`hexis.traces.judge.evaluate`). Prohibitions come first, even if
       the answer is right;
    2. otherwise, if ``acceptance`` was injected and says no => ``rejected``, with ``error_step``
       by the priority below;
    3. otherwise, if ``acceptance`` was injected and says yes => ``accepted``;
    4. **no ``acceptance`` injected** => ``unknown``. Writing ``accepted`` without objective
       acceptance would mix unchecked traces into the accepted set -- this differs from the
       default of :func:`hexis.traces.judge.judged`, deliberately.

    **Priority of ``error_step``** (``rejected`` must carry it, otherwise ``schema.Trace`` raises):

    1. the step of the prohibition violation -- precisely located, and it is exactly the anchor
       the rejection-set exclusion check watches;
    2. **the first divergence**: the earliest check step with ``verify_status != PASS``. This is
       the first observable evidence inside the run that "something is off";
    3. the last step -- the fallback when there is no clue at all (wrong from start to finish, all
       one can point at is the outcome);
    4. an empty run with no steps records ``0`` (before step 1), because ``schema`` does not allow
       rejected without error_step.
    """
    task_in = (raw.task or {}).get("input") if isinstance(raw.task, dict) else None
    outputs = _phases.outputs_of(task_in or {})     # outputs are **declared by the task**, not guessed from file names by the classifier
    values = initial_vars()
    if snapshot_model_vars:
        # Model-side snapshots (MODEL_VARS): judge actions introduced from the document read them,
        # and calibration uses them as samples. Off by default: "every record carries exactly the
        # six variables of initial_vars" is the contract of the single-agent path and of existing
        # traces; the multi-agent collection path turns it on explicitly.
        values.update(model_vars())
    records: list[Record] = []
    first_divergence: Optional[int] = None

    for idx, step in enumerate(raw.steps or (), start=1):
        rec, outcome = _record_for(step, idx, values, artifacts_dir=artifacts_dir,
                                   phase_rules=phase_rules, outputs=outputs)
        records.append(rec)
        if outcome is not None and outcome.status != PASS and first_divergence is None:
            first_divergence = idx

    draft = Trace(task=dict(raw.task or {}), arm=raw.arm, run=raw.run,
                  model=raw.model, harness=raw.harness,
                  verdict="unknown", records=records)
    check_canonical(draft)

    plist = list(prohibitions or ())
    banned = _judge.evaluate(draft, None, plist)      # prohibitions only: acceptance is left for the next step
    if banned.verdict == "rejected":
        verdict, error_step = "rejected", banned.error_step
    elif acceptance is None:
        verdict, error_step = "unknown", None
    elif acceptance(draft):
        verdict, error_step = "accepted", None
    else:
        last = records[-1].step if records else 0
        verdict, error_step = "rejected", (first_divergence
                                           if first_divergence is not None else last)

    return Trace(task=draft.task, arm=draft.arm, run=draft.run, model=draft.model,
                 harness=draft.harness, verdict=verdict, error_step=error_step,
                 records=records)


def check_canonical(trace: Trace) -> None:
    """Check the two hard constraints of cross-arm comparability; raise :class:`TraceAdapterError` on a mismatch.

    1. Every tool name is already canonical (a fixed point of ``canon_tool_name``), and no submit
       alias slipped through: if ``math_verify.py`` and ``math_verify`` are written differently in
       two arms, the compiler opens two states that never form a loop, and replay and path
       agreement are immediately distorted;
    2. the submit step is named only :data:`SUBMIT_TOOL`. P1's ``require_before`` guards exactly
       this name; renaming it switches off the violation check for that arm -- and the report would
       show "violation rate 0%" without revealing that it was switched off.

    Also checks that step numbers are consecutive from 1 (the compiler and replay both locate
    ``error_step`` by index). A view with a begin step prepended by :func:`with_begin` (the first
    record is BEGIN_TOOL at step 0) is equally valid.
    """
    recs = list(trace.records or ())
    if recs and is_begin(recs[0].action) and recs[0].step == 0:
        recs = recs[1:]
    for i, rec in enumerate(recs, start=1):
        if rec.step != i:
            raise TraceAdapterError(
                f"non-consecutive step numbers: record {i} has step={rec.step} (must be consecutive from 1)")
        act = rec.action or {}
        if act.get("kind") != "tool":
            continue
        name = str(act.get("name") or "")
        if name != canon_tool_name(name):
            raise TraceAdapterError(
                f"tool name {name!r} at step {i} is not canonical"
                f" (expected {canon_tool_name(name)!r}); arms would no longer line up")
        if name in SUBMIT_ALIASES and name != SUBMIT_TOOL:
            raise TraceAdapterError(
                f"the submit action at step {i} is named {name!r} and must be unified to {SUBMIT_TOOL!r}, "
                f"otherwise P1's require_before never fires on this arm")


# --------------------------------------------------------------------------- #
# Writing to and reading back from disk
# --------------------------------------------------------------------------- #
_UNSAFE_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _stem(trace: Trace, index: int) -> str:
    """Trace file name: ``<task_id>__<arm>__run<NN>``; missing fields are skipped, illegal characters fold into ``_``."""
    task = trace.task if isinstance(trace.task, dict) else {}
    parts = [str(task.get("task_id") or f"trace{index:04d}")]
    if trace.arm:
        parts.append(str(trace.arm))
    parts.append(f"run{int(trace.run):02d}")
    return _UNSAFE_RE.sub("_", "__".join(parts)).strip("_") or f"trace{index:04d}"


def write_jsonl(traces: Iterable[Trace], out_dir: Any) -> list[Path]:
    """Write traces one by one as ``out_dir/<stem>.jsonl``; returns the written paths (in input order).

    Code bodies not yet moved out (parked in ``Record.meta["code_bodies"]``) are moved to
    ``out_dir/artifacts/<sha>.py`` here as well, and the written JSONL keeps only the hashes.
    ``meta`` takes no part in normalization or judging, so moving or not **never changes any
    verdict** -- the copy on disk judges the same as the one in memory.

    **The inputs are not modified**: moving happens on a deep copy. When names collide (same task,
    same arm, same run) a ``-2``, ``-3`` suffix is added to the file; never overwrite silently --
    what gets overwritten is a trace that cannot be collected again.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    used: set[str] = set()
    for i, trace in enumerate(traces, start=1):
        copy = trace.model_copy(deep=True)
        bodies: dict[str, str] = {}
        for rec in copy.records:
            got = rec.meta.pop("code_bodies", None)
            if isinstance(got, dict):
                bodies.update({str(k): str(v) for k, v in got.items()})
        _spill(bodies, root / ARTIFACTS_DIRNAME)

        stem = _stem(copy, i)
        name, n = stem, 1
        while name in used:
            n += 1
            name = f"{stem}-{n}"
        used.add(name)
        path = root / f"{name}.jsonl"
        path.write_text(copy.to_jsonl(), encoding="utf-8")
        paths.append(path)
    return paths


def read_jsonl(path: Any) -> Trace:
    """Read one trace back. If the file does not exist, say clearly **which path**; don't make the caller guess."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"trace file does not exist: {p}")
    return Trace.from_jsonl(p)


# --------------------------------------------------------------------------- #
# Generic agent traces: records produced by any harness can enter the compiler
# --------------------------------------------------------------------------- #
def ensure_phases(trace: Trace, rules: str = "default") -> Trace:
    """Assign phases (probe / apply / verify / other) to tool steps from the **command body**. Modifies ``action`` in place; idempotent.

    The phase is a function of the content, so as long as the body is still there it is
    **recomputed**, regardless of what the record said before. This rule was learned the hard way:
    ``runtime`` once copied the state's own declaration into the record, so the trace said "what
    the machine thought this step should do" rather than "what this step did", and compilation
    became circular -- the machine only relearned the labels it had attached itself. In one
    measured 4-step trace three steps were mislabelled (a state declared probe had ``wb.save`` in
    its command, actually apply; a state declared apply only read back the output, actually
    verify).

    When the body is unavailable (``_extract_code`` moved it out, leaving only ``code_sha256``),
    the existing label is kept -- a hash cannot tell what this step is doing, so keeping the old
    value is better than changing it blindly. For specialised tools (name is purpose) the
    classifier returns the empty string, and nothing is changed either.
    """
    task = trace.task if isinstance(trace.task, dict) else {}
    outputs = _phases.outputs_of(task.get("input") or {})
    for rec in trace.records:
        act = rec.action
        if not isinstance(act, dict) or act.get("kind") != "tool":
            continue
        inp = dict(act.get("input") or {})
        if not _phases.command_text(inp).strip() and act.get("phase"):
            continue                       # the body is gone; the old label is the only clue
        p = _phases.classify(str(act.get("name") or ""), inp, rules, outputs=outputs)
        if p:
            act["phase"] = p
    return trace


def read_raw_jsonl(path: Any) -> tuple[RawRun, Optional[bool]]:
    """Read a **raw agent event log** (not this repository's Trace format) and fold it into a :class:`RawRun`.

    The first line is the header: ``{"task": {...}, "verdict": "accepted"|"rejected"}`` (also
    accepts ``"ok": true/false``, or ``"input": {...}`` taken directly as the task input). Every
    other line is one step, with fields named as in :class:`RawStep`::

        {"kind": "tool",  "name": "bash",     "args": {"command": "ls"}, "stdout": "...", "returncode": 0}
        {"kind": "tool",  "name": "file_ops", "args": {"op": "read", "path": "a.xlsx"}, "stdout": "..."}
        {"kind": "model", "text": "..."}
        {"kind": "end"}

    Returns ``(RawRun, verdict)``; a verdict of ``None`` means the log does not say whether it was
    right (it enters neither T+ nor T−).
    """
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise TraceAdapterError(f"empty log: {p}")
    head = json.loads(lines[0])
    if not isinstance(head, dict):
        raise TraceAdapterError(f"first line of {p} is not a JSON object")
    task = dict(head.get("task") or {})
    if "input" in head and "input" not in task:
        task["input"] = dict(head["input"] or {})
    task.setdefault("task_id", str(head.get("task_id") or p.stem))
    verdict: Optional[bool] = None
    if "verdict" in head:
        verdict = str(head["verdict"]).lower() == "accepted"
    elif "ok" in head:
        verdict = bool(head["ok"])
    steps: list[RawStep] = []
    for i, ln in enumerate(lines[1:], start=1):
        d = json.loads(ln)
        if not isinstance(d, dict) or not d.get("kind"):
            raise TraceAdapterError(f"step {i} of {p} is missing kind")
        steps.append(RawStep(kind=str(d["kind"]), name=str(d.get("name") or ""),
                             args=dict(d.get("args") or d.get("input") or {}),
                             stdout=str(d.get("stdout") or d.get("output") or ""),
                             stderr=str(d.get("stderr") or ""),
                             returncode=d.get("returncode"),
                             text=str(d.get("text") or ""), meta=dict(d.get("meta") or {})))
    return RawRun(task=task, steps=steps, arm=str(head.get("arm") or "agent"),
                  run=int(head.get("run") or 0), model=str(head.get("model") or ""),
                  harness=str(head.get("harness") or "")), verdict


def tool_output(rec: Any) -> dict:
    """The complete output of a tool record: ``output`` plus the status fields put into ``meta`` at collection time (return code, stderr).

    This is knowledge of the trace **format** and lives only in the adapter: the compiler only sees
    the merged output dict and does not know which fields came from meta.
    """
    out = dict(getattr(rec, "output", None) or {})
    meta = getattr(rec, "meta", None) or {}
    for k in ("returncode", "stderr"):
        if k in meta and k not in out:
            out[k] = meta[k]
    return out


def load_any_trace(path: Any, *, phase_rules: str = "default",
                   artifacts_dir: Any = None) -> Trace:
    """Read a trace file, accepting either this repository's Trace format or a raw agent log, and fill in phases.

    Decided by the second line: Trace record lines have ``step``/``action``, raw log step lines
    have ``kind``.
    """
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    second = json.loads(lines[1]) if len(lines) > 1 else {}
    if isinstance(second, dict) and "action" in second and "step" in second:
        tr = read_jsonl(p)
    else:
        raw, ok = read_raw_jsonl(p)
        tr = to_trace(raw, acceptance=(None if ok is None else (lambda _t, _ok=ok: _ok)),
                      artifacts_dir=artifacts_dir, phase_rules=phase_rules)
    return ensure_phases(tr, phase_rules) if phase_rules else tr
