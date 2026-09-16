"""Deterministic gatekeeper (part 1): the receipted machine-edit interface. **This module does not call a model.**

There is not a single ``model`` parameter, no model client is imported and nothing is read from
the network: whether a machine should be changed into some shape is a purely deterministic graph
and data-flow question and should not be settled by a single sample. The compile agent (built on
top of this layer) can only change the machine through the eight receipted interfaces here; it is
**not allowed** to modify ``Machine`` directly or to write machine files.

**Why this layer exists.** :func:`hexis.legacy.compiler.compile_round` can already "compile a
whole round and undo the whole round if the structural checks fail". But an agentic compile
process is not "one round": it is a series of small proposals (add a state, connect an edge, turn
this branch into a judge action, give this loop a counter). Undoing a whole round at that
granularity is a disaster: if proposal 7 is wrong, the 6 correct proposals before it are lost too,
the agent has to start over, and it does not know which one was wrong. So the rule here is the
other way round:

* **Each proposal is ruled on separately**: it is applied to a **candidate copy** first and the
  checks run on it; only if they pass does the copy become current, otherwise the current machine
  is not touched at all and there is just one more receipt stating the reason (:class:`Receipt`).
* **A rejected proposal does not affect proposals accepted before it.** This is exactly what
  ``compile_round`` cannot express.
* Only :meth:`Checker.commit` is all-or-nothing: it runs acceptance (:mod:`hexis.legacy.verify`)
  and, if that fails, rolls the whole batch back to the machine as of the last commit (or open).

**A receipt must be actionable for a retry.** The rejection reason has to say **which check**
failed on **which state**, otherwise the agent can only guess. So the diagnostic sentences
returned by the structural checks are parsed here into :class:`Finding` objects with
``code``/``state_id``, and the reason string carries a location prefix such as ``[E_OVERLAP@j]``.

The checks themselves are **not reimplemented**: the eight graph and data-flow checks in
:mod:`hexis.machine.checks` are already right (reachability, termination, transition completeness,
finite-configuration enumeration for mutual exclusion and completeness, back-edge bounds, write
before read, output completeness). This module only wraps them and adds two gates they do not
cover (transition support, judge action error rate) plus one they miss (a state has at most one
default edge).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from pydantic import ValidationError

#: File name of the provenance table written next to machine.json on commit. Provenance does
#: **not** go into machine.json: the executable artifact must be byte-stable and private;
#: provenance is for humans and the ledger.
PROVENANCE_FILE = "provenance.json"

from hexis.legacy import fit as _fit
from hexis.machine import cond as _cond
from hexis.machine.checks import _back_edges as _checks_back_edges_raw
from hexis.machine.checks import _reachable as _checks_reachable
from hexis.machine.checks import structural_findings
from hexis.machine.schema import (
    ABSTAIN,
    EndAction,
    JudgeAction,
    Machine,
    Prohibition,
    State,
    Terminal,
    Thresholds,
    Transition,
    Variable,
    empty_machine,
    save_machine,
)

#: Names of the receipted interfaces. The ``op`` of a :class:`Proposal` must be one of them.
#: The first eight are the original eight; the rest (``split_state``, ``mark``, ``rewind``, ...)
#: were added for multi-agent compilation (split by predecessor, set a checkpoint, rewind to a
#: checkpoint) and are likewise ruled on via a candidate copy and receipted.
OPS: tuple[str, ...] = (
    "open_machine", "add_state", "add_transition", "close_loop",
    "add_judge", "set_terminal", "demote_to_fallback", "commit",
    "split_state", "close_loops", "bound_loop", "mark", "rewind",
)

#: Value domain of a provenance row's ``origin``. The ledger groups rows by it: which structures
#: come from the document, which were learned from traces, which the compiler decided on its own,
#: and which enumeration domains have not yet been calibrated on a real executor.
ORIGINS: tuple[str, ...] = (
    "document", "document(弱)", "trace", "compiler",
    "harness(待标定)", "harness(已标定)",
)


def prov_key(kind: str, *parts: str) -> str:
    """Provenance table key: ``state:s1`` / ``edge:s1->s2#cond`` / ``judge:s3`` / ``var:x`` /
    ``terminal:END_VERIFIED`` / ``prohibition:P1`` / ``domain:audit_status``."""
    if kind == "edge":
        src, dst, cond = parts
        return f"edge:{src}->{dst}#{cond}"
    return f"{kind}:{parts[0]}"


# --------------------------------------------------------------------------- #
# Findings and receipts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Finding:
    """One check result. ``severity`` ∈ ``{"error", "warn"}``.

    ``error`` means **the machine is broken** (unreachable, cannot stop, overlapping guards, reads
    an unwritten variable, ...); the receipted interfaces reject the proposal when they see one.
    ``warn`` means **not enough evidence** (this edge was taken by only one trace, this judge's
    error rate is over the cap); the machine structure itself is fine and the edit still lands,
    but it will not pass acceptance in :mod:`hexis.legacy.verify`.

    ``state_id`` is the state where the problem is (empty string when it cannot be located),
    ``code`` is for programmatic dispatch, and ``message`` is the original diagnostic sentence (the
    eight structural check messages are passed through verbatim).
    """

    code: str
    severity: str
    state_id: str
    message: str

    def located(self) -> str:
        """``[E_OVERLAP@j] state j has overlapping guards ...``: one line with a location prefix."""
        where = f"@{self.state_id}" if self.state_id else ""
        return f"[{self.code}{where}] {self.message}"


@dataclass(frozen=True)
class Receipt:
    """A receipt: the audit trail of one edit proposal. Both acceptance and rejection produce one,
    and **rollback does not erase receipts**.

    On acceptance ``reason`` says what changed; on rejection it says which check failed where.
    ``detail`` holds machine-readable attachments (proposal arguments, findings as dicts, the
    acceptance report).
    """

    op: str
    accepted: bool
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    """An edit proposal awaiting a ruling: ``op`` is the receipted interface name, ``args`` its keyword arguments."""

    op: str
    args: dict = field(default_factory=dict)


class _Reject(Exception):
    """A precondition is not met (missing state, duplicate edge, ...). Carries an actionable reason."""

    def __init__(self, reason: str, detail: Optional[dict] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


# --------------------------------------------------------------------------- #
# Structural check diagnostic sentences → Finding with code/state
# --------------------------------------------------------------------------- #
#: ``(regex, code)``. The ``sid`` group in the regex gives the state with the problem. Order is match priority.
_STRUCT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^initial points to a missing state", "E_INITIAL_MISSING"),
    (r"^transition (?P<sid>[^\s→]+)→[^\s]+: \S+ points to a missing state", "E_DANGLING_EDGE"),
    (r"^state (?P<sid>\S+) is unreachable from \S+", "E_UNREACHABLE"),
    (r"^no end state", "E_NO_TERMINAL"),
    (r"^state (?P<sid>\S+) cannot reach any end state", "E_NO_STOP"),
    (r"^state (?P<sid>\S+) has no outgoing transitions", "E_NO_EDGE"),
    (r"^state (?P<sid>\S+) has only guarded transitions", "E_NO_DEFAULT"),
    (r"^state (?P<sid>\S+) has too many guard atoms", "E_ATOM_CAP"),
    (r"^state (?P<sid>\S+) uses variables without a finite domain", "E_UNDECIDABLE_VAR"),
    (r"^state (?P<sid>\S+) has overlapping guards", "E_OVERLAP"),
    (r"^state (?P<sid>\S+) has a gap in its guards", "E_GAP"),
    (r"^back edge (?P<sid>[^\s→]+)→[^\s]+ has no counter", "E_LOOP_UNCOUNTED"),
    (r"^back edge (?P<sid>[^\s→]+)→[^\s]+: counter", "E_LOOP_UNBOUNDED"),
    (r"^state (?P<sid>\S+) reads ", "E_READ_BEFORE_WRITE"),
    (r"^end state (?P<sid>\S+) declares outputs", "E_OUTPUT_MISSING"),
)


def classify(message: str) -> tuple[str, str]:
    """Map a structural diagnostic sentence to ``(code, state_id)``. Unrecognized: ``E_STRUCT`` + empty state."""
    for pat, code in _STRUCT_PATTERNS:
        m = re.match(pat, message)
        if m:
            sid = (m.groupdict().get("sid") or "") if m.groupdict() else ""
            return code, sid
    return "E_STRUCT", ""


# --------------------------------------------------------------------------- #
# Three gates the structural checks do not cover
# --------------------------------------------------------------------------- #
def _default_edges(state: State) -> list[Transition]:
    return [t for t in state.transitions if not t.cond]


def _multi_default_findings(m: Machine) -> list[Finding]:
    """A state has at most one default edge.

    :func:`hexis.execution.runtime.pick_edge` takes the first true edge in declaration order and a
    default edge is always true, so a second default edge is **never reachable**. The eight
    structural checks do not cover this (they only ask "is there a default edge"), but two default
    edges in one machine almost always mean "should have rewritten the existing one, added another
    instead", silently burying the newly connected branch. So it is an error here.
    """
    out: list[Finding] = []
    for sid in sorted(m.states):
        st = m.states[sid]
        if st.action.kind == "end":
            continue
        dflt = _default_edges(st)
        if len(dflt) > 1:
            out.append(Finding(
                "E_MULTI_DEFAULT", "error", sid,
                f"state {sid} has {len(dflt)} default edges (unguarded outgoing edges), targets "
                f"{[t.to for t in dflt]}: only one default edge is allowed and the second one "
                "onward is never reachable; rewrite the existing one, or give the new edge a guard"))
    return out


def _support_rows(m: Machine, thr: Thresholds) -> list[tuple[str, Transition]]:
    """Edges with insufficient support. Three kinds are exempt.

    * **Edges into FALLBACK and FALLBACK's own outgoing edges**: going to FALLBACK means "I did not
      learn this part, hand it back to interpreted execution". By definition it has no trace
      support, and demanding support for it amounts to forbidding the compiler from admitting it
      is stuck;
    * edges with ``origin="document"``: those are claims of the **document skeleton**, not compiled
      from traces. Demanding trace support for them amounts to saying "a branch the document
      describes but this batch of traces happened not to take must not exist", and error handling
      and edge cases would disappear wholesale. Their lack of evidence is a fact, recorded in the
      "in the document, not seen in traces" column, not a failure.

    Evidence is required for **paths compiled from traces**: only a claim of "I have seen this"
    needs to show how many times it was seen.
    """
    out: list[tuple[str, Transition]] = []
    for sid, t in m.transitions_all():
        if sid == m.fallback or t.to == m.fallback:
            continue
        if str(getattr(t, "origin", "") or "") == "document":
            continue
        if t.support < thr.min_support:
            out.append((sid, t))
    return out


def weak_edges(m: Machine, thresholds: Optional[Thresholds] = None) -> list[tuple[str, str]]:
    """Edges with support below ``min_support``, as ``(state id, guard)`` (a default edge's guard is ``""``)."""
    thr = thresholds or m.thresholds
    return [(sid, t.cond) for sid, t in _support_rows(m, thr)]


def hot_judges(m: Machine, thresholds: Optional[Thresholds] = None) -> list[tuple[str, float]]:
    """Judge actions whose calibrated error rate is above ``judge_err_max``, as ``(state id, error rate)``.

    In the inequality "P(a path errs at least once) ≤ Σεᵢ" every εᵢ needs a ceiling, otherwise any
    single judge on the right-hand side can blow through the bound."""
    thr = thresholds or m.thresholds
    out: list[tuple[str, float]] = []
    for sid in sorted(m.states):
        act = m.states[sid].action
        if act.kind == "judge" and act.error_rate > thr.judge_err_max:
            out.append((sid, act.error_rate))
    return out


def _uncalibrated_judge_findings(m: Machine) -> list[Finding]:
    """A judge action **introduced from the document** with no programmatic gold labeler, or never
    calibrated, may only go to FALLBACK.

    Only ``introduced=True`` judges are covered: judges in a hand-written reference machine are
    **targets**, by definition ``support=0`` yet with labeled edges; they are not compiler products
    and this rule does not apply to them. Compiler-introduced judges are different: an empty
    ``gold_from`` or ``support==0`` means there is **no programmatic evidence at all** that the
    judge is accurate, and letting it branch lets an unmeasured εᵢ into the sum
    "P(a path errs at least once) ≤ Σεᵢ".

    * non-FALLBACK outgoing edges ⇒ ``E_JUDGE_UNCALIBRATED_BRANCH`` (error, the interface rejects it);
    * only edges into FALLBACK ⇒ ``W_JUDGE_UNCALIBRATED`` (warn, the ledger calls out "the document
      requires a decision here and we cannot measure it").
    """
    out: list[Finding] = []
    for sid in sorted(m.states):
        act = m.states[sid].action
        if act.kind != "judge" or not act.introduced:
            continue
        uncal = (not act.gold_from) or act.support <= 0
        if not uncal:
            continue
        branching = [t for t in m.states[sid].transitions if t.to != m.fallback]
        why = ("no programmatic labeler registered (gold_from is empty)" if not act.gold_from
               else f"support={act.support}, not yet calibrated on any trace snapshot")
        if branching:
            out.append(Finding(
                "E_JUDGE_UNCALIBRATED_BRANCH", "error", sid,
                f"introduced judge action {sid}: {why}, yet it has {len(branching)} non-FALLBACK "
                f"outgoing edges {[t.to for t in branching]}: an uncalibrated judge can only be a "
                "FALLBACK boundary, not a decision; give it a programmatic labeler and calibrate "
                "it first, or keep only a default edge"))
        else:
            out.append(Finding(
                "W_JUDGE_UNCALIBRATED", "warn", sid,
                f"introduced judge action {sid}: {why}; kept in the graph as a FALLBACK boundary, "
                "the ledger will call it out"))
    return out


def _verified_ends(m: Machine) -> set[str]:
    """States that end claiming "verified" (excluding FALLBACK itself: interpreted execution is
    not a claim made by the machine)."""
    kinds = {t.id: t.kind for t in m.terminals}
    return {sid for sid, s in m.states.items()
            if s.action.kind == "end" and kinds.get(s.action.terminal) == "verified"
            and sid != m.fallback}


def _verified_terminal_findings(m: Machine) -> list[Finding]:
    """A ``verified`` terminal may only be reached via a **programmatically executed audit tool**.

    Empirical evidence from xlsx: when the model stood in for the audit tool, the machine's
    self-audit on two tasks was **fully anti-correlated** with the gold labels (gold PASS got
    self-audit ERROR, gold FAIL got self-audit PASS and reached END_VERIFIED). The "verified"
    terminal category only carries weight when the verifier is a program, so:

    * machines that do not declare ``audit_tools`` are **not checked here**: hand-written toy
      machines and products of the old single-agent path have no such declaration, and suddenly
      warning on them would only turn every "findings is empty" assertion into noise. A missing
      declaration is judged separately by :func:`audit_declaration_findings`, and multi-agent
      compilation enforces it at open_machine;
    * once ``audit_tools`` is declared, there are three errors:
      - ``E_VERIFIED_UNAUDITED``: starting from initial, a verified terminal is still reachable
        while **bypassing every audit tool state** (the argument used on the reference machine,
        turned into a rule for every machine);
      - ``E_VERIFIED_NOT_VIA_STATUS``: an edge from an audit tool state straight to a verified
        terminal (or its submit state) whose guard reads none of the variables the audit tool
        writes: verification ran, but the branch ignores its result;
      - ``E_P1_AUDIT_MISMATCH``: a ``require_before`` prohibition governing verified terminals
        whose requires has no overlap with ``audit_tools``: the judging side and the structural
        side do not mean the same thing by "verification".
    """
    ends = _verified_ends(m)
    if not ends or not m.audit_tools:
        return []
    out: list[Finding] = []
    audit_states = {sid for sid, s in m.states.items()
                    if s.action.kind == "tool" and s.action.name in m.audit_tools}
    # ① reachability while bypassing audit tool states
    seen, stack = set(), [m.initial]
    if m.initial not in audit_states:
        seen.add(m.initial)
    while stack:
        cur = stack.pop()
        if cur in audit_states:
            continue
        for t in m.out_edges(cur):
            if t.to in audit_states or t.to in seen:
                continue
            seen.add(t.to)
            stack.append(t.to)
    for sid in sorted(ends & seen):
        out.append(Finding(
            "E_VERIFIED_UNAUDITED", "error", sid,
            f"verified terminal {sid} is reachable without passing any audit tool state "
            f"{sorted(audit_states) or '(none)'} (audit_tools={m.audit_tools}): a \"verified\" "
            f"ending must, by construction, be impossible to reach without the programmatic audit"))
    # ② edges audit state → (verified terminal | its submit state) must read variables the audit writes
    feeders = set(ends)
    for src, t in m.transitions_all():
        if t.to in ends and src not in audit_states:
            feeders.add(src)
    for a in sorted(audit_states):
        writes = set(m.states[a].action.writes or [])
        for t in m.states[a].transitions:
            if t.to in feeders and t.to != m.fallback:
                used = _cond.vars_of(t.cond) if t.cond else set()
                if not (used & writes):
                    out.append(Finding(
                        "E_VERIFIED_NOT_VIA_STATUS", "error", a,
                        f"audit tool state {a} has an edge on the verified path → {t.to} (guard "
                        f"{t.cond or '(default)'}) that reads none of the variables it writes "
                        f"{sorted(writes)}: verification ran but the branch ignores its result, "
                        "which is the same as not verifying"))
    # ③ P1 and audit_tools must agree on the same set of tools
    for p in m.prohibitions:
        pat = p.pattern if isinstance(p.pattern, dict) else {}
        if p.check != "require_before":
            continue
        only = pat.get("only_when") or {}
        if only.get("terminal_kind") != "verified":
            continue
        req = set(pat.get("requires") or [])
        if req and not (req & set(m.audit_tools)):
            out.append(Finding(
                "E_P1_AUDIT_MISMATCH", "error", "",
                f"prohibition {p.id} requires running {sorted(req)} before a verified ending, but "
                f"the machine's audit_tools={m.audit_tools} has no overlap with it: the judging "
                "side and the structural side do not treat the same tools as \"verification\""))
    return out


def audit_declaration_findings(m: Machine) -> list[Finding]:
    """Verified terminals without a declared ``audit_tools`` ⇒ ``E_AUDIT_TOOLS_MISSING``.

    Not part of :func:`check_machine` (see the note on :func:`_verified_terminal_findings`); the
    multi-agent compile orchestrator calls it after open_machine and refuses to start if the
    declaration is missing: a machine that claims "verified" but cannot name its verifier is an
    empty promise even once compiled.
    """
    ends = _verified_ends(m)
    if not ends or m.audit_tools:
        return []
    return [Finding(
        "E_AUDIT_TOOLS_MISSING", "error", sorted(ends)[0],
        f"machine has verified terminals {sorted(ends)} but declares no audit_tools: \"verified\" "
        "has no program vouching for it and is an empty claim; declare the canonical names of "
        "the audit tools in open_machine")]


# --------------------------------------------------------------------------- #
# All checks on one machine
# --------------------------------------------------------------------------- #
def check_machine(m: Machine, *, thresholds: Optional[Thresholds] = None) -> list[Finding]:
    """Every problem on a machine that can be decided **without looking at traces**. Empty list = all pass.

    = the eight of :func:`hexis.machine.checks.structural_findings` (all ``error``)
      + "at most one default edge" (``error``)
      + the two gates transition support and judge error rate (``warn``: the structure is fine,
        but it will not pass acceptance).

    If ``thresholds`` is not given, the machine's own ``m.thresholds`` is used.
    """
    thr = thresholds or m.thresholds
    out: list[Finding] = []
    for msg in structural_findings(m):
        code, sid = classify(msg)
        out.append(Finding(code, "error", sid, msg))
    out += _multi_default_findings(m)
    out += _uncalibrated_judge_findings(m)
    out += _verified_terminal_findings(m)
    for sid, t in _support_rows(m, thr):
        out.append(Finding(
            "W_LOW_SUPPORT", "warn", sid,
            f"transition {sid}→{t.to} (guard {t.cond or '(default)'}) has support {t.support} "
            f"< minimum {thr.min_support}: compiling it from so few traces mistakes chance for a rule"))
    for sid, rate in hot_judges(m, thr):
        out.append(Finding(
            "W_JUDGE_ERR", "warn", sid,
            f"judge action {sid} has calibrated error rate {rate} > cap {thr.judge_err_max}: "
            "rewrite the question, add examples and recalibrate, or demote_to_fallback to hand "
            "this branch back to interpreted execution"))
    return out


def _reason_from(findings: Sequence[Finding], head: str) -> str:
    errs = [f for f in findings if f.severity == "error"]
    warns = [f for f in findings if f.severity == "warn"]
    body = "; ".join(f.located() for f in errs) or "(none)"
    return f"{head}: {len(errs)} error(s), {len(warns)} warning(s); {body}"


# --------------------------------------------------------------------------- #
# Helpers: normalize dicts / models into models
# --------------------------------------------------------------------------- #
def _mk_transition(x: Any) -> Transition:
    if isinstance(x, Transition):
        return x.model_copy(deep=True)
    return Transition.model_validate(x)


def _mk_variable(x: Any) -> Variable:
    if isinstance(x, Variable):
        return x.model_copy(deep=True)
    return Variable.model_validate(x)


def _mk_terminal(x: Any) -> Terminal:
    if isinstance(x, Terminal):
        return x.model_copy(deep=True)
    return Terminal.model_validate(x)


def _mk_prohibition(x: Any) -> Prohibition:
    if isinstance(x, Prohibition):
        return x.model_copy(deep=True)
    return Prohibition.model_validate(x)


def _checks_back_edges(m: Machine):
    """Back edges in the DFS sense (delegates to checks, so the definition matches the structural checks exactly)."""
    return _checks_back_edges_raw(m, _checks_reachable(m))


def _reachable(m: Machine) -> set[str]:
    seen, stack = {m.initial}, [m.initial]
    while stack:
        cur = stack.pop()
        for t in m.out_edges(cur):
            if t.to not in seen:
                seen.add(t.to)
                stack.append(t.to)
    return seen


# --------------------------------------------------------------------------- #
# Gatekeeper
# --------------------------------------------------------------------------- #
class Checker:
    """Receipted edit desk for one machine. Eight interfaces, each producing a :class:`Receipt`.

    Usage: open with :meth:`open_machine` (an empty machine, or take over an existing one), edit
    one proposal at a time with the other interfaces, and finally :meth:`commit` runs acceptance
    and (if ``root`` is given) writes to disk. Every edit is **candidate copy first**:

    1. deep-copy the current machine → candidate;
    2. apply this edit to the candidate (if a precondition fails, reject immediately; the machine
       was never touched);
    3. run :func:`check_machine` on the candidate and reject on any ``error`` (the machine is still
       untouched);
    4. if everything passes, the candidate becomes current.

    ``doc`` is kept for provenance only (which skill document it was compiled from); the checks do
    not read it.
    """

    def __init__(self, skill_id: str, doc: str = "", *,
                 thresholds: Optional[Thresholds] = None,
                 require_provenance: bool = False) -> None:
        self.skill_id = skill_id
        self.doc = doc
        self.thresholds = thresholds or Thresholds()
        self._thr_given = thresholds is not None
        self._machine: Optional[Machine] = None
        self._baseline: Optional[Machine] = None      # rollback point for commit
        self._receipts: list[Receipt] = []
        #: Provenance table: prov_key → provenance row (dict). Moves with the machine: restored
        #: together on rollback/rewind.
        self._prov: dict[str, dict] = {}
        self._baseline_prov: dict[str, dict] = {}
        #: Checkpoints: label → (machine deep copy, provenance deep copy, receipt count then, commit count then).
        self._marks: dict[str, tuple[Machine, dict, int, int]] = {}
        self._commits = 0                             # successful commits so far (rewind cannot cross them)
        #: Enabled in multi-agent mode: at commit every live structure must have a provenance row,
        #: otherwise commit is refused. Off by default for the old single-agent path: its proposals
        #: carry no prov and should not become uncommittable because of that.
        self.require_provenance = bool(require_provenance)

    # ---- observation surface ---- #
    @property
    def machine(self) -> Machine:
        """A **deep copy** of the current machine.

        The internal object is deliberately not handed out: that would open an in-place
        modification channel bypassing the receipted interfaces, and then no receipt written by
        this layer could prove how the machine came to be what it is.
        """
        if self._machine is None:
            raise ValueError("open_machine has not been called yet: open a machine before using it")
        return self._machine.model_copy(deep=True)

    @property
    def opened(self) -> bool:
        return self._machine is not None

    def receipts(self) -> list[Receipt]:
        """The complete audit trail, in order. Rollback does not erase it: a rollback is itself a record."""
        return list(self._receipts)

    def provenance(self) -> dict[str, dict]:
        """Deep copy of the provenance table: ``prov_key`` → row. Rows come from the ``prov=``
        argument of each receipted interface."""
        return {k: dict(v) for k, v in self._prov.items()}

    def missing_provenance(self) -> list[str]:
        """Keys of live structures in the current machine with **no** provenance row. Must be empty
        before commit in multi-agent mode."""
        return [k for k in _live_keys(self._machine) if k not in self._prov] \
            if self._machine is not None else []

    # ---- internal: candidate copy + ruling ---- #
    def _record(self, receipt: Receipt) -> Receipt:
        self._receipts.append(receipt)
        return receipt

    def _attempt(self, op: str, mutate: Callable[[Machine], str], detail: dict,
                 *, prov: Optional[dict] = None) -> Receipt:
        """Apply one edit on a candidate copy and rule on it. ``mutate`` returns a sentence saying what changed.

        ``mutate`` may put the provenance keys touched by the edit into ``detail["prov_keys"]``;
        when the proposal carries ``prov`` and is accepted, each of those keys gets a row (the same
        ``prov``). The ``origin`` of ``prov`` must be in :data:`ORIGINS`: provenance filed in the
        wrong column is worse than no provenance.
        """
        if self._machine is None:
            return self._record(Receipt(
                op, False, "open_machine has not been called yet: open a machine first, then "
                "propose edits", dict(detail)))
        prov = _norm_prov(prov)
        if prov is not None:
            bad = _prov_problem(prov)
            if bad:
                return self._record(Receipt(op, False, f"invalid provenance row: {bad}",
                                            {**detail, "prov": dict(prov)}))
        cand = self._machine.model_copy(deep=True)
        # Do not copy detail here: each interface's mutate closure puts prov_keys into **this same**
        # dict, and a copy would not see them. The receipt gets a new {**detail, ...} dict, so the
        # original never leaks.
        try:
            note = mutate(cand)
        except _Reject as rej:
            return self._record(Receipt(op, False, rej.reason,
                                        {**detail, **rej.detail}))
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            return self._record(Receipt(
                op, False, f"the proposal itself is invalid ({type(exc).__name__}): {exc}", dict(detail)))
        findings = check_machine(cand, thresholds=self.thresholds)
        errs = [f for f in findings if f.severity == "error"]
        fdicts = [vars(f) for f in findings]
        keys = list(detail.pop("prov_keys", []) or [])
        if errs:
            return self._record(Receipt(
                op, False, _reason_from(findings, "the edited candidate fails the structural checks and was discarded"),
                {**detail, "findings": fdicts}))
        old, self._machine = self._machine, cand
        self._migrate_edge_prov(old, cand)
        if prov is not None:
            for k in keys:
                self._prov[k] = dict(prov)
        return self._record(Receipt(op, True, note,
                                    {**detail, "findings": fdicts,
                                     **({"prov_keys": keys} if prov is not None else {})}))

    @staticmethod
    def _edge_keys_by_pair(m: Optional[Machine]) -> dict:
        out: dict = {}
        if m is None:
            return out
        for sid, st in m.states.items():
            for t in st.transitions:
                out.setdefault((sid, t.to), []).append(prov_key("edge", sid, t.to, t.cond))
        return out

    def _migrate_edge_prov(self, old: Optional[Machine], new: Machine) -> None:
        """When an edge's **guard is rewritten**, move its provenance row along with it.

        When closing a loop, :func:`hexis.legacy.fit.install_counter` ``and``-s ``count < K`` onto
        **every existing** guarded outgoing edge of the target state. It is still the same edge
        (same source, same target, same support, same traces as evidence), but the provenance key
        contains the guard, so the old key goes stale and the new key has no owner. Without moving
        them, commit would reject a perfectly valid machine with ``E_PROVENANCE_MISSING`` and then
        roll the whole batch back to an empty machine (observed: on table_clean the two edges
        ``s3→s4`` and ``s3→s6`` took the whole machine down this way).

        Strict criterion: for the same ``(source, target)``, move only when the old key is **no
        longer in the graph** and the new key **has no row yet**.
        """
        before = self._edge_keys_by_pair(old)
        after = self._edge_keys_by_pair(new)
        live = {k for ks in after.values() for k in ks}
        for pair, new_keys in after.items():
            olds = [k for k in before.get(pair, ())
                    if k in self._prov and k not in live]
            for nk in new_keys:
                if nk in self._prov or not olds:
                    continue
                self._prov[nk] = self._prov.pop(olds.pop(0))

    # ---- ①  open ---- #
    def open_machine(self, *, base: Optional[Machine] = None,
                     variables: Sequence[Any] = (), terminals: Sequence[Any] = (),
                     prohibitions: Sequence[Any] = (),
                     max_steps: Optional[int] = None,
                     version: Optional[str] = None,
                     audit_tools: Sequence[str] = (),
                     prov: Optional[dict] = None) -> Receipt:
        """Open a machine: take over a deep copy of ``base`` if given, otherwise an empty all-fallback machine.

        An empty machine (``initial = FALLBACK``) is a valid foundation: it replays any trace and
        excludes no counterexample, i.e. "nothing learned yet, everything goes to interpreted
        execution".

        ``audit_tools`` are the canonical names of programmatically executed audit tools (see
        :class:`Machine.audit_tools`); if ``prov`` is given, a provenance row is recorded for each
        variable/terminal/prohibition declared in this call.
        """
        detail = {"base": bool(base), "skill_id": self.skill_id}
        prov = _norm_prov(prov)
        if prov is not None:
            bad = _prov_problem(prov)
            if bad:
                return self._record(Receipt("open_machine", False, f"invalid provenance row: {bad}",
                                            {**detail, "prov": prov if isinstance(prov, dict)
                                             else str(prov)}))
        if self._machine is not None:
            return self._record(Receipt(
                "open_machine", False,
                "this gatekeeper has already opened a machine: one Checker manages one machine, "
                "start another Checker for a different machine", detail))
        if base is not None and base.skill_id != self.skill_id:
            return self._record(Receipt(
                "open_machine", False,
                f"base has skill_id {base.skill_id!r}, which does not match this gatekeeper's "
                f"{self.skill_id!r}: machines and skills must correspond one to one", detail))
        cand = base.model_copy(deep=True) if base is not None else empty_machine(self.skill_id)
        if self._thr_given:
            cand.thresholds = self.thresholds
        else:
            self.thresholds = cand.thresholds
        try:
            for v in variables:
                _upsert_variable(cand, _mk_variable(v))
            for t in terminals:
                _upsert_terminal(cand, _mk_terminal(t))
            for p in prohibitions:
                cand.prohibitions.append(_mk_prohibition(p))
            if max_steps is not None:
                cand.max_steps = int(max_steps)
            if version is not None:
                cand.version = str(version)
            if audit_tools:
                cand.audit_tools = sorted(set(cand.audit_tools) | {str(a) for a in audit_tools})
            _ensure_fallback(cand)
        except (ValidationError, ValueError, TypeError) as exc:
            return self._record(Receipt(
                "open_machine", False,
                f"invalid open arguments ({type(exc).__name__}): {exc}", detail))
        findings = check_machine(cand, thresholds=self.thresholds)
        errs = [f for f in findings if f.severity == "error"]
        if errs:
            return self._record(Receipt(
                "open_machine", False,
                _reason_from(findings, "base itself fails the structural checks, refusing to take it over"),
                {**detail, "findings": [vars(f) for f in findings]}))
        self._machine = cand
        self._baseline = cand.model_copy(deep=True)
        if prov is not None:
            for v in variables:
                self._prov[prov_key("var", _mk_variable(v).name)] = dict(prov)
            for t in terminals:
                self._prov[prov_key("terminal", _mk_terminal(t).id)] = dict(prov)
            for p in prohibitions:
                obj = _mk_prohibition(p)
                pat = obj.pattern if isinstance(obj.pattern, dict) else {}
                row = dict(prov)
                if pat.get("clause") or pat.get("quote"):
                    # A prohibition that cites a clause and carries the original sentence comes from
                    # the **document**, not from a compiler decision. The ledger's "compiler
                    # decisions" list is split by this column; mixing them up makes it meaningless.
                    row.update({"origin": "document", "clause": str(pat.get("clause") or ""),
                                "locator": (f"{pat.get('source')}:{pat.get('line')}"
                                            if pat.get("source") and pat.get("line")
                                            else row.get("locator", "")),
                                "note": "prohibition quotes the original document sentence "
                                        "(quote), marked by hand during collection"})
                self._prov[prov_key("prohibition", obj.id)] = row
        self._baseline_prov = {k: dict(v) for k, v in self._prov.items()}
        return self._record(Receipt(
            "open_machine", True,
            f"opened machine {cand.skill_id} (initial {cand.initial}, "
            f"{len(cand.states)} states)", {**detail,
                                        "findings": [vars(f) for f in findings]}))

    # ---- ②  add state ---- #
    def add_state(self, state_id: str, action: Any, *, clause: str = "",
                  transitions: Sequence[Any] = (), initial: bool = False,
                  from_state: Optional[str] = None, from_cond: str = "",
                  from_inc: Optional[str] = None, from_support: int = 0,
                  variables: Sequence[Any] = (),
                  origin: str = "", locator: str = "",
                  prov: Optional[dict] = None) -> Receipt:
        """Add a new state and connect it into the graph **in the same proposal**.

        Adding a state on its own is necessarily invalid (it is either unreachable or has no
        outgoing edges), so incoming and outgoing edges are part of this interface rather than
        something "added later":

        * without ``transitions``, the outgoing edges default to a single default edge into
          FALLBACK: a new state naturally means "this step is learned, the next one is not yet".
        * the incoming edge is given by ``from_state`` (+ ``from_cond``). **An unguarded incoming
          edge extends the spine**: the source state's existing default edge is **rewritten** to
          point to the new state (instead of growing a second default edge), and the new state
          inherits the decision "continue onward or fall back". A guarded incoming edge only adds
          an edge and leaves existing edges alone.
        """
        detail = {"state_id": state_id, "from_state": from_state,
                  "from_cond": from_cond, "initial": initial}

        def mutate(cand: Machine) -> str:
            if state_id in cand.states:
                raise _Reject(f"state {state_id!r} already exists: pick another id, or change it with "
                              "add_transition / demote_to_fallback, do not create it twice")
            _ensure_fallback(cand)
            outs = [_mk_transition(x) for x in transitions] or \
                [Transition(to=cand.fallback)]
            st = State(id=state_id, clause=clause, action=action, transitions=outs,
                       origin=origin, locator=locator)
            cand.states[state_id] = st
            for v in variables:
                _upsert_variable(cand, _mk_variable(v))
            _autodeclare(cand, st)
            if initial:
                cand.initial = state_id
            if from_state is not None:
                _attach(cand, from_state, state_id, from_cond, from_inc, from_support,
                        origin=origin)
            elif not initial and cand.initial != state_id:
                raise _Reject(
                    f"state {state_id!r} has no incoming edge and is not the initial state: connect "
                    "it with from_state, or set initial=True to make it the initial state; otherwise it is dead code")
            keys = [prov_key("state", state_id)]
            keys += [prov_key("edge", state_id, t.to, t.cond) for t in st.transitions]
            if from_state is not None:
                keys.append(prov_key("edge", from_state, state_id, from_cond))
            keys += [prov_key("var", _mk_variable(v).name) for v in variables]
            # variables written by this state (including ones auto-declared by _autodeclare) are attributed to it
            keys += [prov_key("var", w) for w in (getattr(st.action, "writes", []) or [])]
            detail["prov_keys"] = keys
            return (f"added state {state_id} ({st.action.kind}, clause {clause or '(unassigned)'}), "
                    f"incoming edge {from_state or '(none, initial state)'}, outgoing edges "
                    f"{[t.to for t in st.transitions]}")

        return self._attempt("add_state", mutate, detail, prov=prov)

    # ---- ③  add transition ---- #
    def add_transition(self, from_state: str, to: str, *, cond: str = "",
                       inc: Optional[str] = None, support: int = 0,
                       origin: str = "", prov: Optional[dict] = None) -> Receipt:
        """Add an edge between two **existing** states. It only adds the edge; no existing edge is changed.

        An empty ``cond`` means a default edge; if the source state already has one it is rejected
        outright (a state can have only one). This is a deliberate asymmetry with
        :meth:`add_state`'s "unguarded incoming edge rewrites the spine": adding a state grows the
        spine, adding an edge fills in a branch, and the latter should not quietly change existing
        routing.
        """
        detail = {"from_state": from_state, "to": to, "cond": cond,
                  "inc": inc, "support": support,
                  "prov_keys": [prov_key("edge", from_state, to, cond)]}

        def mutate(cand: Machine) -> str:
            src = cand.states.get(from_state)
            if src is None:
                raise _Reject(f"no state {from_state!r}: create it with add_state first, "
                              "then connect edges to it")
            if to not in cand.states:
                raise _Reject(f"edge target {to!r} does not exist: create the target state with add_state first"
                              f" (existing states: {sorted(cand.states)})")
            if src.action.kind == "end":
                raise _Reject(f"state {from_state!r} is an end state (terminal="
                              f"{src.action.terminal!r}); end states cannot have outgoing edges")
            if cond:
                _check_cond(cand, from_state, cond)
                dup = [t for t in src.transitions if t.cond == cond]
                if dup:
                    raise _Reject(
                        f"state {from_state} already has an edge with the identical guard {cond!r} → "
                        f"{dup[0].to}: two edges with the same guard always overlap (violates mutual "
                        "exclusion); change the guard or the target")
            else:
                dflt = _default_edges(src)
                if dflt:
                    raise _Reject(
                        f"state {from_state} already has a default edge → {dflt[0].to}: a state can "
                        "have only one default edge; give the new edge a guard, or "
                        "demote_to_fallback first to rearrange its outgoing edges")
            if inc is not None:
                _require_counter(cand, inc)
            src.transitions.append(
                Transition(cond=cond, to=to, inc=inc, support=support, origin=origin))
            return (f"added transition {from_state}→{to}"
                    f" (guard {cond or '(default)'}, support {support}"
                    f"{', counter ' + inc if inc else ''})")

        return self._attempt("add_transition", mutate, detail, prov=prov)

    # ---- ④  close loop ---- #
    def close_loop(self, from_state: str, to: str, *, cond: str = "",
                   counter: Optional[str] = None, bound: Optional[int] = None,
                   support: int = 0, origin: str = "",
                   prov: Optional[dict] = None) -> Receipt:
        """Form a **bounded** loop: back edge + counter variable + bound exit, all three at once.

        Adding a back edge alone can never pass the checks (``checks._loop_bounds``: back edge
        without inc / counter without a bound exit), so the three must be one proposal. Installing
        the counter calls :func:`hexis.legacy.fit.install_counter` directly (the algorithm should
        exist only once): it inserts an exit ``counter >= K → FALLBACK`` on the loop's **target**
        state and ``and``-s ``counter < K`` onto its existing guarded outgoing edges, guaranteeing
        that "fall back once the count is reached" and "keep looping" are never both true in any
        configuration (the mutual exclusion of Theorem 2). Without ``bound``,
        ``thresholds.retry_budget`` is used.

        ``install_counter`` identifies the back edge by its **target**, so this interface rejects
        the case where "``from_state`` already has an edge to ``to``": which edge should get the
        counter is then ambiguous, and the gatekeeper does not guess for the caller. For a second
        back edge to the same target, use :meth:`add_transition` with the existing counter variable
        (the bound exit is already installed).
        """
        detail = {"from_state": from_state, "to": to, "cond": cond,
                  "counter": counter, "bound": bound}

        def mutate(cand: Machine) -> str:
            src = cand.states.get(from_state)
            if src is None:
                raise _Reject(f"no state {from_state!r}: the back edge's source must exist first")
            if to not in cand.states:
                raise _Reject(f"no state {to!r}: the back edge's target must exist first")
            if src.action.kind == "end":
                raise _Reject(f"state {from_state!r} is an end state; a back edge cannot start from it")
            if from_state not in _forward_closure(cand, to):
                raise _Reject(
                    f"{to} cannot reach {from_state}, so {from_state}→{to} is not a back edge (no cycle): "
                    "add non-cycle edges with add_transition, not close_loop")
            dup_target = [t for t in src.transitions if t.to == to]
            if dup_target:
                raise _Reject(
                    f"state {from_state} already has an edge to {to} (guard "
                    f"{dup_target[0].cond or '(default)'}): a target's bound is set only once. "
                    f"To draw another back edge to the same target, use add_transition with the "
                    f"existing counter variable {dup_target[0].inc or (counter or to + '_count')!r}")
            if cond:
                _check_cond(cand, from_state, cond)
                if any(t.cond == cond for t in src.transitions):
                    raise _Reject(f"state {from_state} already has an edge with guard {cond!r}: it would overlap")
            elif _default_edges(src):
                raise _Reject(
                    f"state {from_state} already has a default edge → {_default_edges(src)[0].to}: "
                    "a back edge must have a guard, or demote_to_fallback first to free the default slot")
            cname = counter or f"{to}_count"
            K = int(bound) if bound is not None else int(self.thresholds.retry_budget)
            if K < 1:
                raise _Reject(f"bound {K} < 1: the loop must allow at least one iteration, "
                              "otherwise this back edge is never taken")
            existing = cand.var(cname)
            if existing is not None and existing.type != "integer":
                raise _Reject(f"variable {cname!r} already exists with type {existing.type}, "
                              "so it cannot be used as a counter (needs integer)")
            src.transitions.append(Transition(cond=cond, to=to, support=support,
                                              origin=origin))
            exit_edge = _fit.install_counter(cand, from_state, to, k=K, name=cname)
            detail["prov_keys"] = [prov_key("edge", from_state, to, cond),
                                   prov_key("var", cname),
                                   prov_key("edge", to, exit_edge.to, exit_edge.cond)]
            return (f"closed loop {from_state}→{to} (guard {cond or '(default)'}), counter "
                    f"{cname} with bound {K}, bound exit installed on {to} → {cand.fallback}")

        return self._attempt("close_loop", mutate, detail, prov=prov)

    # ---- ④''  close a group of interdependent loops at once ---- #
    def close_loops(self, loops: Sequence[Any], *, prov: Optional[dict] = None) -> Receipt:
        """Close **a group** of back edges, with their counters and bounds, in one go, then check once.

        Why not one at a time: ``checks._back_edges`` identifies **edges pointing to an ancestor on
        the DFS stack**, so which edges are back edges depends on traversal order. In a group of
        interdependent loops, closing the first can turn what was a forward edge of the second into
        a back edge; at that moment it has no counter and the structural checks immediately report
        ``E_LOOP_UNCOUNTED``, so **every** proposal fails on its own while **together** they form a
        perfectly valid machine. Observed: s9→s5, s9→s6 and s10→s6 all failed when proposed one by
        one, with reasons pointing at s10→s2 and s11→s5, two edges nobody had touched.

        So this is one proposal with one receipt: first add every edge in the group, each with its
        counter and bound, **then** also bound the remaining back edges that DFS only recognizes
        after the edges are added (as :meth:`bound_loop` does), and finally check once. If the
        group cannot be closed, it is discarded as a whole and the machine is untouched.

        Elements of ``loops`` are dicts: ``{from_state, to, cond?, counter?, bound?, support?}``.
        """
        rows = [dict(x) for x in loops]
        detail = {"loops": rows, "n": len(rows)}

        def mutate(cand: Machine) -> str:
            if not rows:
                raise _Reject("empty loop group: nothing to close")
            keys: list[str] = []
            done: list[str] = []
            for row in rows:
                fs, to = str(row.get("from_state") or ""), str(row.get("to") or "")
                cond = str(row.get("cond") or "")
                src = cand.states.get(fs)
                if src is None:
                    raise _Reject(f"no state {fs!r}: the back edge's source must exist first")
                if to not in cand.states:
                    raise _Reject(f"no state {to!r}: the back edge's target must exist first")
                if src.action.kind == "end":
                    raise _Reject(f"state {fs!r} is an end state; a back edge cannot start from it")
                if any(t.to == to for t in src.transitions):
                    raise _Reject(f"state {fs} already has an edge to {to}: a target's bound is set only once")
                if cond:
                    _check_cond(cand, fs, cond)
                    if any(t.cond == cond for t in src.transitions):
                        raise _Reject(f"state {fs} already has an edge with guard {cond!r}: it would overlap")
                elif _default_edges(src):
                    raise _Reject(f"state {fs} already has a default edge: a back edge must have a "
                                  "guard, or free the default slot first")
                cname = str(row.get("counter") or f"{to}_count")
                K = int(row["bound"]) if row.get("bound") else int(self.thresholds.retry_budget)
                if K < 1:
                    raise _Reject(f"bound {K} < 1: the loop must allow at least one iteration")
                ex = cand.var(cname)
                if ex is not None and ex.type != "integer":
                    raise _Reject(f"variable {cname!r} already exists with type {ex.type}, "
                                  "so it cannot be used as a counter")
                src.transitions.append(Transition(cond=cond, to=to,
                                                  support=int(row.get("support") or 0),
                                                  origin=str(row.get("origin") or "")))
                exit_edge = _fit.install_counter(cand, fs, to, k=K, name=cname)
                keys += [prov_key("edge", fs, to, cond), prov_key("var", cname),
                         prov_key("edge", to, exit_edge.to, exit_edge.cond)]
                done.append(f"{fs}→{to}(K={K})")
            # back edges DFS only recognizes after the edges are added: bound them too, otherwise
            # the group as a whole is still invalid
            extra: list[str] = []
            for _ in range(len(cand.states) + 4):
                todo = [(s, t) for s, t in _checks_back_edges(cand)
                        if not t.inc and t.to != cand.fallback]
                if not todo:
                    break
                s, t = todo[0]
                cname = f"{t.to}_count"
                ex = cand.var(cname)
                if ex is not None and ex.type != "integer":
                    raise _Reject(f"variable {cname!r} has type {ex.type}, so it cannot be used as a counter")
                exit_edge = _fit.install_counter(cand, s, t.to, k=int(self.thresholds.retry_budget),
                                                 name=cname)
                keys += [prov_key("var", cname),
                         prov_key("edge", t.to, exit_edge.to, exit_edge.cond)]
                extra.append(f"{s}→{t.to}")
            detail["prov_keys"] = keys
            return (f"closed {len(done)} loops at once: {', '.join(done)}"
                    + (f"; also bounded back edges recognized only after adding the edges: "
                       f"{', '.join(extra)}" if extra else ""))

        return self._attempt("close_loops", mutate, detail, prov=prov)

    # ---- ④'  bound an existing back edge ---- #
    def bound_loop(self, from_state: str, to: str, *, counter: Optional[str] = None,
                   bound: Optional[int] = None, prov: Optional[dict] = None) -> Receipt:
        """Give an **existing** edge a counter variable and a bound exit: :meth:`close_loop` minus the edge creation.

        Why it is needed: the back edges ``checks._back_edges`` recognizes are **edges pointing to
        an ancestor on the DFS stack**, which depends on traversal order; the ones the ledger marks
        during transcription by "can the target get back here" are only a subset. Observed: after
        closing ``s9→s5``, ``s10→s2``, previously a forward edge, became a DFS back edge, so
        ``E_LOOP_UNCOUNTED`` pointed at an edge **nobody had touched since**, and ``close_loop``
        could not fix it (the edge already exists and would be rejected as "already has an edge to
        X").

        So after adding edges a final pass that "bounds the machine's actual back edges" is needed,
        and it must be receipted in the same way. The algorithm is shared with close_loop via
        :func:`hexis.legacy.fit.install_counter`: insert ``counter >= K → FALLBACK`` on the target
        and ``and`` ``counter < K`` onto the target's existing guarded outgoing edges.
        """
        detail = {"from_state": from_state, "to": to, "counter": counter, "bound": bound}

        def mutate(cand: Machine) -> str:
            src = cand.states.get(from_state)
            if src is None:
                raise _Reject(f"no state {from_state!r}")
            if to not in cand.states:
                raise _Reject(f"no state {to!r}")
            edges = [t for t in src.transitions if t.to == to]
            if not edges:
                raise _Reject(f"edge {from_state}→{to} does not exist: use close_loop to create a new back edge")
            if len(edges) > 1:
                raise _Reject(f"{from_state} has {len(edges)} edges to {to}, "
                              "so attaching a counter is ambiguous: the gatekeeper does not guess for the caller")
            if edges[0].inc:
                raise _Reject(f"{from_state}→{to} already has counter variable {edges[0].inc!r}")
            if from_state not in _forward_closure(cand, to):
                raise _Reject(f"{to} cannot reach {from_state}: this edge does not form a cycle and needs no bound")
            cname = counter or f"{to}_count"
            K = int(bound) if bound is not None else int(self.thresholds.retry_budget)
            if K < 1:
                raise _Reject(f"bound {K} < 1: the loop must allow at least one iteration")
            existing = cand.var(cname)
            if existing is not None and existing.type != "integer":
                raise _Reject(f"variable {cname!r} already exists with type {existing.type}, "
                              "so it cannot be used as a counter")
            exit_edge = _fit.install_counter(cand, from_state, to, k=K, name=cname)
            detail["prov_keys"] = [prov_key("var", cname),
                                   prov_key("edge", to, exit_edge.to, exit_edge.cond)]
            return (f"bounded existing back edge {from_state}→{to} with counter {cname}, bound {K}, "
                    f"bound exit installed on {to} → {cand.fallback}")

        return self._attempt("bound_loop", mutate, detail, prov=prov)

    # ---- ⑤  add judge action ---- #
    def add_judge(self, state_id: str, prompt: str, reads: Sequence[str],
                  writes: Sequence[str], labels: Sequence[str], *,
                  clause: str = "", abstain: str = "",
                  examples: Sequence[Any] = (), error_rate: float = 0.0,
                  support: int = 0, transitions: Sequence[Any] = (),
                  from_state: Optional[str] = None, from_cond: str = "",
                  from_support: int = 0, variables: Sequence[Any] = (),
                  introduced: bool = False, gold_from: str = "",
                  origin: str = "", locator: str = "",
                  prov: Optional[dict] = None) -> Receipt:
        """Install a judge action: semantic decisions that cannot be compiled into deterministic
        guards land here; it is the machine's only model touchpoint.

        ``introduced=True`` means the compiler **introduced it from the document** (the traces had
        no such step); ``gold_from`` names the programmatic labeler that produces its gold labels.
        An introduced judge without a labeler or with ``support==0`` may only have a single edge,
        into FALLBACK (:func:`_uncalibrated_judge_findings`).

        If ``state_id`` already exists, its action is **rewritten in place** (a branch whose guard
        cannot be learned is turned into a judge, outgoing edges unchanged); otherwise a new state
        is created and connected in the same way as :meth:`add_state`.

        A judge whose ``error_rate`` already exceeds ``judge_err_max`` is rejected on the spot:
        knowingly installing a judge noisier than allowed actively blows through the bound of the
        inequality "P(a path errs at least once) ≤ Σεᵢ".
        """
        detail = {"state_id": state_id, "reads": list(reads), "writes": list(writes),
                  "labels": list(labels), "error_rate": error_rate}

        def mutate(cand: Machine) -> str:
            if error_rate > self.thresholds.judge_err_max:
                raise _Reject(
                    f"judge action {state_id} has error rate {error_rate} > cap "
                    f"{self.thresholds.judge_err_max}: rewrite the question / add examples and recalibrate first; "
                    "if unsure, demote_to_fallback to hand this branch back to interpreted execution")
            act = JudgeAction(prompt=prompt, reads=list(reads), writes=list(writes),
                              labels=list(labels), abstain=abstain,
                              examples=list(examples), error_rate=error_rate,
                              support=support, introduced=introduced,
                              gold_from=gold_from)
            _ensure_fallback(cand)
            for v in variables:
                _upsert_variable(cand, _mk_variable(v))
            st = cand.states.get(state_id)
            if st is not None:
                if st.action.kind == "end":
                    raise _Reject(f"state {state_id!r} is an end state; it cannot be turned into a judge action")
                if transitions:
                    st.transitions = [_mk_transition(x) for x in transitions]
                st.action = act
                if clause:
                    st.clause = clause
                if origin:
                    st.origin = origin
                if locator:
                    st.locator = locator
                _autodeclare(cand, st)
                where = "rewritten in place"
            else:
                outs = [_mk_transition(x) for x in transitions] or \
                    [Transition(to=cand.fallback)]
                st = State(id=state_id, clause=clause, action=act, transitions=outs,
                           origin=origin, locator=locator)
                cand.states[state_id] = st
                _autodeclare(cand, st)
                if from_state is not None:
                    _attach(cand, from_state, state_id, from_cond, None, from_support,
                            origin=origin)
                elif cand.initial != state_id:
                    raise _Reject(f"new judge state {state_id!r} has no incoming edge: connect it with from_state")
                where = "created"
            keys = [prov_key("state", state_id), prov_key("judge", state_id)]
            keys += [prov_key("edge", state_id, t.to, t.cond) for t in st.transitions]
            if from_state is not None:
                keys.append(prov_key("edge", from_state, state_id, from_cond))
            keys += [prov_key("var", w) for w in writes]
            keys += [prov_key("var", _mk_variable(v).name) for v in variables]
            detail["prov_keys"] = keys
            return (f"judge action {state_id} ({where}): asks \"{prompt}\", reads {list(reads)} → writes "
                    f"{list(writes)}, labels {list(labels)} (abstain {abstain!r}), "
                    f"error rate {error_rate}")

        return self._attempt("add_judge", mutate, detail, prov=prov)

    # ---- ⑥  set terminal ---- #
    def set_terminal(self, state_id: str, terminal: str, *, kind: str = "",
                     output: Sequence[str] = (), clause: str = "",
                     from_state: Optional[str] = None, from_cond: str = "",
                     from_support: int = 0, origin: str = "", locator: str = "",
                     prov: Optional[dict] = None) -> Receipt:
        """Make a state an end state and register this way of ending.

        ``kind`` is this terminal's **category** (``verified`` / ``unverified``), which is what a
        prohibition's ``only_when`` reads: "run verification before submitting" should only govern
        endings that claim to be verified. An existing state is changed to an ``end`` action and
        **its outgoing edges are cleared** (end states have no outgoing edges).
        """
        detail = {"state_id": state_id, "terminal": terminal, "kind": kind,
                  "output": list(output)}

        def mutate(cand: Machine) -> str:
            _upsert_terminal(cand, Terminal(id=terminal, kind=kind,
                                            output=list(output)))
            st = cand.states.get(state_id)
            if st is not None:
                st.action = EndAction(terminal=terminal)
                st.transitions = []
                if clause:
                    st.clause = clause
                if origin:
                    st.origin = origin
                if locator:
                    st.locator = locator
                where = "rewritten in place"
            else:
                cand.states[state_id] = State(id=state_id, clause=clause,
                                              action=EndAction(terminal=terminal),
                                              origin=origin, locator=locator)
                if from_state is not None:
                    _attach(cand, from_state, state_id, from_cond, None, from_support,
                            origin=origin)
                elif cand.initial != state_id:
                    raise _Reject(f"new end state {state_id!r} has no incoming edge: connect it with from_state, "
                                  "otherwise it is dead code")
                where = "created"
            keys = [prov_key("state", state_id), prov_key("terminal", terminal)]
            if from_state is not None:
                keys.append(prov_key("edge", from_state, state_id, from_cond))
            detail["prov_keys"] = keys
            return (f"end state {state_id} ({where}) → ending {terminal}"
                    f" (kind {kind or '(unspecified)'}, output {list(output)})")

        return self._attempt("set_terminal", mutate, detail, prov=prov)

    # ---- ⑦  demote to fallback ---- #
    def demote_to_fallback(self, state_id: str, *, note: str = "",
                           prov: Optional[dict] = None) -> Receipt:
        """Replace all outgoing edges of a state with one default edge into FALLBACK: the **escape hatch**.

        Guard cannot be learned, judge too noisy, loop cannot be bounded: any "I cannot compile this
        part" can retreat here. On any well-formed machine it is always accepted: a single unguarded
        outgoing edge cannot overlap or leave a gap, FALLBACK is an end state so the machine can
        stop, and removing guards only relaxes write-before-read. So it is the retreat the agent
        can always use.

        The whole segment it cuts off (states reachable only through this state) is **deleted** as
        well: falling back to interpreted execution means giving up that part of the compiled
        artifact, and keeping those states would only leave unreachable dead code.
        """
        detail = {"state_id": state_id, "note": note}

        def mutate(cand: Machine) -> str:
            st = cand.states.get(state_id)
            if st is None:
                raise _Reject(f"no state {state_id!r}: nothing to demote"
                              f" (existing states: {sorted(cand.states)})")
            if st.action.kind == "end":
                raise _Reject(f"state {state_id!r} is an end state and has no outgoing edges to demote; "
                              "to drop it, change its incoming edges")
            _ensure_fallback(cand)
            if state_id == cand.fallback:
                raise _Reject("FALLBACK is the fallback itself; there is nothing to demote")
            st.transitions = [Transition(to=cand.fallback, origin="compiler")]
            keep = _reachable(cand) | {cand.fallback, cand.initial}
            dropped = sorted(set(cand.states) - keep)
            for sid in dropped:
                del cand.states[sid]
            detail["prov_keys"] = [prov_key("edge", state_id, cand.fallback, "")]
            return (f"state {state_id} handed back to interpreted execution (only outgoing edge → {cand.fallback})"
                    f"{'; ' + note if note else ''}"
                    f"; also deleted states that became unreachable: {dropped or '(none)'}")

        return self._attempt("demote_to_fallback", mutate, detail,
                             prov=prov if prov is not None else
                             {"origin": "compiler", "agent_id": "checker",
                              "touchpoint_id": "", "note": note or "demote_to_fallback"})

    # ---- ⑧  commit ---- #
    def commit(self, *, t_plus: Sequence[Any] = (), t_minus: Sequence[Any] = (),
               holdout: Sequence[Any] = (), root: Any = None) -> Receipt:
        """Run acceptance and persist only if it passes. **This step is all-or-nothing.**

        The opposite of the first seven interfaces: a single rejected edit does not affect the
        others, but if commit fails, **all** edits since the last commit (or open_machine) are
        rolled back together. A batch of edits either becomes the new foundation as a whole or does
        not count at all; nobody else should read an intermediate state. The receipts stay:
        rollback is part of the audit trail.

        If ``root`` is given and acceptance passes, machine.json is written there. **This is the
        only place in the repository that writes machine files**: a machine file can only be the
        product of a commit that passed acceptance.
        """
        detail: dict = {"t_plus": len(t_plus), "t_minus": len(t_minus),
                        "holdout": len(holdout), "root": str(root) if root else ""}
        if self._machine is None:
            return self._record(Receipt("commit", False, "open_machine has not been called yet: no machine to commit",
                                        detail))
        from hexis.legacy import verify as _verify  # lazy import: verify in turn uses this module's Finding

        rep = _verify.verify_machine(self._machine, t_plus, t_minus,
                                     thresholds=self.thresholds, holdout=holdout)
        detail = {**detail, "report": _verify.report_dict(rep),
                  "findings": [vars(f) for f in rep.findings]}
        missing = self.missing_provenance() if self.require_provenance else []
        if not rep.ok or missing:
            self._machine = self._baseline.model_copy(deep=True)
            self._prov = {k: dict(v) for k, v in self._baseline_prov.items()}
            why = ("acceptance failed" if not rep.ok else
                   f"[E_PROVENANCE_MISSING] {len(missing)} live structures have no provenance row "
                   f"{missing[:8]}{'…' if len(missing) > 8 else ''}")
            return self._record(Receipt(
                "commit", False,
                f"{why}; the whole batch was rolled back to the machine as of the last commit: " + _verify.summary(rep),
                {**detail, "rolled_back": True, "missing_provenance": missing}))
        self._baseline = self._machine.model_copy(deep=True)
        self._baseline_prov = {k: dict(v) for k, v in self._prov.items()}
        self._commits += 1
        path = ""
        if root is not None:
            path = str(save_machine(self._machine, root))
            if self._prov:
                Path(root).mkdir(parents=True, exist_ok=True)
                (Path(root) / PROVENANCE_FILE).write_text(
                    json.dumps(self._prov, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
        return self._record(Receipt(
            "commit", True,
            "acceptance passed: " + _verify.summary(rep) + (f"; written to {path}" if path else ""),
            {**detail, "path": path}))

    # ---- ⑨  split by predecessor ---- #
    def split_state(self, state_id: str, groups: dict, *, outs: Optional[dict] = None,
                    prov: Optional[dict] = None) -> Receipt:
        """Clone a state into copies grouped **by predecessor** (Myhill-Nerode: these histories
        are not actually equivalent).

        ``groups = {clone_id: [predecessor state id, ...]}``: every predecessor of ``state_id`` must
        fall into exactly one group; clone ids must be new (assigned by the harness, not named by
        the agent); each clone inherits the original state's action and clause, and takes its
        outgoing edges from ``outs[clone_id]`` (a subset of ``[{"if","to"}]``, looked up by
        (cond, to) among the original state's outgoing edges), inheriting all of them if not given.
        The original state is deleted.

        Rejected: splitting the initial state (it has no predecessors to split by); an original
        state with a self-loop or that is the target of a counted back edge (the bound is installed
        on the target, and splitting would misplace the bound exit; break the loop, split, then
        close it again). Replay is **insensitive** to splitting: the clone's action is a deep copy,
        and ``replay._action_matches`` compares only actions, not identity (the split tests check
        this).
        """
        detail = {"state_id": state_id, "groups": {k: list(v) for k, v in groups.items()},
                  "outs": {k: list(v) for k, v in (outs or {}).items()}}

        def mutate(cand: Machine) -> str:
            st = cand.states.get(state_id)
            if st is None:
                raise _Reject(f"no state {state_id!r}: nothing to split")
            if st.action.kind == "end":
                raise _Reject(f"state {state_id!r} is an end state: end states have no outgoing edges to split")
            if state_id == cand.initial:
                raise _Reject(f"state {state_id!r} is the initial state: it has no predecessors to split by")
            if state_id == cand.fallback:
                raise _Reject("FALLBACK cannot be split")
            preds = [(src, t) for src, t in cand.transitions_all() if t.to == state_id]
            if any(src == state_id for src, _t in preds):
                raise _Reject(f"state {state_id!r} has a self-loop: demote to break the loop first, "
                              "then split, then close_loop")
            if any(t.inc for _s, t in preds):
                raise _Reject(f"state {state_id!r} is the target of a counted back edge: splitting "
                              "would misplace the bound exit; break the loop before splitting")
            pred_ids = sorted({src for src, _t in preds})
            if len(groups) < 2:
                raise _Reject("must split into at least two groups, otherwise it is not a split")
            assigned: dict[str, str] = {}
            for cid, members in groups.items():
                if cid in cand.states:
                    raise _Reject(f"clone id {cid!r} already exists: clone ids must be newly assigned")
                for p in members:
                    if p not in pred_ids:
                        raise _Reject(f"{p!r} is not a predecessor of {state_id} (predecessors: {pred_ids})")
                    if p in assigned:
                        raise _Reject(f"predecessor {p!r} is in both group {assigned[p]} and group {cid}")
                    assigned[p] = cid
            left = [p for p in pred_ids if p not in assigned]
            if left:
                raise _Reject(f"predecessors {left} are not assigned to any group: every "
                              "predecessor needs a destination")
            keys: list[str] = []
            for cid, members in groups.items():
                if outs and cid in outs:
                    picked = []
                    for spec in outs[cid]:
                        t = _mk_transition(spec)
                        match = [o for o in st.transitions if o.cond == t.cond and o.to == t.to]
                        if not match:
                            raise _Reject(f"clone {cid} requests outgoing edge {t.cond!r}→{t.to}, which is not among "
                                          f"the outgoing edges of {state_id}")
                        picked.append(match[0].model_copy(deep=True))
                    if not any(not o.cond for o in picked):
                        picked.append(Transition(to=cand.fallback, origin="compiler"))
                else:
                    picked = [o.model_copy(deep=True) for o in st.transitions]
                clone = State(id=cid, clause=st.clause, action=st.action.model_copy(deep=True),
                              transitions=picked, origin=st.origin or "trace",
                              locator=st.locator)
                cand.states[cid] = clone
                keys.append(prov_key("state", cid))
                keys += [prov_key("edge", cid, o.to, o.cond) for o in picked]
                for src, t in preds:
                    if src in members:
                        t.to = cid
                        keys.append(prov_key("edge", src, cid, t.cond))
            del cand.states[state_id]
            detail["prov_keys"] = keys
            return (f"state {state_id} split by predecessor into {sorted(groups)}: "
                    + "; ".join(f"{cid} ← {sorted(m)}" for cid, m in groups.items()))

        return self._attempt("split_state", mutate, detail, prov=prov)

    # ---- ⑩ ⑪  checkpoints and rewind ---- #
    def mark(self, label: str) -> Receipt:
        """Set a checkpoint on the current machine. Receipted, but does not change the machine."""
        detail = {"label": label}
        if self._machine is None:
            return self._record(Receipt("mark", False, "open_machine has not been called yet", detail))
        self._marks[label] = (self._machine.model_copy(deep=True),
                              {k: dict(v) for k, v in self._prov.items()},
                              len(self._receipts), self._commits)
        return self._record(Receipt(
            "mark", True, f"checkpoint {label!r}: {len(self._machine.states)} states, "
            f"{len(self._prov)} provenance rows, receipt #{len(self._receipts)}", detail))

    def rewind(self, label: str) -> Receipt:
        """Return to a checkpoint: machine and provenance table are restored together, **receipts are
        not erased** (the rewind itself is a record).

        Cannot cross the last successful commit: the commit is already on disk, and rewinding past
        it would make the machine on disk and the machine held by the gatekeeper diverge.
        """
        detail = {"label": label}
        if self._machine is None:
            return self._record(Receipt("rewind", False, "open_machine has not been called yet", detail))
        got = self._marks.get(label)
        if got is None:
            return self._record(Receipt(
                "rewind", False, f"no checkpoint {label!r} (existing: {sorted(self._marks)})", detail))
        m, prov, n_receipts, n_commits = got
        if n_commits < self._commits:
            return self._record(Receipt(
                "rewind", False,
                f"[E_REWIND_PAST_COMMIT] checkpoint {label!r} predates the last commit: the commit is already on disk, "
                "cannot rewind to before it", detail))
        self._machine = m.model_copy(deep=True)
        self._prov = {k: dict(v) for k, v in prov.items()}
        return self._record(Receipt(
            "rewind", True,
            f"returned to checkpoint {label!r} (machine as of receipt #{n_receipts}, {len(m.states)} states)",
            {**detail, "receipts_then": n_receipts, "receipts_now": len(self._receipts)}))

    # ---- proposal dispatch ---- #
    def apply(self, proposal: Proposal) -> Receipt:
        """Dispatch to the receipted interface named by ``proposal.op``. Unknown ops / wrong
        arguments produce a rejection receipt."""
        op = proposal.op
        if op not in OPS:
            return self._record(Receipt(
                op, False, f"unknown op {op!r}: only {list(OPS)} are accepted", dict(proposal.args)))
        meth = getattr(self, op)
        try:
            return meth(**dict(proposal.args))
        except TypeError as exc:                       # wrong argument names/count
            return self._record(Receipt(
                op, False, f"wrong arguments: {exc}", dict(proposal.args)))


# --------------------------------------------------------------------------- #
# Batch: a sequence of proposals, ruled on one by one
# --------------------------------------------------------------------------- #
def batch_check(base: Machine, proposals: Sequence[Proposal]) -> tuple[Machine, list[Receipt]]:
    """Rule on proposals against ``base`` one by one; returns ``(edited machine, one receipt per
    proposal)``.

    **A rejected proposal does not affect proposals accepted before it**: this is why the module
    exists, and its single but crucial difference from
    :func:`hexis.legacy.compiler.compile_round` (which undoes the whole round). ``base`` itself is
    never modified in place.

    Receipts correspond one to one with proposals; the only exception is when ``base`` cannot even
    be taken over, in which case ``(copy of base, [that open_machine rejection receipt])`` is
    returned.
    """
    ck = Checker(base.skill_id, thresholds=base.thresholds)
    opened = ck.open_machine(base=base)
    if not opened.accepted:
        return base.model_copy(deep=True), [opened]
    out = [ck.apply(p) for p in proposals]
    return ck.machine, out


# --------------------------------------------------------------------------- #
# Atomic edit operations (all performed on the candidate copy)
# --------------------------------------------------------------------------- #
def _ensure_fallback(m: Machine) -> None:
    """Ensure the FALLBACK state exists. It is a reserved state: entering it = abandon the
    compiled path and switch back to interpreted execution."""
    if m.fallback in m.states:
        return
    term = m.terminals[0].id if m.terminals else "done"
    if not m.terminals:
        m.terminals.append(Terminal(id=term))
    m.states[m.fallback] = State(id=m.fallback, action=EndAction(terminal=term))


def _attach(m: Machine, src_id: str, dst_id: str, cond: str,
            inc: Optional[str], support: int, *, origin: str = "") -> None:
    """Connect ``dst`` after ``src``. An unguarded incoming edge **rewrites** the source state's
    existing default edge (extends the spine).

    Only a default edge **pointing to FALLBACK** is rewritten (the empty slot meaning "next step not
    learned yet"). If the default edge already points to a real state, it is rejected: that is a
    spine someone else connected, and unconditionally hanging another state on it silently steals
    the spine (``E_SPINE_TAKEN``). With a single agent that is a bug that quietly changes routing;
    with multiple agents it is two agents' proposals overwriting each other. To attach a second
    successor, give it a guard.
    """
    src = m.states.get(src_id)
    if src is None:
        raise _Reject(f"no state {src_id!r}: the incoming edge's source must exist first"
                      f" (existing states: {sorted(m.states)})")
    if src.action.kind == "end":
        raise _Reject(f"state {src_id!r} is an end state; no outgoing edge can be connected from it")
    if inc is not None:
        _require_counter(m, inc)
    if cond:
        _check_cond(m, src_id, cond)
        if any(t.cond == cond for t in src.transitions):
            raise _Reject(f"state {src_id} already has an edge with guard {cond!r}: two edges with "
                          "the same guard always overlap")
        src.transitions.append(Transition(cond=cond, to=dst_id, inc=inc, support=support,
                                          origin=origin))
        return
    dflt = _default_edges(src)
    if dflt:                                   # extend the spine: rewrite the original default target
        if dflt[0].to != m.fallback and dflt[0].to != dst_id:
            raise _Reject(
                f"[E_SPINE_TAKEN] the default edge of state {src_id} already points to "
                f"{dflt[0].to}, not an empty slot: unconditionally hanging {dst_id} on it would "
                "steal the already connected spine; give this incoming edge a guard, or "
                "demote_to_fallback first to free the default slot",
                {"code": "E_SPINE_TAKEN", "state_id": src_id, "taken_by": dflt[0].to})
        dflt[0].to = dst_id
        dflt[0].inc = inc
        dflt[0].support = support
        if origin:
            dflt[0].origin = origin
    else:
        src.transitions.append(Transition(to=dst_id, inc=inc, support=support,
                                          origin=origin))


def _upsert_variable(m: Machine, v: Variable) -> None:
    old = m.var(v.name)
    if old is None:
        m.variables.append(v)
        return
    if old.type != v.type:
        raise _Reject(f"variable {v.name!r} is already declared as {old.type}, which conflicts "
                      f"with the new declaration {v.type}: variables with the same name must "
                      "have the same type")
    old.init, old.init_from = v.init, v.init_from


def _upsert_terminal(m: Machine, t: Terminal) -> None:
    old = next((x for x in m.terminals if x.id == t.id), None)
    if old is None:
        m.terminals.append(t)
        return
    if t.kind:
        old.kind = t.kind
    if t.output:
        old.output = list(t.output)


def _autodeclare(m: Machine, st: State) -> None:
    """Variables a state writes but that were never declared are added to the variable table
    automatically (no init: this state produces them).

    Without a better guess the type falls back to ``string``; for precise types such as
    ``array``/``integer`` (or for ``init_from``), declare them explicitly in ``variables=``, and
    :func:`_upsert_variable` gives the explicit declaration precedence.
    """
    for w in getattr(st.action, "writes", []) or []:
        if m.var(w) is None:
            m.variables.append(Variable(name=w, type="string"))


def _require_counter(m: Machine, name: str) -> None:
    v = m.var(name)
    if v is None:
        raise _Reject(f"counter variable {name!r} is not declared: close loops with close_loop "
                      "(it installs the counter variable and the bound exit together), do not "
                      "add inc by hand")
    if v.type != "integer":
        raise _Reject(f"counter variable {name!r} has type {v.type}, not integer")


def _check_cond(m: Machine, sid: str, expr: str) -> None:
    """A guard must parse (whitelisted AST) and use only declared variables."""
    try:
        used = _cond.vars_of(expr)
    except _cond.CondError as exc:
        raise _Reject(f"guard {expr!r} on state {sid} is invalid: {exc}") from exc
    unknown = sorted(v for v in used if m.var(v) is None)
    if unknown:
        raise _Reject(
            f"guard {expr!r} on state {sid} uses undeclared variables {unknown}: "
            "have some state write it first (or declare it in the variables of open_machine/add_state)")


# --------------------------------------------------------------------------- #
# Provenance: keys of live structures, validity of provenance rows
# --------------------------------------------------------------------------- #
def _live_keys(m: Machine) -> list[str]:
    """Keys of **every structure in a machine that should have a provenance row**: states, edges,
    judges, variables, terminals, prohibitions.

    The FALLBACK state and its terminal are reserved structures and need no provenance; default
    edges into FALLBACK do (they record "I did not learn this part", and who decided to give up,
    and when, must be traceable).
    """
    keys: list[str] = []
    for sid in sorted(m.states):
        st = m.states[sid]
        if sid == m.fallback:
            continue
        keys.append(prov_key("state", sid))
        if st.action.kind == "judge":
            keys.append(prov_key("judge", sid))
        for t in st.transitions:
            keys.append(prov_key("edge", sid, t.to, t.cond))
    keys += [prov_key("var", v.name) for v in m.variables]
    fb_term = m.states[m.fallback].action.terminal if m.fallback in m.states \
        and m.states[m.fallback].action.kind == "end" else None
    keys += [prov_key("terminal", t.id) for t in m.terminals if t.id != fb_term]
    keys += [prov_key("prohibition", p.id) for p in m.prohibitions]
    return keys


def _norm_prov(prov: Any) -> Any:
    """Accepts a dict or an object with ``to_dict()`` (a provenance record); anything else is passed
    through unchanged for _prov_problem to reject."""
    if prov is not None and not isinstance(prov, dict) and hasattr(prov, "to_dict"):
        return prov.to_dict()
    return prov


def _prov_problem(prov: Any) -> str:
    """Why a provenance row is invalid; empty string if valid. Checks only shape and column, not
    whether the content is true."""
    if not isinstance(prov, dict):
        return f"expected a dict, got {type(prov).__name__}"
    origin = prov.get("origin")
    if origin not in ORIGINS:
        return f"origin={origin!r} is not in {list(ORIGINS)}"
    if not str(prov.get("agent_id") or ""):
        return "missing agent_id (who proposed this)"
    return ""


# --------------------------------------------------------------------------- #
def _forward_closure(m: Machine, start: str) -> set[str]:
    """All states reachable from ``start`` along edges (including itself). Used to decide
    whether an edge is a back edge."""
    seen, stack = {start}, [start]
    while stack:
        cur = stack.pop()
        for t in m.out_edges(cur):
            if t.to not in seen:
                seen.add(t.to)
                stack.append(t.to)
    return seen


__all__ = [
    "Checker", "Finding", "OPS", "ORIGINS", "PROVENANCE_FILE", "Proposal", "Receipt",
    "audit_declaration_findings", "batch_check", "check_machine", "classify",
    "hot_judges", "prov_key", "weak_edges",
]
