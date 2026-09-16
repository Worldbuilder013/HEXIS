"""Turn-by-turn sequential compilation: walk real execution traces action by action and induce a state machine for the skill.

The compiler is a "compile agent + deterministic gatekeeper" pair. This file is the **gatekeeper**
half: alignment, building states, wiring edges, forming loops, learning branch guards, calibrating
error rates and structural checks, all deterministic computations that can be recomputed and
undone. Semantic decisions (which clause a step belongs to, which variables it reads, what question
to ask at a branch) come from the compile agent in the real system; for the toy skill,
deterministic heuristics stand in for it here, so the whole compilation can self-test offline in
seconds. **The model appears in only two places**: :func:`make_judge` (drafting a judge action when
a branch has no learnable deterministic guard) and :func:`calibrate` (calibrating that judge
action's error rate on snapshots). Nothing else touches the model.

**Only the pipeline skeleton remains in this file**: learning branch guards, installing back edge
counters with bound exits, and calibrating error rates are three pipeline-independent algorithms
that any machine can use, and they live in :mod:`hexis.legacy.fit`; the action signature that
decides "same step" lives in :mod:`hexis.traces.normalize`. This module imports from both and
re-exports them unchanged (callers of ``learn_cond`` / ``calibrate`` need no changes).

The algorithm walks the traces (shortest first) and aligns each action with the current state: if
they match, it advances and records one unit of support for that edge; if the current state has no
action yet, it installs one; if the next action is the same as an existing state's, it wires back
to that state to form a loop (merging ≈-equivalent histories, which corresponds to Myhill-Nerode
state minimality); when a state grows a second edge leading to a different successor, that is a
branch, and a separating guard is learned from the variable snapshots on each side, with a judge
action created only when no guard can be learned. Finally all structural checks run at once, and
any violation undoes the most recent construction.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from hexis.legacy import fit
from hexis.legacy.fit import learn_cond  # noqa: F401  (re-exported: callers have always imported it from here)
from hexis.legacy.replay import excludes, reproduces
from hexis.machine.checks import structural_findings
from hexis.machine.schema import (
    ABSTAIN,
    EndAction,
    Example,
    JudgeAction,
    Machine,
    State,
    Terminal,
    ToolAction,
    Transition,
    Variable,
)
from hexis.traces.normalize import canon_action

_ABSTAIN = ABSTAIN


# --------------------------------------------------------------------------- #
# Splitting the document into clauses (numbering the clauses)
# --------------------------------------------------------------------------- #
_CLAUSE_RE = re.compile(r"^#+\s*(S\d+(?:\.\d+)?|P\d+)\b")


def partition(doc: str) -> list[tuple[str, str]]:
    """Split the SKILL.md body into numbered clause blocks at ``## Sx`` / ``### Sx.y`` / ``## Px`` headings."""
    clauses: list[tuple[str, str]] = []
    cur_id: Optional[str] = None
    buf: list[str] = []
    for line in doc.splitlines():
        m = _CLAUSE_RE.match(line)
        if m:
            if cur_id:
                clauses.append((cur_id, "\n".join(buf)))
            cur_id, buf = m.group(1), [line]
        elif cur_id:
            buf.append(line)
    if cur_id:
        clauses.append((cur_id, "\n".join(buf)))
    return clauses


def _infer_clause(action: dict, clauses: list[tuple[str, str]]) -> str:
    """Assign an action to the best-matching clause. Tools are looked up by name in the text; judges prefer the most specific clause that states a criterion."""
    kind = action.get("kind")
    if kind == "tool":
        name = action.get("name", "")
        hits = [cid for cid, text in clauses if name and name in text]
        return hits[0] if hits else ""
    if kind == "judge":
        specific = [cid for cid, text in clauses if "判据" in text or "criterion" in text.lower()]
        if specific:
            return max(specific, key=len)       # S2.1 is more specific than S2
        hits = [cid for cid, text in clauses if "规范" in text or "well-formed" in text.lower()]
        return max(hits, key=len) if hits else ""
    return ""


# --------------------------------------------------------------------------- #
# Action signature: deciding "same step" (in the real system the compile agent decides new vs
# repeated; the toy uses the signature as a stand-in)
# --------------------------------------------------------------------------- #
def _sig(action: dict) -> tuple:
    """The compile-tier action KEY, delegated to :func:`hexis.traces.normalize.canon_action` (``strict=True``).

    It is given a **bare action dict** (at compile time only ``rec.action`` is at hand), so the
    writes of judge/model steps cannot be inferred and are always empty; the strict tier therefore
    falls back exactly to this function's original grouping of "tools compare by name, judges by
    question, ends by terminal". tests/test_14_normalize.py pins this down pair by pair on the
    table_clean record table.
    """
    return canon_action(action, strict=True)


# --------------------------------------------------------------------------- #
# Inferring an action's reads/writes from trace records (deterministic heuristics standing in for
# the compile agent)
# --------------------------------------------------------------------------- #
def _infer_writes(rec_action: dict, output: dict) -> list[str]:
    if rec_action.get("kind") == "judge":
        return list(output.keys())              # a judge's output is exactly {written variable: label}
    return [k for k in output.keys() if k != "ok"]


def _infer_reads(rec_action: dict, prev_vars: dict) -> list[str]:
    if "reads" in rec_action:                   # judge actions already record reads at runtime
        return list(rec_action["reads"])
    reads: list[str] = []
    for _pk, pv in (rec_action.get("input") or {}).items():
        for var, val in prev_vars.items():
            if val == pv and var not in reads:
                reads.append(var)
    return reads


def _templatize(inp: dict, prev_vars: dict) -> dict:
    """Rewrite concrete input values back into ``${var}`` templates (when a value equals a variable's current value)."""
    out: dict = {}
    for k, v in (inp or {}).items():
        hit = next((var for var, val in prev_vars.items() if val == v), None)
        out[k] = f"${{{hit}}}" if hit else v
    return out


# --------------------------------------------------------------------------- #
# Compile state: the machine + incremental ledgers
# --------------------------------------------------------------------------- #
@dataclass
class _Build:
    machine: Machine
    sig2sid: dict = field(default_factory=dict)      # action signature -> state id
    #: per state, the successor taken after it executed + the variable snapshot at that time (for learning branch guards / calibration)
    branch_obs: dict = field(default_factory=lambda: defaultdict(list))
    #: set of labels observed at each judge state (determines labels)
    judge_labels: dict = field(default_factory=lambda: defaultdict(set))
    #: max number of times each state was entered within a **single trace** (determines the loop bound)
    max_visits: dict = field(default_factory=lambda: defaultdict(int))
    #: field names of the task input (these variables are init_from task.input)
    input_keys: set = field(default_factory=set)
    counter: int = 0

    def new_sid(self) -> str:
        self.counter += 1
        return f"s{self.counter}"


def _seed(skill_id: str) -> Machine:
    """Initial machine: a placeholder start + FALLBACK + a done terminal. The start state's action is installed along the first trace."""
    return Machine(
        skill_id=skill_id,
        initial="s0",
        states={
            "s0": State(id="s0", action=EndAction(terminal="__placeholder__")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
            "end": State(id="end", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done", output=[])],
    )


def _is_placeholder(state: State) -> bool:
    return state.action.kind == "end" and getattr(state.action, "terminal", "") == "__placeholder__"


def _install(build: _Build, sid: str, rec, prev_vars: dict, clauses) -> None:
    """Install a record's action into a state."""
    ra = rec.action
    kind = ra.get("kind")
    clause = _infer_clause(ra, clauses)
    writes = _infer_writes(ra, rec.output)
    reads = _infer_reads(ra, prev_vars)
    st = build.machine.states[sid]
    st.clause = clause
    if kind == "tool":
        st.action = ToolAction(name=ra["name"],
                               input=_templatize(ra.get("input", {}), prev_vars),
                               reads=reads, writes=writes)
    elif kind == "judge":
        wr = writes or ["header_ok"]
        for w in writes:
            build.judge_labels[sid].add(rec.output.get(w))
        first = rec.output.get(wr[0])
        lbls = ([first, _ABSTAIN] if first and first != _ABSTAIN else [_ABSTAIN])
        st.action = JudgeAction(prompt=ra.get("prompt", ""),
                                reads=reads or ["header_row"],
                                writes=wr,
                                labels=lbls)        # _finalize later fills in all observed labels
    st.transitions = []
    build.sig2sid[_sig(ra)] = sid


# --------------------------------------------------------------------------- #
# Walking one trace: build states and edges, accumulate branch observations
# --------------------------------------------------------------------------- #
def _walk(build: _Build, records, clauses, initial_vars=None) -> None:
    m = build.machine
    p = m.initial
    prev_vars: dict = dict(initial_vars or {})       # the start action must be able to rewrite task.input back into ${var}
    visits: dict = defaultdict(int)
    try:
        for i, rec in enumerate(records):
            visits[p] += 1
            if rec.action.get("kind") == "end":
                _add_edge(m, p, "end")
                return
            # install the action into p (if p is a placeholder or the signature matches)
            st = m.states[p]
            if _is_placeholder(st):
                _install(build, p, rec, prev_vars, clauses)
            elif _sig(rec.action) != _sig({"kind": st.action.kind,
                                            "name": getattr(st.action, "name", None),
                                            "prompt": getattr(st.action, "prompt", None)}):
                # p's action does not match this record: add an edge to FALLBACK (compilation is still shallow, hand off to interpretation)
                _add_edge(m, p, m.fallback)
                return
            # accumulate observed judge labels
            if m.states[p].action.kind == "judge":
                for w in m.states[p].action.writes:
                    if w in rec.output:
                        build.judge_labels[p].add(rec.output[w])
            # decide the next state
            nxt = records[i + 1] if i + 1 < len(records) else None
            if nxt is None:
                _add_edge(m, p, "end")
                return
            if nxt.action.get("kind") == "end":
                _add_edge(m, p, "end")
                build.branch_obs[p].append(("end", dict(rec.vars)))
                return
            nsig = _sig(nxt.action)
            if nsig in build.sig2sid:
                tgt = build.sig2sid[nsig]           # repeated: wire back to the existing state (may form a loop)
            else:
                tgt = build.new_sid()
                m.states[tgt] = State(id=tgt, action=EndAction(terminal="__placeholder__"))
            _add_edge(m, p, tgt)
            build.branch_obs[p].append((tgt, dict(rec.vars)))
            prev_vars = dict(rec.vars)
            p = tgt
    finally:
        for sid, c in visits.items():
            build.max_visits[sid] = max(build.max_visits[sid], c)


def _add_edge(m: Machine, src: str, dst: str) -> None:
    """Add an edge (with no guard yet), or add +1 support to an existing edge with the same target."""
    for t in m.states[src].transitions:
        if t.to == dst:
            t.support += 1
            return
    m.states[src].transitions.append(Transition(to=dst, support=1))


# --------------------------------------------------------------------------- #
# Drafting a judge action (make_judge)
#
# Learning branch guards (learn_cond / candidate_atoms / separating) has moved to
# :mod:`hexis.legacy.fit`; this module re-exports them under their original names at the top, so
# callers still use ``compiler.learn_cond``.
# --------------------------------------------------------------------------- #
# WARNING: **deprecated (dead code)**. This function never uses its ``model`` parameter; drafting is
# fully deterministic. Its only call site (``_solve_branches``) passes labels taken from the
# ``__lbl__`` key of the snapshots, and **nothing anywhere in the repository ever writes** that key,
# so in a real compilation labels is always ``[""]``. It is left exactly as is (changing it would
# mean changing a path that has never worked); do not build anything new on it: to create judge
# actions, write a new path with a real source of labels.
def make_judge(prompt: str, reads: list[str], snaps_by_target: dict,
               labels: list[str], model) -> tuple[JudgeAction, dict]:
    """Draft a judge action when a branch has no learnable deterministic guard. **Deprecated, see the comment above.**

    Assigns one label to each target, with examples taken from each target's variable snapshots.
    Returns (judge, {target: guard string}); guards have the form ``verdict == '<label>'`` and read
    the verdict variable written by the judge.
    """
    from hexis.machine.schema import Example
    targets = list(snaps_by_target)
    verdict_var = "verdict"
    tgt_label = {tgt: (labels[i] if i < len(labels) else f"L{i}")
                 for i, tgt in enumerate(targets)}
    examples = []
    for tgt, snaps in snaps_by_target.items():
        for s in snaps[:2]:
            ex = {k: s.get(k) for k in reads}
            ex["label"] = tgt_label[tgt]
            examples.append(Example(**ex))
    lbls = list(tgt_label.values()) + [_ABSTAIN]
    judge = JudgeAction(prompt=prompt or "which branch to take", reads=reads,
                        writes=[verdict_var], labels=lbls, examples=examples)
    conds = {tgt: f"{verdict_var} == {lab!r}" for tgt, lab in tgt_label.items()}
    return judge, conds


def calibrate(judge: JudgeAction, labeled_snaps: list[tuple], model) -> tuple[float, int]:
    """Run the judge on snapshots with correct labels and measure its error rate and support. **The second model touchpoint at compile time.**

    ``labeled_snaps`` = ``[(variable snapshot, correct label), ...]``. Error rate = the fraction of
    wrong answers among non-abstentions; abstentions are counted separately and are not errors.
    Support = the number of snapshots. The algorithm itself is :func:`hexis.legacy.fit.calibrate`.
    """
    rate = fit.calibrate(judge, labeled_snaps, model=model)
    return rate, len(labeled_snaps)


# --------------------------------------------------------------------------- #
# Branch solving: assign guards to a state with multiple outgoing edges
# --------------------------------------------------------------------------- #
def _solve_branches(build: _Build, sid: str, thresholds, model) -> None:
    m = build.machine
    st = m.states[sid]
    obs = build.branch_obs.get(sid, [])
    targets = []
    for t in st.transitions:
        if t.to not in targets:
            targets.append(t.to)
    if len(targets) < 2:
        return                                     # single outgoing edge: no guard needed
    snaps_by_target: dict = defaultdict(list)
    for tgt, snap in obs:
        snaps_by_target[tgt].append(snap)
    # a target with insufficient support -> the whole branch goes to FALLBACK
    if any(len(snaps_by_target.get(t, [])) < thresholds.min_support for t in targets):
        st.transitions = [Transition(to=m.fallback)]
        return
    learned = fit.learn_cond(snaps_by_target, m.variables,
                             min_support=thresholds.min_support,
                             holdout_ratio=thresholds.holdout_ratio,
                             acc_thr=thresholds.acc_thr)
    if learned is None and model is not None:
        # no deterministic guard can be learned -> create a judge action (model touchpoint) and turn the state into that judge's verdict
        judge, conds = make_judge(getattr(st.action, "prompt", ""),
                                  getattr(st.action, "reads", []) or ["header_row"],
                                  snaps_by_target,
                                  sorted({s.get("__lbl__", "") for s in obs}), model)
        st.action = judge
        learned = conds
    if learned is None:
        st.transitions = [Transition(to=m.fallback)]
        return
    # apply the guards: write them back per target; the target with the most support becomes the default edge (empty guard)
    fallback_tgt = max(targets, key=lambda t: len(snaps_by_target.get(t, [])))
    new_edges = []
    for tgt in targets:
        if tgt == fallback_tgt:
            continue
        new_edges.append(Transition(cond=learned[tgt], to=tgt,
                                     support=len(snaps_by_target.get(tgt, []))))
    new_edges.append(Transition(to=fallback_tgt,
                                support=len(snaps_by_target.get(fallback_tgt, []))))
    st.transitions = new_edges


# --------------------------------------------------------------------------- #
# Back edge counter variables + bound exits
# --------------------------------------------------------------------------- #
def _install_counters(build: _Build, thresholds) -> list[dict]:
    """Give every back edge a counter variable and a bound exit, and make the existing guards at the back edge's target mutually exclusive with that exit.

    Both the bound K and "who set K" are computed by :func:`hexis.legacy.fit.loop_bound_detail`: if
    the document states an iteration bound, the document's value is used; only when it does not
    does the compiler supply one as ``ceil(loop_margin × max visits to that target within a single
    trace)``. Returns the ledger of K values, which the compilation result copies into its report;
    the coverage report uses it to state which bounds the compiler added on its own.
    """
    return fit.install_counters(build.machine, build.max_visits,
                                margin=thresholds.loop_margin)


# --------------------------------------------------------------------------- #
# Set judge labels / complete the variable table / clean up placeholders
# --------------------------------------------------------------------------- #
def _finalize(build: _Build) -> None:
    m = build.machine
    for sid, st in m.states.items():
        if st.action.kind == "judge":
            observed = sorted(x for x in build.judge_labels.get(sid, set()) if x)
            if _ABSTAIN not in observed:
                observed = observed + [_ABSTAIN]
            st.action.labels = observed
            # examples come from the traces: keep one variable snapshot per observed label as its representative
            wkey = st.action.writes[0]
            exs, seen = [], set()
            for _tgt, snap in build.branch_obs.get(sid, []):
                lbl = snap.get(wkey)
                if lbl and lbl not in seen:
                    ex = {k: snap.get(k) for k in st.action.reads}
                    ex["label"] = lbl
                    exs.append(Example(**ex))
                    seen.add(lbl)
            st.action.examples = exs
    # variable table: add every variable that appeared; mark task input fields with init_from.
    known = {v.name for v in m.variables}
    used: dict[str, str] = {}
    for st in m.states.values():
        for w in getattr(st.action, "writes", []) or []:
            used.setdefault(w, "array" if w == "rows" else "string")
        for r in getattr(st.action, "reads", []) or []:
            used.setdefault(r, "array" if r == "rows" else "string")
    for name, t in used.items():
        if name not in known:
            ifrom = f"task.input.{name}" if name in build.input_keys else None
            m.variables.append(Variable(name=name, type=t, init_from=ifrom))
            known.add(name)
    for v in m.variables:                            # existing variables that are input fields get init_from filled in
        if v.name in build.input_keys and v.init is None and v.init_from is None:
            v.init_from = f"task.input.{v.name}"


# --------------------------------------------------------------------------- #
# Top level: one compilation round
# --------------------------------------------------------------------------- #
@dataclass
class CompileResult:
    machine: Machine
    calibration: dict
    report: dict
    findings: list[str]


def compile(doc: str, t_plus: list, t_minus: Optional[list] = None,
            thresholds=None, *, skill_id: str = "compiled", model=None,
            prohibitions: Optional[list] = None) -> CompileResult:
    """Compile a machine from accepted traces (+ optional rejected traces) by turn-by-turn sequential compilation.

    ``prohibitions`` are the human-annotated prohibitions, written straight into the machine: they
    cannot be compiled into the graph (a prohibition violation looks structurally the same as a
    normal run), so the judging layer enforces them.
    """
    from hexis.machine.schema import Thresholds
    thresholds = thresholds or Thresholds()
    t_minus = t_minus or []
    clauses = partition(doc)
    build = _Build(machine=_seed(skill_id))
    build.machine.prohibitions = list(prohibitions or [])
    for trace in t_plus:
        build.input_keys |= set(trace.task.get("input", {}))

    for trace in sorted(t_plus, key=lambda t: len(t.records)):
        _walk(build, trace.records, clauses, trace.task.get("input", {}))

    for sid in list(build.machine.states):
        _solve_branches(build, sid, thresholds, model)

    loop_bounds = _install_counters(build, thresholds)
    _finalize(build)

    # calibrate judge action error rates (if a model is given): rerun the judge on the observed snapshots and compare with the actual outcome
    calibration: dict = {}
    if model is not None:
        for sid, st in build.machine.states.items():
            if st.action.kind != "judge":
                continue
            wkey = st.action.writes[0]
            labeled = [(snap, snap.get(wkey))
                       for _t, snap in build.branch_obs.get(sid, [])
                       if snap.get(wkey)]
            if labeled:
                rate, sup = calibrate(st.action, labeled, model)
                st.action.error_rate = rate
                st.action.support = sup
                calibration[sid] = {"error_rate": rate, "support": sup}

    # rejected traces: confirm they are excluded (active repair is left to a later stage; only recorded here)
    unexcluded = [i for i, neg in enumerate(t_minus) if not excludes(build.machine, neg)]

    findings = structural_findings(build.machine)
    report = {
        "n_states": build.machine.n_states(),
        "clauses_seen": sorted({s.clause for s in build.machine.states.values() if s.clause}),
        "t_plus": len(t_plus),
        "t_plus_reproduced": sum(1 for t in t_plus if reproduces(build.machine, t)),
        "t_minus": len(t_minus),
        "t_minus_excluded": len(t_minus) - len(unexcluded),
        "loop_bounds": loop_bounds,          # each back edge's K and its source (document / compiler)
    }
    return CompileResult(machine=build.machine, calibration=calibration,
                         report=report, findings=findings)


# --------------------------------------------------------------------------- #
# One incremental compilation round (with whole-round undo)
# --------------------------------------------------------------------------- #
def compile_round(base: Optional[Machine], doc: str, traces: list, *,
                  thresholds=None, skill_id: str = "compiled", model=None,
                  prohibitions: Optional[list] = None) -> tuple[CompileResult, bool]:
    """Compile one round and validate the whole round. If structural checks fail, **undo** and fall back to ``base`` (the machine file is unchanged).

    Returns ``(result, applied)``: a false ``applied`` means this round was rolled back, and
    ``result.machine`` is ``base`` (an empty machine when there is no base). This implements "an
    extension lands only when validation passes": inject a trace that makes validation fail, and
    the whole round does not land.
    """
    from hexis.machine.schema import Thresholds, empty_machine
    cr = compile(doc, traces, thresholds=thresholds or Thresholds(),
                 skill_id=skill_id, model=model, prohibitions=prohibitions)
    if cr.findings:
        fallback_machine = base if base is not None else empty_machine(skill_id)
        rolled = CompileResult(machine=fallback_machine, calibration={},
                               report={"rolled_back": True, "findings": cr.findings},
                               findings=cr.findings)
        return rolled, False
    return cr, True


# --------------------------------------------------------------------------- #
# Split: one action signature collapsed into a single state, but two places need different
# successors and this state's variables cannot tell them apart
# --------------------------------------------------------------------------- #
def split_groups(preds: Sequence[str], outs: Sequence[str],
                 contingency: Optional[Mapping[str, Mapping[str, int]]] = None,
                 ) -> Optional[dict[str, list[str]]]:
    """Group predecessors by "which predecessors should share one clone": the **pure pairing core** of a split, which never touches the machine.

    ``contingency[pred][out]`` is an observed count: how many times execution entered from
    predecessor ``pred`` and then left through exit ``out``. When it is given, grouping follows the
    evidence: if every predecessor goes to exactly **one** exit (each row of the contingency table
    has exactly one nonzero cell), predecessors are grouped by exit and ``{exit: [predecessors...]}``
    is returned; if any predecessor has gone to two or more exits, predecessors cannot separate
    them and ``None`` is returned (that is a job for a judge action, not for a split).

    Without a contingency table it falls back to the **positional rule**: the i-th predecessor pairs
    with the i-th outgoing edge (the compiler builds edges in trace order, so order is the
    correspondence); ``None`` is returned when the numbers of predecessors and outgoing edges differ
    or are below 2. This is the original rule of :func:`split_by_predecessor`, kept as is for the
    old path.
    """
    preds, outs = list(preds), list(outs)
    if len(preds) < 2 or len(outs) < 2:
        return None
    if contingency is None:
        if len(preds) != len(outs):
            return None
        return {o: [p] for p, o in zip(preds, outs)}
    groups: dict[str, list[str]] = {}
    for p in preds:
        row = contingency.get(p) or {}
        hit = [o for o in outs if int(row.get(o, 0) or 0) > 0]
        if len(hit) != 1:
            return None                     # this predecessor went to 0 or >= 2 exits: predecessors cannot separate them
        groups.setdefault(hit[0], []).append(p)
    if len(groups) < 2:
        return None                         # every predecessor goes to the same exit: nothing to split
    return groups


def split_by_predecessor(machine: Machine, sid: str) -> bool:
    """Split a state that "was merged by signature but behaves inconsistently" by **predecessor**.

    When a state's outgoing edges cannot be told apart by its own variables (no branch guard can be
    learned), but the conflict corresponds one to one with "which state it was entered from", the
    state is cloned once per predecessor, and each clone keeps only its matching outgoing edge. This
    is the belated correction, in the Myhill-Nerode sense, that "these two histories are not
    actually equivalent". Returns whether a split happened.

    Pairing is delegated to the positional rule of :func:`split_groups`: the i-th predecessor pairs
    with the i-th outgoing edge. The receipt-issuing version is
    :meth:`hexis.legacy.checker.Checker.split_state` (explicit groups, clone ids assigned by the
    harness).
    """
    st = machine.states.get(sid)
    if st is None:
        return False
    preds = [(src, t) for src, t in machine.transitions_all() if t.to == sid]
    outs = list(st.transitions)
    out_ids = [f"{i}" for i in range(len(outs))]
    grouping = split_groups([f"{i}" for i in range(len(preds))], out_ids)
    if grouping is None:
        return False
    for oi, members in grouping.items():
        i = int(members[0])
        out_edge = outs[int(oi)]
        _psrc, pedge = preds[i]
        clone_id = f"{sid}_{i}"
        machine.states[clone_id] = State(
            id=clone_id, clause=st.clause,
            action=st.action.model_copy(deep=True),
            transitions=[Transition(cond=out_edge.cond, to=out_edge.to,
                                    inc=out_edge.inc, support=out_edge.support)])
        pedge.to = clone_id                     # redirect the predecessor to its own clone
    if machine.initial == sid:
        machine.initial = f"{sid}_0"
    del machine.states[sid]
    return True
