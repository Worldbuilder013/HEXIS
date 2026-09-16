"""Fitting: the few **truly reusable** algorithms in the compiler, pulled out of the gatekeeper into their own module.

:mod:`hexis.legacy.compiler` is the pipeline skeleton that "walks traces, builds states and wires
edges", and it is tightly bound to the table_clean record format; but the skeleton embeds three
pipeline-independent algorithms that any machine can use, extracted here:

* :func:`learn_cond`: **learn a branch guard from variable snapshots**. Given "the variable
  snapshots observed for each successor", find a set of pairwise mutually exclusive predicates
  that turn "which branch to take" from an on-the-spot model judgment into a deterministic
  transition in the machine. This is the most valuable step of the whole compilation: once a guard
  is learned, that branch no longer needs the model.
* :func:`install_counter` / :func:`install_counters`: **give back edges a counter and a bound
  exit**. Loops are the bit of extra expressiveness an EFSM has over a plain automaton, and the
  only place where it might not halt; the bound exit turns "a learned loop" into "a loop that is
  guaranteed to stop".
* :func:`calibrate`: **calibrate a judge action's error rate**. The only step at compile time that
  still touches the model: rerun a judge action on snapshots with correct labels, measure how often
  it is wrong, and write that into the machine for review (the source of each judge's error rate).

The three gates (which learn_cond requires a guard to pass before accepting it)
-------------------------------------------------------------------------------
1. **Support**: each branch has at least ``min_support`` observed snapshots. A branch seen only
   once is not a branch, it is a coincidence.
2. **Holdout accuracy**: guards are selected on the **fit split** and their accuracy is then
   measured on the **holdout split**; below ``acc_thr`` they are rejected. Without this gate, the
   learned guard is often a predicate that "memorizes the training snapshots", for example one that
   separates the two branches at a midpoint of a counter variable: perfect on the observations,
   wrong as soon as the task changes.
3. **Provable mutual exclusion**: the guards of the branches are never true at the same time **at
   the symbolic level** (variables are reduced to finitely many configurations and enumerated one
   by one, the same method :func:`hexis.machine.checks.structural_findings` uses to decide mutual
   exclusion), not merely non-overlapping by chance on the few observed snapshots. Determinism of
   the machine requires the former.

If any gate fails, ``None`` is returned, and the caller (``compiler._solve_branches``) then routes
the whole branch to FALLBACK: **when it cannot compute a guard, it says so plainly**, which is
safer than learning a half-correct guard.

Where the loop bound K comes from
---------------------------------
:func:`loop_bound` makes the choice of K explicit: if the document states an iteration bound, **the
document decides**; only when it does not does the compiler supply one as
``ceil(margin × max observed iterations)`` and mark it ``"compiler"`` in the ledger. This
distinction must be recorded: when a skill's document never states an iteration bound, every K is
introduced by the compiler, and the coverage report must say truthfully "this bound is not a
document requirement, the compiler added it so the machine halts".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence

from hexis.machine import cond
from hexis.machine.checks import _back_edges, _reachable
from hexis.machine.schema import Machine, Transition, Variable

__all__ = [
    "LoopBound", "calibrate", "candidate_atoms", "install_counter",
    "install_counters", "learn_cond", "loop_bound", "loop_bound_detail",
    "mutually_exclusive", "separating",
]

#: Upper bound on the number of configurations for the mutual exclusion enumeration: beyond it the result is "cannot prove" (better to reject the guard than to pretend it was proven).
_CONFIG_CAP = 4096

#: Representative value for string variables whose domain cannot be fixed: stands for "any value other than the listed literals".
_OTHER = "__other__"

#: When gate three falls back: max alternative guards kept per branch, and max combinations tried for the whole set (guards against combinatorial explosion).
_MAX_ALT = 6
_ASSIGN_CAP = 256


# --------------------------------------------------------------------------- #
# Candidate atomic predicates
# --------------------------------------------------------------------------- #
def _flatten(snaps: Any) -> list[dict]:
    """Flatten either ``[snapshot, ...]`` or ``{target: [snapshot, ...]}`` into a list of snapshots."""
    if isinstance(snaps, Mapping):
        return [s for group in snaps.values() for s in group]
    return list(snaps or [])


def _task_input_names(variables: Sequence) -> set:
    """Names in the variable table that are initialized from the task input (non-empty ``init_from``). They never take part in branch guards."""
    return {v.name for v in variables if getattr(v, "init_from", None)}


def candidate_atoms(snaps: Any, variables: Sequence) -> list[str]:
    """Enumerate candidate atomic predicates: numbers use midpoints between adjacent observations and the observed endpoints, strings use equality/inequality.

    ``snaps`` may be a list of snapshots or ``{target: [snapshots]}`` (which is flattened).
    ``variables`` is the machine's variable table, used only to read types: a variable typed as a
    number is treated as a number, and an unregistered variable is judged by its observed values.
    Sets/arrays do not produce atoms directly: either they cannot enter a finite configuration, or
    they should be expressed with ``empty()``/``nonempty()``.

    **Task inputs produce no atoms** (variables with a non-empty ``init_from``, see
    :data:`_task_input_names`). They are the identity of "which problem this is": the input workbook
    path, the problem text, which differ on every run. Using them as branch predicates inevitably
    learns something like "if the input workbook is /tmp/<sandbox>/input.xlsx, go this way": 100%
    separable on the few training traces, and forever false on another run in another sandbox. This
    actually happened in real compiled output. Branch guards should read **what happened during
    execution** (a tool's exit code, a judge's label, a counter), not what the problem is called.

    Returns an **order-preserving, deduplicated** list of expression strings: the order is the
    search order, and two calls on the same snapshots return the same list.
    """
    allsnaps = _flatten(snaps)
    vtypes = {v.name: v.type for v in variables}
    skip = _task_input_names(variables)
    out: list[str] = []
    keys: set = set()
    for s in allsnaps:
        keys |= set(s.keys())
    for k in sorted(keys - skip):
        vals = [s.get(k) for s in allsnaps if k in s]
        t = vtypes.get(k)
        if t in ("integer", "number") or all(isinstance(v, (int, float))
                                             and not isinstance(v, bool) for v in vals):
            nums = sorted({float(v) for v in vals
                           if isinstance(v, (int, float)) and not isinstance(v, bool)})
            for a, b in zip(nums, nums[1:]):
                mid = (a + b) / 2
                out += [f"{k} < {mid}", f"{k} >= {mid}"]
            for n in nums:
                out += [f"{k} < {n}", f"{k} >= {n}", f"{k} == {int(n) if n.is_integer() else n}"]
        else:
            for v in {v for v in vals if isinstance(v, str)}:
                out += [f"{k} == {v!r}", f"{k} != {v!r}"]
    seen, uniq = set(), []
    for e in out:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


# --------------------------------------------------------------------------- #
# Searching for separating predicates
# --------------------------------------------------------------------------- #
def _truth(expr: str, snap: dict) -> Optional[bool]:
    """Evaluate on one snapshot. An invalid guard or an undefined variable -> ``None`` (neither true nor false)."""
    try:
        return bool(cond.evaluate(expr, snap))
    except cond.CondError:
        return None


def _iter_exprs(atoms: Sequence[str], max_atoms: int) -> Iterator[str]:
    """Search order: single atoms first, then pairwise conjunctions, up to ``max_atoms``-ary conjunctions."""
    for r in range(1, max(1, int(max_atoms)) + 1):
        for combo in itertools.combinations(atoms, r):
            yield " and ".join(combo)


def _separates(expr: str, mine: Sequence[dict], others: Sequence[dict]) -> bool:
    """Always true on ``mine`` and always false on ``others`` (an evaluation error never counts as separating)."""
    return (all(_truth(expr, s) is True for s in mine)
            and all(_truth(expr, s) is False for s in others))


def separating(atoms: Sequence[str], mine: Sequence[dict], others: Sequence[dict],
               *, max_atoms: int = 2) -> Optional[str]:
    """Find a predicate that fully separates ``mine`` from ``others``. Returns ``None`` if there is none."""
    for expr in _iter_exprs(atoms, max_atoms):
        if _separates(expr, mine, others):
            return expr
    return None


# --------------------------------------------------------------------------- #
# Gate two: the holdout split
# --------------------------------------------------------------------------- #
def _holdout_split(snaps: Sequence[dict], ratio: float) -> tuple[list[dict], list[dict]]:
    """Deterministically split one branch's snapshots into (fit split, holdout split). **Uses no random numbers.**

    The holdout split is taken by evenly spaced sampling (``ratio=0.2`` means one in every 5), not by
    cutting off the tail: traces are sorted by length, so cutting the tail would systematically put
    only long traces in the holdout split. A branch with fewer than 2 snapshots cannot hold any out,
    and its holdout split is empty.
    """
    items = list(snaps)
    if ratio <= 0 or len(items) < 2:
        return items, []
    stride = max(2, int(round(1.0 / ratio)))
    hold = items[::stride]
    holdset = set(range(0, len(items), stride))
    fit = [s for i, s in enumerate(items) if i not in holdset]
    if not fit:                                  # do not let an extreme ratio empty the fit split
        return items, []
    return fit, hold


def _holdout_rate(expr: str, hold_mine: Sequence[dict],
                  hold_others: Sequence[dict]) -> float:
    """Accuracy of the guard on the holdout split: true on its own branch, false on the others. Counts as 1.0 when nothing is held out."""
    total = len(hold_mine) + len(hold_others)
    if not total:
        return 1.0
    ok = sum(1 for s in hold_mine if _truth(expr, s) is True)
    ok += sum(1 for s in hold_others if _truth(expr, s) is False)
    return ok / total


# --------------------------------------------------------------------------- #
# Gate three: provable mutual exclusion (at the symbolic level, not "non-overlapping by chance on the observations")
# --------------------------------------------------------------------------- #
def _domain(var: str, atoms: Sequence[cond.Atom], observed: Sequence,
            vtypes: Mapping[str, str]) -> Optional[list]:
    """Fix a finite enumeration domain for a variable: thresholds/literals from the guards + observed values + one "other" representative."""
    va = [a for a in atoms if a.var == var]
    if any(a.op in ("empty", "nonempty") for a in va):
        return [[], [1]]                                  # two representatives: empty / non-empty
    nums = sorted({float(a.const) for a in va
                   if a.op in ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")
                   and isinstance(a.const, (int, float)) and not isinstance(a.const, bool)})
    obs_nums = sorted({float(v) for v in observed
                       if isinstance(v, (int, float)) and not isinstance(v, bool)})
    if vtypes.get(var) == "boolean":
        return [True, False]
    if nums or obs_nums or vtypes.get(var) in ("integer", "number"):
        reps = set(nums) | set(obs_nums)
        if nums:
            reps |= {min(nums) - 1, max(nums) + 1}
        elif not reps:
            reps = {0, 1}
        return sorted(reps)
    strs = {a.const for a in va if a.op in ("Eq", "NotEq") and isinstance(a.const, str)}
    strs |= {v for v in observed if isinstance(v, str)}
    if strs:
        return sorted(strs) + [_OTHER]
    return None


def mutually_exclusive(exprs: Sequence[str], snaps: Any = (),
                       variables: Sequence = ()) -> bool:
    """Prove that no two guards in a set hold at the same time: reduce variables to finitely many configurations and enumerate them one by one.

    The enumeration domain is spanned by the atoms that appear in the guards (thresholds, literals)
    plus the observed values: numbers take the thresholds themselves and one representative on each
    side, strings take the literals that appeared plus one "other". If a domain cannot be fixed, or
    the number of configurations exceeds :data:`_CONFIG_CAP`, the result is always **cannot prove**
    (returns false): what cannot be proven is not accepted, and FALLBACK is preferred.
    """
    exprs = [e for e in exprs if e]
    if len(exprs) < 2:
        return True                                       # 0 or 1 guards are trivially mutually exclusive
    atoms: list[cond.Atom] = []
    allvars: set[str] = set()
    for e in exprs:
        try:
            atoms += cond.atoms_of(e)
            allvars |= cond.vars_of(e)
        except cond.CondError:
            return False
    allsnaps = _flatten(snaps)
    vtypes = {v.name: v.type for v in variables}
    domains: dict[str, list] = {}
    size = 1
    for v in sorted(allvars):
        observed = [s[v] for s in allsnaps if v in s]
        dom = _domain(v, atoms, observed, vtypes)
        if not dom:
            return False
        domains[v] = dom
        size *= len(dom)
        if size > _CONFIG_CAP:
            return False
    names = list(domains)
    for combo in itertools.product(*(domains[n] for n in names)):
        env = dict(zip(names, combo))
        fired = sum(1 for e in exprs if _truth(e, env) is True)
        if fired >= 2:
            return False
    return True


# --------------------------------------------------------------------------- #
# Learning branch guards (passing the three gates)
# --------------------------------------------------------------------------- #
def learn_cond(snaps_by_target: Mapping[str, Sequence[dict]], variables: Sequence,
               *, max_atoms: int = 2, min_support: int = 2,
               holdout_ratio: float = 0.2, acc_thr: float = 0.9) -> Optional[dict]:
    """Find a set of pairwise mutually exclusive separating predicates from each target's variable snapshots. Returns ``None`` if none can be learned.

    ``snaps_by_target`` = ``{target state: [variable snapshot, ...]}``. Returns ``{target state:
    guard string}``; whether the default target keeps its guard is up to the caller
    (``compiler._solve_branches`` leaves the branch with the most support unguarded as the default
    edge).

    All three gates are required (see the module docs): support ``min_support``, holdout accuracy
    ``acc_thr``, and provable mutual exclusion. ``max_atoms`` bounds the arity of conjunctions
    (default 2: single atoms first, then pairwise conjunctions).
    """
    targets = list(snaps_by_target)
    if len(targets) < 2:
        return None
    # gate one: support, every branch needs enough observations
    for tgt in targets:
        if len(snaps_by_target[tgt]) < min_support:
            return None

    atoms = candidate_atoms(snaps_by_target, variables)
    split = {t: _holdout_split(snaps_by_target[t], holdout_ratio) for t in targets}

    def passing(tgt: str, limit: int) -> list[str]:
        """Candidates for this branch that pass gates one and two (in search order, at most ``limit``)."""
        fit_mine, hold_mine = split[tgt]
        fit_others = [s for o in targets if o != tgt for s in split[o][0]]
        hold_others = [s for o in targets if o != tgt for s in split[o][1]]
        out: list[str] = []
        for cand in _iter_exprs(atoms, max_atoms):
            # gate two: separates on the fit split and is still right on the holdout split; predicates that memorize the training snapshots are filtered out here
            if not _separates(cand, fit_mine, fit_others):
                continue
            if _holdout_rate(cand, hold_mine, hold_others) < acc_thr:
                continue
            out.append(cand)
            if len(out) >= limit:
                break
        return out

    # fast path: take the first passing candidate for each branch and go straight to gate three
    first = {t: passing(t, 1) for t in targets}
    if any(not v for v in first.values()):
        return None
    chosen = {t: v[0] for t, v in first.items()}
    if mutually_exclusive(list(chosen.values()), snaps_by_target, variables):
        return chosen

    # gate three: provable mutual exclusion (symbolic enumeration, not just non-overlapping observations). When the
    # branches' first choices do not form a mutually exclusive set, step back and look for one among each branch's
    # first few candidates: "best for this branch alone" is not the same as "best together".
    alts = {t: passing(t, _MAX_ALT) for t in targets}
    if any(not v for v in alts.values()):
        return None
    tried = 0
    for combo in itertools.product(*(alts[t] for t in targets)):
        tried += 1
        if tried > _ASSIGN_CAP:
            break
        cand = dict(zip(targets, combo))
        if mutually_exclusive(list(cand.values()), snaps_by_target, variables):
            return cand
    return None


# --------------------------------------------------------------------------- #
# Loop bound K: the document decides, and the compiler supplies one only when the document is silent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoopBound:
    """Ledger entry for one choice of K. ``source`` is ``"document"`` or ``"compiler"``."""

    k: int
    source: str
    observed_max: int
    margin: float
    doc_bound: Optional[int] = None


def loop_bound_detail(observed_max: int, *, margin: float = 1.5,
                      doc_bound: Optional[int] = None) -> LoopBound:
    """Compute K and **record who set it**. If the document gives a bound it is used, otherwise the compiler supplies one with a margin."""
    obs = max(int(observed_max or 0), 1)
    if doc_bound is not None:
        return LoopBound(k=max(1, int(doc_bound)), source="document",
                         observed_max=obs, margin=float(margin), doc_bound=int(doc_bound))
    k = max(1, math.ceil(float(margin) * obs))
    return LoopBound(k=k, source="compiler", observed_max=obs, margin=float(margin))


def loop_bound(observed_max: int, *, margin: float = 1.5,
               doc_bound: Optional[int] = None) -> int:
    """K = ``ceil(margin × max observed iterations)``; **if the document states a bound, the document wins**.

    To get it together with its source (the coverage report must say truthfully whether the
    compiler added this bound on its own), use :func:`loop_bound_detail`.
    """
    return loop_bound_detail(observed_max, margin=margin, doc_bound=doc_bound).k


# --------------------------------------------------------------------------- #
# Back edge counter variables + bound exits
# --------------------------------------------------------------------------- #
def install_counter(machine: Machine, src: str, dst: str, *, k: int,
                    name: str) -> Transition:
    """Give back edge ``src->dst`` a counter variable ``name`` and an exit at bound ``k``, and return that exit edge.

    It does three things, and all three are required: attach ``inc`` to the back edge; conjoin
    ``name < k`` onto every **existing guarded outgoing edge** of the target state; insert
    ``name >= k -> FALLBACK`` at the **front** of the target state's outgoing edges. The second is the
    crux of mutual exclusion (which machine determinism relies on): without the conjunction, "counter
    full, go to FALLBACK" and "keep looping" both hold at the full-counter configuration, and the
    structural check immediately reports an overlap.

    Idempotent: if the target state already has a default exit that reads ``name``, that edge is
    returned as is and nothing is installed twice.

    **When the document already closes this loop, it is not closed a second time.** Document
    skeletons often come with their own bound exit (``repair_count >= 3 -> s_done``: when the rounds
    are used up, hand in an unverified result). Inserting another ``repair_count >= 3 -> FALLBACK``
    with the same name and threshold makes both hold at the full-counter configuration, and the
    structural check immediately reports overlapping guards (observed in practice: once document
    edges were no longer removed, a whole batch of back edges was rejected for this). So the
    document's exit is accepted first, and only ``inc`` plus ``count < k`` on the other guards are
    added.
    """
    st = machine.states[src]
    edge = next((t for t in st.transitions if t.to == dst), None)
    if edge is None:
        raise KeyError(f"{src} has no edge to {dst}, cannot install a counter")
    if not machine.var(name):
        machine.variables.append(Variable(name=name, type="integer", init=0))
    edge.inc = name
    tgt = machine.states[dst]
    exist = next((g for g in tgt.transitions
                  if g.cond and name in cond.vars_of(g.cond) and g.to == machine.fallback),
                 None)
    if exist is not None:
        return exist
    # the document's own bound exit (wherever it goes, not necessarily FALLBACK): accept it instead of inserting another with the same name
    doc_exit = next((g for g in tgt.transitions
                     if g.cond and name in cond.vars_of(g.cond)), None)
    for g in tgt.transitions:                    # conjoin count<k onto existing guards, keeping mutual exclusion
        if g.cond and name not in cond.vars_of(g.cond):
            g.cond = f"({g.cond}) and {name} < {k}"
    if doc_exit is not None:
        return doc_exit
    exit_edge = Transition(cond=f"{name} >= {k}", to=machine.fallback)
    tgt.transitions.insert(0, exit_edge)
    return exit_edge


def install_counters(machine: Machine, max_visits: Mapping[str, int], *,
                     margin: float = 1.5,
                     doc_bounds: Optional[Mapping[str, int]] = None) -> list[dict]:
    """Install a counter and a bound exit on every back edge in the machine that has no counter yet, and return the ledger of K values.

    ``max_visits`` = ``{state: max number of times the state was entered within a single trace}``;
    ``doc_bounds`` are iteration bounds stated explicitly in the document (``{state: K}``), which
    override the value inferred from observations when given. Each ledger entry looks like
    ``{"back_edge","var","k","source","observed_max","margin"}``, where ``source`` says whether this
    K is a document requirement or was supplied by the compiler; the coverage report copies it
    verbatim.
    """
    ledger: list[dict] = []
    reachable = _reachable(machine)
    for src, edge in _back_edges(machine, reachable):
        if edge.inc:
            continue
        tgt_id = edge.to
        name = f"{tgt_id}_count"
        lb = loop_bound_detail(max_visits.get(tgt_id, 1), margin=margin,
                               doc_bound=(doc_bounds or {}).get(tgt_id))
        install_counter(machine, src, tgt_id, k=lb.k, name=name)
        ledger.append({"back_edge": f"{src}->{tgt_id}", "var": name, "k": lb.k,
                       "source": lb.source, "observed_max": lb.observed_max,
                       "margin": lb.margin})
    return ledger


# --------------------------------------------------------------------------- #
# Calibrating a judge action's error rate (the only model touchpoint at compile time)
# --------------------------------------------------------------------------- #
def calibrate(judge, samples: Sequence[tuple], *, model) -> float:
    """Run the judge on snapshots with correct labels and measure its error rate.

    ``samples`` = ``[(variable snapshot, correct label), ...]``. Error rate = the fraction of wrong
    answers among **non-abstentions**; abstentions are counted separately and are not errors (an
    abstention means "I don't know", and its cost is going to FALLBACK, not going the wrong way).
    Returns 0.0 when every answer is an abstention.
    """
    errors = nonabstain = 0
    for values, gold in samples:
        vread = {k: values.get(k) for k in judge.reads}
        pred = model.classify(prompt=judge.prompt, values=vread,
                              labels=judge.labels,
                              examples=tuple(e.model_dump() for e in judge.examples))
        if pred == judge.abstain:
            continue
        nonabstain += 1
        if pred != gold:
            errors += 1
    return errors / nonabstain if nonabstain else 0.0
