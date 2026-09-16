"""Mechanical part of stepwise incremental compilation.

The judge (an agent) is not here. This module does only three deterministic things:

1. cut a trace into steps and compute the candidate set of every step: tier 1 is the states of the same kind and
   tool reachable in one hop from the current position, tier 2 is the remaining states of the same kind and tool
   (they need a new transition and must not bypass document states that the rules require);
2. turn the agent's per-step decisions (match / new / ignore) into an alignment path, hand it to
   :func:`modify.build_candidate` to build the candidate, then run Check, replay of the new trace and replay of the
   accepted traces; the acceptance rule is the same as in :mod:`update`;
3. persist progress (machine.json + progress.json) so that a batch of traces can be decided in parts and resumed at
   any time.

The "proposal" is a deterministic default rule (if tier 1 has candidates take the one with the closest labels,
otherwise new) that only saves the judge some writing: the judge still reviews every step, and an accepted proposal
is recorded as its decision too (source=proposal).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from hexis.compiler import check as _check
from hexis.compiler.align import NEW, Alignment, Slot, loop_cost
from hexis.compiler.common import anchors, entry_anchors, required_states, skip_blocked, status_known, zero_paths
from hexis.compiler.context import CompileContext, apply_labels, check_requirements, terminal_for
from hexis.compiler.modify import build_candidate
from hexis.compiler.traces import Event, Prepared, end_state_for, prepare, violates_prohibitions
from hexis.machine.schema import Machine, Trace


# --------------------------------------------------------------------------- #
# Machine fingerprint and trace preparation
# --------------------------------------------------------------------------- #
def fingerprint(m: Machine) -> str:
    parts: list = []
    for sid in sorted(m.states):
        st = m.states[sid]
        a = st.action
        parts.append([sid, a.kind, getattr(a, "name", ""), getattr(a, "phase", ""),
                      sorted(getattr(a, "labels", []) or []), bool(getattr(a, "observable", False))])
        for t in st.transitions:
            parts.append([sid, t.cond, t.to, t.inc or ""])
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def strip_tools(trace: Trace, names: Sequence[str]) -> Trace:
    """Remove the records of harness marker tools (such as skill2fsm_begin)."""
    bad = set(names)
    trace.records = [r for r in trace.records
                     if not ((r.action or {}).get("kind") == "tool" and str((r.action or {}).get("name")) in bad)]
    return trace


def prepare_steps(trace: Trace, ctx: CompileContext, *, source: str = "", ignore: Sequence[int] = (),
                  ignore_calls: Sequence[int] = ()) -> Prepared:
    """After prepare(), drop the steps the agent decided are not part of the skill workflow (``ignore``, by event
    index) or individual calls inside a step (``ignore_calls``, by record step number: environment checks or temporary
    directory recomputation mixed into a step), then recompute labels, requirements and the end class.

    Every event carries ``orig``: its index before steps were dropped. Decisions are always keyed by ``orig``, so
    dropping steps does not shift the numbering."""
    prep = prepare(trace, ctx, source=source)
    for ev in prep.events:
        ev.orig = ev.index                                   # type: ignore[attr-defined]
    if ignore or ignore_calls:
        ign = {int(i) for i in ignore}
        bad_calls = {int(i) for i in ignore_calls}
        kept = []
        for ev in prep.events:
            if getattr(ev, "orig", -1) in ign:
                continue
            if ev.kind == "tool" and bad_calls:
                ev.calls = [c for c in ev.calls if c.step not in bad_calls]
                if not ev.calls:
                    continue
            kept.append(ev)
        prep.events = _remerge(kept)
        for i, ev in enumerate(prep.events):
            ev.index = i
            ev.labels = set()
        task_in = prep.task_input
        apply_labels(prep.events, ctx, task_in)
        prep.requirement_violations = check_requirements(prep.events, ctx, task_in)
        prep.tau, prep.evidence = terminal_for(prep.events, ctx, task_in)
        prep.claims = prep.events[-1].terminal if prep.events else ""
        conditioned = set(ctx.conditioned_terminals())
        prep.violation = (f"claimed to reach {prep.claims}, but the evidence only supports {prep.tau}"
                          if prep.claims in conditioned and prep.claims != prep.tau else None)
    return prep


def _remerge(events: list) -> list:
    """After dropping steps, merge consecutive calls with the same tool and base label into one step again (calls
    separated only by narration count as consecutive), by the same rule as segment_trace."""
    out: list = []
    for ev in events:
        prev = None
        for cand in reversed(out):
            if cand.kind == "model" and cand.role == "narration":
                continue
            prev = cand
            break
        if prev is not None and prev.kind == "tool" and ev.kind == "tool" and prev.tool == ev.tool and prev.label == ev.label:
            prev.calls.extend(ev.calls)
            if ev.intent:
                prev.intent = (prev.intent + " " + ev.intent).strip()
            prev.merged = list(getattr(prev, "merged", [])) + [getattr(ev, "orig", ev.index)]   # type: ignore[attr-defined]
            continue
        out.append(ev)
    return out


def pre_status(prep: Prepared, trace: Trace, machine: Machine) -> Optional[tuple[str, str]]:
    """Statuses that can be settled without a decision: unsupported / excluded / violation / skipped. None = needs a
    decision."""
    banned = violates_prohibitions(trace, machine)
    if prep.unsupported:
        return "unsupported", "unrecognized events: " + ", ".join(f"step {s} kind={k}" for s, k in prep.unsupported)
    if banned:
        return "excluded", banned
    if prep.requirement_violations:
        return "excluded", "; ".join(prep.requirement_violations)
    if prep.violation:
        return "violation", prep.violation
    if not prep.observable:
        return "skipped", "no observable events"
    return None


# --------------------------------------------------------------------------- #
# Candidate sets and proposals
# --------------------------------------------------------------------------- #
def same_kind(m: Machine, ev: Event, sid: str) -> bool:
    """Deterministic pre-filter: same kind; tool steps also need the same tool name and the same base label (a probe
    state does not write files, so it cannot produce an apply step)."""
    a = m.states[sid].action
    if ev.kind == "tool":
        return a.kind == "tool" and a.name == ev.tool and (a.phase or "") == (ev.label or "")
    if ev.kind == "model":
        return a.kind == "model" and bool(getattr(a, "observable", False))
    if ev.kind == "user":
        return a.kind == "user"
    if ev.kind == "end":
        return a.kind == "end"
    return False


def candidates(m: Machine, ctx: CompileContext, ev: Event, pos: Optional[str], b: Optional[bool],
               term: str, *, allow_tier2: bool = True) -> tuple[list[str], list[str]]:
    """C(p, b): (tier 1, tier 2). An end step only has the end state corresponding to τ*."""
    if ev.kind == "end":
        return ([term] if term in m.states else []), []
    if pos is None:
        t1 = list(entry_anchors(m))
    elif pos.startswith(NEW):
        t1 = []
    else:
        t1 = list(zero_paths(m, pos, status_known(ctx, m.states[pos], b)))
    t1 = [s for s in t1 if same_kind(m, ev, s)]
    t2: list[str] = []
    if allow_tier2:
        req = required_states(m, ctx)
        for s in anchors(m):
            if s in t1 or not same_kind(m, ev, s) or m.states[s].action.kind == "end":
                continue
            if pos is not None and not pos.startswith(NEW) and skip_blocked(m, pos, s, req):
                continue
            t2.append(s)
    return t1, t2


def propose(m: Machine, ev: Event, t1: list[str], t2: list[str], term: str = "") -> dict:
    """Default proposal: if tier 1 has candidates take the first one with the same derived labels, otherwise the first;
    for a deliverable step prefer candidates that lead to the τ* terminal; no tier 1 → new."""
    if ev.kind == "end":
        return {"d": "match", "state": t1[0]} if t1 else {"d": "new"}
    if t1:
        pool = t1
        if ev.kind == "model" and term:
            reach = [s for s in t1 if term in zero_paths(m, s, {})]
            pool = reach or t1
        same = [s for s in pool if set(getattr(m.states[s].action, "labels", []) or []) == set(ev.labels)]
        return {"d": "match", "state": (same or pool)[0]}
    return {"d": "new"}


# --------------------------------------------------------------------------- #
# Decisions → alignment → candidate → check → acceptance
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    d: str                      # match / new / ignore
    state: str = ""
    purpose: str = ""
    clause: str = ""
    source: str = "agent"       # agent / proposal


def parse_decisions(spec: Any) -> tuple[list[int], dict[int, Decision], bool, list[int]]:
    """Decisions for one trace: {"accept_proposals": bool, "ignore_calls": [record step number…],
    "steps": {"<orig>": {"d", "state"?, "purpose"?, "clause"?}}}. Steps in "steps" with d=ignore are dropped before
    alignment."""
    spec = spec or {}
    accept = bool(spec.get("accept_proposals", False))
    ignore_calls = [int(x) for x in (spec.get("ignore_calls") or [])]
    ignore: list[int] = []
    steps: dict[int, Decision] = {}
    for k, v in (spec.get("steps") or {}).items():
        if isinstance(v, str):
            v = {"d": v}
        d = str(v.get("d") or v.get("decision") or "").strip()
        if d == "ignore":
            ignore.append(int(k))
            continue
        steps[int(k)] = Decision(d=d, state=str(v.get("state") or ""), purpose=str(v.get("purpose") or ""),
                                 clause=str(v.get("clause") or ""))
    return ignore, steps, accept, ignore_calls


def align_from_decisions(m: Machine, ctx: CompileContext, prep: Prepared, steps: dict[int, Decision],
                         accept_proposals: bool, *, allow_tier2: bool) -> tuple[Optional[Alignment], str, list]:
    """Turn per-step decisions into an alignment path. The strict attempt (allow_tier2=False) turns tier 2 matches
    into new."""
    term = end_state_for(m, prep.tau)
    ea = entry_anchors(m)
    slots: list[Slot] = []
    log: list[dict] = []
    pos: Optional[str] = None
    b: Optional[bool] = None
    for ev in prep.observable:
        orig = getattr(ev, "orig", ev.index)
        t1, t2 = candidates(m, ctx, ev, pos, b, term, allow_tier2=True)
        rec: dict = {"i": orig, "event": ev.describe(), "tier1": t1, "tier2": t2}
        if ev.kind == "end":
            if not t1:
                return None, f"the machine has no end state {term}", log
            slots.append(Slot(index=ev.index, state=term, how="end", edge="keep" if pos is not None else "start"))
            rec.update({"decision": "end", "state": term})
            log.append(rec)
            break
        d = steps.get(orig)
        if d is None:
            if not accept_proposals:
                return None, f"step {orig} has no decision", log
            p = propose(m, ev, t1, t2, term)
            d = Decision(d=p["d"], state=p.get("state", ""), source="proposal")
        if d.d not in ("match", "new"):
            return None, f"step {orig} has an unrecognized decision {d.d!r}", log
        tier = 0
        if d.d == "match":
            if d.state in t1:
                tier = 1
            elif d.state in t2 and allow_tier2:
                tier = 2
            elif d.state in t2:
                d = Decision(d="new", purpose=d.purpose, clause=d.clause, source=d.source + "→new(strict attempt)")
            else:
                return None, f"step {orig} was matched to {d.state!r}, which is not a candidate (T1 {t1} / T2 {t2})", log
        if d.d == "match":
            st = m.states[d.state]
            how = "label" if set(getattr(st.action, "labels", []) or []) != set(ev.labels) else "match"
            has_loop = d.state in zero_paths(m, d.state, status_known(ctx, st, True))
            edge = ("keep" if d.state in ea else "start") if pos is None else ("keep" if tier == 1 else "add")
            slots.append(Slot(index=ev.index, state=d.state, how=how, c_loop=loop_cost(ev, has_loop), edge=edge))
            pos = d.state
        else:
            if d.purpose:
                ev.intent = (d.purpose + (" — " + ev.intent if ev.intent else ""))[:600]
            slots.append(Slot(index=ev.index, state=f"{NEW}{ev.index}", how="new", c_loop=loop_cost(ev, False),
                              edge="start" if pos is None else "add"))
            pos = f"{NEW}{ev.index}"
        b = ev.ok if ev.kind == "tool" else None
        rec.update({"decision": d.d, "state": d.state, "tier": tier, "source": d.source,
                    "purpose": d.purpose, "clause": d.clause, "slot": slots[-1].how + "/" + slots[-1].edge
                    + ("+loop" if slots[-1].loop else "")})
        log.append(rec)
    if not slots or slots[-1].how != "end":
        return None, "the path does not finish with an end state", log
    cost = sum(3 if s.edge in ("add", "start") else 0 for s in slots) + sum(4 for s in slots if s.is_new) \
        + sum(1 for s in slots if s.how == "label") + sum(s.c_loop for s in slots)
    return Alignment(slots=slots, cost=cost, end_state=term), "", log


def _stamp_clauses(build_machine: Machine, alignment: Alignment, anchors_out: list[str], log: list,
                   ctx: CompileContext) -> None:
    """Stamp the clause id given by the agent on new states (and their gates); log and slots correspond one to one
    (one entry per observable event)."""
    ids = {c[0] for c in ctx.clauses}
    for slot, sid, rec in zip(alignment.slots, anchors_out, log):
        rec["state_id"] = sid
        if not slot.is_new or sid not in build_machine.states:
            continue
        clause = str(rec.get("clause") or "")
        if clause and clause in ids:
            build_machine.states[sid].clause = clause
            for gs in build_machine.states.values():
                if gs.action.kind == "model" and not getattr(gs.action, "observable", False) \
                        and [t.to for t in gs.transitions] == [sid]:
                    gs.clause = clause


@dataclass
class Accepted:
    trace: str
    source: str
    anchors: list
    ignore: list
    prep: Optional[Prepared] = None
    ignore_calls: list = field(default_factory=list)


def update_with_decisions(machine: Machine, trace: Trace, ctx: CompileContext, *, source: str, spec: Any,
                          accepted: list[Accepted], attempts: int = 2,
                          key: str = "") -> tuple[Machine, dict, Optional[Accepted]]:
    """Process one trace. Returns (new or unchanged machine, record, accepted item or None).

    ``key`` identifies the trace in the record and in the accepted item (default: the task id, or the file stem when
    the trace has no task id). Callers that process several traces of the same task must pass distinct keys.

    The judge may give ``{"exclude": "<reason>"}`` at trace level: the steps of this trace cannot be classified by the
    skill workflow (for example the real modification is hidden in a script file written by the write tool and run
    by a command with no write signal), so the trace does not take part in the update."""
    if isinstance(spec, dict) and spec.get("exclude"):
        prep0 = prepare_steps(trace, ctx, source=source)
        return machine, {"trace": key or prep0.trace_id, "verdict": prep0.verdict, "tau": prep0.tau,
                         "events": [e.describe() for e in prep0.observable], "status": "excluded",
                         "why": "excluded by the judge: " + str(spec["exclude"])}, None
    ignore, steps, accept_props, ignore_calls = parse_decisions(spec)
    prep = prepare_steps(trace, ctx, source=source, ignore=ignore, ignore_calls=ignore_calls)
    entry: dict = {"trace": key or prep.trace_id, "verdict": prep.verdict, "tau": prep.tau, "ignored": sorted(ignore),
                   "ignored_calls": sorted(ignore_calls),
                   "events": [e.describe() for e in prep.observable], "attempts": [], "fingerprint_before": fingerprint(machine)}
    pre = pre_status(prep, trace, machine)
    if pre is not None:
        entry["status"], entry["why"] = pre
        return machine, entry, None
    modes = [True, False][:max(1, attempts)]
    for k, allow in enumerate(modes, 1):
        att: dict = {"attempt": k, "allow_tier2": allow}
        prep_k = prepare_steps(trace, ctx, source=source, ignore=ignore, ignore_calls=ignore_calls)   # intent is rewritten by new decisions, so redo this every attempt
        al, why, log = align_from_decisions(machine, ctx, prep_k, steps, accept_props, allow_tier2=allow)
        att["decisions"] = log
        if al is None:
            att["result"], att["why"] = "no_path", why
            entry["attempts"].append(att)
            break
        att["cost"] = al.cost
        att["path"] = [f"{s.state}:{s.how}" + ("+loop" if s.loop else "") + f"/{s.edge}" for s in al.slots]
        bd = build_candidate(machine, prep_k, al, ctx)
        _stamp_clauses(bd.machine, al, bd.anchors, log, ctx)
        att["changes"] = list(bd.changes)
        errs = _check.check(bd.machine, ctx)
        if errs:
            att["result"], att["why"] = "check_failed", errs[:8]
            entry["attempts"].append(att)
            continue
        rp = _check.replay(bd.machine, prep_k, bd.anchors)
        if not rp.ok:
            att["result"], att["why"] = "replay_failed", rp.why
            att["replay_path"] = rp.path
            entry["attempts"].append(att)
            continue
        broken = None
        for acc in accepted:
            if acc.prep is None:
                raise ValueError(f"accepted trace {acc.trace} is not available, so the candidate cannot be checked "
                                 "against it")
            r2 = _check.replay(bd.machine, acc.prep, acc.anchors)
            if not r2.ok:
                broken = (acc.trace, r2.why)
                break
        if broken is not None:
            att["result"], att["why"] = "protected_failed", f"{broken[0]}: {broken[1]}"
            entry["attempts"].append(att)
            continue
        att["result"] = "accepted"
        entry["attempts"].append(att)
        entry.update({"status": "accepted", "cost": al.cost, "anchors": list(bd.anchors), "changes": list(bd.changes),
                      "path": rp.path, "inserted": [c for c in bd.changes if c.startswith("added state ") or c.startswith("added observable ")],
                      "fingerprint_after": fingerprint(bd.machine)})
        return bd.machine, entry, Accepted(trace=key or prep.trace_id, source=source, anchors=list(bd.anchors),
                                           ignore=sorted(ignore), prep=prep_k, ignore_calls=sorted(ignore_calls))
    entry["status"] = "rejected"
    last = entry["attempts"][-1] if entry["attempts"] else {}
    entry["why"] = f"{last.get('result', '')}: {last.get('why', '')}"[:600]
    return machine, entry, None


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
@dataclass
class Progress:
    machine: Machine
    entries: list = field(default_factory=list)
    accepted: list = field(default_factory=list)       # list[Accepted]
    agent_log: list = field(default_factory=list)      # decisions for every step of every trace

    def done(self) -> dict:
        return {e["trace"]: e["status"] for e in self.entries}

    def counts(self) -> dict:
        out: dict = {}
        for e in self.entries:
            out[e["status"]] = out.get(e["status"], 0) + 1
        return out

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "machine.json").write_text(json.dumps(json.loads(self.machine.model_dump_json(by_alias=True)),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "progress.json").write_text(json.dumps({
            "fingerprint": fingerprint(self.machine), "counts": self.counts(),
            "accepted": [{"trace": a.trace, "source": a.source, "anchors": a.anchors, "ignore": a.ignore,
                          "ignore_calls": a.ignore_calls} for a in self.accepted],
            "entries": self.entries, "agent_log": self.agent_log}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out: Path, m0: Machine, ctx: CompileContext, traces: dict[str, Trace]) -> "Progress":
        p = out / "progress.json"
        if not p.is_file():
            return cls(machine=m0.model_copy(deep=True))
        data = json.loads(p.read_text(encoding="utf-8"))
        from hexis.machine.schema import load_machine
        m = load_machine(out / "machine.json")
        pr = cls(machine=m, entries=list(data.get("entries") or []), agent_log=list(data.get("agent_log") or []))
        for a in data.get("accepted") or []:
            tr = traces.get(a["trace"])
            if tr is None:
                raise ValueError(f"accepted trace {a['trace']} from {p} is not in the trace directory; every accepted "
                                 "trace must stay available because the machine is checked against it")
            prep = prepare_steps(tr, ctx, source=a["source"], ignore=a.get("ignore") or [],
                                 ignore_calls=a.get("ignore_calls") or [])
            pr.accepted.append(Accepted(trace=a["trace"], source=a["source"], anchors=list(a["anchors"]),
                                        ignore=list(a.get("ignore") or []), prep=prep,
                                        ignore_calls=list(a.get("ignore_calls") or [])))
        return pr


__all__ = ["Accepted", "Decision", "Progress", "align_from_decisions", "candidates", "fingerprint",
           "parse_decisions", "pre_status", "prepare_steps", "propose", "strip_tools", "update_with_decisions"]
