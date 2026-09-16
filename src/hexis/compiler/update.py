"""Trace-by-trace update.

Accept(M', T, P_k) ⟺ Check(M') ∧ ⋀_{T'∈P_k∪{T}} Replay(M', T', π̃_{T'}).
If everything passes, the change is committed at once; if anything fails, the copy is discarded and the machine and
the accepted set stay unchanged. Every trace gets at most two attempts by default: the first may change the tool of an
existing state, the second keeps the original tools.

Whether a trace can take part in the update is decided by **the current skill's rules**, not by fixed phase rules:
traces with unrecognized events do not (unsupported); traces that violate skill requirements or external prohibition
rules do not (excluded); traces whose claimed terminal disagrees with the evidence do not (violation); traces without
observable events are skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from hexis.compiler import check as _check
from hexis.compiler.align import align
from hexis.compiler.context import CompileContext
from hexis.compiler.modify import build_candidate
from hexis.compiler.traces import Prepared, prepare, violates_prohibitions
from hexis.machine.schema import Machine, Trace


@dataclass
class Accepted:
    prep: Prepared
    anchors: list


@dataclass
class UpdateResult:
    machine: Machine
    entries: list = field(default_factory=list)
    accepted: list = field(default_factory=list)

    def counts(self) -> dict:
        out: dict = {}
        for e in self.entries:
            out[e["status"]] = out.get(e["status"], 0) + 1
        return out


def _events_brief(prep: Prepared) -> list[str]:
    return [e.describe() for e in prep.events if e.kind != "model" or e.role == "output"]


def update_one(machine: Machine, prep: Prepared, accepted: list, ctx: CompileContext, *,
               attempts: int = 2) -> tuple[Machine, dict, Optional[Accepted]]:
    """Process one trace. Returns (new or unchanged machine, record, accepted item or None)."""
    entry: dict = {"trace": prep.trace_id, "verdict": prep.verdict, "events": _events_brief(prep),
                   "tau": prep.tau, "attempts": []}
    for k in range(1, max(1, attempts) + 1):
        allow = (k == 1)
        att: dict = {"attempt": k, "allow_realize": allow}
        al, why = align(machine, prep, ctx, allow_realize=allow)
        if al is None:
            att["result"], att["why"] = "no_path", why
            entry["attempts"].append(att)
            if k == 1:
                entry["status"], entry["why"] = "rejected", f"no alignment path: {why}"
                return machine, entry, None
            continue
        att["cost"] = al.cost
        att["path"] = [f"{s.state}:{s.how}" + ("+loop" if s.loop else "") + f"/{s.edge}" for s in al.slots]
        bd = build_candidate(machine, prep, al, ctx)
        att["changes"] = list(bd.changes)
        errs = _check.check(bd.machine, ctx)
        if errs:
            att["result"], att["why"] = "check_failed", errs[:8]
            entry["attempts"].append(att)
            continue
        rp = _check.replay(bd.machine, prep, bd.anchors)
        if not rp.ok:
            att["result"], att["why"] = "replay_failed", rp.why
            att["replay_path"] = rp.path
            entry["attempts"].append(att)
            continue
        broken = None
        for acc in accepted:
            r2 = _check.replay(bd.machine, acc.prep, acc.anchors)
            if not r2.ok:
                broken = (acc.prep.trace_id, r2.why)
                break
        if broken is not None:
            att["result"], att["why"] = "protected_failed", f"{broken[0]}: {broken[1]}"
            entry["attempts"].append(att)
            continue
        att["result"] = "accepted"
        entry["attempts"].append(att)
        entry.update({"status": "accepted", "cost": al.cost, "anchors": list(bd.anchors),
                      "changes": list(bd.changes), "path": rp.path})
        return bd.machine, entry, Accepted(prep=prep, anchors=list(bd.anchors))
    entry["status"] = "rejected"
    last = entry["attempts"][-1] if entry["attempts"] else {}
    entry["why"] = f"{last.get('result', '')}: {last.get('why', '')}"[:600]
    return machine, entry, None


def update(machine: Machine, traces: Sequence[tuple[Any, Optional[Trace], str]], ctx: CompileContext, *,
           attempts: int = 2, accepted_only: bool = False, progress=None) -> UpdateResult:
    """Process all traces. ``traces`` is the output of load_traces: (path, Trace or None, read error)."""
    res = UpdateResult(machine=machine)
    for path, trace, err in traces:
        name = Path(str(path)).stem
        if trace is None:
            res.entries.append({"trace": name, "status": "unreadable", "why": err})
        elif accepted_only and trace.verdict != "accepted":
            res.entries.append({"trace": name, "status": "skipped", "why": f"verdict {trace.verdict}, only accepted traces are kept"})
        else:
            prep = prepare(trace, ctx, source=str(path))
            base = {"trace": prep.trace_id, "verdict": prep.verdict, "events": _events_brief(prep),
                    "tau": prep.tau, "notes": list(prep.notes)}
            banned = violates_prohibitions(trace, res.machine)
            if prep.unsupported:
                res.entries.append({**base, "status": "unsupported",
                                    "why": "unrecognized events: " + ", ".join(f"step {s} kind={k}" for s, k in prep.unsupported)})
            elif banned:
                res.entries.append({**base, "status": "excluded", "why": banned})
            elif prep.requirement_violations:
                res.entries.append({**base, "status": "excluded", "why": "; ".join(prep.requirement_violations)})
            elif prep.violation:
                res.entries.append({**base, "status": "violation", "why": prep.violation})
            elif not prep.observable:
                res.entries.append({**base, "status": "skipped", "why": "no observable events"})
            else:
                m2, entry, acc = update_one(res.machine, prep, res.accepted, ctx, attempts=attempts)
                entry["notes"] = list(prep.notes)
                res.entries.append(entry)
                if acc is not None:
                    res.machine = m2
                    res.accepted.append(acc)
        if progress:
            e = res.entries[-1]
            progress(f"{e['trace']:<12} {e['status']:<11} {str(e.get('why') or '')[:110]}")
    return res


__all__ = ["Accepted", "UpdateResult", "update", "update_one"]
