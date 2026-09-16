"""Fold new traces into a compiled machine, one trace at a time.

A trace is accepted only if the candidate machine passes the static checks, replays the trace, and still replays
every trace accepted before (in this run and in earlier runs of the same build directory). Otherwise the machine
does not change. Three deciders produce the candidate for a trace:

* ``model``: a model decides every step (:class:`hexis.step_judge.ModelDecider`);
* ``file``: decisions are read from a JSON file written by a person or another tool;
* ``align``: no decisions; the deterministic alignment of :func:`hexis.compiler.update.update_one`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from hexis.builddir import BuildDir, TraceItem, counts
from hexis.compiler import check as _check
from hexis.compiler import stepwise as SW
from hexis.compiler.context import CompileContext
from hexis.compiler.update import update_one
from hexis.machine.schema import Machine


class BaselineError(RuntimeError):
    """The current machine does not pass the checks or no longer replays an accepted trace."""


def rebuild_accepted(progress: dict, stored: dict[str, TraceItem], ctx: CompileContext) -> list:
    out = []
    for a in progress.get("accepted") or []:
        item = stored.get(a["trace"])
        if item is None:
            raise BaselineError(f"accepted trace {a['trace']} is not stored in the build directory; "
                                "it cannot be protected")
        prep = SW.prepare_steps(item.trace, ctx, source=item.file, ignore=a.get("ignore") or [],
                                ignore_calls=a.get("ignore_calls") or [])
        out.append(SW.Accepted(trace=item.key, source=item.file, anchors=list(a["anchors"]),
                               ignore=list(a.get("ignore") or []), prep=prep,
                               ignore_calls=list(a.get("ignore_calls") or [])))
    return out


def baseline_check(machine: Machine, ctx: CompileContext, accepted: Sequence) -> tuple[list[str], list[str]]:
    """(errors, warnings). Check failures are errors once some trace is accepted; replay failures always are."""
    problems = [f"check: {e}" for e in _check.check(machine, ctx)]
    replays = []
    for acc in accepted:
        r = _check.replay(machine, acc.prep, acc.anchors)
        if not r.ok:
            replays.append(f"replay of {acc.trace}: {r.why}")
    if accepted:
        return problems + replays, []
    return replays, problems


# --------------------------------------------------------------------------- #
# deciders
# --------------------------------------------------------------------------- #
class AlignDecider:
    name = "align"


class FileDecider:
    """Decisions from a JSON object keyed by build key, file stem or (unique) task id."""

    name = "file"

    def __init__(self, spec: dict):
        self.spec = dict(spec or {})

    def spec_for(self, machine: Machine, ctx: CompileContext, trace: Any, *, source: str = "", key: str = "",
                 item: Optional[TraceItem] = None) -> Optional[dict]:
        if key in self.spec:
            return self.spec[key]
        if item is not None:
            stem = item.key.rsplit("-", 1)[0]
            if stem in self.spec:
                return self.spec[stem]
            if item.task_id and item.task_id in self.spec:
                return self.spec[item.task_id]
        return None


@dataclass
class Outcome:
    machine: Machine
    processed: int = 0
    stopped_after_insert: str = ""


def pending_items(progress: dict, stored: dict[str, TraceItem], *, redo: bool = False,
                  tasks: Sequence[str] = ()) -> list[TraceItem]:
    done = {e["trace"]: e.get("status") for e in progress.get("entries") or []}
    out = []
    for rec in progress.get("traces") or []:
        key = rec["key"]
        status = done.get(key)
        if status is not None and not (redo and status != "accepted"):
            continue
        item = stored.get(key)
        if item is None:
            continue
        if tasks:
            stem = key.rsplit("-", 1)[0]
            if not any(w in (key, stem, item.task_id) or stem.endswith(w) or item.task_id.endswith(w) for w in tasks):
                continue
        out.append(item)
    return out


def _put_entry(progress: dict, entry: dict) -> None:
    entries = progress.setdefault("entries", [])
    for i, e in enumerate(entries):
        if e.get("trace") == entry["trace"]:
            entries[i] = entry
            return
    entries.append(entry)


def process(bd: BuildDir, items: Sequence[TraceItem], *, machine: Machine, ctx: CompileContext, accepted: list,
            progress: dict, decider: Any, attempts: int = 2, max_traces: int = 0, accepted_only: bool = False,
            continue_after_insert: bool = False, run_n: int = 0, say: Callable[[str], None] = print) -> Outcome:
    outcome = Outcome(machine=machine)
    for item in items:
        if max_traces and outcome.processed >= max_traces:
            break
        m = outcome.machine
        before = set(m.states)
        if accepted_only and item.trace.verdict != "accepted":
            entry = {"trace": item.key, "status": "skipped", "why": f"verdict is {item.trace.verdict}; only accepted traces are used"}
            m2, acc = m, None
        elif decider.name == "align":
            prep = SW.prepare_steps(item.trace, ctx, source=item.file)
            pre = SW.pre_status(prep, item.trace, m)
            if pre is not None:
                entry = {"trace": item.key, "verdict": prep.verdict, "tau": prep.tau, "status": pre[0], "why": pre[1]}
                m2, acc = m, None
            else:
                m2, entry, acc0 = update_one(m, prep, accepted, ctx, attempts=attempts)
                acc = (SW.Accepted(trace=item.key, source=item.file, anchors=list(acc0.anchors), ignore=[], prep=prep)
                       if acc0 is not None else None)
        else:
            say(f"{item.key}:")
            spec = (decider.spec_for(m, ctx, item.trace, source=item.file, key=item.key, item=item)
                    if isinstance(decider, FileDecider) else
                    decider.spec_for(m, ctx, item.trace, source=item.file, key=item.key))
            if spec is None:
                continue
            m2, entry, acc = SW.update_with_decisions(m, item.trace, ctx, source=item.file, spec=spec,
                                                      accepted=accepted, attempts=attempts)
            if acc is not None:
                acc.trace = item.key
        entry["trace"] = item.key
        entry.update({"task_id": item.task_id, "run": run_n, "decider": decider.name})
        inserted = sorted(s for s in set(m2.states) - before)
        if acc is not None:
            outcome.machine = m2
            accepted.append(acc)
            progress["accepted"] = [a for a in progress.get("accepted") or [] if a["trace"] != item.key]
            progress["accepted"].append({"trace": item.key, "anchors": list(acc.anchors), "ignore": list(acc.ignore),
                                         "ignore_calls": list(acc.ignore_calls)})
            entry["new_states"] = inserted
        _put_entry(progress, entry)
        bd.save_progress(outcome.machine, progress)
        outcome.processed += 1
        why = entry.get("why") or (" → ".join(entry.get("anchors") or []) if entry.get("status") == "accepted" else "")
        say(f"{item.key:<32} {entry.get('status', ''):<11} {str(why)[:140]}")
        if (decider.name == "file" and acc is not None and inserted and not continue_after_insert):
            outcome.stopped_after_insert = item.key
            break
    return outcome


def preview(machine: Machine, ctx: CompileContext, items: Sequence[TraceItem], n: int,
            say: Callable[[str], None] = print) -> None:
    from hexis.cli import stepwise_view as SV
    shown = 0
    for item in items:
        if shown >= n:
            break
        prep = SW.prepare_steps(item.trace, ctx, source=item.file)
        pre = SW.pre_status(prep, item.trace, machine)
        if pre is not None:
            say(f"{item.key:<32} {pre[0]:<11} {pre[1][:120]}")
            continue
        view = SV.show_trace(machine, ctx, prep)
        view["trace"] = item.key
        say(SV.render_show(machine, view))
        say("")
        shown += 1
    steps = sum(len([e for e in SW.prepare_steps(i.trace, ctx, source=i.file).observable if e.kind != "end"])
                for i in items)
    say(f"{len(items)} pending traces, {steps} steps to decide (a model decider asks about one question per step)")


def write_report(bd: BuildDir, machine: Machine, ctx: CompileContext, progress: dict) -> str:
    from hexis.cli.compile import describe
    c = counts(progress)
    lines = ["# Build report", "",
             f"Skill {ctx.skill_id}: {len(progress.get('traces') or [])} stored traces, task inputs "
             f"{list(ctx.task_inputs)}, tools {sorted(ctx.tools)}; {len(ctx.label_rules)} label rules, "
             f"{len(ctx.requirements)} requirements, {len(ctx.terminal_conditions)} terminal conditions.",
             f"Machine: {machine.n_states()} states ({len(machine.states)} including end states), "
             f"{len(machine.transitions_all())} transitions, {len(machine.variables)} variables.", "",
             "| Outcome | Traces |", "|---|---:|"]
    for k in ("accepted", "rejected", "excluded", "violation", "unsupported", "skipped", "unreadable"):
        if c.get(k):
            lines.append(f"| {k} | {c[k]} |")
    lines += ["", "## Runs", "", "| Run | Command | Decider | Model | Status | Outcomes |", "|---:|---|---|---|---|---|"]
    for r in bd.manifest.get("runs") or []:
        model = (r.get("model") or {}).get("model", "")
        lines.append(f"| {r.get('n')} | {r.get('command')} | {r.get('decider', '')} | {model} | {r.get('status')} | "
                     f"{json.dumps(r.get('counts') or {}, sort_keys=True)} |")
    lines += ["", "## Traces", "", "| Trace | Run | Decider | Outcome | Reason / path |", "|---|---:|---|---|---|"]
    for e in progress.get("entries") or []:
        why = e.get("why") or ""
        if e.get("status") == "accepted":
            why = " → ".join(e.get("anchors", []))
        lines.append(f"| {e['trace']} | {e.get('run', '')} | {e.get('decider', '')} | {e.get('status')} | "
                     f"{str(why)[:150].replace('|', '/')} |")
    lines += ["", "## Machine", "", "```"] + describe(machine) + ["```", ""]
    errs = _check.check(machine, ctx)
    lines.append("Final checks: " + ("all passed" if not errs else "; ".join(errs)))
    text = "\n".join(lines) + "\n"
    (bd.root / "report.md").write_text(text, encoding="utf-8")
    return text


def refresh_docs(bd: BuildDir, machine: Machine, ctx: CompileContext, progress: dict, *, guide: bool = True) -> None:
    (bd.root / "context.json").write_text(json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(bd, machine, ctx, progress)
    if guide:
        from hexis.cli.guide import generate
        g = bd.manifest.get("guide") or {}
        init_log = json.loads((bd.root / "init_log.json").read_text(encoding="utf-8")) \
            if (bd.root / "init_log.json").is_file() else {}
        generate(machine_path=bd.root / "machine.json", out=bd.root, skill_dir=bd.skill_dir,
                 tools_path=bd.root / "tools.json", embed_skill=bool(g.get("embed_skill")),
                 retries=int(g.get("retries", 3)), manifest=bd.manifest, progress=progress,
                 clause_map=init_log.get("clause_map") or None)


__all__ = ["AlignDecider", "BaselineError", "FileDecider", "Outcome", "baseline_check", "pending_items", "preview",
           "process", "rebuild_accepted", "refresh_docs", "write_report"]
