"""Update a compiled machine with new traces; by default a model decides every trace step.

For each new trace, the decider accounts for every step: the step is produced by an existing state (match), needs
a new state (new), or is not part of the skill's workflow (ignore); a whole trace can be excluded. The candidate
machine is accepted only if it passes the static checks, replays the new trace, and still replays every trace
accepted earlier in this build directory. Progress is saved after every trace, model answers are cached in
``BUILD/decisions.jsonl``, and running the command again continues where it stopped.

    hexis-agent update --build BUILD --traces NEW_TRACES --model MODEL_ID --base-url URL
    hexis-agent update --build BUILD --show 3                  # preview pending traces, no model calls
    hexis-agent update --build BUILD --decider file --decisions decisions.json
    hexis-agent update --build BUILD --traces NEW_TRACES --decider align   # deterministic, no model

Exit status: 0 done (some traces may be rejected), 2 usage, configuration or build problem, 3 the model endpoint
failed or the run was interrupted (progress is saved; run the command again to continue).
"""
from __future__ import annotations

import argparse
import json
import pathlib

from hexis.cli import _model


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--build", required=True, help="build directory written by `compile`")
    ap.add_argument("--traces", default=None, help="directory of new traces (*.jsonl); without it, pending traces "
                                                   "already stored in the build directory are processed")
    ap.add_argument("--decider", choices=("model", "file", "align"), default=None,
                    help="who decides the steps: a model (default), a decisions file (default when --decisions is "
                         "given), or deterministic alignment without decisions")
    ap.add_argument("--decisions", default=None, help="decisions JSON for --decider file")
    ap.add_argument("--show", type=int, default=0, help="print the next N pending traces with their candidates "
                                                        "and exit (no model calls, nothing written)")
    ap.add_argument("--attempts", type=int, default=2, help="candidate attempts per trace")
    ap.add_argument("--task", action="append", default=None,
                    help="only process traces of this task (build key, file name or task_id); repeatable")
    ap.add_argument("--accepted-only", action="store_true", help="skip traces whose verdict is not accepted")
    ap.add_argument("--max-traces", type=int, default=0, help="process at most N traces in this run (0 = all)")
    ap.add_argument("--redo", action="store_true", help="process again the traces that were not accepted")
    ap.add_argument("--no-cache", action="store_true", help="ask the model again even when an answer is cached")
    ap.add_argument("--continue-after-insert", action="store_true",
                    help="with --decider file: keep going after a trace added states (by default the run stops so "
                         "the remaining decisions can be reviewed against the changed machine)")
    ap.add_argument("--no-guide", action="store_true", help="do not regenerate GUIDE.md and PROMPT.md")
    _model.add_endpoint_args(ap)
    a = ap.parse_args(argv)
    decider_name = a.decider or ("file" if a.decisions else "model")
    if decider_name == "file" and not a.decisions:
        ap.error("--decider file needs --decisions")

    from hexis import updater as U
    from hexis.builddir import BuildDir, BuildError, trace_files
    from hexis.compiler.context import build_context
    from hexis.llm.model_iface import ModelUnavailable
    from hexis.skill_loader import load_agent_skill, markdown_clauses

    try:
        bd = BuildDir.open(pathlib.Path(a.build))
        progress = bd.load_progress()
        machine = bd.load_machine()
        stored = bd.stored_traces(progress)
    except BuildError as exc:
        print(f"error: {exc}")
        return 2
    if bd.machine_changed_outside(machine, progress):
        print("note: machine.json changed since the last run; it is checked before any trace is processed")

    skill = load_agent_skill(bd.skill_dir)
    clauses = markdown_clauses(skill.body)
    new_files = trace_files(pathlib.Path(a.traces)) if a.traces else []
    if a.traces and not new_files:
        print(f"no *.jsonl traces in {a.traces}")

    if a.show:
        preview_progress = json.loads(json.dumps(progress))
        added, _stats = [], {}
        if new_files:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                shadow = BuildDir(pathlib.Path(tmp), dict(bd.manifest))
                added, _stats = shadow.stage(new_files, preview_progress, run=0, say=print)
        items = dict(stored)
        items.update({i.key: i for i in added})
        ctx = build_context(skill.slug, skill.body, clauses, [i.trace for i in items.values()],
                            registry=bd.registry(), rules=bd.rules())
        pending = U.pending_items(preview_progress, items, redo=a.redo, tasks=a.task or ())
        U.preview(machine, ctx, pending, a.show)
        return 0

    run = bd.start_run("update", decider=decider_name, traces={"directory": str(a.traces) if a.traces else ""})
    try:
        added, stats = bd.stage(new_files, progress, run=run["n"], say=print)
        bd.save_progress(machine, progress)
        stored.update({i.key: i for i in added})
        ctx = build_context(skill.slug, skill.body, clauses, [stored[r["key"]].trace for r in progress["traces"]],
                            registry=bd.registry(), rules=bd.rules())
        accepted = U.rebuild_accepted(progress, stored, ctx)
        errors, warnings = U.baseline_check(machine, ctx, accepted)
        for w in warnings:
            print(f"warning: {w}")
        if errors:
            print("the current machine does not pass the baseline check, nothing was changed:")
            for e in errors[:20]:
                print(f"  {e}")
            print("if the build relies on tool definitions inferred from traces, pass an explicit tool registry "
                  "when compiling (--tools)")
            bd.finish_run(run, status="failed", traces=stats, errors=errors[:20])
            return 2
        pending = U.pending_items(progress, stored, redo=a.redo, tasks=a.task or ())
        print(f"skill {skill.slug}: {len(stored)} stored traces ({stats['staged']} new, {stats['duplicates']} already "
              f"stored); {len(accepted)} accepted; {len(pending)} pending; decider {decider_name}")
    except U.BaselineError as exc:
        print(f"error: {exc}")
        bd.finish_run(run, status="failed")
        return 2

    before = machine
    usage: dict = {}
    questions: dict = {}
    model_rec: dict = {}
    status = "ok"
    code = 0
    try:
        if decider_name == "align":
            outcome = U.process(bd, pending, machine=machine, ctx=ctx, accepted=accepted, progress=progress,
                                decider=U.AlignDecider(), attempts=a.attempts, max_traces=a.max_traces,
                                accepted_only=a.accepted_only, run_n=run["n"])
        elif decider_name == "file":
            spec = json.loads(pathlib.Path(a.decisions).read_text(encoding="utf-8"))
            outcome = U.process(bd, pending, machine=machine, ctx=ctx, accepted=accepted, progress=progress,
                                decider=U.FileDecider(spec), attempts=a.attempts, max_traces=a.max_traces,
                                accepted_only=a.accepted_only, continue_after_insert=a.continue_after_insert,
                                run_n=run["n"])
            if outcome.stopped_after_insert:
                print(f"{outcome.stopped_after_insert} added states; stopped so the remaining decisions can be "
                      "reviewed (use --show, or --continue-after-insert)")
        else:
            from hexis.step_judge import DecisionLog, ModelDecider
            with _model.open_model(a) as (adapter, model_rec):
                print(f"endpoint: {model_rec.get('model')} @ {model_rec.get('base_url')}")
                decider = ModelDecider(adapter, model_id=str(model_rec.get("model") or ""),
                                       log=DecisionLog(bd.root / "decisions.jsonl"), use_cache=not a.no_cache,
                                       say=print)
                try:
                    outcome = U.process(bd, pending, machine=machine, ctx=ctx, accepted=accepted, progress=progress,
                                        decider=decider, attempts=a.attempts, max_traces=a.max_traces,
                                        accepted_only=a.accepted_only, run_n=run["n"])
                finally:
                    usage = adapter.usage()
                    questions = dict(decider.stats)
        machine = outcome.machine
    except ModelUnavailable as exc:
        print(f"model endpoint failed: {exc}")
        print("progress is saved; run the same command again to continue")
        status, code = "interrupted", 3
        machine = bd.load_machine()
    except KeyboardInterrupt:
        print("interrupted; progress is saved; run the same command again to continue")
        status, code = "interrupted", 3
        machine = bd.load_machine()
    except Exception as exc:
        if exc.__class__.__name__ == "EnvError":
            print(f"error: {exc}")
            bd.finish_run(run, status="failed")
            return 2
        raise
    from hexis.builddir import counts
    from hexis.compiler.stepwise import fingerprint
    run_counts = {}
    for e in progress.get("entries") or []:
        if e.get("run") == run["n"]:
            run_counts[e.get("status")] = run_counts.get(e.get("status"), 0) + 1
    bd.finish_run(run, status=status, model=model_rec, traces={**stats}, counts=run_counts, questions=questions,
                  usage=usage, machine_before=fingerprint(before), machine_after=fingerprint(machine))
    U.refresh_docs(bd, machine, ctx, progress, guide=not a.no_guide)
    print(f"\nthis run: {run_counts}; build total: {counts(progress)}; machine {machine.n_states()} states, "
          f"{len(machine.transitions_all())} transitions")
    if questions:
        print(f"questions: {questions}; usage: {usage}")
    print(f"outputs: {bd.root}/machine.json, progress.json, report.md" + ("" if a.no_guide else ", GUIDE.md, PROMPT.md"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
