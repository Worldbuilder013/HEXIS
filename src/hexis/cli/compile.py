"""Build a machine from a skill document and traces (initialization + trace update).

The compiler only depends on the skill document, tool definitions and the trace format:

0. **Compile context.** Read all traces and discover the task input fields and tools (observed
   arguments and outputs). Tool interfaces come from a registry (``--backend`` / ``--tools``); missing
   ones are inferred and marked as such. Skill rules (terminals, derived labels, requirements, terminal
   conditions) come from ``<skill>/compile.json`` or ``--rules``; without either, a model extracts them
   from the document and every rule is checked against its quoted text.
1. **Initialization** (model). The model drafts an efsm-v1 machine from the document, its clauses,
   the tools, task inputs, terminals and rules, and redrafts under check feedback for up to
   ``--rounds`` rounds. The accepted draft is normalized and must pass the update-stage checks.
   ``--machine-init`` skips this step.
2. **Update** (no model). Traces are processed in file-name order: normalize, label, check
   requirements, classify the ending, align, build a candidate, check, replay the new trace and all
   accepted traces, then accept or reject.

   With ``--decider model`` a model decides every trace step instead (see ``hexis-agent update``).

Without ``--traces`` only the initial machine is built. The model endpoint is chosen with ``--model``,
``--base-url`` and ``--api-key-env`` (or ``--provider`` and the environment; see ``.env.example``).

Outputs in ``--out``: ``context.json``, ``machine_init.json``, ``init_log.json``, ``machine.json``,
``update_log.json``, ``report.md``, and the build files that ``hexis-agent update`` continues from
(``build.json``, ``skill/SKILL.md``, ``tools.json``, ``rules.json``, ``traces/``, ``progress.json``), plus
``GUIDE.md`` and ``PROMPT.md`` (see ``hexis-agent guide``).

    hexis-agent compile --skill SKILL_DIR --out BUILD --model MODEL_ID --base-url URL
    hexis-agent compile --skill SKILL_DIR --traces TRACE_DIR --out BUILD
    hexis-agent compile --skill SKILL_DIR --traces TRACE_DIR --out BUILD --machine-init M0.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib

from hexis.cli import _model
from hexis.compiler import check as _check
from hexis.compiler.context import build_context, load_rules
from hexis.compiler.init import extract_rules, initialize, install_rules, normalize
from hexis.compiler.traces import load_traces
from hexis.compiler.update import update
from hexis.machine.schema import Machine, load_machine
from hexis.skill_loader import load_agent_skill, markdown_clauses
from hexis.tools.backends import REGISTRIES, registry
from hexis.tools.toolspec import load_registry

RULES_FILE = "compile.json"


def dump(m: Machine, path: pathlib.Path) -> None:
    path.write_text(json.dumps(json.loads(m.model_dump_json(by_alias=True)),
                               ensure_ascii=False, indent=2), encoding="utf-8")


def describe(m: Machine) -> list[str]:
    """One or two lines per state: action and ordered outgoing transitions."""
    lines = []
    for sid, st in m.states.items():
        a = st.action
        if a.kind == "tool":
            labs = "+".join(getattr(a, "labels", []) or [])
            tail = (f"{a.name}{'/' + a.phase if a.phase else ''}{'[' + labs + ']' if labs else ''} "
                    f"{json.dumps(a.input, ensure_ascii=False)[:60]} writes={a.writes}")
        elif a.kind == "model":
            tail = f"model{'*' if getattr(a, 'observable', False) else ''} reads={a.reads} writes={a.writes}"
        elif a.kind == "judge":
            tail = f"judge {a.labels} reads={a.reads}"
        elif a.kind == "user":
            tail = f"user writes={a.writes}"
        else:
            tail = f"end {a.terminal}"
        edges = ", ".join(f"[{t.cond or '·'}]→{t.to}" + (f"(+{t.inc})" if t.inc else "")
                          + (f"×{t.support}" if t.support else "") for t in st.ordered_transitions())
        lines.append(f"  {sid:<16} {st.origin:<9} {tail}")
        if edges:
            lines.append(f"  {'':<16} {'':<9} → {edges}")
    return lines


def report(m0: Machine, m: Machine, res, init_res, ctx, out: pathlib.Path) -> str:
    c = res.counts()
    lines = ["# Compilation report", "",
             f"Skill {ctx.skill_id}: {ctx.n_traces} traces, task inputs {list(ctx.task_inputs)}, "
             f"tools {sorted(ctx.tools)}; {len(ctx.label_rules)} label rules, {len(ctx.requirements)} requirements, "
             f"{len(ctx.terminal_conditions)} terminal conditions.",
             f"Initial machine: {m0.n_states()} states, {len(m0.transitions_all())} transitions"
             + (f", initialized in {len(init_res.attempts)} rounds." if init_res else ", loaded from --machine-init."),
             f"Final machine: {m.n_states()} states ({len(m.states)} including terminals), "
             f"{len(m.transitions_all())} transitions, {len(m.variables)} variables.", "",
             "| Outcome | Traces |", "|---|---:|"]
    for k in ("accepted", "rejected", "excluded", "violation", "unsupported", "skipped", "unreadable"):
        if c.get(k):
            lines.append(f"| {k} | {c[k]} |")
    lines += ["", "## Traces", "", "| Trace | Verdict | Outcome | Cost | Reason / path |", "|---|---|---|---:|---|"]
    for e in res.entries:
        why = e.get("why") or ""
        if e.get("status") == "accepted":
            why = " → ".join(e.get("anchors", []))
        lines.append(f"| {e['trace']} | {e.get('verdict', '')} | {e['status']} | {e.get('cost', '')} | {str(why)[:140]} |")
    lines += ["", "## Final machine", "", "```"] + describe(m) + ["```", ""]
    errs = _check.check(m, ctx)
    lines.append("Final checks: " + ("all passed" if not errs else "; ".join(errs)))
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--skill", required=True, help="skill directory containing SKILL.md")
    ap.add_argument("--traces", default=None,
                    help="directory of traces (*.jsonl); without it only the initial machine is built")
    ap.add_argument("--out", required=True, help="output (build) directory")
    ap.add_argument("--backend", default="opencode",
                    help=f"tool registry name ({', '.join(sorted(REGISTRIES))}); 'none' infers tools from the traces only")
    ap.add_argument("--tools", default=None, help="tool registry JSON file (overrides --backend)")
    ap.add_argument("--rules", default=None,
                    help=f"skill rules JSON (default: <skill>/{RULES_FILE}; without one the model extracts rules)")
    ap.add_argument("--machine-init", default=None, help="existing initial machine; skips the model call")
    ap.add_argument("--provider", default="default",
                    help="endpoint profile: 'default' reads MODEL / BASE_URL / API_KEY; NAME reads NAME_MODEL / "
                         "NAME_BASE_URL / NAME_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=16384, help="maximum tokens per model response")
    ap.add_argument("--rounds", type=int, default=3, help="maximum initialization rounds")
    ap.add_argument("--attempts", type=int, default=2, help="candidate attempts per trace")
    ap.add_argument("--task", action="append", default=None,
                    help="only use traces of this task (header task_id or file name); repeatable")
    ap.add_argument("--accepted-only", action="store_true", help="only use traces whose verdict is accepted")
    ap.add_argument("--lenient", action="store_true", help="continue with the update even if the initial machine fails the checks")
    ap.add_argument("--init-only", action="store_true", help="stop after initialization")
    ap.add_argument("--decider", choices=("align", "model"), default="align",
                    help="how traces update the machine: deterministic alignment (default, no model calls) or a "
                         "model deciding every step (as in `update`)")
    ap.add_argument("--no-cache", action="store_true", help="with --decider model: ask again even when an answer is cached")
    ap.add_argument("--no-guide", action="store_true", help="do not write GUIDE.md and PROMPT.md")
    ap.add_argument("--embed-skill", action="store_true", help="append the skill document to PROMPT.md")
    _model.add_endpoint_args(ap, skip=("provider", "max_tokens"))
    a = ap.parse_args(argv)

    from hexis.builddir import BuildDir, dump_machine, trace_files
    from hexis.compiler.context import rules_dict
    from hexis.llm.env import EnvError
    from hexis.llm.model_iface import ModelUnavailable

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skill = load_agent_skill(pathlib.Path(a.skill))
    doc = skill.body
    clauses = markdown_clauses(doc)

    # 0. compile context
    traces = load_traces(pathlib.Path(a.traces)) if a.traces else []
    for p, t, e in traces:
        if t is None:
            print(f"unreadable trace {pathlib.Path(str(p)).name}: {e}")
    if a.tools:
        reg = load_registry(a.tools)
        tools_source = f"file:{a.tools}"
    elif a.backend and a.backend != "none":
        reg = registry(a.backend)
        tools_source = f"backend:{a.backend}"
    else:
        reg = {}
        tools_source = "none"
    rules_path = pathlib.Path(a.rules) if a.rules else pathlib.Path(a.skill) / RULES_FILE
    rules = load_rules(rules_path) if rules_path.is_file() else None
    ctx = build_context(skill.slug, doc, clauses, [t for _p, t, _e in traces if t is not None],
                        registry=reg, rules=rules)
    print(f"skill {skill.slug}: document {len(doc)} chars, {len(clauses)} clauses; {ctx.n_traces} traces")
    print(f"task inputs {list(ctx.task_inputs)}; tools " + ", ".join(
        f"{n} ({s.source}, {s.calls} calls)" for n, s in ctx.tools.items()))
    print("rules: " + (str(rules_path) if rules is not None else "no rules file; extracted from the document by the model"))
    for n in ctx.notes:
        print("  " + n)

    # the build directory keeps the inputs so that `update` can continue from this compilation
    if BuildDir.is_build(out):
        print(f"note: {out} already holds a build; it is replaced")
    bd = BuildDir.create(out, skill_dir=pathlib.Path(a.skill), skill=skill, tools=reg, tools_source=tools_source,
                         rules=None, rules_source=(f"file:{rules_path}" if rules is not None else
                                                   ("default" if a.machine_init else "model")))
    progress = {"traces": [], "entries": [], "accepted": []}
    run = bd.start_run("compile", decider=a.decider if a.traces else "",
                       init={"source": "import" if a.machine_init else "model"})
    staged, stage_stats = bd.stage(trace_files(pathlib.Path(a.traces)) if a.traces else [], progress, run=run["n"],
                                   say=lambda _m: None)
    key_of = {item.source: item for item in staged}

    def save_context() -> None:
        (out / "context.json").write_text(json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    def fail(status: str = "failed") -> int:
        save_context()
        bd.save_manifest()
        bd.finish_run(run, status=status, traces=stage_stats)
        return 2

    need_model = (not a.machine_init) or (bool(a.traces) and a.decider == "model" and not a.init_only)
    stack = contextlib.ExitStack()
    model = None
    endpoint: dict = {}
    usage: dict = {}
    try:
        if need_model:
            model, endpoint = stack.enter_context(_model.open_model(a, repair_chars=None))
            print(f"endpoint: {endpoint.get('model')} @ {endpoint.get('base_url')}")
    except EnvError as exc:
        print(f"error: {exc}")
        stack.close()
        return fail()

    with stack:
        # 1. initialization
        init_res = None
        try:
            if a.machine_init:
                if rules is None:
                    print("no rules file and initialization skipped: using a single default terminal and no requirements")
                m0 = load_machine(pathlib.Path(a.machine_init))
                normalize(m0, ctx)
                errs = _check.check(m0, ctx)
                print(f"initial machine from {a.machine_init}: {m0.n_states()} states; checks "
                      + ("passed" if not errs else f"failed {errs[:3]}"))
                if errs and not a.lenient:
                    return fail()
            else:
                rule_notes: list = []
                if rules is None:
                    extracted, rule_notes = extract_rules(ctx, model, progress=print)
                    install_rules(ctx, extracted)
                    (out / "rules_extracted.json").write_text(json.dumps(
                        {k: v for k, v in ctx.to_dict().items()
                         if k in ("terminals", "label_rules", "requirements", "terminal_conditions")},
                        ensure_ascii=False, indent=2), encoding="utf-8")
                init_res = initialize(ctx, model=model, rounds=a.rounds, lenient=a.lenient, progress=print)
                usage = model.usage()
                init_res.notes = rule_notes + init_res.notes
                (out / "init_log.json").write_text(json.dumps({
                    "attempts": init_res.attempts, "notes": init_res.notes,
                    "check_errors": init_res.check_errors, "clause_map": init_res.clause_map,
                    "usage": usage}, ensure_ascii=False, indent=2), encoding="utf-8")
                if init_res.machine is None:
                    print("initialization failed: " + (("update-stage checks failed: " + "; ".join(init_res.check_errors[:5]))
                                                       if init_res.check_errors else f"no draft passed G_init in {a.rounds} rounds"))
                    return fail()
                m0 = init_res.machine
                print(f"initial machine M0: {m0.n_states()} states, {len(m0.transitions_all())} transitions"
                      + (f"; checks failed, continuing (--lenient): {init_res.check_errors[:3]}" if init_res.check_errors else ""))
        except ModelUnavailable as exc:
            print(f"model endpoint failed: {exc}")
            return fail()
        save_context()
        (out / "rules.json").write_text(json.dumps(rules_dict(ctx), ensure_ascii=False, indent=2), encoding="utf-8")
        dump(m0, out / "machine_init.json")
        print("\n".join(describe(m0)))
        run["model"] = endpoint
        if a.init_only:
            bd.finish_run(run, status="ok", traces=stage_stats, usage=usage)
            return 0
        if not a.traces:
            bd.save_progress(m0, progress)
            bd.finish_run(run, status="ok", usage=usage, machine_after=_fingerprint(m0))
            _write_guide(bd, a)
            print(f"outputs: {out}/context.json, machine_init.json, machine.json, rules.json, build.json"
                  + ("" if a.no_guide else ", GUIDE.md, PROMPT.md"))
            return 0

        # 2. trace update
        if a.task:
            want = set(a.task)

            def keep(item) -> bool:
                p, t, _e = item
                stem = pathlib.Path(str(p)).stem
                tid = str((t.task or {}).get("task_id") or "") if t is not None else ""
                return stem in want or tid in want or any(stem.endswith(w) or tid.endswith(w) for w in want)

            traces = [x for x in traces if keep(x)]
            if not traces:
                print(f"no traces match {sorted(want)}")
        print(f"\nupdating with {len(traces)} traces:")
        if a.decider == "align":
            res = update(m0, traces, ctx, attempts=a.attempts, accepted_only=a.accepted_only, progress=print)
            machine = res.machine
            dump(machine, out / "machine.json")
            (out / "update_log.json").write_text(json.dumps({
                "counts": res.counts(), "entries": res.entries}, ensure_ascii=False, indent=2), encoding="utf-8")
            text = report(m0, machine, res, init_res, ctx, out)
            accepted = iter(res.accepted)
            for (p, _t, _e), entry in zip(traces, res.entries):
                item = key_of.get(str(p))
                acc = next(accepted) if entry.get("status") == "accepted" else None
                if item is None:
                    continue
                progress["entries"].append({**entry, "trace": item.key, "task_id": item.task_id, "run": run["n"],
                                            "decider": "align"})
                if acc is not None:
                    progress["accepted"].append({"trace": item.key, "anchors": list(acc.anchors), "ignore": [],
                                                 "ignore_calls": []})
            bd.save_progress(machine, progress)
        else:
            from hexis import updater as U
            from hexis.step_judge import DecisionLog, ModelDecider
            bd.save_progress(m0, progress)
            items = [key_of[str(p)] for p, t, _e in traces if t is not None and str(p) in key_of]
            decider = ModelDecider(model, model_id=str(endpoint.get("model") or ""),
                                   log=DecisionLog(out / "decisions.jsonl"), use_cache=not a.no_cache, say=print)
            try:
                outcome = U.process(bd, items, machine=m0, ctx=ctx, accepted=[], progress=progress, decider=decider,
                                    attempts=a.attempts, accepted_only=a.accepted_only, run_n=run["n"])
            except ModelUnavailable as exc:
                print(f"model endpoint failed: {exc}")
                print("progress is saved; continue with `hexis-agent update --build " + str(out) + "`")
                bd.finish_run(run, status="interrupted", traces=stage_stats, usage=model.usage(),
                              questions=dict(decider.stats))
                return 3
            machine = outcome.machine
            usage = model.usage()
            run["questions"] = dict(decider.stats)
            (out / "update_log.json").write_text(json.dumps({
                "counts": _counts(progress), "entries": progress["entries"]}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            text = U.write_report(bd, machine, ctx, progress)
        bd.finish_run(run, status="ok", traces=stage_stats, counts=_counts(progress), usage=usage,
                      machine_before=_fingerprint(m0), machine_after=_fingerprint(machine))
        _write_guide(bd, a)
        print()
        print(text)
        print(f"outputs: {out}/context.json, machine_init.json, machine.json, update_log.json, report.md, "
              "progress.json, build.json" + ("" if a.no_guide else ", GUIDE.md, PROMPT.md"))
        return 0


def _counts(progress: dict) -> dict:
    from hexis.builddir import counts
    return counts(progress)


def _fingerprint(m: Machine) -> str:
    from hexis.compiler.stepwise import fingerprint
    return fingerprint(m)


def _write_guide(bd, a) -> None:
    bd.manifest["guide"] = {"embed_skill": bool(a.embed_skill), "retries": 3}
    bd.save_manifest()
    if a.no_guide:
        return
    from hexis.cli.guide import generate
    progress = json.loads((bd.root / "progress.json").read_text(encoding="utf-8")) \
        if (bd.root / "progress.json").is_file() else None
    init_log = json.loads((bd.root / "init_log.json").read_text(encoding="utf-8")) \
        if (bd.root / "init_log.json").is_file() else {}
    generate(machine_path=bd.root / "machine.json", out=bd.root, skill_dir=bd.skill_dir, tools_path=bd.root / "tools.json",
             embed_skill=bool(a.embed_skill), manifest=bd.manifest, progress=progress,
             clause_map=init_log.get("clause_map") or None)


if __name__ == "__main__":
    raise SystemExit(main())
