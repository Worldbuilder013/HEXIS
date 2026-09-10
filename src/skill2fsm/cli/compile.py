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

Outputs in ``--out``: ``context.json``, ``machine_init.json``, ``init_log.json``, ``machine.json``,
``update_log.json`` and ``report.md``.

    skill2fsm compile --skill SKILL_DIR --traces TRACE_DIR --out OUT
    skill2fsm compile --skill SKILL_DIR --traces TRACE_DIR --out OUT --machine-init M0.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

from skill2fsm.backends import REGISTRIES, registry
from skill2fsm.compile_agent import markdown_clauses
from skill2fsm.fsm import check as _check
from skill2fsm.fsm.context import build_context, load_rules
from skill2fsm.fsm.init import extract_rules, initialize, install_rules, normalize
from skill2fsm.fsm.traces import load_traces
from skill2fsm.fsm.update import update
from skill2fsm.schema import Machine, load_machine
from skill2fsm.skill_loader import load_agent_skill
from skill2fsm.toolspec import load_registry

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
    ap.add_argument("--traces", required=True, help="directory of traces (*.jsonl)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--backend", default="opencode",
                    help=f"tool registry name ({', '.join(sorted(REGISTRIES))}); 'none' infers tools from the traces only")
    ap.add_argument("--tools", default=None, help="tool registry JSON file (overrides --backend)")
    ap.add_argument("--rules", default=None,
                    help=f"skill rules JSON (default: <skill>/{RULES_FILE}; without one the model extracts rules)")
    ap.add_argument("--machine-init", default=None, help="existing initial machine; skips the model call")
    ap.add_argument("--provider", default="default",
                    help="endpoint profile: 'default' reads MODEL / BASE_URL / API_KEY; minimax / deepseek read prefixed keys")
    ap.add_argument("--max-tokens", type=int, default=16384, help="maximum tokens per model response")
    ap.add_argument("--rounds", type=int, default=3, help="maximum initialization rounds")
    ap.add_argument("--attempts", type=int, default=2, help="candidate attempts per trace")
    ap.add_argument("--task", action="append", default=None,
                    help="only use traces of this task (header task_id or file name); repeatable")
    ap.add_argument("--accepted-only", action="store_true", help="only use traces whose verdict is accepted")
    ap.add_argument("--lenient", action="store_true", help="continue with the update even if the initial machine fails the checks")
    ap.add_argument("--init-only", action="store_true", help="stop after initialization")
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skill = load_agent_skill(pathlib.Path(a.skill))
    doc = skill.body
    clauses = markdown_clauses(doc)

    # 0. compile context
    traces = load_traces(pathlib.Path(a.traces))
    for p, t, e in traces:
        if t is None:
            print(f"unreadable trace {pathlib.Path(str(p)).name}: {e}")
    if a.tools:
        reg = load_registry(a.tools)
    elif a.backend and a.backend != "none":
        reg = registry(a.backend)
    else:
        reg = {}
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

    def save_context() -> None:
        (out / "context.json").write_text(json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    # 1. initialization
    init_res = None
    if a.machine_init:
        if rules is None:
            print("no rules file and initialization skipped: using a single default terminal and no requirements")
        m0 = load_machine(pathlib.Path(a.machine_init))
        normalize(m0, ctx)
        errs = _check.check(m0, ctx)
        print(f"initial machine from {a.machine_init}: {m0.n_states()} states; checks "
              + ("passed" if not errs else f"failed {errs[:3]}"))
        if errs and not a.lenient:
            save_context()
            return 2
    else:
        from skill2fsm.llm_client import ModelAdapter, client_from_env
        prov = "" if a.provider in ("default", "") else a.provider
        rule_notes: list = []
        with client_from_env(profile=prov) as cl:
            print(f"endpoint: {cl.model} @ {cl.base_url}")
            model = ModelAdapter(cl, temperature=0.0, max_tokens=a.max_tokens)
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
            save_context()
            return 2
        m0 = init_res.machine
        print(f"initial machine M0: {m0.n_states()} states, {len(m0.transitions_all())} transitions"
              + (f"; checks failed, continuing (--lenient): {init_res.check_errors[:3]}" if init_res.check_errors else ""))
    save_context()
    dump(m0, out / "machine_init.json")
    print("\n".join(describe(m0)))
    if a.init_only:
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
    res = update(m0, traces, ctx, attempts=a.attempts, accepted_only=a.accepted_only, progress=print)
    dump(res.machine, out / "machine.json")
    (out / "update_log.json").write_text(json.dumps({
        "counts": res.counts(), "entries": res.entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    text = report(m0, res.machine, res, init_res, ctx, out)
    print()
    print(text)
    print(f"outputs: {out}/context.json, machine_init.json, machine.json, update_log.json, report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
