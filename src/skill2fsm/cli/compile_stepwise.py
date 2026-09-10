"""Update a machine trace by trace with externally supplied step decisions.

A judge (a person or a coding agent) decides each trace step; this command computes the candidates,
applies the decisions, validates and records progress. Progress lives in ``<out>/progress.json`` and
``<out>/machine.json``, so work can resume at any time.

    # 1. show the next batch of undecided traces: per-step event summaries, candidates
    #    (tier 1: reachable in one step; tier 2: needs a new transition) and default proposals
    skill2fsm compile-stepwise --skill SKILL_DIR --traces TRACE_DIR --out OUT --machine-init M0.json --show 8

    # 2. write the decisions as JSON (keys are the step numbers printed by --show) and apply them:
    #    {"trace_id": {"accept_proposals": true,
    #                  "steps": {"1": {"d": "ignore"},
    #                            "4": {"d": "match", "state": "s7"},
    #                            "6": {"d": "new", "purpose": "...", "clause": "S2.3.3"}}}}
    #    applying stops after a trace that inserted new states: later traces need fresh candidates
    skill2fsm compile-stepwise ... --apply decisions.json

    # 3. write the report once every trace is processed
    skill2fsm compile-stepwise ... --report
"""
from __future__ import annotations

import argparse
import json
import pathlib

from skill2fsm import stepwise_view as SV
from skill2fsm.backends import REGISTRIES, registry
from skill2fsm.cli.compile import describe
from skill2fsm.compile_agent import markdown_clauses
from skill2fsm.fsm import check as _check
from skill2fsm.fsm import stepwise as SW
from skill2fsm.fsm.context import build_context, load_rules
from skill2fsm.fsm.init import normalize
from skill2fsm.fsm.traces import load_traces
from skill2fsm.schema import load_machine
from skill2fsm.skill_loader import load_agent_skill
from skill2fsm.toolspec import load_registry

RULES_FILE = "compile.json"


def setup(a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skill = load_agent_skill(pathlib.Path(a.skill))
    clauses = markdown_clauses(skill.body)
    loaded = load_traces(pathlib.Path(a.traces))
    traces: dict = {}
    order: list = []
    for p, t, e in loaded:
        name = pathlib.Path(str(p)).stem
        if t is None:
            print(f"unreadable trace {name}: {e}")
            continue
        SW.strip_tools(t, a.ignore_tool or [])
        traces[name] = (p, t)
        order.append(name)
    if a.tools:
        reg = load_registry(a.tools)
    elif a.backend and a.backend != "none":
        reg = registry(a.backend)
    else:
        reg = {}
    rules_path = pathlib.Path(a.rules) if a.rules else pathlib.Path(a.skill) / RULES_FILE
    rules = load_rules(rules_path) if rules_path.is_file() else None
    ctx = build_context(skill.slug, skill.body, clauses, [t for _p, t in traces.values()], registry=reg, rules=rules)
    m0 = load_machine(pathlib.Path(a.machine_init))
    normalize(m0, ctx)
    errs = _check.check(m0, ctx)
    if errs:
        raise SystemExit(f"initial machine fails the checks: {errs[:5]}")
    prog = SW.Progress.load(out, m0, ctx, {k: v[1] for k, v in traces.items()})
    return out, ctx, m0, traces, order, prog


def pending(order, prog, traces, ctx):
    """Unprocessed traces in order; traces that need no decision (unsupported / excluded / violation /
    skipped) are recorded directly."""
    done = prog.done()
    todo = []
    for name in order:
        if name in done:
            continue
        p, t = traces[name]
        prep = SW.prepare_steps(t, ctx, source=str(p))
        pre = SW.pre_status(prep, t, prog.machine)
        if pre is not None:
            prog.entries.append({"trace": prep.trace_id, "verdict": prep.verdict, "tau": prep.tau,
                                 "events": [e.describe() for e in prep.observable], "status": pre[0], "why": pre[1]})
            print(f"{prep.trace_id:<12} {pre[0]:<11} {pre[1][:100]}")
            continue
        todo.append((name, p, t, prep))
    return todo


def cmd_show(a, out, ctx, traces, order, prog):
    todo = pending(order, prog, traces, ctx)
    prog.save(out)
    batch = todo[:a.show]
    shown = []
    for name, p, t, prep in batch:
        s = SV.show_trace(prog.machine, ctx, prep)
        shown.append(s)
        print(SV.render_show(prog.machine, s))
        print()
    (out / "pending.json").write_text(json.dumps(shown, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"current machine fp={SW.fingerprint(prog.machine)}: {prog.machine.n_states()} states, "
          f"{len(prog.machine.transitions_all())} transitions; processed {len(prog.entries)}, "
          f"pending {len(todo)}, this batch {len(batch)}")
    print("\ncurrent machine:")
    print("\n".join(describe(prog.machine)))


def cmd_apply(a, out, ctx, traces, order, prog):
    spec = json.loads(pathlib.Path(a.apply).read_text(encoding="utf-8"))
    if a.redo:
        redo = {n for n in spec if prog.done().get(n) not in (None, "accepted")}
        prog.entries = [e for e in prog.entries if e["trace"] not in redo]
        prog.agent_log = [d for d in prog.agent_log if d.get("trace") not in redo]
        if redo:
            print(f"re-deciding (earlier non-accepted records dropped): {sorted(redo)}")
    done = prog.done()
    names = [n for n in order if n in spec and n not in done]
    skipped = [n for n in spec if n in done]
    if skipped:
        print(f"already processed, skipped: {skipped}")
    for name in names:
        p, t = traces[name]
        m2, entry, acc = SW.update_with_decisions(prog.machine, t, ctx, source=str(p), spec=spec[name],
                                                  accepted=prog.accepted, attempts=a.attempts)
        prog.entries.append(entry)
        for att in entry.get("attempts", []):
            for d in att.get("decisions", []):
                prog.agent_log.append({"trace": entry["trace"], "attempt": att["attempt"],
                                       **{k: v for k, v in d.items() if k != "tier1" and k != "tier2"},
                                       "tier1": d.get("tier1"), "tier2": d.get("tier2")})
        inserted = entry.get("inserted") or []
        if acc is not None:
            prog.machine = m2
            prog.accepted.append(acc)
        prog.save(out)
        why = entry.get("why") or ("; ".join(entry.get("changes") or [])[:160] if entry["status"] == "accepted" else "")
        print(f"{entry['trace']:<12} {entry['status']:<9} cost={entry.get('cost', '')!s:<3} {str(why)[:150]}")
        if inserted and not a.continue_after_insert:
            rest = [n for n in names if order.index(n) > order.index(name)]
            print(f"\n{entry['trace']} inserted new states {inserted}; the machine changed "
                  f"(fp={SW.fingerprint(prog.machine)}). Stopping here: run --show again before deciding "
                  f"the remaining {len(rest)} traces: {rest[:12]}")
            break
    print(f"\nprogress: {prog.counts()}; machine {prog.machine.n_states()} states, "
          f"{len(prog.machine.transitions_all())} transitions")


def cmd_report(a, out, ctx, m0, prog):
    m = prog.machine
    c = prog.counts()
    lines = ["# Stepwise compilation report", "",
             f"Skill {ctx.skill_id}: {ctx.n_traces} traces, task inputs {list(ctx.task_inputs)}, tools {sorted(ctx.tools)}; "
             f"{len(ctx.label_rules)} label rules, {len(ctx.requirements)} requirements, "
             f"{len(ctx.terminal_conditions)} terminal conditions.",
             f"Initial machine: {m0.n_states()} states, {len(m0.transitions_all())} transitions.",
             f"Final machine: {m.n_states()} states ({len(m.states)} including terminals), "
             f"{len(m.transitions_all())} transitions, {len(m.variables)} variables.",
             "", "| Outcome | Traces |", "|---|---:|"]
    for k in ("accepted", "rejected", "excluded", "violation", "unsupported", "skipped"):
        if c.get(k):
            lines.append(f"| {k} | {c[k]} |")
    n_dec = len(prog.agent_log)
    n_new = sum(1 for d in prog.agent_log if d.get("decision") == "new")
    n_t2 = sum(1 for d in prog.agent_log if d.get("decision") == "match" and d.get("tier") == 2)
    n_prop = sum(1 for d in prog.agent_log if d.get("source") == "proposal")
    n_ign = sum(len(e.get("ignored") or []) for e in prog.entries)
    lines += ["", f"Judge decisions on {n_dec} steps: {n_new} new, {n_t2} tier-2 matches, "
                  f"{n_prop} accepted proposals; {n_ign} harness steps ignored.", "",
              "## Traces", "", "| Trace | Verdict | Outcome | Cost | Ignored | Reason / path |", "|---|---|---|---:|---|---|"]
    for e in prog.entries:
        why = e.get("why") or ""
        if e.get("status") == "accepted":
            why = " → ".join(e.get("anchors", []))
        lines.append(f"| {e['trace']} | {e.get('verdict', '')} | {e['status']} | {e.get('cost', '')} | "
                     f"{','.join(map(str, e.get('ignored') or [])) or '-'} | {str(why)[:150]} |")
    lines += ["", "## Final machine", "", "```"] + describe(m) + ["```", ""]
    errs = _check.check(m, ctx)
    lines.append("Final checks: " + ("all passed" if not errs else "; ".join(errs)))
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "update_log.json").write_text(json.dumps({"counts": c, "entries": prog.entries}, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    (out / "agent_log.json").write_text(json.dumps(prog.agent_log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--skill", required=True, help="skill directory containing SKILL.md")
    ap.add_argument("--traces", required=True, help="directory of traces (*.jsonl)")
    ap.add_argument("--out", required=True, help="output directory (progress is kept here)")
    ap.add_argument("--machine-init", required=True, help="initial machine")
    ap.add_argument("--backend", default="opencode", help=f"tool registry name ({', '.join(sorted(REGISTRIES))})")
    ap.add_argument("--tools", default=None, help="tool registry JSON file (overrides --backend)")
    ap.add_argument("--rules", default=None, help=f"skill rules JSON (default: <skill>/{RULES_FILE})")
    ap.add_argument("--ignore-tool", action="append", default=["skill2fsm_begin"],
                    help="harness marker tool to drop from traces; repeatable")
    ap.add_argument("--attempts", type=int, default=2, help="candidate attempts per trace")
    ap.add_argument("--show", type=int, default=0, help="print the next N undecided traces")
    ap.add_argument("--apply", default=None, help="decisions JSON file")
    ap.add_argument("--continue-after-insert", action="store_true",
                    help="keep applying later decisions after a trace inserted new states")
    ap.add_argument("--redo", action="store_true",
                    help="re-decide traces in the decisions file that were processed but not accepted")
    ap.add_argument("--report", action="store_true", help="write report.md, update_log.json and agent_log.json")
    a = ap.parse_args(argv)
    out, ctx, m0, traces, order, prog = setup(a)
    if a.apply:
        cmd_apply(a, out, ctx, traces, order, prog)
    if a.show:
        cmd_show(a, out, ctx, traces, order, prog)
    if a.report:
        cmd_report(a, out, ctx, m0, prog)
    if not (a.apply or a.show or a.report):
        print(f"progress: {prog.counts()}; pending {len([n for n in order if n not in prog.done()])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
