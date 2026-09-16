"""Compare arms on a shared task set: pass rate, cost, paired tests.

    hexis-agent summarize OUT.md "title" fsm=RUN_DIR skill=RUN_DIR [awm=RUN_DIR rbank=RUN_DIR ...]

Each ``RUN_DIR`` contains a ``results.jsonl``. Rows are selected by their ``arm`` field when the arm
name is one of fsm / skill / awm / rbank; for any other name every row of the file is used. When a
task appears more than once, its last row counts. Only tasks present in every arm are compared, and
each arm is compared with ``fsm`` by an exact binomial test on the discordant pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

DISPLAY = {"fsm": "hexis", "skill": "Skill + ReAct", "awm": "AWM", "rbank": "ReasoningBank"}


def load_arm(name: str, path: str) -> dict:
    rows = [json.loads(line) for line in (pathlib.Path(path) / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [r for r in rows if r.get("arm") == name or name not in DISPLAY]
    return {r["task"]: r for r in rows if r.get("arm") == name} or {r["task"]: r for r in rows}


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output markdown file")
    ap.add_argument("title", help="table title")
    ap.add_argument("arms", nargs="+", metavar="ARM=RUN_DIR", help="arm name and the directory holding its results.jsonl")
    a = ap.parse_args(argv)

    arms = {}
    for spec in a.arms:
        if "=" not in spec:
            ap.error(f"expected ARM=RUN_DIR, got {spec!r}")
        name, path = spec.split("=", 1)
        arms[name] = load_arm(name, path)
    tasks = sorted(set.intersection(*[set(v) for v in arms.values()]))
    if not tasks:
        ap.error("the arms share no tasks")

    def stat(d):
        rs = [d[t] for t in tasks]
        p = sum(bool(r["passed"]) for r in rs)
        dur = [r.get("duration_s") or 0 for r in rs]
        tok = [(r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in rs]
        calls = [r.get("llm_calls") or 0 for r in rs]
        return p, statistics.mean(dur), statistics.median(dur), statistics.mean(tok), statistics.median(tok), statistics.mean(calls)

    lines = [f"# {a.title}", "", f"{len(tasks)} tasks shared by all arms.", "",
             "| Arm | Passed | Pass rate | Time mean / median (s) | Tokens mean / median | Model calls (mean) |",
             "|---|---|---|---|---|---|"]
    for name, d in arms.items():
        p, dm, dmed, tm, tmed, cm = stat(d)
        lines.append(f"| {DISPLAY.get(name, name)} | {p} / {len(tasks)} | {p / len(tasks):.1%} | {dm:.0f} / {dmed:.0f} | "
                     f"{tm:,.0f} / {tmed:,.0f} | {cm:.1f} |")
    lines += ["", "## Paired comparison with fsm (exact binomial test on discordant pairs)", ""]
    if "fsm" in arms:
        for name, d in arms.items():
            if name == "fsm":
                continue
            fo = sum(1 for t in tasks if arms["fsm"][t]["passed"] and not d[t]["passed"])
            so = sum(1 for t in tasks if d[t]["passed"] and not arms["fsm"][t]["passed"])
            both = sum(1 for t in tasks if arms["fsm"][t]["passed"] and d[t]["passed"])
            n, k = fo + so, min(fo, so)
            pv = min(1.0, sum(math.comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)) if n else 1.0
            lines.append(f"- fsm vs {name}: both passed {both}, only fsm {fo}, only {name} {so}, "
                         f"neither {len(tasks) - both - fo - so}; p = {pv:.3f}")
    text = "\n".join(lines) + "\n"
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
