"""Fold the OpenCode event streams of a benchmark run into traces.

    hexis-agent fold-traces --bench RUN_DIR --tasks-file TASKS.yaml --out TRACE_DIR

Every skill-arm job ``<RUN_DIR>/skill/<task>/r1/events.jsonl`` becomes ``<TRACE_DIR>/<task>.jsonl``;
verdicts come from ``<RUN_DIR>/results.jsonl``. Jobs that only replied with text still produce a trace
without tool steps; compilation excludes such traces because they contain no modifying step.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from hexis.cli.collect import fold_events


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--bench", required=True, help="benchmark run directory (output of hexis-agent bench with a skill arm)")
    ap.add_argument("--tasks-file", required=True, help="task YAML used for that run")
    ap.add_argument("--out", required=True, help="output trace directory")
    a = ap.parse_args(argv)

    import yaml
    pool = {str(r["id"]): r for r in yaml.safe_load(pathlib.Path(a.tasks_file).read_text(encoding="utf-8"))}
    bench = pathlib.Path(a.bench)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for line in (bench / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["arm"] == "skill":
                results[r["task"]] = r
    n = n_tool = 0
    for job in sorted(bench.glob("skill/*/r1/events.jsonl")):
        tid = job.parents[1].name
        r = results.get(tid) or {}
        text = fold_events(job.read_text(encoding="utf-8"), task_id=tid, request=str(pool[tid]["turns"][0]),
                           input_path="", output_path=str(job.parent / "answer.txt"),
                           passed=bool(r.get("passed")), model=str(r.get("model") or ""))
        (out / f"{tid}.jsonl").write_text(text, encoding="utf-8")
        n += 1
        n_tool += any(json.loads(line).get("kind") == "tool" for line in text.splitlines()[1:])
    print(f"folded {n} traces into {out}; {n_tool} contain tool calls, {n - n_tool} are text only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
