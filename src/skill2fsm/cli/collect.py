"""Collect skill-execution traces with OpenCode on spreadsheet tasks.

One OpenCode session per task, with the skill document in the agent's system prompt and only
OpenCode's native tools enabled (:data:`TOOLS`). Tool names are kept verbatim so that machine states
and trace steps share one alphabet; sub-agents, web access and to-do tools are disabled.

For every task the output directory keeps the raw evidence next to a folded trace::

    <out>/<task_id>/events.jsonl   OpenCode event stream (stdout of ``--format json``)
    <out>/<task_id>/input.xlsx     input workbook
    <out>/<task_id>/output.xlsx    workbook written by the agent
    <out>/<task_id>/prompt.txt     task prompt
    <out>/<task_id>.jsonl          folded trace, the input of ``skill2fsm compile --traces``
    <out>/report.json              grading per task (SpreadsheetBench comparison after LibreOffice recalculation)

Each task runs in its own directory via ``opencode run --dir``: OpenCode ignores the working directory
of the subprocess, and without ``--dir`` all tasks would share one directory and overwrite each other's
workbooks. Runs resume: tasks with an event stream and a graded entry in ``report.json`` are skipped
unless ``--no-resume`` is given, and a change of :data:`PROTOCOL` invalidates earlier reports.

    skill2fsm collect --tasks-file tasks.yaml --skill SKILL_DIR --out traces/ --model MODEL
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import time

#: OpenCode's native tools given to the agent. Deliberately left out: ``task`` (spawns sub-agents,
#: which turns a trace step into a black box), ``webfetch`` (tasks are offline) and ``todowrite`` /
#: ``todoread`` (a scratchpad rather than an action on the task).
TOOLS = ("bash", "read", "write", "edit", "list", "glob", "grep")

AGENT = "xlsx-collect"

#: Collection protocol version; increase it whenever the isolation of runs changes. Resuming only
#: accepts a report with the same version.
PROTOCOL = 2

PROMPT_HEAD = """\
Complete the spreadsheet request and save the result at the exact requested output path.

Your tools are a shell (bash) and ordinary file tools (read, write, edit, list, glob, grep).
python3 with openpyxl is available in the shell. There are no spreadsheet-specific tools — do
the work yourself. Do not stop after merely explaining a formula; the output workbook must exist.

Follow the skill below.

--- SKILL ---
{skill}
--- END SKILL ---
"""


PROMPT_HEAD_LIVEMATH = """\
Answer the multiple-choice mathematics question and write the answer file exactly as requested.

Your tools are a shell (bash) and ordinary file tools (read, write, edit, list, glob, grep).
Do not stop after explaining; answer.txt must exist in the working directory and contain only \\boxed{{X}}.

Follow the skill below.

--- SKILL ---
{skill}
--- END SKILL ---
"""


PROMPT_HEAD_FILETASK = """\
Complete the task and write the answer file exactly as requested.

Your tools are a shell (bash) and ordinary file tools (read, write, edit, list, glob, grep).
python3 with pandas, numpy, scipy and scikit-learn is available in the shell. The task's data
files are in the working directory. Do not stop after explaining; answer.txt must exist in the
working directory and contain only the answer in the required format.

Follow the skill below.

--- SKILL ---
{skill}
--- END SKILL ---
"""


def agent_config(opencode_model: str, upstream_model: str, skill_text: str, head: str = PROMPT_HEAD) -> str:
    """OpenCode configuration: an OpenAI-compatible provider and one agent restricted to :data:`TOOLS`.
    ``head`` is the task instruction placed before the skill, chosen by task type."""
    tools = {"*": False}
    for t in TOOLS:
        tools[t] = True
    return json.dumps({
        "provider": {
            "openai-compatible": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "OpenAI-compatible endpoint",
                "options": {"baseURL": "{env:BASE_URL}", "apiKey": "{env:API_KEY}"},
                "models": {upstream_model: {
                    "name": upstream_model,
                    "limit": {"context": 200000, "output": 16384}}},
            },
        },
        "agent": {
            AGENT: {
                "description": "skill2fsm trace collection: skill + two primitives",
                "mode": "primary",
                "model": opencode_model,
                "prompt": head.format(skill=skill_text.strip()),
                "tools": tools,
            },
        },
    }, ensure_ascii=False)


def fold_events(events_text: str, *, task_id: str, request: str, input_path: str,
                output_path: str, passed: bool, model: str) -> str:
    """Fold an OpenCode event stream into a trace: one header line, then one line per step.

    OpenCode's ``events.jsonl`` is its own UI event stream: one tool call is split into
    ``step-start`` / ``tool`` (pending → completed) / ``step-finish`` events with text in between, and
    the format changes between OpenCode versions. The raw stream is kept as evidence; this function
    produces the action sequence that :func:`skill2fsm.trace_adapter.read_raw_jsonl` reads.

    Tool names are not renamed: ``bash`` / ``read`` / ``write`` … appear as in :data:`TOOLS`, which is
    the alphabet of the machine.
    """
    lines = [json.dumps({
        "task_id": task_id, "arm": "agent", "harness": "opencode", "model": model,
        "verdict": "accepted" if passed else "rejected",
        "input": {"request": request, "input_path": input_path, "output_path": output_path},
    }, ensure_ascii=False)]
    for raw in events_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except ValueError:                      # the last line may be cut off when a run times out
            continue
        part = ev.get("part") or {}
        kind = part.get("type")
        if kind == "tool":
            st = part.get("state") or {}
            if st.get("status") not in ("completed", "error"):
                continue                        # pending / running are intermediate states of the same step
            meta = st.get("metadata") or {}
            err = st.get("status") == "error"
            lines.append(json.dumps({
                "kind": "tool", "name": str(part.get("tool") or ""),
                "args": st.get("input") or {},
                "stdout": "" if err else str(st.get("output") or ""),
                "stderr": str(st.get("error") or st.get("output") or "") if err else "",
                "returncode": 1 if err else int(meta.get("exit") or 0),
            }, ensure_ascii=False))
        elif kind == "text":
            text = str(part.get("text") or "").strip()
            if text:
                lines.append(json.dumps({"kind": "model", "text": text}, ensure_ascii=False))
    lines.append(json.dumps({"kind": "end"}))
    return "\n".join(lines) + "\n"


def task_prompt(task: dict) -> str:
    """Spreadsheet prompt: the benchmark instruction plus local file names (the job directory is the
    working directory)."""
    original = str(task["turns"][0])
    # Spreadsheet task files end each instruction with a note on asset paths that starts with this
    # marker; the note is replaced by the local file names below.
    instruction = original.split(" 工作簿在 assets/", 1)[0].strip()
    return f"{instruction}\n\nThe workbook is input.xlsx. Save the result as output.xlsx."


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--tasks-file", required=True,
                    help="task YAML whose entries carry metadata.init_asset, metadata.golden and metadata.answer_position")
    ap.add_argument("--skill", required=True, help="skill directory containing SKILL.md")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--data-root", default=".",
                    help="directory that asset paths in the task file are relative to (default: current directory)")
    ap.add_argument("--provider", default="default",
                    help="endpoint profile (skill2fsm.env); the model name must belong to that endpoint")
    ap.add_argument("--model", default="", help="upstream model id (default: the profile's model)")
    ap.add_argument("--opencode-bin", default="opencode", help="OpenCode executable")
    ap.add_argument("--only", default="", help="comma-separated task ids (default: all tasks)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (0 = all)")
    ap.add_argument("--timeout", type=float, default=900.0, help="wall-clock limit per task in seconds")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing outputs and rerun every task")
    a = ap.parse_args(argv)

    import yaml
    from skill2fsm.env import llm_config
    from skill2fsm.evaluators.spreadsheet_golden import compare_workbooks

    cfg = llm_config(profile=("" if a.provider in ("default", "") else a.provider))
    upstream = a.model or cfg.model
    print(f"endpoint: {cfg.base_url}  model: {upstream}  tools: {list(TOOLS)}", flush=True)
    opencode_model = f"openai-compatible/{upstream}"
    skill_dir = pathlib.Path(a.skill).resolve()
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise SystemExit(f"no SKILL.md in {skill_dir}")
    skill_md = skill_file.read_text(encoding="utf-8")
    print(f"skill: {skill_dir.name} ({len(skill_md)} chars in the system prompt)", flush=True)
    rows = yaml.safe_load(pathlib.Path(a.tasks_file).read_text(encoding="utf-8"))
    by_id = {str(r["id"]): r for r in rows}
    ids = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else list(by_id)
    if a.limit:
        ids = ids[:a.limit]
    missing = [t for t in ids if t not in by_id]
    if missing:
        raise SystemExit(f"unknown task ids: {missing}")
    root = pathlib.Path(a.data_root).resolve()

    out_root = pathlib.Path(a.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "report.json"
    previous: dict = {}
    if report_path.is_file() and not a.no_resume:
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if (loaded.get("model") == upstream and loaded.get("protocol") == PROTOCOL
                    and loaded.get("skill") == skill_dir.name):
                previous = loaded.get("tasks") or {}
            elif loaded.get("protocol") != PROTOCOL:
                print(f"the existing report uses collection protocol v{loaded.get('protocol', 1)} and this run "
                      f"uses v{PROTOCOL}: earlier results cannot be resumed, all tasks are rerun.", flush=True)
        except (OSError, ValueError):
            previous = {}

    env = dict(os.environ)
    env["OPENCODE_CONFIG_CONTENT"] = agent_config(opencode_model, upstream, skill_md)
    # Overwrite rather than setdefault: BASE_URL / API_KEY from the environment may belong to another
    # endpoint, which would receive a model name it does not serve.
    env["BASE_URL"] = cfg.base_url
    env["API_KEY"] = cfg.api_key

    report = {"protocol": PROTOCOL,
              "model": upstream, "opencode_model": opencode_model, "agent": AGENT,
              "tool_policy": f"OpenCode native tools {list(TOOLS)}",
              "skill": skill_dir.name, "tasks": {}}
    passed = 0
    for i, tid in enumerate(ids, 1):
        task = by_id[tid]
        case = out_root / tid
        case.mkdir(parents=True, exist_ok=True)
        if tid in previous and (case / "events.jsonl").is_file():
            report["tasks"][tid] = previous[tid]
            passed += bool(previous[tid].get("passed"))
            print(f"[{i}/{len(ids)}] {tid} {'PASS' if previous[tid].get('passed') else 'FAIL'} (resumed)", flush=True)
            continue
        meta = task["metadata"]
        init = (root / str(meta["init_asset"])).resolve()
        if not init.is_file():
            init = (skill_dir / str(meta["init_asset"])).resolve()
        golden = (root / str(meta["golden"])).resolve()
        shutil.copy2(init, case / "input.xlsx")
        (case / "output.xlsx").unlink(missing_ok=True)
        prompt = task_prompt(task)
        (case / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        print(f"[{i}/{len(ids)}] {tid} running...", flush=True)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [a.opencode_bin, "run", "--format", "json", "--model", opencode_model,
                 "--dir", str(case), "--agent", AGENT, "--title", f"collect-{tid}", prompt],
                cwd=case, env=env, capture_output=True, text=True, timeout=a.timeout)
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc, stdout, stderr = -1, (exc.stdout or ""), "timed out"
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        dur = round(time.monotonic() - t0, 2)
        (case / "events.jsonl").write_text(stdout, encoding="utf-8")
        if stderr.strip():
            (case / "stderr.txt").write_text(stderr[:20000], encoding="utf-8")

        produced = case / "output.xlsx"
        if produced.is_file():
            ok, why = compare_workbooks(golden, produced, str(meta["answer_position"]))
        else:
            ok, why = False, "no output workbook"
        passed += bool(ok)
        # The verdict is known only after grading, so the trace is folded here.
        (out_root / f"{tid}.jsonl").write_text(
            fold_events(stdout, task_id=tid, request=prompt,
                        input_path=str(case / "input.xlsx"),
                        output_path=str(produced), passed=bool(ok), model=upstream),
            encoding="utf-8")
        report["tasks"][tid] = {"passed": bool(ok), "why": why, "returncode": rc,
                                "duration_s": dur, "answer_position": str(meta["answer_position"])}
        print(f"    {'PASS' if ok else 'FAIL'}  {why[:100]}  ({dur}s)", flush=True)
        report["passed"] = passed
        report["n"] = len(report["tasks"])
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    report["passed"] = passed
    report["n"] = len(report["tasks"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\npassed {passed}/{len(report['tasks'])} -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
