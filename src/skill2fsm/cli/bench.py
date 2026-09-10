"""Run machine and skill-execution arms on a task set in parallel.

All arms share one model endpoint and OpenCode's native tools:

* ``fsm``: the compiled machine (``skill2fsm run``; the fallback state retries ``--retries`` times
  before interpreted execution);
* ``skill``: OpenCode with the skill document in the agent's system prompt (the agent configuration
  of ``skill2fsm collect``);
* ``awm`` / ``rbank``: the skill arm with Agent Workflow Memory workflows (``--awm-file``) or
  ReasoningBank retrieval results (``--rbank``) placed before the task prompt.

Every (arm, task, repetition) is a job with its own working directory, run as a subprocess in a thread
pool. Spreadsheet tasks are graded with the SpreadsheetBench comparison (LibreOffice recalculation,
then the ``answer_position`` range), one grading at a time. Runs resume: jobs already present in
``results.jsonl`` are skipped.

    skill2fsm bench --mode livemath --tasks-file livemath.yaml --tasks "$(cat test_tasks.txt)" \\
        --arms fsm,skill --reps 1 --machine machine.json --skill skills/livemath --out runs/livemath

Outputs::

    <out>/results.jsonl        one line per job: arm, task, repetition, verdict, reason, time, tokens, model calls, path
    <out>/summary.json / .md   per-arm summary and per-task table
    <out>/fsm/<task>/r<k>/     machine jobs: trace.jsonl, result.json, produced file, stdout.txt
    <out>/<arm>/<task>/r<k>/   OpenCode jobs: events.jsonl, prompt.txt, stderr.txt, input and output files
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from skill2fsm.cli.collect import (AGENT, PROMPT_HEAD, PROMPT_HEAD_FILETASK, PROMPT_HEAD_LIVEMATH,
                                   agent_config, task_prompt)
from skill2fsm.cli.run import DEFAULT_BENCH_DIR

MODE = {"name": "xlsx", "tasks_file": None}          # set from --mode / --tasks-file
PY = sys.executable
_print_lock = threading.Lock()
_grade_lock = threading.Lock()
_results_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------- #
# task pool
# --------------------------------------------------------------------------- #
def load_pool() -> dict:
    import yaml
    rows = yaml.safe_load(pathlib.Path(MODE["tasks_file"]).read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in rows}


def pick_tasks(pool: dict, n: int, seed: int, explicit: str) -> list[str]:
    if explicit:
        ids = [x.strip() for x in explicit.split(",") if x.strip()]
        missing = [t for t in ids if t not in pool]
        if missing:
            raise SystemExit(f"unknown task ids: {missing}")
        return ids
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(pool), n))


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
def opencode_tokens(events_text: str) -> dict:
    """Sum the tokens of every step-finish event: prompt = input + cache.read, completion = output + reasoning."""
    prompt = completion = calls = 0
    for raw in events_text.splitlines():
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        part = ev.get("part") or {}
        tok = part.get("tokens")
        if part.get("type") == "step-finish" and isinstance(tok, dict):
            calls += 1
            cache = tok.get("cache") or {}
            prompt += int(tok.get("input") or 0) + int(cache.get("read") or 0)
            completion += int(tok.get("output") or 0) + int(tok.get("reasoning") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "llm_calls": calls}


def _reply_text(stdout: str) -> str:
    texts = []
    for line in stdout.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        part = e.get("part") or e
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(str(part.get("text") or ""))
    return "\n".join(texts)


def run_skill_job(a, task: dict, tid: str, rep: int, job_dir: pathlib.Path, env: dict, grader,
                  extra_context: str = "") -> dict:
    meta = task["metadata"]
    livemath = MODE["name"] == "livemath"
    filetask = MODE["name"] == "filetask"
    root = pathlib.Path(a.data_root)
    job_dir.mkdir(parents=True, exist_ok=True)
    if livemath or filetask:
        (job_dir / "answer.txt").unlink(missing_ok=True)
        prompt = str(task["turns"][0])
        if filetask:
            from skill2fsm.evaluators.filetask import stage_assets
            stage_assets(meta, job_dir, root=root)
    else:
        init = (root / str(meta["init_asset"])).resolve()
        if not init.is_file():
            init = (pathlib.Path(a.skill) / str(meta["init_asset"])).resolve()
        golden = (root / str(meta["golden"])).resolve()
        shutil.copy2(init, job_dir / "input.xlsx")
        (job_dir / "output.xlsx").unlink(missing_ok=True)
        prompt = task_prompt(task)
    if extra_context:
        prompt = extra_context.rstrip() + "\n\n" + prompt
    (job_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    # A private HOME / XDG tree per job: parallel `opencode run` processes sharing one SQLite database
    # fail with "database is locked". The configuration comes from OPENCODE_CONFIG_CONTENT only.
    private = job_dir / ".oc_home"
    for sub in ("config", "data", "cache"):
        (private / sub).mkdir(parents=True, exist_ok=True)
    job_env = dict(env)
    job_env.update({"HOME": str(private), "XDG_CONFIG_HOME": str(private / "config"),
                    "XDG_DATA_HOME": str(private / "data"), "XDG_CACHE_HOME": str(private / "cache"),
                    "OPENCODE_DISABLE_PROJECT_CONFIG": "true", "OPENCODE_DISABLE_AUTOUPDATE": "true",
                    "OPENCODE_DISABLE_LSP_DOWNLOAD": "true"})
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [a.opencode_bin, "run", "--format", "json", "--model", f"openai-compatible/{a.model}",
             "--dir", str(job_dir), "--agent", AGENT, "--title", f"bench-{tid}-r{rep}", prompt],
            cwd=job_dir, env=job_env, capture_output=True, text=True, timeout=a.timeout)
        rc, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, stdout, stderr = -1, str(exc.stdout or ""), "timed out"
    dur = round(time.monotonic() - t0, 2)
    (job_dir / "events.jsonl").write_text(stdout, encoding="utf-8")
    shutil.rmtree(private, ignore_errors=True)      # tens of MB per job; many jobs would fill the disk
    if stderr.strip():
        (job_dir / "stderr.txt").write_text(stderr[:20000], encoding="utf-8")
    if filetask:
        from skill2fsm.evaluators.filetask import grade as grade_filetask
        reply = _reply_text(stdout)
        ok, why, source = grade_filetask(meta, job_dir / "answer.txt", reply)
        if source == "reply_text":
            (job_dir / "reply_text.txt").write_text(reply, encoding="utf-8")
        return {"passed": ok, "why": why, "returncode": rc, "duration_s": dur, "answer_source": source,
                **opencode_tokens(stdout)}
    if livemath:
        ok, why = grader(job_dir / "answer.txt", str(meta["answer"]))
        source = "answer.txt"
        if not (job_dir / "answer.txt").is_file():
            # skill arms often answer \boxed{X} in the reply without writing the file: grade the last boxed
            # answer of the reply instead and record the source
            from skill2fsm.evaluators.mcq_boxed import extract_choice
            got = extract_choice(_reply_text(stdout))
            if got is not None:
                source = "reply_text"
                expected = str(meta["answer"]).upper()
                ok, why = (got == expected), ("" if got == expected else f"reference {meta['answer']}, reply text {got}")
        return {"passed": bool(ok), "why": why, "returncode": rc, "duration_s": dur, "answer_source": source,
                **opencode_tokens(stdout)}
    produced = job_dir / "output.xlsx"
    with _grade_lock:
        ok, why = grader(golden, produced, str(meta["answer_position"])) if produced.is_file() \
            else (False, "no output workbook")
    return {"passed": bool(ok), "why": why, "returncode": rc, "duration_s": dur, **opencode_tokens(stdout)}


def run_fsm_job(a, task: dict, tid: str, rep: int, job_dir: pathlib.Path) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    result = job_dir / "result.json"
    result.unlink(missing_ok=True)
    keep_name = "output.xlsx" if MODE["name"] == "xlsx" else "answer.txt"
    cmd = [PY, "-m", "skill2fsm", "run", "--machine", a.machine, "--skill", a.skill,
           "--mode", MODE["name"], "--task", tid, "--quiet", "--provider", a.provider, "--model", a.model,
           "--json", str(job_dir / "trace.jsonl"), "--keep", str(job_dir / keep_name),
           "--result-json", str(result), "--opencode-bin", a.opencode_bin,
           "--tool-timeout", str(a.tool_timeout), "--data-root", a.data_root, "--bench-dir", a.bench_dir]
    if MODE["name"] != "xlsx":
        cmd += ["--tasks-file", str(MODE["tasks_file"])]
    cmd += ["--retries", str(a.retries), "--executor", a.executor]
    if a.tools:
        cmd += ["--tools", a.tools]
    if a.no_interpret:
        cmd.append("--no-interpret")
    if a.no_think_judge:
        cmd.append("--no-think-judge")
    if a.no_think:
        cmd.append("--no-think")
    if a.think_budget:
        cmd += ["--think-budget", str(a.think_budget)]
    if a.judge_think_budget:
        cmd += ["--judge-think-budget", str(a.judge_think_budget)]
    cmd += ["--llm-timeout", str(a.llm_timeout), "--llm-retries", str(a.llm_retries)]
    cmd += ["--max-tokens", str(a.max_tokens)]
    if a.stream:
        cmd.append("--stream")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        rc, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, stdout, stderr = -1, str(exc.stdout or ""), "timed out"
    dur = round(time.monotonic() - t0, 2)
    (job_dir / "stdout.txt").write_text(stdout + ("\n--- stderr ---\n" + stderr if stderr.strip() else ""),
                                        encoding="utf-8")
    out: dict = {"passed": False, "why": "no result.json (the run subprocess failed; see stdout.txt)",
                 "returncode": rc, "duration_s": dur, "prompt_tokens": 0, "completion_tokens": 0,
                 "total_tokens": 0, "llm_calls": 0}
    if result.is_file():
        r = json.loads(result.read_text(encoding="utf-8"))
        out.update({k: r.get(k) for k in ("passed", "why", "stopped", "error", "path", "llm_calls",
                                          "prompt_tokens", "completion_tokens", "unmeasured_calls",
                                          "fallback_entry", "fallback_steps", "tool_calls",
                                          "machine_steps", "machine_prompt_tokens", "machine_completion_tokens",
                                          "fallback_prompt_tokens", "fallback_completion_tokens", "retries",
                                          "realized_calls")})
        out["passed"] = bool(r.get("passed"))
        out["duration_s"] = dur
        out["total_tokens"] = int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
    return out


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def summarize(rows: list[dict], tasks: list[str], arms: list[str], reps: int) -> tuple[dict, str]:
    by_arm: dict = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        durs = [r["duration_s"] for r in rs]
        toks = [r.get("total_tokens") or 0 for r in rs]
        by_arm[arm] = {
            "runs": len(rs), "passed": sum(bool(r["passed"]) for r in rs),
            "pass_rate": round(sum(bool(r["passed"]) for r in rs) / len(rs), 3) if rs else None,
            "duration_mean_s": round(statistics.mean(durs), 1) if durs else None,
            "duration_median_s": round(statistics.median(durs), 1) if durs else None,
            "prompt_tokens_mean": round(statistics.mean(r.get("prompt_tokens") or 0 for r in rs)) if rs else None,
            "completion_tokens_mean": round(statistics.mean(r.get("completion_tokens") or 0 for r in rs)) if rs else None,
            "total_tokens_mean": round(statistics.mean(toks)) if toks else None,
            "llm_calls_mean": round(statistics.mean(r.get("llm_calls") or 0 for r in rs), 1) if rs else None,
        }
        if arm == "fsm" and rs:
            fb = [r for r in rs if (r.get("fallback_steps") or 0) > 0]
            by_arm[arm].update({
                "runs_entered_fallback": len(fb),
                "fallback_steps_mean": round(statistics.mean(r.get("fallback_steps") or 0 for r in rs), 1),
                "fallback_tokens_mean": round(statistics.mean((r.get("fallback_prompt_tokens") or 0)
                                                              + (r.get("fallback_completion_tokens") or 0) for r in rs)),
                "machine_tokens_mean": round(statistics.mean((r.get("machine_prompt_tokens") or 0)
                                                             + (r.get("machine_completion_tokens") or 0) for r in rs)),
                "passed_without_fallback": sum(1 for r in rs if r["passed"] and not (r.get("fallback_steps") or 0)),
                "retries_mean": round(statistics.mean(r.get("retries") or 0 for r in rs), 1),
            })
    per_task = {}
    for tid in tasks:
        per_task[tid] = {arm: {"passed": sum(bool(r["passed"]) for r in rows if r["arm"] == arm and r["task"] == tid),
                               "runs": sum(1 for r in rows if r["arm"] == arm and r["task"] == tid)}
                         for arm in arms}
    lines = ["# Machine vs skill execution", "", f"{len(tasks)} tasks × {reps} repetitions.", "",
             "| Arm | Passed / runs | Pass rate | Mean time (s) | Median time (s) | Mean prompt tokens | "
             "Mean completion tokens | Mean total tokens | Mean model calls |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        s = by_arm[arm]
        lines.append(f"| {arm} | {s['passed']} / {s['runs']} | {s['pass_rate']} | {s['duration_mean_s']} | "
                     f"{s['duration_median_s']} | {s['prompt_tokens_mean']} | {s['completion_tokens_mean']} | "
                     f"{s['total_tokens_mean']} | {s['llm_calls_mean']} |")
    lines += ["", "Token accounting: OpenCode arms sum the tokens of every step (prompt = input + cache.read, "
                  "completion = output + reasoning); the fsm arm sums the usage reported by the endpoint."]
    if "fsm" in by_arm and by_arm["fsm"].get("runs_entered_fallback") is not None:
        s = by_arm["fsm"]
        lines += ["", f"fsm fallback: {s['retries_mean']} retries per run on average; {s['runs_entered_fallback']} / "
                      f"{s['runs']} runs entered interpreted execution; {s['fallback_steps_mean']} interpreted steps on "
                      f"average; mean tokens {s['machine_tokens_mean']} in the machine and {s['fallback_tokens_mean']} "
                      f"in interpreted execution; {s['passed_without_fallback']} runs passed without interpreted execution."]
    lines += ["", "| Task | " + " | ".join(f"{arm} passed" for arm in arms) + " |",
              "|---|" + "---:|" * len(arms)]
    for tid in tasks:
        lines.append(f"| {tid} | " + " | ".join(f"{per_task[tid][arm]['passed']} / {per_task[tid][arm]['runs']}"
                                              for arm in arms) + " |")
    fails = [r for r in rows if not r["passed"]]
    if fails:
        lines += ["", "## Failures", ""]
        for r in fails:
            lines.append(f"- {r['arm']} {r['task']} r{r['rep']}: {str(r.get('why') or r.get('error') or '')[:140]}")
    return {"by_arm": by_arm, "per_task": per_task, "n_tasks": len(tasks), "reps": reps}, "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("xlsx", "livemath", "filetask"), default="xlsx",
                    help="task type: xlsx (SpreadsheetBench workbooks), livemath (\\boxed{X} in answer.txt), "
                         "filetask (assets copied in, answer.txt graded by metadata.verifier)")
    ap.add_argument("--tasks-file", required=True, help="task YAML")
    ap.add_argument("--skill", required=True, help="skill directory containing SKILL.md")
    ap.add_argument("--machine", default=None, help="machine JSON (required by the fsm arm)")
    ap.add_argument("--arms", default="fsm,skill", help="comma-separated arms: fsm, skill, awm, rbank")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=10, help="number of tasks to sample")
    ap.add_argument("--reps", type=int, default=3, help="repetitions per task")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--tasks", default="", help="explicit comma-separated task ids (disables sampling)")
    ap.add_argument("--workers", type=int, default=4, help="parallel jobs")
    ap.add_argument("--data-root", default=".", help="directory that asset paths in the task file are relative to")
    ap.add_argument("--bench-dir", default=DEFAULT_BENCH_DIR,
                    help="SpreadsheetBench verified-400 directory (fsm arm in xlsx mode)")
    ap.add_argument("--provider", default="default",
                    help="endpoint profile: 'default' reads MODEL / BASE_URL / API_KEY; minimax / deepseek read prefixed keys")
    ap.add_argument("--model", default="", help="upstream model id (default: the profile's model)")
    ap.add_argument("--timeout", type=float, default=900.0, help="wall-clock limit per job in seconds")
    ap.add_argument("--tool-timeout", type=float, default=120.0, help="fsm arm: timeout of each tool call in seconds")
    ap.add_argument("--opencode-bin", default="opencode", help="OpenCode executable")
    ap.add_argument("--executor", choices=("opencode", "local"), default="opencode",
                    help="fsm arm: OpenCode native tools, or bash in a local subprocess")
    ap.add_argument("--retries", type=int, default=3, help="fsm arm: retries at the fallback state")
    ap.add_argument("--tools", default=None,
                    help="fsm arm: tool registry JSON for machine tools the backend lacks (realized by the model)")
    ap.add_argument("--no-interpret", action="store_true",
                    help="fsm arm: stop when retries are exhausted instead of interpreted execution")
    ap.add_argument("--no-think-judge", action="store_true", help="fsm arm: enable_thinking=false for judge requests")
    ap.add_argument("--no-think", action="store_true", help="fsm arm: enable_thinking=false for every model request")
    ap.add_argument("--think-budget", type=int, default=0, help="fsm arm: thinking_budget for generation requests")
    ap.add_argument("--judge-think-budget", type=int, default=0, help="fsm arm: thinking_budget for judge requests")
    ap.add_argument("--llm-timeout", type=float, default=900.0, help="fsm arm: HTTP read timeout of a model request")
    ap.add_argument("--llm-retries", type=int, default=2, help="fsm arm: retries of a failed model request")
    ap.add_argument("--max-tokens", type=int, default=32768, help="fsm arm: max_tokens of a generation")
    ap.add_argument("--stream", action="store_true", help="fsm arm: stream model responses")
    ap.add_argument("--awm-file", default=None, help="awm arm: induced workflows placed before the task prompt")
    ap.add_argument("--rbank", default=None,
                    help="rbank arm: precomputed retrieval JSON, task id -> injected text (skill2fsm memory precompute)")
    ap.add_argument("--no-resume", action="store_true", help="ignore results of earlier runs")
    ap.add_argument("--dry-run", action="store_true", help="list the jobs without running them")
    a = ap.parse_args(argv)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if "fsm" in arms and not a.machine:
        ap.error("the fsm arm needs --machine")
    if "awm" in arms and not a.awm_file:
        ap.error("the awm arm needs --awm-file")
    if "rbank" in arms and not a.rbank:
        ap.error("the rbank arm needs --rbank")
    for attr in ("tasks_file", "skill", "machine", "data_root", "bench_dir", "awm_file", "rbank"):
        if getattr(a, attr):                       # job subprocesses must resolve paths the same way
            setattr(a, attr, str(pathlib.Path(getattr(a, attr)).resolve()))
    MODE["name"] = a.mode
    MODE["tasks_file"] = a.tasks_file

    pool = load_pool()
    tasks = pick_tasks(pool, a.n, a.seed, a.tasks)
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, tid, rep) for tid in tasks for rep in range(1, a.reps + 1) for arm in arms]
    say(f"task pool {len(pool)}; selected {len(tasks)}: {', '.join(tasks)}")
    say(f"arms {arms} × {a.reps} repetitions = {len(jobs)} jobs, {a.workers} workers; "
        f"model {a.model or '(profile default)'}; output {out}")
    if a.dry_run:
        for arm, tid, rep in jobs:
            say(f"  {arm:<5} {tid} r{rep}")
        return 0

    from skill2fsm.env import llm_config
    cfg = llm_config(profile=("" if a.provider in ("default", "") else a.provider))
    a.model = a.model or cfg.model
    skill_md = (pathlib.Path(a.skill) / "SKILL.md").read_text(encoding="utf-8")
    env = dict(os.environ)
    head = PROMPT_HEAD_LIVEMATH if a.mode == "livemath" else PROMPT_HEAD_FILETASK if a.mode == "filetask" else PROMPT_HEAD
    env["OPENCODE_CONFIG_CONTENT"] = agent_config(f"openai-compatible/{a.model}", a.model, skill_md, head=head)
    env["BASE_URL"], env["API_KEY"] = cfg.base_url, cfg.api_key
    if a.mode == "livemath":
        from skill2fsm.evaluators.mcq_boxed import grade_answer_file as grader
    elif a.mode == "filetask":
        grader = None
    else:
        from skill2fsm.evaluators.spreadsheet_golden import compare_workbooks as grader

    memory_ctx = {"awm": ("## Workflows induced from past experience\n\n"
                          + pathlib.Path(a.awm_file).read_text(encoding="utf-8")) if a.awm_file else ""}
    memory_ctx["rbank"] = json.loads(pathlib.Path(a.rbank).read_text(encoding="utf-8")) if a.rbank else {}
    results_path = out / "results.jsonl"
    done: dict = {}
    if results_path.is_file() and not a.no_resume:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["arm"], r["task"], r["rep"])] = r
    todo = [j for j in jobs if j not in done]
    if done:
        say(f"resuming: {len(done)} jobs already have results; {len(todo)} left")

    t_start = time.monotonic()
    tally = {arm: [0, 0] for arm in arms}          # [passed, finished]
    for r in done.values():
        if r["arm"] in tally:
            tally[r["arm"]][0] += bool(r["passed"])
            tally[r["arm"]][1] += 1

    def run(job):
        arm, tid, rep = job
        job_dir = out / arm / tid / f"r{rep}"
        say(f"▶ start  {arm:<5} {tid} r{rep}")
        if arm == "fsm":
            r = run_fsm_job(a, pool[tid], tid, rep, job_dir)
        elif arm == "awm":
            r = run_skill_job(a, pool[tid], tid, rep, job_dir, env, grader, extra_context=memory_ctx["awm"])
        elif arm == "rbank":
            r = run_skill_job(a, pool[tid], tid, rep, job_dir, env, grader, extra_context=memory_ctx["rbank"][tid])
        else:
            r = run_skill_job(a, pool[tid], tid, rep, job_dir, env, grader)
        r.update({"arm": arm, "task": tid, "rep": rep, "model": a.model})
        return r

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool_exec:
        futures = {pool_exec.submit(run, j): j for j in todo}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:                          # noqa: BLE001
                r = {"arm": job[0], "task": job[1], "rep": job[2], "passed": False,
                     "why": f"{type(exc).__name__}: {exc}"[:300], "duration_s": 0.0,
                     "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}
            with _results_lock:
                done[job] = r
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                tally[r["arm"]][0] += bool(r["passed"])
                tally[r["arm"]][1] += 1
                elapsed = time.monotonic() - t_start
                status = " ".join(f"{arm}:{tally[arm][0]}/{tally[arm][1]}" for arm in arms)
                say(f"✔ [{len(done)}/{len(jobs)}] "
                    f"{r['arm']:<5} {r['task']} r{r['rep']}  {'PASS' if r['passed'] else 'FAIL'}  "
                    f"{r['duration_s']:.0f}s  {r.get('total_tokens') or 0:,} tok"
                    + (f"  retries {r['retries']}" if r.get("retries") else "")
                    + (f"  interpreted {r['fallback_steps']} steps" if r.get("fallback_steps") else "")
                    + f"  |  passed {status}  elapsed {elapsed / 60:.1f} min"
                    + ("" if r["passed"] else f"  ← {str(r.get('why') or r.get('error') or '')[:80]}"))

    rows = [done[j] for j in jobs if j in done]
    summary, text = summarize(rows, tasks, arms, a.reps)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"results: {results_path}  summary: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
