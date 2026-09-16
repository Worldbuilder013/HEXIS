"""Execute a machine on one task.

Every model and tool call is printed as it happens. Modes:

* ``xlsx``: spreadsheet task. Give ``--workbook`` and ``--prompt`` (plus ``--golden`` and
  ``--answer-position`` to grade), or ``--task`` with a SpreadsheetBench verified-400 directory
  (``--bench-dir``). Grading follows SpreadsheetBench: LibreOffice recalculates the output workbook
  and only the cached values in the answer range are compared.
* ``livemath``: multiple-choice question answered in ``answer.txt`` as ``\\boxed{X}``. Give ``--prompt``
  (and ``--answer`` to grade) or ``--task`` with ``--tasks-file``.
* ``filetask``: task from ``--tasks-file`` whose assets are copied into the working directory; the
  answer goes to ``answer.txt`` and is graded according to ``metadata.verifier``.
* ``task``: any machine on any inputs. Give every task input the machine declares with ``--input KEY=VALUE``
  (repeatable) or ``--input-file inputs.json``; ``--prompt`` sets ``request``. Tools run in ``--workdir``, which
  is kept. Nothing is graded.

Tool steps run through OpenCode's native tools (``--executor opencode``) or a local ``bash``
subprocess (``--executor local``); rendered argument templates are passed unchanged. When a step fails,
the fallback state first retries from the most recent tool step (``--retries``) and then hands the task
to interpreted execution of ``SKILL.md``, unless ``--no-interpret`` is given. The model endpoint is
configured through ``.env`` (see :mod:`hexis.llm.env`).

    hexis-agent run --mode livemath --machine machine.json --skill skills/livemath \\
        --task lm_202606_001 --tasks-file tasks/livemath.yaml --model qwen3.6-flash
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import shutil
import sys
import time
from tempfile import TemporaryDirectory

from hexis.llm.llm_client import client_from_env
from hexis.machine.schema import load_machine

DEFAULT_BENCH_DIR = "third_party/SpreadsheetBench/spreadsheetbench_verified_400"


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--machine", required=True,
                    help="machine JSON file, or a directory containing machine.json (for example a build directory)")
    ap.add_argument("--skill", default=None,
                    help="skill directory; its SKILL.md is used by interpreted execution after fallback "
                         "(default for a build directory: BUILD/skill)")
    ap.add_argument("--mode", choices=("xlsx", "livemath", "filetask", "task"), default="xlsx",
                    help="task type (see above)")
    ap.add_argument("--input", action="append", default=None, metavar="KEY=VALUE",
                    help="task mode: a task input; repeatable")
    ap.add_argument("--input-file", default=None, help="task mode: JSON object with the task inputs")
    ap.add_argument("--workdir", default=None,
                    help="working directory for tool calls, created if needed and kept (default: a temporary "
                         "directory that is removed afterwards)")
    ap.add_argument("--task", default=None,
                    help="task id: a SpreadsheetBench id such as sb_49801 (xlsx) or an id in --tasks-file")
    ap.add_argument("--tasks-file", default=None, help="task YAML for the livemath and filetask modes")
    ap.add_argument("--bench-dir", default=DEFAULT_BENCH_DIR,
                    help="SpreadsheetBench verified-400 directory used by --task in xlsx mode")
    ap.add_argument("--data-root", default=".", help="directory that asset paths in the task file are relative to")
    ap.add_argument("--prompt", default=None, help="task text (taken from the task when --task is given)")
    ap.add_argument("--answer", default=None, help="reference answer letter (livemath)")
    ap.add_argument("--workbook", default=None, help="input workbook (xlsx)")
    ap.add_argument("--golden", default=None, help="golden workbook (xlsx); grading needs --answer-position too")
    ap.add_argument("--answer-position", default=None, help="graded range (xlsx), e.g. C1 or B2:B17")
    ap.add_argument("--max-steps", type=int, default=None, help="step limit (default: the machine's max_steps)")
    ap.add_argument("--retries", type=int, default=3,
                    help="retries at the fallback state, each re-entering the most recent tool step")
    ap.add_argument("--no-interpret", action="store_true",
                    help="stop when the retries are exhausted instead of handing the task to interpreted execution")
    ap.add_argument("--json", default=None, help="write the trace as JSONL to this file")
    ap.add_argument("--result-json", default=None,
                    help="write a structured result (verdict, stop reason, path, tokens, time) to this file")
    ap.add_argument("--keep", default=None, help="copy the produced file to this path")
    ap.add_argument("--quiet", action="store_true", help="print the summary only, not every step")
    ap.add_argument("--provider", default="default",
                    help="endpoint profile: 'default' reads MODEL / BASE_URL / API_KEY; minimax / deepseek read prefixed keys")
    ap.add_argument("--model", default="", help="upstream model id (default: the profile's model)")
    ap.add_argument("--base-url", default="", help="base URL of an OpenAI-compatible endpoint (overrides BASE_URL)")
    ap.add_argument("--api-key-env", default="",
                    help="name of the environment variable that holds the API key (default API_KEY)")
    ap.add_argument("--executor", choices=("opencode", "local"), default="opencode",
                    help="tool executor: OpenCode native tools, or bash in a local subprocess")
    ap.add_argument("--opencode-bin", default="opencode", help="OpenCode executable")
    ap.add_argument("--tools", default=None,
                    help="tool registry JSON; machine tools the backend lacks are realized by the model as shell commands")
    ap.add_argument("--tool-timeout", type=float, default=120, help="timeout of each tool call in seconds")
    ap.add_argument("--no-think-judge", action="store_true",
                    help="send enable_thinking=false with judge requests (Qwen-compatible endpoints)")
    ap.add_argument("--no-think", action="store_true",
                    help="send enable_thinking=false with every model request (Qwen-compatible endpoints)")
    ap.add_argument("--think-budget", type=int, default=0,
                    help="thinking_budget for generation requests (Qwen-compatible endpoints); 0 = unset")
    ap.add_argument("--judge-think-budget", type=int, default=0,
                    help="thinking_budget for judge requests (Qwen-compatible endpoints); 0 = unset")
    ap.add_argument("--llm-timeout", type=float, default=900.0, help="HTTP read timeout of a model request in seconds")
    ap.add_argument("--llm-retries", type=int, default=2, help="retries of a failed model request")
    ap.add_argument("--max-tokens", type=int, default=32768, help="max_tokens of a generation, including inline reasoning")
    ap.add_argument("--stream", action="store_true", help="stream model responses (avoids gateway timeouts on long generations)")
    return ap


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = build_parser(prog)
    a = ap.parse_args(argv)
    if a.tool_timeout <= 0:
        ap.error("--tool-timeout must be positive")
    if a.skill is None:
        build_skill = pathlib.Path(a.machine) / "skill"
        if pathlib.Path(a.machine).is_dir() and (build_skill / "SKILL.md").is_file():
            a.skill = str(build_skill)
        else:
            ap.error("--skill is required unless --machine is a build directory")
    if a.mode == "task":
        a.task_inputs = _task_inputs(ap, a)
        return run_machine(a)
    if a.mode == "livemath":
        if a.task:
            if not a.tasks_file:
                ap.error("--task in livemath mode needs --tasks-file")
            _fill_from_tasks_file(a)
        elif not a.prompt:
            ap.error("livemath mode needs --task or --prompt")
    elif a.mode == "filetask":
        if not a.task or not a.tasks_file:
            ap.error("filetask mode needs --task and --tasks-file")
        _fill_from_tasks_file(a)
    else:
        if a.task:
            _fill_from_bench(a)
        if not a.workbook or not a.prompt:
            ap.error("xlsx mode needs --workbook and --prompt, or --task with --bench-dir")
    return run_machine(a)


def _task_inputs(ap, a) -> dict:
    """Task inputs for the task mode, checked against the variables the machine initializes from the task."""
    inputs: dict = {}
    if a.input_file:
        data = json.loads(pathlib.Path(a.input_file).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            ap.error("--input-file must contain a JSON object")
        inputs.update(data)
    for item in a.input or []:
        if "=" not in item:
            ap.error(f"--input expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        inputs[k.strip()] = v
    if a.prompt and "request" not in inputs:
        inputs["request"] = a.prompt
    m = load_machine(pathlib.Path(a.machine))
    needed = [v.init_from.split(".")[-1] for v in m.variables if v.init_from]
    missing = [k for k in needed if k not in inputs]
    if missing:
        ap.error(f"the machine needs the task inputs {missing}; give them with --input KEY=VALUE")
    return inputs


def _fill_from_bench(a) -> None:
    """``--task 49801`` or ``sb_49801``: prompt, input and golden workbooks and answer range from the
    SpreadsheetBench ``dataset.json``. Options given explicitly take precedence."""
    bench = pathlib.Path(a.bench_dir)
    if not (bench / "dataset.json").is_file():
        raise SystemExit(f"{bench / 'dataset.json'} not found: download SpreadsheetBench verified-400 or pass --bench-dir")
    sid = str(a.task).replace("sb_", "")
    rows = {str(r["id"]): r for r in json.loads((bench / "dataset.json").read_text(encoding="utf-8"))}
    if sid not in rows:
        raise SystemExit(f"task {sid} not found in {bench / 'dataset.json'}")
    r = rows[sid]
    d = bench / r["spreadsheet_path"]
    a.prompt = a.prompt or r["instruction"]
    a.workbook = a.workbook or str(sorted(d.glob("*_init.xlsx"))[0])
    a.golden = a.golden or str(sorted(d.glob("*_golden.xlsx"))[0])
    a.answer_position = a.answer_position or r["answer_position"]


def _fill_from_tasks_file(a) -> None:
    """``--task ID --tasks-file tasks.yaml``: prompt, and the reference answer (livemath) or the whole
    task entry (filetask)."""
    import yaml
    rows = {str(r["id"]): r for r in yaml.safe_load(pathlib.Path(a.tasks_file).read_text(encoding="utf-8"))}
    if str(a.task) not in rows:
        raise SystemExit(f"task {a.task} not found in {a.tasks_file}")
    r = rows[str(a.task)]
    a.prompt = a.prompt or str(r["turns"][0])
    if a.mode == "livemath":
        a.answer = str(r["metadata"]["answer"])
    else:
        a.filetask = r


def run_machine(a) -> int:
    """Run the machine on the task described by the parsed options ``a``."""
    from hexis.execution import runtime
    from hexis.llm.llm_client import ModelAdapter
    from hexis.tools.opencode_tools import FALLBACK_PROTOCOL, PRIMITIVES, OpenCodeTools

    m = load_machine(pathlib.Path(a.machine))
    mode = getattr(a, "mode", "xlsx")
    unsupported = sorted({st.action.name for st in m.states.values()
                          if st.action.kind == "tool" and st.action.name not in PRIMITIVES})
    realize_specs = {}
    tools_path = getattr(a, "tools", None)
    if unsupported and tools_path:
        from hexis.tools.toolspec import load_registry
        reg = load_registry(tools_path)
        realize_specs = {n: reg[n] for n in unsupported if n in reg}
        unsupported = [n for n in unsupported if n not in reg]
    if unsupported:
        why = (f"the machine uses tools that OpenCode does not provide: {unsupported}. This is usually an initial "
               f"machine that has not been updated with traces (hypothetical tools are replaced by observed ones "
               f"during the update). To run it anyway, pass --tools with a registry that defines these tools; the "
               f"model then realizes each call as a shell command.")
        if getattr(a, "result_json", None):
            rp = pathlib.Path(a.result_json)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps({"passed": False, "why": why, "stopped": "unrunnable", "error": why,
                                      "path": [], "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                      "duration_s": 0.0}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(why)
    # Interpreted execution after fallback receives the action protocol followed by the skill document.
    md = pathlib.Path(a.skill) / "SKILL.md"
    doc = FALLBACK_PROTOCOL + (md.read_text(encoding="utf-8") if md.is_file() else "")
    print(f"machine: {a.machine}  ({m.n_states()} states, initial {m.initial})")
    if mode == "task":
        print(f"task inputs: {json.dumps(a.task_inputs, ensure_ascii=False)[:300]}")
    else:
        print(f"task: {a.prompt[:300]}")
        print(f"input workbook: {a.workbook}" if mode == "xlsx" else "output: answer.txt")
    print("=" * 78, flush=True)
    t0 = time.time()

    class _Live:
        """Print each model question and tool call as it happens instead of after the run."""

        def __init__(self, model, tools):
            self._m, self._t = model, tools
            self.n = 0
            # the same lines go to <trace>.live, so a running job can be followed
            self._live_path = (str(a.json) + ".live") if getattr(a, "json", None) else None

        def _say(self, line):
            print(line, flush=True)
            if self._live_path:
                try:
                    with open(self._live_path, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except OSError:
                    pass

        def _stamp(self):
            self.n += 1
            return f"[{time.time() - t0:6.0f}s #{self.n:02d}]"

        # ---- model protocol ----
        def classify(self, *, prompt, values, labels, examples=()):
            self._say(f"{self._stamp()} judge... prompt={prompt[:60]!r} labels={labels}")
            out = self._m.classify(prompt=prompt, values=values, labels=labels, examples=examples)
            self._say(f"{'':>15}→ {out!r}")
            return out

        def generate(self, *, prompt, values, history=()):
            self._say(f"{self._stamp()} generate... prompt={prompt[:60]!r}")
            out = self._m.generate(prompt=prompt, values=values, history=history)
            self._say(f"{'':>15}→ {json.dumps(out, ensure_ascii=False)[:200]}")
            return out

        def usage(self):
            return self._m.usage()

        def __getattr__(self, k):
            return getattr(self._m, k)

        # ---- tool protocol ----
        def call(self, name, inp):
            self._say(f"{self._stamp()} tool {name} args={json.dumps(inp, ensure_ascii=False)[:160]}")
            out = self._t.call(name, inp)
            self._say(f"{'':>15}→ rc={'ok' if out.get('ok') else 'ERR'} "
                      f"stdout={str(out.get('stdout', out.get('error', '')))[:160]!r}")
            return out

    prov = "" if str(a.provider) in ("default", "") else str(a.provider)
    if getattr(a, "workdir", None):
        pathlib.Path(a.workdir).mkdir(parents=True, exist_ok=True)
        work_cm = contextlib.nullcontext(str(pathlib.Path(a.workdir).resolve()))
    else:
        work_cm = TemporaryDirectory(prefix="fsm-run-")
    with client_from_env(profile=prov, timeout=float(getattr(a, "llm_timeout", 900.0)),
                         max_retries=int(getattr(a, "llm_retries", 2)), model=getattr(a, "model", "") or "",
                         base_url=getattr(a, "base_url", "") or "",
                         api_key_env=getattr(a, "api_key_env", "") or "") as cl, \
            work_cm as directory:
        if a.model:
            cl.model = a.model
        print(f"endpoint: {cl.model} @ {cl.base_url}", flush=True)
        work = pathlib.Path(directory)
        if mode == "xlsx":
            if a.workbook:
                shutil.copy(a.workbook, work / "input.xlsx")
            task = {"task_id": "prompt", "input": {
                "request": a.prompt, "problem": a.prompt, "input_path": str(work / "input.xlsx"),
                "output_path": str(work / "output.xlsx")}}
        elif mode == "filetask":
            from hexis.evaluators.filetask import stage_assets
            extra = stage_assets(a.filetask.get("metadata") or {}, work, root=getattr(a, "data_root", "."))
            task = {"task_id": str(a.task), "input": {
                "request": a.prompt, "problem": a.prompt, "output_path": str(work / "answer.txt"),
                "work_dir": str(work), **extra}}
        elif mode == "task":
            task = {"task_id": str(getattr(a, "task", None) or "task"), "input": dict(a.task_inputs)}
        else:
            task = {"task_id": str(getattr(a, "task", "prompt") or "prompt"), "input": {
                "request": a.prompt, "problem": a.prompt, "output_path": str(work / "answer.txt")}}
        max_tokens = int(getattr(a, "max_tokens", 32768))
        no_think = {"enable_thinking": False}
        gen_extra = (no_think if getattr(a, "no_think", False)
                     else ({"enable_thinking": True, "thinking_budget": int(a.think_budget)}
                           if getattr(a, "think_budget", 0) else None))
        judge_extra = ({"enable_thinking": True, "thinking_budget": int(a.judge_think_budget)}
                       if getattr(a, "judge_think_budget", 0)
                       else (no_think if (getattr(a, "no_think_judge", False) or getattr(a, "no_think", False)) else None))
        cl.max_tokens_ceiling = max(int(getattr(cl, "max_tokens_ceiling", 0)), max_tokens)
        cl.stream = bool(getattr(a, "stream", False))
        model = ModelAdapter(cl, temperature=0.0, max_tokens=max_tokens, extra_body=gen_extra,
                             classify_extra_body=judge_extra)
        if getattr(a, "executor", "opencode") == "local":
            from hexis.tools.local_tools import LocalTools
            native_cm = LocalTools(work, timeout_s=a.tool_timeout, python_bin=sys.executable)
        else:
            native_cm = OpenCodeTools(work, binary=a.opencode_bin, timeout_s=a.tool_timeout)
        with native_cm as native:
            tools = native
            if realize_specs:
                from hexis.tools.backends.realized import RealizedTools
                tools = RealizedTools(native, realize_specs, model)
            missing = {st.action.name for st in m.states.values()
                       if st.action.kind == "tool"} - set(tools.available_tools)
            if missing:
                raise SystemExit(f"the tool backend lacks {sorted(missing)}; tools are never substituted")
            live = _Live(model, tools)
            if realize_specs:
                tools.model = live                 # model calls that realize tools are printed and counted too
            print(("tool backend: local subprocess (bash)" if getattr(a, "executor", "opencode") == "local"
                   else "tool backend: OpenCode native tools")
                  + (f" + model-realized {sorted(realize_specs)}" if realize_specs else ""), flush=True)
            print("--- live ---", flush=True)
            rr = runtime.run_task(m, task, model=live, tools=live, doc=doc,
                                  max_steps=a.max_steps, on_error="fallback",
                                  retries=int(getattr(a, "retries", 3)),
                                  interpret=not getattr(a, "no_interpret", False))
        print(f"--- finished in {time.time() - t0:.0f}s ---\n", flush=True)
        produced = None if mode == "task" else work / ("output.xlsx" if mode == "xlsx" else "answer.txt")
        kept = pathlib.Path(a.keep) if a.keep else None
        if kept and produced is not None and produced.is_file():
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(produced, kept)
        ok_produced = produced.is_file() if produced is not None else False
        verdict = None
        if mode == "livemath":
            if getattr(a, "answer", None) is not None:
                from hexis.evaluators.mcq_boxed import grade_answer_file
                verdict = grade_answer_file(produced, a.answer)
        elif mode == "filetask":
            from hexis.evaluators.filetask import grade as grade_filetask
            verdict = grade_filetask(a.filetask.get("metadata") or {}, produced)[:2]
        elif a.golden and a.answer_position:
            from hexis.evaluators.spreadsheet_golden import compare_workbooks
            verdict = (compare_workbooks(pathlib.Path(a.golden), produced, a.answer_position)
                       if ok_produced else (False, "no output workbook"))

    if getattr(a, "result_json", None):
        rp = pathlib.Path(a.result_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        # token accounting per segment: records in the fallback state belong to interpreted execution
        seg = {"machine": [0, 0, 0], "fallback": [0, 0, 0]}
        for rec in rr.trace.records:
            key = "fallback" if rec.state == m.fallback else "machine"
            meta = rec.meta or {}
            seg[key][0] += int(meta.get("prompt_tokens") or 0)
            seg[key][1] += int(meta.get("completion_tokens") or 0)
            seg[key][2] += 1
        rp.write_text(json.dumps({
            "machine_prompt_tokens": seg["machine"][0], "machine_completion_tokens": seg["machine"][1],
            "machine_steps": seg["machine"][2],
            "fallback_prompt_tokens": seg["fallback"][0], "fallback_completion_tokens": seg["fallback"][1],
            "passed": bool(verdict[0]) if verdict is not None else None,
            "why": verdict[1] if verdict is not None else "not graded",
            "stopped": rr.stopped, "error": rr.error or "", "path": rr.path(),
            "llm_calls": rr.llm_calls, "prompt_tokens": rr.prompt_tokens,
            "completion_tokens": rr.completion_tokens, "unmeasured_calls": rr.unmeasured_calls,
            "fallback_entry": rr.fallback_entry, "fallback_steps": rr.fallback_steps,
            "retries": rr.retries,
            "tool_calls": len(getattr(tools, "calls", []) or []), "produced": bool(ok_produced),
            "realized_calls": len(getattr(tools, "realized", []) or []),
            "duration_s": round(time.time() - t0, 2), "model": cl.model,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.json:
        pathlib.Path(a.json).write_text(rr.trace.to_jsonl(), encoding="utf-8")
        print("trace written to", a.json)
    print(f"stopped: {rr.stopped}  error: {(rr.error or '')[:200] or '(none)'}")
    print(f"path: {' -> '.join(rr.path())}")
    print(f"fallback entry: {rr.fallback_entry or '(no fallback)'}  retries {rr.retries}  "
          f"interpreted steps {rr.fallback_steps}  model calls {rr.llm_calls}")
    _p, _c = rr.prompt_tokens, rr.completion_tokens
    if _p is None and _c is None:
        print(f"tokens: the endpoint reported no usage ({rr.unmeasured_calls} calls unmeasured)")
    else:
        print(f"tokens: prompt {_p or 0} + completion {_c or 0} = {(_p or 0) + (_c or 0)}"
              + (f" ({rr.unmeasured_calls} calls without reported usage)" if rr.unmeasured_calls else ""))
    if not a.quiet:
        print("\n--- steps ---")
        for rec in rr.trace.records:
            act = rec.action or {}
            k = act.get("kind")
            if k == "tool":
                print(f"[{rec.state:>10}] tool {act.get('name')}")
                print(f"{'':>13}args   = {json.dumps(act.get('input'), ensure_ascii=False)[:400]}")
                o = rec.output or {}
                print(f"{'':>13}stdout = {str(o.get('stdout', ''))[:300]}")
                if o.get("stderr"):
                    print(f"{'':>13}stderr = {str(o['stderr'])[:200]}")
            elif k == "judge":
                lab = next(iter((rec.output or {}).values()), "")
                print(f"[{rec.state:>10}] judge → {lab!r}   prompt = {str(act.get('prompt'))[:120]}")
            elif k == "model":
                print(f"[{rec.state:>10}] generate: {json.dumps(rec.output, ensure_ascii=False)[:300]}")
            else:
                print(f"[{rec.state:>10}] {k} {act}")

    if mode == "task":
        print(f"\nworking directory: {a.workdir if getattr(a, 'workdir', None) else '(temporary, removed; use --workdir to keep it)'}")
        print("final variables: " + json.dumps({k: v for k, v in rr.values.items() if k not in a.task_inputs},
                                               ensure_ascii=False, default=str)[:1000])
    elif ok_produced:
        print(f"\noutput: {kept if kept else f'{produced.name} in a temporary directory (use --keep to save it)'}")
    else:
        print(f"\noutput: no {produced.name} produced")
    if verdict is not None:
        ok, why = verdict
        print(f"verdict: {'PASS' if ok else 'FAIL'}  {why}")
        return 0 if ok else 1
    print("verdict: not graded")
    return 1 if rr.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
