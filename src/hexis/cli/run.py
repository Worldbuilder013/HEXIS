"""Execute a machine on one task.

Give every task input the machine declares with ``--input KEY=VALUE`` (repeatable) or ``--input-file
inputs.json``; ``--prompt`` sets ``request``. Every model and tool call is printed as it happens.

Tool steps run through OpenCode's native tools (``--executor opencode``) or a local ``bash`` subprocess
(``--executor local``) in ``--workdir``; rendered argument templates are passed unchanged. When a step fails,
the fallback state first retries from the most recent tool step (``--retries``) and then hands the task to
interpreted execution of ``SKILL.md``, unless ``--no-interpret`` is given. The model endpoint is configured with
``--model``, ``--base-url`` and ``--api-key-env`` or through the environment (see :mod:`hexis.llm.env`).

    hexis-agent run --machine BUILD_DIR --input request="Clean data.csv" --workdir work/ --executor local \\
        --model MODEL_ID --base-url URL
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys
import time
from tempfile import TemporaryDirectory

from hexis.llm.llm_client import client_from_env
from hexis.machine.schema import load_machine

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--machine", required=True,
                    help="machine JSON file, or a directory containing machine.json (for example a build directory)")
    ap.add_argument("--skill", default=None,
                    help="skill directory; its SKILL.md is used by interpreted execution after fallback "
                         "(default for a build directory: BUILD/skill)")
    ap.add_argument("--input", action="append", default=None, metavar="KEY=VALUE",
                    help="a task input; repeatable")
    ap.add_argument("--input-file", default=None, help="JSON object with the task inputs")
    ap.add_argument("--workdir", default=None,
                    help="working directory for tool calls, created if needed and kept (default: a temporary "
                         "directory that is removed afterwards)")
    ap.add_argument("--prompt", default=None, help="task text, passed as the input request")
    ap.add_argument("--max-steps", type=int, default=None, help="step limit (default: the machine's max_steps)")
    ap.add_argument("--retries", type=int, default=3,
                    help="retries at the fallback state, each re-entering the most recent tool step")
    ap.add_argument("--no-interpret", action="store_true",
                    help="stop when the retries are exhausted instead of handing the task to interpreted execution")
    ap.add_argument("--json", default=None, help="write the trace as JSONL to this file")
    ap.add_argument("--result-json", default=None,
                    help="write a structured result (stop reason, path, tokens, time) to this file")
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
    a.task_inputs = _task_inputs(ap, a)
    return run_machine(a)


def _task_inputs(ap, a) -> dict:
    """The task inputs, checked against the variables the machine initializes from the task."""
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


def run_machine(a) -> int:
    """Run the machine on the task described by the parsed options ``a``."""
    from hexis.execution import runtime
    from hexis.llm.llm_client import ModelAdapter
    from hexis.tools.opencode_tools import FALLBACK_PROTOCOL, PRIMITIVES, OpenCodeTools

    m = load_machine(pathlib.Path(a.machine))
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
    print(f"task inputs: {json.dumps(a.task_inputs, ensure_ascii=False)[:300]}")
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
        task = {"task_id": "task", "input": dict(a.task_inputs)}
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
            "stopped": rr.stopped, "error": rr.error or "", "path": rr.path(),
            "llm_calls": rr.llm_calls, "prompt_tokens": rr.prompt_tokens,
            "completion_tokens": rr.completion_tokens, "unmeasured_calls": rr.unmeasured_calls,
            "fallback_entry": rr.fallback_entry, "fallback_steps": rr.fallback_steps,
            "retries": rr.retries,
            "tool_calls": len(getattr(tools, "calls", []) or []),
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

    print(f"\nworking directory: {a.workdir if getattr(a, 'workdir', None) else '(temporary, removed; use --workdir to keep it)'}")
    print("final variables: " + json.dumps({k: v for k, v in rr.values.items() if k not in a.task_inputs},
                                           ensure_ascii=False, default=str)[:1000])
    return 1 if rr.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
