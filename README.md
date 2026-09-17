<div align="center">

# hexis

**Compile agent skills into extended finite state machines.**

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)
[![Status](https://img.shields.io/badge/status-alpha-orange)](CHANGELOG.md)

[Why](#why-hexis) · [Installation](#installation) · [Quick start](#quick-start) · [Concepts](#concepts) ·
[Updating](#updating-a-machine-with-a-model) · [Using a machine](#using-a-machine) · [CLI](#command-line-interface) ·
[Evaluation](#evaluation) · [Development](#development) · [Citation](#citation)

</div>

---

## Why hexis

An agent skill (a `SKILL.md` document) tells an agent how to carry out a class of tasks. In the usual way of
executing a skill, the document sits in the model's context and the model chooses every next step, so
requirements that the document states clearly can still be skipped, reordered or applied in the wrong situation.

hexis compiles the skill into an **extended finite state machine**. The machine records progress in a current
state and a set of variables, executes the operation assigned to the current state and evaluates transition
guards over the recorded values to decide what comes next. Language models do the reasoning and generation inside
states, with the state's prompt and the variables it reads; the order of operations is enforced by the program.

This package accompanies the paper *Compiling Agent Skills into Extended Finite State Machines*.

## Highlights

- **Compile with any OpenAI-compatible model.** Choose the model, endpoint and key variable on the command line
  (`--model`, `--base-url`, `--api-key-env`); the model drafts the machine and redrafts it under check feedback.
- **Update with new traces, one decision per step.** A model decides for every step of a new trace whether an
  existing state produces it, a new state is needed, or the step is noise. Static checks and replay of *every*
  previously accepted trace guard each change, so a wrong decision cannot break the machine. Answers are cached
  and runs resume where they stopped.
- **Usage guides.** Every build gets `GUIDE.md` (inputs, tools, a diagram, every state and transition) and
  `PROMPT.md`, a system prompt that lets any tool-using agent execute the machine step by step.
- **Readable, executable machines.** Plain JSON (`efsm-v1`) with typed variables, `tool` / `model` / `judge` /
  `user` / `end` actions, ordered guarded transitions, bounded loops and a fallback state.
- **Graceful degradation.** The fallback state retries from the most recent tool step and then hands the task to
  interpreted execution of the skill, so a partially learned machine can still finish a task.
- **Real tool backends.** OpenCode's native tools, a local `bash` backend, or tools that the model realizes from
  registry definitions.
- **Evaluation harness.** Parallel comparison of machines with direct skill execution and memory baselines (Agent
  Workflow Memory, ReasoningBank); graders for spreadsheet, multiple-choice and file-answer tasks; paired exact tests.
- **Hermetic test suite.** More than 350 tests that need no network access, model endpoint or API key.

## How it works

<p align="center">
  <img src="docs/figures/overview.png" width="820"
       alt="Native skill execution, where the model infers the next operation, compared with hexis, where a program evaluates the transition condition and selects the next state">
</p>

*Native execution uses model inference to select the next operation; skill optimization, reflection and memory
improve the model's inputs. hexis uses a program to evaluate transition conditions and execute the operation
assigned to the selected state.*

1. **Compile.** The model reads the skill document, its clauses, the tool definitions, the task input fields and
   the skill rules, and drafts a machine. Static checks (format, schema, guards, reachability and termination,
   clause coverage, tools) return errors to the model until a draft passes.
2. **Update.** Each trace is normalized into events. For every step the decider (a model, a decisions file, or
   deterministic alignment) chooses how the machine accounts for it, a candidate machine is built by reusing or
   adding states and transitions, and the candidate is accepted only if it passes the variable, evidence,
   requirement and structure checks and replays the new trace and every previously accepted trace.
3. **Execution.** The runtime executes the action of the current state, writes the declared variables and takes
   the first transition whose guard holds. Models are called only for `model` and `judge` states.

## Installation

```bash
pip install hexis-agent             # the package is imported as `hexis`; the command is `hexis-agent`
pip install "hexis-agent[all]"      # with every optional extra
pip install .                       # from a checkout; Python >= 3.11
```

| Extra | Adds | Needed for |
|---|---|---|
| `xlsx` | openpyxl | grading spreadsheet tasks |
| `memory` | numpy, sentence-transformers | `hexis-agent memory index / precompute / retrieve` |
| `dev` | pytest, build, twine, ruff | tests, linting and packaging |

Running machines on real tasks also requires:

| Requirement | Used for |
|---|---|
| An OpenAI-compatible chat endpoint | model calls (see [Configuration](#configuration)) |
| [OpenCode](https://opencode.ai) on `PATH` | native tool execution and the skill-execution baseline (`--executor local` runs `bash` without it) |
| LibreOffice (`soffice`) | recalculating spreadsheets before grading |

## Quick start

The package ships a small hermetic example skill, `table_clean`, with scripted tools and a scripted model, so a
machine runs without network access:

```python
from hexis.execution import runtime
from hexis.examples import table_clean as tc

machine = tc.reference_machine()
task = tc.gen_tasks(1, seed=0)[0]
result = runtime.run_task(
    machine, task,
    model=tc.build_model(),
    tools=tc.build_registry(tc.MemFS(task["files"])),
    doc=tc.skill_doc(),
)
print(result.stopped, " -> ".join(result.path()), tc.verify(task, result.trace))
# terminal s1 -> s2 -> s4 -> end True
```

With a real skill and a model endpoint:

```bash
export API_KEY=...                         # or put MODEL / BASE_URL / API_KEY in a .env file
M="--model qwen3.6-flash --base-url https://your-endpoint/v1"

# 1. compile a machine from the skill document
hexis-agent compile --skill path/to/skill --out build/ $M

# 2. collect traces (for example with `hexis-agent collect`) and fold them into the machine
hexis-agent update --build build/ --traces traces/ --show 3    # preview: no model calls, nothing written
hexis-agent update --build build/ --traces traces/ $M

# 3. use the machine
cat build/GUIDE.md                          # how the machine works; build/PROMPT.md is the agent prompt
hexis-agent run --mode task --machine build/ --input request="..." --workdir work/ --executor local $M
```

## Concepts

### Machines

A machine is a JSON document in the `efsm-v1` format (`hexis.machine.schema.Machine`):

```json
{
  "format": "efsm-v1",
  "skill_id": "answer-file",
  "initial": "draft",
  "fallback": "FALLBACK",
  "variables": [
    {"name": "request", "init_from": "task.input.request"},
    {"name": "output_path", "init_from": "task.input.output_path"},
    {"name": "write_count", "type": "integer", "init": 0}
  ],
  "states": {
    "draft": {"id": "draft",
              "action": {"kind": "model", "reads": ["request", "output_path"], "writes": ["command"],
                         "prompt": "Solve the request. Write command: one shell command that writes only the final answer to output_path."},
              "transitions": [{"if": "write_count >= 2", "to": "FALLBACK"}, {"if": "", "to": "write"}]},
    "write": {"id": "write",
              "action": {"kind": "tool", "name": "bash", "input": {"command": "${command}"},
                         "writes": ["returncode", "stdout", "stderr"], "phase": "apply"},
              "transitions": [{"if": "returncode == 0", "to": "check"},
                              {"if": "", "to": "draft", "inc": "write_count"}]},
    "check": {"id": "check",
              "action": {"kind": "tool", "name": "read", "input": {"filePath": "${output_path}"},
                         "writes": ["ok", "stdout"], "phase": "verify"},
              "transitions": [{"if": "ok", "to": "END_VERIFIED"}, {"if": "", "to": "END_UNVERIFIED"}]},
    "END_VERIFIED": {"id": "END_VERIFIED", "action": {"kind": "end", "terminal": "END_VERIFIED"}},
    "END_UNVERIFIED": {"id": "END_UNVERIFIED", "action": {"kind": "end", "terminal": "END_UNVERIFIED"}},
    "FALLBACK": {"id": "FALLBACK", "action": {"kind": "end", "terminal": "END_FALLBACK"}}
  },
  "terminals": [
    {"id": "END_VERIFIED", "kind": "verified"},
    {"id": "END_UNVERIFIED", "kind": "unverified"},
    {"id": "END_FALLBACK", "kind": "fallback"}
  ]
}
```

| Element | Meaning |
|---|---|
| `variables` | Initialized from task inputs (`init_from`) or constants (`init`) and written by actions |
| `tool` action | Runs a named tool; `${var}` in the argument template is substituted at run time; outputs are written to `writes` |
| `model` action | Generates the listed variables from the state's prompt and the variables it reads |
| `judge` action | Chooses one label from a fixed set, or abstains (the abstain label is `abstain`) |
| `user` / `end` action | Asks for input / finishes with a terminal |
| `transitions` | Evaluated in order after the action; the first guard that holds is taken; an empty guard is unconditional and comes last; `inc` increments a loop counter |
| `fallback` | Retries from the most recent tool step, then hands the task to interpreted execution of the skill |

Guards use a small whitelisted expression language (`hexis.machine.cond`) that supports static checks of mutual
exclusion and loop bounds; it never calls `eval`.

### Traces

Traces are JSON Lines files: a header with the task and its verdict, then one line per step.

```json
{"task_id": "t1", "arm": "agent", "harness": "opencode", "model": "qwen3.6-flash", "verdict": "accepted", "input": {"request": "...", "input_path": "", "output_path": "answer.txt"}}
{"kind": "tool", "name": "bash", "args": {"command": "python3 solve.py > answer.txt"}, "stdout": "", "stderr": "", "returncode": 0}
{"kind": "model", "text": "The answer has been written to answer.txt."}
{"kind": "end"}
```

`hexis-agent collect` and `hexis-agent fold-traces` produce this format from OpenCode event streams.

### Skill rules

A `compile.json` next to `SKILL.md` declares terminals, derived labels, requirements and terminal conditions. Every
rule quotes the sentence of the skill document it comes from. Without the file, the compiler asks the model to
extract the rules and checks each quote against the document; the effective rules are saved as `rules.json` in the
build directory.

```json
{
  "terminals": [{"id": "END_VERIFIED", "kind": "verified"}, {"id": "END_UNVERIFIED", "kind": "unverified"}],
  "labels": [
    {"label": "verify",
     "when": {"kind": "tool", "label": "probe", "args_contain": "${output_path}", "after": {"label": "apply"}},
     "quote": "Read the output file back after writing it."}
  ],
  "requirements": [
    {"id": "R1", "kind": "before", "a": {"kind": "tool", "label": "probe"}, "b": {"kind": "tool", "label": "apply"},
     "quote": "Inspect the input before changing anything."}
  ],
  "terminal_conditions": [
    {"terminal": "END_VERIFIED",
     "required_evidence": [{"kind": "tool", "label": "verify", "success": true}],
     "invalidating_events": [{"kind": "tool", "label": "apply"}],
     "quote": "Read the output file back after writing it."}
  ]
}
```

### Tools

Tool definitions come from a registry (`hexis/tools/backends/opencode.json` describes OpenCode's native tools, or
pass `--tools registry.json`) or are inferred from traces. The compiler treats tool names as opaque identifiers.
At run time a machine may only use tools that the backend provides, or tools defined in a `--tools` registry,
which the model then realizes as shell commands; tools are never silently substituted.

### Build directory

`compile --out BUILD` writes everything a later `update` needs:

| File | Contents |
|---|---|
| `machine.json` / `machine_init.json` | the current machine / the initial machine |
| `build.json` | manifest: skill, tool registry and rules sources, one record per run (command, decider, model id and base URL, outcomes, token usage); never an API key |
| `skill/SKILL.md`, `tools.json`, `rules.json` | the inputs of the compilation |
| `traces/` | copies of every trace used, named `<file stem>-<content hash>.jsonl` |
| `progress.json` | the outcome of every trace and the accepted traces that protect the machine |
| `decisions.jsonl` | every model question and answer (the cache of `update`) |
| `report.md`, `context.json`, `init_log.json`, `update_log.json` | reports and logs |
| `GUIDE.md`, `PROMPT.md` | usage guide and agent prompt |

## Updating a machine with a model

```bash
hexis-agent update --build build/ --traces new_traces/ --model MODEL --base-url URL
```

For each new trace, and for each step of it, the model sees the step (tool, arguments, results, narration), the
task inputs, the skill's clauses and the candidate states that could produce the step, and answers:

```json
{"decision": "match | new | ignore | exclude", "state": "<candidate id or null>",
 "purpose": "<one sentence>", "clause": "<clause id or empty>"}
```

- `match` reuses a state (adding a transition if needed); `new` adds a state whose argument prompt is built from
  the purpose; `ignore` removes harness noise from the trace; `exclude` leaves the whole trace out.
- A candidate machine is accepted only if it passes the checks, replays the new trace and replays **every trace
  accepted before**, including those of earlier runs. Otherwise the machine is unchanged and the trace is
  recorded as rejected.
- Answers are cached in `decisions.jsonl`: running the same command again asks nothing new, and after an
  interruption (exit status 3) it continues where it stopped. `--no-cache` asks again.
- The deterministic proposal is never shown to the model; it is used only when the model gives no valid answer
  after one repair turn.
- Cost: about one model call per trace step. `--show N` previews pending traces and the number of steps without
  calling the model; `--max-traces N` limits a run.
- Other deciders: `--decisions FILE` applies decisions written by a person or another tool (the same JSON shape
  per step, keyed by trace), and `--decider align` uses deterministic alignment without any model calls. `compile
  --traces` uses deterministic alignment by default and a model with `--decider model`.

Before processing anything, `update` checks that the current machine still passes the checks and replays every
accepted trace; if it does not (for example after a manual edit, or when tool definitions inferred from traces
changed), it stops with exit status 2 without changing anything.

## Using a machine

- **Read the guide.** `GUIDE.md` lists the task inputs, the tools, a Mermaid diagram, every state with its prompt
  or tool call, the transitions in evaluation order, loop limits and fallback behaviour.
- **Run it.** `hexis-agent run --mode task --machine BUILD --input KEY=VALUE ... --workdir DIR` runs the machine
  on any inputs with the hexis runtime (`--executor local` runs `bash` in a subprocess, the default executor uses
  OpenCode's tools). The benchmark modes `xlsx`, `livemath` and `filetask` also grade the result.
- **Give it to an agent.** `PROMPT.md` is a system prompt that tells a tool-using agent (for example Claude Code
  or OpenCode) how to execute the machine state by state: the execution loop, variables, the guard language, the
  tools, every state and its transitions, the finishing report and the fallback procedure. `hexis-agent guide
  --embed-skill` appends the skill document so the agent can finish a task after falling back.
- **From Python.** `hexis.execution.runtime.run_task(machine, {"input": {...}}, model=..., tools=..., doc=...)`.

`hexis-agent guide --build BUILD` (or `--machine machine.json`) regenerates both files; `compile` and `update`
write them automatically unless `--no-guide` is given.

## Command-line interface

| Command | Purpose |
|---|---|
| `hexis-agent compile` | Initialize a machine from a skill document with a model, optionally fold in traces; writes a build directory |
| `hexis-agent update` | Fold new traces into a build; a model (or a decisions file, or deterministic alignment) decides every step |
| `hexis-agent guide` | Write `GUIDE.md` and `PROMPT.md` for a machine |
| `hexis-agent run` | Execute a machine on one task (`task`, `xlsx`, `livemath` or `filetask` mode) |
| `hexis-agent compile-stepwise` | Update a machine trace by trace with step decisions from a file (`--show`, `--apply`, `--report`) |
| `hexis-agent collect` | Execute a skill with OpenCode on spreadsheet tasks and save traces |
| `hexis-agent fold-traces` | Turn the skill-arm event streams of a benchmark run into traces |
| `hexis-agent bench` | Run machine and skill-execution arms on a task set in parallel |
| `hexis-agent memory` | Build Agent Workflow Memory workflows and ReasoningBank memories |
| `hexis-agent summarize` | Compare arms on shared tasks: pass rate, time, tokens, paired exact tests |

Run `hexis-agent <command> --help` for all options; `python -m hexis` is equivalent to `hexis-agent`.

### Configuration

| Setting | Effect |
|---|---|
| `--model`, `--base-url` | Model id and OpenAI-compatible base URL; override `MODEL` / `BASE_URL` |
| `--api-key-env NAME` | Environment variable that holds the API key (default `API_KEY`); keys are never passed on the command line |
| `--provider NAME` | Read `NAME_MODEL`, `NAME_BASE_URL`, `NAME_API_KEY` instead (built-in defaults for `minimax` and `deepseek`) |
| `MODEL`, `BASE_URL`, `API_KEY` | Default endpoint |
| `.env` | Read from the working directory upward; variables already set in the environment take precedence |
| `--temperature`, `--max-tokens`, `--llm-timeout`, `--llm-retries`, `--stream`, `--extra-body JSON` | Request settings for `compile` and `update` (`--extra-body` is merged into every request, e.g. `'{"enable_thinking": false}'`) |
| `--no-think`, `--think-budget`, `--judge-think-budget` | Reasoning controls of `run` for Qwen-compatible endpoints |
| `LABEL=self` | Memory baselines label trajectories by model self-judgement (`hexis-agent memory judge`) instead of the grader |

See `.env.example` for a template. Exit statuses: 0 success, 2 usage, configuration or build problem, 3 model
endpoint failure or interruption (progress is saved).

## Evaluation

`hexis-agent bench` runs arms on a task set: `fsm` (a compiled machine), `skill` (OpenCode with the skill in the
system prompt), and `awm` / `rbank` (the skill arm with Agent Workflow Memory workflows or ReasoningBank memories
placed before the prompt). Task files are YAML lists:

```yaml
- id: t1
  turns: ["Compute the mean of column x in data.csv. Write @mean[value] to answer.txt."]
  metadata:
    verifier: dabench            # file-answer graders: dabench, sealqa_judge
    answers: [[mean, "34.65"]]
    assets: [data/data.csv]      # copied into the job directory; relative to --data-root
```

| Mode | Required metadata | Grading |
|---|---|---|
| `xlsx` | `init_asset`, `golden`, `answer_position` | SpreadsheetBench comparison after LibreOffice recalculation |
| `livemath` | `answer` | last `\boxed{X}` in `answer.txt` |
| `filetask` | `verifier`, `answers` or `answer`, `assets` or `assets_dir` + `assets_target` | `dabench`: `@name[value]` with the reference precision; `sealqa_judge`: recorded for an external judge |

```bash
hexis-agent bench --mode filetask --tasks-file tasks.yaml --tasks t1,t2 --reps 1 \
    --arms fsm,skill --machine build/machine.json --skill path/to/skill --out runs/test

# memory baselines built from a development run and frozen for the test run
hexis-agent memory convert runs/dev dev_traj.jsonl
hexis-agent memory awm dev_traj.jsonl workflows.txt
hexis-agent memory rbank dev_traj.jsonl bank.jsonl
hexis-agent memory index bank.jsonl
hexis-agent memory precompute bank.jsonl tasks.yaml test_ids.txt rbank.json --mode filetask
hexis-agent bench --mode filetask --tasks-file tasks.yaml --tasks t1,t2 --reps 1 \
    --arms awm,rbank --awm-file workflows.txt --rbank rbank.json --skill path/to/skill --out runs/test

hexis-agent summarize table.md "Test split" fsm=runs/test skill=runs/test awm=runs/test rbank=runs/test
```

## Project layout

```text
src/hexis/
├── machine/              efsm-v1 schema, guard language, structural checks
├── compiler/             compile context, initialization, trace normalization, alignment,
│                         candidate construction, checks and replay, update, stepwise decisions
├── execution/            runtime interpreter with retries and fallback
├── llm/                  OpenAI-compatible client, endpoint configuration, model protocol
├── tools/                tool registries and backends (OpenCode, local subprocess, model-realized)
├── traces/               trace formats, normalization, phase classification, judging
├── evaluators/           graders
├── examples/table_clean/ hermetic example skill
├── builddir.py           build directories
├── updater.py            the update loop shared by compile and update
├── step_judge.py         the model that decides trace steps
├── guide.py              GUIDE.md and PROMPT.md
├── cli/                  the hexis-agent command
└── legacy/               modules of an earlier compiler iteration (not used by the CLI)
tests/                    hermetic test suite (tests/legacy covers the legacy modules)
```

### Python API

| Module | Contents |
|---|---|
| `hexis.machine.schema` | `Machine`, `State`, actions, `Transition`, `Variable`, `Trace`, `load_machine` |
| `hexis.execution.runtime` | `run_task`: execute a machine with a model and tools |
| `hexis.compiler` | `context.build_context`, `init.initialize`, `traces.load_traces`, `update.update`, `stepwise`, `decide.decide_trace` |
| `hexis.llm.llm_client` | OpenAI-compatible client (`client_from_env`) and `ModelAdapter` |
| `hexis.tools.opencode_tools`, `hexis.tools.local_tools` | Tool backends |
| `hexis.guide` | `render_guide`, `render_prompt`, `mermaid` |
| `hexis.evaluators` | Graders |

## Development

```bash
pip install -e ".[dev,xlsx]"
pytest                                   # hermetic; OpenCode tests are skipped without the opencode executable
ruff check src tests
python -m build && twine check --strict dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CHANGELOG.md](CHANGELOG.md) for release notes.

## Security and privacy

hexis executes tool calls whose arguments are generated by language models, including shell commands. Run it in an
isolated environment and only with machines and task files you trust. See [SECURITY.md](SECURITY.md).

`update` sends trace steps (tool arguments and shortened results) and the skill's clauses to the configured model
endpoint, and a build directory keeps copies of the traces and every question sent. `PROMPT.md` contains the whole
machine, including prompts derived from the skill document; the runtime itself only ever shows a model the prompt
of the state it is executing. Review build directories and prompts before sharing them.

## License

MIT (see [LICENSE](LICENSE)), except for the portions listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): the spreadsheet comparison adapted from SpreadsheetBench
(CC BY-SA 4.0) and the workflow-induction instruction adapted from Agent Workflow Memory (Apache-2.0).

## Citation

```bibtex
@misc{hexis2026,
  title  = {Compiling Agent Skills into Extended Finite State Machines},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Under review}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Acknowledgements

hexis builds on [SpreadsheetBench](https://github.com/RUCKBReasoning/SpreadsheetBench) for spreadsheet grading, on
[Agent Workflow Memory](https://github.com/zorazrw/agent-workflow-memory) and ReasoningBank for the memory
baselines, and on [OpenCode](https://opencode.ai) for tool execution.
