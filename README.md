# skill2fsm

Compile agent skills into extended finite state machines.

An agent skill (a `SKILL.md` document) describes in natural language how an agent should carry out a
class of tasks. The usual way to use a skill is to place the document in the model's context and let the
model choose every next step. skill2fsm compiles the skill into an extended finite state machine (EFSM)
instead. The machine keeps the current state and a set of variables, executes the operation assigned to
the current state (a tool call, a model generation, a branch decision, a user input or a terminal) and
evaluates guards over the variables to choose the next state. Models are called only inside states, with
the prompt of the state and the variables it reads.

This package accompanies the paper *Compiling Agent Skills into Extended Finite State Machines*.

## Installation

```bash
pip install .            # from a checkout, Python >= 3.11
pip install ".[all]"     # with the optional extras below
```

| Extra | Adds | Needed for |
|---|---|---|
| `xlsx` | openpyxl | grading spreadsheet tasks |
| `memory` | numpy, sentence-transformers | `skill2fsm memory index / precompute / retrieve` |
| `dev` | pytest, build, twine | tests and packaging |

Running machines on real tasks also needs:

- an OpenAI-compatible chat endpoint: set `MODEL`, `BASE_URL` and `API_KEY` in the environment or in a
  `.env` file in the working directory (see `.env.example`). `--provider minimax` and
  `--provider deepseek` read `MINIMAX_*` and `DEEPSEEK_*` keys instead;
- [OpenCode](https://opencode.ai) on `PATH`, which executes tool calls (`--executor local` runs `bash`
  without it) and provides the skill-execution baseline;
- LibreOffice (`soffice`) to grade spreadsheet tasks.

## Quick start

The package ships a small hermetic example skill, `table_clean`, with scripted tools and a scripted
model, so a machine runs without network access:

```python
from skill2fsm import runtime
from skill2fsm.examples import table_clean as tc

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

With a real skill the pipeline is: collect traces, compile, run.

```bash
# 1. Execute the skill with OpenCode; event streams are folded into traces
skill2fsm collect --tasks-file tasks.yaml --skill path/to/skill --out traces/ --model qwen3.6-flash

# 2. Initialize a machine from SKILL.md with the model, then update it over the traces (no model)
skill2fsm compile --skill path/to/skill --traces traces/ --out build/

# 3. Execute the machine on a task
skill2fsm run --machine build/machine.json --skill path/to/skill \
    --workbook input.xlsx --prompt "..." --golden golden.xlsx --answer-position B2:B17
```

## Concepts

### Machines

A machine is a JSON document in the `efsm-v1` format (`skill2fsm.schema.Machine`):

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

- **Variables** are initialized from task inputs (`init_from`) or constants (`init`) and written by actions.
- **Actions**: `tool` runs a named tool with an argument template (`${var}` is substituted at run time)
  and writes its outputs; `model` generates the listed variables; `judge` chooses one of a fixed set of
  labels or abstains; `user` asks for input; `end` finishes with a terminal.
- **Transitions** are evaluated in order after the action. The first guard that holds is taken; an empty
  guard is unconditional and comes last; `inc` increments a loop counter. Guards use a small whitelisted
  expression language (`skill2fsm.cond`).
- Entering the **fallback** state first retries from the most recent tool step and then hands the task to
  interpreted execution of the skill document, so a partially learned machine can still finish a task.

### Traces

Traces are JSON Lines files: a header with the task and its verdict, then one line per step.

```json
{"task_id": "t1", "arm": "agent", "harness": "opencode", "model": "qwen3.6-flash", "verdict": "accepted", "input": {"request": "...", "input_path": "", "output_path": "answer.txt"}}
{"kind": "tool", "name": "bash", "args": {"command": "python3 solve.py > answer.txt"}, "stdout": "", "stderr": "", "returncode": 0}
{"kind": "model", "text": "The answer has been written to answer.txt."}
{"kind": "end"}
```

`skill2fsm collect` and `skill2fsm fold-traces` produce this format from OpenCode event streams.

### Skill rules

A `compile.json` next to `SKILL.md` declares terminals, derived labels, requirements and terminal
conditions. Every rule quotes the sentence of the skill document it comes from. Without the file, the
compiler asks the model to extract the rules and checks each quote against the document.

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

Tool definitions come from a registry (`skill2fsm/backends/opencode.json` describes OpenCode's native
tools) or are inferred from the traces. The compiler treats tool names as opaque identifiers. At run
time a machine may only use tools that the backend provides, or tools defined in a `--tools` registry,
which the model then realizes as shell commands.

## Command-line interface

| Command | Purpose |
|---|---|
| `skill2fsm compile` | Initialize a machine from a skill document (model) and update it over traces (no model) |
| `skill2fsm compile-stepwise` | Update a machine trace by trace with step decisions supplied by a judge |
| `skill2fsm run` | Execute a machine on one task (`xlsx`, `livemath` or `filetask` mode) |
| `skill2fsm collect` | Execute a skill with OpenCode on spreadsheet tasks and save traces |
| `skill2fsm fold-traces` | Turn the skill-arm event streams of a benchmark run into traces |
| `skill2fsm bench` | Run machine and skill-execution arms on a task set in parallel |
| `skill2fsm memory` | Build Agent Workflow Memory workflows and ReasoningBank memories |
| `skill2fsm summarize` | Compare arms on shared tasks: pass rate, time, tokens, paired exact tests |

Run `skill2fsm <command> --help` for all options.

### Evaluation

`skill2fsm bench` runs arms on a task set: `fsm` (a compiled machine), `skill` (OpenCode with the skill in
the system prompt), and `awm` / `rbank` (the skill arm with Agent Workflow Memory workflows or
ReasoningBank memories placed before the prompt). Task files are YAML lists:

```yaml
- id: t1
  turns: ["Compute the mean of column x in data.csv. Write @mean[value] to answer.txt."]
  metadata:
    verifier: dabench            # file-answer graders: dabench, sealqa_judge
    answers: [[mean, "34.65"]]
    assets: [data/data.csv]      # copied into the job directory; relative to --data-root
```

The metadata depends on the mode: `xlsx` tasks carry `init_asset`, `golden` and `answer_position`;
`livemath` tasks carry `answer`; `filetask` tasks carry `verifier` with `answers` (`dabench`) or
`answer` (`sealqa_judge`, graded afterwards by a judge) and either `assets` or `assets_dir` with
`assets_target`.

```bash
skill2fsm bench --mode filetask --tasks-file tasks.yaml --tasks t1,t2 --reps 1 \
    --arms fsm,skill --machine build/machine.json --skill path/to/skill --out runs/test

# memory baselines from a development run, frozen for the test run
skill2fsm memory convert runs/dev dev_traj.jsonl
skill2fsm memory awm dev_traj.jsonl workflows.txt
skill2fsm memory rbank dev_traj.jsonl bank.jsonl
skill2fsm memory index bank.jsonl
skill2fsm memory precompute bank.jsonl tasks.yaml test_ids.txt rbank.json --mode filetask
skill2fsm bench --mode filetask --tasks-file tasks.yaml --tasks t1,t2 --reps 1 \
    --arms awm,rbank --awm-file workflows.txt --rbank rbank.json --skill path/to/skill --out runs/test

skill2fsm summarize table.md "Test split" fsm=runs/test skill=runs/test awm=runs/test rbank=runs/test
```

## Python API

| Module | Contents |
|---|---|
| `skill2fsm.schema` | `Machine`, `State`, actions, `Transition`, `Variable`, `Trace`, `load_machine` |
| `skill2fsm.runtime` | `run_task`: execute a machine with a model and tools |
| `skill2fsm.fsm` | Compiler: `context.build_context`, `init.initialize`, `traces.load_traces`, `update.update`, `stepwise` |
| `skill2fsm.llm_client` | OpenAI-compatible client (`client_from_env`) and `ModelAdapter` |
| `skill2fsm.opencode_tools`, `skill2fsm.local_tools` | Tool backends |
| `skill2fsm.evaluators` | Graders |

Code comments and some diagnostic messages are in Chinese. A few modules from earlier iterations of the
compiler (such as `compile_agent`, `compiler` and `checker`) are still used internally.

## Development

```bash
pip install -e ".[dev]"
pytest
python -m build
twine check dist/*
```

## License

MIT (see `LICENSE`).
