# Example machines

Four machines in the `efsm-v1` format, compiled from skills for four task families. Each directory holds the
machine (`machine.json`) and its usage guide (`GUIDE.md`, written by `hexis-agent guide`). The directory names
follow the benchmarks the skills were written for.

| Directory | `skill_id` | Task | Inputs | Tools | States | Transitions | Step limit |
|---|---|---|---|---|---|---|---|
| [`dabench/`](dabench/) | `dabench` | answer a data-analysis question about one data file with a computation | `request`, `data_path`, `work_dir`, `output_path` | `bash`, `read` | 15 | 29 | 60 |
| [`livemath/`](livemath/) | `livemath` | answer a theorem-grounded multiple-choice mathematics question | `request`, `output_path` | `bash`, `read` | 12 | 19 | 40 |
| [`sealqa/`](sealqa/) | `sealqa` | answer a question from a local corpus of page files, with evidence | `request`, `docs_dir`, `work_dir`, `output_path` | `bash`, `read` | 21 | 39 | 80 |
| [`spreadsheet/`](spreadsheet/) | `structure-preserving-spreadsheet-edits` | edit a workbook without changing its structure | `request`, `input_path`, `output_path` | `bash` | 17 | 35 | 67 |

## What the machines have in common

- **Generate, execute, check.** A `model` state writes a shell command into a variable, a `tool` state runs it,
  and the transitions branch on `returncode`. A failed command goes back to the generating state with a loop
  counter incremented (`inc`); when the counter reaches its bound the machine moves on or falls back.
- **Verified and unverified terminals.** Every machine writes its answer to `output_path`, reads the file back
  with the `read` tool (or a verification program) and lets a `judge` state decide `pass` / `wrong_content`.
  Only a passing verdict reaches `END_VERIFIED`; after the allowed repairs the machine reports honestly through
  `END_UNVERIFIED`.
- **Bounded loops.** Every cycle carries a counter (`inspect_count`, `code_count`, `repair_count`, ...) and a
  guard that leaves the cycle when the counter reaches its bound, so the structural checks can prove termination.
- **Fallback.** Exhausted budgets lead to `FALLBACK`, where the runtime retries from the most recent tool step and
  then hands the task to interpreted execution of the skill document.

## The four flows

**dabench** inspects the data file (`s1`/`s2`, at most 3 attempts), designs a plan (`s3`), implements and runs the
computation (`s4`/`s5`, at most 4 attempts), reviews the output (`s6`: `pass` / `revise_code` / `revise_plan`),
writes the answer lines to the output file (`s7`/`s8`), reads the file back (`s9`) and judges it (`s10`).

**livemath** analyses the question (`s1`), decides the option letter and its justification (`s2`), checks once
whether a "none of the above" style option was chosen for the right reason (`s2m`), writes exactly one `\boxed{X}`
line (`s3`/`s4`), reads it back (`s5`) and judges it (`s6`).

**sealqa** classifies the question and rewrites its target as a schema (`s1`), indexes the local pages (`s2`), then
loops over opening pages and searching them (`s3` to `s5b`, at most 8 pages), derives the answer with a checked
calculation (`s6`/`s7`), audits it (`s8`: `pass` / `revise_answer` / `more_evidence`), writes the final line
(`s9`/`s10`), reads it back (`s11`) and judges it (`s12`).

**spreadsheet** inspects the workbook (`s1`/`s2`), derives the boundary of the edit (`s3`) and judges whether the
evidence is sufficient (`s4`), plans and applies the edit (`s5`/`s6`, with `s6_loop` / `s6_j` deciding whether to
run again or continue), verifies the saved workbook with a program written for it (`s7g`/`s7`/`s7_loop`), audits
the report (`s8`: `pass` / `wrong_cells` / `boundary_wrong`) and reports (`s9` / `s10`).

## Using them

```bash
M="--model qwen3.6-flash --base-url https://your-endpoint/v1"     # API_KEY in the environment or .env

# read the guide (inputs, tools, Mermaid diagram, every state and transition)
cat examples/machines/livemath/GUIDE.md

# run a machine on one task with the local bash executor
hexis-agent run --machine examples/machines/livemath \
    --input request="$(cat question.txt)" --input output_path=answer.txt \
    --workdir work/ --executor local $M

# regenerate GUIDE.md and PROMPT.md (the system prompt for a tool-using agent)
hexis-agent guide --machine examples/machines/sealqa --embed-skill
```

From Python, a machine loads and passes the structural checks without a model:

```python
from hexis.machine.schema import load_machine
from hexis.machine.checks import structural_findings

machine = load_machine("examples/machines/dabench/machine.json")
assert structural_findings(machine) == []
```

## Notes

- Every `judge` state lists `abstain` among its labels: a judge that cannot decide abstains, and the transitions
  send an abstention back to a repair step or on to `FALLBACK` instead of forcing an answer.
- The `bash` commands assume a POSIX shell and `python3` on `PATH`; `dabench` and `sealqa` also expect the task to
  provide a writable `work_dir`.
- These machines are not part of the hermetic test suite and have no skill document or traces in this repository;
  use them as references for the format and for running the runtime on real tasks.
