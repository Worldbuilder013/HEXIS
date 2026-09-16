# Third-party notices

hexis is released under the MIT License (see `LICENSE`), except for the portions listed below, which
are derived from third-party work and remain under the licenses of that work.

## SpreadsheetBench

- Source: SpreadsheetBench, <https://github.com/RUCKBReasoning/SpreadsheetBench> (`evaluation.py` and
  `open_spreadsheet.py`).
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0),
  <https://creativecommons.org/licenses/by-sa/4.0/>.
- Used in `src/hexis/evaluators/spreadsheet_golden.py`: the cell-value normalization and comparison
  (`_datetime_to_float`, `_transform_value`, `compare_cell_value`), the column and range helpers
  (`_col_num2name`, `_col_name2num`, `_parse_cell_range`, `cell_names`) and the LibreOffice command line
  used to recalculate workbooks before grading.
- Changes: type annotations; helpers renamed; booleans are no longer normalized as numbers; range parsing
  rewritten with the same results. These portions are distributed under CC BY-SA 4.0.

## Agent Workflow Memory

- Source: Z. Z. Wang, J. Mao, D. Fried and G. Neubig, *Agent Workflow Memory*, arXiv:2409.07429;
  code at <https://github.com/zorazrw/agent-workflow-memory>.
- License: Apache License 2.0, <https://www.apache.org/licenses/LICENSE-2.0>.
- Used in `src/hexis/cli/memory.py`: `AWM_INSTRUCTION`, the offline workflow-induction instruction.
- Changes: the task description is rewritten for trajectories of shell and file tools, and the variable
  and output conventions are stated for that setting.

## ReasoningBank

`src/hexis/cli/memory.py` implements a ReasoningBank-style memory baseline: memory items with a title,
a one-sentence description and content, extracted as strategies from successful trajectories and as
lessons from failed ones, and retrieved by embedding similarity.

## Programs used at run time (not distributed)

- OpenCode, <https://opencode.ai> (MIT License): executes native tool calls and the skill-execution
  baseline. `src/hexis/tools/backends/opencode.json` lists its tool and argument names for interoperability.
- LibreOffice: recalculates spreadsheets before grading.

## Python dependencies (installed from PyPI, not bundled)

| Package | License |
|---|---|
| pydantic, PyYAML | MIT |
| httpx | BSD-3-Clause (its dependency certifi is MPL-2.0) |
| openpyxl (extra `xlsx`) | MIT |
| numpy (extra `memory`) | BSD-3-Clause |
| sentence-transformers (extra `memory`) | Apache-2.0 |
| pytest, build (extra `dev`) | MIT |
| twine (extra `dev`) | Apache-2.0 |
