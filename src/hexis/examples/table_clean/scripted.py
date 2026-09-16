"""Hermetic scripts for table_clean: the judge stub, the FALLBACK interpretation script, the task generator, and a hand-written reference machine.

Nothing here is randomly nondeterministic: judgments take their "ground truth" from
:func:`~.tools.is_canonical`, the error rate is injected deterministically from a seed by
:class:`~hexis.llm.model_iface.ScriptedModel`, and task generation uses a fixed seed. The reference
machine :func:`reference_machine` is a **correct** table_clean EFSM, fed to the runner and to replay
for comparison; it is also the yardstick for "what a compiled machine should look like".
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from hexis.examples.table_clean.tools import is_canonical
from hexis.llm.model_iface import ScriptedModel
from hexis.machine.schema import (
    ABSTAIN,
    EndAction,
    JudgeAction,
    Machine,
    Prohibition,
    State,
    Terminal,
    ToolAction,
    Transition,
    Variable,
)

JUDGE_Q = "Is header_row well-formed according to the criterion in SKILL.md S2.1?"
LABELS = ["well_formed", "malformed", ABSTAIN]

_GOOD_HEADERS = [
    ["name", "quantity", "date"],
    ["product", "unit_price", "stock"],
    ["full_name", "age", "city"],
    ["order_id", "amount", "status"],
]


# --------------------------------------------------------------------------- #
# Judge stub and FALLBACK interpretation script
# --------------------------------------------------------------------------- #
def make_judge(abstain_on: Optional[Callable[[dict], bool]] = None):
    """Judge function: reads ``header_row`` and picks well_formed or malformed. Abstains when ``abstain_on`` matches."""

    def judge(prompt: str, values: dict) -> str:
        if abstain_on and abstain_on(values):
            return ABSTAIN
        return "well_formed" if is_canonical(values.get("header_row", "")) else "malformed"

    return judge


def interpret(prompt: str, values: dict, history: tuple = ()) -> dict:
    """FALLBACK interpretation script: the model reads document + history + variables and walks table_clean step by step.

    The next action is fully determined by the current variables and the history, so it is
    deterministic and repeatable. The judgment (is the header well-formed) is made **inline** by the
    model here (interpretation mode builds no separate judge state), simulated with
    :func:`is_canonical`.
    """
    fixes = sum(1 for r in history
                if (r.get("action") or {}).get("name") == "fix_header")
    exported = any((r.get("action") or {}).get("name") == "export" for r in history)
    if exported:
        return {"kind": "end", "terminal": "done"}
    if "header_row" not in values:
        return {"kind": "tool", "name": "read_csv",
                "input": {"path": values["path"]},
                "writes": ["header_row", "rows"]}
    hr = values["header_row"]
    if not is_canonical(hr) and fixes < 3:
        return {"kind": "tool", "name": "fix_header",
                "input": {"header_row": hr}, "writes": ["header_row"]}
    return {"kind": "tool", "name": "export",
            "input": {"header_row": hr, "rows": values.get("rows", []),
                      "output_path": values["output_path"],
                      "source_path": values["path"]},
            "writes": ["output_path"]}


def build_model(*, error_rate: float = 0.0, seed: int = 0,
                abstain_on: Optional[Callable[[dict], bool]] = None) -> ScriptedModel:
    """Build the hermetic model stub: judge goes through :func:`make_judge`, generate through :func:`interpret`."""
    return ScriptedModel(judge=make_judge(abstain_on), gen=interpret,
                         error_rate=error_rate, seed=seed)


# --------------------------------------------------------------------------- #
# Task generator
# --------------------------------------------------------------------------- #
def gen_tasks(n: int, *, seed: int = 0, bad_ratio: float = 0.5) -> list[dict]:
    """Produce n tasks, mixing well-formed headers (exported directly) and malformed headers (triggering 1..3 repairs).

    Each task carries its own in-memory files (``files``), and ``seed`` makes generation
    deterministic. ``output_path`` never equals ``path`` (the happy path); tasks that violate P1 are
    built separately by the tests.
    """
    rng = random.Random(seed)
    tasks: list[dict] = []
    for i in range(n):
        good = list(rng.choice(_GOOD_HEADERS))
        make_bad = rng.random() < bad_ratio
        header = list(good)
        if make_bad:
            k = rng.randint(1, min(3, len(header)))
            for j in rng.sample(range(len(header)), k):
                header[j] = "" if rng.random() < 0.5 else f"Unnamed: {j}"
        rows = [[f"v{i}{c}" for c in range(len(good))]]
        path, out = f"in{i}.csv", f"out{i}.csv"
        tasks.append({
            "task_id": f"t{i}",
            "input": {"path": path, "output_path": out,
                      "request": "Clean this table and export it"},
            "files": {path: {"header": header, "rows": rows}},
            "acceptance": {"kind": "script"},
            "_expected_header": good,   # for acceptance comparison (underscore prefix = internal to tests)
        })
    return tasks


# --------------------------------------------------------------------------- #
# Hand-written reference machine (a correct table_clean EFSM)
# --------------------------------------------------------------------------- #
def reference_machine() -> Machine:
    """A hand-written, correct table_clean state machine: read -> judge -> (repair loop) -> export, with P1."""
    return Machine(
        skill_id="table-clean",
        initial="s1",
        variables=[
            Variable(name="path", type="string", init_from="task.input.path"),
            Variable(name="output_path", type="string", init_from="task.input.output_path"),
            Variable(name="request", type="string", init_from="task.input.request"),
            Variable(name="header_row", type="string"),
            Variable(name="rows", type="array"),
            Variable(name="header_ok", type="string"),
            Variable(name="fix_count", type="integer", init=0),
        ],
        states={
            "s1": State(id="s1", clause="S1",
                        action=ToolAction(name="read_csv",
                                          input={"path": "${path}"},
                                          reads=["path"],
                                          writes=["header_row", "rows"]),
                        transitions=[Transition(to="s2")]),
            "s2": State(id="s2", clause="S2.1",
                        action=JudgeAction(prompt=JUDGE_Q,
                                           reads=["header_row"],
                                           writes=["header_ok"],
                                           labels=list(LABELS)),
                        transitions=[
                            Transition(cond="header_ok == 'well_formed'", to="s4"),
                            Transition(cond="header_ok == 'malformed' and fix_count < 3",
                                       to="s3"),
                            Transition(to="FALLBACK"),
                        ]),
            "s3": State(id="s3", clause="S3",
                        action=ToolAction(name="fix_header",
                                          input={"header_row": "${header_row}"},
                                          reads=["header_row"],
                                          writes=["header_row"]),
                        transitions=[Transition(to="s2", inc="fix_count")]),
            "s4": State(id="s4", clause="S4",
                        action=ToolAction(name="export",
                                          input={"header_row": "${header_row}",
                                                 "rows": "${rows}",
                                                 "output_path": "${output_path}",
                                                 "source_path": "${path}"},
                                          reads=["header_row", "rows",
                                                 "output_path", "path"],
                                          writes=["output_path"]),
                        transitions=[Transition(to="end")]),
            "end": State(id="end", action=EndAction(terminal="done")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done", output=["output_path"])],
        prohibitions=[Prohibition(
            id="P1", check="forbid_action",
            pattern={"name": "export",
                     "equal": ["input.output_path", "input.source_path"]})],
    )
