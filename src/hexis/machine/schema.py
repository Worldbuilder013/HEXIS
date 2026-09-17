"""Data definitions for EFSM artifacts and traces.

A skill is idealized as a function ``f: E* → A∪Z``: it reads a history (the actions taken so far and their results)
and yields the next action, or ends in some way. Turning that function into something people can read and programs
can run gives an **extended finite state machine**: finite control (the state ``q``) carries "which step we are at",
and typed variables (``ν``) carry data (a repair count ranging over 0..∞ does not fit into a plain automaton, hence
*extended*).

This file holds only the data model: no execution and no model calls. The field shapes follow the machine.json /
trace JSONL formats: `clause` (which clause a state belongs to), `action` (who performs this step), `transitions`
embedded in states (guarded out-edges), `variables` (with init_from), the `FALLBACK` state, and the judge action's
`error_rate`/`support`. All of these fields are **auditable**: dump a whole machine and a person can check, entry by
entry, which trace and which clause each step was learned from.

The syntax and evaluation of guard expressions (``Transition.if``) live in :mod:`hexis.machine.cond`; here they are
stored only as strings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

MACHINE_FILE = "machine.json"

#: Reserved identifier of the fallback state. Entering it = abandoning the compiled path and going back to "the model
#: reads the whole document and interprets it", while the trace keeps being recorded as usual. Values no guard covers,
#: judge abstentions and repeated validation failures all end up here. See runtime.run_task.
FALLBACK = "FALLBACK"

#: Abstain label of judge actions: a judge that cannot decide returns it.
ABSTAIN = "abstain"
#: Abstain label written by earlier versions; machines that use it keep working.
LEGACY_ABSTAIN = "弃权"
ABSTAIN_LABELS = (ABSTAIN, LEGACY_ABSTAIN)


def abstain_label(labels: Any) -> str:
    """The last label in ``labels`` that is a known abstain label, or an empty string."""
    return next((lab for lab in reversed(list(labels or ())) if lab in ABSTAIN_LABELS), "")

VarType = Literal["string", "integer", "number", "boolean", "array", "object"]


# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #
class Variable(BaseModel):
    """A typed variable. Its initial value comes from ``init`` (a literal) or ``init_from`` (a field of the task input).

    ``init_from`` looks like ``"task.input.path"``: when the machine starts, that field is taken from the task input.
    Counter variables (repair count) use ``init: 0`` plus ``inc`` on a back edge.
    """

    name: str
    type: VarType = "string"
    init: Optional[Any] = None
    init_from: Optional[str] = None

    @model_validator(mode="after")
    def _init_xor(self) -> "Variable":
        if self.init is not None and self.init_from is not None:
            raise ValueError(f"variable {self.name}: give only one of init and init_from")
        return self


# --------------------------------------------------------------------------- #
# Actions: who performs a state's step
# --------------------------------------------------------------------------- #
class ToolAction(BaseModel):
    """One tool call. ``input`` is a parameter template whose values may contain ``${var}`` placeholders filled at run time."""

    kind: Literal["tool"] = "tool"
    name: str
    input: dict = Field(default_factory=dict)
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    #: **Phase** (probe / apply / verify / other). Only for **generic** tools such as ``bash``/``run_python``: they
    #: share a name but serve different purposes, and without refinement they would collapse into a single state.
    #: Derived **at collection time** from the command text by the pure functions in :mod:`hexis.traces.phases` (at
    #: compile time the text has already been moved into artifacts and can no longer be read). When non-empty it takes
    #: part in the KEY of ``canon_action``, symmetrically on both sides, and replay works unchanged.
    #: Dedicated tools (whose name is their purpose) leave it empty and behave exactly as before.
    phase: str = ""
    #: Derived labels: labels that skill rules attach to trace events (such as "read the output after modifying"),
    #: stored with the state; static checks match rule patterns against them. ``phase`` is the base label; these are
    #: the rest.
    labels: list[str] = Field(default_factory=list)
    #: **Data binding**: tool output key → semantic variable name, such as ``{"stdout": "workbook_content"}``.
    #: At run time the outputs are first renamed with it, then collected into the variable table through the
    #: ``writes`` allowlist. Without it, "read the workbook into workbook_content" in the document can never connect
    #: to the ``stdout`` that bash actually emits: ``rebuild`` takes values by name only, so a state with
    #: ``writes=["workbook_content"]`` gets nothing at all out of ``{ok, stdout, returncode}`` (observed in practice:
    #: every one of four rounds over eight tasks broke here).
    #: Bindings are determined from traces during alignment; they are a first-class product of it, not an
    #: after-the-fact patch.
    binds: dict[str, str] = Field(default_factory=dict)


class ModelAction(BaseModel):
    """A content-generating step: a **private** prompt and one model call without tools.

    Outputs are collected through the writes allowlist. The prompt lives on the private side (the compile ledger) and
    never enters the conversation sequence.
    """

    kind: Literal["model"] = "model"
    prompt: str
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    #: A generation state **introduced** by the compiler (input gate: when a tool input varies from task to task and no
    #: variable can stand in for it, a generation step is inserted before the tool). The trace has no such step, so in
    #: replay it is **zero-width**: it consumes no record, and the variables it writes take the real input of the tool
    #: record that immediately follows. At run time it is a real model call.
    introduced: bool = False
    #: An **observable** model state: what it writes is deliverable content (summary, answer, report), and it
    #: corresponds to a model-output event in the trace, which alignment and replay treat as a primary state. Default
    #: False = intermediate generation (generating inputs, drafting a plan), zero-width.
    observable: bool = False
    #: Derived labels (same convention as ToolAction.phase); skill rules use them to match model states.
    labels: list[str] = Field(default_factory=list)


class Example(BaseModel):
    """One calibration example for a judge action: the ``reads`` values (carried as extras) plus its true ``label``.

    Examples **come from traces**: variable snapshots of the traces on both sides of a branch point, not invented.
    """

    model_config = ConfigDict(extra="allow")
    label: str


class JudgeAction(BaseModel):
    """Judge action: for semantic decisions that cannot become a deterministic guard, e.g. is this header well-formed.

    One fixed question, with the answer restricted to ``labels`` (**which must include the abstain label**), written
    into the variable declared in ``writes``; later guards look only at that variable. The error rate ``error_rate``
    can be calibrated (see compiler.calibrate): run this judgment offline on the variable snapshots at the branch point
    and compare with the direction actually taken. Abstaining is the release valve of the inequality "the probability
    that a path errs at least once ≤ Σεᵢ": when unsure, abstain and go to FALLBACK instead of forcing an answer.
    """

    kind: Literal["judge"] = "judge"
    #: The prompt sent to the model. Matches the field of the same name on :class:`ModelAction`: both are "what this
    #: step sends to the model". Older files call it ``question``; that name is still accepted on load.
    prompt: str = Field(validation_alias=AliasChoices("prompt", "question"))
    reads: list[str]
    writes: list[str]
    labels: list[str]
    abstain: str = ABSTAIN
    examples: list[Example] = Field(default_factory=list)
    error_rate: float = 0.0
    support: int = 0
    #: **Introduced from the document** by the compiler (the traces have no such judge step). In replay it is
    #: zero-width: it consumes no trace record, and its label is computed on the spot by the program labeler named in
    #: ``gold_from``.
    introduced: bool = False
    #: Name of a **program** labeler registered in trace_adapter.LABELERS. Empty string = no program can give it a gold
    #: label, so it cannot be calibrated, ``support`` stays 0, and it may only carry a default edge to FALLBACK.
    gold_from: str = ""

    @model_validator(mode="before")
    @classmethod
    def _default_abstain(cls, data: Any) -> Any:
        """Without an explicit abstain label, use the abstain label found in ``labels`` (the older or the current one)."""
        if isinstance(data, dict) and not data.get("abstain"):
            data = {**data, "abstain": abstain_label(data.get("labels")) or ABSTAIN}
        return data

    @model_validator(mode="after")
    def _abstain_in_labels(self) -> "JudgeAction":
        if self.abstain not in self.labels:
            raise ValueError(f"the abstain label {self.abstain!r} of a judge action must be in labels")
        if not self.reads or not self.writes:
            raise ValueError("a judge action needs non-empty reads and writes")
        return self


class UserAction(BaseModel):
    """Ask the user once. Outputs are collected through the writes allowlist."""

    kind: Literal["user"] = "user"
    prompt: str = ""
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class EndAction(BaseModel):
    """End action: reaching it halts the machine; ``terminal`` refers to an entry of machine.terminals."""

    kind: Literal["end"] = "end"
    terminal: str


Action = Annotated[
    Union[ToolAction, ModelAction, JudgeAction, UserAction, EndAction],
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- #
# States, transitions, machine
# --------------------------------------------------------------------------- #
class Transition(BaseModel):
    """A guarded out-edge. Empty ``if`` = default edge (evaluated **last** within the state).

    The JSON key is ``if`` (a Python keyword, so the field is named ``cond`` and mapped with an alias); ``to`` is the
    target state. A non-empty ``inc`` means taking this edge increments that counter variable by 1 (repair loops rely
    on it). ``support`` records how many traces took this edge.
    """

    model_config = ConfigDict(populate_by_name=True)

    cond: str = Field(default="", alias="if")
    to: str
    inc: Optional[str] = None
    support: int = 0
    #: Why this edge exists: document / trace / compiler / harness (pending calibration) ... (empty string = not
    #: recorded). Only a short tag that can be dumped with the machine is kept here.
    origin: str = ""


class State(BaseModel):
    """A state: **one** action plus guarded out-edges. ``clause`` is the id of the document clause it belongs to.

    State = an equivalence class of histories (Myhill-Nerode). ``id`` is assigned by the compiler (``s1``/``s2``) and
    carries no meaning; the meaning is in the original document sentence that ``clause`` traces back to.
    """

    id: str
    clause: str = ""
    action: Action
    transitions: list[Transition] = Field(default_factory=list)
    #: Why this state exists (same as Transition.origin); ``locator`` is the position of the clause's original sentence
    #: (``SKILL.md:14``), copied from the clause table, never free text.
    origin: str = ""
    locator: str = ""

    def ordered_transitions(self) -> list[Transition]:
        """Guarded edges first, the default edge (empty ``cond``) always last. The order is the priority."""
        guarded = [t for t in self.transitions if t.cond]
        fallback = [t for t in self.transitions if not t.cond]
        return guarded + fallback


class Terminal(BaseModel):
    """One way to end. ``output`` is the keys filtered according to the interface declaration on arrival.

    ``kind`` is the **category** of this terminal, empty string by default (no claim). ``id`` is only an identifier
    (``done``, ``END_UNVERIFIED``); the category is the semantics a grading program can read: a math machine has to
    distinguish "submit after verification passed" (``kind="verified"``) from "budget exhausted, submit marked as
    unverified" (``kind="unverified"``). Both are "finished", but only the former **claims** that the result was
    verified, so a prohibition such as "verification must run before submitting" should govern only the former (see
    ``only_when`` in judge._require_before).
    """

    id: str
    kind: str = ""
    output: list[str] = Field(default_factory=list)


class Prohibition(BaseModel):
    """A prohibition (marked by hand). Checked on the trace during grading; a violation rejects even a correct result.

    The five forms of ``check`` / ``pattern``:

    * ``absent`` -- ``pattern`` (a string) must not appear in the input/output text of any action.
    * ``present`` -- it must appear.
    * ``regex`` -- a regex matching any action text is a violation.
    * ``forbid_action`` -- ``pattern`` is a dict; matching "some action + a relation between variables" is a
      violation, for example ``{"name":"export","equal":["input.output_path","input.source_path"]}`` means "the
      export target equals the source file (overwriting the original)". This is the shape of a structured prohibition.
    * ``require_before`` -- ``pattern`` is a dict: an ordering requirement on the **event stream**: before a given
      action appears, one of the actions in ``requires`` must already have appeared, otherwise it is a violation. A
      math skill's verification rule ("every non-trivial result must be independently verified at least once") has
      this shape:

      .. code-block:: python

          {"action": "submit_answer",
           "requires": ["math_verify", "run_python"],
           "only_when": {"terminal_kind": "verified"},
           "clause": "RV.0.1", "quote": "<original sentence from the skill document>"}

      For how the semantics relate to terminal categories see :func:`hexis.traces.judge._require_before`.
      ``clause``/``quote`` are extra keys for provenance; the check itself does not read them (pattern is ``Any``, and
      extra keys are ignored).
    """

    id: str
    check: Literal["absent", "present", "regex", "forbid_action", "require_before"]
    pattern: Any


class Thresholds(BaseModel):
    """A set of compile-time thresholds. See the fields for the defaults.

    ``judge_err_max`` is the **upper bound on a judge action's calibrated error rate**: if the
    ``JudgeAction.error_rate`` computed by calibration (compiler.calibrate) exceeds it, the judgment should not stay in
    the machine (rewrite the question, or fall back to FALLBACK). It is the ceiling on each εᵢ in the inequality "the
    probability that a path errs at least once ≤ Σεᵢ".
    """

    min_support: int = 2
    holdout_ratio: float = 0.2
    acc_thr: float = 0.9
    retry_budget: int = 3
    loop_margin: float = 1.5
    fallback_rate_target: float = 0.15
    judge_rewrite_max: int = 2
    judge_err_max: float = 0.2


class Machine(BaseModel):
    """A compiled extended finite state machine. The runtime never places the whole machine in a model's context.

    ``format`` is self-describing; loading dispatches on it. ``fallback`` points at the reserved fallback state.
    ``initial`` is the start state.
    """

    format: Literal["efsm-v1"] = "efsm-v1"
    skill_id: str
    version: str = "0.1.0"
    initial: str
    fallback: str = FALLBACK
    max_steps: int = 24
    states: dict[str, State] = Field(default_factory=dict)
    variables: list[Variable] = Field(default_factory=list)
    terminals: list[Terminal] = Field(default_factory=list)
    prohibitions: list[Prohibition] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    #: Canonical names of the **program-executed audit tools** (``math_verify`` / ``audit_workbook``). Every terminal
    #: with ``kind="verified"`` is meant to be reachable only through one of them, and at
    #: run time these names must have real executors in the tool table. They come from the skill configuration / the
    #: ``requires`` of the verification prohibition, never from the model. Empty list = this machine claims no
    #: "verified" way of ending.
    audit_tools: list[str] = Field(default_factory=list)
    #: Which set of **phase classification rules** this machine's tool states use to determine their identity (a key
    #: of :data:`hexis.traces.phases.CLASSIFIERS`). Self-describing: different rules make a different machine, so a
    #: silent mismatch such as "compiled with rules A, run with rules B" cannot happen.
    #: Empty string = no phase refinement (skills with dedicated tools do not need it).
    phase_rules: str = ""

    # ---- convenience accessors ---- #
    def state(self, sid: str) -> Optional[State]:
        return self.states.get(sid)

    def out_edges(self, sid: str) -> list[Transition]:
        st = self.states.get(sid)
        return st.ordered_transitions() if st else []

    def transitions_all(self) -> list[tuple[str, Transition]]:
        """All edges as ``(source state id, edge)`` pairs, for graph algorithms to traverse."""
        return [(sid, t) for sid, s in self.states.items() for t in s.transitions]

    def var(self, name: str) -> Optional[Variable]:
        return next((v for v in self.variables if v.name == name), None)

    def terminal_ids(self) -> set[str]:
        return {t.id for t in self.terminals}

    def is_counter(self, name: str) -> bool:
        v = self.var(name)
        return bool(v and v.type == "integer")

    def n_states(self) -> int:
        """Number of states for complexity purposes: end states are not counted."""
        return sum(1 for s in self.states.values() if s.action.kind != "end")

    def initial_values(self, task_input: dict) -> dict:
        """Compute the machine's starting working state from the variable table; ``init_from`` reads ``task_input``."""
        vals: dict = {}
        for v in self.variables:
            if v.init_from:
                # of the form "task.input.path": take the path after input
                key = v.init_from.split(".")[-1]
                if key in task_input:
                    vals[v.name] = task_input[key]
            elif v.init is not None:
                vals[v.name] = v.init
        return vals


# --------------------------------------------------------------------------- #
# Traces: the complete record of one execution
# --------------------------------------------------------------------------- #
class Record(BaseModel):
    """One step of a trace. ``action`` is the action executed (``{kind,name?,input?,prompt?...}``),
    ``output`` is what it produced (tool results are filled in by the host), and ``vars`` holds all variable values
    after this step.

    ``meta`` holds **execution-side bookkeeping unrelated to semantics**: this step's token counts, latency, the argv
    actually executed, the model id. It takes no part in normalization (normalize looks only at action/output) or in
    grading; it is purely what the experiment report aggregates. It is a separate field rather than being stuffed into
    ``output`` because ``output`` is collected through the writes allowlist and goes into canon_output; mixing it in
    would pollute state identity.
    """

    step: int
    state: str = ""
    clause: str = ""
    action: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    vars: dict = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)


class Trace(BaseModel):
    """The trace of one execution plus a header (task input, grading result, provenance of this run).

    ``verdict`` is filled in by grading (accepted/rejected); rejected always carries ``error_step`` (the first
    position that deviates from correct behavior), which anchors the exclusion check for rejected traces.

    ``arm``/``run``/``model``/``harness`` are **run-level provenance**, all optional: the report of a three-arm
    experiment must be able to answer, for every trace, "which arm, which repetition, which model endpoint, which
    executor produced it". They do not affect compilation or grading (compilation looks only at records and task);
    they are read only for reports and reproduction.
    """

    task: dict = Field(default_factory=dict)
    arm: str = ""
    run: int = 0
    model: str = ""
    harness: str = ""
    verdict: Literal["accepted", "rejected", "unknown"] = "unknown"
    error_step: Optional[int] = None
    records: list[Record] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rejected_has_error_step(self) -> "Trace":
        if self.verdict == "rejected" and self.error_step is None:
            raise ValueError("a rejected trace must set error_step (the first deviating position)")
        return self

    def to_jsonl(self) -> str:
        """Header on the first line, one record per remaining line.

        The header is written in the format's key order: ``{"header": true, "task_id", "arm", "run", "input", "task",
        "model", "harness", "verdict", "error_step"}``. Two deliberate choices:

        * ``task_id``/``input`` are **mirrored** from ``task`` (the format puts them at the top level of the header),
          while ``task`` is still written in full: it may also hold keys such as ``files`` that the mirrors alone would
          lose. On reading, ``task`` wins; the mirrors are used to rebuild it only when it is missing.
        * ``arm``/``run``/``model``/``harness`` are not written when they hold their default values, so the header is
          not bloated by a pile of empty strings; on reading, ``.get`` restores the same defaults.
        """
        head: dict[str, Any] = {"header": True}
        task = self.task if isinstance(self.task, dict) else {}
        if "task_id" in task:
            head["task_id"] = task["task_id"]
        if self.arm:
            head["arm"] = self.arm
        if self.run:
            head["run"] = self.run
        if "input" in task:
            head["input"] = task["input"]
        head["task"] = self.task
        if self.model:
            head["model"] = self.model
        if self.harness:
            head["harness"] = self.harness
        head["verdict"] = self.verdict
        if self.error_step is not None:
            head["error_step"] = self.error_step
        lines = [json.dumps(head, ensure_ascii=False)]
        for r in self.records:
            lines.append(json.dumps(r.model_dump(), ensure_ascii=False))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, text_or_path: Any) -> "Trace":
        """Read back from JSONL text or a file path. The first line is the header, the rest are records.

        Fully compatible with **old headers** (only ``task``/``verdict``/``error_step``, no ``header`` marker, no
        provenance fields): missing keys take their default values. Headers **written in the documented format**
        (``task_id``/``input`` at the top level, no ``task``) are accepted too: those two keys are used to rebuild
        ``task``.

        Only short strings without newlines are tried as paths, so multi-line JSONL content is never mistaken for a
        file name.
        """
        text = str(text_or_path)
        if isinstance(text_or_path, Path) or ("\n" not in text and len(text) < 4096):
            try:
                p = Path(text)
                if p.is_file():
                    text = p.read_text(encoding="utf-8")
            except OSError:
                pass
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        if not rows:
            raise ValueError("empty trace")
        head, body = rows[0], rows[1:]
        task = head.get("task")
        if not isinstance(task, dict):                  # documented shape: task_id/input at the top level
            task = {k: head[k] for k in ("task_id", "input") if k in head}
        return cls(
            task=task,
            arm=head.get("arm") or "",
            run=head.get("run") or 0,
            model=head.get("model") or "",
            harness=head.get("harness") or "",
            verdict=head.get("verdict", "unknown"),
            error_step=head.get("error_step"),
            records=[Record(**r) for r in body],
        )


# --------------------------------------------------------------------------- #
# Empty machine: the foundation of incremental construction (initially everything falls back)
# --------------------------------------------------------------------------- #
def empty_machine(skill_id: str) -> Machine:
    """A valid machine that has learned nothing: the start goes straight to FALLBACK (interpretation).

    It runs and passes structural checks.

    Compilation starts here: every trace learned makes one small, checkable change on top of it. At this point every
    trace is "trivially replayed" by it (because everything is handed to FALLBACK for interpretation), which is exactly
    the foundation state required: initially everything falls back.
    """
    return Machine(
        skill_id=skill_id,
        initial=FALLBACK,
        states={FALLBACK: State(id=FALLBACK, action=EndAction(terminal="done"))},
        terminals=[Terminal(id="done")],
    )


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def save_machine(machine: Machine, root: Any) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MACHINE_FILE
    path.write_text(
        json.dumps(json.loads(machine.model_dump_json(by_alias=True)),
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return path


def load_machine(root: Any) -> Machine:
    p = Path(root)
    if p.is_dir():
        p = p / MACHINE_FILE
    if not p.is_file():
        raise FileNotFoundError(f"no machine definition: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("format") != "efsm-v1":
        raise ValueError(f"{p} is not an efsm-v1 machine (format={data.get('format')!r})")
    return Machine.model_validate(data)
