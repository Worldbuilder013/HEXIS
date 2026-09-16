"""The **compile agent** half of Algorithm 1 ("sequential transcription compilation"). The deterministic
gatekeeper is :mod:`hexis.legacy.checker`.

:mod:`hexis.legacy.compiler` is a one-shot compiler: it computes a whole round, checks the whole round and
reverts the whole round if it fails. An **agentic** compilation is not "one round" but a series of small
proposals -- add a state, connect an edge, turn this branch into a judge action, give this loop a counter.
So this module no longer touches :class:`~hexis.machine.schema.Machine` itself: it changes the machine
**only** through the eight receipt-issuing interfaces of :class:`hexis.legacy.checker.Checker`, one receipt
per proposal, and a rejected proposal does not affect the proposals accepted before it. This module
therefore does **not import** and does not need ``save_machine``.

Where the model may and may not appear
--------------------------------------
The core claim of the method: **the compiled artifact is deterministic; the agent's nondeterminism is paid
off once, at compile time**. So the model may appear in only four places (:data:`MODEL_TOUCHPOINTS`):

(a) the semantic **new step vs repeat** decision (L6);
(b) **clause attribution** (which sentence of the document this step implements);
(c) when a branch has no learnable deterministic guard, **drafting a judge action**'s question and label
    set (L10, :func:`draft_judge`);
(d) **calibrating the error rate** of that judge action (:func:`hexis.legacy.fit.calibrate`).

Everything else stays away from the model: alignment, building states, connecting edges, closing loops,
learning branch guards, computing loop bounds, support pruning, acceptance -- all reproducible
deterministic computation. **Every model reply goes through a schema check**; a reply that cannot be
parsed or does not fit the schema is a **REJECT, not a guess**. Two consecutive rejections at the same
point (the model reply does not fit the schema, or the gatekeeper rejects the proposal) trigger
:meth:`~hexis.legacy.checker.Checker.demote_to_fallback`, and compilation moves on -- **better to compile
less than to compile wrong**.

``model=None`` must work (the hermetic self-tests take exactly this path); it then falls back to
deterministic heuristics:

* new step vs repeat -- decided by the normalized action KEY
  (:func:`hexis.traces.normalize.canon_action`, strict level);
* clause attribution -- **always left empty** (attribution is a semantic judgement; without a model we do
  not pretend to have one);
* judge actions -- **not drafted** (judge steps already present in the traces are transcribed as usual;
  that is not drafting).

Two passes, and why there must be two
-------------------------------------
Every edge-building interface of the gatekeeper (``from_support`` of ``add_state``, ``support`` of
``add_transition`` / ``close_loop``) requires **support to be given when the edge is built** -- none of the
eight interfaces can add support to an existing edge afterwards. And
:func:`hexis.legacy.verify.verify_machine` requires every non-fallback edge to have support >=
``min_support``. So "build edges while walking sequentially" would pin every edge at support=1 on the
first trace: acceptance always fails and the whole batch is rolled back. This module therefore splits
Algorithm 1 into two passes, **while the order of decisions is still the order of the traces**:

* **Pass 1, transcribe (**:func:`transcribe`**)** -- walk the traces "fewest steps first", action by
  action, making all of Algorithm 1's L3-L11 **decisions** on a *ledger* (new step/repeat/branch, clause
  attribution, judge labels, visit counts, support). This pass **changes no machine**, so it needs no
  gatekeeper either: it is the agent's thinking.
* **Pass 2, post (**:func:`apply_plan`**)** -- turn the decision sequence, in original order, into
  receipt proposals carrying the final support, and hand them one by one to the gatekeeper for a ruling.
  One receipt per proposal; two rejections in a row fall back to interpreted execution at that point.

The cost, stated openly: a proposal rejected in pass 2 cannot go back and change the decisions of pass 1
(e.g. a mutual-exclusion conflict only surfaces when posting). The same escape route applies -- two
rejections in a row => ``demote_to_fallback``, and that segment falls back to interpreted execution.

Why back edges always carry a guard
-----------------------------------
``Checker.close_loop`` **rejects an unguarded back edge** when the source state already has a default
edge, and a state built by ``add_state`` naturally carries a default edge to FALLBACK -- in other words,
**no unguarded back edge can be built** through the receipt interfaces. This is not a pitfall to work
around but the intended shape: a back edge must carry its own predicate for "when to go round again", and
the default slot is left to FALLBACK (when the loop cannot continue, fall back to interpreted execution).
So this module learns, for each back edge, a predicate that is **always true on all observed snapshots**
of that state (:func:`hexis.legacy.fit.separating`, with empty ``others``); if none can be learned, the
whole loop is dropped.

Judge actions are only rewritten in place
-----------------------------------------
L10's "draft a judge action" has a hard limit on real traces: replay
(:func:`hexis.legacy.replay._action_matches`) compares actions step by step, and **inserting a judge state
out of thin air** makes the machine take one more step than the trace, so that trace can no longer be
replayed. So this module only drafts when **the step already is a judge step** (``kind == "judge"``) --
rewriting its question and label set while keeping ``writes`` unchanged, so the loose KEY stays the same
and replay still matches. A branch on a tool step whose guard cannot be learned simply cannot be learned:
the whole branch falls back to FALLBACK.

Deliverables
------------
Besides the machine, :class:`CompileResult` delivers a **coverage report** (``coverage``): per clause,
supported / thin / not reached by any trace; which structures come from the document and which the
compiler added itself (**the loop bound K is added by the compiler -- SKILL.md never states any iteration
bound**); how large the fallback surface is; and which additional traces would be most valuable.
``coverage`` is structured data that :func:`hexis.legacy.report.render` can render directly.
"""

from __future__ import annotations

import json as _json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from hexis.execution import runtime as _runtime
from hexis.legacy import checker as _checker
from hexis.legacy import compiler as _compiler
from hexis.legacy import fit as _fit
from hexis.legacy import replay as _replay
from hexis.legacy import report as _report
from hexis.legacy import verify as _verify
from hexis.machine import cond as _cond
from hexis.machine.checks import structural_findings
from hexis.machine.schema import (
    ABSTAIN,
    FALLBACK,
    JudgeAction,
    Machine,
    Thresholds,
    Trace,
    Variable,
)
from hexis.skill_loader import markdown_clauses
from hexis.traces.normalize import canon_action, canon_tool_name

__all__ = [
    "ABSTAIN", "HARNESS_PRIMITIVES", "MAX_STRIKES", "MODEL_TOUCHPOINTS", "SKELETON_EXAMPLE",
    "SKELETON_FORMAT",
    "ClauseRow", "CompileResult", "Plan", "PlanResult", "Proposal", "apply_plan",
    "clause_rows", "compile_skill", "draft_judge", "markdown_clauses", "transcribe",
]

#: The abstain label of judge actions. Consistent across the repository (see
#: :class:`hexis.machine.schema.JudgeAction`).

#: How many consecutive rejections at the same point fall back to interpreted execution.
#: **Better to compile less than to compile wrong.**
MAX_STRIKES = 2

#: The **only** places where the model may appear. A model call anywhere else contradicts the method.
#: The first four belong to the single-agent compiler; the rest were added for multi-agent
#: compilation -- they pass the same schema check and the same REJECT-not-guess rule, all contracts are
#: registered in :data:`REGISTRY`, and an agent may only ask the model through
#: :class:`TouchpointGuard` according to the registered contract.
MODEL_TOUCHPOINTS: tuple[str, ...] = (
    "new_or_repeat",        # (a) L6 new step vs repeat
    "clause_attribution",   # (b) which clause of the document this step implements
    "draft_judge",          # (c) L10 draft the question and label set of a judge action
    "calibrate_judge",      # (d) calibrate the error rate of that judge action
    "introduce_judge",      # (e) introduce a judge action from a document clause (no such step in the traces)
    "split_context",        # (f) whether the same action under two predecessor contexts is the same step
    "annotate_judge",       # (g) online collection probe: label the current snapshot with the judge's question
    "draft_skeleton",       # (h) document -> skeleton machine (first step of document-first compilation; traces calibrate it online later)
    "classify_clauses",     # (i) pipeline drafting: classify each clause of a section as "step / constraint / skip" and "who does it, which phase"
)

#: At most this many candidate labels when asking for clause attribution (putting all 269 clauses into
#: the label set makes no sense).
_CLAUSE_LABEL_CAP = 60

#: At most this many repair rounds when acceptance fails (each round demotes the "offending states" to
#: interpreted execution and verifies again).
_MAX_REPAIR = 4

_Q_NEW_OR_REPEAT = "Is this step a new step in the procedure, or a return to a step already taken earlier?"
_Q_CLAUSE = "Which clause of the skill document does this step implement? If unsure, answer abstain."

#: The **efsm-v1 format description** shown to the model in the ``draft_skeleton`` touchpoint. The
#: earlier description only stated the rules, not the shape, so the model had to guess field names
#: from the word "efsm-v1"; one wrong guess fails Machine validation, and this touchpoint gets only
#: one attempt. The example is a real machine that passes Machine validation and the structural
#: checks, so the description does not drift from the schema.
#: The **complete** set of names allowed for tool states in a machine: the two harness primitives.
#: This is not a tool library; drafting does not take any tool list.
HARNESS_PRIMITIVES: tuple[str, ...] = ("bash", "file_ops")

SKELETON_EXAMPLE: dict = {
    "format": "efsm-v1", "skill_id": "example", "initial": "s1", "fallback": "FALLBACK",
    "max_steps": 24,
    "variables": [
        {"name": "input_path", "type": "string", "init_from": "task.input.input_path"},
        {"name": "output_path", "type": "string", "init_from": "task.input.output_path"},
        {"name": "content", "type": "string", "init": ""},
        {"name": "plan", "type": "string", "init": ""},
        {"name": "apply_cmd", "type": "string", "init": ""},
        {"name": "verify_cmd", "type": "string", "init": ""},
        {"name": "plan_conf", "type": "string", "init": ""},
        {"name": "returncode", "type": "integer", "init": 0},
        {"name": "repair_count", "type": "integer", "init": 0},
    ],
    "states": {
        "s1": {"id": "s1", "clause": "S1",
               "action": {"kind": "tool", "name": "file_ops", "phase": "probe",
                          "input": {"op": "read", "path": "${input_path}"},
                          "reads": ["input_path"], "writes": ["content"]},
               "transitions": [{"if": "empty(content)", "to": "FALLBACK"},
                               {"to": "s2"}]},
        "s2": {"id": "s2", "clause": "S2",
               "action": {"kind": "model",
                          "prompt": "Following the document, draft a minimal change plan and write the shell command that applies it and the shell command that reads the result back to check it",
                          "reads": ["content", "input_path", "output_path"],
                          "writes": ["plan", "apply_cmd", "verify_cmd"]},
               "transitions": [{"if": "repair_count >= 3", "to": "s7"},
                               {"to": "s3"}]},
        "s3": {"id": "s3", "clause": "S2",
               "action": {"kind": "judge", "prompt": "Is the evidence for this plan sufficient? If unsure, answer abstain.",
                          "reads": ["plan"], "writes": ["plan_conf"],
                          "labels": ["sufficient", "insufficient", "abstain"], "abstain": "abstain"},
               "transitions": [{"if": "plan_conf == 'sufficient'", "to": "s4"},
                               {"if": "plan_conf == 'insufficient'", "to": "s2", "inc": "repair_count"},
                               {"to": "FALLBACK"}]},
        "s4": {"id": "s4", "clause": "S2",
               "action": {"kind": "tool", "name": "bash", "phase": "apply",
                          "input": {"command": "${apply_cmd}"},
                          "reads": ["apply_cmd"], "writes": ["returncode"]},
               "transitions": [{"if": "returncode != 0", "to": "s2", "inc": "repair_count"},
                               {"to": "s5"}]},
        "s5": {"id": "s5", "clause": "S3",
               "action": {"kind": "tool", "name": "bash", "phase": "verify",
                          "input": {"command": "${verify_cmd}"},
                          "reads": ["verify_cmd"], "writes": ["returncode"]},
               "transitions": [{"if": "returncode == 0", "to": "s6"},
                               {"if": "returncode != 0", "to": "s2", "inc": "repair_count"},
                               {"to": "FALLBACK"}]},
        "s6": {"id": "s6", "clause": "S3", "action": {"kind": "end", "terminal": "END_VERIFIED"}},
        "s7": {"id": "s7", "clause": "S2", "action": {"kind": "end", "terminal": "END_UNVERIFIED"}},
        "FALLBACK": {"id": "FALLBACK", "action": {"kind": "end", "terminal": "END_FALLBACK"}},
    },
    "terminals": [{"id": "END_VERIFIED", "kind": "verified"},
                  {"id": "END_UNVERIFIED", "kind": "unverified"},
                  {"id": "END_FALLBACK", "kind": "fallback"}],
    "audit_tools": ["bash"],
    "phase_rules": "default",
}

SKELETON_FORMAT = """\
Shape of efsm-v1 (a JSON object):
- Top level: format="efsm-v1", skill_id, initial (id of the start state), fallback="FALLBACK", max_steps,
  states (id -> state), variables, terminals, audit_tools.
- State: {id, clause, action, transitions, origin}. Use ids s1, s2, ...; clause is the id of the clause it implements.
  origin marks **why this state exists**: "document" = a step the document explicitly requires (e.g. "read back and check after modifying"),
  "compiler" = an implementation choice you made to turn the procedure into a machine (e.g. inserting an "is the plan good enough" judge, setting a retry bound).
  Each edge in transitions may also carry origin, with the same meaning. When the machine is later corrected with real traces, the document parts are
  constraints and must not be bypassed; the compiler parts may be changed. If unsure, write document.
- action has four kinds, distinguished by kind:
  tool  {name, phase, input, reads, writes}  execute one step. name can only be bash or file_ops:
        bash     input={"command": "..."}, output keys returncode / stdout;
        file_ops input={"op": "read"|"write"|"list", "path": "...", "content"?: "..."}, a read yields output key content;
        writes may use **semantic names** (e.g. workbook_content); then add binds={"stdout": "workbook_content"}
        to say which output key carries it; without binds, every name in writes must be an output key itself;
        phase is the purpose of the step: probe (read the input to see the current state) / apply (write the output) / verify (read back what you just wrote to check it).
        The parts of commands and paths that vary per task are written as "${variable}", produced by some earlier model state;
  model {prompt, reads, writes}       the model writes some content; outputs are collected according to writes;
  judge {prompt, reads, writes, labels, abstain}  a fixed question whose answer is locked to labels; labels must include "abstain",
        the answer is written to the variable in writes, and later branches read only that variable;
  end   {terminal}                    halt; terminal refers to an entry in terminals.
- transitions: [{if, to, inc}]. if is a predicate over variables, and **only** these forms are allowed:
  x == 'A', x != 'A', n >= 3, n < 3, empty(x), nonempty(x), joined with infix and / or / not,
  e.g. "returncode == 0 and empty(stdout)". These are not function calls: and(...), &&, || are all illegal;
  only names declared in variables may be referenced. An edge without if is the default edge: at most one per state, evaluated last.
  The outgoing edges of a state must be pairwise mutually exclusive and cover all values. inc is the counter variable incremented by 1 when the edge is taken.
- A back edge (an edge that can loop back) must carry inc, and the state it loops back to must have an exit edge "counter >= K".
- variables: [{name, type, init | init_from}], type in string/integer/number/boolean/array/object;
  task inputs use init_from="task.input.<key>", counter variables use init=0. Every variable a state reads must be written first on every
  path leading to that state.
- terminals: [{id, kind}], kind in verified / unverified / fallback. There must be a state with id FALLBACK whose
  action is end and points to a terminal with kind=fallback; any state that is unsure may have an edge to FALLBACK.
- audit_tools: ["bash"]. A terminal with kind=verified can only be reached via a bash state with phase=verify, and the if of that edge
  must read the returncode that state writes.
- phase_rules: "default".
Minimal example:
"""

_JUDGE_PROMPT = """\
You are compiling a skill document into a state machine. A branch appears after state {state}: after the same step,
execution sometimes goes to {targets}, and these branches **cannot** be separated by a deterministic predicate over the existing variables.

Draft one **fixed question** whose answer decides which branch to take. Requirements:
1. read only these variables: {reads};
2. the answer is locked to a finite label set, one label per branch, plus an abstain label {abstain};
3. reply with a JSON object: {{"prompt": "...", "labels": ["...", ...], "abstain": "{abstain}"}}.

Sample variable snapshots from both sides of the branch:
{samples}
"""


# --------------------------------------------------------------------------- #
# Touchpoint registry: per touchpoint, the question, reads whitelist, reply schema, REJECT rules,
# and model=None fallback
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Touchpoint:
    """The **complete** contract of one model touchpoint. An agent may only ask the model through
    :class:`TouchpointGuard` according to it.

    ``reads`` is the whitelist of value keys that may be passed to the model (intersection rule, same as
    ``want = [...]`` in draft_judge); ``parse(raw, values) -> dict | None`` performs the schema check,
    ``None`` meaning REJECT; ``fallback(values)`` is the deterministic fallback for ``model=None``
    (``None`` means "do nothing"); ``allowed_ops`` limits which receipt interfaces may appear in this
    touchpoint's proposal batch (empty = no proposals, only fields are written).
    """

    id: str
    prompt: str
    reads: frozenset
    reply_kind: str                         # "classify" | "generate"
    parse: Any
    reject_rules: tuple = ()
    fallback: Any = None
    max_strikes: int = MAX_STRIKES
    allowed_ops: frozenset = frozenset()


def _parse_in_labels(raw: Any, values: Mapping) -> Optional[dict]:
    labels = list(values.get("labels") or ())
    if not isinstance(raw, str) or raw not in labels or raw == ABSTAIN:
        return None
    return {"label": raw}


def _parse_draft_judge(raw: Any, _values: Mapping) -> Optional[dict]:
    if not isinstance(raw, Mapping):
        return None
    if not isinstance(raw.get("prompt"), str) or not raw["prompt"].strip():
        return None
    if not isinstance(raw.get("labels"), (list, tuple)) or len(raw["labels"]) < 2:
        return None
    return dict(raw)


def _parse_introduce_judge(raw: Any, values: Mapping) -> Optional[dict]:
    """``{clause_id, prompt, labels[2..9], abstain?, reads?, label_locators{label: locator}}``.

    REJECT: not a Mapping; empty prompt; fewer than 2 or more than 9 labels, or duplicates;
    ``clause_id`` not in the given clause table; ``label_locators`` given but some label has no locator.
    The part of ``reads`` outside the whitelist is **discarded** (as in draft_judge: a model replying
    with a variable name outside the whitelist is sneaking extra context into the judge).
    """
    if not isinstance(raw, Mapping):
        return None
    q = raw.get("prompt")
    labs = raw.get("labels")
    cid = str(raw.get("clause_id") or raw.get("clause") or "")
    if not isinstance(q, str) or not q.strip() or not isinstance(labs, (list, tuple)):
        return None
    labs = list(dict.fromkeys(str(x) for x in labs
                              if isinstance(x, (str, int, float)) and not isinstance(x, bool)))
    labs = [l for l in labs if l != ABSTAIN]
    if len(labs) < 2 or len(labs) > 9:
        return None
    rows = set(values.get("clause_ids") or ())
    if rows and cid not in rows:
        return None
    locs = raw.get("label_locators")
    if locs is not None and not isinstance(locs, Mapping):
        return None
    if isinstance(locs, Mapping) and any(l not in locs for l in labs):
        return None
    want = list(values.get("reads") or ())
    reads = raw.get("reads")
    if reads is not None:
        if not isinstance(reads, (list, tuple)):
            return None
        reads = [r for r in reads if r in set(want)]
    return {"clause": cid, "prompt": q.strip(), "labels": labs,
            "abstain": ABSTAIN, "reads": list(reads) if reads else want,
            "label_locators": dict(locs or {})}


_ROLES = ("step", "constraint", "skip")
_KINDS = ("tool", "model", "judge")
_PHASES = ("probe", "apply", "verify")


def _parse_classify_clauses(raw: Any, values: Mapping) -> Optional[dict]:
    """``{"items": [{clause_id, role, kind?, phase?, summary?}, ...]}``.

    Items whose id is not in the clause table are dropped; items whose role is not one of the three
    values are dropped; items with role=step but no kind, or kind=tool but no phase, are dropped. No
    valid item at all => REJECT. Clauses not mentioned are filled in by the pipeline as "constraint".
    """
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except ValueError:
            return None
    if not isinstance(raw, Mapping):
        return None
    items = raw.get("items")
    if not isinstance(items, (list, tuple)):
        return None
    ids = set(values.get("clause_ids") or ())
    out = []
    for it in items:
        if not isinstance(it, Mapping):
            continue
        cid = str(it.get("clause_id") or it.get("id") or "")
        role = str(it.get("role") or "").strip().lower()
        kind = str(it.get("kind") or "").strip().lower()
        phase = str(it.get("phase") or "").strip().lower()
        if (ids and cid not in ids) or role not in _ROLES:
            continue
        if role == "step":
            if kind not in _KINDS:
                continue
            if kind == "tool" and phase not in _PHASES:
                continue
        out.append({"clause_id": cid, "role": role, "kind": kind if role == "step" else "",
                    "phase": phase if (role == "step" and kind == "tool") else "",
                    "summary": str(it.get("summary") or "")[:200]})
    return {"items": out} if out else None


def _parse_split_context(raw: Any, _values: Mapping) -> Optional[dict]:
    if not isinstance(raw, str) or raw not in ("same step", "different step", ABSTAIN):
        return None
    return {"label": raw}


REGISTRY: dict[str, Touchpoint] = {
    "new_or_repeat": Touchpoint(
        "new_or_repeat", _Q_NEW_OR_REPEAT,
        frozenset({"current position", "this step", "existing states of the same kind", "labels"}), "classify",
        _parse_in_labels, ("reply not in label set", ABSTAIN), None, MAX_STRIKES,
        frozenset({"add_state", "add_transition", "close_loop", "set_terminal"})),
    "clause_attribution": Touchpoint(
        "clause_attribution", _Q_CLAUSE, frozenset({"this step", "candidate clauses", "labels"}),
        "classify", _parse_in_labels, ("reply not among the candidate clauses",),
        lambda v: {"label": ""}, MAX_STRIKES, frozenset()),
    "draft_judge": Touchpoint(
        "draft_judge", _JUDGE_PROMPT,
        frozenset({"state", "reads", "targets", "abstain", "samples", "writes", "clause"}),
        "generate", _parse_draft_judge, ("not a Mapping / empty prompt / fewer than 2 labels",),
        None, MAX_STRIKES, frozenset({"add_judge"})),
    "calibrate_judge": Touchpoint(
        "calibrate_judge", "(the judge action's own question)", frozenset(), "classify",
        lambda raw, v: ({"label": raw} if isinstance(raw, str) else None), (), None,
        MAX_STRIKES, frozenset()),
    "introduce_judge": Touchpoint(
        "introduce_judge",
        "Which sentence of the document requires a judgement at this step? Give a fixed question, a "
        "finite label set (with each label's locator in the clause's original text) and the variables "
        "it reads. Reply with a JSON object: "
        "{\"clause_id\": ..., \"prompt\": ..., \"labels\": [...], \"abstain\": \"abstain\", "
        "\"reads\": [...], \"label_locators\": {label: locator}}",
        frozenset({"state", "clause_text", "clause_ids", "reads", "prev_action",
                   "next_actions", "samples"}),
        "generate", _parse_introduce_judge,
        ("clause_id not in clause table", "labels <2 or >9", "label without a source locator",
         "reads outside the whitelist"),
        None, 1, frozenset({"add_judge"})),
    "split_context": Touchpoint(
        "split_context",
        "Is the same action under these two predecessor contexts the same step? Reply "
        "\"same step\" / \"different step\" / \"abstain\".",
        frozenset({"action", "pred_a", "pred_b", "sample_vars", "labels"}), "classify",
        _parse_split_context, ("not one of the three labels",), None, MAX_STRIKES,
        frozenset({"split_state"})),
    "annotate_judge": Touchpoint(
        "annotate_judge", "(the judge action's own question)", frozenset(), "classify",
        _parse_in_labels, ("not in label set => write the abstain label",), None, MAX_STRIKES, frozenset()),
    "draft_skeleton": Touchpoint(
        "draft_skeleton",
        "Transcribe this skill document into an efsm-v1 state machine (JSON). Rules: attach one clause "
        "to each state (clause uses the given clause ids); every tool input value that varies per task "
        "is written as ${variable}, and some earlier model state must write that variable; branch on "
        "variable guards or judge actions; give back edges a counter variable and a bound; verified "
        "terminals can only be reached via audit tools. Reply with the JSON object only.\n"
        "You are not given any tool list: the name of a tool state can only be one of the two "
        "primitives bash or file_ops, and the concrete commands are written by earlier model states "
        "following the document. In VARIABLES, clauses is the clause table (id -> original text): set "
        "each state's clause to the id of the closest clause; one state may cover several clauses, "
        "there is no need for one state per clause, and do not copy the number of states in the "
        "example. Do not walk through the clauses one by one in your reasoning; once the main path is "
        "clear, write the JSON directly.\n"
        "If VARIABLES contains previous and errors: previous is your previous machine and errors are "
        "the errors the deterministic checks reported on it; fix only those errors, keep everything "
        "else unchanged, and still reply with the **complete** machine JSON.\n\n" + SKELETON_FORMAT
        + _json.dumps(SKELETON_EXAMPLE, ensure_ascii=False, indent=1),
        frozenset({"doc", "clause_ids", "clauses", "tool_names", "input_keys", "audit_tools",
                   "skill_id", "previous", "errors"}),
        "generate", None, ("not valid efsm-v1", "structural check reports an error",
                           "tool name not in the allowed set"),
        None, 1, frozenset({"open_machine", "add_state", "add_transition", "close_loop",
                            "add_judge", "set_terminal"})),
    "classify_clauses": Touchpoint(
        "classify_clauses",
        "Below is one section of a skill document; classify each clause. role is one of three: step "
        "(this clause requires performing an action), constraint (a restriction on how something is "
        "done, not a step of its own), skip (heading, background, unrelated to execution). For "
        "role=step also give kind: tool (runs a command or reads/writes files; also give phase: "
        "probe=read the input to see the current state / apply=write the output / verify=read back "
        "what you just wrote to check it), model (needs thinking, writing content or drafting a "
        "plan), judge (decides which of a few finite cases applies, with different paths "
        "afterwards). Reply in the order the clauses appear in the document. Reply with a single "
        "JSON object only: "
        "{\"items\": [{\"clause_id\": ..., \"role\": ..., \"kind\": ..., \"phase\": ..., "
        "\"summary\": \"at most ten words\"}]}",
        frozenset({"section", "clauses", "clause_ids", "primitives"}),
        "generate", _parse_classify_clauses, ("items is not a list", "no valid item"),
        None, MAX_STRIKES, frozenset()),
}
# The parse of draft_skeleton needs Machine validation; it is filled in later by its user to avoid a
# circular import.
assert tuple(REGISTRY) == MODEL_TOUCHPOINTS, "registry and MODEL_TOUCHPOINTS must match item by item"


class TouchpointGuard:
    """The **only** model handle an agent gets. Every question passes the whitelist and schema check;
    a mismatch records a strike instead of guessing.

    ``ask`` returns the parsed dict, or ``None`` on REJECT (having already recorded a strike for
    ``point`` on ``ctx``); with ``model=None`` it uses the touchpoint's ``fallback`` (``None`` when
    there is none: do nothing).
    """

    def __init__(self, model: Any, ctx: "_Ctx", registry: Optional[Mapping] = None) -> None:
        self._model = model
        self.ctx = ctx
        self.registry = dict(registry or REGISTRY)
        self.calls = 0
        self.rejects: list[dict] = []

    @property
    def has_model(self) -> bool:
        return self._model is not None

    def ask(self, tp_id: str, values: Mapping, *, point: str = "",
            labels: Sequence[str] = (), history: tuple = ()) -> Optional[dict]:
        tp = self.registry.get(tp_id)
        if tp is None:
            self.ctx.strike(point, f"touchpoint {tp_id!r} is not registered")
            return None
        vals = dict(values)
        if labels:
            vals["labels"] = list(labels)
        extra = set(vals) - set(tp.reads) - {"labels"}
        if tp.reads and extra:
            self.ctx.strike(point, f"[E_READS_WHITELIST] touchpoint {tp_id} passed values outside the "
                                   f"whitelist {sorted(extra)}")
            self.rejects.append({"touchpoint": tp_id, "why": "reads", "extra": sorted(extra)})
            return None
        if self._model is None:
            return tp.fallback(vals) if tp.fallback is not None else None
        self.calls += 1
        self.ctx.model_calls += 1
        try:
            if tp.reply_kind == "classify":
                labs = list(vals.get("labels") or [])
                raw = self._model.classify(
                    prompt=tp.prompt,
                    values={k: v for k, v in vals.items() if k != "labels"}, labels=labs)
            else:
                raw = self._model.generate(prompt=tp.prompt, values=vals, history=history)
        except Exception as exc:                                    # noqa: BLE001
            self.ctx.strike(point, f"touchpoint {tp_id} call failed: {type(exc).__name__}: {exc}")
            return None
        parsed = tp.parse(raw, vals)
        if parsed is None:
            self.ctx.strike(point, f"touchpoint {tp_id} reply does not fit the schema: {str(raw)[:120]!r}")
            self.rejects.append({"touchpoint": tp_id, "why": "schema", "raw": str(raw)[:200]})
            return None
        return parsed


# --------------------------------------------------------------------------- #
# Public data shapes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proposal:
    """One rewrite proposal from the compile agent.

    ``op`` is one of the gatekeeper's eight receipt interfaces, ``payload`` its keyword arguments,
    ``rationale`` **the justification for this proposal** (human-readable), and ``trace_id``/``step``
    point back to the trace and step it was learned from -- when auditing a machine, these three together
    answer "which trace and which document clause this edge exists because of".

    This class is not :class:`hexis.legacy.checker.Proposal`: that one is the gatekeeper's input shape
    (only ``op``/``args``); this one additionally carries provenance. :meth:`to_checker` converts.
    """

    op: str
    payload: dict = field(default_factory=dict)
    rationale: str = ""
    trace_id: str = ""
    step: int = 0
    #: Provenance row (an object with ``to_dict()`` or a dict of the same shape). Required in
    #: multi-agent mode; the old single-agent path omits it, and the gatekeeper does not require it.
    prov: Any = None

    def to_checker(self) -> _checker.Proposal:
        """Convert to the proposal shape the gatekeeper accepts. Provenance, if present, is passed to
        the receipt interface's ``prov=``."""
        args = dict(self.payload)
        if self.prov is not None and self.op not in ("commit", "mark", "rewind"):
            args["prov"] = (self.prov.to_dict() if hasattr(self.prov, "to_dict")
                            else dict(self.prov))
        return _checker.Proposal(op=self.op, args=args)

    @property
    def point(self) -> str:
        """**The point this proposal acts on** -- the one demoted to interpreted execution after two
        rejections in a row.

        For edge-building proposals the point is **the edge's source state** (compilation got stuck
        there); only when there is no source state does it fall back to the state being built.
        ``open_machine``/``commit`` do not act on any state and return an empty string.
        """
        p = self.payload
        return str(p.get("from_state") or p.get("state_id") or "")


@dataclass
class PlanResult:
    """Result of one posting pass: receipts, accepted/rejected counts, points demoted to interpreted
    execution, skipped proposals."""

    receipts: list = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    demoted: list[str] = field(default_factory=list)
    skipped: list[Proposal] = field(default_factory=list)


@dataclass
class CompileResult:
    """All deliverables of one compilation. ``machine`` is the only artifact; the rest is its ledger."""

    machine: Machine
    receipts: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    judges: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    diff_vs_reference: Optional[dict] = None


@dataclass(frozen=True)
class ClauseRow:
    """One row of the clause table. ``locator`` looks like ``SKILL.md:12-20``; empty string if absent."""

    id: str
    title: str
    text: str
    locator: str = ""


# --------------------------------------------------------------------------- #
# The ledger of pass 1
# --------------------------------------------------------------------------- #
@dataclass
class _PState:
    """A state in the ledger: one step's action plus its origin."""

    sid: str
    key: tuple
    kind: str                                   # tool / judge / model / end
    payload: dict = field(default_factory=dict)  # action dict for tool/model
    clause: str = ""
    terminal: str = ""
    prompt: str = ""
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    labels_seen: list = field(default_factory=list)
    examples: list = field(default_factory=list)   # [(label, {read: value})]
    drafted: bool = False                          # whether the judge action was drafted by the model
    error_rate: float = 0.0
    support: int = 0
    origin: tuple = ("", 0)                        # (trace_id, step)
    # ---- provenance and introduced-judge fields added for multi-agent compilation ---- #
    # ---- (empty by default: single-agent output is byte-for-byte unchanged)         ---- #
    introduced: bool = False                       # judge introduced from the document (no such step in the traces)
    gold_from: str = ""                            # programmatic labeler name (trace_adapter.LABELERS)
    origin_kind: str = ""                          # document / trace / compiler ...
    locator: str = ""                              # location of the clause's original sentence


@dataclass
class _PEdge:
    """An edge in the ledger. ``cond`` is filled in only in the second stage, "fixing guards"."""

    src: str
    dst: str
    support: int = 0
    back: bool = False
    creating: bool = False                          # this edge also creates its target state
    cond: str = ""
    counter: str = ""
    bound: int = 0
    origin: tuple = ("", 0)


@dataclass
class Plan:
    """The product of the pass-1 transcription: a compile ledger **not yet applied to any machine**.

    It is the agent's trail of thought -- states, edges, support, variable snapshots at branches,
    labels observed for judge actions, and the maximum number of times each state is entered within a
    single trace (used to set loop bounds). Pass 2 generates receipt proposals from it.
    """

    skill_id: str = "compiled"
    states: dict = field(default_factory=dict)          # sid -> _PState
    order: list = field(default_factory=list)           # state creation order
    by_key: dict = field(default_factory=lambda: defaultdict(list))
    edges: dict = field(default_factory=dict)           # (src,dst) -> _PEdge
    edge_order: list = field(default_factory=list)
    out: dict = field(default_factory=lambda: defaultdict(list))
    initial: str = ""
    obs: dict = field(default_factory=lambda: defaultdict(list))   # sid -> [(dst, snap)]
    #: Context ledger parallel to obs: sid -> [(dst, predecessor sid, number of times sid had already
    #: been visited in this trace before entering it)]. The split agent uses it to build the contingency
    #: table for "can the branch be separated by predecessor / by visit count". Kept separate instead of
    #: widening obs's pairs into 4-tuples: obs's consumers (fit_guards/_unlink/_judge_branch) all unpack pairs.
    obs_ctx: dict = field(default_factory=lambda: defaultdict(list))
    #: Where a branch originally led when it was sent to FALLBACK (recorded by _block): the split /
    #: introduce-judge agents need to know "which states this branch originally split into", and after
    #: _block out[p] is already empty.
    blocked_targets: dict = field(default_factory=dict)
    #: sid -> variable name: when fixing this state's branch guards, look **only** at this variable
    #: (used for introduced judge states, so the predicate learn_cond learns is always "judge variable
    #: == label" and not some other variable that happens to separate the branches too).
    guard_vars: dict = field(default_factory=dict)
    visits: dict = field(default_factory=lambda: defaultdict(int))
    var_types: dict = field(default_factory=dict)
    input_keys: set = field(default_factory=set)
    trace_states: dict = field(default_factory=lambda: defaultdict(list))
    blocked: set = field(default_factory=set)           # states whose branch could not be compiled; only the default edge remains
    thin: list = field(default_factory=list)            # edges pruned for insufficient support
    pruned: list = field(default_factory=list)          # states pruned because they became unreachable
    loop_bounds: list = field(default_factory=list)     # ledger of K values
    notes: list = field(default_factory=list)           # places where transcription could not proceed
    max_records: int = 0
    _n: int = 0

    # ---- helpers ---- #
    def new_sid(self) -> str:
        self._n += 1
        return f"s{self._n}"

    def reaches(self, src: str, dst: str) -> bool:
        """Whether ``dst`` is reachable from ``src`` on the ledger graph (including ``src is dst``)."""
        seen, stack = {src}, [src]
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            for nxt in self.out.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return dst in seen

    def add_edge(self, src: str, dst: str, *, back: bool, creating: bool,
                 origin: tuple) -> _PEdge:
        e = self.edges.get((src, dst))
        if e is None:
            e = _PEdge(src=src, dst=dst, back=back, creating=creating, origin=origin)
            self.edges[(src, dst)] = e
            self.edge_order.append((src, dst))
            self.out[src].append(dst)
        return e


# --------------------------------------------------------------------------- #
# Compile context (model touchpoints + counters)
# --------------------------------------------------------------------------- #
@dataclass
class _Ctx:
    model: Any = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    rows: list = field(default_factory=list)
    doc: str = ""
    progress: bool = False
    model_calls: int = 0
    strikes: dict = field(default_factory=lambda: defaultdict(int))
    rejects: list = field(default_factory=list)
    #: Hook for state identity: ``key_of(rec, prev_rec, counts) -> tuple``. Default ``None`` =
    #: ``canon_action(rec, strict=True)``. The multi-agent transcription agent uses it to realize the
    #: split agent's ``Refinement`` as "same action, different predecessor / different visit count =>
    #: different identity" without changing the transcription loop.
    #: ``counts`` is a dict of counts per base KEY within the current trace, maintained by the hook itself.
    key_of: Any = None

    def say(self, msg: str) -> None:
        if self.progress:
            print(f"[compile_agent] {msg}", file=sys.stderr)

    def strike(self, point: str, why: str) -> bool:
        """Record one rejection at a point. Returns **whether the consecutive-rejection limit has been
        reached** (time to fall back to interpreted execution)."""
        self.strikes[point] += 1
        self.rejects.append({"point": point, "why": why, "n": self.strikes[point]})
        self.say(f"point {point} rejected (#{self.strikes[point]}): {why}")
        return self.strikes[point] >= MAX_STRIKES

    def clear(self, point: str) -> None:
        self.strikes[point] = 0


# --------------------------------------------------------------------------- #
# Model touchpoints (four of them, each schema-checked; schema mismatch = REJECT, not a guess)
# --------------------------------------------------------------------------- #
def _brief(action: Mapping) -> str:
    """Short summary of a step's action, shown to the model. The full private prompt never goes in."""
    kind = str(action.get("kind") or "")
    if kind == "tool":
        return f"tool:{canon_tool_name(action.get('name') or '')}"
    if kind == "judge":
        return f"judge:{str(action.get('prompt') or '')[:60]}"
    if kind == "end":
        return f"end:{action.get('terminal') or 'done'}"
    return kind or "?"


def _ask_new_or_repeat(ctx: _Ctx, point: str, rec: Any,
                       cands: Sequence[str]) -> Optional[str]:
    """(a) New step vs repeat. Returns ``"new"`` / an existing state id; ``None`` on schema mismatch."""
    labels = ["new step"] + [f"repeat:{s}" for s in cands] + [ABSTAIN]
    values = {"current position": point, "this step": _brief(rec.action),
              "existing states of the same kind": ",".join(cands) or "(none)"}
    ctx.model_calls += 1
    try:
        ans = ctx.model.classify(prompt=_Q_NEW_OR_REPEAT, values=values,
                                 labels=list(labels))
    except Exception:                                       # noqa: BLE001
        return None
    if not isinstance(ans, str) or ans not in labels or ans == ABSTAIN:
        return None
    if ans == "new step":
        return "new"
    return ans.split(":", 1)[1]


def _clause_candidates(rec: Any, rows: Sequence[ClauseRow]) -> list[ClauseRow]:
    """Narrow the candidate clauses down to a range that fits in a label set. **This is retrieval, not
    attribution** -- attribution is the model's job."""
    act = rec.action if isinstance(rec.action, Mapping) else {}
    name = canon_tool_name(act.get("name") or "")
    hits = [r for r in rows if name and (name in r.text.lower()
                                         or name.replace("_", "-") in r.text.lower()
                                         or name.replace("_", " ") in r.text.lower())]
    pool = hits or list(rows)
    return pool[:_CLAUSE_LABEL_CAP]


def _ask_clause(ctx: _Ctx, rec: Any) -> str:
    """(b) Clause attribution. ``model=None`` or a schema mismatch always returns an empty string
    (**no pretend attribution**)."""
    if ctx.model is None or not ctx.rows:
        return ""
    pool = _clause_candidates(rec, ctx.rows)
    if not pool:
        return ""
    labels = [r.id for r in pool] + [ABSTAIN]
    values = {"this step": _brief(rec.action),
              "candidate clauses": " | ".join(f"{r.id} {r.title}" for r in pool)}
    ctx.model_calls += 1
    try:
        ans = ctx.model.classify(prompt=_Q_CLAUSE, values=values, labels=list(labels))
    except Exception:                                       # noqa: BLE001
        return ""
    if not isinstance(ans, str) or ans not in labels or ans == ABSTAIN:
        return ""
    return ans


def draft_judge(question_ctx: Mapping, *, model: Any) -> Optional[JudgeAction]:
    """(c) When a branch has no learnable deterministic guard, **the agent drafts** a judge action;
    the gatekeeper validates it afterwards.

    ``question_ctx`` must provide at least ``reads`` (whitelist of variables that may be read) and
    ``targets`` (``{target state: [variable snapshots]}``); it may provide ``state``/``writes``/``clause``.

    The three cases that return ``None`` instead of raising all mean **the same thing** -- this draft
    is void and the caller treats it as one rejection: no model; the model call blew up; the reply cannot
    be parsed or does not fit the schema (missing ``prompt``, fewer than two labels, non-scalar labels).
    **Never assemble a judge action from half a reply.**

    ``reads`` accepts only whitelisted variables: a model replying with a variable name outside the
    whitelist is sneaking extra context into this judge action, and narrow reads are exactly what this
    compilation must protect.
    """
    if model is None:
        return None
    ctx = dict(question_ctx or {})
    reads = [str(r) for r in (ctx.get("reads") or [])]
    if not reads:
        return None
    writes = [str(w) for w in (ctx.get("writes") or [])]
    if not writes:
        writes = [f"{ctx.get('state') or 'branch'}_verdict"]
    targets = list((ctx.get("targets") or {}))
    abstain = str(ctx.get("abstain") or ABSTAIN)
    prompt = _JUDGE_PROMPT.format(
        state=ctx.get("state") or "?", targets=", ".join(targets) or "?",
        reads=", ".join(reads), abstain=abstain,
        samples=_sample_text(ctx.get("targets") or {}, reads))
    try:
        raw = model.generate(prompt=prompt, values={"reads": reads, "targets": targets})
    except Exception:                                       # noqa: BLE001
        return None
    if not isinstance(raw, Mapping):
        return None
    prompt = raw.get("prompt")
    labels = raw.get("labels")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if not isinstance(labels, (list, tuple)):
        return None
    labs: list[str] = []
    for x in labels:
        if isinstance(x, (str, int, float)) and not isinstance(x, bool):
            s = str(x).strip()
            if s and s not in labs:
                labs.append(s)
    if len(labs) < 2:
        return None
    abstain = str(raw.get("abstain") or abstain)
    if abstain not in labs:
        labs.append(abstain)
    want = [r for r in (raw.get("reads") or reads) if r in reads] or reads
    try:
        return JudgeAction(prompt=prompt.strip(), reads=want, writes=writes,
                           labels=labs, abstain=abstain)
    except (ValidationError, ValueError):
        return None


def _sample_text(targets: Mapping, reads: Sequence[str]) -> str:
    lines = []
    for tgt in list(targets)[:4]:
        for snap in list(targets[tgt])[:2]:
            vals = ", ".join(f"{k}={snap.get(k)!r}" for k in reads)
            lines.append(f"  → {tgt}: {vals}")
    return "\n".join(lines) or "  (none)"


def _calibrate(ctx: _Ctx, judge: JudgeAction,
               samples: Sequence[tuple]) -> Optional[float]:
    """(d) Calibrate the error rate of a judge action. Returns ``None`` if it cannot be calibrated
    (the caller treats that as a rejection)."""
    if ctx.model is None or not samples:
        return None
    ctx.model_calls += len(samples)
    try:
        return float(_fit.calibrate(judge, list(samples), model=ctx.model))
    except Exception:                                       # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Pass 1: sequential transcription (decisions of Algorithm 1, L2-L11)
# --------------------------------------------------------------------------- #
def _tid(trace: Trace) -> str:
    task = trace.task if isinstance(trace.task, dict) else {}
    return str(task.get("task_id") or "")


def _type_of(v: Any) -> Optional[str]:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, (list, tuple)):
        return "array"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, str):
        return "string"
    return None


def _order_traces(t_plus: Sequence[Trace]) -> list[tuple[int, Trace]]:
    """L2's "fewest steps first". Ties are broken by task_id, then by original index, so **the order is
    fully deterministic**."""
    return sorted(enumerate(t_plus),
                  key=lambda it: (len(it[1].records), _tid(it[1]), it[0]))


def _create_state(plan: Plan, sid: str, rec: Any, key: tuple, prev_vars: dict,
                  ctx: _Ctx, tid: str) -> _PState:
    """L7: turn a record's action into a new state (including clause attribution and reads/writes)."""
    act = rec.action if isinstance(rec.action, Mapping) else {}
    kind = str(act.get("kind") or "")
    clause = _ask_clause(ctx, rec)
    st = _PState(sid=sid, key=key, kind=kind, clause=clause,
                 origin=(tid, int(getattr(rec, "step", 0) or 0)))
    if kind == "end":
        st.terminal = str(act.get("terminal") or "done")
    elif kind == "judge":
        st.prompt = str(act.get("prompt") or "")
        st.reads = [str(r) for r in (act.get("reads") or [])]
        st.writes = _compiler._infer_writes(dict(act), dict(rec.output or {}))
        if not st.reads:
            st.reads = sorted(k for k in prev_vars if k not in st.writes)
        if not st.writes:
            st.writes = [f"{sid}_verdict"]
    elif kind == "model":
        st.reads = ([str(r) for r in (act.get("reads") or [])]
                    or _compiler._infer_reads(dict(act), prev_vars))
        st.writes = _compiler._infer_writes(dict(act), dict(rec.output or {}))
        st.payload = {"kind": "model",
                      "prompt": str(act.get("prompt") or act.get("template") or ""),
                      "reads": list(st.reads), "writes": list(st.writes)}
    else:                                              # tool (an unrecognized kind is also treated as a tool)
        st.reads = _compiler._infer_reads(dict(act), prev_vars)
        st.writes = _compiler._infer_writes(dict(act), dict(rec.output or {}))
        st.payload = {"kind": "tool", "name": str(act.get("name") or ""),
                      "input": _compiler._templatize(dict(act.get("input") or {}),
                                                     prev_vars),
                      "reads": list(st.reads), "writes": list(st.writes)}
        if act.get("phase"):            # phase comes with the record and goes into the state and the KEY (symmetric on both sides)
            st.payload["phase"] = str(act["phase"])
    plan.states[sid] = st
    plan.order.append(sid)
    plan.by_key[key].append(sid)
    return st


def _resolve_target(plan: Plan, p: str, rec: Any, key: tuple, prev_vars: dict,
                    ctx: _Ctx, tid: str) -> tuple[Optional[str], bool]:
    """The L6-L9 decision: is this step **a new step**, or **a return to an existing step**.

    Returns ``(target state, whether it was newly created)``; returns ``(None, False)`` when
    compilation cannot proceed at this point.

    Deterministic level: normalized action KEY matches an existing state => repeat, otherwise new step.
    With a model, ask the model (touchpoint (a) of :data:`MODEL_TOUCHPOINTS`); a reply that does not fit
    the schema records one rejection and falls back to the deterministic level; once the same point
    reaches the rejection limit, compilation cannot proceed.
    """
    # KEYs aligned only by successor, never deduplicated globally (model states in the document
    # skeleton fold to ("model",): a model step anywhere in a trace "looks like" them, so global
    # deduplication would align the first model step to the wrong place in the skeleton; they are aligned
    # only in _step's L5 (successors of the current state) and are always treated as new steps here).
    local_only = set(getattr(ctx, "local_only_keys", ()) or ())
    cands = [] if key in local_only else list(plan.by_key.get(key, []))
    exact = _aligned_by_state(plan, p, rec, key)
    if exact is not None:
        cands = [exact] + [c for c in cands if c != exact]  # the self-reported state comes first
    choice = cands[0] if cands else "new"
    if ctx.model is not None:
        ans = _ask_new_or_repeat(ctx, p, rec, cands)
        if ans is None:
            if ctx.strike(p, "new step/repeat reply does not fit the schema"):
                return None, False
        elif ans == "new" or ans in cands:
            choice = ans
            ctx.clear(p)
        elif ctx.strike(p, f"model points to a nonexistent state {ans!r}"):
            return None, False
    if choice == "new":
        sid = plan.new_sid()
        _create_state(plan, sid, rec, key, prev_vars, ctx, tid)
        return sid, True
    return choice, False


def _observe(plan: Plan, sid: str, rec: Any, tid: str) -> None:
    """Accumulate this step's observations into the ledger: judge labels and examples, variable types,
    trace membership."""
    st = plan.states[sid]
    if tid and tid not in plan.trace_states[sid]:
        plan.trace_states[sid].append(tid)
    out = dict(rec.output or {})
    for w in st.writes:
        t = _type_of(out.get(w))
        if t and w not in plan.var_types:
            plan.var_types[w] = t
    if st.kind != "judge" or not st.writes:
        return
    label = out.get(st.writes[0])
    if isinstance(label, str) and label and label not in st.labels_seen:
        st.labels_seen.append(label)
        snap = {k: dict(rec.vars).get(k) for k in st.reads if k != "label"}
        st.examples.append((label, snap))


def transcribe(t_plus: Sequence[Trace], *, ctx: Optional[_Ctx] = None,
               skill_id: str = "compiled", plan: Optional[Plan] = None,
               on_trace: Any = None, ordered: bool = True) -> Plan:
    """**Pass 1**: walk every trace in L2 order (fewest steps first) and produce a :class:`Plan`.

    This pass does not change a single byte of any machine -- all it does is decide: alignment, new
    step/repeat, where branches are, how much support, which labels judge actions have seen, and the
    maximum number of times each state is entered within a single trace.

    If ``plan`` is given, transcription **continues** on it (the path used when a document skeleton is
    the seed and traces update it online); ``on_trace(plan, trace, before, after)`` is called after each
    trace, where ``before/after`` are ``(number of states, number of edges, sum of edge support)``
    before and after -- the online per-trace ledger relies on it; ``ordered=False`` feeds the traces in
    the given order (online arrival order) instead of re-sorting by step count.
    """
    ctx = ctx or _Ctx()
    plan = plan if plan is not None else Plan(skill_id=skill_id)
    seq = _order_traces(t_plus) if ordered else list(enumerate(t_plus))
    for _, trace in seq:
        task = trace.task if isinstance(trace.task, dict) else {}
        inp = task.get("input", {}) if isinstance(task.get("input", {}), dict) else {}
        plan.input_keys |= set(inp)
        for k, v in inp.items():
            t = _type_of(v)
            if t and k not in plan.var_types:
                plan.var_types[k] = t
        plan.max_records = max(plan.max_records, len(trace.records))
        before = _plan_size(plan)
        _transcribe_one(plan, trace, ctx)
        if on_trace is not None:
            on_trace(plan, trace, before, _plan_size(plan))
    return plan


def _plan_size(plan: Plan) -> tuple[int, int, int]:
    return (len(plan.states), len(plan.edges), sum(e.support for e in plan.edges.values()))


def _transcribe_one(plan: Plan, trace: Trace, ctx: _Ctx) -> None:
    """L3-L11: walk one accepted trace action by action."""
    recs = list(trace.records)
    if not recs:
        return
    tid = _tid(trace)
    task = trace.task if isinstance(trace.task, dict) else {}
    task_input = task.get("input", {}) if isinstance(task.get("input", {}), dict) else {}
    visits: dict = defaultdict(int)
    prev_sid, prev_rec = "", None
    pprev_sid = ""
    counts: dict = defaultdict(int)                 # counts per base KEY within this trace (for key_of)

    try:
        for i, rec in enumerate(recs):
            key = (ctx.key_of(rec, prev_rec, counts) if ctx.key_of is not None
                   else canon_action(rec, strict=True))
            prev_vars = dict(prev_rec.vars) if prev_rec is not None else dict(task_input)
            if not prev_sid:                               # L3: start
                if not plan.initial:
                    sid = plan.new_sid()
                    _create_state(plan, sid, rec, key, prev_vars, ctx, tid)
                    plan.initial = sid
                elif plan.states[plan.initial].key == key:
                    sid = plan.initial
                else:
                    plan.notes.append({
                        "trace": tid, "step": i, "kind": "start_mismatch",
                        "why": f"the opening action {_brief(rec.action)} of this trace is not the same "
                               f"step as the compiled start {plan.initial}; a machine has only one start, "
                               "and this shape cannot express branching at the very beginning, so the "
                               "whole trace is not transcribed"})
                    return
            else:
                sid = _step(plan, prev_sid, rec, key, prev_vars, ctx, tid, i)
                if sid is None:
                    return
                e = plan.edges[(prev_sid, sid)]
                e.support += 1
                plan.obs[prev_sid].append((sid, dict(prev_rec.vars)))
                # The 4th element is the trace **object**, not task_id: the same task may have several
                # runs that task_id cannot tell apart, and a labeler given the wrong run produces wrong
                # labels (observed in practice: labels did not match successors).
                plan.obs_ctx[prev_sid].append((sid, pprev_sid, visits[prev_sid], trace, i - 1))
            visits[sid] += 1
            _observe(plan, sid, rec, tid)
            pprev_sid, prev_sid, prev_rec = prev_sid, sid, rec
    finally:
        # For a trace that gets stuck halfway, **the part already walked still counts**: visit counts
        # are the only basis for the loop bound K, and dropping half a trace would make K too small and
        # push loops that could have finished into FALLBACK too early.
        for s, c in visits.items():
            plan.visits[s] = max(plan.visits[s], c)


def _aligned_by_state(plan: Plan, p: str, rec: Any, key: tuple) -> Optional[str]:
    """The state id self-reported by the trace record is the **exact** alignment target, provided it is
    in the ledger and its identity matches.

    Folded identities collide: model states in the seed all fold to ``("model",)`` (the document's
    private prompt is not comparable to what the model said in a trace), so if both a state's back edge
    and its main-path edge point to model states, L5, which takes the first outgoing edge in order, may
    pick the back edge. Observed in practice: a 17-step trace that went round the repair loop three times
    collapsed into a self-loop at the start, the branch guard could not be learned and was blocked, and a
    23-state skeleton compiled into 2 states.

    ``Record.state`` is **what the execution side faithfully recorded** as "the state at the time", which
    is more precise than the folded key. It is used only when both conditions hold: the id exists in the
    ledger, and its identity equals the identity computed for this step -- otherwise it is a state name
    from another machine and is ignored. External agent logs lack this field (``to_trace`` always records
    FALLBACK), so this path naturally does not apply.
    """
    sid = str(getattr(rec, "state", "") or "")
    if not sid or sid == FALLBACK:
        return None
    st = plan.states.get(sid)
    if st is None or st.key != key:
        return None
    return sid


def _step(plan: Plan, p: str, rec: Any, key: tuple, prev_vars: dict,
          ctx: _Ctx, tid: str, step: int) -> Optional[str]:
    """Take one step from ``p``: L5 alignment / L6-L8 new step or loop / L9 branch. Returns ``None``
    when compilation cannot proceed."""
    if p in plan.blocked:
        return None
    exact = _aligned_by_state(plan, p, rec, key)
    if exact is not None and exact in plan.out.get(p, []):
        return exact                                       # state self-reported by the record, and indeed a successor of p
    for dst in plan.out.get(p, []):                        # L5: outgoing edge with the same KEY, just advance
        if plan.states[dst].key == key:
            return dst
    target, created = _resolve_target(plan, p, rec, key, prev_vars, ctx, tid)
    if target is None:
        plan.blocked.add(p)
        plan.notes.append({"trace": tid, "step": step, "kind": "blocked",
                           "state": p, "why": "the same point reached the rejection limit; this "
                                              "segment falls back to interpreted execution"})
        return None
    back = (not created) and plan.reaches(target, p)        # L8: loop
    plan.add_edge(p, target, back=back, creating=created, origin=(tid, step))
    return target


# --------------------------------------------------------------------------- #
# Pass 1.5: fixing guards (fit.learn_cond for L9 / judge actions for L10)
# --------------------------------------------------------------------------- #
def _plan_variables(plan: Plan) -> list[Variable]:
    """The variable table derived from the ledger. Three sources:

    * variables **written by some state**;
    * **task input fields** (with ``init_from``);
    * **counter variables** -- an edge's ``inc`` increments them and guards read them, but no state
      "writes" them. ``repair_count`` in a document skeleton is like that: it only has an initial value
      and is incremented by back edges. Missing it, every guard that reads it is flagged by the
      gatekeeper as "uses an undeclared variable" and the whole proposal batch is rejected with it
      (observed in practice: once document edges were no longer removed, three batches were rejected
      for this).

    A trace's ``vars`` also contain other things (e.g. the counter variables of the machine that
    produced these traces); those are **never collected**: a machine only recognizes variables it can
    write itself, and collecting them would make guards read a name nobody ever writes.
    """
    names: set[str] = set(plan.input_keys)
    for st in plan.states.values():
        names |= set(st.writes)
    counters = {e.counter for e in plan.edges.values() if getattr(e, "counter", None)}
    names |= counters
    out: list[Variable] = []
    for n in sorted(names):
        if n in counters and n not in plan.input_keys:
            out.append(Variable(name=n, type="integer", init=0))
            continue
        out.append(Variable(name=n, type=plan.var_types.get(n, "string"),
                            init_from=f"task.input.{n}" if n in plan.input_keys else None))
    return out


def _tighten(conds: dict, snaps: Mapping, variables: Sequence[Variable]) -> dict:
    """**Tighten** learned guards **to the minimal form supported by observations**: ``x != 'b'`` =>
    ``x == 'a'`` (if ``x`` is always ``'a'`` on this branch).

    Two reasons, both essential:

    * **Safety**. Situations outside the branch (the cell where the judge action abstains) should land
      on the default edge, i.e. FALLBACK; keeping the ``!=`` form would also swallow "values never
      seen" into some branch. Better to compile less.
    * **Determinism**. :func:`hexis.legacy.fit.candidate_atoms` enumerates string literals through a
      ``set``, so for the same snapshots different processes may yield ``==`` first or ``!=`` first,
      making learned guards **unstable across processes**. Tightening to the equality form makes both
      search orders converge on the same answer.
    """
    out = dict(conds)
    for tgt in list(out):
        expr = out[tgt]
        try:
            used = sorted(_cond.vars_of(expr))
        except _cond.CondError:
            continue
        if len(used) != 1:
            continue
        v = used[0]
        vals = {s.get(v) for s in snaps.get(tgt, []) if v in s}
        if len(vals) != 1:
            continue
        val = next(iter(vals))
        if isinstance(val, bool) or not isinstance(val, (str, int, float)):
            continue
        eq = f"{v} == {val!r}"
        if eq == expr:
            continue
        mine = list(snaps.get(tgt, []))
        others = [s for o in snaps if o != tgt for s in snaps[o]]
        if not _fit._separates(eq, mine, others):
            continue
        trial = {**out, tgt: eq}
        if _fit.mutually_exclusive(list(trial.values()), snaps, variables):
            out = trial
    return out


def _always_guard(snaps: Sequence[dict], variables: Sequence[Variable]) -> Optional[str]:
    """Learn, for a **back edge**, a predicate that is "always true on all observed snapshots of this
    state". Returns ``None`` if none can be learned.

    Why back edges must have a guard: see the module docs, "Why back edges always carry a guard".
    """
    if not snaps:
        return None
    atoms = _fit.candidate_atoms(list(snaps), variables)
    expr = _fit.separating(atoms, list(snaps), (), max_atoms=1)
    if expr is None:
        return None
    return _tighten({"__loop__": expr}, {"__loop__": list(snaps)}, variables)["__loop__"]


def _judge_branch(plan: Plan, p: str, snaps: Mapping, ctx: _Ctx) -> Optional[dict]:
    """L10: when a branch has no learnable deterministic guard, draft a judge action and use its
    verdict variable as the guard.

    **Only done when the step already is a judge step** (see the module docs, "Judge actions are only
    rewritten in place"): inserting a judge state out of thin air makes the machine take one more step
    than the trace, so that trace can no longer be replayed -- not worth it.
    """
    st = plan.states.get(p)
    if st is None or st.kind != "judge" or ctx.model is None:
        return None
    reads = list(st.reads) or sorted({k for s in snaps.values() for x in s for k in x})
    judge = draft_judge({"state": p, "reads": reads, "writes": list(st.writes),
                         "targets": {t: list(v) for t, v in snaps.items()},
                         "clause": st.clause}, model=ctx.model)
    if judge is None:
        ctx.strike(p, "judge action drafting failed or the reply does not fit the schema")
        return None
    usable = [l for l in judge.labels if l != judge.abstain]
    targets = list(snaps)
    if len(usable) < len(targets):
        ctx.strike(p, f"the drafted label set has only {len(usable)} non-abstain labels, not enough "
                      f"for {len(targets)} branches")
        return None
    assign = {t: usable[i] for i, t in enumerate(targets)}
    samples = [(dict(s), assign[t]) for t in targets for s in snaps[t]]
    rate = _calibrate(ctx, judge, samples)
    if rate is None:
        ctx.strike(p, "cannot calibrate the judge action's error rate")
        return None
    if rate > ctx.thresholds.judge_err_max:
        ctx.strike(p, f"calibrated error rate {rate} > bound {ctx.thresholds.judge_err_max}")
        return None
    st.prompt = judge.prompt
    st.labels_seen = [l for l in judge.labels if l != judge.abstain]
    st.reads = list(judge.reads)
    st.error_rate = rate
    st.support = len(samples)
    st.drafted = True
    st.examples = [(assign[t], {k: s.get(k) for k in judge.reads})
                   for t in targets for s in snaps[t][:1]]
    w = st.writes[0]
    return {t: f"{w} == {assign[t]!r}" for t in targets}


def _drop_thin(plan: Plan, min_support: int) -> None:
    """First half of L14: remove edges with insufficient support entirely, and prune the states only
    they could reach -- those fall back to interpreted execution."""
    for kk in list(plan.edge_order):
        e = plan.edges[kk]
        if e.support >= min_support:
            continue
        if e.origin and e.origin[0] == "document":
            # An edge of the document skeleton that no trace walked: **keep it**. This is not "taking
            # chance for a rule" but "required by the document, not yet observed in traces" -- the two
            # must be handled differently. Removing it would leave the machine able to do only the paths
            # already seen; branches the document describes but this batch of traces happened not to
            # take (error handling, edge cases) would vanish from the artifact entirely.
            #
            # The cost is the support threshold: this edge has support=0 and can never pass
            # min_support, and a failed acceptance makes _settle demote from the **source state** to
            # FALLBACK, taking along the main path that already matched the traces (observed in practice:
            # the whole machine collapsed into begin->FALLBACK). So the threshold side must exempt it too
            # -- see checker._support_rows: support requires "evidence for the compiled path", and
            # document edges were never compiled from traces in the first place.
            plan.notes.append({"kind": "doc_unobserved", "edge": f"{e.src}->{e.dst}",
                               "cond": e.cond, "support": e.support,
                               "why": "this edge of the document skeleton was not walked by this batch "
                                      "of traces: **it stays in the artifact anyway** (a document's "
                                      "claim does not vanish just because it was not observed), but it "
                                      "is recorded here -- it is one of the most valuable targets when "
                                      "adding traces"})
            continue
        plan.thin.append({"edge": f"{e.src}->{e.dst}", "support": e.support,
                          "min_support": min_support,
                          "why": "compiling this from so few traces would take chance for a rule; "
                                 "this branch falls back to interpreted execution"})
        _unlink(plan, kk)
    _prune(plan)
    # During transcription, "back edge" was decided by whether the graph at that time could loop back.
    # A document skeleton brings a set of support=0 loops (the s4->s2 repair loop), so a trace going
    # straight from s2 to s4 looked like closing a loop and was recorded as a back edge; once those
    # document edges are removed above it can no longer loop back at all, and learning "when to go round
    # again" for it would only fail and block s2 entirely (observed in practice: document skeleton + 4
    # linear traces compiled into a machine with only a start). Only demote, never promote: real loops
    # stay as they are.
    for e in plan.edges.values():
        if e.back and not plan.reaches(e.dst, e.src):
            e.back = False
            plan.notes.append({"kind": "back_edge_demoted", "edge": f"{e.src}->{e.dst}",
                               "why": "during transcription this edge only formed a loop through "
                                      "document edges; those were not walked by traces and have been "
                                      "removed, so this edge is now a forward edge"})
    # A second pitfall from the same source: posting only issues proposals for states that "have a
    # creating edge or are the start". In the skeleton, s4's creating edge is the document edge s3->s4;
    # when a trace walks straight from s2 to s4, s4 already exists and that edge is recorded with
    # creating=False. Once the document edge is removed, s4 has no creating edge, and the whole state plus
    # the main path after it cannot be posted (observed in practice: state:s5 was rejected for
    # "referencing nonexistent state s4"). Fix: if a state that still has incoming edges has no creating
    # edge, promote its earliest forward incoming edge to the creating edge.
    creators = {kk[1] for kk in plan.edge_order if plan.edges[kk].creating}
    for sid in plan.order:
        if sid == plan.initial or sid in creators:
            continue
        incoming = [kk for kk in plan.edge_order if kk[1] == sid]
        pick = next((kk for kk in incoming if not plan.edges[kk].back), None) or \
            (incoming[0] if incoming else None)
        if pick is None:
            continue
        plan.edges[pick].creating = True
        creators.add(sid)
        plan.notes.append({"kind": "creator_reanchored", "state": sid,
                           "edge": f"{pick[0]}->{pick[1]}",
                           "why": "the original creating edge was a document edge that no trace "
                                  "walked and has been removed; this trace-supported incoming edge "
                                  "creates the state instead"})


def _unlink(plan: Plan, kk: tuple) -> None:
    e = plan.edges.pop(kk, None)
    if e is None:
        return
    plan.edge_order = [x for x in plan.edge_order if x != kk]
    plan.out[e.src] = [d for d in plan.out.get(e.src, []) if d != e.dst]
    plan.obs[e.src] = [(d, s) for d, s in plan.obs.get(e.src, []) if d != e.dst]


def _prune(plan: Plan) -> None:
    """Recompute reachability from the start along the edges **still present**; prune unreachable
    states together with their edges."""
    if not plan.initial:
        return
    seen, stack = {plan.initial}, [plan.initial]
    while stack:
        cur = stack.pop()
        if cur in plan.blocked:
            continue
        for nxt in plan.out.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    dead = [s for s in plan.order if s not in seen]
    for s in dead:
        plan.pruned.append(s)
        plan.order.remove(s)
        st = plan.states.pop(s)
        plan.by_key[st.key] = [x for x in plan.by_key.get(st.key, []) if x != s]
        plan.out.pop(s, None)
        plan.obs.pop(s, None)
    if dead:
        for kk in list(plan.edge_order):
            if kk[0] in dead or kk[1] in dead:
                _unlink(plan, kk)


def _avail_map(plan: Plan) -> dict:
    """Variables guaranteed to have been written **on entry** to each state (intersection over paths).

    Same criterion as :func:`hexis.machine.checks._write_before_read`: the union asks "is there a path
    on which it exists", the intersection asks "does it exist on every path". Only the latter guarantees
    that no undefined variable is hit at run time.
    """
    seed = set(plan.input_keys)
    universe = set(seed)
    for st in plan.states.values():
        universe |= set(st.writes)
    avail = {sid: set(universe) for sid in plan.states}
    if plan.initial in avail:
        avail[plan.initial] = set(seed)
    incoming: dict = defaultdict(list)
    for src, dst in plan.edge_order:
        incoming[dst].append(src)
    for _ in range(len(plan.states) + 2):
        changed = False
        for sid in plan.order:
            if sid == plan.initial:
                continue
            srcs = incoming.get(sid, [])
            new = set() if not srcs else set(universe)
            for src in srcs:
                st = plan.states.get(src)
                new &= avail.get(src, set()) | (set(st.writes) if st else set())
            if new != avail.get(sid):
                avail[sid] = new
                changed = True
        if not changed:
            break
    return avail


def _template_vars(payload: Any) -> list[str]:
    """Which variables the input template of a tool state in the ledger references. Same criterion as
    checks.template_vars."""
    if not isinstance(payload, Mapping):
        return []
    from hexis.machine.checks import template_vars
    return template_vars(type("A", (), {"input": payload.get("input") or {}})())


def _prune_reads(plan: Plan, avail: dict) -> None:
    """Narrow each state's ``reads`` to what is "guaranteed written when reaching it".

    ``compiler._infer_reads`` is a **value-equality** heuristic: if some input value happens to equal a
    variable's current value, the variable counts as read. In practice it infers entirely unrelated
    variables (a submit step reading ``verify_exit``) that nobody writes on some other path -- the
    gatekeeper then flags ``E_READ_BEFORE_WRITE`` and the whole compile decision is discarded. Rather
    than let one speculative read drag down a whole section of the graph, narrow it honestly here:
    **what cannot be read does not count as read**, and record the removals in the ledger (so the
    coverage report can say "this step seemed to read something else too, dropped because nobody
    writes it on some path").
    """
    for sid in plan.order:
        st = plan.states[sid]
        if not st.reads:
            continue
        ok = avail.get(sid, set())
        # Variables referenced by the input template **must not be removed**: they were not guessed by
        # _infer_reads, the step really needs them to render its input. Removing the read declaration
        # while keeping the template renders that slot empty at run time (observed in practice: bash got
        # an empty command). If nobody writes it, that should be an E_READ_BEFORE_WRITE, left for a later
        # step to supply a producer.
        pinned = set(_template_vars(st.payload))
        dropped = [r for r in st.reads if r not in ok and r not in pinned]
        if not dropped:
            continue
        st.reads = [r for r in st.reads if r in ok]
        if isinstance(st.payload, dict) and "reads" in st.payload:
            st.payload["reads"] = list(st.reads)
        plan.notes.append({
            "kind": "read_pruned", "state": sid, "dropped": dropped,
            "why": "nobody writes these variables on some path leading to this state (intersection "
                   "over paths); keeping them would make the whole compile decision fail the "
                   "write-before-read check -- removed honestly"})


def _fix_doc_branch(plan: Plan, p: str, targets: list, fixed: dict, snaps: dict,
                    variables: list) -> bool:
    """Outgoing edges of a document-skeleton state: the document's guards are used as is, and each new
    target grown from traces gets a guard mutually exclusive with them.

    A skeleton branch looks like: ``if repair_count >= 3 -> s_done`` plus a **default edge** ``-> s2``.
    The old criterion reused the document guards only if "every edge has a non-empty guard" -- the
    default edge's guard is the empty string, so the criterion failed on the spot and the guards were
    relearned; targets the document mentions but no trace walked have no snapshots, cannot be learned,
    and the whole branch got blocked. Previously this bug was masked by pruning: all document edges were
    removed, the state had only one target left, and the main path was taken. Once they were no longer
    removed, all 23 branches were blocked.

    The correct reading treats a branch as "several guarded edges + at most one default edge": the
    guarded ones only need to be mutually exclusive, and the default edge catches the rest. So:

    * the document's guarded edges -- kept as is;
    * the document's default edge -- stays in the default slot;
    * a new target grown from traces -- learn a predicate from its own snapshots, **then conjoin the
      negations of the document guards**, guaranteeing pairwise mutual exclusion with the document edges
      (:func:`hexis.machine.checks._determinism` requires pairwise exclusion, regardless of order).

    If nothing can be learned, return ``False`` so the caller takes the original relearn /
    introduce-judge / block path.
    """
    doc_guarded = [t for t in targets if fixed.get(t)]
    doc_default = [t for t in targets if t in fixed and not fixed.get(t)]
    fresh = [t for t in targets if t not in fixed]
    if len(doc_default) > 1:
        return False                                   # the skeleton itself is invalid; let downstream report it
    if not fresh:
        for t in doc_guarded:
            plan.edges[(p, t)].cond = fixed[t]
        for t in doc_default:
            plan.edges[(p, t)].cond = ""
        return True
    if len(fresh) > 1 or not doc_default:
        # several new targets must also be mutually exclusive, or there is no default slot left: both
        # cases go down the normal guard-learning path
        return False
    t = fresh[0]
    g = _always_guard(snaps.get(t) or [], variables)
    if g is None:
        return False
    negs = " and ".join(f"not ({fixed[d]})" for d in doc_guarded)
    plan.edges[(p, t)].cond = f"({g}) and {negs}" if negs else g
    for d in doc_guarded:
        plan.edges[(p, d)].cond = fixed[d]
    plan.edges[(p, doc_default[0])].cond = ""
    plan.notes.append({"kind": "doc_branch_extended", "state": p, "new_target": t,
                       "cond": plan.edges[(p, t)].cond,
                       "why": "a new target grew on a document branch: the document's edges are "
                              "kept as is, the new edge gets a predicate true only on its own "
                              "observations and mutually exclusive with the document guards, and the "
                              "document's default edge stays in the default slot"})
    return True


def fit_guards(plan: Plan, ctx: _Ctx) -> None:
    """Fix a guard for every edge (L9/L10), and compute a counter bound for every back edge (K of L8).

    * a state with only **one** target => main-path edge, unguarded (except back edges, see
      :func:`_always_guard`).
    * **two or more** targets => :func:`hexis.legacy.fit.learn_cond` learns a set of pairwise mutually
      exclusive predicates; if that fails, try an L10 judge action; if that fails too, the whole branch
      falls back to interpreted execution (the state goes into ``plan.blocked`` and none of its outgoing
      edges are posted; only the default edge provided by the gatekeeper remains).
    * all branch edges **always carry a guard**, and the default slot is left to FALLBACK: unseen
      situations fall back to interpreted execution instead of being swallowed by some branch.
    """
    avail = _avail_map(plan)
    _prune_reads(plan, avail)
    variables = _plan_variables(plan)
    declared = {v.name for v in variables}
    thr = ctx.thresholds
    for p in list(plan.order):
        if p in plan.blocked:
            # a point that already hit the rejection limit during transcription: none of its outgoing
            # edges are posted, and the default slot is left to FALLBACK.
            for kk in [k for k in plan.edge_order if k[0] == p]:
                _unlink(plan, kk)
            continue
        targets = list(plan.out.get(p, []))
        if not targets:
            continue
        snaps: dict = {t: [] for t in targets}
        only = plan.guard_vars.get(p)                  # introduced judge state: only look at the judge variable
        # Outgoing guards are evaluated **after this state runs**, so the usable set is "available on
        # entry, union written by this state". Without this filter a predicate reading "a variable nobody
        # writes on some path" could be learned, and the gatekeeper would flag write-before-read on the spot.
        usable = (avail.get(p, set()) | set(plan.states[p].writes)) & declared
        for dst, snap in plan.obs.get(p, []):
            if dst in snaps:
                snaps[dst].append({k: v for k, v in snap.items()
                                   if k in usable and (only is None or k == only)})
        # ---- document skeleton branch: guards are set by the document, not relearned ---- #
        fixed = getattr(plan, "fixed_conds", {}).get(p) or {}
        if fixed and _fix_doc_branch(plan, p, targets, fixed, snaps, variables):
            continue
        if len(targets) == 1:
            e = plan.edges[(p, targets[0])]
            if not e.back:
                e.cond = ""                                # main path
                continue
            g = _always_guard(snaps[targets[0]], variables)
            if g is None:
                _block(plan, p, "no predicate for \"when to go round again\" could be learned for the "
                                "back edge; the whole loop is dropped")
                continue
            e.cond = g
            continue
        learned = _fit.learn_cond(snaps, variables, min_support=thr.min_support,
                                  holdout_ratio=thr.holdout_ratio,
                                  acc_thr=thr.acc_thr)
        if learned is None:
            learned = _judge_branch(plan, p, snaps, ctx)
        if learned is None:
            _block(plan, p, "no pairwise mutually exclusive guards could be learned for the branch "
                            "and no judge action could be set up: the whole branch falls back to "
                            "interpreted execution")
            continue
        learned = _tighten(learned, snaps, variables)
        for t in targets:
            plan.edges[(p, t)].cond = learned.get(t, "")
        if any(not plan.edges[(p, t)].cond for t in targets):
            _block(plan, p, "some branch got no guard (the learned guards are incomplete)")
    _prune(plan)
    _install_bounds(plan, ctx)


def _block(plan: Plan, p: str, why: str) -> None:
    plan.blocked.add(p)
    plan.blocked_targets[p] = list(plan.out.get(p, []))
    plan.notes.append({"kind": "blocked", "state": p, "why": why,
                       "targets": list(plan.out.get(p, []))})
    for kk in list(plan.edge_order):
        if kk[0] == p:
            _unlink(plan, kk)


def _install_bounds(plan: Plan, ctx: _Ctx) -> None:
    """Give every back edge a counter variable and bound K, and **record who set K**.

    ``doc_bound`` is always ``None``: this module does not extract iteration bounds from the document
    (an extracted number would be taken as a document requirement, when it is really just the
    compiler's reading of the document). So :func:`hexis.legacy.fit.loop_bound_detail` always yields
    ``source="compiler"`` -- which lets the coverage report **state** "this bound was added by the
    compiler to guarantee halting; it is not a document requirement".
    """
    for kk in plan.edge_order:
        e = plan.edges[kk]
        if not e.back:
            continue
        lb = _fit.loop_bound_detail(plan.visits.get(e.dst, 1),
                                    margin=ctx.thresholds.loop_margin)
        # If the document skeleton already named this back edge's counter (``inc: repair_count``), keep
        # that name; do not rename: the skeleton's guards use that name (``repair_count >= 3``), and
        # renaming it to ``sN_count`` would make those guards read an undeclared variable, so the whole
        # proposal batch would be rejected by the gatekeeper (observed in practice: once document edges
        # were no longer removed, three batches were rejected for this). Only loops grown from traces
        # get a harness-chosen name.
        e.counter = e.counter or f"{e.dst}_count"
        e.bound = e.bound or lb.k
        plan.loop_bounds.append({
            "back_edge": f"{e.src}->{e.dst}", "var": e.counter, "k": lb.k,
            "source": lb.source, "observed_max": lb.observed_max, "margin": lb.margin,
            "why": "the skill document states no iteration bound; K = ceil(margin x the maximum "
                   "number of times this state is entered in a single trace), added by the compiler "
                   "so that \"the loop is guaranteed to halt\""})


# --------------------------------------------------------------------------- #
# Pass 2: posting (turning decisions into receipt proposals)
# --------------------------------------------------------------------------- #
#: Names and order of the five batches; the multi-agent merge stage order matches it.
BATCH_NAMES: tuple[str, ...] = ("entry", "trunk", "judges", "transitions", "loops")


def build_batches(plan: Plan, ctx: _Ctx, *, prohibitions: Sequence = (),
                  audit_tools: Sequence[str] = ()) -> dict[str, list[Proposal]]:
    """Turn the ledger into five **batches**: ``entry`` (open_machine) / ``trunk`` (building states,
    including judges and terminals with creating edges) / ``judges`` (judges rewritten in place; empty
    on the single-agent path) / ``transitions`` (forward branch edges) / ``loops`` (back edges).
    :func:`build_proposals` is their concatenation in order.

    Putting back edges last is not laziness: ``close_loop`` calls
    :func:`hexis.legacy.fit.install_counter`, which ``and``s ``count < K`` onto **every guarded outgoing
    edge the loop's target state has at that moment**, then inserts ``count >= K -> FALLBACK``. Adding a
    new guarded edge to that state after closing the loop leaves ``count < K`` off the new edge, so it
    holds together with the bound exit once the counter is full -- the structural check flags overlapping
    guards on the spot. So loops must be closed last.
    """
    out: dict[str, list[Proposal]] = {name: [] for name in BATCH_NAMES}
    variables = _plan_variables(plan)
    open_payload = {"variables": [v.model_dump() for v in variables],
                    "prohibitions": [dict(p) if isinstance(p, Mapping) else p
                                     for p in (prohibitions or [])],
                    "max_steps": max(24, plan.max_records * 2)}
    if audit_tools:
        open_payload["audit_tools"] = list(audit_tools)
    out["entry"].append(Proposal(
        "open_machine", open_payload,
        rationale=f"L1: open machine {plan.skill_id}, declaring {len(variables)} variables"
                  f" (task inputs {sorted(plan.input_keys)} use init_from)"))

    creator = {kk[1]: plan.edges[kk] for kk in plan.edge_order if plan.edges[kk].creating}
    for sid in plan.order:
        st = plan.states[sid]
        e = creator.get(sid)
        attach: dict = {}
        if e is not None:
            attach = {"from_state": e.src, "from_cond": e.cond,
                      "from_support": e.support}
        elif sid != plan.initial:
            continue                       # no creating edge and not the start: this state cannot be posted
        out["trunk"].append(_state_proposal(plan, st, attach, sid == plan.initial))

    for kk in plan.edge_order:
        e = plan.edges[kk]
        if e.creating or e.back:
            continue
        out["transitions"].append(Proposal(
            "add_transition",
            {"from_state": e.src, "to": e.dst, "cond": e.cond, "support": e.support},
            rationale=f"L9: branch at {e.src}, guard {e.cond or '(default)'} learned by "
                      f"fit.learn_cond from variable snapshots of {e.support} observations",
            trace_id=e.origin[0], step=e.origin[1]))

    for kk in plan.edge_order:
        e = plan.edges[kk]
        if not e.back:
            continue
        out["loops"].append(Proposal(
            "close_loop",
            {"from_state": e.src, "to": e.dst, "cond": e.cond,
             "counter": e.counter, "bound": e.bound, "support": e.support},
            rationale=f"L8: {e.src}->{e.dst} is a back edge, closed into a bounded loop; counter {e.counter}"
                      f" bound {e.bound} (**the bound is added by the compiler; the document does not state it**)",
            trace_id=e.origin[0], step=e.origin[1]))
    return out


def build_proposals(plan: Plan, ctx: _Ctx, *, prohibitions: Sequence = ()) -> list[Proposal]:
    """Turn the ledger into a sequence of receipt proposals. Order = decision order, with the single
    exception that **back edges come last** -- i.e. the concatenation of the five batches of
    :func:`build_batches` in :data:`BATCH_NAMES` order."""
    batches = build_batches(plan, ctx, prohibitions=prohibitions)
    return [p for name in BATCH_NAMES for p in batches[name]]


def _state_proposal(plan: Plan, st: _PState, attach: dict, is_initial: bool) -> Proposal:
    src = attach.get("from_state") or "(start)"
    if st.kind == "end":
        payload = {"state_id": st.sid, "terminal": st.terminal, "clause": st.clause}
        seed_t = (getattr(plan, "seed_terminals", {}) or {}).get(st.terminal)
        if seed_t is not None:                     # terminal kind declared by the document skeleton (verified/unverified)
            payload["kind"] = seed_t.kind
            payload["output"] = list(seed_t.output)
        if st.origin_kind:
            payload["origin"] = st.origin_kind
        if st.locator:
            payload["locator"] = st.locator
        payload.update(attach)
        return Proposal("set_terminal", payload,
                        rationale=f"L11: the trace ends after {src}, with terminal {st.terminal}",
                        trace_id=st.origin[0], step=st.origin[1])
    if st.kind == "judge":
        labels = sorted(st.labels_seen)
        if ABSTAIN not in labels:
            labels.append(ABSTAIN)
        examples = []
        for lbl, snap in st.examples:
            ex = {k: v for k, v in snap.items() if k != "label"}
            ex["label"] = lbl
            examples.append(ex)
        payload = {"state_id": st.sid, "prompt": st.prompt,
                   "reads": list(st.reads), "writes": list(st.writes),
                   "labels": labels, "abstain": ABSTAIN, "examples": examples,
                   "error_rate": st.error_rate, "support": st.support,
                   "clause": st.clause}
        if st.introduced:
            payload.update({"introduced": True, "gold_from": st.gold_from})
        if st.origin_kind:
            payload["origin"] = st.origin_kind
        if st.locator:
            payload["locator"] = st.locator
        payload.update(attach)
        how = ("introduced from the document (introduce_judge)" if st.introduced else
               "drafted by the model (L10)" if st.drafted else "a judge step already in the traces, transcribed as is")
        return Proposal("add_judge", payload,
                        rationale=f"L7: a judgement follows {src}, {how}; labels {labels} taken from "
                                  f"{'the draft' if st.drafted else 'trace observations'}",
                        trace_id=st.origin[0], step=st.origin[1])
    payload = {"state_id": st.sid, "action": dict(st.payload), "clause": st.clause}
    if st.origin_kind:
        payload["origin"] = st.origin_kind
    if st.locator:
        payload["locator"] = st.locator
    if is_initial and not attach:
        payload["initial"] = True
    payload.update(attach)
    return Proposal("add_state", payload,
                    rationale=f"L7: a new step {_brief(st.payload)} follows {src}, "
                              f"reads {st.reads} -> writes {st.writes}, clause "
                              f"{st.clause or '(unattributed: clause attribution needs a model)'}",
                    trace_id=st.origin[0], step=st.origin[1])


def apply_plan(ck: _checker.Checker, proposals: Sequence[Proposal], *,
               max_strikes: int = MAX_STRIKES, progress: bool = False) -> PlanResult:
    """**Pass 2**: hand the proposals one by one to the gatekeeper for a ruling.

    Two rules, both directly matching Algorithm 1's requirements:

    * **A rejected proposal does not affect the proposals accepted before it** -- this is a property
      the gatekeeper already has; this function merely does not break it: after a rejection, it moves
      on to the next proposal.
    * **``max_strikes`` consecutive rejections at the same point => ``demote_to_fallback`` that point,
      then move on.** Falling back to interpreted execution also removes the states reachable only
      through that point, so every later proposal touching a dead state is skipped (no point running
      into a certain rejection again).

    ``ck`` must either not have run ``open_machine`` yet (the first proposal in the list is exactly
    that) or already be open -- both work; opening twice is rejected by the gatekeeper itself, which
    records a receipt.
    """
    res = PlanResult()
    strikes: dict = defaultdict(int)
    dead: set[str] = set()
    alive: set[str] = set()
    for prop in proposals:
        point = prop.point
        touched = {str(prop.payload.get(k) or "")
                   for k in ("from_state", "to", "state_id")} - {""}
        if touched & dead:
            res.skipped.append(prop)
            continue
        if alive and prop.op != "open_machine":
            need = {str(prop.payload.get(k) or "") for k in ("from_state", "to")} - {""}
            if need - alive:
                res.skipped.append(prop)
                continue
        receipt = ck.apply(prop.to_checker())
        res.receipts.append(receipt)
        if progress:
            print(f"[compile_agent] {prop.op} "
                  f"{'✓' if receipt.accepted else '✗'} {receipt.reason[:110]}",
                  file=sys.stderr)
        if receipt.accepted:
            res.accepted += 1
            strikes[point] = 0
            if ck.opened:
                alive = set(ck.machine.states)
            continue
        res.rejected += 1
        if not ck.opened:
            continue
        strikes[point] += 1
        if point and strikes[point] >= max_strikes:
            dr = ck.demote_to_fallback(
                point, note=f"the same point was rejected {strikes[point]} times in a row (last: "
                            f"{receipt.reason[:80]}) -- better to compile less than to compile wrong")
            res.receipts.append(dr)
            strikes[point] = 0
            if dr.accepted:
                res.demoted.append(point)
                dead.add(point)
                alive = set(ck.machine.states)
                dead |= {s for s in touched if s not in alive}
    return res


# --------------------------------------------------------------------------- #
# L12/L13: rejected set and acceptance
# --------------------------------------------------------------------------- #
def _machine_walk(machine: Machine, trace: Trace) -> tuple[list, Optional[int]]:
    """Drive the machine along a trace; returns ``([(record index, state)], record index of the
    divergence or None)``.

    The driving logic matches :func:`hexis.legacy.replay.replay` (variables advanced from the trace
    records' output through the writes whitelist, back edges increment themselves, ``pick_edge`` picks
    edges); what is needed here is the **position**, to know which state to demote.
    """
    r = _replay.walk(machine, trace)
    # the virtual opening step and zero-width judges may record index -1 / the previous record: to locate
    # "which state to demote", only look at states that actually consumed a record
    seq = [(i, sid) for i, sid in r.seq if i >= 0]
    return seq, (None if r.ok else r.diverged_at)


def _offenders(machine: Machine, rep: Any, t_plus: Sequence[Trace],
               t_minus: Sequence[Trace]) -> list[str]:
    """Acceptance failed: which states should fall back to interpreted execution. Order-preserving and
    deduplicated, excluding FALLBACK and nonexistent states."""
    out: list[str] = []

    def push(sid: str) -> None:
        if sid and sid != machine.fallback and sid in machine.states and sid not in out:
            out.append(sid)

    for sid, _c in rep.weak_edges:
        push(sid)
    for sid, _r in rep.hot_judges:
        push(sid)
    for i in rep.unreproduced:
        seq, _d = _machine_walk(machine, t_plus[i])
        push(seq[-1][1] if seq else machine.initial)
    for i in rep.unexcluded:
        neg = t_minus[i]
        cut = neg.error_step if neg.error_step is not None else 10 ** 9
        seq, _d = _machine_walk(machine, neg)
        # Which state to demote? Demoting ``p`` sends the machine into FALLBACK at **the step after
        # p**, so pick the last state "whose next record's step is still <= the error position" --
        # demoting it makes the fallback segment cover the error, and this negative example moves from
        # "missed in the compiled section" to "not compiled yet, not yet excludable".
        at = [sid for idx, sid in seq
              if idx + 1 < len(neg.records) and neg.records[idx + 1].step <= cut]
        push(at[-1] if at else (seq[-1][1] if seq else machine.initial))
    for f in rep.findings:
        if f.severity == "error":
            push(f.state_id)
    return out


def _settle(ck: _checker.Checker, t_plus: Sequence[Trace], t_minus: Sequence[Trace],
            thr: Thresholds, ctx: _Ctx, *, commit: bool = True) -> tuple[Any, list, bool]:
    """L12+L13: run acceptance ourselves first; on failure demote the offending states to interpreted
    execution, and only ``commit`` once it passes.

    Why not just ``commit`` and let it fail: ``Checker.commit`` is **all-or-nothing** -- on failure the
    whole batch is rolled back to the previous commit (here ``open_machine``, i.e. an empty machine). One
    real failure wastes every accepted rewrite together with its receipts, and **after the rollback no
    receipt interface can put them back**. So this first takes a look with
    :func:`hexis.legacy.verify.verify_machine` (read-only, does not change the machine), demotes the
    offending points according to the report, and commits once it passes. If it cannot be repaired, it
    **does not commit**: the machine stays "passed all structural checks, but without the acceptance
    stamp", which is far more honest than rolling back to an empty machine; this is recorded in
    ``stats["commit"]``.
    """
    receipts: list = []
    rep = _verify.verify_machine(ck.machine, t_plus, t_minus, thresholds=thr)
    seen_bad: list = []
    for _ in range(_MAX_REPAIR):
        if rep.ok:
            break
        bad = _offenders(ck.machine, rep, t_plus, t_minus)
        if not bad or bad == seen_bad:
            break                    # nothing more to demote (e.g. the opening action does not match the start): do not spin
        seen_bad = list(bad)
        moved = False
        for sid in bad:
            r = ck.demote_to_fallback(
                sid, note="L12/L13: acceptance points here as failing -- " + _verify.summary(rep)[:120])
            receipts.append(r)
            moved = moved or r.accepted
            ctx.say(f"demote {sid} to interpreted execution: {'✓' if r.accepted else '✗'}")
        if not moved:
            break
        rep = _verify.verify_machine(ck.machine, t_plus, t_minus, thresholds=thr)
    if not rep.ok:
        return rep, receipts, False
    if not commit:
        # multi-agent round: do not commit here even if acceptance passed -- only the orchestrator
        # commits once, on the incumbent
        return rep, receipts, True
    receipts.append(ck.commit(t_plus=t_plus, t_minus=t_minus))
    return rep, receipts, receipts[-1].accepted


# --------------------------------------------------------------------------- #
# Skill and clause table
# --------------------------------------------------------------------------- #
def _resolve_skill(skill: Any) -> tuple[str, str, list, Optional[Machine]]:
    """Extract ``(skill_id, document body, prohibitions, reference machine or None)`` from ``skill``.

    Four shapes are recognized: :class:`~hexis.skill_loader.AgentSkill`; a module or object with
    ``skill_doc()`` / ``reference_machine()`` (``examples.table_clean`` is one); a ``dict``; and a body
    string or a skill directory path.

    **The reference machine is only used for diffing** (``CompileResult.diff_vs_reference``); not a
    single byte of it goes into the compiled artifact. Prohibitions are hand-annotated things that cannot
    be compiled into the graph, and are only collected when ``skill`` itself carries ``prohibitions``
    -- picking them up from the reference machine would be using the target as the arrow.
    """
    if skill is None:
        return "compiled", "", [], None
    if isinstance(skill, Mapping):
        doc = str(skill.get("doc") or skill.get("body") or skill.get("text") or "")
        sid = str(skill.get("skill_id") or skill.get("slug") or skill.get("name")
                  or "compiled")
        ref = skill.get("reference_machine")
        return sid, doc, list(skill.get("prohibitions") or []), \
            (ref if isinstance(ref, Machine) else None)
    if isinstance(skill, (str, Path)):
        text = str(skill)
        p = Path(text) if (isinstance(skill, Path)
                           or ("\n" not in text and len(text) < 4096)) else None
        if p is not None and p.is_dir():
            try:
                from hexis.skill_loader import load_agent_skill
                loaded = load_agent_skill(p)
                return loaded.slug, loaded.body, [], None
            except Exception:                               # noqa: BLE001
                pass
        return "compiled", text, [], None

    doc = ""
    for attr in ("skill_doc", "full_text"):
        fn = getattr(skill, attr, None)
        if callable(fn):
            try:
                doc = str(fn())
                break
            except Exception:                               # noqa: BLE001
                doc = ""
    if not doc:
        for attr in ("body", "doc", "text"):
            v = getattr(skill, attr, None)
            if isinstance(v, str) and v:
                doc = v
                break
    ref: Optional[Machine] = None
    rm = getattr(skill, "reference_machine", None) or \
        getattr(skill, "build_reference_machine", None)
    if isinstance(rm, Machine):
        ref = rm
    elif callable(rm):
        try:
            got = rm()
            ref = got if isinstance(got, Machine) else None
        except Exception:                                   # noqa: BLE001
            ref = None
    sid = ""
    for attr in ("skill_id", "slug", "name"):
        v = getattr(skill, attr, None)
        if isinstance(v, str) and v:
            sid = v
            break
    if not sid and ref is not None:
        sid = ref.skill_id
    proh = list(getattr(skill, "prohibitions", ()) or [])
    return sid or "compiled", doc, proh, ref


def clause_rows(doc: str, clauses: Sequence = ()) -> list[ClauseRow]:
    """L1's "list the clause table". If ``clauses`` is given it is used (both :class:`Clause` records
    and ``(id, text)`` pairs are accepted); otherwise first try :func:`hexis.legacy.compiler.partition`
    (numbered ``## Sx`` headings), and if that yields nothing, split by structure with
    :func:`markdown_clauses` -- every SKILL.md has a clause table."""
    rows: list[ClauseRow] = []
    src: Sequence = clauses if clauses else _compiler.partition(doc or "")
    if not src and not clauses:
        src = markdown_clauses(doc or "")
    for item in src:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            cid, text = str(item[0]), str(item[1])
            title = next((l.strip("# ").strip() for l in text.splitlines() if l.strip()),
                         cid)
            rows.append(ClauseRow(id=cid, title=title, text=text,
                                  locator=str(item[2]) if len(item) > 2 else ""))
            continue
        cid = str(getattr(item, "id", "") or "")
        if not cid:
            continue
        loc = ""
        fn = getattr(item, "locator", None)
        if callable(fn):
            try:
                loc = str(fn())
            except Exception:                               # noqa: BLE001
                loc = ""
        rows.append(ClauseRow(id=cid, title=str(getattr(item, "title", "") or cid),
                              text=str(getattr(item, "text", "") or ""), locator=loc))
    return rows


# --------------------------------------------------------------------------- #
# Coverage report (one of the deliverables)
# --------------------------------------------------------------------------- #
def _coverage(machine: Machine, plan: Plan, rows: Sequence[ClauseRow],
              t_plus: Sequence[Trace], t_minus: Sequence[Trace], ctx: _Ctx,
              rep: Any, committed: bool, demoted: Sequence[str]) -> dict:
    """Per clause supported / thin / not reached by any trace + structure origins + fallback surface +
    "which additional traces would be most valuable".

    The base is :func:`hexis.legacy.report.cover_report` directly, so
    :func:`hexis.legacy.report.render` can render this dict; this function only adds entries on top.
    """
    thr = ctx.thresholds
    base = _report.cover_report(machine, [(r.id, r.text) for r in rows],
                                t_plus=t_plus, t_minus=t_minus)

    cite: dict = defaultdict(list)
    for sid, st in sorted(machine.states.items()):
        if st.clause:
            cite[st.clause].append(sid)
    in_support: dict = defaultdict(int)
    for src, t in machine.transitions_all():
        if t.to != machine.fallback:
            in_support[t.to] += t.support

    table: list[dict] = []
    supported, thin, untouched = [], [], []
    for r in rows:
        states = cite.get(r.id, [])
        traces = sorted({tid for s in states for tid in plan.trace_states.get(s, [])})
        # Support counts **the number of traces that walked these states**, not incoming-edge support:
        # the start has no incoming edge, and measuring it by incoming edges would judge "the first step
        # every trace took" as thin. Incoming-edge support is kept in a separate column; each number
        # tells its own story.
        sup = len(traces)
        if not states:
            status = "untouched"
            untouched.append(r.id)
        elif sup < thr.min_support:
            status = "thin"
            thin.append(r.id)
        else:
            status = "supported"
            supported.append(r.id)
        table.append({"id": r.id, "title": r.title, "locator": r.locator,
                      "status": status, "states": states, "traces": traces,
                      "support": sup,
                      "edge_support": sum(in_support.get(s, 0) for s in states)})

    # The K ledger marks each entry live by "is this back edge still on the machine": falling back to
    # interpreted execution during acceptance also removes a whole segment, and that segment's K is just
    # history that should no longer count as a compiler-introduced structure in the artifact.
    live_edges = {f"{sid}->{t.to}" for sid, t in machine.transitions_all()}
    bounds = [{**lb, "live": lb["back_edge"] in live_edges} for lb in plan.loop_bounds]

    introduced: list[dict] = []
    for lb in bounds:
        if not lb["live"]:
            continue
        introduced.append({"kind": "loop_bound", **lb})
        introduced.append({
            "kind": "counter_variable", "var": lb["var"],
            "why": "the counter variable was introduced by the compiler to bound the loop; the "
                   "document has no such variable"})
    for sid, t in machine.transitions_all():
        if t.cond and sid in plan.states and not t.inc:
            introduced.append({
                "kind": "branch_guard", "state": sid, "cond": t.cond, "to": t.to,
                "why": "the branch guard was learned by fit.learn_cond from variable snapshots; the "
                       "document does not state this predicate explicitly"})
    fb_states = sorted({sid for sid, t in machine.transitions_all()
                        if t.to == machine.fallback and sid != machine.fallback})
    introduced.append({
        "kind": "fallback_surface", "states": fb_states,
        "why": "default edges to FALLBACK are escape hatches left by the compiler: when guards do "
               "not cover a case, a judge abstains, or a loop is exhausted, execution falls back "
               "from here to \"the model reads the whole document and interprets it\""})

    from_doc = [{"kind": "state", "id": sid, "clause": st.clause,
                 "action": st.action.kind,
                 "locator": next((r.locator for r in rows if r.id == st.clause), "")}
                for sid, st in sorted(machine.states.items()) if st.clause]

    next_traces: list[dict] = []
    for cid in untouched:
        next_traces.append({"kind": "clause", "target": cid,
                            "why": "no trace reaches this clause; it currently lies entirely in "
                                   "FALLBACK"})
    for row in plan.thin:
        next_traces.append({"kind": "edge", "target": row["edge"],
                            "why": f"support {row['support']} < {thr.min_support}, "
                                   "so it was pruned; a few more traces taking this branch would get "
                                   "it compiled"})
    for note in plan.notes:
        if note.get("kind") == "blocked":
            next_traces.append({"kind": "branch", "target": note.get("state", ""),
                                "why": note.get("why", "")})
    starts = [n for n in plan.notes if n.get("kind") == "start_mismatch"]
    if starts:
        next_traces.append({
            "kind": "start", "target": machine.initial,
            "why": f"the **opening action** of {len(starts)} accepted traces is not the same step "
                   "as the compiled start, so those traces were not transcribed at all. A machine has "
                   "only one start, and this shape cannot express \"branching at the very "
                   "beginning\"; either group the traces by opening action and compile one machine "
                   "per group, or add a common opening step and collect another round of traces"})
    for sid in fb_states:
        next_traces.append({"kind": "fallback", "target": sid,
                            "why": "this state still has a default edge to interpreted execution: "
                                   "the more traces walk past it, the better the chance of compiling "
                                   "the default branch as well"})

    judge_states = [s for s in machine.states.values() if s.action.kind == "judge"]
    base.update({
        "skill_id": machine.skill_id,
        "clause_table": table,
        "supported": supported, "thin": thin, "untouched": untouched,
        "clause_attribution": "model" if ctx.model is not None else "none",
        "structures": {"from_document": from_doc, "compiler_introduced": introduced},
        "loop_bounds": bounds,
        "fallback_surface": {
            "states_with_fallback_edge": fb_states,
            "n_edges_to_fallback": sum(1 for _s, t in machine.transitions_all()
                                       if t.to == machine.fallback),
            "demoted": list(demoted),
            "blocked_branches": sorted(plan.blocked),
            "thin_edges_dropped": list(plan.thin),
            "pruned_states": list(plan.pruned),
        },
        "next_traces": next_traces,
        "notes": list(plan.notes),
        "model_dependence": {
            "model_used": ctx.model is not None,
            "touchpoints": list(MODEL_TOUCHPOINTS),
            "model_free": [
                "states and main path (traces aligned by normalized action KEY)",
                "repeats and loops (a KEY matching an existing state is a repeat)",
                "branch guards (fit.learn_cond: three gates -- support / holdout accuracy / provable mutual exclusion)",
                "back-edge counter variables and bound K (fit.loop_bound_detail)",
                "support pruning, structural checks and acceptance (checker / verify)",
                "end states and terminals (taken from the traces' end records)",
            ],
            "needs_model": [
                f"clause attribution (this run: {'attributed by the model' if ctx.model is not None else 'all left empty'})",
                f"semantic new step vs repeat decision (this run: "
                f"{'ask the model' if ctx.model is not None else 'structural heuristic, by action KEY'})",
                "drafting a judge action when no branch guard can be learned (this run: "
                f"{'may draft' if ctx.model is not None else 'no drafting, the whole branch falls back to FALLBACK'})",
                "error-rate calibration of judge actions (without calibration there is no bound sum(eps_i))",
                "introducing judge actions from document clauses (multi-agent: introduce_judge; model=None uses the skill package's judges library)",
                "whether the same action under two predecessor contexts is the same step (multi-agent: split_context; model=None decides by whether guards can be learned separately)",
                "online collection probes labelling snapshots (multi-agent: annotate_judge; without running it there are no judge records)",
            ],
        },
        "verify": _verify.report_dict(rep),
        "committed": committed,
        "judge_states": {s.id: {"error_rate": s.action.error_rate,
                                "support": s.action.support} for s in judge_states},
    })
    return base


def _diff(machine: Machine, ref: Optional[Machine]) -> Optional[dict]:
    """Item-by-item diff between the compiled artifact and the hand-written target machine. Returns
    ``None`` without a reference machine."""
    if ref is None:
        return None

    def keys(m: Machine) -> dict:
        return {sid: canon_action(st.action, strict=False)
                for sid, st in m.states.items() if sid != m.fallback}

    got, want = keys(machine), keys(ref)
    gset, wset = set(got.values()), set(want.values())
    return {
        "reference_skill_id": ref.skill_id,
        "n_states_reference": ref.n_states(), "n_states_compiled": machine.n_states(),
        "n_transitions_reference": len(ref.transitions_all()),
        "n_transitions_compiled": len(machine.transitions_all()),
        "actions_matched": sorted("|".join(k) for k in (gset & wset)),
        "actions_missing": sorted("|".join(k) for k in (wset - gset)),
        "actions_extra": sorted("|".join(k) for k in (gset - wset)),
        "judges_reference": sorted(s.id for s in ref.states.values()
                                   if s.action.kind == "judge"),
        "judges_compiled": sorted(s.id for s in machine.states.values()
                                  if s.action.kind == "judge"),
        "prohibitions_reference": [p.id for p in ref.prohibitions],
        "prohibitions_compiled": [p.id for p in machine.prohibitions],
        "note": "the reference machine is the **target shape**, not a deliverable; this table only "
                "shows where the compiled artifact differs.",
    }


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
def compile_skill(skill: Any, t_plus: Sequence[Trace], t_minus: Sequence[Trace] = (), *,
                  model: Any = None, thresholds: Optional[Thresholds] = None,
                  clauses: Sequence = (), progress: bool = False,
                  begin: bool = False) -> CompileResult:
    """Algorithm 1: compile a machine by sequential transcription of real execution traces. **The only
    channel that writes the machine is the gatekeeper.**

    ``skill`` accepts a skill object/module/``dict``/body/directory path (see :func:`_resolve_skill`);
    ``t_plus`` are accepted traces, ``t_minus`` rejected traces (L12 uses them to verify "the machine
    diverges at or before the error"); ``clauses`` are split from the body if not given. ``model=None``
    uses deterministic heuristics, with no network at all.

    In the returned :class:`CompileResult`, ``machine`` is the artifact, ``receipts`` the audit trail of
    how each step of it grew, ``coverage`` the coverage report (rendered directly by
    :func:`hexis.legacy.report.render`), ``judges`` lists every judge action in the machine with its
    origin, ``stats`` holds counters, and ``diff_vs_reference`` gives an item-by-item diff when a
    hand-written target machine is available.
    """
    thr = thresholds or Thresholds()
    skill_id, doc, prohibitions, ref = _resolve_skill(skill)
    rows = clause_rows(doc, clauses)
    ctx = _Ctx(model=model, thresholds=thr, rows=rows, doc=doc, progress=progress)
    ctx.say(f"L1: skill {skill_id}, clause table {len(rows)} rows, "
            f"T+ {len(t_plus)} traces, T- {len(t_minus)} traces, "
            f"model {'present' if model is not None else 'absent (deterministic heuristics)'}")
    if begin:
        # Opening tool: prepend a BEGIN_TOOL step to every trace, so the machine has exactly one start
        # and the real first step becomes a branch after it -- start_mismatch drops to zero. Replay is
        # transparent to this (replay.walk virtually adds the same step), so the T+/T- exclusion and
        # replay checks still run on the original traces.
        from hexis.traces.trace_adapter import with_begin
        t_plus = [with_begin(t) for t in t_plus]
        t_minus = [with_begin(t) for t in t_minus]

    # ---- pass 1: transcription (decisions of L2-L11) ---- #
    plan = transcribe(t_plus, ctx=ctx, skill_id=skill_id)
    _drop_thin(plan, thr.min_support)                       # first half of L14
    fit_guards(plan, ctx)                                   # L9/L10 fix guards + L8 set K
    ctx.say(f"transcription done: {len(plan.order)} states, {len(plan.edge_order)} edges, "
            f"{len(plan.blocked)} branches could not be compiled")

    # ---- pass 2: posting (one receipt per rewrite) ---- #
    ck = _checker.Checker(skill_id, doc=doc, thresholds=thr)
    proposals = build_proposals(plan, ctx, prohibitions=prohibitions)
    filed = apply_plan(ck, proposals, progress=progress)
    if not ck.opened:                                       # not even the machine was opened: return empty-handed
        from hexis.machine.schema import empty_machine
        return CompileResult(machine=empty_machine(skill_id), receipts=filed.receipts,
                             coverage={}, judges=[],
                             stats={"committed": False,
                                    "commit": "not even open_machine passed; nothing was compiled"},
                             diff_vs_reference=None)

    # ---- L12 + L13: rejected set and acceptance ---- #
    rep, settle_receipts, committed = _settle(ck, t_plus, t_minus, thr, ctx)
    machine = ck.machine
    receipts = filed.receipts + settle_receipts
    demoted = list(filed.demoted) + [r.detail.get("state_id", "")
                                     for r in settle_receipts
                                     if r.op == "demote_to_fallback" and r.accepted]

    judges = [{
        "state": sid, "prompt": st.action.prompt, "reads": list(st.action.reads),
        "writes": list(st.action.writes), "labels": list(st.action.labels),
        "abstain": st.action.abstain, "error_rate": st.action.error_rate,
        "support": st.action.support, "clause": st.clause,
        "source": ("drafted" if (sid in plan.states and plan.states[sid].drafted)
                   else "trace"),
    } for sid, st in sorted(machine.states.items()) if st.action.kind == "judge"]

    coverage = _coverage(machine, plan, rows, t_plus, t_minus, ctx, rep, committed,
                         demoted)
    stats = {
        "traces_in": len(t_plus), "traces_negative": len(t_minus),
        "clauses": len(rows),
        "states_added": sum(1 for r in receipts
                            if r.op in ("add_state", "add_judge", "set_terminal")
                            and r.accepted),
        # Only the two edge-only interfaces. The state-building interfaces **bring their own incoming
        # edge** (add_state's from_state), so the machine has more edges than this number -- see
        # n_transitions for the machine's actual count.
        "transitions_added": sum(1 for r in receipts
                                 if r.op in ("add_transition", "close_loop")
                                 and r.accepted),
        "n_states": machine.n_states(),
        "n_transitions": len(machine.transitions_all()),
        "judges_calibrated": sum(1 for j in judges if j["support"] > 0),
        "judges_added": sum(1 for r in receipts if r.op == "add_judge" and r.accepted),
        "judges_drafted": sum(1 for j in judges if j["source"] == "drafted"),
        "fallback_demotions": len([d for d in demoted if d]),
        "proposals": len(proposals), "accepted": filed.accepted,
        "rejected": filed.rejected, "skipped": len(filed.skipped),
        "model_calls": ctx.model_calls,
        "model_rejects": len(ctx.rejects),
        "prompt_tokens": None, "completion_tokens": None,
        "blocked_branches": sorted(plan.blocked),
        "thin_edges_dropped": len(plan.thin),
        # Accepted traces whose opening action does not match the start and were not transcribed at
        # all. A machine has only one start, and this shape cannot express "branching at the very
        # beginning" -- such traces can never be replayed, so acceptance fails and must be reported
        # honestly.
        "traces_not_transcribed": sum(1 for n in plan.notes
                                      if n.get("kind") == "start_mismatch"),
        "committed": committed,
        "commit": ("acceptance passed and committed" if committed
                   else "acceptance failed, **not committed** (commit is all-or-nothing; one failure "
                        "rolls every accepted rewrite back to an empty machine): " + _verify.summary(rep)),
        "structural_findings": structural_findings(machine),
    }
    usage = _runtime._usage_of(model)
    if usage:
        stats["prompt_tokens"] = usage.get("prompt_tokens")
        stats["completion_tokens"] = usage.get("completion_tokens")
    ctx.say(f"L13: {stats['commit']}")
    return CompileResult(machine=machine, receipts=receipts, coverage=coverage,
                         judges=judges, stats=stats,
                         diff_vs_reference=_diff(machine, ref))
