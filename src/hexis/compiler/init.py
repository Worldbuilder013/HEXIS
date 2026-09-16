"""Initial machine generation (context driven).

The model receives a :class:`~hexis.compiler.context.CompileContext`: the document text, the clause table, the tool
definitions (names, parameters, outputs, success condition, marked as guaranteed by the interface or inferred from
traces), the task input fields, the terminal table and the skill rules. It returns an efsm-v1 machine. Every round's
candidate goes through G_init = format ∧ schema ∧ guard ∧ structure ∧ clause ∧ tool ∧ terminal; if it fails, the
candidate and its errors are handed back to the model, by default for at most three rounds. Once a candidate passes,
its clause ids are cleared, it is normalized and it goes through the update stage's Check.

When the skill rules (labels, requirements, terminal conditions) have no explicit file, :func:`extract_rules` has the
model extract them from the document; every entry carries its document quote, and entries whose quote does not check
out are discarded. The rules contain only event patterns, no interpretation of tools or fields.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from hexis.compiler import check as _check
from hexis.compiler.context import (
    CompileContext,
    EventPattern,
    LabelRule,
    Requirement,
    TerminalCondition,
    parse_rules,
    quote_in_document,
)
from hexis.llm.model_iface import ModelUnavailable
from hexis.machine import checks as _checks
from hexis.machine import cond as _cond
from hexis.machine.schema import ABSTAIN, Machine

FORMAT_EXAMPLE = {
    "format": "efsm-v1", "skill_id": "<skill_id>", "initial": "s1", "fallback": "FALLBACK",
    "max_steps": 32, "phase_rules": "default", "audit_tools": [],
    "variables": [
        {"name": "task_field_a", "type": "string", "init_from": "task.input.task_field_a"},
        {"name": "s2_param", "type": "string"}, {"name": "s2_result", "type": "string"},
        {"name": "verdict", "type": "string"},
        {"name": "status_field", "type": "integer", "init": 0}, {"name": "primary_field", "type": "string"},
        {"name": "retry_count", "type": "integer", "init": 0},
    ],
    "terminals": [{"id": "DONE", "kind": "done"}, {"id": "GAVE_UP", "kind": "gave_up"},
                  {"id": "END_FALLBACK", "kind": "fallback"}],
    "states": {
        "s1": {"id": "s1", "clause": "S1.1", "origin": "document",
               "action": {"kind": "model",
                          "prompt": "Produce the parameter s2_param for the tool call in s2, based on task_field_a. Return exactly one JSON object {\"s2_param\": \"...\"}.",
                          "reads": ["task_field_a"], "writes": ["s2_param"]},
               "transitions": [{"if": "retry_count >= 2", "to": "GAVE_UP"}, {"if": "", "to": "s2"}]},
        "s2": {"id": "s2", "clause": "S1.1", "origin": "document",
               "action": {"kind": "tool", "name": "<a tool name from tools>", "phase": "",
                          "input": {"<param key>": "${s2_param}"}, "reads": [],
                          "writes": ["status_field", "primary_field", "s2_result"],
                          "binds": {"primary_field": "s2_result"}},
               "transitions": [{"if": "<the tool's success condition>", "to": "s3"},
                               {"if": "", "to": "s1", "inc": "retry_count"}]},
        "s3": {"id": "s3", "clause": "S1.2", "origin": "document",
               "action": {"kind": "judge", "prompt": "Is s2_result what the task asked for? Answer 'ok' or 'redo'.",
                          "reads": ["s2_result"], "writes": ["verdict"],
                          "labels": ["ok", "redo", ABSTAIN], "abstain": ABSTAIN},
               "transitions": [{"if": "verdict == 'redo'", "to": "s1", "inc": "retry_count"},
                               {"if": "", "to": "s4"}]},
        "s4": {"id": "s4", "clause": "S1.3", "origin": "document",
               "action": {"kind": "model", "observable": True,
                          "prompt": "Write the final deliverable from s2_result. Return exactly one JSON object {\"final_text\": \"...\"}.",
                          "reads": ["s2_result"], "writes": ["final_text"]},
               "transitions": [{"if": "", "to": "DONE"}]},
        "DONE": {"id": "DONE", "origin": "document", "action": {"kind": "end", "terminal": "DONE"}},
        "GAVE_UP": {"id": "GAVE_UP", "origin": "document", "action": {"kind": "end", "terminal": "GAVE_UP"}},
        "FALLBACK": {"id": "FALLBACK", "origin": "compiler", "action": {"kind": "end", "terminal": "END_FALLBACK"}},
    },
}

PROMPT = """You compile a skill document into one executable state machine (format efsm-v1) and return it as a single JSON object. The document is source material; follow the machine format below exactly.

Inputs in VARIABLES: document (the skill text), clauses (numbered clause table), skill_id, tools (the only tools allowed, with their parameter fields, result fields, success condition and whether the definition is guaranteed or inferred from observed calls), task_inputs (task input fields observed in traces, with how often each appears), terminals (the terminal ids you must provide, with their kinds), rules (labels, requirements and terminal conditions the skill imposes, each with its document quote), format_example (a syntax-only example for a DIFFERENT, generic skill), and optionally previous_candidate and errors from the last round.

Machine format:
- Top level: format="efsm-v1", skill_id, initial, fallback="FALLBACK", max_steps, phase_rules="default", audit_tools (may be []), variables, terminals, states.
- variables: [{name, type, init?, init_from?}]. Every task input listed in task_inputs gets a variable with init_from="task.input.<key>". Other variables have no init except integer counters (init 0). Declare every variable you use, including generated tool parameters.
- terminals: exactly the ids in terminals plus one with kind="fallback" (id END_FALLBACK). The FALLBACK state is an end state with that fallback terminal; its id is exactly "FALLBACK".
- states: object keyed by state id; each {id, clause, origin, action, transitions}. clause = a clause id from clauses or "". origin = "document" for what the document requires, "compiler" for implementation choices.
- action kinds:
  tool:  {kind:"tool", name:<tool name from tools>, phase:<label or "">, labels:[...], input:<parameter template>, reads:[], writes:[], binds:{}}
         input uses the tool's own parameter keys. Values that depend on the task are "${variable}" templates whose variable is produced by a model state placed right before the tool state. Never hard-code task-specific values.
         writes lists the result fields to keep (use the tool's result field names); a semantic name may be added with binds={"<primary result field>": "semantic_name"} and listed in writes.
         phase = the base label of the call: for a tool whose definition in tools carries a label use that label; for generic command tools write "apply" when the command writes something and "probe" when it only reads; otherwise "". labels = derived labels defined by the rules that this state realizes (never the base label).
  model: {kind:"model", prompt, reads, writes, observable?}. One generation without tools; the prompt states what to produce and instructs to return exactly one JSON object with the keys in writes; reads includes everything the prompt needs. observable=true only for a model state whose output is the deliverable itself (a summary, answer or report the skill hands back).
  judge: {kind:"judge", prompt, reads, writes:[label_var], labels:[..., "abstain"], abstain:"abstain"}. reads must already exist; writes is one new label variable; include a conservative unconditional default edge.
  user:  {kind:"user", prompt, writes:[answer_var]} only if the skill requires asking the user.
  end:   {kind:"end", terminal:<terminal id>}.
- transitions: [{if, to, inc?}], evaluated in order; "if" is a condition or "" for the default. Conditions support ==, !=, <, <=, >, >=, and/or/not, empty(x), nonempty(x), string, integer and boolean constants over variables. No arithmetic, attribute access or indexing. inc names one integer counter incremented when the edge is taken. Success of a tool call is exactly the tool's success condition from tools (over its result fields, which must be in writes).
- Rules the checker enforces: every state reachable from initial; every state can reach an end state; every non-end state has a default ("") edge; guarded edges of one state are mutually exclusive; every cycle has an edge with inc and the cycle target has an exit "counter >= K" going to FALLBACK, with the other guarded edges of that state combined with "counter < K"; a variable is only read after some state on every path wrote it (template ${var} counts as a read).
- Terminal conditions in rules say which evidence a terminal needs (events matching the patterns, on the success branch) and which events invalidate it. Build the graph so that the evidence necessarily holds on every path into that terminal, and route paths without the evidence to the default terminal. Requirements of kind must_occur / before / forbid must hold on every path.
- Keep the machine small: typically 4–12 states. Every clause you keep must be reflected as a state or inside a model prompt. Do not copy the example's states or topology; the number of states, their order, the branches and loops must come from the document, the tools and the rules.

If previous_candidate and errors are given, fix exactly those errors and return the full corrected machine.
Return only the machine JSON object.
"""

RULES_PROMPT = """Extract the compile rules that this skill document imposes, as one JSON object with keys "terminals", "labels", "requirements", "terminal_conditions". Use only the pattern language below; every entry must carry a "quote": a verbatim sentence or clause from the document that states it (entries whose quote is not found verbatim in the document are discarded).

Inputs in VARIABLES: document, clauses, tools (tool names and fields), task_inputs (field names), base_labels (labels the trace processor already assigns: for generic command tools "apply" when a call writes, "probe" otherwise; plus the fixed labels given per tool).

Event pattern (all keys optional; omitted keys match anything):
  {"kind": "tool"|"model"|"user"|"end", "tool": <tool name>, "label": <label>, "role": "output" (model deliverable),
   "args_contain": <text; may contain ${task_input_field} which is substituted per task>, "arg_regex": <regex over the call arguments>,
   "success": true|false, "after": <another pattern that must have occurred earlier in the same trace>}

- terminals: [{"id": <UPPER_SNAKE id>, "kind": <short kind>}] — the ways a run can end that the document distinguishes (e.g. finished with checks passed vs finished without). Give at least one; the first terminal without conditions is the default.
- labels: [{"label": <name>, "when": <pattern>, "quote": ...}] — derived labels for events the document treats specially (e.g. "a read of the produced output after a write" → label "verify" with after={"label":"apply"}).
- requirements: [{"id": "R1", "kind": "must_occur"|"before"|"forbid", "a": <pattern>, "b": <pattern, for before: a must occur before b>, "quote": ...}] — only what the document explicitly requires.
- terminal_conditions: [{"terminal": <id>, "required_evidence": [<pattern with success true>], "invalidating_events": [<pattern>], "quote": ...}] — what must have happened (and not been undone) for a run to legitimately end at that terminal.

Do not invent requirements the document does not state. Prefer labels over tool names when the document speaks about kinds of actions. Return only the JSON object.
"""


@dataclass
class InitResult:
    machine: Optional[Machine] = None
    attempts: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    check_errors: list = field(default_factory=list)
    clause_map: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Rule extraction (model touchpoint, quote verification)
# --------------------------------------------------------------------------- #
def _valid_pattern(p: Optional[EventPattern], ctx: CompileContext, labels: set) -> bool:
    if p is None:
        return True
    if p.kind is not None and p.kind not in ("tool", "model", "judge", "user", "end"):
        return False
    if p.tool is not None and p.tool not in ctx.tools:
        return False
    if p.label is not None and p.label not in labels:
        return False
    return _valid_pattern(p.after, ctx, labels)


def extract_rules(ctx: CompileContext, model: Any, *, progress=None) -> tuple[dict, list[str]]:
    """Have the model extract rules from the document; returns (rules dict, notes). Quote verification and pattern
    validity checks both happen here."""
    notes: list[str] = []
    base = {"apply", "probe"} | {s.label for s in ctx.tools.values() if s.label}
    values = {"document": ctx.skill_text, "clauses": {c[0]: c[1][:300] for c in ctx.clauses},
              "tools": {n: {"input": list(s.input_keys()), "output": list(s.output_keys()),
                            "label": s.label} for n, s in ctx.tools.items()},
              "task_inputs": list(ctx.task_inputs), "base_labels": sorted(base)}
    try:
        raw = model.generate(prompt=RULES_PROMPT, values=values)
    except ModelUnavailable:
        raise                                                       # the endpoint failed: do not continue without rules
    except Exception as exc:                                        # noqa: BLE001
        return {}, [f"rule extraction failed: {type(exc).__name__}: {exc}"]
    if not isinstance(raw, dict):
        return {}, ["rule extraction: the reply is not a JSON object"]
    parsed = parse_rules(raw, source="document")
    labels = set(base) | {r.label for r in parsed["label_rules"]}
    keep_labels = []
    for r in parsed["label_rules"]:
        if not quote_in_document(r.quote, ctx.skill_text):
            notes.append(f"label {r.label}: quote not found in the document, discarded")
        elif not _valid_pattern(r.when, ctx, labels):
            notes.append(f"label {r.label}: pattern refers to an unknown tool or label, discarded")
        else:
            keep_labels.append(r)
    labels = set(base) | {r.label for r in keep_labels}
    keep_reqs = []
    for r in parsed["requirements"]:
        if not quote_in_document(r.quote, ctx.skill_text):
            notes.append(f"requirement {r.id}: quote not found in the document, discarded")
        elif not (_valid_pattern(r.a, ctx, labels) and _valid_pattern(r.b, ctx, labels)):
            notes.append(f"requirement {r.id}: pattern refers to an unknown tool or label, discarded")
        else:
            keep_reqs.append(r)
    keep_tcs = []
    for tc in parsed["terminal_conditions"]:
        pats = list(tc.required_evidence) + list(tc.invalidating_events)
        if not quote_in_document(tc.quote, ctx.skill_text):
            notes.append(f"terminal condition {tc.terminal}: quote not found in the document, discarded")
        elif not all(_valid_pattern(p, ctx, labels) for p in pats):
            notes.append(f"terminal condition {tc.terminal}: pattern refers to an unknown tool or label, discarded")
        else:
            keep_tcs.append(tc)
    terminals = parsed["terminals"] or [{"id": "DONE", "kind": "done"}]
    ids = {t["id"] for t in terminals}
    for tc in keep_tcs:
        if tc.terminal not in ids:
            terminals.append({"id": tc.terminal, "kind": tc.terminal.lower()})
            ids.add(tc.terminal)
    if progress:
        progress(f"  rules: {len(terminals)} terminals, {len(keep_labels)} labels, {len(keep_reqs)} requirements, "
                 f"{len(keep_tcs)} terminal conditions" + (f"; {len(notes)} discarded" if notes else ""))
    return {"terminals": terminals, "label_rules": keep_labels, "requirements": keep_reqs,
            "terminal_conditions": keep_tcs}, notes


def install_rules(ctx: CompileContext, rules: dict) -> None:
    ctx.terminals = list(rules.get("terminals") or ctx.terminals)
    ctx.label_rules = list(rules.get("label_rules") or [])
    ctx.requirements = list(rules.get("requirements") or [])
    ctx.terminal_conditions = list(rules.get("terminal_conditions") or [])
    known = {t["id"] for t in ctx.terminals}
    for tc in ctx.terminal_conditions:
        if tc.terminal not in known:
            ctx.terminals.append({"id": tc.terminal, "kind": tc.terminal.lower()})
            known.add(tc.terminal)


# --------------------------------------------------------------------------- #
# G_init
# --------------------------------------------------------------------------- #
def g_init(machine: Machine, ctx: CompileContext) -> list[str]:
    """Guard syntax, structure, clause ids, tool names, task input declarations, terminal table, fallback state."""
    errs: list[str] = []
    for sid, st in machine.states.items():
        for t in st.transitions:
            if t.cond:
                try:
                    _cond.parse(t.cond)
                except _cond.CondError as exc:
                    errs.append(f"guard: {sid}→{t.to} {t.cond!r}: {exc}")
    if errs:
        return errs
    errs += [f"structure: {f}" for f in _checks.structural_findings(machine)]
    ids = {c[0] for c in ctx.clauses}
    if ids:
        bad = sorted({st.clause for st in machine.states.values() if st.clause and st.clause not in ids})
        if bad:
            errs.append(f"clause: {bad} not in the clause table (available: {sorted(ids)[:8]}…)")
    for sid, st in machine.states.items():
        if st.action.kind == "tool" and st.action.name not in ctx.tools:
            errs.append(f"tool: {sid} uses {st.action.name!r}, only {sorted(ctx.tools)} are allowed")
    declared = {v.init_from for v in machine.variables if v.init_from}
    for k, fs in ctx.task_inputs.items():
        if fs.always_present and f"task.input.{k}" not in declared:
            errs.append(f"variable: missing variable declaration for task input task.input.{k}")
    have = {t.id: t.kind for t in machine.terminals}
    for t in ctx.terminals:
        if t["id"] not in have:
            errs.append(f"terminal: missing terminal {t['id']}")
    if "fallback" not in have.values():
        errs.append("terminal: missing a terminal with kind=fallback")
    if machine.fallback not in machine.states:
        errs.append(f"fallback: no {machine.fallback} state")
    return list(dict.fromkeys(errs))


def normalize(machine: Machine, ctx: CompileContext) -> list[str]:
    """Normalization before the update: origin marks, output field mappings, step limit. Modifies the machine in
    place and returns a log of what was done."""
    notes: list[str] = []
    base_labels = {"apply", "probe"} | {s.label for s in ctx.tools.values() if s.label}
    for sid, st in machine.states.items():
        if not st.origin:
            st.origin = "document"
        for t in st.transitions:
            if not t.origin:
                t.origin = "document"
        a = st.action
        if a.kind != "tool":
            continue
        spec = ctx.spec(a.name)
        # the base label goes to phase (for tools with a fixed label in the registry, the registry wins), derived labels
        # go to labels
        labels = list(dict.fromkeys(getattr(a, "labels", []) or []))
        if spec.label and a.phase != spec.label:
            if a.phase:
                notes.append(f"{sid}: base label {a.phase!r} changed to {spec.label!r} from the registry")
            a.phase = spec.label
        if not a.phase:
            base = [l for l in labels if l in base_labels]
            if base:
                a.phase = base[0]
                notes.append(f"{sid}: base label {base[0]!r} moved from labels to phase")
        a.labels = [l for l in labels if l not in base_labels and l != a.phase]
        known = set(spec.output_keys())
        binds = dict(a.binds or {})
        semantic = [w for w in a.writes if w not in known and w not in binds.values()]
        if semantic and spec.primary and spec.primary not in binds:
            binds[spec.primary] = semantic[0]
            notes.append(f"{sid}: output mapping {spec.primary} → {semantic[0]}")
            semantic = semantic[1:]
        if semantic:
            notes.append(f"{sid}: {semantic} have no output field to bind to and read as empty at run time")
        a.binds = binds
    machine.max_steps = max(int(machine.max_steps or 0), 3 * len(machine.states) + 16)
    return notes


def _unwrap(raw: Any) -> Any:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and "machine" in raw and isinstance(raw["machine"], dict):
        return raw["machine"]
    return raw


def _tools_for_prompt(ctx: CompileContext) -> dict:
    out = {}
    for n, s in ctx.tools.items():
        out[n] = {"description": s.description, "parameters": s.input_schema or {k: {"observed": c} for k, c in s.observed_inputs.items()},
                  "result_fields": s.output_keys(), "success": s.success, "primary_result": s.primary,
                  "definition": "guaranteed" if s.source in ("registry", "backend") else "inferred from traces",
                  "label": s.label, "observed_calls": s.calls}
    return out


def _rules_for_prompt(ctx: CompileContext) -> dict:
    return {"labels": [{"label": r.label, "when": r.when.to_dict(), "quote": r.quote} for r in ctx.label_rules],
            "requirements": [{"id": r.id, "kind": r.kind, "a": r.a.to_dict(),
                              "b": r.b.to_dict() if r.b else None, "quote": r.quote} for r in ctx.requirements],
            "terminal_conditions": [{"terminal": tc.terminal,
                                     "required_evidence": [p.to_dict() for p in tc.required_evidence],
                                     "invalidating_events": [p.to_dict() for p in tc.invalidating_events],
                                     "quote": tc.quote} for tc in ctx.terminal_conditions]}


def initialize(ctx: CompileContext, *, model: Any, rounds: int = 3, lenient: bool = False,
               progress=None) -> InitResult:
    """M_0 = the first candidate that passes G_init; then clear clauses, normalize, and run the update stage's check."""
    res = InitResult()
    previous, errors = None, []
    for attempt in range(1, max(1, rounds) + 1):
        if progress:
            progress(f"initialization round {attempt}/{rounds}")
        values = {"document": ctx.skill_text, "clauses": {c[0]: c[1][:300] for c in ctx.clauses},
                  "skill_id": ctx.skill_id, "tools": _tools_for_prompt(ctx),
                  "task_inputs": {k: {"present_in": v.present_in, "total": v.total}
                                  for k, v in ctx.task_inputs.items()},
                  "terminals": list(ctx.terminals), "rules": _rules_for_prompt(ctx),
                  "format_example": FORMAT_EXAMPLE}
        if previous is not None:
            values["previous_candidate"] = previous
            values["errors"] = errors
        raw = None
        machine = None
        try:
            raw = model.generate(prompt=PROMPT, values=values)
            data = _unwrap(raw)
            if not isinstance(data, dict):
                raise ValueError("the reply is not a JSON object")
            data.setdefault("format", "efsm-v1")
            data["skill_id"] = ctx.skill_id
            data.setdefault("fallback", "FALLBACK")
            machine = Machine.model_validate(data)
            errors = g_init(machine, ctx)
        except ModelUnavailable:
            raise                                                 # the endpoint failed: another round cannot help
        except Exception as exc:                                  # noqa: BLE001
            errors = [f"format: {type(exc).__name__}: {str(exc)[:600]}"]
            machine = None
        res.attempts.append({"attempt": attempt, "errors": list(errors),
                             "n_states": len(machine.states) if machine else 0})
        if progress:
            progress("  " + ("passed" if not errors else "failed: " + "; ".join(e[:120] for e in errors[:4])))
        if machine is not None and not errors:
            res.machine = machine
            break
        previous = _unwrap(raw) if isinstance(raw, (dict, str)) else raw
    if res.machine is None:
        return res
    m = res.machine
    res.clause_map = {sid: st.clause for sid, st in m.states.items() if st.clause}
    for st in m.states.values():
        st.clause = ""
    res.notes = normalize(m, ctx)
    res.check_errors = _check.check(m, ctx)
    if res.check_errors and not lenient:
        res.machine = None
    return res


__all__ = ["FORMAT_EXAMPLE", "InitResult", "PROMPT", "RULES_PROMPT", "extract_rules", "g_init",
           "initialize", "install_rules", "normalize"]
