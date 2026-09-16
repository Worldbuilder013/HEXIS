"""A model that decides trace steps for the stepwise update.

For each step of a new trace, :class:`ModelDecider` shows the model the step (tool, arguments, results, narration),
the task inputs, the skill's clauses and the candidate states computed by :func:`hexis.compiler.stepwise.candidates`,
and asks for one decision::

    {"decision": "match" | "new" | "ignore" | "exclude", "state": <candidate id or null>,
     "purpose": "<one sentence>", "clause": "<clause id or empty>"}

The deterministic proposal is not shown to the model, so it does not anchor the answer; it is used only when the
model gives no valid answer after one repair turn. Answers are cached by (model, question) in ``decisions.jsonl``:
running an update again asks nothing that was already answered.

Wrong decisions cannot break the machine: every candidate machine must pass the static checks and replay the new
trace and every previously accepted trace before it is accepted. A wrong decision can add an unnecessary state or
transition, or drop a step that no rule protects; the decisions are logged so they can be audited.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from hexis.compiler.context import CompileContext
from hexis.compiler.decide import StepView, decide_trace
from hexis.machine.schema import Machine, Trace

PROMPT_VERSION = "step-v1"
MAX_CANDIDATES = 16
CLAUSE_CHARS = 160
CLAUSES_TOTAL = 6000
INPUT_CHARS = 300
NARRATION_CHARS = 400
ARG_CHARS = 600
RESULT_CHARS = 400
PURPOSE_CHARS = 300

STEP_PROMPT = """You map one step of a recorded agent run onto a state machine compiled from a skill document.

The machine has one state per workflow action: a tool call, a deliverable written by the model, or a question to the user. A recorded run of the skill is being walked through step by step. For the CURRENT step, decide how the machine accounts for it:

- "match": a candidate state produces this step. It calls the same tool for the same purpose in the skill's workflow, in this situation. Arguments and results differ between tasks; judge by purpose. Prefer candidates with "reachable": true (the machine already continues there from the previous step). Choose a candidate with "reachable": false only when the step clearly serves that state's purpose and the machine merely lacks the transition. A candidate with "repeat": true is the state of the previous step; choose it when this step retries or continues that action.
- "new": the step is part of the skill's workflow, but no candidate serves its purpose. A new state will be added.
- "ignore": the step is not part of the skill's workflow: harness or environment noise such as listing the working directory, reading the task statement, checking installed tools or versions, or a throwaway experiment whose result is not used. Ignored steps are removed before the machine is updated. Do not ignore a step only because it failed or repeats an earlier step.
- "exclude": the whole run cannot be mapped onto the skill's workflow, for example because the real work happens inside a script whose effects the steps do not show, or because the run does not attempt the task. Give the reason in "purpose". Use this rarely.

Inputs in VARIABLES: skill and clauses (the skill document's clauses, id -> text, possibly shortened); task (the task inputs of this run); steps_before (earlier steps with the decision taken for each) and ignored_steps; position (the state of the previous step: "start" before the first step, "new state" when the previous step was new) and whether that step succeeded; step (the current step: kind, tool, base label, derived labels, number of calls, success, the narration before it, and its calls with arguments and shortened results); steps_after (a preview of the following steps); candidates (states that could produce this step, each with id, reachable, repeat, action, labels, argument template, description, clause, and the states that follow it). The candidate list can be empty; then answer "new", "ignore" or "exclude".

Return exactly one JSON object:
{"decision": "match" | "new" | "ignore" | "exclude", "state": "<candidate id when decision is match, otherwise null>", "purpose": "<one sentence in the skill's terms: what this step does; for exclude, the reason>", "clause": "<id from clauses that this step carries out, or an empty string>"}"""

REPAIR_NOTE = ("Your previous answer was invalid: {problem}. Answer again with exactly one JSON object that follows the "
               "rules above.")


# --------------------------------------------------------------------------- #
# question
# --------------------------------------------------------------------------- #
def _short(value: Any, n: int) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _clauses(ctx: CompileContext) -> dict:
    out: dict = {}
    total = 0
    for i, c in enumerate(ctx.clauses):
        text = _short(c[1], CLAUSE_CHARS)
        if total + len(text) > CLAUSES_TOTAL:
            out["..."] = f"{len(ctx.clauses) - i} more clauses omitted"
            break
        out[str(c[0])] = text
        total += len(text)
    return out


def _step(view: StepView) -> dict:
    ev = view.event
    d: dict = {"step": view.step, "kind": ev.kind}
    if ev.kind == "tool":
        calls = list(ev.calls)
        shown = calls if len(calls) <= 3 else calls[:2] + calls[-1:]
        d.update({"tool": ev.tool, "base_label": ev.label, "labels": sorted(ev.labels), "calls": len(calls),
                  "succeeded": ev.ok, "narration": _short(ev.intent, NARRATION_CHARS),
                  "call_details": [{"arguments": _short(c.input, ARG_CHARS), "result": _short(c.output, RESULT_CHARS),
                                    "succeeded": bool(c.ok)} for c in shown],
                  "omitted_calls": len(calls) - len(shown)})
    elif ev.kind == "model":
        d.update({"role": ev.role, "text": _short(ev.text, NARRATION_CHARS), "narration": _short(ev.intent, NARRATION_CHARS)})
    else:
        d.update({"text": _short(ev.text, NARRATION_CHARS)})
    return d


def _card(machine: Machine, sid: str, view: StepView) -> dict:
    st = machine.states[sid]
    a = st.action
    card: dict = {"id": sid, "reachable": sid in view.tier1, "repeat": sid == view.position,
                  "action": f"{a.kind} {getattr(a, 'name', '')}".strip(), "labels": sorted(getattr(a, "labels", []) or [])}
    if a.kind == "tool":
        card["arguments"] = _short(a.input, ARG_CHARS)
    desc = str(getattr(st, "description", "") or "")
    if not desc:
        gate = next((g for g, gs in machine.states.items() if gs.action.kind == "model"
                     and not getattr(gs.action, "observable", False) and [t.to for t in gs.transitions] == [sid]), None)
        desc = machine.states[gate].action.prompt if gate else getattr(a, "prompt", "")
    card["description"] = _short(desc, 240)
    card["clause"] = st.clause
    card["next"] = [t.to for t in st.ordered_transitions()]
    return card


def question(view: StepView, machine: Machine, ctx: CompileContext) -> tuple[dict, list[str]]:
    """The VARIABLES of one question and the candidate ids shown in it."""
    ordered = list(view.tier1) + [s for s in view.tier2 if s not in view.tier1]
    shown = ordered[:MAX_CANDIDATES]
    position = ("start" if view.position is None else "new state" if view.is_new_position else view.position)
    values = {
        "skill": ctx.skill_id,
        "clauses": _clauses(ctx),
        "task": {k: _short(v, INPUT_CHARS) for k, v in sorted((view.prep.task_input or {}).items())},
        "steps_before": [{"step": s, "event": e, "decision": dec} for s, e, dec in view.before],
        "ignored_steps": list(view.ignored),
        "position": {"state": position, "previous_step_succeeded": view.previous_ok},
        "step": _step(view),
        "steps_after": list(view.after),
        "candidates": [_card(machine, sid, view) for sid in shown],
        "omitted_candidates": len(ordered) - len(shown),
    }
    return values, shown


def validate(answer: Any, shown: list[str], clause_ids: set) -> tuple[Optional[dict], str]:
    """Return (normalized answer, "") or (None, problem)."""
    if not isinstance(answer, dict):
        return None, "the answer is not a JSON object"
    d = str(answer.get("decision") or "").strip().lower()
    if d not in ("match", "new", "ignore", "exclude"):
        return None, f"decision must be one of match, new, ignore, exclude (got {answer.get('decision')!r})"
    state = answer.get("state")
    purpose = str(answer.get("purpose") or "").strip()[:PURPOSE_CHARS]
    clause = str(answer.get("clause") or "").strip()
    if clause not in clause_ids:
        clause = ""
    if d == "match":
        if not shown:
            return None, "there are no candidates, so the decision cannot be match"
        if str(state) not in shown:
            return None, f"state {state!r} is not one of the candidates {shown}"
        return {"decision": d, "state": str(state), "purpose": purpose, "clause": clause}, ""
    if d in ("new", "exclude") and not purpose:
        return None, f"a {d} decision needs a purpose"
    return {"decision": d, "state": None, "purpose": purpose, "clause": clause}, ""


def cache_key(model_id: str, values: dict) -> str:
    blob = json.dumps({"v": PROMPT_VERSION, "model": model_id, "values": values}, ensure_ascii=False, sort_keys=True,
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# cache / audit log
# --------------------------------------------------------------------------- #
class DecisionLog:
    """``decisions.jsonl``: one record per question; the last record of a key wins."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path is not None else None
        self._index: dict = {}
        if self.path is not None and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("key"):
                        self._index[rec["key"]] = rec

    def get(self, key: str) -> Optional[dict]:
        return self._index.get(key)

    def append(self, record: dict) -> None:
        self._index[record["key"]] = record
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# --------------------------------------------------------------------------- #
# decider
# --------------------------------------------------------------------------- #
class ModelDecider:
    """Decide trace steps with a model (any object with ``generate(prompt=, values=)``)."""

    name = "model"

    def __init__(self, model: Any, *, model_id: str, log: Optional[DecisionLog] = None, use_cache: bool = True,
                 say: Optional[Callable[[str], None]] = None):
        self.model = model
        self.model_id = model_id
        self.log = log or DecisionLog(None)
        self.use_cache = use_cache
        self.say = say
        self.stats = {"asked": 0, "cached": 0, "repaired": 0, "fallback": 0}

    def _usage(self) -> dict:
        u = getattr(self.model, "usage", None)
        return dict(u()) if callable(u) else {}

    def decide(self, view: StepView, machine: Machine, ctx: CompileContext) -> dict:
        values, shown = question(view, machine, ctx)
        key = cache_key(self.model_id, values)
        if self.use_cache:
            hit = self.log.get(key)
            if hit and isinstance(hit.get("decision"), dict):
                self.stats["cached"] += 1
                return dict(hit["decision"])
        clause_ids = {str(c[0]) for c in ctx.clauses}
        attempts: list = []
        before = self._usage()
        decision: Optional[dict] = None
        problem = ""
        for turn in range(2):
            vals = values if turn == 0 else {**values, "previous_answer": attempts[-1]["answer"], "problem": problem}
            prompt = STEP_PROMPT if turn == 0 else STEP_PROMPT + "\n\n" + REPAIR_NOTE.format(problem=problem)
            try:
                raw = self.model.generate(prompt=prompt, values=vals)
            except Exception as exc:
                from hexis.llm.model_iface import ModelUnavailable
                if isinstance(exc, ModelUnavailable):
                    raise
                raw = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            self.stats["asked" if turn == 0 else "repaired"] += 1
            decision, problem = validate(raw, shown, clause_ids)
            attempts.append({"answer": raw, "problem": problem})
            if decision is not None:
                break
        source = "model"
        if decision is None:
            self.stats["fallback"] += 1
            p = view.proposal or {"d": "new"}
            decision = {"decision": p.get("d", "new"), "state": p.get("state"), "purpose": "", "clause": ""}
            if decision["decision"] == "new":
                decision["purpose"] = "added after the model gave no valid decision"
            source = "fallback"
        after = self._usage()
        usage = {k: after.get(k, 0) - before.get(k, 0) for k in ("llm_calls", "prompt_tokens", "completion_tokens")
                 if isinstance(after.get(k), int)}
        self.log.append({"key": key, "prompt_version": PROMPT_VERSION, "model": self.model_id, "trace": view.trace,
                         "step": view.step, "question": values, "attempts": attempts, "decision": decision,
                         "source": source, "proposal": view.proposal, "usage": usage,
                         "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if self.say:
            self.say(f"    step {view.step:>3} {view.event.describe():<28} → {decision['decision']}"
                     + (f" {decision['state']}" if decision.get("state") else "") + (" (fallback)" if source == "fallback" else ""))
        return decision

    def spec_for(self, machine: Machine, ctx: CompileContext, trace: Trace, *, source: str = "", key: str = "") -> dict:
        return decide_trace(machine, ctx, trace, source=source, key=key,
                            ask=lambda view: self.decide(view, machine, ctx))


__all__ = ["DecisionLog", "ModelDecider", "PROMPT_VERSION", "STEP_PROMPT", "cache_key", "question", "validate"]
