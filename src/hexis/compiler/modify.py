"""Candidate machine construction: M' = Modify(M_k, π*). All modifications are made on a copy.

How a state is built is decided by its :class:`StepContract`: the purpose of the step, what it reads, what it
writes, which tool it uses, its preconditions and postconditions, and the document text it implements. The contract
content comes from events (narration, call arguments), tool definitions and skill rules; this module only turns the
contract into prompts and parameter templates and contains no skill, tool or field names.

* Parameter template of a tool state: a field whose value equals some task input value exactly → ``${field}``;
  fields the registry marks as constant, and non-string values → kept verbatim; other string fields → generated at
  run time by a preceding generation state (the model).
* Output fields are collected according to the tool definition; semantic variables are bound to the tool's primary
  output.
* Existing transitions get their usage count increased; missing ones are added, guarded by the tool's own success
  condition. The same guard with several targets, or several unconditional edges → judge state. Several calls →
  loop judge. First event not reachable from the entry → adjust the entry.
* Back edges get a counter with bound K_q = ⌈1.5·max{1,N_q}⌉; reaching the bound leads to the fallback state;
  existing bounds are kept and raised when the state is visited more often.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from hexis.compiler import check as _check
from hexis.compiler.align import NEW, Alignment
from hexis.compiler.common import (
    ABSTAIN,
    back_edges,
    branch_class,
    counter_exit,
    entry_anchors,
    guaranteed,
    maybe_true,
    need,
    reaches,
    seed_vars,
    status_known,
    summary_of,
    task_inputs,
    zero_paths,
)
from hexis.compiler.context import CompileContext, event_matches
from hexis.compiler.traces import Event, Prepared
from hexis.machine import cond as _cond
from hexis.machine.schema import (
    EndAction,
    JudgeAction,
    Machine,
    ModelAction,
    State,
    Terminal,
    ToolAction,
    Transition,
    UserAction,
    Variable,
)
from hexis.tools.toolspec import ToolSpec

ORIGIN_TRACE = "trace"
ORIGIN_COMPILER = "compiler"


# --------------------------------------------------------------------------- #
# Step contract
# --------------------------------------------------------------------------- #
@dataclass
class StepContract:
    purpose: str
    input_fields: list = field(default_factory=list)
    output_fields: list = field(default_factory=list)
    tool_name: Optional[str] = None
    preconditions: list = field(default_factory=list)
    postconditions: list = field(default_factory=list)
    source_requirements: list = field(default_factory=list)
    examples: list = field(default_factory=list)          # call arguments observed in traces (verbatim, truncated)

    def prompt(self, sid: str, kind: str) -> str:
        """Turn the contract into a prompt. The wording carries no skill or tool knowledge; everything comes from the
        contract fields."""
        what = {"gate": f"produce the values {self.output_fields} that the next step needs",
                "output": f"produce the deliverable content of this step as {self.output_fields}",
                "user": f"ask the user and record the answer as {self.output_fields}"}[kind]
        parts = [f"Step {sid}: {what}."]
        if self.tool_name:
            parts.append(f"The next step calls the tool '{self.tool_name}'.")
        if self.purpose:
            parts.append(f"Purpose of this step: {self.purpose}")
        if self.input_fields:
            parts.append(f"Inputs available in VARIABLES: {self.input_fields}. Use their values verbatim "
                         f"where they identify task objects; do not invent identifiers.")
        if self.preconditions:
            parts.append("Constraints: " + " ".join(self.preconditions))
        if self.postconditions:
            parts.append("Expected outcome: " + " ".join(self.postconditions))
        if self.source_requirements:
            parts.append("Skill document requirements for this step: "
                         + " | ".join(f"\"{q}\"" for q in self.source_requirements))
        if self.examples:
            parts.append("Examples of parameter values used for this step in recorded runs (adapt the shape, "
                         "not the task-specific content): " + " || ".join(self.examples))
        parts.append(f"Return exactly one JSON object with the keys {self.output_fields}; every value must be "
                     f"a plain JSON string (escape quotes and newlines correctly).")
        return " ".join(parts)


@dataclass
class Build:
    machine: Machine
    anchors: list = field(default_factory=list)
    changes: list = field(default_factory=list)


class Builder:
    def __init__(self, m: Machine, prep: Prepared, ctx: CompileContext) -> None:
        self.m = m.model_copy(deep=True)
        self.prep = prep
        self.ctx = ctx
        self.changes: list[str] = []
        self._ensure_fallback()

    # ---- helpers ---- #
    def note(self, s: str) -> None:
        self.changes.append(s)

    def new_id(self, prefix: str) -> str:
        n = 1
        while f"{prefix}{n}" in self.m.states:
            n += 1
        return f"{prefix}{n}"

    def unique(self, base: str) -> str:
        if base not in self.m.states:
            return base
        n = 2
        while f"{base}{n}" in self.m.states:
            n += 1
        return f"{base}{n}"

    def ensure_var(self, name: str, vtype: str = "string", init=None) -> None:
        if self.m.var(name) is None:
            self.m.variables.append(Variable(name=name, type=vtype, init=init))

    def _ensure_fallback(self) -> None:
        fb = self.m.fallback
        if fb not in self.m.states:
            tid = next((t.id for t in self.m.terminals if t.kind == "fallback"), "END_FALLBACK")
            if not any(t.id == tid for t in self.m.terminals):
                self.m.terminals.append(Terminal(id=tid, kind="fallback"))
            self.m.states[fb] = State(id=fb, action=EndAction(terminal=tid), origin=ORIGIN_COMPILER)
            self.note(f"added missing fallback state {fb}")

    def _sort(self, sid: str) -> None:
        st = self.m.states[sid]
        st.transitions = [t for t in st.transitions if t.cond] + [t for t in st.transitions if not t.cond]

    def counter_guards(self, sid: str, cond: str) -> str:
        """Conjoin a new guarded transition with the counter bounds the state already has (cnt < K), so that it is
        mutually exclusive with the bound exits."""
        if not cond:
            return cond
        terms = [cond]
        for t in self.m.states[sid].transitions:
            ce = counter_exit(t.cond)
            if ce and f"{ce[0]} <" not in cond:
                terms.append(f"{ce[0]} < {ce[1]}")
        return " and ".join(terms)

    # ---- contract ---- #
    def contract(self, ev: Event, *, tool: Optional[ToolSpec] = None,
                 inputs: Sequence[str] = (), outputs: Sequence[str] = ()) -> StepContract:
        """Assemble the contract of this step from the event, the tool definition and the skill rules."""
        task_in = self.prep.task_input
        earlier = self.prep.events[:ev.index]
        quotes: list[str] = []
        for rule in self.ctx.label_rules:
            if rule.quote and event_matches(rule.when, ev, task_in, earlier):
                quotes.append(rule.quote)
        for r in self.ctx.requirements:
            if r.quote and (event_matches(r.a, ev, task_in, earlier)
                            or (r.b is not None and event_matches(r.b, ev, task_in, earlier))):
                quotes.append(r.quote)
        for tc in self.ctx.terminal_conditions:
            if tc.quote and any(event_matches(p, ev, task_in, earlier) for p in tc.required_evidence):
                quotes.append(tc.quote)
        pre, post = [], []
        if tool is not None:
            if tool.description:
                pre.append(f"Tool '{tool.name}': {tool.description}")
            req = [k for k, v in tool.input_schema.items() if isinstance(v, dict) and v.get("required")]
            if req:
                pre.append(f"Required parameters: {req}.")
            if tool.success:
                post.append(f"The call succeeds when '{tool.success}'.")
            if tool.source == "inferred":
                pre.append("This tool has no formal definition; its interface is inferred from observed calls.")
        purpose = (ev.intent or "").strip()[:400]
        examples: list[str] = []
        if ev.kind == "tool":
            for call in [c for c in ev.calls if c.ok][-2:] or ev.calls[-1:]:
                examples.append(json.dumps(call.input, ensure_ascii=False)[:700])
        return StepContract(purpose=purpose, input_fields=list(inputs), output_fields=list(outputs),
                            tool_name=tool.name if tool else None, preconditions=pre,
                            postconditions=post, source_requirements=list(dict.fromkeys(quotes)),
                            examples=examples)

    # ---- state construction ---- #
    def make_gate(self, sid: str, ev: Event, vars_: list[str], spec: ToolSpec,
                  extra_reads: Sequence[str] = ()) -> str:
        gid = self.unique(f"{sid}_gate")
        reads = list(dict.fromkeys(list(task_inputs(self.m))
                                   + [r for r in extra_reads if self.m.var(r) is not None]))
        for v in vars_:
            self.ensure_var(v)
        ct = self.contract(ev, tool=spec, inputs=reads, outputs=vars_)
        self.m.states[gid] = State(
            id=gid, origin=ORIGIN_TRACE,
            action=ModelAction(prompt=ct.prompt(sid, "gate"), reads=reads, writes=list(vars_)),
            transitions=[Transition(cond="", to=sid, origin=ORIGIN_TRACE)])
        return gid

    def tool_template(self, sid: str, ev: Event, spec: ToolSpec) -> tuple[dict, list[str]]:
        """Parameter template and the variables that must be generated."""
        base = dict(ev.calls[-1].input) if ev.calls else {}
        task_in = self.prep.task_input
        by_value: dict = {}
        for k, v in task_in.items():
            if isinstance(v, (str, int, float)) and str(v):
                by_value.setdefault(str(v), k)
                if isinstance(v, str) and "/" in v and v.rsplit("/", 1)[-1]:
                    by_value.setdefault(v.rsplit("/", 1)[-1], k)     # relative spelling of the same file
        constants = spec.constant_keys()
        template: dict = {}
        generated: list[str] = []
        for key, value in base.items():
            if isinstance(value, str) and value in by_value:
                template[key] = "${" + by_value[value] + "}"
            elif key in constants or not isinstance(value, str):
                template[key] = value
            else:
                var = f"{sid}_{key}"
                template[key] = "${" + var + "}"
                generated.append(var)
        return template, generated

    def make_tool_state(self, sid: str, ev: Event, *, origin: str,
                        keep_writes: Sequence[str] = (), keep_binds: Optional[dict] = None,
                        extra_reads: Sequence[str] = ()) -> str:
        """Create (or rebuild) the action of a tool state and return its entry (its gate or the state itself).

        When rebuilding, keep the existing output mappings whose source field is still produced by the new tool;
        semantic variables that remain unbound are bound to the primary output."""
        spec = self.ctx.spec(ev.tool)
        known = set(spec.output_keys())
        semantic = [w for w in keep_writes if w not in known]
        template, generated = self.tool_template(sid, ev, spec)
        writes = list(dict.fromkeys(sorted(spec.sure_outputs()) + ([spec.primary] if spec.primary else [])
                                    + [w for w in semantic]))
        binds: dict = {src: dst for src, dst in (keep_binds or {}).items()
                       if src in known and dst in semantic}
        unbound = [w for w in semantic if w not in binds.values()]
        if unbound and spec.primary and spec.primary not in binds:
            binds[spec.primary] = unbound[0]
        action = ToolAction(name=ev.tool, input=template, reads=[], writes=writes, phase=ev.label,
                            labels=sorted(ev.labels), binds=binds)
        if sid in self.m.states:
            self.m.states[sid].action = action
        else:
            self.m.states[sid] = State(id=sid, action=action, origin=origin)
        return self.make_gate(sid, ev, generated, spec, extra_reads) if generated else sid

    def make_model_state(self, sid: str, ev: Event) -> str:
        var = f"{sid}_output"
        self.ensure_var(var)
        reads = list(task_inputs(self.m))
        ct = self.contract(ev, inputs=reads, outputs=[var])
        self.m.states[sid] = State(id=sid, origin=ORIGIN_TRACE,
                                   action=ModelAction(prompt=ct.prompt(sid, "output"), reads=reads,
                                                      writes=[var], observable=True,
                                                      labels=sorted(ev.labels)))
        return sid

    def make_user_state(self, sid: str, ev: Event) -> str:
        var = f"{sid}_answer"
        self.ensure_var(var)
        ct = self.contract(ev, outputs=[var])
        self.m.states[sid] = State(id=sid, origin=ORIGIN_TRACE,
                                   action=UserAction(prompt=ct.prompt(sid, "user"), writes=[var],
                                                     labels=sorted(ev.labels)))
        return sid

    def add_state(self, ev: Event) -> str:
        sid = self.new_id("t")
        if ev.kind == "tool":
            self.make_tool_state(sid, ev, origin=ORIGIN_TRACE)
            self.note(f"added state {sid}: {ev.describe()}")
        elif ev.kind == "model":
            self.make_model_state(sid, ev)
            self.note(f"added observable model state {sid}")
        elif ev.kind == "user":
            self.make_user_state(sid, ev)
            self.note(f"added user state {sid}")
        return sid

    def gates_of(self, sid: str) -> list[str]:
        """Model states that only generate parameters for sid: zero-width, and apart from counter bound exits their only
        successor is sid."""
        out = []
        for g, st in self.m.states.items():
            if st.action.kind != "model" or getattr(st.action, "observable", False):
                continue
            succ = [t.to for t in st.transitions if not (t.to == self.m.fallback and counter_exit(t.cond))]
            if succ == [sid]:
                out.append(g)
        return out

    def retire_gate(self, old: str, new: str) -> None:
        """The old gate gives way to the new gate: edges into the old gate now enter the new gate, the old gate's bound
        exits move over, and the old gate is deleted."""
        if self.m.initial == old:
            self.m.initial = new
        exits = [t for t in self.m.states[old].transitions if counter_exit(t.cond)]
        for src, st in self.m.states.items():
            if src == old:
                continue
            for t in st.transitions:
                if t.to == old:
                    t.to = new
        for e in exits:
            self.m.states[new].transitions.insert(0, e)
        del self.m.states[old]
        self.note(f"gate {old} replaced by {new}")

    def realize(self, sid: str, ev: Event) -> None:
        st = self.m.states[sid]
        old = st.action
        if old.kind != "tool":
            return
        old_gates = self.gates_of(sid)
        old_vars = list(dict.fromkeys(list(old.reads) + list(need(st))))
        old_tool = old.name
        entry = self.make_tool_state(sid, ev, origin=st.origin, keep_writes=list(old.writes),
                                     keep_binds=dict(getattr(old, "binds", {}) or {}), extra_reads=old_vars)
        if entry != sid:
            self.redirect_in_edges(sid, entry)
            for g in old_gates:
                if g != entry and g in self.m.states:
                    self.retire_gate(g, entry)
        self.note(f"changed tool of {sid}: {old_tool} → {ev.tool}")

    def add_labels(self, sid: str, ev: Event) -> None:
        a = self.m.states[sid].action
        have = set(getattr(a, "labels", []) or [])
        if ev.labels - have:
            a.labels = sorted(have | set(ev.labels))
            self.note(f"{sid}: labels {sorted(have)} → {a.labels}")

    def redirect_in_edges(self, sid: str, entry: str) -> None:
        """Redirect the edges into sid to its gate; counter bound exits move along."""
        if self.m.initial == sid:
            self.m.initial = entry
        for src, st in self.m.states.items():
            if src == entry:
                continue
            for t in st.transitions:
                if t.to == sid:
                    t.to = entry
                    if t.inc:
                        self._move_exit(sid, entry, t.inc)

    def _move_exit(self, frm: str, to: str, cnt: str) -> None:
        src = self.m.states[frm]
        exits = [t for t in src.transitions if counter_exit(t.cond) and counter_exit(t.cond)[0] == cnt]
        for e in exits:
            src.transitions.remove(e)
            self.m.states[to].transitions.insert(0, e)

    def complete_binds(self, sid: str) -> None:
        """When a state is reused, add missing output mappings: an unbound semantic variable is bound to the primary
        output (only the first one each time)."""
        st = self.m.states[sid]
        a = st.action
        if a.kind != "tool":
            return
        spec = self.ctx.spec(a.name)
        known = set(spec.output_keys())
        bound = set((a.binds or {}).values())
        semantic = [w for w in a.writes if w not in known and w not in bound]
        if semantic and spec.primary and spec.primary not in (a.binds or {}):
            a.binds = dict(a.binds or {})
            a.binds[spec.primary] = semantic[0]
            self.note(f"{sid}: output mapping {spec.primary} → {semantic[0]}")

    # ---- variable supply ---- #
    def generator_for(self, missing: set[str], q: str) -> Optional[str]:
        """A model state that can write missing and can reach q; prefer one that reaches q through zero-width states
        only, then the one with the shortest path."""
        cands = [sid for sid, st in self.m.states.items()
                 if st.action.kind == "model" and not getattr(st.action, "observable", False)
                 and missing <= set(st.action.writes)]
        best: Optional[tuple] = None
        for sid in cands:
            zp = zero_paths(self.m, sid, {}, include_src_zero=True)
            if q in zp:
                key = (0, len(zp[q]), sid)
            elif reaches(self.m, sid, q):
                key = (1, 0, sid)
            else:
                continue
            if best is None or key < best:
                best = key
        return best[2] if best else None

    def entry_of(self, q: str) -> str:
        """Where to enter q: when the variables it needs are not in A_0, find the state that generates them."""
        st = self.m.states[q]
        if st.action.kind != "tool":
            return q
        missing = set(need(st)) - set(seed_vars(self.m))
        if not missing:
            return q
        return self.generator_for(missing, q) or q

    # ---- transitions ---- #
    def bump(self, path) -> None:
        for _sid, t in path:
            t.support = int(t.support or 0) + 1

    def success_cond(self, p: str, b: Optional[bool]) -> str:
        """Guard of a new transition leaving tool state p with outcome b: the tool's success condition or its negation."""
        st = self.m.states[p]
        if b is None or st.action.kind != "tool":
            return ""
        spec = self.ctx.spec(st.action.name)
        if not spec.success:
            return ""
        try:
            vars_ = _cond.vars_of(spec.success)
        except _cond.CondError:
            return ""
        if not set(vars_) <= set(guaranteed(st, self.ctx)):
            return ""
        return spec.success if b else f"not ({spec.success})"

    def connect(self, p: str, q: str, b: Optional[bool]) -> str:
        paths = zero_paths(self.m, p, status_known(self.ctx, self.m.states[p], b))
        if q in paths:
            self.bump(paths[q])
            return "keep"
        pst = self.m.states[p]
        cond = self.counter_guards(p, self.success_cond(p, b))
        an = _check.analyze(self.m, self.ctx)
        avail = set(an.avail.get(p, seed_vars(self.m))) | set(guaranteed(pst, self.ctx))
        target = q
        missing = set(need(self.m.states[q])) - avail
        if missing:
            g = self.generator_for(missing, q)
            if g is not None:
                target = g
            else:
                self.note(f"{p}→{q} lacks {sorted(missing)} and no model state can supply it; left to the check")
        self.add_edge(p, cond, target)
        return "add"

    def is_our_judge(self, sid: str) -> bool:
        """Judge states introduced by the update stage (branch judges and loop judges): more options may be added."""
        st = self.m.states.get(sid)
        return st is not None and st.action.kind == "judge" and st.origin == ORIGIN_TRACE

    def same_condition(self, p: str, existing: str, new: str) -> bool:
        """Whether two guards count as "the same condition": both unconditional, or the same success / failure branch
        class (counter terms are ignored)."""
        if not new or not existing:
            return (not new) and (not existing)
        st = self.m.states[p]
        return branch_class(existing, self.ctx, st) == branch_class(new, self.ctx, st)

    def add_edge(self, p: str, cond: str, target: str) -> None:
        st = self.m.states[p]
        same = [t for t in st.transitions if self.same_condition(p, t.cond, cond)]
        for t in same:
            if t.to == target:
                t.support = int(t.support or 0) + 1
                return
        if same:                                       # same guard with several targets → judge state
            e = same[0]
            if self.is_our_judge(e.to):
                self.extend_judge(e.to, target)
                return
            if cond == "":                             # several unconditional transitions: the most used one is the default
                default, other = (e.to, target) if int(e.support or 0) >= 1 else (target, e.to)
            else:
                default, other = e.to, target
            jid = self.make_judge(p, default, other)
            e.to = jid
            e.support = int(e.support or 0) + 1
            self._sort(p)
            return
        uncond = [t for t in st.transitions if not t.cond]
        if cond and uncond:                            # the new guarded edge would shadow the existing unconditional edge
            e = uncond[0]
            if self.is_our_judge(e.to):
                self.extend_judge(e.to, target)
                jid = e.to
            else:
                jid = self.make_judge(p, e.to, target)
            st.transitions.append(Transition(cond=cond, to=jid, support=1, origin=ORIGIN_TRACE))
            self._sort(p)
            return
        st.transitions.append(Transition(cond=cond, to=target, support=1, origin=ORIGIN_TRACE))
        if cond and not any(not t.cond for t in st.transitions):
            st.transitions.append(Transition(cond="", to=self.m.fallback, origin=ORIGIN_COMPILER))
        self._sort(p)
        self.note(f"added transition {p} → {target}" + (f" [{cond}]" if cond else ""))

    def judge_reads(self, p: str) -> list[str]:
        """A judge preferably reads the content outputs of the previous step (at most three, status fields excluded);
        otherwise the first task input; failing that, a status field."""
        st = self.m.states[p]
        g = guaranteed(st, self.ctx)
        status = set(self.ctx.spec(st.action.name).status_keys) if st.action.kind == "tool" else set()
        outs = [w for w in getattr(st.action, "writes", []) if w in g and w not in status][:3]
        if outs:
            return outs
        x = task_inputs(self.m)
        if x:
            return [x[0]]
        rest = [w for w in getattr(st.action, "writes", []) if w in g]
        return rest[:1] or sorted(seed_vars(self.m))[:1] or ["__none__"]

    def make_judge(self, p: str, default: str, other: str) -> str:
        jid = self.unique(f"{p}_j")
        var = f"{jid}_choice"
        self.ensure_var(var)
        prompt = (f"After {summary_of(self.m.states[p])}, decide which step comes next. "
                  f"Options: '{default}' = {summary_of(self.m.states[default])}; "
                  f"'{other}' = {summary_of(self.m.states[other])}. Answer '{other}' only when the "
                  f"outputs show that step is needed now; otherwise answer '{default}'.")
        self.m.states[jid] = State(
            id=jid, origin=ORIGIN_TRACE,
            action=JudgeAction(prompt=prompt, reads=self.judge_reads(p), writes=[var],
                               labels=[default, other, ABSTAIN], abstain=ABSTAIN),
            transitions=[Transition(cond=f"{var} == '{other}'", to=other, support=1, origin=ORIGIN_TRACE),
                         Transition(cond="", to=default, origin=ORIGIN_TRACE)])
        self.note(f"added judge {jid} (after {p}: default {default}, alternative {other})")
        return jid

    def extend_judge(self, jid: str, target: str) -> None:
        st = self.m.states[jid]
        a = st.action
        var = a.writes[0]
        if target in a.labels:
            for t in st.transitions:
                if t.to == target:
                    t.support = int(t.support or 0) + 1
            return
        a.labels = [l for l in a.labels if l != a.abstain] + [target, a.abstain]
        a.prompt += f" Option '{target}' = {summary_of(self.m.states[target])}."
        st.transitions.append(Transition(cond=f"{var} == '{target}'", to=target, support=1,
                                         origin=ORIGIN_TRACE))
        self._sort(jid)
        self.note(f"judge {jid} gained option {target}")

    # ---- loops ---- #
    def loop_judge(self, q: str) -> None:
        lid = f"{q}_loop"
        if lid in self.m.states:
            for t in self.m.states[lid].transitions:
                if t.cond == f"{lid}_choice == 'again'":
                    t.support = int(t.support or 0) + 1
            return
        st = self.m.states[q]
        var = f"{lid}_choice"
        self.ensure_var(var)
        back = self.entry_of(q)
        succ = self.success_cond(q, True)
        orig = list(st.transitions)
        exits = [t for t in orig if counter_exit(t.cond)]                  # bound exits belong to q only
        rest = [t for t in orig if not counter_exit(t.cond)]
        if succ:
            ok_known, fail_known = status_known(self.ctx, st, True), status_known(self.ctx, st, False)
            keep = exits + [t for t in rest if maybe_true(t.cond, fail_known)]   # the failure side stays on q
            moved = [t for t in rest if maybe_true(t.cond, ok_known)]           # the success side moves into the loop judge
            first_cond = self.counter_guards(q, succ)
        else:
            keep, moved, first_cond = exits, rest, ""
        st.transitions = [Transition(cond=first_cond, to=lid, support=1, origin=ORIGIN_TRACE)] + keep
        self._sort(q)
        edges = [Transition(cond=f"{var} == 'again'", to=back, support=1, origin=ORIGIN_TRACE)]
        for t in moved:
            c = f"{var} == 'continue' and ({t.cond})" if t.cond else ""
            edges.append(Transition(cond=c, to=t.to, inc=t.inc, support=t.support, origin=t.origin))
        if not any(not e.cond for e in edges):
            edges.append(Transition(cond="", to=(moved[-1].to if moved else self.m.fallback),
                                    origin=ORIGIN_COMPILER))
        prompt = (f"Step {summary_of(st)} may need to run more than once. Given its output, answer "
                  f"'again' if the step must be repeated with new parameters, otherwise 'continue'.")
        self.m.states[lid] = State(
            id=lid, origin=ORIGIN_TRACE,
            action=JudgeAction(prompt=prompt, reads=self.judge_reads(q), writes=[var],
                               labels=["again", "continue", ABSTAIN], abstain=ABSTAIN),
            transitions=edges)
        self.note(f"added loop judge {lid} (back to {back})")

    # ---- counters ---- #
    def install_counters(self, visits: dict) -> None:
        for src, e in back_edges(self.m):
            tgt = e.to
            if not e.inc:
                e.inc = f"{tgt}_count"
                self.note(f"back edge {src}→{tgt} gets counter {e.inc}")
            self.ensure_var(e.inc, "integer", 0)
            cnt = e.inc
            t = self.m.states[tgt]
            k = math.ceil(1.5 * max(1, int(visits.get(tgt, 1))))
            old = next((counter_exit(g.cond) for g in t.transitions if counter_exit(g.cond)
                        and counter_exit(g.cond)[0] == cnt), None)
            if old is not None:                           # an existing counter and bound exit are kept
                if k > old[1]:                            # the candidate path visits more often: raise the bound (every reference in the machine changes too)
                    for st2 in self.m.states.values():
                        for g in st2.transitions:
                            g.cond = re.sub(rf"\b{re.escape(cnt)} (>=|<) {old[1]}\b",
                                            lambda mm: f"{cnt} {mm.group(1)} {k}", g.cond or "")
                    self.note(f"{tgt}: bound {cnt} {old[1]} → {k}")
                continue
            if any(g.cond and cnt in _cond.vars_of(g.cond) for g in t.transitions):
                continue
            for g in t.transitions:
                if g.cond:
                    g.cond = f"{cnt} < {k} and ({g.cond})"
            t.transitions.insert(0, Transition(cond=f"{cnt} >= {k}", to=self.m.fallback,
                                               origin=ORIGIN_COMPILER))
            self.note(f"{tgt}: bound {cnt} >= {k} → {self.m.fallback}")


def build_candidate(m: Machine, prep: Prepared, alignment: Alignment, ctx: CompileContext) -> Build:
    """Build the candidate machine along the alignment path. Returns the candidate, the aligned path of observable
    states and the change log."""
    bd = Builder(m, prep, ctx)
    events = prep.events
    resolved: list[str] = []
    # 1. states: add / change tool / complete labels and mappings
    for slot in alignment.slots:
        ev = events[slot.index]
        if slot.is_new:
            sid = bd.add_state(ev)
        else:
            sid = slot.state
            if slot.how == "realize":
                bd.realize(sid, ev)
            elif slot.how == "label":
                bd.add_labels(sid, ev)
                bd.complete_binds(sid)
            elif ev.kind == "tool":
                bd.complete_binds(sid)
        resolved.append(sid)
    # 2. entry
    first = alignment.slots[0]
    q1 = resolved[0]
    if first.edge == "start":
        entry = bd.entry_of(q1)
        bd.m.initial = entry
        bd.note(f"entry changed to {entry}")
    else:
        ea = entry_anchors(bd.m)
        if q1 in ea:
            bd.bump(ea[q1])
    # 3. transitions
    for i in range(1, len(resolved)):
        prev = events[alignment.slots[i - 1].index]
        b = prev.ok if prev.kind == "tool" else None
        bd.connect(resolved[i - 1], resolved[i], b)
    # 4. loops
    for slot, sid in zip(alignment.slots, resolved):
        if slot.loop and bd.m.states[sid].action.kind == "tool":
            bd.loop_judge(sid)
    # 5. visit counts along the candidate path → counter bounds
    rp = _check.replay(bd.m, prep, resolved, ignore_counters=True)
    visits = rp.visits if rp.ok else {}
    if not rp.ok:
        bd.note(f"path replay before installing counters failed: {rp.why}")
    bd.install_counters(visits)
    bd.m.max_steps = max(int(bd.m.max_steps or 0), 3 * len(bd.m.states) + 16)
    return Build(machine=bd.m, anchors=resolved, changes=bd.changes)


__all__ = ["Build", "Builder", "ORIGIN_COMPILER", "ORIGIN_TRACE", "StepContract", "build_candidate"]
