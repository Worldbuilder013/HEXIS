"""候选机器构造（算法文档第 5 节）：M' = Modify(M_k, π*)。全部修改在副本上进行。

状态怎么造，由 :class:`StepContract` 决定：这一步的目的、读什么、写什么、用哪个工具、
前置与后置条件、它落实的文档原文。合同的内容来自事件（叙述、调用参数）、工具定义与技能规则；
本模块只把合同组织成提示词与参数模板，不含任何技能、工具或字段的名字。

* 工具状态的参数模板：与某个任务输入取值完全相同的字段 → ``${field}``；注册表标为常量的
  字段与非字符串值 → 原样保留；其余字符串字段 → 由前置的生成状态（模型）在运行时生成。
* 产出字段按工具定义收；语义变量绑到工具的主要输出。
* 已有转移增加使用次数；缺少的新增，条件用工具自己的成功判据。同一条件多目标、多条无条件
  → 判断状态。多次调用 → 循环判断。首个事件够不到入口 → 调整入口。
* 回边装计数，上限 K_q = ⌈1.5·max{1,N_q}⌉，达到上限转入回退状态；已有上限继续保留，
  访问更多次时抬高。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .. import cond as _cond
from ..schema import (EndAction, JudgeAction, Machine, ModelAction, State, Terminal, ToolAction,
                      Transition, UserAction, Variable)
from ..toolspec import ToolSpec
from .align import NEW, Alignment
from .common import (ABSTAIN, back_edges, branch_class, counter_exit, entry_anchors, guaranteed,
                     maybe_true, need, reaches, seed_vars, status_known, summary_of, task_inputs,
                     zero_paths)
from .context import CompileContext, event_matches
from .traces import Event, Prepared
from . import check as _check

ORIGIN_TRACE = "trace"
ORIGIN_COMPILER = "compiler"


# --------------------------------------------------------------------------- #
# 步骤合同
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
    examples: list = field(default_factory=list)          # 轨迹里观察到的调用参数（原样，截断）

    def prompt(self, sid: str, kind: str) -> str:
        """把合同组织成一段提示词。措辞不含任何技能与工具知识，全部来自合同字段。"""
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

    # ---- 小工具 ---- #
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
            self.note(f"补上回退状态 {fb}")

    def _sort(self, sid: str) -> None:
        st = self.m.states[sid]
        st.transitions = [t for t in st.transitions if t.cond] + [t for t in st.transitions if not t.cond]

    def counter_guards(self, sid: str, cond: str) -> str:
        """给一条新的带条件转移并上该状态已有的计数上限（cnt < K），与上限出口互斥。"""
        if not cond:
            return cond
        terms = [cond]
        for t in self.m.states[sid].transitions:
            ce = counter_exit(t.cond)
            if ce and f"{ce[0]} <" not in cond:
                terms.append(f"{ce[0]} < {ce[1]}")
        return " and ".join(terms)

    # ---- 合同 ---- #
    def contract(self, ev: Event, *, tool: Optional[ToolSpec] = None,
                 inputs: Sequence[str] = (), outputs: Sequence[str] = ()) -> StepContract:
        """从事件、工具定义和技能规则拼出这一步的合同。"""
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

    # ---- 状态构造 ---- #
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
        """参数模板与需要生成的变量。"""
        base = dict(ev.calls[-1].input) if ev.calls else {}
        task_in = self.prep.task_input
        by_value: dict = {}
        for k, v in task_in.items():
            if isinstance(v, (str, int, float)) and str(v):
                by_value.setdefault(str(v), k)
                if isinstance(v, str) and "/" in v and v.rsplit("/", 1)[-1]:
                    by_value.setdefault(v.rsplit("/", 1)[-1], k)     # 同名文件的相对写法
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
        """新建（或重建）一个工具状态的动作，返回它的入口（生成门或自身）。

        重建时保留原有的输出映射里源字段在新工具产出中仍存在的那些；仍未绑定的语义变量绑到主要输出。"""
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
            self.note(f"新增状态 {sid}：{ev.describe()}")
        elif ev.kind == "model":
            self.make_model_state(sid, ev)
            self.note(f"新增可观察模型状态 {sid}")
        elif ev.kind == "user":
            self.make_user_state(sid, ev)
            self.note(f"新增用户状态 {sid}")
        return sid

    def gates_of(self, sid: str) -> list[str]:
        """专门给 sid 生成参数的模型状态：零宽、除计数上限出口外唯一后继是 sid。"""
        out = []
        for g, st in self.m.states.items():
            if st.action.kind != "model" or getattr(st.action, "observable", False):
                continue
            succ = [t.to for t in st.transitions if not (t.to == self.m.fallback and counter_exit(t.cond))]
            if succ == [sid]:
                out.append(g)
        return out

    def retire_gate(self, old: str, new: str) -> None:
        """旧生成门让位给新生成门：进旧门的边改进新门，旧门的上限出口搬过去，旧门删除。"""
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
        self.note(f"生成门 {old} 由 {new} 取代")

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
        self.note(f"更换工具 {sid}：{old_tool} → {ev.tool}")

    def add_labels(self, sid: str, ev: Event) -> None:
        a = self.m.states[sid].action
        have = set(getattr(a, "labels", []) or [])
        if ev.labels - have:
            a.labels = sorted(have | set(ev.labels))
            self.note(f"{sid}: 标签 {sorted(have)} → {a.labels}")

    def redirect_in_edges(self, sid: str, entry: str) -> None:
        """把进入 sid 的边改到它的生成门；计数上限出口跟着搬。"""
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
        """复用状态时补充缺失的输出映射：未绑定的语义变量绑到主要输出（每次只绑首个）。"""
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
            self.note(f"{sid}: 输出映射 {spec.primary} → {semantic[0]}")

    # ---- 变量供给 ---- #
    def generator_for(self, missing: set[str], q: str) -> Optional[str]:
        """能写出 missing 且能到达 q 的模型状态；优先只经零宽状态到达的，其次路径最短的。"""
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
        """进入 q 该从哪儿进：所需变量不在 A_0 里时，找它的生成状态。"""
        st = self.m.states[q]
        if st.action.kind != "tool":
            return q
        missing = set(need(st)) - set(seed_vars(self.m))
        if not missing:
            return q
        return self.generator_for(missing, q) or q

    # ---- 转移 ---- #
    def bump(self, path) -> None:
        for _sid, t in path:
            t.support = int(t.support or 0) + 1

    def success_cond(self, p: str, b: Optional[bool]) -> str:
        """从工具状态 p 出发、结果为 b 的新转移条件：工具的成功判据或它的否定。"""
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
                self.note(f"{p}→{q} 缺少 {sorted(missing)}，没有模型状态能补，交由检查决定")
        self.add_edge(p, cond, target)
        return "add"

    def is_our_judge(self, sid: str) -> bool:
        """由更新阶段引入的判断状态（分岔判断与循环判断）：可以继续加选项。"""
        st = self.m.states.get(sid)
        return st is not None and st.action.kind == "judge" and st.origin == ORIGIN_TRACE

    def same_condition(self, p: str, existing: str, new: str) -> bool:
        """两条护卫算不算「同一条件」：都无条件，或成功 / 失败分支类别相同（计数项不参与）。"""
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
        if same:                                       # 同一条件对应多个目标 → 判断状态
            e = same[0]
            if self.is_our_judge(e.to):
                self.extend_judge(e.to, target)
                return
            if cond == "":                             # 多条无条件转移：使用次数最多的做默认
                default, other = (e.to, target) if int(e.support or 0) >= 1 else (target, e.to)
            else:
                default, other = e.to, target
            jid = self.make_judge(p, default, other)
            e.to = jid
            e.support = int(e.support or 0) + 1
            self._sort(p)
            return
        uncond = [t for t in st.transitions if not t.cond]
        if cond and uncond:                            # 新的条件边会遮住已有的无条件边
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
        self.note(f"新增转移 {p} → {target}" + (f" [{cond}]" if cond else ""))

    def judge_reads(self, p: str) -> list[str]:
        """判断优先读前一步的内容产出（最多三个，排除状态字段）；缺少时读首个任务输入；再没有读状态字段。"""
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
        self.note(f"新增判断 {jid}（{p} 之后：默认 {default}，另选 {other}）")
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
        self.note(f"判断 {jid} 增加选项 {target}")

    # ---- 循环 ---- #
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
        exits = [t for t in orig if counter_exit(t.cond)]                  # 上限出口只属于 q
        rest = [t for t in orig if not counter_exit(t.cond)]
        if succ:
            ok_known, fail_known = status_known(self.ctx, st, True), status_known(self.ctx, st, False)
            keep = exits + [t for t in rest if maybe_true(t.cond, fail_known)]   # 失败侧留在 q
            moved = [t for t in rest if maybe_true(t.cond, ok_known)]           # 成功侧搬进循环判断
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
        self.note(f"增加循环判断 {lid}（回到 {back}）")

    # ---- 计数 ---- #
    def install_counters(self, visits: dict) -> None:
        for src, e in back_edges(self.m):
            tgt = e.to
            if not e.inc:
                e.inc = f"{tgt}_count"
                self.note(f"回边 {src}→{tgt} 装计数 {e.inc}")
            self.ensure_var(e.inc, "integer", 0)
            cnt = e.inc
            t = self.m.states[tgt]
            k = math.ceil(1.5 * max(1, int(visits.get(tgt, 1))))
            old = next((counter_exit(g.cond) for g in t.transitions if counter_exit(g.cond)
                        and counter_exit(g.cond)[0] == cnt), None)
            if old is not None:                           # 已有计数和上限出口继续保留
                if k > old[1]:                            # 候选路径访问更多次：抬高上限（全机器同名引用一起改）
                    for st2 in self.m.states.values():
                        for g in st2.transitions:
                            g.cond = re.sub(rf"\b{re.escape(cnt)} (>=|<) {old[1]}\b",
                                            lambda mm: f"{cnt} {mm.group(1)} {k}", g.cond or "")
                    self.note(f"{tgt}: 上限 {cnt} {old[1]} → {k}")
                continue
            if any(g.cond and cnt in _cond.vars_of(g.cond) for g in t.transitions):
                continue
            for g in t.transitions:
                if g.cond:
                    g.cond = f"{cnt} < {k} and ({g.cond})"
            t.transitions.insert(0, Transition(cond=f"{cnt} >= {k}", to=self.m.fallback,
                                               origin=ORIGIN_COMPILER))
            self.note(f"{tgt}: 上限 {cnt} >= {k} → {self.m.fallback}")


def build_candidate(m: Machine, prep: Prepared, alignment: Alignment, ctx: CompileContext) -> Build:
    """按对齐路径构造候选机器。返回候选、对齐后的可观察状态路径与修改记录。"""
    bd = Builder(m, prep, ctx)
    events = prep.events
    resolved: list[str] = []
    # 1. 状态：新增 / 更换工具 / 补标签与映射
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
    # 2. 入口
    first = alignment.slots[0]
    q1 = resolved[0]
    if first.edge == "start":
        entry = bd.entry_of(q1)
        bd.m.initial = entry
        bd.note(f"入口改为 {entry}")
    else:
        ea = entry_anchors(bd.m)
        if q1 in ea:
            bd.bump(ea[q1])
    # 3. 转移
    for i in range(1, len(resolved)):
        prev = events[alignment.slots[i - 1].index]
        b = prev.ok if prev.kind == "tool" else None
        bd.connect(resolved[i - 1], resolved[i], b)
    # 4. 循环
    for slot, sid in zip(alignment.slots, resolved):
        if slot.loop and bd.m.states[sid].action.kind == "tool":
            bd.loop_judge(sid)
    # 5. 候选路径上的访问次数 → 计数上限
    rp = _check.replay(bd.m, prep, resolved, ignore_counters=True)
    visits = rp.visits if rp.ok else {}
    if not rp.ok:
        bd.note(f"装计数前的路径推演未通过：{rp.why}")
    bd.install_counters(visits)
    bd.m.max_steps = max(int(bd.m.max_steps or 0), 3 * len(bd.m.states) + 16)
    return Build(machine=bd.m, anchors=resolved, changes=bd.changes)


__all__ = ["Build", "Builder", "ORIGIN_COMPILER", "ORIGIN_TRACE", "StepContract", "build_candidate"]
