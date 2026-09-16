"""编译上下文：这份技能的编译输入，全部来自文档、工具定义、轨迹与显式规则。

编译器不预设技能处理文件、执行修改或用某个库。它知道的只有：

* 任务输入字段        ← 轨迹头部的 ``task.input``（只说明出现过，``present_in`` 记次数）
* 工具及其接口        ← 工具注册表 / 后端定义（接口保证），轨迹观察补充（推断，不是保证）
* 标签规则            ← 技能规则：给事件打派生标签（如「修改之后读产出」），附文档原文
* 必须执行的步骤      ← 技能规则：must_occur / before / forbid，附文档原文
* 完成与验收条件      ← 技能规则：进入某终点需要哪些证据、哪些事件使证据失效

规则来自技能目录里的 ``compile.json``、``--rules`` 文件，或初始化时由模型从文档抽取并经
原文核对（见 :mod:`.init`）。更新算法本身只执行这些检查，不含任何技能名称判断。

事件模式（:class:`EventPattern`）是规则的原子：按事件类型、工具名、标签、参数文本、成败匹配。
``args_contain`` 里的 ``${field}`` 按当前任务的输入取值代入（路径类取值同时认最后一段文件名）。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from hexis.tools.toolspec import STATUS_KEYS, ToolSpec, observe
from hexis.traces.trace_adapter import tool_output

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class FieldSpec:
    name: str
    present_in: int = 0
    total: int = 0
    sample: Any = None
    source: str = "traces"

    @property
    def always_present(self) -> bool:
        return self.total > 0 and self.present_in == self.total


@dataclass
class EventPattern:
    kind: Optional[str] = None            # tool / model / judge / user / end
    tool: Optional[str] = None
    label: Optional[str] = None           # 基础或派生标签
    role: Optional[str] = None            # model 事件：narration / output
    args_contain: Optional[str] = None    # 参数文本须包含（可含 ${field}）
    arg_regex: Optional[str] = None
    success: Optional[bool] = None
    after: Optional["EventPattern"] = None    # 之前须出现过匹配它的事件

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None and k != "after"}
        if self.after is not None:
            d["after"] = self.after.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "EventPattern":
        after = d.get("after")
        return cls(kind=d.get("kind"), tool=d.get("tool"), label=d.get("label"), role=d.get("role"),
                   args_contain=d.get("args_contain"), arg_regex=d.get("arg_regex"),
                   success=d.get("success"),
                   after=cls.from_dict(after) if isinstance(after, Mapping) else None)

    def describe(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self.to_dict().items() if k != "after"]
        if self.after is not None:
            parts.append(f"after=({self.after.describe()})")
        return " ".join(parts) or "任意事件"


@dataclass
class LabelRule:
    label: str
    when: EventPattern
    quote: str = ""
    source: str = "rules"


@dataclass
class Requirement:
    id: str
    kind: str                              # must_occur / before / forbid
    a: EventPattern
    b: Optional[EventPattern] = None
    quote: str = ""
    source: str = "rules"

    def describe(self) -> str:
        if self.kind == "before":
            return f"{self.id}: ({self.a.describe()}) 须先于 ({self.b.describe() if self.b else ''})"
        return f"{self.id}: {self.kind} ({self.a.describe()})"


@dataclass
class TerminalCondition:
    terminal: str
    required_evidence: list = field(default_factory=list)      # list[EventPattern]
    invalidating_events: list = field(default_factory=list)    # list[EventPattern]
    quote: str = ""
    source: str = "rules"


@dataclass
class CompileContext:
    skill_id: str
    skill_text: str
    clauses: list = field(default_factory=list)                # [(id, text, locator)]
    task_inputs: dict = field(default_factory=dict)            # name → FieldSpec
    tools: dict = field(default_factory=dict)                  # name → ToolSpec
    terminals: list = field(default_factory=list)              # [{"id", "kind"}]
    label_rules: list = field(default_factory=list)
    requirements: list = field(default_factory=list)
    terminal_conditions: list = field(default_factory=list)
    event_kinds: Counter = field(default_factory=Counter)
    n_traces: int = 0
    notes: list = field(default_factory=list)

    # ---- 便捷 ---- #
    def spec(self, tool: str) -> ToolSpec:
        s = self.tools.get(tool)
        if s is None:
            s = ToolSpec(name=tool, source="inferred")
            self.tools[tool] = s
        return s

    def input_keys(self) -> list[str]:
        return list(self.task_inputs)

    def default_terminal(self) -> str:
        """没有证据条件的首个非回退终点。"""
        conditioned = {tc.terminal for tc in self.terminal_conditions if tc.required_evidence}
        for t in self.terminals:
            if t["id"] not in conditioned and t.get("kind") != "fallback":
                return t["id"]
        return self.terminals[0]["id"] if self.terminals else "done"

    def conditioned_terminals(self) -> list[str]:
        return [tc.terminal for tc in self.terminal_conditions if tc.required_evidence]

    def patterns(self) -> list[EventPattern]:
        """规则里出现的全部模式（判「必经状态」用）。"""
        out: list[EventPattern] = []
        for r in self.requirements:
            out.append(r.a)
            if r.b is not None:
                out.append(r.b)
        for tc in self.terminal_conditions:
            out.extend(tc.required_evidence)
            out.extend(tc.invalidating_events)
        return out

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id, "n_traces": self.n_traces,
            "task_inputs": {k: {"present_in": v.present_in, "total": v.total, "source": v.source,
                                "sample": _clip(v.sample)} for k, v in self.task_inputs.items()},
            "tools": {k: v.to_dict() for k, v in self.tools.items()},
            "terminals": list(self.terminals),
            "label_rules": [{"label": r.label, "when": r.when.to_dict(), "quote": r.quote,
                             "source": r.source} for r in self.label_rules],
            "requirements": [{"id": r.id, "kind": r.kind, "a": r.a.to_dict(),
                              "b": r.b.to_dict() if r.b else None, "quote": r.quote,
                              "source": r.source} for r in self.requirements],
            "terminal_conditions": [{"terminal": tc.terminal,
                                     "required_evidence": [p.to_dict() for p in tc.required_evidence],
                                     "invalidating_events": [p.to_dict() for p in tc.invalidating_events],
                                     "quote": tc.quote, "source": tc.source}
                                    for tc in self.terminal_conditions],
            "event_kinds": dict(self.event_kinds), "notes": list(self.notes),
        }


def _clip(v: Any, n: int = 120) -> Any:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


# --------------------------------------------------------------------------- #
# 规则文件
# --------------------------------------------------------------------------- #
def parse_rules(data: Mapping, *, source: str = "rules") -> dict:
    """规则 JSON → (terminals, label_rules, requirements, terminal_conditions)。"""
    terminals = [dict(t) if isinstance(t, Mapping) else {"id": str(t), "kind": ""}
                 for t in (data.get("terminals") or [])]
    labels = [LabelRule(label=str(r["label"]), when=EventPattern.from_dict(r.get("when") or {}),
                        quote=str(r.get("quote") or ""), source=source)
              for r in (data.get("labels") or []) if r.get("label")]
    reqs = []
    for i, r in enumerate(data.get("requirements") or [], 1):
        kind = str(r.get("kind") or "must_occur")
        if kind not in ("must_occur", "before", "forbid"):
            continue
        reqs.append(Requirement(id=str(r.get("id") or f"R{i}"), kind=kind,
                                a=EventPattern.from_dict(r.get("a") or r.get("when") or {}),
                                b=EventPattern.from_dict(r["b"]) if isinstance(r.get("b"), Mapping) else None,
                                quote=str(r.get("quote") or ""), source=source))
    tcs = [TerminalCondition(terminal=str(t["terminal"]),
                             required_evidence=[EventPattern.from_dict(p) for p in (t.get("required_evidence") or [])],
                             invalidating_events=[EventPattern.from_dict(p) for p in (t.get("invalidating_events") or [])],
                             quote=str(t.get("quote") or ""), source=source)
           for t in (data.get("terminal_conditions") or []) if t.get("terminal")]
    return {"terminals": terminals, "label_rules": labels, "requirements": reqs,
            "terminal_conditions": tcs}


def load_rules(path: Any) -> dict:
    return parse_rules(json.loads(Path(path).read_text(encoding="utf-8")))


def quote_in_document(quote: str, doc: str) -> bool:
    """原文核对：忽略空白差异后是文档的子串。空引文不算。"""
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()
    q = norm(quote)
    return bool(q) and q in norm(doc)


# --------------------------------------------------------------------------- #
# 事件匹配（动态：在轨迹事件上）
# --------------------------------------------------------------------------- #
def substitute(template: str, task_input: Mapping) -> list[str]:
    """``${field}`` 代入任务输入。路径类取值另给一个只含最后一段的版本。返回可选的文本列表。"""
    if not template:
        return []
    outs = [template]
    for m in _VAR.findall(template):
        val = task_input.get(m)
        if val is None:
            return []
        s = str(val)
        alts = [s]
        if "/" in s and s.rsplit("/", 1)[-1]:
            alts.append(s.rsplit("/", 1)[-1])
        outs = [o.replace("${" + m + "}", a) for o in outs for a in alts]
    return outs


def event_matches(pat: EventPattern, ev: Any, task_input: Mapping, earlier: Sequence[Any] = ()) -> bool:
    """事件 ``ev`` 是否匹配模式。``ev`` 须有 kind / tool / label / labels / role / ok / args_text。"""
    if pat.kind is not None and getattr(ev, "kind", None) != pat.kind:
        return False
    if pat.tool is not None and getattr(ev, "tool", None) != pat.tool:
        return False
    if pat.label is not None:
        labels = set(getattr(ev, "labels", ()) or ()) | {getattr(ev, "label", "") or ""}
        if pat.label not in labels:
            return False
    if pat.role is not None and getattr(ev, "role", None) != pat.role:
        return False
    if pat.success is not None and bool(getattr(ev, "ok", True)) != pat.success:
        return False
    text = getattr(ev, "args_text", "") or ""
    if pat.args_contain is not None:
        alts = substitute(pat.args_contain, task_input)
        if not alts or not any(a in text for a in alts):
            return False
    if pat.arg_regex is not None:
        try:
            if not re.search(pat.arg_regex, text):
                return False
        except re.error:
            return False
    if pat.after is not None:
        if not any(event_matches(pat.after, e, task_input, ()) for e in earlier):
            return False
    return True


def apply_labels(events: Sequence[Any], ctx: CompileContext, task_input: Mapping) -> None:
    """按标签规则给事件加派生标签（就地，按规则顺序，后面的规则能看到前面的标签）。"""
    for i, ev in enumerate(events):
        if not hasattr(ev, "labels"):
            continue
        for rule in ctx.label_rules:
            if event_matches(rule.when, ev, task_input, events[:i]):
                ev.labels.add(rule.label)


def check_requirements(events: Sequence[Any], ctx: CompileContext, task_input: Mapping) -> list[str]:
    """轨迹违反了哪些要求。返回说明列表（空 = 全满足）。"""
    out: list[str] = []
    for r in ctx.requirements:
        hits_a = [i for i, e in enumerate(events) if event_matches(r.a, e, task_input, events[:i])]
        if r.kind == "must_occur" and not hits_a:
            out.append(f"{r.id}: 缺少必须出现的事件 ({r.a.describe()})" + (f"「{r.quote}」" if r.quote else ""))
        elif r.kind == "forbid" and hits_a:
            out.append(f"{r.id}: 出现了禁止的事件 ({r.a.describe()})" + (f"「{r.quote}」" if r.quote else ""))
        elif r.kind == "before" and r.b is not None:
            for j, e in enumerate(events):
                if event_matches(r.b, e, task_input, events[:j]) and not any(i < j for i in hits_a):
                    out.append(f"{r.id}: 第 {j + 1} 个事件 ({r.b.describe()}) 之前没有 ({r.a.describe()})"
                               + (f"「{r.quote}」" if r.quote else ""))
                    break
    return out


def evidence_held(events: Sequence[Any], ctx: CompileContext, task_input: Mapping) -> set:
    """走完全部事件后仍成立的证据 {(terminal, i)}。成功的证据事件加入，失效事件与失败的证据事件移除。"""
    held: set = set()
    for j, ev in enumerate(events):
        earlier = events[:j]
        for tc in ctx.terminal_conditions:
            for inv in tc.invalidating_events:
                if event_matches(inv, ev, task_input, earlier):
                    held -= {(tc.terminal, i) for i in range(len(tc.required_evidence))}
            for i, pat in enumerate(tc.required_evidence):
                relaxed = EventPattern(**{**pat.__dict__, "success": None})
                if not event_matches(relaxed, ev, task_input, earlier):
                    continue
                ok = bool(getattr(ev, "ok", True))
                if pat.success is None or ok == pat.success:
                    held.add((tc.terminal, i))
                elif pat.success and not ok:
                    held.discard((tc.terminal, i))
    return held


def terminal_for(events: Sequence[Any], ctx: CompileContext, task_input: Mapping) -> tuple[str, set]:
    """τ*(T)：证据齐全的首个带条件终点；没有就是默认终点。"""
    held = evidence_held(events, ctx, task_input)
    for tc in ctx.terminal_conditions:
        if tc.required_evidence and all((tc.terminal, i) in held for i in range(len(tc.required_evidence))):
            return tc.terminal, held
    return ctx.default_terminal(), held


# --------------------------------------------------------------------------- #
# 构造上下文
# --------------------------------------------------------------------------- #
def _args_text(inp: Any) -> str:
    try:
        return json.dumps(inp, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(inp)


def build_context(skill_id: str, skill_text: str, clauses: Sequence[tuple], traces: Iterable[Any], *,
                  registry: Optional[Mapping[str, ToolSpec]] = None,
                  rules: Optional[Mapping] = None) -> CompileContext:
    """从技能文档、工具注册表、轨迹与规则构造编译上下文。``traces`` 是 Trace 对象序列。"""
    ctx = CompileContext(skill_id=skill_id, skill_text=skill_text, clauses=list(clauses))
    for name, spec in (registry or {}).items():
        ctx.tools[name] = ToolSpec(**{**spec.__dict__, "observed_inputs": Counter(),
                                      "observed_outputs": Counter(), "calls": 0})
    traces = list(traces)
    ctx.n_traces = len(traces)
    for tr in traces:
        task_in = tr.task.get("input") if isinstance(tr.task, dict) else None
        for k, v in (task_in or {}).items():
            fs = ctx.task_inputs.setdefault(str(k), FieldSpec(name=str(k), sample=v))
            fs.present_in += 1
        for r in tr.records:
            act = r.action or {}
            kind = str(act.get("kind") or "")
            ctx.event_kinds[kind] += 1
            if kind == "tool":
                name = str(act.get("name") or "")
                inp = act.get("input") or act.get("args") or {}
                observe(ctx.spec(name), inp if isinstance(inp, Mapping) else {}, tool_output(r))
    for fs in ctx.task_inputs.values():
        fs.total = len(traces)
    if rules:
        parsed = parse_rules(rules) if not isinstance(rules, dict) or "label_rules" not in rules else rules
        ctx.terminals = list(parsed["terminals"])
        ctx.label_rules = list(parsed["label_rules"])
        ctx.requirements = list(parsed["requirements"])
        ctx.terminal_conditions = list(parsed["terminal_conditions"])
    if not ctx.terminals:
        ctx.terminals = [{"id": "done", "kind": "done"}]
        ctx.notes.append("规则没有给终点表，用单一终点 done")
    known = {t["id"] for t in ctx.terminals}
    for tc in ctx.terminal_conditions:
        if tc.terminal not in known:
            ctx.terminals.append({"id": tc.terminal, "kind": tc.terminal.lower()})
            known.add(tc.terminal)
    inferred = sorted(n for n, s in ctx.tools.items() if s.source == "inferred")
    if inferred:
        ctx.notes.append(f"工具 {inferred} 没有注册表定义，接口只来自轨迹观察，不是保证")
    for k, v in ctx.task_inputs.items():
        if not v.always_present:
            ctx.notes.append(f"任务输入 {k} 只在 {v.present_in}/{v.total} 条轨迹里出现")
    return ctx


__all__ = ["CompileContext", "EventPattern", "FieldSpec", "LabelRule", "Requirement",
           "STATUS_KEYS", "TerminalCondition", "apply_labels", "build_context", "check_requirements",
           "event_matches", "evidence_held", "load_rules", "parse_rules", "quote_in_document",
           "substitute", "terminal_for"]
