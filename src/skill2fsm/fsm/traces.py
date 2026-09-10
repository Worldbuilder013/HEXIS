"""轨迹预处理（算法文档第 3 节，事件化）。

轨迹按文件名顺序读取。每条记录折成一个**事件**，五种：模型生成、工具调用、判断结果、
用户输入、任务结束。每个事件保留自己的输入、输出与前后依赖；工具名保持原样，不合并、不换名。
认不出的记录类型不会被丢掉：整条轨迹标为「含无法识别的事件」，不参与更新。

* 同一工具、同一基础标签的连续调用合并成一个事件（多次调用 = 循环需求）。
* 基础标签只对通用命令类工具按「有没有写操作」分 apply / probe；其他工具用注册表给的标签。
  此外一律空。**派生标签**（比如「修改之后读产出」）全部来自技能规则（见 :mod:`.context`）。
* 模型事件分 narration（两次操作之间的叙述，附到下一个事件的意图里）和 output（结构化产出，
  或结束前的最后一段生成）。后者是可观察事件。
* 结束类别 τ*(T) 由技能的终点条件决定；轨迹声明的终点与证据不符即违规。
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .. import cond as _cond
from .. import judge as _judge
from .. import phases as _phases
from ..schema import Machine, Trace
from ..toolspec import ToolSpec
from ..trace_adapter import load_any_trace, tool_output
from .context import CompileContext, apply_labels, check_requirements, terminal_for

EVENT_KINDS = ("tool", "model", "judge", "user", "end")


@dataclass
class Call:
    step: int
    input: dict
    output: dict
    ok: bool
    narration: str = ""


@dataclass
class Event:
    kind: str                                   # tool / model / judge / user / end
    index: int = 0
    tool: str = ""                              # tool 事件
    label: str = ""                             # 基础标签
    labels: set = field(default_factory=set)    # 派生标签
    calls: list = field(default_factory=list)   # tool 事件的全部调用
    role: str = ""                              # model 事件：narration / output
    text: str = ""                              # model 事件的正文 / user 事件的输入
    output: dict = field(default_factory=dict)  # model / judge 事件的产出
    step: int = 0                               # 非 tool 事件的记录步号
    terminal: str = ""                          # end 事件声明的终点
    intent: str = ""                            # 附上来的叙述

    @property
    def ok(self) -> bool:
        if self.kind == "tool":
            return bool(self.calls) and bool(self.calls[-1].ok)
        return True

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def observable(self) -> bool:
        return self.kind in ("tool", "user", "end") or (self.kind == "model" and self.role == "output")

    @property
    def args_text(self) -> str:
        if self.kind == "tool":
            return "\n".join(_dumps(c.input) for c in self.calls)
        if self.kind == "model":
            return self.text
        return _dumps(self.output)

    def describe(self) -> str:
        if self.kind == "tool":
            labs = "+".join(sorted(self.labels))
            return f"{self.tool}{'/' + self.label if self.label else ''}{'[' + labs + ']' if labs else ''}" \
                   f"×{self.n_calls}{'✓' if self.ok else '✗'}"
        if self.kind == "model":
            return f"model:{self.role}"
        if self.kind == "end":
            return f"end{':' + self.terminal if self.terminal else ''}"
        return self.kind


Segment = Event


@dataclass
class Prepared:
    trace_id: str
    source: str
    verdict: str
    task_input: dict
    events: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)      # [(step, kind)]
    claims: str = ""                                     # 轨迹自己声明的终点
    tau: str = ""                                        # 按终点条件算出的结束类别
    evidence: set = field(default_factory=set)
    violation: Optional[str] = None
    requirement_violations: list = field(default_factory=list)
    n_records: int = 0

    @property
    def observable(self) -> list:
        return [e for e in self.events if e.observable]

    @property
    def tool_events(self) -> list:
        return [e for e in self.events if e.kind == "tool"]

    def obs(self) -> list[tuple]:
        """Obs(T)：可观察事件序列，(类型, 名字, 成败)，末尾 (end, τ*)。"""
        out: list[tuple] = []
        for e in self.observable:
            if e.kind == "tool":
                out.append(("tool", e.tool, e.ok))
            elif e.kind == "model":
                out.append(("model", "", True))
            elif e.kind == "user":
                out.append(("user", "", True))
            elif e.kind == "end":
                out.append(("end", self.tau, True))
        return out


# --------------------------------------------------------------------------- #
def _dumps(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(v)


def _restore_code(inp: dict, meta: dict) -> dict:
    """采集时抽走的代码正文（meta.code_bodies）放回入参；参数只用于记录不用于比较。"""
    bodies = meta.get("code_bodies") if isinstance(meta, dict) else None
    if not isinstance(bodies, dict):
        return inp
    out = dict(inp)
    for k in list(out):
        if k.endswith("_sha256") and out[k] in bodies:
            out[k[:-7]] = bodies[out[k]]
            del out[k]
    return out


def base_label(tool: str, inp: Mapping, spec: ToolSpec) -> str:
    """基础标签。通用命令类工具按写操作分 apply / probe；其余按注册表；再无则空。"""
    if tool.lower() in _phases.GENERIC_TOOLS:
        text = _phases.command_text(inp)
        if text.strip():
            return _phases.PHASE_APPLY if _phases.writes_something(text) else _phases.PHASE_PROBE
    return spec.label or ""


def call_ok(spec: ToolSpec, out: Mapping) -> bool:
    """一次调用成败：按工具的成功判据在产出上求值；判据缺失或求不出按成功。"""
    if not spec.success:
        return True
    try:
        return bool(_cond.evaluate(spec.success, dict(out)))
    except _cond.CondError:
        return True


def segment_trace(trace: Trace, ctx: CompileContext) -> tuple[list, list, list]:
    """轨迹 → 事件序列。返回 (事件, 说明, 无法识别的记录)。"""
    events: list[Event] = []
    notes: list[str] = []
    unsupported: list = []
    pending: list[str] = []
    for r in trace.records:
        act = dict(r.action or {})
        kind = str(act.get("kind") or "")
        if kind == "model":
            out = dict(r.output or {})
            said = str(out.get("reply") or act.get("text") or "").strip()
            structured = {k: v for k, v in out.items() if k not in ("reply", "text")}
            ev = Event(kind="model", step=r.step, text=said[:2000], output=structured,
                       role="output" if structured else "narration", intent=" ".join(pending))
            events.append(ev)
            if said:
                pending.append(said[:400])
            continue
        if kind == "end":
            events.append(Event(kind="end", step=r.step, terminal=str(act.get("terminal") or ""),
                                intent=" ".join(pending)))
            pending = []
            continue
        if kind == "judge":
            events.append(Event(kind="judge", step=r.step, output=dict(r.output or {}),
                                text=str(act.get("prompt") or ""), intent=" ".join(pending)))
            pending = []
            continue
        if kind == "user":
            out = dict(r.output or {})
            events.append(Event(kind="user", step=r.step, output=out,
                                text=str(out.get("answer") or act.get("prompt") or ""),
                                intent=" ".join(pending)))
            pending = []
            continue
        if kind != "tool":
            unsupported.append((r.step, kind or "(空)"))
            continue
        name = str(act.get("name") or "")
        spec = ctx.spec(name)
        inp = act.get("input") or act.get("args") or {}
        inp = _restore_code(deepcopy(inp) if isinstance(inp, dict) else {}, r.meta or {})
        out = deepcopy(tool_output(r))
        ok = call_ok(spec, out)
        label = base_label(name, inp, spec)
        call = Call(step=r.step, input=inp, output=out, ok=ok, narration=" ".join(pending))
        # 同工具、同标签的连续调用合并；中间只隔着叙述（不是产出）的也算连续
        prev = None
        for cand in reversed(events):
            if cand.kind == "model" and cand.role == "narration":
                continue
            prev = cand
            break
        if prev is not None and prev.kind == "tool" and prev.tool == name and prev.label == label:
            prev.calls.append(call)
            if pending:
                prev.intent = (prev.intent + " " + " ".join(pending)).strip()
        else:
            events.append(Event(kind="tool", tool=name, label=label, calls=[call],
                                intent=" ".join(pending)))
        pending = []
    if pending and not (events and events[-1].kind == "end"):
        notes.append(f"末尾 {len(pending)} 句叙述后面没有任何事件")
    # 结束前的最后一段模型生成是交付内容（output），中间的是叙述
    last_model = None
    for i, ev in enumerate(events):
        if ev.kind == "model":
            last_model = i
        elif ev.kind in ("tool", "user"):
            last_model = None
    if last_model is not None and events and events[-1].kind == "end":
        events[last_model].role = "output"
    for i, ev in enumerate(events):
        ev.index = i
    return events, notes, unsupported


def prepare(trace: Trace, ctx: CompileContext, *, source: str = "") -> Prepared:
    """一条轨迹的全部预处理结果：事件、派生标签、要求检查、结束类别、违规。"""
    tid = str(trace.task.get("task_id") or Path(source).stem) if isinstance(trace.task, dict) else source
    task_in = dict((trace.task.get("input") if isinstance(trace.task, dict) else None) or {})
    events, notes, unsupported = segment_trace(trace, ctx)
    p = Prepared(trace_id=tid, source=source, verdict=trace.verdict, task_input=task_in,
                 events=events, notes=notes, unsupported=unsupported, n_records=len(trace.records))
    if not events:
        return p
    if events[-1].kind != "end":
        events.append(Event(kind="end", index=len(events), terminal=""))
        p.notes.append("轨迹没有结束记录，补一个结束事件")
    apply_labels(events, ctx, task_in)
    p.requirement_violations = check_requirements(events, ctx, task_in)
    p.tau, p.evidence = terminal_for(events, ctx, task_in)
    p.claims = events[-1].terminal
    conditioned = set(ctx.conditioned_terminals())
    if p.claims in conditioned and p.claims != p.tau:
        p.violation = f"声明到达 {p.claims}，但证据只支持 {p.tau}"
    return p


def load_traces(tdir: Path, *, phase_rules: str = "") -> list[tuple[Path, Optional[Trace], str]]:
    """按文件名顺序读目录里的 *.jsonl。读不动的 (path, None, 错误)。不做阶段分类：标签由本模块定。"""
    out: list = []
    for f in sorted(Path(tdir).glob("*.jsonl")):
        try:
            out.append((f, load_any_trace(f, phase_rules=phase_rules), ""))
        except Exception as exc:                               # noqa: BLE001
            out.append((f, None, f"{type(exc).__name__}: {exc}"[:300]))
    return out


def violates_prohibitions(trace: Trace, machine: Machine) -> Optional[str]:
    """外部禁止规则（机器的 prohibitions）。触犯返回原因。"""
    if not machine.prohibitions:
        return None
    v = _judge.evaluate(trace, None, list(machine.prohibitions))
    return v.reason if v.verdict == "rejected" else None


def end_state_for(machine: Machine, terminal: str) -> str:
    """结束类别（终点 id）→ 结束状态：对应终点；缺少时非回退终点；再没有就回退状态。"""
    kinds = terminal_kinds(machine)
    for sid, st in machine.states.items():
        if st.action.kind == "end" and st.action.terminal == terminal:
            return sid
    for sid, st in machine.states.items():
        if st.action.kind == "end" and sid != machine.fallback and kinds.get(st.action.terminal) != "fallback":
            return sid
    return machine.fallback


def terminal_kinds(m: Machine) -> dict[str, str]:
    return {t.id: t.kind for t in m.terminals}


__all__ = ["Call", "EVENT_KINDS", "Event", "Prepared", "Segment", "base_label", "call_ok",
           "end_state_for", "load_traces", "prepare", "segment_trace", "violates_prohibitions"]
