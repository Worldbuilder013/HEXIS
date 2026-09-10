"""逐步增量编译的机械部分（docs/COMPILE_STEPWISE.md 第 3–6 节）。

判定者（agent）不在这里。本模块只做三件确定性的事：

1. 把轨迹切成步，算每一步的候选集——tier 1 是从当前位置一跳可达的同类同工具状态，
   tier 2 是其余同类同工具状态（需新增转移，且不得绕过规则提到的文档必经状态）；
2. 把 agent 的逐步判定（match / new / ignore）翻成对齐路径，交给 :func:`modify.build_candidate`
   构造候选，再过 Check、新轨迹回放、已接受轨迹重放——接受规则与 :mod:`update` 相同；
3. 持久化进度（machine.json + progress.json），让一批轨迹可以分批判定、随时续跑。

"提案"是一条确定性的默认规则（tier 1 有候选就取标签最接近的那个，没有就 new），只是为了让
判定者少写字：判定者仍然逐步过目，接受提案也记为它的判定（source=proposal）。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..schema import Machine, Trace
from . import check as _check
from .align import NEW, Alignment, Slot, loop_cost
from .common import anchors, entry_anchors, required_states, skip_blocked, status_known, zero_paths
from .context import CompileContext, apply_labels, check_requirements, terminal_for
from .modify import build_candidate
from .traces import Event, Prepared, end_state_for, prepare, violates_prohibitions


# --------------------------------------------------------------------------- #
# 机器指纹与轨迹准备
# --------------------------------------------------------------------------- #
def fingerprint(m: Machine) -> str:
    parts: list = []
    for sid in sorted(m.states):
        st = m.states[sid]
        a = st.action
        parts.append([sid, a.kind, getattr(a, "name", ""), getattr(a, "phase", ""),
                      sorted(getattr(a, "labels", []) or []), bool(getattr(a, "observable", False))])
        for t in st.transitions:
            parts.append([sid, t.cond, t.to, t.inc or ""])
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def strip_tools(trace: Trace, names: Sequence[str]) -> Trace:
    """剔除 harness 标记工具的记录（如 skill2fsm_begin）。"""
    bad = set(names)
    trace.records = [r for r in trace.records
                     if not ((r.action or {}).get("kind") == "tool" and str((r.action or {}).get("name")) in bad)]
    return trace


def prepare_steps(trace: Trace, ctx: CompileContext, *, source: str = "", ignore: Sequence[int] = (),
                  ignore_calls: Sequence[int] = ()) -> Prepared:
    """prepare() 之后按 agent 的判定删掉不属于技能工作流的步（``ignore``，按事件下标）或某一步里的个别调用
    （``ignore_calls``，按记录步号：一步里混进的环境检查、临时目录重算），再重算标签、要求、结束类别。

    每个事件带 ``orig``：删步之前的下标。判定一律按 ``orig`` 记，删步不会让编号漂移。"""
    prep = prepare(trace, ctx, source=source)
    for ev in prep.events:
        ev.orig = ev.index                                   # type: ignore[attr-defined]
    if ignore or ignore_calls:
        ign = {int(i) for i in ignore}
        bad_calls = {int(i) for i in ignore_calls}
        kept = []
        for ev in prep.events:
            if getattr(ev, "orig", -1) in ign:
                continue
            if ev.kind == "tool" and bad_calls:
                ev.calls = [c for c in ev.calls if c.step not in bad_calls]
                if not ev.calls:
                    continue
            kept.append(ev)
        prep.events = _remerge(kept)
        for i, ev in enumerate(prep.events):
            ev.index = i
            ev.labels = set()
        task_in = prep.task_input
        apply_labels(prep.events, ctx, task_in)
        prep.requirement_violations = check_requirements(prep.events, ctx, task_in)
        prep.tau, prep.evidence = terminal_for(prep.events, ctx, task_in)
        prep.claims = prep.events[-1].terminal if prep.events else ""
        conditioned = set(ctx.conditioned_terminals())
        prep.violation = (f"声明到达 {prep.claims}，但证据只支持 {prep.tau}"
                          if prep.claims in conditioned and prep.claims != prep.tau else None)
    return prep


def _remerge(events: list) -> list:
    """删步之后，重新把连续的同工具、同基础标签调用并成一步（中间只隔叙述也算连续），与 segment_trace 同一规则。"""
    out: list = []
    for ev in events:
        prev = None
        for cand in reversed(out):
            if cand.kind == "model" and cand.role == "narration":
                continue
            prev = cand
            break
        if prev is not None and prev.kind == "tool" and ev.kind == "tool" and prev.tool == ev.tool and prev.label == ev.label:
            prev.calls.extend(ev.calls)
            if ev.intent:
                prev.intent = (prev.intent + " " + ev.intent).strip()
            prev.merged = list(getattr(prev, "merged", [])) + [getattr(ev, "orig", ev.index)]   # type: ignore[attr-defined]
            continue
        out.append(ev)
    return out


def pre_status(prep: Prepared, trace: Trace, machine: Machine) -> Optional[tuple[str, str]]:
    """不需要判定就能定下的状态：unsupported / excluded / violation / skipped。None = 要判定。"""
    banned = violates_prohibitions(trace, machine)
    if prep.unsupported:
        return "unsupported", "无法识别的事件：" + ", ".join(f"第 {s} 步 kind={k}" for s, k in prep.unsupported)
    if banned:
        return "excluded", banned
    if prep.requirement_violations:
        return "excluded", "；".join(prep.requirement_violations)
    if prep.violation:
        return "violation", prep.violation
    if not prep.observable:
        return "skipped", "没有可观察事件"
    return None


# --------------------------------------------------------------------------- #
# 候选集与提案
# --------------------------------------------------------------------------- #
def same_kind(m: Machine, ev: Event, sid: str) -> bool:
    """确定性预筛：类型相同；工具步还要同名工具、同基础标签（probe 状态不会写文件，生成不了 apply 步）。"""
    a = m.states[sid].action
    if ev.kind == "tool":
        return a.kind == "tool" and a.name == ev.tool and (a.phase or "") == (ev.label or "")
    if ev.kind == "model":
        return a.kind == "model" and bool(getattr(a, "observable", False))
    if ev.kind == "user":
        return a.kind == "user"
    if ev.kind == "end":
        return a.kind == "end"
    return False


def candidates(m: Machine, ctx: CompileContext, ev: Event, pos: Optional[str], b: Optional[bool],
               term: str, *, allow_tier2: bool = True) -> tuple[list[str], list[str]]:
    """C(p, b)：(tier 1, tier 2)。end 步只有 τ* 对应的结束状态。"""
    if ev.kind == "end":
        return ([term] if term in m.states else []), []
    if pos is None:
        t1 = list(entry_anchors(m))
    elif pos.startswith(NEW):
        t1 = []
    else:
        t1 = list(zero_paths(m, pos, status_known(ctx, m.states[pos], b)))
    t1 = [s for s in t1 if same_kind(m, ev, s)]
    t2: list[str] = []
    if allow_tier2:
        req = required_states(m, ctx)
        for s in anchors(m):
            if s in t1 or not same_kind(m, ev, s) or m.states[s].action.kind == "end":
                continue
            if pos is not None and not pos.startswith(NEW) and skip_blocked(m, pos, s, req):
                continue
            t2.append(s)
    return t1, t2


def propose(m: Machine, ev: Event, t1: list[str], t2: list[str], term: str = "") -> dict:
    """默认提案：tier 1 有候选取派生标签相同的首个，否则首个；交付步优先取能通向 τ* 终点的；没有 tier 1 → new。"""
    if ev.kind == "end":
        return {"d": "match", "state": t1[0]} if t1 else {"d": "new"}
    if t1:
        pool = t1
        if ev.kind == "model" and term:
            reach = [s for s in t1 if term in zero_paths(m, s, {})]
            pool = reach or t1
        same = [s for s in pool if set(getattr(m.states[s].action, "labels", []) or []) == set(ev.labels)]
        return {"d": "match", "state": (same or pool)[0]}
    return {"d": "new"}


# --------------------------------------------------------------------------- #
# 判定 → 对齐 → 候选 → 检查 → 接受
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    d: str                      # match / new / ignore
    state: str = ""
    purpose: str = ""
    clause: str = ""
    source: str = "agent"       # agent / proposal


def parse_decisions(spec: Any) -> tuple[list[int], dict[int, Decision], bool, list[int]]:
    """一条轨迹的判定：{"accept_proposals": bool, "ignore_calls": [记录步号…],
    "steps": {"<orig>": {"d", "state"?, "purpose"?, "clause"?}}}。"steps" 里 d=ignore 的步在对齐前删掉。"""
    spec = spec or {}
    accept = bool(spec.get("accept_proposals", False))
    ignore_calls = [int(x) for x in (spec.get("ignore_calls") or [])]
    ignore: list[int] = []
    steps: dict[int, Decision] = {}
    for k, v in (spec.get("steps") or {}).items():
        if isinstance(v, str):
            v = {"d": v}
        d = str(v.get("d") or v.get("decision") or "").strip()
        if d == "ignore":
            ignore.append(int(k))
            continue
        steps[int(k)] = Decision(d=d, state=str(v.get("state") or ""), purpose=str(v.get("purpose") or ""),
                                 clause=str(v.get("clause") or ""))
    return ignore, steps, accept, ignore_calls


def align_from_decisions(m: Machine, ctx: CompileContext, prep: Prepared, steps: dict[int, Decision],
                         accept_proposals: bool, *, allow_tier2: bool) -> tuple[Optional[Alignment], str, list]:
    """把逐步判定翻成对齐路径。严格轮（allow_tier2=False）把 tier 2 匹配改为 new。"""
    term = end_state_for(m, prep.tau)
    ea = entry_anchors(m)
    slots: list[Slot] = []
    log: list[dict] = []
    pos: Optional[str] = None
    b: Optional[bool] = None
    for ev in prep.observable:
        orig = getattr(ev, "orig", ev.index)
        t1, t2 = candidates(m, ctx, ev, pos, b, term, allow_tier2=True)
        rec: dict = {"i": orig, "event": ev.describe(), "tier1": t1, "tier2": t2}
        if ev.kind == "end":
            if not t1:
                return None, f"机器里没有结束状态 {term}", log
            slots.append(Slot(index=ev.index, state=term, how="end", edge="keep" if pos is not None else "start"))
            rec.update({"decision": "end", "state": term})
            log.append(rec)
            break
        d = steps.get(orig)
        if d is None:
            if not accept_proposals:
                return None, f"第 {orig} 步没有判定", log
            p = propose(m, ev, t1, t2, term)
            d = Decision(d=p["d"], state=p.get("state", ""), source="proposal")
        if d.d not in ("match", "new"):
            return None, f"第 {orig} 步的判定 {d.d!r} 不认识", log
        tier = 0
        if d.d == "match":
            if d.state in t1:
                tier = 1
            elif d.state in t2 and allow_tier2:
                tier = 2
            elif d.state in t2:
                d = Decision(d="new", purpose=d.purpose, clause=d.clause, source=d.source + "→new(严格轮)")
            else:
                return None, f"第 {orig} 步判为 {d.state!r}，它不在候选里（T1 {t1} / T2 {t2}）", log
        if d.d == "match":
            st = m.states[d.state]
            how = "label" if set(getattr(st.action, "labels", []) or []) != set(ev.labels) else "match"
            has_loop = d.state in zero_paths(m, d.state, status_known(ctx, st, True))
            edge = ("keep" if d.state in ea else "start") if pos is None else ("keep" if tier == 1 else "add")
            slots.append(Slot(index=ev.index, state=d.state, how=how, c_loop=loop_cost(ev, has_loop), edge=edge))
            pos = d.state
        else:
            if d.purpose:
                ev.intent = (d.purpose + (" — " + ev.intent if ev.intent else ""))[:600]
            slots.append(Slot(index=ev.index, state=f"{NEW}{ev.index}", how="new", c_loop=loop_cost(ev, False),
                              edge="start" if pos is None else "add"))
            pos = f"{NEW}{ev.index}"
        b = ev.ok if ev.kind == "tool" else None
        rec.update({"decision": d.d, "state": d.state, "tier": tier, "source": d.source,
                    "purpose": d.purpose, "clause": d.clause, "slot": slots[-1].how + "/" + slots[-1].edge
                    + ("+loop" if slots[-1].loop else "")})
        log.append(rec)
    if not slots or slots[-1].how != "end":
        return None, "路径没有以结束状态收尾", log
    cost = sum(3 if s.edge in ("add", "start") else 0 for s in slots) + sum(4 for s in slots if s.is_new) \
        + sum(1 for s in slots if s.how == "label") + sum(s.c_loop for s in slots)
    return Alignment(slots=slots, cost=cost, end_state=term), "", log


def _stamp_clauses(build_machine: Machine, alignment: Alignment, anchors_out: list[str], log: list,
                   ctx: CompileContext) -> None:
    """新状态（及其生成门）写上 agent 给的条款号；log 与 slots 一一对应（每个可观察事件一条）。"""
    ids = {c[0] for c in ctx.clauses}
    for slot, sid, rec in zip(alignment.slots, anchors_out, log):
        rec["state_id"] = sid
        if not slot.is_new or sid not in build_machine.states:
            continue
        clause = str(rec.get("clause") or "")
        if clause and clause in ids:
            build_machine.states[sid].clause = clause
            for gs in build_machine.states.values():
                if gs.action.kind == "model" and not getattr(gs.action, "observable", False) \
                        and [t.to for t in gs.transitions] == [sid]:
                    gs.clause = clause


@dataclass
class Accepted:
    trace: str
    source: str
    anchors: list
    ignore: list
    prep: Optional[Prepared] = None
    ignore_calls: list = field(default_factory=list)


def update_with_decisions(machine: Machine, trace: Trace, ctx: CompileContext, *, source: str, spec: Any,
                          accepted: list[Accepted], attempts: int = 2) -> tuple[Machine, dict, Optional[Accepted]]:
    """处理一条轨迹。返回 (新机器或原机器, 记录, 接受项或 None)。

    判定者可以在轨迹层面给 ``{"exclude": "<原因>"}``：这条轨迹的步无法按技能工作流归类
    （例如真正的修改藏在 write 工具写出的脚本文件里、由一条没有写信号的命令执行），不参与更新。"""
    if isinstance(spec, dict) and spec.get("exclude"):
        prep0 = prepare_steps(trace, ctx, source=source)
        return machine, {"trace": prep0.trace_id, "verdict": prep0.verdict, "tau": prep0.tau,
                         "events": [e.describe() for e in prep0.observable], "status": "excluded",
                         "why": "判定者排除：" + str(spec["exclude"])}, None
    ignore, steps, accept_props, ignore_calls = parse_decisions(spec)
    prep = prepare_steps(trace, ctx, source=source, ignore=ignore, ignore_calls=ignore_calls)
    entry: dict = {"trace": prep.trace_id, "verdict": prep.verdict, "tau": prep.tau, "ignored": sorted(ignore),
                   "ignored_calls": sorted(ignore_calls),
                   "events": [e.describe() for e in prep.observable], "attempts": [], "fingerprint_before": fingerprint(machine)}
    pre = pre_status(prep, trace, machine)
    if pre is not None:
        entry["status"], entry["why"] = pre
        return machine, entry, None
    modes = [True, False][:max(1, attempts)]
    for k, allow in enumerate(modes, 1):
        att: dict = {"attempt": k, "allow_tier2": allow}
        prep_k = prepare_steps(trace, ctx, source=source, ignore=ignore, ignore_calls=ignore_calls)   # intent 会被 new 判定改写，每轮重来
        al, why, log = align_from_decisions(machine, ctx, prep_k, steps, accept_props, allow_tier2=allow)
        att["decisions"] = log
        if al is None:
            att["result"], att["why"] = "no_path", why
            entry["attempts"].append(att)
            break
        att["cost"] = al.cost
        att["path"] = [f"{s.state}:{s.how}" + ("+loop" if s.loop else "") + f"/{s.edge}" for s in al.slots]
        bd = build_candidate(machine, prep_k, al, ctx)
        _stamp_clauses(bd.machine, al, bd.anchors, log, ctx)
        att["changes"] = list(bd.changes)
        errs = _check.check(bd.machine, ctx)
        if errs:
            att["result"], att["why"] = "check_failed", errs[:8]
            entry["attempts"].append(att)
            continue
        rp = _check.replay(bd.machine, prep_k, bd.anchors)
        if not rp.ok:
            att["result"], att["why"] = "replay_failed", rp.why
            att["replay_path"] = rp.path
            entry["attempts"].append(att)
            continue
        broken = None
        for acc in accepted:
            if acc.prep is None:
                continue
            r2 = _check.replay(bd.machine, acc.prep, acc.anchors)
            if not r2.ok:
                broken = (acc.trace, r2.why)
                break
        if broken is not None:
            att["result"], att["why"] = "protected_failed", f"{broken[0]}：{broken[1]}"
            entry["attempts"].append(att)
            continue
        att["result"] = "accepted"
        entry["attempts"].append(att)
        entry.update({"status": "accepted", "cost": al.cost, "anchors": list(bd.anchors), "changes": list(bd.changes),
                      "path": rp.path, "inserted": [c for c in bd.changes if c.startswith("新增状态") or c.startswith("新增可观察")],
                      "fingerprint_after": fingerprint(bd.machine)})
        return bd.machine, entry, Accepted(trace=prep.trace_id, source=source, anchors=list(bd.anchors),
                                           ignore=sorted(ignore), prep=prep_k, ignore_calls=sorted(ignore_calls))
    entry["status"] = "rejected"
    last = entry["attempts"][-1] if entry["attempts"] else {}
    entry["why"] = f"{last.get('result', '')}: {last.get('why', '')}"[:600]
    return machine, entry, None


# --------------------------------------------------------------------------- #
# 进度
# --------------------------------------------------------------------------- #
@dataclass
class Progress:
    machine: Machine
    entries: list = field(default_factory=list)
    accepted: list = field(default_factory=list)       # list[Accepted]
    agent_log: list = field(default_factory=list)      # 每条轨迹每一步的判定

    def done(self) -> dict:
        return {e["trace"]: e["status"] for e in self.entries}

    def counts(self) -> dict:
        out: dict = {}
        for e in self.entries:
            out[e["status"]] = out.get(e["status"], 0) + 1
        return out

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "machine.json").write_text(json.dumps(json.loads(self.machine.model_dump_json(by_alias=True)),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "progress.json").write_text(json.dumps({
            "fingerprint": fingerprint(self.machine), "counts": self.counts(),
            "accepted": [{"trace": a.trace, "source": a.source, "anchors": a.anchors, "ignore": a.ignore,
                          "ignore_calls": a.ignore_calls} for a in self.accepted],
            "entries": self.entries, "agent_log": self.agent_log}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out: Path, m0: Machine, ctx: CompileContext, traces: dict[str, Trace]) -> "Progress":
        p = out / "progress.json"
        if not p.is_file():
            return cls(machine=m0.model_copy(deep=True))
        data = json.loads(p.read_text(encoding="utf-8"))
        from ..schema import load_machine
        m = load_machine(out / "machine.json")
        pr = cls(machine=m, entries=list(data.get("entries") or []), agent_log=list(data.get("agent_log") or []))
        for a in data.get("accepted") or []:
            tr = traces.get(a["trace"])
            prep = (prepare_steps(tr, ctx, source=a["source"], ignore=a.get("ignore") or [],
                                  ignore_calls=a.get("ignore_calls") or []) if tr is not None else None)
            pr.accepted.append(Accepted(trace=a["trace"], source=a["source"], anchors=list(a["anchors"]),
                                        ignore=list(a.get("ignore") or []), prep=prep,
                                        ignore_calls=list(a.get("ignore_calls") or [])))
        return pr


__all__ = ["Accepted", "Decision", "Progress", "align_from_decisions", "candidates", "fingerprint",
           "parse_decisions", "pre_status", "prepare_steps", "propose", "strip_tools", "update_with_decisions"]
