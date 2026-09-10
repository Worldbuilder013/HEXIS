"""逐条轨迹更新（算法文档第 7 节）。

Accept(M', T, P_k) ⟺ Check(M') ∧ ⋀_{T'∈P_k∪{T}} Replay(M', T', π̃_{T'})。
全部通过一次性提交；任一失败丢弃副本，机器与已接受集合不变。每条轨迹默认最多两次尝试：
第一次允许更换已有状态的工具，第二次保留原工具。

一条轨迹能不能参与更新，由**当前技能的规则**决定，不由固定的阶段规则决定：
含无法识别事件的轨迹不参与（unsupported）；违反技能要求或外部禁止规则的不参与（excluded）；
声明的终点与证据不符的不参与（violation）；没有可观察事件的跳过。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..schema import Machine, Trace
from . import check as _check
from .align import align
from .context import CompileContext
from .modify import build_candidate
from .traces import Prepared, prepare, violates_prohibitions


@dataclass
class Accepted:
    prep: Prepared
    anchors: list


@dataclass
class UpdateResult:
    machine: Machine
    entries: list = field(default_factory=list)
    accepted: list = field(default_factory=list)

    def counts(self) -> dict:
        out: dict = {}
        for e in self.entries:
            out[e["status"]] = out.get(e["status"], 0) + 1
        return out


def _events_brief(prep: Prepared) -> list[str]:
    return [e.describe() for e in prep.events if e.kind != "model" or e.role == "output"]


def update_one(machine: Machine, prep: Prepared, accepted: list, ctx: CompileContext, *,
               attempts: int = 2) -> tuple[Machine, dict, Optional[Accepted]]:
    """处理一条轨迹。返回 (新机器或原机器, 记录, 接受项或 None)。"""
    entry: dict = {"trace": prep.trace_id, "verdict": prep.verdict, "events": _events_brief(prep),
                   "tau": prep.tau, "attempts": []}
    for k in range(1, max(1, attempts) + 1):
        allow = (k == 1)
        att: dict = {"attempt": k, "allow_realize": allow}
        al, why = align(machine, prep, ctx, allow_realize=allow)
        if al is None:
            att["result"], att["why"] = "no_path", why
            entry["attempts"].append(att)
            if k == 1:
                entry["status"], entry["why"] = "rejected", f"无对齐路径：{why}"
                return machine, entry, None
            continue
        att["cost"] = al.cost
        att["path"] = [f"{s.state}:{s.how}" + ("+loop" if s.loop else "") + f"/{s.edge}" for s in al.slots]
        bd = build_candidate(machine, prep, al, ctx)
        att["changes"] = list(bd.changes)
        errs = _check.check(bd.machine, ctx)
        if errs:
            att["result"], att["why"] = "check_failed", errs[:8]
            entry["attempts"].append(att)
            continue
        rp = _check.replay(bd.machine, prep, bd.anchors)
        if not rp.ok:
            att["result"], att["why"] = "replay_failed", rp.why
            att["replay_path"] = rp.path
            entry["attempts"].append(att)
            continue
        broken = None
        for acc in accepted:
            r2 = _check.replay(bd.machine, acc.prep, acc.anchors)
            if not r2.ok:
                broken = (acc.prep.trace_id, r2.why)
                break
        if broken is not None:
            att["result"], att["why"] = "protected_failed", f"{broken[0]}：{broken[1]}"
            entry["attempts"].append(att)
            continue
        att["result"] = "accepted"
        entry["attempts"].append(att)
        entry.update({"status": "accepted", "cost": al.cost, "anchors": list(bd.anchors),
                      "changes": list(bd.changes), "path": rp.path})
        return bd.machine, entry, Accepted(prep=prep, anchors=list(bd.anchors))
    entry["status"] = "rejected"
    last = entry["attempts"][-1] if entry["attempts"] else {}
    entry["why"] = f"{last.get('result', '')}: {last.get('why', '')}"[:600]
    return machine, entry, None


def update(machine: Machine, traces: Sequence[tuple[Any, Optional[Trace], str]], ctx: CompileContext, *,
           attempts: int = 2, accepted_only: bool = False, progress=None) -> UpdateResult:
    """处理全部轨迹。``traces`` 是 load_traces 的输出：(路径, Trace 或 None, 读取错误)。"""
    res = UpdateResult(machine=machine)
    for path, trace, err in traces:
        name = Path(str(path)).stem
        if trace is None:
            res.entries.append({"trace": name, "status": "unreadable", "why": err})
        elif accepted_only and trace.verdict != "accepted":
            res.entries.append({"trace": name, "status": "skipped", "why": f"判分 {trace.verdict}，只保留判对轨迹"})
        else:
            prep = prepare(trace, ctx, source=str(path))
            base = {"trace": prep.trace_id, "verdict": prep.verdict, "events": _events_brief(prep),
                    "tau": prep.tau, "notes": list(prep.notes)}
            banned = violates_prohibitions(trace, res.machine)
            if prep.unsupported:
                res.entries.append({**base, "status": "unsupported",
                                    "why": "无法识别的事件：" + ", ".join(f"第 {s} 步 kind={k}" for s, k in prep.unsupported)})
            elif banned:
                res.entries.append({**base, "status": "excluded", "why": banned})
            elif prep.requirement_violations:
                res.entries.append({**base, "status": "excluded", "why": "；".join(prep.requirement_violations)})
            elif prep.violation:
                res.entries.append({**base, "status": "violation", "why": prep.violation})
            elif not prep.observable:
                res.entries.append({**base, "status": "skipped", "why": "没有可观察事件"})
            else:
                m2, entry, acc = update_one(res.machine, prep, res.accepted, ctx, attempts=attempts)
                entry["notes"] = list(prep.notes)
                res.entries.append(entry)
                if acc is not None:
                    res.machine = m2
                    res.accepted.append(acc)
        if progress:
            e = res.entries[-1]
            progress(f"{e['trace']:<12} {e['status']:<11} {str(e.get('why') or '')[:110]}")
    return res


__all__ = ["Accepted", "UpdateResult", "update", "update_one"]
