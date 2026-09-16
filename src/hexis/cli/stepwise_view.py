"""逐步增量编译的展示层：给判定者（agent）看的步摘要、候选描述与提案渲染。

放在 ``hexis/compiler/`` 之外：这里为了让人读得快，会识别命令里的工作簿读写、重算、目录列举等特征，
带有工具名与领域词；编译器核心（``hexis/compiler/``）不得含这些词（test_46 钉住这一点）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from hexis.compiler.align import NEW
from hexis.compiler.stepwise import candidates, fingerprint, propose
from hexis.compiler.traces import Event, Prepared, end_state_for
from hexis.compiler.context import CompileContext
from hexis.machine.schema import Machine


def _short(s: Any, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[:n] + "…"


def summarize_command(cmd: str) -> str:
    """一条 shell 命令的特征：读了什么簿（公式视图/缓存视图）、存到哪、是否对比、是否设重算。"""
    c = cmd or ""
    flags: list[str] = []
    for arg in re.findall(r"load_workbook\(([^)]*)\)", c):
        tgt = "output" if "output" in arg else ("input" if "input" in arg else _short(arg, 24))
        mode = "cached" if re.search(r"data_only\s*=\s*True", arg) else "formula"
        flags.append(f"load:{tgt}/{mode}")
    for arg in re.findall(r"\.save\(([^)]*)\)", c):
        flags.append("save:" + ("output" if "output" in arg else ("INPUT!" if "input" in arg else _short(arg, 24))))
    if re.search(r"(^|[;&|\s])(cp|mv)\s", c) or "shutil.copy" in c:
        flags.append("copy")
    if "fullCalcOnLoad" in c or "calcMode" in c or "calculation" in c:
        flags.append("recalc-on-open")
    if any(f.startswith("load:input") for f in flags) and any(f.startswith("load:output") for f in flags):
        flags.append("compare-in-out")
    if re.search(r"(^|\s)(ls|pwd|find|cat|head|tail|file|wc|unzip)\b", c) and "python" not in c:
        flags.append("shell-probe")
    if re.search(r"pip\s|--version|import sys|which\s|python3 -c \"import openpyxl\"$", c):
        flags.append("env-check")
    if "python" in c and "openpyxl" not in c and "load_workbook" not in c:
        flags.append("python-no-openpyxl")
    body = re.sub(r"python3?\s*(-\s*)?<<\s*'?EOF'?", "", c)
    return (" ".join(flags) + " | " if flags else "") + _short(body, 180)


def describe_event(ev: Event) -> dict:
    d: dict = {"kind": ev.kind, "index": getattr(ev, "orig", ev.index)}
    if ev.kind == "tool":
        d.update({"tool": ev.tool, "label": ev.label, "labels": sorted(ev.labels), "n_calls": ev.n_calls, "ok": ev.ok,
                  "intent": _short(ev.intent, 160), "calls": []})
        for k, c in enumerate(ev.calls):
            inp = c.input or {}
            brief = 0 < k < len(ev.calls) - 1 and len(ev.calls) > 3
            if ev.tool == "bash" or "command" in inp:
                what = summarize_command(str(inp.get("command", "")))
                if brief:
                    what = _short(what, 110)
            else:
                what = _short(json.dumps(inp, ensure_ascii=False), 160)
            out = c.output or {}
            d["calls"].append({"step": c.step, "ok": c.ok, "rc": out.get("returncode"), "what": what,
                               "stdout": "" if brief else _short(out.get("stdout", "") or out.get("error", ""), 70)})
    elif ev.kind == "model":
        d.update({"role": ev.role, "text": _short(ev.text, 260), "output_keys": sorted(ev.output)})
    elif ev.kind == "end":
        d.update({"terminal": ev.terminal})
    else:
        d.update({"text": _short(ev.text, 200)})
    return d


def describe_state(m: Machine, sid: str) -> str:
    st = m.states[sid]
    a = st.action
    if a.kind == "tool":
        gate = next((g for g, gs in m.states.items() if gs.action.kind == "model"
                     and not getattr(gs.action, "observable", False) and [t.to for t in gs.transitions] == [sid]), None)
        purpose = _short(m.states[gate].action.prompt, 110) if gate else ""
        labs = "+".join(getattr(a, "labels", []) or [])
        return (f"{sid}: {a.name}/{a.phase or '-'}{'[' + labs + ']' if labs else ''} "
                f"{_short(json.dumps(a.input, ensure_ascii=False), 60)}"
                + (f" ← {gate}: {purpose}" if gate else "") + (f" @{st.clause}" if st.clause else ""))
    if a.kind == "model":
        return f"{sid}: model{'*' if getattr(a, 'observable', False) else ''} → {a.writes} {_short(a.prompt, 110)!r}" + (f" @{st.clause}" if st.clause else "")
    if a.kind == "end":
        return f"{sid}: end {a.terminal}"
    return f"{sid}: {a.kind}"


def show_trace(m: Machine, ctx: CompileContext, prep: Prepared, *, allow_tier2: bool = True) -> dict:
    """沿提案链算每一步的候选与提案。判定者改了某一步，后面几步的候选会随之变化——apply 时按实际判定重算。"""
    term = end_state_for(m, prep.tau)
    out: dict = {"trace": prep.trace_id, "verdict": prep.verdict, "tau": prep.tau, "term": term,
                 "request": _short(prep.task_input.get("request", ""), 220), "fingerprint": fingerprint(m), "steps": []}
    pos: Optional[str] = None
    b: Optional[bool] = None
    for ev in prep.observable:
        t1, t2 = candidates(m, ctx, ev, pos, b, term, allow_tier2=allow_tier2)
        prop = propose(m, ev, t1, t2, term)
        out["steps"].append({"i": getattr(ev, "orig", ev.index), "event": describe_event(ev),
                             "tier1": t1, "tier2": t2, "propose": prop})
        pos = prop.get("state") if prop["d"] == "match" else f"{NEW}{ev.index}"
        b = ev.ok if ev.kind == "tool" else None
    return out


def render_show(m: Machine, shown: dict) -> str:
    lines = [f"### {shown['trace']}  verdict={shown['verdict']}  tau={shown['tau']}→{shown['term']}  "
             f"steps={len(shown['steps'])}  fp={shown['fingerprint']}",
             f"    request: {shown['request']}"]
    for s in shown["steps"]:
        e = s["event"]
        if e["kind"] == "tool":
            head = (f" [{s['i']:>2}] {e['tool']}/{e['label'] or '-'}{'[' + '+'.join(e['labels']) + ']' if e['labels'] else ''}"
                    f" ×{e['n_calls']} {'✓' if e['ok'] else '✗'}")
            if e["intent"]:
                head += f"   intent: {e['intent']}"
            lines.append(head)
            for c in e["calls"]:
                if "skipped" in c:
                    lines.append(f"        … {c['skipped']} more calls …")
                else:
                    lines.append(f"        - #{c['step']} rc={c['rc']} {c['what']}")
                    if c["stdout"]:
                        lines.append(f"            → {c['stdout']}")
        elif e["kind"] == "model":
            lines.append(f" [{s['i']:>2}] model:{e['role']} {e['text']!r}")
        elif e["kind"] == "end":
            lines.append(f" [{s['i']:>2}] end  (声明 {e['terminal'] or '-'}, τ*={shown['tau']})")
        else:
            lines.append(f" [{s['i']:>2}] {e['kind']} {e.get('text', '')!r}")
        if e["kind"] != "end":
            t1 = ", ".join(describe_state(m, x) for x in s["tier1"]) or "—"
            t2 = ", ".join(x for x in s["tier2"]) or "—"
            p = s["propose"]
            lines.append(f"        cand T1: {t1}" + (f"   | T2: {t2}" if s["tier2"] else "")
                         + f"    ⇒ 提案 {p['d']}{' ' + p['state'] if p.get('state') else ''}")
    return "\n".join(lines)


__all__ = ["describe_event", "describe_state", "render_show", "show_trace", "summarize_command"]
