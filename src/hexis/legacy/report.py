"""Coverage report, context probe, and two renderings of the three-arm experiment report (plain text / HTML).

Compilation delivers not just a machine but also an account of "why it goes this way and what it has not learned":
which clauses were triggered by traces and which were not (those all go to FALLBACK), the error rate of every judge
action, the accumulated error bound along a path, and the share of runs that fall into FALLBACK over a batch of tasks.
This account makes "how much the machine preserves and how much is still missing" reviewable, instead of hiding it
behind a single success-rate number.

:func:`context_probe` compares two contexts for a judge action, only the variables it declares it reads (narrow read,
which is what the compiled artifact uses) versus somewhat more (wide read), to see whether the narrow read degrades
output quality. A correct path whose content quietly gets worse is the most insidious failure of this kind of
compilation; the probe puts it in plain view.

:func:`render_experiment` and :func:`render_html` render the six numbers of a three-arm experiment report. **The HTML
page is fully self-contained**: no external styles, no CDN, no JS libraries; the charts are inline SVG that this module
assembles from plain strings. The reason is practical: the report has to fit into an attachment, open on a machine
without network access, and still open ten years from now; any external link would one day turn it into a blank page.
The charts paint their own opaque background and pick foreground colors against it, so they stay readable when
embedded in light or dark pages, without depending on the host's default background.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from hexis.execution import runtime
from hexis.legacy.replay import excludes, reproduces
from hexis.machine.schema import FALLBACK, Machine

__all__ = [
    "context_probe", "cover_report", "render", "render_experiment", "render_html",
]


# --------------------------------------------------------------------------- #
# Coverage account
# --------------------------------------------------------------------------- #
def _min_support(machine: Machine, state: Any) -> int:
    """The **weakest** piece of evidence on a state: a judge action's own support, a branch's out-edge supports."""
    vals = [t.support for t in (state.transitions or [])]
    if state.action.kind == "judge":
        vals.append(state.action.support)
    return min(vals) if vals else 0


def cover_report(machine: Machine, clauses: list, *, t_plus=(), t_minus=(),
                 runs=()) -> dict:
    """Coverage account of a machine. ``clauses`` is the list of document clauses ``[(id, text)]``; ``runs`` is a
    number of :class:`~hexis.execution.runtime.RunResult` used to compute the FALLBACK rate.

    ``clauses_thin`` are the clauses with **thin evidence**: on their states the weakest support has not reached
    ``thresholds.min_support``. Support 0 counts as thin too: it means the edge has no recorded trace evidence at all
    (a hand-written reference machine is therefore thin throughout; that is the truth, not a bug).
    """
    covered = {s.clause for s in machine.states.values() if s.clause}
    doc_clauses = {cid for cid, _ in clauses if cid.startswith("S")}
    judge_states = [s for s in machine.states.values() if s.action.kind == "judge"]
    thr = machine.thresholds.min_support
    thin = {s.clause for s in machine.states.values()
            if s.clause and s.action.kind != "end" and _min_support(machine, s) < thr}

    fallback_runs = sum(1 for r in runs
                        if any(rec.state == FALLBACK for rec in r.trace.records))
    return {
        "n_states": machine.n_states(),
        "clauses_total": len(doc_clauses),
        "clauses_covered": sorted(covered & doc_clauses),
        "clauses_untriggered": sorted(doc_clauses - covered),
        "clauses_thin": sorted(thin),
        "min_support": thr,
        "judge_error_sum": round(sum(s.action.error_rate for s in judge_states), 4),
        "judge_states": {s.id: {"error_rate": s.action.error_rate,
                                "support": s.action.support} for s in judge_states},
        "t_plus_reproduced": sum(1 for t in t_plus if reproduces(machine, t)),
        "t_plus_total": len(t_plus),
        "t_minus_excluded": sum(1 for t in t_minus if excludes(machine, t)),
        "t_minus_total": len(t_minus),
        "fallback_rate": round(fallback_runs / len(runs), 4) if runs else None,
    }


def context_probe(machine: Machine, tasks: list, *, model, tools, doc: str,
                  wide_extra: Optional[dict] = None) -> dict:
    """For each judge action, compare whether the narrow-read and wide-read outputs agree.

    Collect input snapshots of judge actions from a real task stream, run the judgment once with "declared variables
    only" and once with "extra ``wide_extra`` context", and report the agreement rate between the two. A low agreement
    rate means the narrow read loses information: the path is right but the judgment quality is declining. A toy stub
    judges both contexts the same, so its agreement rate is 1; that completes the probe formally, and its real value
    shows on a real model.
    """
    wide_extra = wide_extra or {}
    snaps_by_state: dict[str, list[dict]] = {}
    for task in tasks:
        fs_tools = tools(task) if callable(tools) else tools
        res = runtime.run_task(machine, task, model=model, tools=fs_tools, doc=doc)
        for rec in res.trace.records:
            if rec.action.get("kind") == "judge":
                snaps_by_state.setdefault(rec.state, []).append(dict(rec.vars))

    report: dict = {}
    for sid, st in machine.states.items():
        if st.action.kind != "judge":
            continue
        snaps = snaps_by_state.get(sid, [])
        agree = total = 0
        dist: dict[str, int] = {}
        for snap in snaps:
            narrow_vals = {k: snap.get(k) for k in st.action.reads}
            wide_vals = {**narrow_vals, **{k: snap.get(k) for k in wide_extra}}
            narrow = model.classify(prompt=st.action.prompt, values=narrow_vals,
                                    labels=st.action.labels,
                                    examples=tuple(e.model_dump() for e in st.action.examples))
            wide = model.classify(prompt=st.action.prompt, values=wide_vals,
                                  labels=st.action.labels,
                                  examples=tuple(e.model_dump() for e in st.action.examples))
            total += 1
            agree += int(narrow == wide)
            dist[narrow] = dist.get(narrow, 0) + 1
        report[sid] = {
            "samples": total,
            "narrow_vs_wide_agree": round(agree / total, 4) if total else None,
            "label_dist": dist,
            "error_rate": st.action.error_rate,
        }
    return report


def render(report: dict) -> str:
    """Render the coverage account as human-readable text."""
    lines = ["Skill compilation coverage report", "=" * 32,
             f"States (excluding end states): {report.get('n_states')}",
             f"Clauses covered: {report.get('clauses_covered')}",
             f"Untriggered clauses: {report.get('clauses_untriggered') or '(none)'}",
             f"Sum of judge action error rates (a component of the path error bound): {report.get('judge_error_sum')}"]
    for sid, d in (report.get("judge_states") or {}).items():
        lines.append(f"  judge {sid}: error rate {d['error_rate']}, support {d['support']}")
    lines.append(f"T+ replayed: {report.get('t_plus_reproduced')}/{report.get('t_plus_total')}")
    lines.append(f"T- excluded: {report.get('t_minus_excluded')}/{report.get('t_minus_total')}")
    if report.get("fallback_rate") is not None:
        lines.append(f"FALLBACK rate: {report.get('fallback_rate')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Three-arm experiment report: shared helpers
# --------------------------------------------------------------------------- #
#: Titles of the six numbers; the order is their numbering in the deliverable.
_METRIC_TITLES = (
    ("score", "1. Score"),
    ("path_consistency", "2. Path consistency"),
    ("compliance", "3. Process compliance"),
    ("cost", "4. Cost"),
    ("fallback", "5. Fallback rate"),
    ("judge_error", "6. Judge action error rate"),
)

_ARM_LABEL = {"bare": "Arm 1: bare (no skill installed)",
              "skill": "Arm 2: skill (skill installed, interpreted)",
              "machine": "Arm 3: machine (compiled machine + fallback segment)"}

#: The opaque background the charts paint themselves, and the foreground colors chosen against it. **The host's
#: default background is not used**: these charts get embedded in light pages as well as dark pages, and only fixed
#: colors stay readable in both.
_PANEL = "#232a36"
_INK = "#e8ecf3"
_MUTED = "#9aa4b8"
_GRID = "#39404e"
#: Foreground palette: blue (main segment), light blue (second series), amber (fallback segment), green (consistency).
_C_MAIN = "#5b8ff9"
_C_ALT = "#a7c7ff"
_C_FALLBACK = "#f6bd16"
_C_OK = "#5ad8a6"


def _as_dict(report: Any) -> dict:
    """Accept an experiment report object (anything with ``to_dict``) or its dict, and normalize both to a dict."""
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return dict(report or {})


def _arms_of(rep: Mapping) -> list:
    """The arms that appear in the report, in the fixed order bare / skill / machine (any others after them)."""
    seen: list = []
    for key in ("score", "cost", "compliance", "path_consistency"):
        seen += list(((rep.get(key) or {}).get("per_arm") or {}).keys())
    order = ["bare", "skill", "machine"]
    known = [a for a in order if a in seen]
    return known + sorted({a for a in seen if a not in order})


def _num(v: Any, dash: str = "—") -> str:
    """Number to string. ``None`` is always shown as a dash: "not measured" must not look like 0."""
    if v is None:
        return dash
    if isinstance(v, float):
        return ("%.4f" % v).rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


def _pct(v: Any) -> str:
    return "—" if v is None else "%.1f%%" % (100.0 * float(v))


def _esc(s: Any) -> str:
    """XML/HTML text escaping. SVG fragments must parse as XML, so not one of the five characters may be missed."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #
def render_experiment(report: Any) -> str:
    """Render the three-arm experiment report as human-readable text. Readable directly in a terminal, no browser needed."""
    rep = _as_dict(report)
    arms = _arms_of(rep)
    prov = rep.get("provenance") or {}
    counts = rep.get("counts") or {}
    out: list = ["Three-arm experiment report (hexis)", "=" * 60]

    n_prob = prov.get("n_problems")
    runs = ((prov.get("config") or {}).get("runs"))
    out.append("Sample size n=%s test problems × %s runs × %d arms: supports directional conclusions only, "
               "no claims of significance" % (_num(n_prob), _num(runs), len(arms)))
    out.append("Model %s | machine %s | data sha %s"
               % (prov.get("model_id") or "—",
                  (prov.get("machine") or {}).get("skill_id") or "(none)",
                  str(prov.get("data_sha256") or "")[:12]))
    out.append("Skill %s @ %s"
               % ((prov.get("skill") or {}).get("slug") or "—",
                  ((prov.get("skill") or {}).get("commit") or "—")[:12]))
    out.append("Wall clock %s s | planned %s runs (reused %s, executed now %s, failed %s) | report %s"
               % (_num(prov.get("wall_s")), _num(counts.get("planned")),
                  _num(counts.get("reused")), _num(counts.get("executed")),
                  _num(counts.get("failed")),
                  "complete" if rep.get("complete") else "**incomplete**"))

    # ---- 1. score ---- #
    score = rep.get("score") or {}
    out += ["", "1. Score (raw counts)", "-" * 60]
    out.append("%-12s %6s %8s %8s %8s %10s" % ("arm", "runs", "strict", "lenient",
                                               "no_boxed", "strict acc"))
    for a in arms:
        d = (score.get("per_arm") or {}).get(a) or {}
        out.append("%-12s %6s %8s %8s %8s %10s"
                   % (a, _num(d.get("runs")), _num(d.get("correct")),
                      _num(d.get("lenient_correct")), _num(d.get("no_boxed")),
                      _pct(d.get("accuracy"))))
    for a in arms:
        d = (score.get("per_arm") or {}).get(a) or {}
        out.append("  route distribution of %s: %s" % (a, d.get("routes") or {}))
    out.append("  " + str(score.get("note") or ""))
    out.append("  per-problem table:")
    out.append("  %-20s %3s %s" % ("problem", "lvl", " ".join("%-9s" % a for a in arms)))
    for tid, row in (score.get("per_problem") or {}).items():
        cells = []
        for a in arms:
            c = (row.get("arms") or {}).get(a) or {}
            cells.append("%-9s" % ("%s/%s" % (_num(c.get("correct", 0)),
                                              _num(c.get("runs", 0)))))
        out.append("  %-20s %3s %s" % (tid, _num(row.get("level")), " ".join(cells)))

    # ---- 2. path consistency ---- #
    pc = rep.get("path_consistency") or {}
    out += ["", "2. Path consistency (share of same-problem run pairs with identical action sequences)", "-" * 60]
    out.append("%-12s %8s %10s %10s %10s" % ("arm", "problems", "run pairs", "identical", "rate"))
    for a in arms:
        d = (pc.get("per_arm") or {}).get(a) or {}
        out.append("%-12s %8s %10s %10s %10s"
                   % (a, _num(d.get("problems")), _num(d.get("pairs_total")),
                      _num(d.get("pairs_identical")), _pct(d.get("consistency"))))
    out.append("  " + str(pc.get("note") or ""))

    # ---- 3. process compliance ---- #
    comp = rep.get("compliance") or {}
    out += ["", "3. Process compliance", "-" * 60]
    out.append("%-12s %8s %10s %10s %12s %10s"
               % ("arm", "verified", "verif rate", "exceeded", "unverified", "labelled"))
    for a in arms:
        d = (comp.get("per_arm") or {}).get(a) or {}
        out.append("%-12s %8s %10s %10s %12s %10s"
                   % (a, _num(d.get("ran_verification")),
                      _pct(d.get("verification_rate")),
                      _num(d.get("repair_exceeded")),
                      _num(d.get("unverified_submissions")),
                      "%s/%s" % (_num(d.get("unverified_labelled")),
                                 _num(d.get("unverified_submissions")))))
    for a in arms:
        d = (comp.get("per_arm") or {}).get(a) or {}
        out.append("  %s: P1 violations %s (claimed verified without verifying: %s), terminal kinds %s"
                   % (a, _num(d.get("p1_violations")),
                      _num(d.get("mislabelled_verified")),
                      d.get("terminal_kinds") or {}))
    out.append("  " + str(comp.get("note") or ""))

    # ---- 4. cost ---- #
    cost = rep.get("cost") or {}
    out += ["", "4. Cost (one run = one problem attempted once)", "-" * 60]
    out.append("%-12s %8s %12s %12s %12s %10s"
               % ("arm", "calls", "prompt", "completion", "fb prompt", "unmeasured"))
    for a in arms:
        d = (cost.get("per_arm") or {}).get(a) or {}
        out.append("%-12s %8s %12s %12s %12s %10s"
                   % (a, _num(d.get("llm_calls")), _num(d.get("prompt_tokens")),
                      _num(d.get("completion_tokens")),
                      _num(d.get("fallback_prompt_tokens")),
                      _num(d.get("runs_unmeasured"))))
    out.append("  " + str(cost.get("note") or ""))

    # ---- 5. fallback rate ---- #
    fb = rep.get("fallback") or {}
    out += ["", "5. Fallback rate (arm 3 only)", "-" * 60]
    out.append("Entered the fallback segment in %s / %s runs = %s; the machine finished on its own %s times"
               % (_num(fb.get("entered")), _num(fb.get("runs")),
                  _pct(fb.get("rate")), _num(fb.get("finished_in_machine"))))
    out.append("States fallen back from: %s" % (fb.get("entry_states") or {}))
    out.append("Fallback reasons: %s" % (fb.get("entry_reasons") or {}))
    out.append("  " + str(fb.get("note") or ""))

    # ---- 6. judge action error rate ---- #
    je = rep.get("judge_error") or {}
    ct = je.get("compile_time") or {}
    tt = je.get("test_time") or {}
    oa = tt.get("outcome_anchored") or {}
    out += ["", "6. Judge action error rate (compile-time calibration vs test-time observation)", "-" * 60]
    out.append("Compile time: sum of error rates %s (threshold %s)"
               % (_num(ct.get("error_sum")), _num(ct.get("threshold"))))
    for sid, d in (ct.get("judge_states") or {}).items():
        out.append("  %s calibrated error rate %s, support %s"
                   % (sid, _num(d.get("error_rate")), _num(d.get("support"))))
    out.append("Test time: %s judge calls, %s abstentions (%s)"
               % (_num(tt.get("judge_calls")), _num(tt.get("abstained")),
                  _pct(tt.get("abstain_rate"))))
    for sid, d in (tt.get("per_state") or {}).items():
        out.append("  %s: %s calls, abstain rate %s, label distribution %s"
                   % (sid, _num(d.get("calls")), _pct(d.get("abstain_rate")),
                      d.get("labels") or {}))
    out.append("Outcome-anchored proxy: %s submissions claimed verified, %s of them wrong (%s)"
               % (_num(oa.get("verified_submissions")),
                  _num(oa.get("verified_but_wrong")), _pct(oa.get("rate"))))
    out.append("  " + str(je.get("note") or ""))

    out += ["", "What this report cannot show", "-" * 60]
    for line in _limits(rep):
        out.append("  · " + line)
    return "\n".join(out)


def _limits(rep: Mapping) -> list:
    """Honesty list: what these numbers **cannot measure**. The report needs this section, or readers over-read it."""
    prov = rep.get("provenance") or {}
    n = (prov.get("n_problems") or "?")
    runs = ((prov.get("config") or {}).get("runs") or "?")
    iso = prov.get("sandbox_isolation") or {}
    return [
        "The sample is only this large: %s test problems × %s runs. Any difference between arms is only a direction, "
        "not a conclusion; this report runs no significance test and gives no confidence intervals" % (n, runs),
        "The endpoint ignores the seed (measured: the same input run twice does not take the same path), so there is "
        "no guarantee that \"a rerun gives the same numbers\". Path consistency measures exactly that",
        "The network is not blocked at the operating-system level (sandbox network_blocked = %s). Child processes could "
        "in principle still go online; this experiment merely has no reason to, which is not the same as being unable to"
        % (iso.get("network_blocked") if iso else "not recorded"),
        "The loop bound K is **introduced by the compiler**; the original skill text has no such number. Arm 3 "
        "therefore follows one more rule than the skill document, and that rule must not be credited to \"the skill\"",
        "Judge actions have no per-call gold labels at test time; the test-time column of the sixth number is the abstain "
        "rate plus an outcome-anchored proxy, measured differently from the compile-time calibration: do not subtract them",
        "Scores are graded by a program, with LaTeX equivalence decided by sympy; equivalent forms it cannot recognize "
        "count as wrong. All three arms bear this error equally, but it is not zero",
        "Arm 2 rereads SKILL.md at every step, so the token difference includes the cost of this executor's "
        "implementation and cannot be attributed entirely to \"having the skill installed\"",
    ]


# --------------------------------------------------------------------------- #
# Inline SVG charts: assembled from plain strings, no matplotlib, no JS library
# --------------------------------------------------------------------------- #
def _nice_max(v: float) -> float:
    """Raise the y-axis upper bound to a nice round number. All zeros give 1, to avoid division by zero."""
    if v <= 0:
        return 1.0
    mag = 10.0 ** (len(str(int(v))) - 1)
    for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if v <= step * mag:
            return step * mag
    return 10.0 * mag


def _bar_chart(title: str, categories: Sequence[str], groups: Sequence[Mapping], *,
               width: int = 720, height: int = 300, fmt: str = "{:,.0f}",
               y_label: str = "") -> str:
    """A grouped + stacked bar chart, returned as an inline SVG string.

    ``groups`` = ``[{"name": group name, "segments": [{"label":…, "color":…, "values":[one per category]}]}]``.
    Each category draws ``len(groups)`` bars side by side, and each bar stacks its own segments from bottom to top;
    that is how arm 3's fallback segment gets a color of its own.

    It paints its own background and picks its own foreground colors: this chart gets embedded in light pages as well
    as dark pages.
    """
    n_cat = max(1, len(categories))
    n_grp = max(1, len(groups))

    # Lay out the legend first: entry widths are estimated with wide (CJK) characters counting as two cells, entries
    # wrap when a row is full, and the number of rows decides the bottom margin. Without laying it out first, wide
    # labels would be measured at half width and overlap: the most easily overlooked collapse in a chart.
    legend: list = []
    for grp in groups:
        for seg in (grp.get("segments") or []):
            label = str(seg.get("label") or "")
            if n_grp > 1 and grp.get("name"):
                label = ("%s · %s" % (grp.get("name"), label) if label
                         else str(grp["name"]))
            pair = (label, seg.get("color") or _C_MAIN)
            if pair not in legend:
                legend.append(pair)

    ml, mr, mt = 62, 18, 38
    rows: list = [[]]
    x = ml
    for label, color in legend:
        w = 22 + 5.4 * sum(2 if ord(c) > 0x2E80 else 1 for c in label)
        if x + w > width - mr and rows[-1]:
            rows.append([])
            x = ml
        rows[-1].append((label, color, x))
        x += w
    mb = 40 + 14 * max(1, len(rows))
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    totals = []
    for gi in range(n_grp):
        segs = list(groups[gi].get("segments") or [])
        for ci in range(n_cat):
            totals.append(sum(float((s.get("values") or [0] * n_cat)[ci] or 0)
                              for s in segs))
    top = _nice_max(max(totals) if totals else 0.0)

    def y_of(v: float) -> float:
        return mt + plot_h - (float(v) / top) * plot_h

    parts = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d" role="img" '
             'aria-label="%s">' % (width, height, height, _esc(title))]
    parts.append('<rect x="0" y="0" width="%d" height="%d" rx="8" fill="%s"/>'
                 % (width, height, _PANEL))
    parts.append('<text x="%d" y="24" fill="%s" font-family="system-ui,sans-serif" '
                 'font-size="14" font-weight="600">%s</text>'
                 % (ml - 46, _INK, _esc(title)))

    # grid and y-axis ticks
    for i in range(5):
        v = top * i / 4.0
        y = y_of(v)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                     'stroke-width="1"/>' % (ml, y, width - mr, y, _GRID))
        parts.append('<text x="%d" y="%.1f" fill="%s" font-family="system-ui,sans-serif"'
                     ' font-size="10" text-anchor="end">%s</text>'
                     % (ml - 6, y + 3, _MUTED, _esc(fmt.format(v))))
    if y_label:
        parts.append('<text x="%d" y="%d" fill="%s" font-family="system-ui,sans-serif" '
                     'font-size="10">%s</text>' % (ml - 46, mt + plot_h + 34, _MUTED,
                                                   _esc(y_label)))

    band = plot_w / n_cat
    bar_w = min(46.0, (band * 0.68) / n_grp)
    for ci, cat in enumerate(categories):
        cx = ml + band * (ci + 0.5)
        start = cx - (bar_w * n_grp) / 2.0
        for gi, grp in enumerate(groups):
            x = start + bar_w * gi
            base = 0.0
            for seg in (grp.get("segments") or []):
                vals = seg.get("values") or [0] * n_cat
                v = float(vals[ci] or 0) if ci < len(vals) else 0.0
                if v <= 0:
                    continue
                y0, y1 = y_of(base + v), y_of(base)
                parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                             'fill="%s" rx="2"/>'
                             % (x + 2, y0, bar_w - 4, max(1.0, y1 - y0),
                                seg.get("color") or _C_MAIN))
                base += v
            if base > 0:
                parts.append('<text x="%.1f" y="%.1f" fill="%s" font-size="10" '
                             'font-family="system-ui,sans-serif" text-anchor="middle">'
                             '%s</text>'
                             % (x + bar_w / 2.0, y_of(base) - 4, _INK,
                                _esc(fmt.format(base))))
        parts.append('<text x="%.1f" y="%d" fill="%s" font-size="11" '
                     'font-family="system-ui,sans-serif" text-anchor="middle">%s</text>'
                     % (cx, mt + plot_h + 18, _INK, _esc(cat)))

    # legend: positions were laid out above; only drawing here
    for ri, row in enumerate(rows):
        ly = height - 12 - 14 * (len(rows) - 1 - ri)
        for label, color, lx in row:
            parts.append('<rect x="%.1f" y="%.1f" width="10" height="10" rx="2" '
                         'fill="%s"/>' % (lx, ly - 9, color))
            parts.append('<text x="%.1f" y="%.1f" fill="%s" font-size="10" '
                         'font-family="system-ui,sans-serif">%s</text>'
                         % (lx + 14, ly, _MUTED, _esc(label)))
    parts.append("</svg>")
    return "".join(parts)


def _chart_accuracy(rep: Mapping, arms: Sequence[str]) -> str:
    per = (rep.get("score") or {}).get("per_arm") or {}
    strict = [(per.get(a) or {}).get("correct") or 0 for a in arms]
    lenient = [(per.get(a) or {}).get("lenient_correct") or 0 for a in arms]
    return _bar_chart(
        "1. Correct runs (strict requires \\boxed{}, lenient does not)", list(arms),
        [{"name": "strict", "segments": [{"label": "strict correct", "color": _C_MAIN,
                                        "values": strict}]},
         {"name": "lenient", "segments": [{"label": "lenient correct", "color": _C_ALT,
                                        "values": lenient}]}],
        fmt="{:,.0f}", y_label="runs (arm × problem × repetition)")


def _chart_tokens(rep: Mapping, arms: Sequence[str]) -> str:
    per = (rep.get("cost") or {}).get("per_arm") or {}
    p_main = [(per.get(a) or {}).get("prompt_tokens") or 0 for a in arms]
    p_fb = [(per.get(a) or {}).get("fallback_prompt_tokens") or 0 for a in arms]
    c_main = [(per.get(a) or {}).get("completion_tokens") or 0 for a in arms]
    c_fb = [(per.get(a) or {}).get("fallback_completion_tokens") or 0 for a in arms]
    return _bar_chart(
        "4. Total tokens (arm 3's fallback segment stacked separately)", list(arms),
        [{"name": "prompt",
          "segments": [{"label": "main segment", "color": _C_MAIN, "values": p_main},
                       {"label": "fallback segment", "color": _C_FALLBACK, "values": p_fb}]},
         {"name": "completion",
          "segments": [{"label": "main segment", "color": _C_ALT, "values": c_main},
                       {"label": "fallback segment", "color": _C_FALLBACK, "values": c_fb}]}],
        fmt="{:,.0f}", y_label="token")


def _chart_consistency(rep: Mapping, arms: Sequence[str]) -> str:
    per = (rep.get("path_consistency") or {}).get("per_arm") or {}
    vals = [100.0 * float((per.get(a) or {}).get("consistency") or 0.0) for a in arms]
    return _bar_chart(
        "2. Path consistency", list(arms),
        [{"name": "", "segments": [{"label": "share of run pairs with identical action sequences",
                                    "color": _C_OK, "values": vals}]}],
        fmt="{:,.0f}", y_label="%")


# --------------------------------------------------------------------------- #
# Self-contained HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root{--bg:#f7f8fa;--fg:#1b2029;--muted:#5d6675;--card:#ffffff;--line:#e2e6ec;
      --accent:#2f5fd0;--warn:#8a5a00;--warnbg:#fff6df;}
@media (prefers-color-scheme: dark){
  :root{--bg:#12151c;--fg:#e7ebf2;--muted:#9aa4b8;--card:#1a1f29;--line:#2b323e;
        --accent:#8ab0ff;--warn:#f0c460;--warnbg:#2a2312;}
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC",sans-serif;}
main{max-width:1040px;margin:0 auto;padding:28px 20px 72px;}
h1{font-size:24px;margin:0 0 6px;}
h2{font-size:18px;margin:34px 0 10px;padding-top:14px;border-top:1px solid var(--line);}
h3{font-size:14px;margin:18px 0 6px;color:var(--muted);font-weight:600;}
p{margin:8px 0;}
.sub{color:var(--muted);margin:0 0 14px;}
.banner{background:var(--warnbg);color:var(--warn);border:1px solid var(--line);
        border-radius:10px;padding:12px 14px;margin:14px 0 6px;font-weight:600;}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
      padding:14px 16px;margin:12px 0;}
.chart{background:var(--card);border:1px solid var(--line);border-radius:12px;
       padding:10px;margin:14px 0;}
.scroll{overflow-x:auto;}
table{border-collapse:collapse;width:100%;font-size:13px;}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;
      white-space:nowrap;}
th:first-child,td:first-child{text-align:left;}
thead th{color:var(--muted);font-weight:600;border-bottom:2px solid var(--line);}
tbody tr:hover{background:rgba(127,127,127,.08);}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;}
.note{color:var(--muted);font-size:12.5px;margin:8px 0 0;}
ul{margin:8px 0;padding-left:20px;}
li{margin:6px 0;}
.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;
     padding:1px 9px;margin:2px 4px 2px 0;font-size:12px;color:var(--muted);}
.dash{color:var(--muted);}
"""


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A table. Wide tables scroll horizontally on their own; the page body never scrolls horizontally."""
    out = ['<div class="scroll"><table><thead><tr>']
    out += ["<th>%s</th>" % _esc(h) for h in headers]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join("<td>%s</td>" % _cell(c) for c in row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _cell(v: Any) -> str:
    if v is None or v == "—":
        return '<span class="dash">—</span>'
    if isinstance(v, (dict, list)):
        return '<span class="mono">%s</span>' % _esc(json.dumps(v, ensure_ascii=False))
    return _esc(v)


def _tags(items: Sequence[Any], empty: str = "(none)") -> str:
    if not items:
        return '<span class="dash">%s</span>' % _esc(empty)
    return "".join('<span class="tag">%s</span>' % _esc(x) for x in items)


def render_html(report: Any, coverage: Optional[Mapping] = None) -> str:
    """Render the three-arm experiment report as one **self-contained** HTML page: no external styles, no CDN, no JS library.

    The charts are inline SVG assembled by this module. ``coverage`` is the output of :func:`cover_report`; if it is
    not given, the copy carried by the report itself (its ``coverage`` entry) is used.
    """
    rep = _as_dict(report)
    cov = dict(coverage or rep.get("coverage") or {})
    arms = _arms_of(rep) or ["bare", "skill", "machine"]
    prov = rep.get("provenance") or {}
    cfg = prov.get("config") or {}
    counts = rep.get("counts") or {}
    h: list = []

    h.append("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">")
    h.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    h.append("<title>Three-arm experiment report · hexis</title>")
    h.append("<style>%s</style></head><body><main>" % _CSS)

    # ---- header ---- #
    h.append("<h1>Three-arm experiment report</h1>")
    h.append('<p class="sub">An Agent Skill compiled into an extended finite state machine, run side by side with '
             "\"no skill\" and \"skill installed, interpreted\" on the same batch of MATH-500 test problems.</p>")
    h.append('<div class="banner">Sample size n=%s test problems × %s runs × %d arms. At this scale only '
             "<strong>directional</strong> conclusions are supported: this page holds only raw counts and derived ratios, with "
             "no significance tests, no confidence intervals and no claim that any arm is \"better\".%s</div>"
             % (_esc(_num(prov.get("n_problems"))), _esc(_num(cfg.get("runs"))),
                len(arms),
                "" if rep.get("complete") else
                " <strong>Note: this version of the report is not finished (complete=false).</strong>"))

    # ---- provenance ---- #
    skill = prov.get("skill") or {}
    caps = prov.get("harness_caps") or {}
    iso = prov.get("sandbox_isolation") or {}
    mach = prov.get("machine") or {}
    h.append("<h2>Provenance (where these numbers come from)</h2>")
    h.append(_table(["Item", "Value"], [
        ["Model id", prov.get("model_id") or "—"],
        ["Skill", "%s @ %s" % (skill.get("slug") or "—",
                              (skill.get("commit") or "—")[:12])],
        ["SKILL.md sha256", (skill.get("skill_md_sha256") or "—")[:16]],
        ["Data sha256", (prov.get("data_sha256") or "—")[:16]],
        ["Split seed", prov.get("split_seed")],
        ["Repository commit", (prov.get("repo_commit") or "—")[:12]],
        ["Executor", "%s (%s)" % (caps.get("engine") or "—",
                                 "native tool-calling" if caps.get("supports_native_tools")
                                 else "no native tool-calling, uses the JSON tool protocol")],
        ["Machine", ("%s v%s, %s states (including %s judge actions), repair budget %s"
                  % (mach.get("skill_id"), mach.get("version"), mach.get("n_states"),
                     mach.get("judge_states"), mach.get("retry_budget")))
         if mach else "—"],
        ["Sandbox isolation", "timeout %s | kill tree %s | memory limit %s | network blocked %s"
         % (iso.get("timeout"), iso.get("kill_tree"),
            (iso.get("mem_limit") if "mem_limit" in iso else iso.get("mem_limit_mb")),
            iso.get("network_blocked"))],
        ["Platform", "%s / Python %s" % (prov.get("platform") or "—",
                                     prov.get("python") or "—")],
        ["Wall clock", "%s s" % _num(prov.get("wall_s"))],
        ["Run counts", "planned %s | reused %s | executed now %s | failed %s"
         % (_num(counts.get("planned")), _num(counts.get("reused")),
            _num(counts.get("executed")), _num(counts.get("failed")))],
        ["Determinism", prov.get("determinism") or "—"],
    ]))

    # ---- 1. score ---- #
    score = rep.get("score") or {}
    per = score.get("per_arm") or {}
    h.append("<h2>1. Score</h2>")
    h.append('<div class="chart">%s</div>' % _chart_accuracy(rep, arms))
    h.append(_table(["Arm", "Runs", "Strict correct", "Strict accuracy", "Lenient correct", "Lenient accuracy",
                     "no_boxed", "No answer", "Crashed"],
                    [[_ARM_LABEL.get(a, a), (per.get(a) or {}).get("runs"),
                      (per.get(a) or {}).get("correct"),
                      _pct((per.get(a) or {}).get("accuracy")),
                      (per.get(a) or {}).get("lenient_correct"),
                      _pct((per.get(a) or {}).get("lenient_accuracy")),
                      (per.get(a) or {}).get("no_boxed"),
                      (per.get(a) or {}).get("no_answer"),
                      (per.get(a) or {}).get("failed_runs")] for a in arms]))
    h.append('<h3>Route distribution (the comparator that matched; no_boxed is a format failure, not a math failure)</h3>')
    h.append(_table(["Arm", "route → count"],
                    [[a, (per.get(a) or {}).get("routes") or {}] for a in arms]))
    h.append("<h3>Per-problem table (correct / runs)</h3>")
    h.append(_table(["Problem", "Level", "Subject"] + [_ARM_LABEL.get(a, a) for a in arms],
                    [[tid, row.get("level"), row.get("subject")]
                     + ["%s / %s" % (((row.get("arms") or {}).get(a) or {}).get("correct", 0),
                                     ((row.get("arms") or {}).get(a) or {}).get("runs", 0))
                        for a in arms]
                     for tid, row in (score.get("per_problem") or {}).items()]))
    h.append('<p class="note">%s</p>' % _esc(score.get("note") or ""))

    # ---- 2. path consistency ---- #
    pc = rep.get("path_consistency") or {}
    pper = pc.get("per_arm") or {}
    h.append("<h2>2. Path consistency</h2>")
    h.append('<div class="chart">%s</div>' % _chart_consistency(rep, arms))
    h.append(_table(["Arm", "Problems", "Run pairs", "Pairs with identical sequences", "Consistency"],
                    [[_ARM_LABEL.get(a, a), (pper.get(a) or {}).get("problems"),
                      (pper.get(a) or {}).get("pairs_total"),
                      (pper.get(a) or {}).get("pairs_identical"),
                      _pct((pper.get(a) or {}).get("consistency"))] for a in arms]))
    h.append('<p class="note">%s</p>' % _esc(pc.get("note") or ""))

    # ---- 3. process compliance ---- #
    comp = rep.get("compliance") or {}
    cper = comp.get("per_arm") or {}
    h.append("<h2>3. Process compliance</h2>")
    h.append(_table(["Arm", "Ran verification", "Verification rate", "Verified before submitting",
                     "Repairs over budget", "Over-budget rate", "Unverified submissions", "Labelled honestly",
                     "Claimed verified without verifying", "P1 violations"],
                    [[_ARM_LABEL.get(a, a), (cper.get(a) or {}).get("ran_verification"),
                      _pct((cper.get(a) or {}).get("verification_rate")),
                      (cper.get(a) or {}).get("verified_before_submit"),
                      (cper.get(a) or {}).get("repair_exceeded"),
                      _pct((cper.get(a) or {}).get("repair_exceeded_rate")),
                      (cper.get(a) or {}).get("unverified_submissions"),
                      (cper.get(a) or {}).get("unverified_labelled"),
                      (cper.get(a) or {}).get("mislabelled_verified"),
                      (cper.get(a) or {}).get("p1_violations")] for a in arms]))
    h.append("<h3>Terminal kind distribution</h3>")
    h.append(_table(["Arm", "terminal_kind → count"],
                    [[a, (cper.get(a) or {}).get("terminal_kinds") or {}]
                     for a in arms]))
    h.append('<p class="note">%s</p>' % _esc(comp.get("note") or ""))

    # ---- 4. cost ---- #
    cost = rep.get("cost") or {}
    kper = cost.get("per_arm") or {}
    h.append("<h2>4. Cost</h2>")
    h.append('<div class="chart">%s</div>' % _chart_tokens(rep, arms))
    h.append(_table(["Arm", "Model calls", "Calls per problem", "prompt", "completion",
                     "Fallback prompt", "Fallback completion", "Total tokens", "Unmeasured runs"],
                    [[_ARM_LABEL.get(a, a), (kper.get(a) or {}).get("llm_calls"),
                      (kper.get(a) or {}).get("llm_calls_per_problem"),
                      (kper.get(a) or {}).get("prompt_tokens"),
                      (kper.get(a) or {}).get("completion_tokens"),
                      (kper.get(a) or {}).get("fallback_prompt_tokens"),
                      (kper.get(a) or {}).get("fallback_completion_tokens"),
                      ((kper.get(a) or {}).get("total_prompt_tokens") or 0)
                      + ((kper.get(a) or {}).get("total_completion_tokens") or 0),
                      (kper.get(a) or {}).get("runs_unmeasured")] for a in arms]))
    h.append('<p class="note">%s</p>' % _esc(cost.get("note") or ""))

    # ---- 5. fallback rate ---- #
    fb = rep.get("fallback") or {}
    h.append("<h2>5. Fallback rate (arm 3 only)</h2>")
    h.append(_table(["Item", "Value"], [
        ["Runs", fb.get("runs")],
        ["Entered the fallback segment", fb.get("entered")],
        ["Fallback rate", _pct(fb.get("rate"))],
        ["Finished within the machine", fb.get("finished_in_machine")],
        ["Mean steps in the machine segment", fb.get("machine_steps_mean")],
    ]))
    h.append("<h3>States the runs fell back from</h3>")
    h.append(_table(["State", "Count"],
                    sorted((fb.get("entry_states") or {}).items())) if fb.get("entry_states")
             else '<p class="note">No run in this batch entered the fallback segment.</p>')
    h.append(_table(["Reason", "Count"], sorted((fb.get("entry_reasons") or {}).items())))
    h.append('<p class="note">%s</p>' % _esc(fb.get("note") or ""))

    # ---- 6. judge action error rate ---- #
    je = rep.get("judge_error") or {}
    ct = je.get("compile_time") or {}
    tt = je.get("test_time") or {}
    oa = tt.get("outcome_anchored") or {}
    h.append("<h2>6. Judge action error rate: compile-time calibration vs test-time observation</h2>")
    h.append(_table(["Judge state", "Clause", "Calibrated error rate (compile time)", "Calibration support",
                     "Calls (test time)", "Abstain rate (test time)", "Label distribution (test time)"],
                    [[sid, d.get("clause"), d.get("error_rate"), d.get("support"),
                      ((tt.get("per_state") or {}).get(sid) or {}).get("calls", 0),
                      _pct(((tt.get("per_state") or {}).get(sid) or {}).get("abstain_rate")),
                      ((tt.get("per_state") or {}).get(sid) or {}).get("labels") or {}]
                     for sid, d in (ct.get("judge_states") or {}).items()]))
    h.append(_table(["Item", "Value"], [
        ["Sum of compile-time error rates (a component of the path error bound)", ct.get("error_sum")],
        ["Threshold judge_err_max", ct.get("threshold")],
        ["Total judge calls (test time)", tt.get("judge_calls")],
        ["Abstain rate (test time)", _pct(tt.get("abstain_rate"))],
        ["Outcome-anchored: submissions claimed verified", oa.get("verified_submissions")],
        ["Outcome-anchored: of which the answer was wrong", oa.get("verified_but_wrong")],
        ["Outcome-anchored error rate", _pct(oa.get("rate"))],
    ]))
    h.append('<p class="note">%s</p>' % _esc(je.get("note") or ""))
    h.append('<p class="note">%s</p>' % _esc(oa.get("definition") or ""))

    # ---- coverage ---- #
    h.append("<h2>Coverage report (how much compilation learned)</h2>")
    if cov:
        h.append(_table(["Item", "Value"], [
            ["States (excluding end states)", cov.get("n_states")],
            ["Document clauses in total", cov.get("clauses_total")],
            ["Supported clauses", len(cov.get("clauses_covered") or [])],
            ["T+ replayed", "%s / %s" % (cov.get("t_plus_reproduced"),
                                     cov.get("t_plus_total"))],
            ["T- excluded", "%s / %s" % (cov.get("t_minus_excluded"),
                                     cov.get("t_minus_total"))],
            ["Sum of judge error rates", cov.get("judge_error_sum")],
            ["Support threshold min_support", cov.get("min_support")],
        ]))
        h.append("<h3>Supported clauses (with states and trace evidence)</h3>")
        h.append('<div class="card">%s</div>' % _tags(cov.get("clauses_covered") or []))
        h.append("<h3>Clauses with thin evidence (weakest support below min_support)</h3>")
        h.append('<div class="card">%s</div>' % _tags(cov.get("clauses_thin") or []))
        h.append("<h3>Clauses never reached (no compile-set trace reached them; at run time they go to FALLBACK)</h3>")
        h.append('<div class="card">%s</div>'
                 % _tags(cov.get("clauses_never_touched")
                         or cov.get("clauses_untriggered") or []))
    else:
        h.append('<p class="note">This version of the report carries no coverage account (it can be produced separately'
                 " with <code>cover_report</code>).</p>")

    # ---- honesty list ---- #
    h.append("<h2>What this report <strong>cannot</strong> show</h2><ul>")
    for line in _limits(rep):
        h.append("<li>%s</li>" % _esc(line))
    h.append("</ul>")
    h.append('<p class="note">This page contains no external resources: styles are inline and the charts are '
             "hand-written inline SVG, so it opens unchanged without a network connection.</p>")
    h.append("</main></body></html>")
    return "".join(h)
