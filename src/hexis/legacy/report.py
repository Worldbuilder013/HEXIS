"""覆盖报告、上下文探针，以及三臂实验报告的两种渲染（纯文本 / 自包含 HTML）。

编译交出的不只是一台机器，还有一份「它凭什么这么走、哪些地方没学到」的账：哪些条款被轨迹
触发过、哪些没有（一律接 FALLBACK）、每个判断动作的误差率、一条路径累计的误差上界、跑一
批任务时落进 FALLBACK 的比例。这份账让「机器保住了多少、还漏着多少」可审阅，而不是藏在
一个通过率数字后面。

:func:`context_probe` 对照判断动作的两种上下文——只给它声明要读的变量（窄读，编译产物用的
就是这个）与多给一些（宽读）——看窄读会不会让产出质量下降。路径对了但内容悄悄变差，是这类
编译最隐蔽的失败，探针把它显式地摆出来。

:func:`render_experiment` 与 :func:`render_html` 渲染 :mod:`hexis.experiment` 跑出来的
六个数。**HTML 页面完全自包含**：没有外部样式、没有 CDN、没有 JS 库，图表是本模块用普通
字符串拼出来的内联 SVG。理由很实际——这份报告要能塞进附件、能在没有网的机器上打开、十年后
还打得开；任何一个外链都会让它在某一天变成一页空白。图表自己画一层不透明底色并按那层底色
挑前景色，因此贴进浅色或深色页面都读得清，不依赖宿主的默认背景。
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
# 覆盖账
# --------------------------------------------------------------------------- #
def _min_support(machine: Machine, state: Any) -> int:
    """一个状态身上**最弱**的那处证据。判断动作比它的 support，分岔比出边的 support。"""
    vals = [t.support for t in (state.transitions or [])]
    if state.action.kind == "judge":
        vals.append(state.action.support)
    return min(vals) if vals else 0


def cover_report(machine: Machine, clauses: list, *, t_plus=(), t_minus=(),
                 runs=()) -> dict:
    """一台机器的覆盖账。``clauses`` 是文档条款列表 ``[(id, text)]``；``runs`` 是若干
    :class:`~hexis.execution.runtime.RunResult`，用来算 FALLBACK 落入率。

    ``clauses_thin`` 是**证据薄**的那些条款：它们对应的状态上，最弱的一处支持度还没到
    ``thresholds.min_support``。支持度 0 也算薄——那表示这条边根本没有记录在案的轨迹证据
    （手写的参考机器因此整台都是薄的，这是实话，不是 bug）。
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
    """对每个判断动作，比较窄读与宽读的产出是否一致。

    在真实任务流里采集判断动作的输入快照，分别用「只读声明变量」和「额外多给
    ``wide_extra`` 的上下文」跑判断，报告两者的一致率。一致率低说明窄读丢了信息、路径虽对
    但判断质量在下降。玩具桩两种上下文同判，一致率为 1——形式上完成这道探针，真实价值在
    真实模型上显现。
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
    """把覆盖账渲染成一段人读的文本。"""
    lines = ["技能编译覆盖报告", "=" * 32,
             f"状态数（不含终止）: {report.get('n_states')}",
             f"条款覆盖: {report.get('clauses_covered')}",
             f"未触发条款: {report.get('clauses_untriggered') or '（无）'}",
             f"判断动作误差率之和（路径误差上界的构件）: {report.get('judge_error_sum')}"]
    for sid, d in (report.get("judge_states") or {}).items():
        lines.append(f"  判断 {sid}: 误差率 {d['error_rate']}, 支持度 {d['support']}")
    lines.append(f"T+ 复述: {report.get('t_plus_reproduced')}/{report.get('t_plus_total')}")
    lines.append(f"T- 排除: {report.get('t_minus_excluded')}/{report.get('t_minus_total')}")
    if report.get("fallback_rate") is not None:
        lines.append(f"FALLBACK 落入率: {report.get('fallback_rate')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 三臂实验报告：共用的小工具
# --------------------------------------------------------------------------- #
#: 六个数的标题，顺序即交付物里的编号。
_METRIC_TITLES = (
    ("score", "① 分数"),
    ("path_consistency", "② 路径一致率"),
    ("compliance", "③ 过程遵从"),
    ("cost", "④ 开销"),
    ("fallback", "⑤ 回退率"),
    ("judge_error", "⑥ 判断动作错误率"),
)

_ARM_LABEL = {"bare": "臂一 bare（不装技能）",
              "skill": "臂二 skill（装技能，解释执行）",
              "machine": "臂三 machine（编译出的状态机 + 回退段）"}

#: 图表自带的一层不透明底色，以及按它挑的前景色。**不吃宿主的默认背景**——这几张图会被贴进
#: 浅色页面，也会被贴进深色页面，颜色写死才两边都读得清。
_PANEL = "#232a36"
_INK = "#e8ecf3"
_MUTED = "#9aa4b8"
_GRID = "#39404e"
#: 前景色板：蓝（主段）、浅蓝（第二序列）、琥珀（回退段）、绿（一致率）。
_C_MAIN = "#5b8ff9"
_C_ALT = "#a7c7ff"
_C_FALLBACK = "#f6bd16"
_C_OK = "#5ad8a6"


def _as_dict(report: Any) -> dict:
    """吃 :class:`~hexis.experiment.ExperimentReport` 或它的 dict，都归一成 dict。"""
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return dict(report or {})


def _arms_of(rep: Mapping) -> list:
    """报告里出现过的臂，按 bare / skill / machine 的固定顺序（多出来的排在后面）。"""
    seen: list = []
    for key in ("score", "cost", "compliance", "path_consistency"):
        seen += list(((rep.get(key) or {}).get("per_arm") or {}).keys())
    order = ["bare", "skill", "machine"]
    known = [a for a in order if a in seen]
    return known + sorted({a for a in seen if a not in order})


def _num(v: Any, dash: str = "—") -> str:
    """数字转字符串。``None`` 一律显示成破折号——「没测到」不能长得像 0。"""
    if v is None:
        return dash
    if isinstance(v, float):
        return ("%.4f" % v).rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


def _pct(v: Any) -> str:
    return "—" if v is None else "%.1f%%" % (100.0 * float(v))


def _esc(s: Any) -> str:
    """XML/HTML 文本转义。SVG 片段要能当 XML 解析，所以五个字符一个都不能漏。"""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


# --------------------------------------------------------------------------- #
# 纯文本
# --------------------------------------------------------------------------- #
def render_experiment(report: Any) -> str:
    """把三臂实验报告渲染成一段人读的文本。终端里直接能看，不需要浏览器。"""
    rep = _as_dict(report)
    arms = _arms_of(rep)
    prov = rep.get("provenance") or {}
    counts = rep.get("counts") or {}
    out: list = ["三臂实验报告（hexis）", "=" * 60]

    n_prob = prov.get("n_problems")
    runs = ((prov.get("config") or {}).get("runs"))
    out.append("样本量 n=%s 道测试题 × %s 次运行 × %d 条臂 —— 只支持方向性结论，"
               "不做显著性声称。" % (_num(n_prob), _num(runs), len(arms)))
    out.append("模型 %s ｜ 机器 %s ｜ 数据 sha %s"
               % (prov.get("model_id") or "—",
                  (prov.get("machine") or {}).get("skill_id") or "（无）",
                  str(prov.get("data_sha256") or "")[:12]))
    out.append("技能 %s @ %s"
               % ((prov.get("skill") or {}).get("slug") or "—",
                  ((prov.get("skill") or {}).get("commit") or "—")[:12]))
    out.append("墙钟 %s 秒 ｜ 计划 %s 次运行（复用 %s，本次跑 %s，失败 %s）｜ 报告%s"
               % (_num(prov.get("wall_s")), _num(counts.get("planned")),
                  _num(counts.get("reused")), _num(counts.get("executed")),
                  _num(counts.get("failed")),
                  "完整" if rep.get("complete") else "**未跑完**"))

    # ---- ① 分数 ---- #
    score = rep.get("score") or {}
    out += ["", "① 分数（原始计数）", "-" * 60]
    out.append("%-12s %6s %8s %8s %8s %10s" % ("臂", "运行", "严档对", "松档对",
                                               "no_boxed", "严档准确率"))
    for a in arms:
        d = (score.get("per_arm") or {}).get(a) or {}
        out.append("%-12s %6s %8s %8s %8s %10s"
                   % (a, _num(d.get("runs")), _num(d.get("correct")),
                      _num(d.get("lenient_correct")), _num(d.get("no_boxed")),
                      _pct(d.get("accuracy"))))
    for a in arms:
        d = (score.get("per_arm") or {}).get(a) or {}
        out.append("  %s 的 route 分布: %s" % (a, d.get("routes") or {}))
    out.append("  " + str(score.get("note") or ""))
    out.append("  逐题表：")
    out.append("  %-20s %3s %s" % ("题", "级", " ".join("%-9s" % a for a in arms)))
    for tid, row in (score.get("per_problem") or {}).items():
        cells = []
        for a in arms:
            c = (row.get("arms") or {}).get(a) or {}
            cells.append("%-9s" % ("%s/%s" % (_num(c.get("correct", 0)),
                                              _num(c.get("runs", 0)))))
        out.append("  %-20s %3s %s" % (tid, _num(row.get("level")), " ".join(cells)))

    # ---- ② 路径一致率 ---- #
    pc = rep.get("path_consistency") or {}
    out += ["", "② 路径一致率（同题多次运行，动作序列逐对相同的比例）", "-" * 60]
    out.append("%-12s %8s %10s %10s %10s" % ("臂", "题数", "运行对", "相同对", "一致率"))
    for a in arms:
        d = (pc.get("per_arm") or {}).get(a) or {}
        out.append("%-12s %8s %10s %10s %10s"
                   % (a, _num(d.get("problems")), _num(d.get("pairs_total")),
                      _num(d.get("pairs_identical")), _pct(d.get("consistency"))))
    out.append("  " + str(pc.get("note") or ""))

    # ---- ③ 过程遵从 ---- #
    comp = rep.get("compliance") or {}
    out += ["", "③ 过程遵从", "-" * 60]
    out.append("%-12s %8s %10s %10s %12s %10s"
               % ("臂", "跑核验", "核验率", "超预算", "未验证提交", "如实标注"))
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
        out.append("  %s: P1 违规 %s 次（自称已验证却没核验 %s 次），终点类别 %s"
                   % (a, _num(d.get("p1_violations")),
                      _num(d.get("mislabelled_verified")),
                      d.get("terminal_kinds") or {}))
    out.append("  " + str(comp.get("note") or ""))

    # ---- ④ 开销 ---- #
    cost = rep.get("cost") or {}
    out += ["", "④ 开销（每次运行 = 一题一次）", "-" * 60]
    out.append("%-12s %8s %12s %12s %12s %10s"
               % ("臂", "调用", "prompt", "completion", "回退段prompt", "没测到"))
    for a in arms:
        d = (cost.get("per_arm") or {}).get(a) or {}
        out.append("%-12s %8s %12s %12s %12s %10s"
                   % (a, _num(d.get("llm_calls")), _num(d.get("prompt_tokens")),
                      _num(d.get("completion_tokens")),
                      _num(d.get("fallback_prompt_tokens")),
                      _num(d.get("runs_unmeasured"))))
    out.append("  " + str(cost.get("note") or ""))

    # ---- ⑤ 回退率 ---- #
    fb = rep.get("fallback") or {}
    out += ["", "⑤ 回退率（只有臂三有）", "-" * 60]
    out.append("落进回退段 %s / %s 次运行 = %s；机器自己跑完 %s 次"
               % (_num(fb.get("entered")), _num(fb.get("runs")),
                  _pct(fb.get("rate")), _num(fb.get("finished_in_machine"))))
    out.append("退出状态清单: %s" % (fb.get("entry_states") or {}))
    out.append("退出原因: %s" % (fb.get("entry_reasons") or {}))
    out.append("  " + str(fb.get("note") or ""))

    # ---- ⑥ 判断动作错误率 ---- #
    je = rep.get("judge_error") or {}
    ct = je.get("compile_time") or {}
    tt = je.get("test_time") or {}
    oa = tt.get("outcome_anchored") or {}
    out += ["", "⑥ 判断动作错误率（编译期标定 vs 测试期实际）", "-" * 60]
    out.append("编译期：误差率之和 %s（阈值 %s）"
               % (_num(ct.get("error_sum")), _num(ct.get("threshold"))))
    for sid, d in (ct.get("judge_states") or {}).items():
        out.append("  %s 标定误差率 %s，支持度 %s"
                   % (sid, _num(d.get("error_rate")), _num(d.get("support"))))
    out.append("测试期：判断调用 %s 次，弃权 %s 次（%s）"
               % (_num(tt.get("judge_calls")), _num(tt.get("abstained")),
                  _pct(tt.get("abstain_rate"))))
    for sid, d in (tt.get("per_state") or {}).items():
        out.append("  %s 调用 %s 次，弃权率 %s，标签分布 %s"
                   % (sid, _num(d.get("calls")), _pct(d.get("abstain_rate")),
                      d.get("labels") or {}))
    out.append("结局锚定代理量：自称已验证 %s 次，其中答错 %s 次（%s）"
               % (_num(oa.get("verified_submissions")),
                  _num(oa.get("verified_but_wrong")), _pct(oa.get("rate"))))
    out.append("  " + str(je.get("note") or ""))

    out += ["", "这份报告不能证明什么", "-" * 60]
    for line in _limits(rep):
        out.append("  · " + line)
    return "\n".join(out)


def _limits(rep: Mapping) -> list:
    """诚实清单：这份数**测不到**的东西。报告里必须有这一节，不然读者会多读出结论。"""
    prov = rep.get("provenance") or {}
    n = (prov.get("n_problems") or "?")
    runs = ((prov.get("config") or {}).get("runs") or "?")
    iso = prov.get("sandbox_isolation") or {}
    return [
        "样本量就这么大：%s 道测试题 × %s 次运行。任何臂间差值都只是方向，不是结论；"
        "本报告不做显著性检验，也不给置信区间。" % (n, runs),
        "本端点的 seed 不生效（实测：同一入参两次跑不出同一条路径），所以「重跑一遍会得到"
        "同样的数」这件事没有保证。路径一致率量的正是这件事本身。",
        "网络没有在操作系统层被阻断（sandbox 的 network_blocked = %s）。子进程理论上仍能"
        "联网，本实验只是没有理由这么做，不是做不到。"
        % (iso.get("network_blocked") if iso else "未记录"),
        "循环上界 K 是**编译器引入的**，技能原文里没有这个数。臂三因此比技能文档多守了一条"
        "纪律，这一条不能记在「技能」头上。",
        "判断动作在测试期没有逐次的金标准，第六个数里的测试期一栏是弃权率与结局锚定的代理"
        "量，与编译期标定值口径不同，不能相减。",
        "分数由程序判定（grader），LaTeX 等价性靠 sympy；它判不出的等价形式会被计成答错，"
        "这部分误差三条臂同等承受，但不为零。",
        "臂二每一步都要把 SKILL.md 重读一遍，token 差里含着这个执行器实现的代价，不能整个"
        "归给「装了技能」。",
    ]


# --------------------------------------------------------------------------- #
# 内联 SVG 图表：普通字符串拼装，不用 matplotlib、不用任何 JS 库
# --------------------------------------------------------------------------- #
def _nice_max(v: float) -> float:
    """把纵轴上界抬到一个好看的整数。全零时给 1，免得除零。"""
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
    """一张分组 + 堆叠柱状图，返回内联 SVG 字符串。

    ``groups`` = ``[{"name": 组名, "segments": [{"label":…, "color":…, "values":[每类一个]}]}]``。
    一个类目里画 ``len(groups)`` 根柱子并排，每根柱子把自己的 segments 从下往上堆起来——臂三
    的回退段就是这么单列成一段颜色的。

    自己画底色、自己定前景色：这张图会被贴进浅色页面，也会被贴进深色页面。
    """
    n_cat = max(1, len(categories))
    n_grp = max(1, len(groups))

    # 图例先排：条目宽度按「汉字算两格」估，够宽就换行，行数决定底部留多少。
    # 不先排的话，中日韩标签会按半角宽度算而互相压在一起——图上最容易被忽略的一处塌陷。
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

    # 网格与纵轴刻度
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

    # 图例：位置在上面就排好了，这里只画
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
        "① 正确次数（严档要求 \\boxed{}，松档不要求）", list(arms),
        [{"name": "严档", "segments": [{"label": "严档正确", "color": _C_MAIN,
                                        "values": strict}]},
         {"name": "松档", "segments": [{"label": "松档正确", "color": _C_ALT,
                                        "values": lenient}]}],
        fmt="{:,.0f}", y_label="次（臂 × 题 × 第几次）")


def _chart_tokens(rep: Mapping, arms: Sequence[str]) -> str:
    per = (rep.get("cost") or {}).get("per_arm") or {}
    p_main = [(per.get(a) or {}).get("prompt_tokens") or 0 for a in arms]
    p_fb = [(per.get(a) or {}).get("fallback_prompt_tokens") or 0 for a in arms]
    c_main = [(per.get(a) or {}).get("completion_tokens") or 0 for a in arms]
    c_fb = [(per.get(a) or {}).get("fallback_completion_tokens") or 0 for a in arms]
    return _bar_chart(
        "④ token 合计（臂三的回退段单堆一段）", list(arms),
        [{"name": "prompt",
          "segments": [{"label": "主段", "color": _C_MAIN, "values": p_main},
                       {"label": "回退段", "color": _C_FALLBACK, "values": p_fb}]},
         {"name": "completion",
          "segments": [{"label": "主段", "color": _C_ALT, "values": c_main},
                       {"label": "回退段", "color": _C_FALLBACK, "values": c_fb}]}],
        fmt="{:,.0f}", y_label="token")


def _chart_consistency(rep: Mapping, arms: Sequence[str]) -> str:
    per = (rep.get("path_consistency") or {}).get("per_arm") or {}
    vals = [100.0 * float((per.get(a) or {}).get("consistency") or 0.0) for a in arms]
    return _bar_chart(
        "② 路径一致率", list(arms),
        [{"name": "", "segments": [{"label": "相同动作序列的运行对占比",
                                    "color": _C_OK, "values": vals}]}],
        fmt="{:,.0f}", y_label="%")


# --------------------------------------------------------------------------- #
# 自包含 HTML
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
    """一张表。宽表自己横向滚动，页面本体永远不横向滚。"""
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


def _tags(items: Sequence[Any], empty: str = "（无）") -> str:
    if not items:
        return '<span class="dash">%s</span>' % _esc(empty)
    return "".join('<span class="tag">%s</span>' % _esc(x) for x in items)


def render_html(report: Any, coverage: Optional[Mapping] = None) -> str:
    """把三臂实验报告渲染成一页**自包含** HTML：无外部样式、无 CDN、无 JS 库。

    图表是本模块拼出来的内联 SVG。``coverage`` 是 :func:`cover_report` 的产物；不给就用报告
    自带的那份（``ExperimentReport.coverage``）。
    """
    rep = _as_dict(report)
    cov = dict(coverage or rep.get("coverage") or {})
    arms = _arms_of(rep) or ["bare", "skill", "machine"]
    prov = rep.get("provenance") or {}
    cfg = prov.get("config") or {}
    counts = rep.get("counts") or {}
    h: list = []

    h.append("<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">")
    h.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    h.append("<title>三臂实验报告 · hexis</title>")
    h.append("<style>%s</style></head><body><main>" % _CSS)

    # ---- 头 ---- #
    h.append("<h1>三臂实验报告</h1>")
    h.append('<p class="sub">把一个 Agent Skill 编译成扩展有限状态机，再与「不装技能」'
             "「装技能解释执行」并排跑同一批 MATH-500 测试题。</p>")
    h.append('<div class="banner">样本量 n=%s 道测试题 × %s 次运行 × %d 条臂。'
             "这个规模只支持<strong>方向性</strong>结论：本页全是原始计数加派生比例，"
             "不做显著性检验、不给置信区间、不声称任何一条臂「更好」。%s</div>"
             % (_esc(_num(prov.get("n_problems"))), _esc(_num(cfg.get("runs"))),
                len(arms),
                "" if rep.get("complete") else
                " <strong>注意：这一版报告还没跑完（complete=false）。</strong>"))

    # ---- 出身 ---- #
    skill = prov.get("skill") or {}
    caps = prov.get("harness_caps") or {}
    iso = prov.get("sandbox_isolation") or {}
    mach = prov.get("machine") or {}
    h.append("<h2>出身（这份数是怎么来的）</h2>")
    h.append(_table(["项", "值"], [
        ["模型 id", prov.get("model_id") or "—"],
        ["技能", "%s @ %s" % (skill.get("slug") or "—",
                              (skill.get("commit") or "—")[:12])],
        ["SKILL.md sha256", (skill.get("skill_md_sha256") or "—")[:16]],
        ["数据 sha256", (prov.get("data_sha256") or "—")[:16]],
        ["划分种子", prov.get("split_seed")],
        ["仓库提交", (prov.get("repo_commit") or "—")[:12]],
        ["执行器", "%s（%s）" % (caps.get("engine") or "—",
                                 "有原生 tool-calling" if caps.get("supports_native_tools")
                                 else "无原生 tool-calling，走 JSON 工具协议")],
        ["机器", ("%s v%s，%s 个状态（含 %s 个判断动作），修复预算 %s"
                  % (mach.get("skill_id"), mach.get("version"), mach.get("n_states"),
                     mach.get("judge_states"), mach.get("retry_budget")))
         if mach else "—"],
        ["沙箱隔离", "超时 %s ｜ 杀进程树 %s ｜ 内存上限 %s ｜ 网络阻断 %s"
         % (iso.get("timeout"), iso.get("kill_tree"),
            (iso.get("mem_limit") if "mem_limit" in iso else iso.get("mem_limit_mb")),
            iso.get("network_blocked"))],
        ["平台", "%s / Python %s" % (prov.get("platform") or "—",
                                     prov.get("python") or "—")],
        ["墙钟", "%s 秒" % _num(prov.get("wall_s"))],
        ["运行计数", "计划 %s ｜ 复用 %s ｜ 本次执行 %s ｜ 失败 %s"
         % (_num(counts.get("planned")), _num(counts.get("reused")),
            _num(counts.get("executed")), _num(counts.get("failed")))],
        ["确定性", prov.get("determinism") or "—"],
    ]))

    # ---- ① 分数 ---- #
    score = rep.get("score") or {}
    per = score.get("per_arm") or {}
    h.append("<h2>① 分数</h2>")
    h.append('<div class="chart">%s</div>' % _chart_accuracy(rep, arms))
    h.append(_table(["臂", "运行", "严档正确", "严档准确率", "松档正确", "松档准确率",
                     "no_boxed", "无答案", "跑挂"],
                    [[_ARM_LABEL.get(a, a), (per.get(a) or {}).get("runs"),
                      (per.get(a) or {}).get("correct"),
                      _pct((per.get(a) or {}).get("accuracy")),
                      (per.get(a) or {}).get("lenient_correct"),
                      _pct((per.get(a) or {}).get("lenient_accuracy")),
                      (per.get(a) or {}).get("no_boxed"),
                      (per.get(a) or {}).get("no_answer"),
                      (per.get(a) or {}).get("failed_runs")] for a in arms]))
    h.append('<h3>route 分布（命中的比较器；no_boxed 是格式失败，不是数学失败）</h3>')
    h.append(_table(["臂", "route → 次数"],
                    [[a, (per.get(a) or {}).get("routes") or {}] for a in arms]))
    h.append("<h3>逐题表（正确次数 / 运行次数）</h3>")
    h.append(_table(["题", "级", "学科"] + [_ARM_LABEL.get(a, a) for a in arms],
                    [[tid, row.get("level"), row.get("subject")]
                     + ["%s / %s" % (((row.get("arms") or {}).get(a) or {}).get("correct", 0),
                                     ((row.get("arms") or {}).get(a) or {}).get("runs", 0))
                        for a in arms]
                     for tid, row in (score.get("per_problem") or {}).items()]))
    h.append('<p class="note">%s</p>' % _esc(score.get("note") or ""))

    # ---- ② 路径一致率 ---- #
    pc = rep.get("path_consistency") or {}
    pper = pc.get("per_arm") or {}
    h.append("<h2>② 路径一致率</h2>")
    h.append('<div class="chart">%s</div>' % _chart_consistency(rep, arms))
    h.append(_table(["臂", "题数", "运行对", "序列相同的对", "一致率"],
                    [[_ARM_LABEL.get(a, a), (pper.get(a) or {}).get("problems"),
                      (pper.get(a) or {}).get("pairs_total"),
                      (pper.get(a) or {}).get("pairs_identical"),
                      _pct((pper.get(a) or {}).get("consistency"))] for a in arms]))
    h.append('<p class="note">%s</p>' % _esc(pc.get("note") or ""))

    # ---- ③ 过程遵从 ---- #
    comp = rep.get("compliance") or {}
    cper = comp.get("per_arm") or {}
    h.append("<h2>③ 过程遵从</h2>")
    h.append(_table(["臂", "跑过核验", "核验率", "提交前已核验", "修复超预算", "超限率",
                     "未验证提交", "如实标注", "自称已验证却没核验", "P1 违规"],
                    [[_ARM_LABEL.get(a, a), (cper.get(a) or {}).get("ran_verification"),
                      _pct((cper.get(a) or {}).get("verification_rate")),
                      (cper.get(a) or {}).get("verified_before_submit"),
                      (cper.get(a) or {}).get("repair_exceeded"),
                      _pct((cper.get(a) or {}).get("repair_exceeded_rate")),
                      (cper.get(a) or {}).get("unverified_submissions"),
                      (cper.get(a) or {}).get("unverified_labelled"),
                      (cper.get(a) or {}).get("mislabelled_verified"),
                      (cper.get(a) or {}).get("p1_violations")] for a in arms]))
    h.append("<h3>终点类别分布</h3>")
    h.append(_table(["臂", "terminal_kind → 次数"],
                    [[a, (cper.get(a) or {}).get("terminal_kinds") or {}]
                     for a in arms]))
    h.append('<p class="note">%s</p>' % _esc(comp.get("note") or ""))

    # ---- ④ 开销 ---- #
    cost = rep.get("cost") or {}
    kper = cost.get("per_arm") or {}
    h.append("<h2>④ 开销</h2>")
    h.append('<div class="chart">%s</div>' % _chart_tokens(rep, arms))
    h.append(_table(["臂", "模型调用", "每题调用", "prompt", "completion",
                     "回退段 prompt", "回退段 completion", "合计 token", "没测到的运行"],
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

    # ---- ⑤ 回退率 ---- #
    fb = rep.get("fallback") or {}
    h.append("<h2>⑤ 回退率（只有臂三有）</h2>")
    h.append(_table(["项", "值"], [
        ["运行次数", fb.get("runs")],
        ["落进回退段", fb.get("entered")],
        ["回退率", _pct(fb.get("rate"))],
        ["机器自己跑完", fb.get("finished_in_machine")],
        ["机器段平均步数", fb.get("machine_steps_mean")],
    ]))
    h.append("<h3>从哪些状态退下去的</h3>")
    h.append(_table(["退出状态", "次数"],
                    sorted((fb.get("entry_states") or {}).items())) if fb.get("entry_states")
             else '<p class="note">这一批没有任何一次落进回退段。</p>')
    h.append(_table(["退出原因", "次数"], sorted((fb.get("entry_reasons") or {}).items())))
    h.append('<p class="note">%s</p>' % _esc(fb.get("note") or ""))

    # ---- ⑥ 判断动作错误率 ---- #
    je = rep.get("judge_error") or {}
    ct = je.get("compile_time") or {}
    tt = je.get("test_time") or {}
    oa = tt.get("outcome_anchored") or {}
    h.append("<h2>⑥ 判断动作错误率：编译期标定 vs 测试期实际</h2>")
    h.append(_table(["判断状态", "条款", "编译期标定误差率", "标定支持度",
                     "测试期调用次数", "测试期弃权率", "测试期标签分布"],
                    [[sid, d.get("clause"), d.get("error_rate"), d.get("support"),
                      ((tt.get("per_state") or {}).get(sid) or {}).get("calls", 0),
                      _pct(((tt.get("per_state") or {}).get(sid) or {}).get("abstain_rate")),
                      ((tt.get("per_state") or {}).get(sid) or {}).get("labels") or {}]
                     for sid, d in (ct.get("judge_states") or {}).items()]))
    h.append(_table(["项", "值"], [
        ["编译期误差率之和（路径误差上界的构件）", ct.get("error_sum")],
        ["阈值 judge_err_max", ct.get("threshold")],
        ["测试期判断调用总数", tt.get("judge_calls")],
        ["测试期弃权率", _pct(tt.get("abstain_rate"))],
        ["结局锚定：自称已验证的提交", oa.get("verified_submissions")],
        ["结局锚定：其中答案是错的", oa.get("verified_but_wrong")],
        ["结局锚定错误率", _pct(oa.get("rate"))],
    ]))
    h.append('<p class="note">%s</p>' % _esc(je.get("note") or ""))
    h.append('<p class="note">%s</p>' % _esc(oa.get("definition") or ""))

    # ---- 覆盖 ---- #
    h.append("<h2>覆盖报告（编译学到了多少）</h2>")
    if cov:
        h.append(_table(["项", "值"], [
            ["状态数（不含终止）", cov.get("n_states")],
            ["文档条款总数", cov.get("clauses_total")],
            ["被支持的条款数", len(cov.get("clauses_covered") or [])],
            ["T+ 复述", "%s / %s" % (cov.get("t_plus_reproduced"),
                                     cov.get("t_plus_total"))],
            ["T- 排除", "%s / %s" % (cov.get("t_minus_excluded"),
                                     cov.get("t_minus_total"))],
            ["判断误差率之和", cov.get("judge_error_sum")],
            ["支持度门槛 min_support", cov.get("min_support")],
        ]))
        h.append("<h3>被支持的条款（有状态、有轨迹证据）</h3>")
        h.append('<div class="card">%s</div>' % _tags(cov.get("clauses_covered") or []))
        h.append("<h3>证据薄的条款（最弱的一处支持度低于 min_support）</h3>")
        h.append('<div class="card">%s</div>' % _tags(cov.get("clauses_thin") or []))
        h.append("<h3>从未触达的条款（编译集里没有一条轨迹走到过，运行时一律接 FALLBACK）</h3>")
        h.append('<div class="card">%s</div>'
                 % _tags(cov.get("clauses_never_touched")
                         or cov.get("clauses_untriggered") or []))
    else:
        h.append('<p class="note">这一版报告里没有带覆盖账（跑 <code>hexis-agent report'
                 " --machine … --doc …</code> 可以单独出）。</p>")

    # ---- 诚实清单 ---- #
    h.append("<h2>这份报告<strong>不能</strong>说明什么</h2><ul>")
    for line in _limits(rep):
        h.append("<li>%s</li>" % _esc(line))
    h.append("</ul>")
    h.append('<p class="note">本页不含任何外部资源：样式内联、图表是手写的内联 SVG，'
             "断网也能原样打开。</p>")
    h.append("</main></body></html>")
    return "".join(h)
