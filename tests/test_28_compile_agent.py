"""㉘ 算法 1「顺序转写编译」：编译智能体 + 确定性守门程序。

盯的是 :mod:`skill2fsm.compile_agent` 与 :mod:`skill2fsm.checker` 之间的分工：**智能体只
提议，守门程序才写机器**。所以这套测试问四件事——

1. ``model=None`` 走确定性启发式时，编出来的机器结构上一条问题都没有，且每条接受轨迹都
   复述得出来（**「没有模型也必须能编出一台正确的机器」是这套方法的地板**）；
2. 会破坏互斥的提议被守门程序**拒掉**，而**它之前被接受的提议一个都不掉**；
3. 同一个点上连拒两次 ⇒ ``demote_to_fallback``，退回解释执行，**不是崩**；
4. 覆盖报告说得出「这条条款没有任何轨迹碰过」「这条循环上限是我加的」。

外加一条底线：同样的输入编两次，机器**逐字节相同**。

密闭：``model=None``，不碰网络、不写文件；轨迹由 conftest 的 ``accepted`` 夹具用手写参考
机器 + 脚本模型桩现造。
"""

import json

import pytest

from skill2fsm import checks, compile_agent, replay, report, runtime
from skill2fsm.checker import Checker
from skill2fsm.compile_agent import Proposal, apply_plan, compile_skill, draft_judge
from skill2fsm.examples import table_clean as tc
from skill2fsm.model_iface import ScriptedModel
from skill2fsm.schema import FALLBACK, Record, Trace

#: 轨迹与编译结果都是确定性的，按 (条数, 种子) 缓存一份，别让每个用例都重跑一遍 24 个任务。
_TRACES: dict = {}
_COMPILED: dict = {}


def _traces(accepted, n=24, seed=2):
    if (n, seed) not in _TRACES:
        _TRACES[(n, seed)] = accepted(n, seed=seed)
    return _TRACES[(n, seed)]


def _compiled(accepted, n=24, seed=2):
    if (n, seed) not in _COMPILED:
        _COMPILED[(n, seed)] = compile_skill(tc, _traces(accepted, n, seed), (),
                                             model=None)
    return _COMPILED[(n, seed)]


def _judge_state(machine):
    return next(s for s in machine.states.values() if s.action.kind == "judge")


def _neg_trace() -> Trace:
    """一条拒绝轨迹：读完直接导出，**跳过了表头检查**，出错位置就在导出那一步。"""
    inp = {"path": "in.csv", "output_path": "out.csv", "request": "清洗这张表并导出"}
    bad = ",单价,库存"
    rows = [["v0", "v1", "v2"]]
    v1 = {**inp, "header_row": bad, "rows": rows}
    v2 = {**v1, "output_path": "out.csv"}
    return Trace(
        task={"task_id": "neg-skip-check", "input": dict(inp)},
        verdict="rejected", error_step=2,
        records=[
            Record(step=1, state="", action={"kind": "tool", "name": "read_csv",
                                             "input": {"path": "in.csv"}},
                   output={"ok": True, "header_row": bad, "rows": rows}, vars=v1),
            Record(step=2, state="",
                   action={"kind": "tool", "name": "export",
                           "input": {"header_row": bad, "rows": rows,
                                     "output_path": "out.csv",
                                     "source_path": "in.csv"}},
                   output={"ok": True, "output_path": "out.csv"}, vars=v2),
            Record(step=3, state="", action={"kind": "end", "terminal": "done"},
                   vars=v2),
        ])


# --------------------------------------------------------------------------- #
# ① model=None：确定性启发式也必须编出一台正确的机器
# --------------------------------------------------------------------------- #
def test_compiles_the_toy_with_no_model(accepted):
    res = _compiled(accepted)
    assert checks.structural_findings(res.machine) == [], \
        checks.structural_findings(res.machine)
    assert res.stats["model_calls"] == 0            # 一次模型都没调
    assert res.stats["committed"] is True, res.stats["commit"]


def test_every_accepted_trace_replays(accepted):
    res = _compiled(accepted)
    traces = _traces(accepted)
    bad = [t.task.get("task_id") for t in traces if not replay.reproduces(res.machine, t)]
    assert bad == [], f"这些接受轨迹复述不出来: {bad}"


def test_learns_read_judge_repair_export(accepted):
    """四段主干都学到了，且修复成了一个**带计数、带上限出口**的环。"""
    m = _compiled(accepted).machine
    names = {s.action.name for s in m.states.values() if s.action.kind == "tool"}
    assert {"read_csv", "fix_header", "export"} <= names
    assert any(s.action.kind == "judge" for s in m.states.values())

    back = [(src, t) for src, t in m.transitions_all() if t.inc]
    assert back, "修复该形成一条带计数的回边"
    for _src, t in back:
        tgt = m.states[t.to]
        assert any(g.cond and t.inc in g.cond for g in tgt.transitions), \
            "回边的计数变量必须在目标状态上有一条上限出口"


def test_branch_condition_is_built_on_the_judge_output(accepted):
    m = _compiled(accepted).machine
    j = _judge_state(m)
    guarded = [t for t in j.transitions if t.cond and t.to != FALLBACK]
    assert len(guarded) >= 2, "判断之后应当有两条带条件的分支"
    assert all(j.action.writes[0] in t.cond for t in guarded)


def test_compiled_machine_runs_fresh_tasks(accepted):
    """学出的机器要泛化到没见过的任务，而不是把训练轨迹背下来。"""
    m = _compiled(accepted).machine
    ok = 0
    for task in tc.gen_tasks(20, seed=99):
        fs = tc.MemFS(task["files"])
        r = runtime.run_task(m, task, model=tc.build_model(),
                             tools=tc.build_registry(fs), doc=tc.skill_doc())
        if r.stopped == "terminal" and tc.verify(task, r.trace):
            ok += 1
    assert ok == 20


def _neg_overwrite() -> Trace:
    """一条**结构上与正例一模一样**的拒绝轨迹：导出目标就是源文件（触犯 P1）。

    这种反例在图上没有可偏离之处——它要靠禁止项拦，而编译产物没有禁止项（``table_clean``
    这个技能对象本身不带 ``prohibitions``，编译器也不会从手写参考机器上顺手拿）。所以它逼
    出的正是 L12 的最后一招：修不动就退回解释执行。
    """
    inp = {"path": "in.csv", "output_path": "in.csv", "request": "清洗这张表并导出"}
    good, rows = "名称,数量,日期", [["v0", "v1", "v2"]]
    v1 = {**inp, "header_row": good, "rows": rows}
    v2 = {**v1, "header_ok": "规范"}
    return Trace(
        task={"task_id": "neg-overwrite", "input": dict(inp)},
        verdict="rejected", error_step=3,
        records=[
            Record(step=1, action={"kind": "tool", "name": "read_csv",
                                   "input": {"path": "in.csv"}},
                   output={"ok": True, "header_row": good, "rows": rows}, vars=v1),
            Record(step=2, action={"kind": "judge", "prompt": tc.JUDGE_Q,
                                   "reads": ["header_row"]},
                   output={"header_ok": "规范"}, vars=v2),
            Record(step=3, action={"kind": "tool", "name": "export",
                                   "input": {"header_row": good, "rows": rows,
                                             "output_path": "in.csv",
                                             "source_path": "in.csv"}},
                   output={"ok": True, "output_path": "in.csv"}, vars=v2),
            Record(step=4, action={"kind": "end", "terminal": "done"}, vars=v2),
        ])


def test_negative_trace_is_excluded(accepted):
    """L12：跳过表头检查的反例，机器要在它的出错位置或更早偏离。"""
    res = compile_skill(tc, _traces(accepted), [_neg_trace()], model=None)
    assert replay.excludes(res.machine, _neg_trace())
    assert res.coverage["verify"]["excluded"] == 1
    assert res.stats["committed"] is True, res.stats["commit"]


def test_unexcludable_negative_forces_a_demotion(accepted):
    """L12 的最后一招：排除不掉的反例 ⇒ 把那一段退回解释执行，而不是硬编下去。

    退完之后机器仍然合法、正例仍然复述得出来、那条反例落进「尚不可排除」而不是「漏掉了」
    ——少编了一截，但没编错。
    """
    neg = _neg_overwrite()
    res = compile_skill(tc, _traces(accepted), [neg], model=None)

    assert res.stats["fallback_demotions"] >= 1
    assert res.coverage["fallback_surface"]["demoted"], "该记下退了哪个点"
    assert checks.structural_findings(res.machine) == []
    assert all(replay.reproduces(res.machine, t) for t in _traces(accepted))
    v = res.coverage["verify"]
    assert v["unexcluded"] == [] and len(v["fallback_deferred"]) == 1
    assert res.stats["committed"] is True, res.stats["commit"]


def test_one_trace_is_not_enough_to_compile_anything(accepted):
    """L14：支持度不足的边整条拿掉。一条轨迹上的边支持度都是 1，**什么都不该编下来**。"""
    one = [t for t in _traces(accepted) if len(t.records) == 4][:1]
    res = compile_skill(tc, one, (), model=None)
    assert res.stats["thin_edges_dropped"] >= 1
    assert res.machine.n_states() <= 1                 # 顶多剩个起点，之后全交给解释执行
    assert checks.structural_findings(res.machine) == []
    assert replay.reproduces(res.machine, one[0])


def test_no_traces_yields_a_valid_all_fallback_machine():
    """没有轨迹就没有可学的东西：交一台合法的、全回退的空机器，而不是崩。"""
    res = compile_skill(tc, (), (), model=None)
    assert res.machine.initial == FALLBACK
    assert checks.structural_findings(res.machine) == []
    assert res.stats["committed"] is True
    assert report.render(res.coverage).startswith("技能编译覆盖报告")


@pytest.mark.parametrize("form,want", [
    ("module", "table-clean"),                          # examples.table_clean 模块
    ("dict", "x"),                                      # {"skill_id": ..., "doc": ...}
    ("dir", "table-clean"),                             # 一个 Agent Skill 目录
    ("text", "compiled"),                               # 光一段正文
])
def test_skill_argument_forms(accepted, form, want):
    skill = {"module": tc, "dict": {"skill_id": "x", "doc": tc.skill_doc()},
             "dir": str(tc.SKILL_PATH.parent), "text": tc.skill_doc()}[form]
    res = compile_skill(skill, _traces(accepted)[:6], (), model=None)
    assert res.machine.skill_id == want
    assert len(res.coverage["clause_table"]) == 6       # SKILL.md 切得出 6 条条款


# --------------------------------------------------------------------------- #
# ② 坏提议被拒，已接受的前缀一个不掉
# --------------------------------------------------------------------------- #
def _open_on(machine) -> Checker:
    ck = Checker(machine.skill_id, thresholds=machine.thresholds)
    assert ck.open_machine(base=machine).accepted
    return ck


def _overlapping(sid: str, cond: str) -> Proposal:
    """一条会与 ``sid`` 上已有条件重叠的加边提议——违互斥（定理2），必被拒。"""
    return Proposal("add_transition",
                    {"from_state": sid, "to": FALLBACK, "cond": cond, "support": 4},
                    rationale="故意与已有分支条件重叠的坏提议")


def test_mutual_exclusion_breaking_proposal_is_rejected_and_prefix_survives(accepted):
    m = _compiled(accepted).machine
    sid = _judge_state(m).id
    ck = _open_on(m)
    before = ck.machine.model_dump_json(by_alias=True)

    res = apply_plan(ck, [_overlapping(sid, "header_ok != '弃权'")])

    assert res.rejected == 1 and res.accepted == 0
    receipt = res.receipts[0]
    assert not receipt.accepted
    assert "重叠" in receipt.reason or "E_OVERLAP" in receipt.reason, receipt.reason
    # 被拒的提议**一个字节都没落到机器上**，先前接受的那一批完好无损
    assert ck.machine.model_dump_json(by_alias=True) == before
    assert res.demoted == []


def test_two_consecutive_rejections_demote_to_fallback(accepted):
    """同一个点连拒两次 ⇒ 退回解释执行。**不是崩，也不是硬塞。**"""
    m = _compiled(accepted).machine
    sid = _judge_state(m).id
    ck = _open_on(m)

    res = apply_plan(ck, [_overlapping(sid, "header_ok != '弃权'"),
                          _overlapping(sid, "header_ok != '规范'")])

    assert res.rejected == 2
    assert res.demoted == [sid], res.demoted
    after = ck.machine
    assert [t.to for t in after.states[sid].transitions] == [FALLBACK]
    assert after.states[sid].action.kind == "judge"      # 动作还在，只是不再往下编
    assert checks.structural_findings(after) == []       # 退回之后机器仍然合法


def test_one_rejection_alone_does_not_demote(accepted):
    """一次被拒只记一笔，不该动机器——两次才退。"""
    m = _compiled(accepted).machine
    sid = _judge_state(m).id
    ck = _open_on(m)
    res = apply_plan(ck, [_overlapping(sid, "header_ok != '弃权'")])
    assert res.demoted == []
    assert len(ck.machine.states[sid].transitions) > 1


def test_receipts_cover_every_proposal(accepted):
    """每条提议一张回执——回滚也不擦，这是审计痕迹。"""
    res = _compiled(accepted)
    ops = [r.op for r in res.receipts]
    assert ops[0] == "open_machine"
    assert ops[-1] == "commit"
    assert all(r.accepted for r in res.receipts), \
        [r.reason for r in res.receipts if not r.accepted]


# --------------------------------------------------------------------------- #
# ③ 覆盖报告
# --------------------------------------------------------------------------- #
def test_coverage_lists_a_clause_no_trace_touched(accepted):
    """P1 是禁止性要求，编不进图、也没有任何轨迹「走」到它——必须在报告里点名。"""
    cov = _compiled(accepted).coverage
    assert "P1" in cov["untouched"], cov["untouched"]
    row = next(r for r in cov["clause_table"] if r["id"] == "P1")
    assert row["status"] == "untouched"
    assert row["states"] == [] and row["traces"] == []
    # model=None ⇒ 条款归属整个留空，报告要如实说是为什么
    assert cov["clause_attribution"] == "none"
    assert set(cov["untouched"]) == {r["id"] for r in cov["clause_table"]}


def test_coverage_says_the_loop_bound_is_compiler_introduced(accepted):
    """SKILL.md 没写过任何圈数上限，K 是编译器为了停机补的——报告必须照实说。"""
    cov = _compiled(accepted).coverage
    assert cov["loop_bounds"], "有回边就该有 K 的台账"
    for lb in cov["loop_bounds"]:
        assert lb["source"] == "compiler"
        assert "文档" in lb["why"]
    kinds = {x["kind"] for x in cov["structures"]["compiler_introduced"]}
    assert {"loop_bound", "counter_variable", "branch_guard", "fallback_surface"} <= kinds


def test_coverage_names_what_needs_a_model(accepted):
    cov = _compiled(accepted).coverage
    dep = cov["model_dependence"]
    assert dep["model_used"] is False
    assert dep["touchpoints"] == list(compile_agent.MODEL_TOUCHPOINTS)
    assert any("learn_cond" in s for s in dep["model_free"])
    assert any("条款归属" in s for s in dep["needs_model"])


def test_coverage_reports_the_fallback_surface(accepted):
    cov = _compiled(accepted).coverage
    fb = cov["fallback_surface"]
    assert fb["n_edges_to_fallback"] >= 1
    assert fb["demoted"] == [] and fb["blocked_branches"] == []
    assert cov["next_traces"], "报告要说得出再补哪些轨迹最值钱"


def test_report_renders_the_coverage(accepted):
    text = report.render(_compiled(accepted).coverage)
    assert "覆盖报告" in text and "T+ 复述" in text


def test_diff_vs_reference_matches_the_target_shape(accepted):
    d = _compiled(accepted).diff_vs_reference
    assert d is not None
    assert d["actions_missing"] == [] and d["actions_extra"] == []
    assert d["n_states_compiled"] == d["n_states_reference"]


# --------------------------------------------------------------------------- #
# ④ 没有模型就不起草判断动作；起草本身要过模式检查
# --------------------------------------------------------------------------- #
def test_no_judge_is_drafted_without_a_model(accepted):
    res = _compiled(accepted)
    assert res.stats["judges_drafted"] == 0
    assert [j["source"] for j in res.judges] == ["trace"]   # 轨迹里本来就有的判断步


def test_draft_judge_needs_a_model():
    assert draft_judge({"reads": ["x"], "targets": {"a": [{"x": 1}]}}, model=None) is None


@pytest.mark.parametrize("reply", [
    {"labels": ["甲", "乙"]},                      # 缺 question
    {"prompt": "  ", "labels": ["甲", "乙"]},     # question 是空白
    {"prompt": "走哪支", "labels": ["甲"]},        # 标签少于两个
    {"prompt": "走哪支"},                         # 没有标签
    "不是一个 JSON 对象",                            # 形状就不对
])
def test_off_schema_reply_is_a_reject_not_a_guess(reply):
    model = ScriptedModel(gen=lambda p, v, h: reply)
    assert draft_judge({"state": "s2", "reads": ["x"],
                        "targets": {"a": [{"x": 1}]}}, model=model) is None


def test_draft_judge_accepts_a_well_formed_reply():
    model = ScriptedModel(gen=lambda p, v, h: {
        "prompt": "这一支该走哪边", "labels": ["甲", "乙"]})
    j = draft_judge({"state": "s2", "reads": ["x", "y"], "writes": ["verdict"],
                     "targets": {"a": [{"x": 1}], "b": [{"x": 2}]}}, model=model)
    assert j is not None
    assert j.writes == ["verdict"] and j.reads == ["x", "y"]
    assert compile_agent.ABSTAIN in j.labels        # 弃权是硬性的，模型忘了也要补上


def test_draft_judge_refuses_reads_outside_the_whitelist():
    """模型想多读一个变量，就是在给判断动作偷偷加上下文——只收白名单里的。"""
    model = ScriptedModel(gen=lambda p, v, h: {
        "prompt": "走哪支", "labels": ["甲", "乙"], "reads": ["x", "偷偷加的"]})
    j = draft_judge({"state": "s2", "reads": ["x"], "writes": ["verdict"],
                     "targets": {"a": [{"x": 1}]}}, model=model)
    assert j is not None and j.reads == ["x"]


# --------------------------------------------------------------------------- #
# ⑤ 有模型时的两处触点（脚本桩，仍然密闭、零成本）
# --------------------------------------------------------------------------- #
_CLAUSE_BY_STEP = {"tool:read_csv": "S1", "tool:fix_header": "S3", "tool:export": "S4"}


def _scripted_agent(prompt: str, values: dict) -> str:
    """一个把两处判定答对的模型桩：新步/重复照 KEY 判，条款归属照动作查表。"""
    if prompt.startswith("这一步是流程里新的一步"):
        cands = values.get("同型的已有状态", "（没有）")
        return "新步" if cands == "（没有）" else "重复:" + cands.split(",")[0]
    if prompt.startswith("这一步在落实技能文档"):
        step = values.get("这一步", "")
        if step.startswith("judge:"):
            return "S2.1"
        return _CLAUSE_BY_STEP.get(step, compile_agent.ABSTAIN)
    return compile_agent.ABSTAIN


def test_model_fills_in_clause_attribution(accepted):
    """条款归属是**要模型**的那一半：给了模型，覆盖报告才说得出每个状态凭哪句话存在。"""
    res = compile_skill(tc, _traces(accepted), (), model=ScriptedModel(judge=_scripted_agent))
    assert checks.structural_findings(res.machine) == []
    got = {sid: s.clause for sid, s in res.machine.states.items() if s.clause}
    assert set(got.values()) == {"S1", "S2.1", "S3", "S4"}
    assert res.coverage["clause_attribution"] == "model"
    assert set(res.coverage["supported"]) == {"S1", "S2.1", "S3", "S4"}
    assert "P1" in res.coverage["untouched"]     # 禁止项永远没有轨迹「走」到
    assert res.stats["model_calls"] > 0 and res.stats["model_rejects"] == 0
    assert res.stats["committed"] is True, res.stats["commit"]


def test_model_only_changes_attribution_not_the_shape(accepted):
    """(a)(b) 两处触点答对时，机器的**形状**该和 model=None 那台一模一样——
    模型只补语义，不改结构。"""
    a = _compiled(accepted).machine
    b = compile_skill(tc, _traces(accepted), (),
                      model=ScriptedModel(judge=_scripted_agent)).machine
    assert sorted(a.states) == sorted(b.states)
    for sid in a.states:
        assert [(t.cond, t.to, t.inc, t.support) for t in a.states[sid].transitions] == \
            [(t.cond, t.to, t.inc, t.support) for t in b.states[sid].transitions]


def test_off_schema_new_or_repeat_reply_is_rejected_then_demoted(accepted):
    """(a) 的回复恒不合模式 ⇒ 同一个点连拒两次 ⇒ 那一段退回解释执行，机器仍然合法。"""
    def always_abstain(prompt, values):
        return compile_agent.ABSTAIN if prompt.startswith("这一步是流程里新的一步") \
            else compile_agent.ABSTAIN

    res = compile_skill(tc, _traces(accepted), (),
                        model=ScriptedModel(judge=always_abstain))
    assert res.stats["model_rejects"] >= 2
    assert res.stats["blocked_branches"], "连拒到上限的那个点该被记下来"
    assert checks.structural_findings(res.machine) == []
    assert all(replay.reproduces(res.machine, t) for t in _traces(accepted))


# --------------------------------------------------------------------------- #
# ⑥ 确定性
# --------------------------------------------------------------------------- #
def test_compile_skill_is_deterministic(accepted):
    traces = _traces(accepted)
    a = compile_skill(tc, traces, (), model=None)
    b = compile_skill(tc, traces, (), model=None)
    assert a.machine.model_dump_json(by_alias=True) == \
        b.machine.model_dump_json(by_alias=True)
    assert json.dumps(a.coverage["clause_table"], ensure_ascii=False, sort_keys=True) == \
        json.dumps(b.coverage["clause_table"], ensure_ascii=False, sort_keys=True)
    assert [r.op for r in a.receipts] == [r.op for r in b.receipts]


def test_trace_order_does_not_change_the_machine(accepted):
    """L2 说「步数少的优先」——喂进来的顺序因此不该影响产物。"""
    traces = _traces(accepted)
    a = compile_skill(tc, traces, (), model=None)
    b = compile_skill(tc, list(reversed(traces)), (), model=None)
    assert a.machine.model_dump_json(by_alias=True) == \
        b.machine.model_dump_json(by_alias=True)
