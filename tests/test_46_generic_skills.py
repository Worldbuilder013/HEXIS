"""编译器不预设技能形态：只读检索、纯文本生成、文件修改、自定义工具流程各编一遍。

全部密闭：工具定义手写在测试里，初始机器手写（不调模型），轨迹合成。检查的是：
上下文只来自文档 / 工具定义 / 轨迹 / 规则；没有 apply 的技能照样更新；工具名原样保留；
未知工具接口标为推断；认不出的事件不被丢掉；``hexis/compiler`` 里没有任何技能或字段名。
"""
from __future__ import annotations

import pathlib
import re

import pytest

from hexis.compiler import check as C
from hexis.compiler.context import build_context, parse_rules
from hexis.compiler.traces import prepare
from hexis.compiler.update import update
from hexis.machine.schema import Machine, Record, Trace
from hexis.tools.toolspec import ToolSpec

FSM_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "hexis" / "compiler"


# --------------------------------------------------------------------------- #
# 合成轨迹
# --------------------------------------------------------------------------- #
def tool(step, name, args, out=None, ok=True):
    out = dict(out or {})
    out.setdefault("ok", ok)
    return Record(step=step, action={"kind": "tool", "name": name, "input": args}, output=out)


def model(step, text="thinking", **structured):
    return Record(step=step, action={"kind": "model"}, output={"reply": text, **structured})


def end(step, terminal="done"):
    return Record(step=step, action={"kind": "end", "terminal": terminal})


def trace(records, inputs, tid="t1", verdict="accepted"):
    return Trace(task={"task_id": tid, "input": dict(inputs)}, verdict=verdict, records=records)


def machine(states: dict, variables: list, terminals: list, initial: str) -> Machine:
    data = {"format": "efsm-v1", "skill_id": "test", "initial": initial, "fallback": "FALLBACK",
            "variables": [{"name": n, **spec} for n, spec in variables],
            "terminals": [{"id": t, "kind": k} for t, k in terminals] + [{"id": "END_FALLBACK", "kind": "fallback"}],
            "states": {}}
    for sid, (action, edges) in states.items():
        data["states"][sid] = {"id": sid, "origin": "document", "action": action,
                               "transitions": [{"if": c, "to": t, **({"inc": inc} if inc else {})}
                                               for c, t, inc in edges]}
    for t, _k in terminals:
        data["states"][t] = {"id": t, "origin": "document", "action": {"kind": "end", "terminal": t}}
    data["states"]["FALLBACK"] = {"id": "FALLBACK", "origin": "compiler",
                                  "action": {"kind": "end", "terminal": "END_FALLBACK"}}
    return Machine.model_validate(data)


def spec(name, inputs, outputs, success, primary, description="", label=""):
    return ToolSpec(name=name, description=description,
                    input_schema={k: {"type": "string", "required": True} for k in inputs},
                    output_schema={k: "string" for k in outputs}, success=success, primary=primary,
                    label=label, source="registry")


# --------------------------------------------------------------------------- #
# 1. 只读检索：search → open → 总结。没有任何修改，照样能更新。
# --------------------------------------------------------------------------- #
RETRIEVAL_DOC = """# Research
Search for sources with the search tool. Open at least one result before answering.
The answer is the final message."""


@pytest.fixture
def retrieval():
    reg = {"search": spec("search", ["query"], ["ok", "results"], "ok == True", "results"),
           "open": spec("open", ["url"], ["ok", "text"], "ok == True", "text")}
    rules = parse_rules({
        "terminals": [{"id": "ANSWERED", "kind": "answered"}, {"id": "INCOMPLETE", "kind": "incomplete"}],
        "requirements": [{"id": "R1", "kind": "before", "a": {"tool": "open"},
                          "b": {"kind": "model", "role": "output"},
                          "quote": "Open at least one result before answering."}],
        "terminal_conditions": [{"terminal": "ANSWERED",
                                 "required_evidence": [{"tool": "open", "success": True},
                                                       {"kind": "model", "role": "output"}],
                                 "invalidating_events": [],
                                 "quote": "The answer is the final message."}]})
    inputs = {"question": "why is the sky blue"}
    good = trace([model(1, "let me search"), tool(2, "search", {"query": "sky blue"}, {"results": "r1"}),
                  tool(3, "open", {"url": "http://a"}, {"text": "rayleigh"}),
                  model(4, "Because of Rayleigh scattering."), end(5)], inputs, "good")
    bad = trace([tool(1, "search", {"query": "sky"}, {"results": "r"}), model(2, "guess"), end(3)],
                inputs, "no_open")
    ctx = build_context("research", RETRIEVAL_DOC, [], [good, bad], registry=reg, rules=rules)
    m0 = machine({
        "s1": ({"kind": "model", "prompt": "Write the search query. Return {\"query\": ...}",
                "reads": ["question"], "writes": ["query"]}, [("", "s2", None)]),
        "s2": ({"kind": "tool", "name": "search", "input": {"query": "${query}"},
                "writes": ["ok", "results"]}, [("ok == True", "s3", None), ("", "INCOMPLETE", None)]),
        "s3": ({"kind": "model", "prompt": "Pick a url from results. Return {\"url\": ...}",
                "reads": ["results"], "writes": ["url"]}, [("", "s4", None)]),
        "s4": ({"kind": "tool", "name": "open", "input": {"url": "${url}"},
                "writes": ["ok", "text"]}, [("ok == True", "s5", None), ("", "INCOMPLETE", None)]),
        "s5": ({"kind": "model", "observable": True, "prompt": "Answer from text. Return {\"answer\": ...}",
                "reads": ["question", "text"], "writes": ["answer"]}, [("", "ANSWERED", None)]),
    }, [("question", {"init_from": "task.input.question"}), ("query", {}), ("results", {}),
        ("url", {}), ("text", {}), ("answer", {}), ("ok", {"type": "boolean"})],
        [("ANSWERED", "answered"), ("INCOMPLETE", "incomplete")], "s1")
    return ctx, m0, good, bad


def test_retrieval_context_from_traces_and_registry(retrieval):
    ctx, _m0, _good, _bad = retrieval
    assert list(ctx.task_inputs) == ["question"] and ctx.task_inputs["question"].always_present
    assert ctx.tools["search"].source == "registry" and ctx.tools["search"].calls == 2
    assert ctx.default_terminal() == "INCOMPLETE"


def test_retrieval_accepts_read_only_trace_and_excludes_requirement_violation(retrieval):
    ctx, m0, good, bad = retrieval
    assert C.check(m0, ctx) == []
    p = prepare(good, ctx)
    assert [e.describe() for e in p.observable] == ["search×1✓", "open×1✓", "model:output", "end:done"]
    assert p.tau == "ANSWERED"
    res = update(m0, [("good.jsonl", good, ""), ("no_open.jsonl", bad, "")], ctx)
    assert res.counts() == {"accepted": 1, "excluded": 1}, res.entries
    assert "R1" in res.entries[1]["why"]
    assert C.check(res.machine, ctx) == []
    assert {st.action.name for st in res.machine.states.values() if st.action.kind == "tool"} == {"search", "open"}


# --------------------------------------------------------------------------- #
# 2. 纯文本生成：草稿是结构化的模型产出，lint 通过才交付；重写草稿使证据失效
# --------------------------------------------------------------------------- #
@pytest.fixture
def textgen():
    reg = {"lint": spec("lint", ["text"], ["ok", "issues"], "ok == True", "issues")}
    rules = parse_rules({
        "terminals": [{"id": "DELIVERED", "kind": "delivered"}, {"id": "GAVE_UP", "kind": "gave_up"}],
        "terminal_conditions": [{"terminal": "DELIVERED",
                                 "required_evidence": [{"tool": "lint", "success": True}],
                                 "invalidating_events": [{"kind": "model", "role": "output"}],
                                 "quote": "Only deliver when lint passes."}]})
    inputs = {"topic": "otters"}
    good = trace([model(1, "drafting", draft="Otters are..."), tool(2, "lint", {"text": "Otters are..."}, {"issues": ""}),
                  end(3)], inputs, "good")
    redo = trace([model(1, "d1", draft="v1"), tool(2, "lint", {"text": "v1"}, {"issues": "x"}, ok=False),
                  model(3, "d2", draft="v2"), tool(4, "lint", {"text": "v2"}, {"issues": ""}), end(5)], inputs, "redo")
    ctx = build_context("textgen", "# Write\nDraft the text. Only deliver when lint passes.", [],
                        [good, redo], registry=reg, rules=rules)
    m0 = machine({
        "s1": ({"kind": "model", "observable": True, "prompt": "Draft. Return {\"draft\": ...}",
                "reads": ["topic"], "writes": ["draft"]},
               [("tries >= 3", "GAVE_UP", None), ("", "s2", None)]),
        "s2": ({"kind": "tool", "name": "lint", "input": {"text": "${draft}"}, "writes": ["ok", "issues"]},
               [("ok == True", "DELIVERED", None), ("", "s1", "tries")]),
    }, [("topic", {"init_from": "task.input.topic"}), ("draft", {}), ("issues", {}),
        ("ok", {"type": "boolean"}), ("tries", {"type": "integer", "init": 0})],
        [("DELIVERED", "delivered"), ("GAVE_UP", "gave_up")], "s1")
    return ctx, m0, good, redo


def test_textgen_structured_model_output_is_observable_and_drives_evidence(textgen):
    ctx, m0, good, redo = textgen
    assert C.check(m0, ctx) == []
    p = prepare(redo, ctx)
    assert [e.describe() for e in p.observable] == ["model:output", "lint×1✗", "model:output", "lint×1✓", "end:done"]
    assert p.tau == "DELIVERED"
    res = update(m0, [("good.jsonl", good, ""), ("redo.jsonl", redo, "")], ctx)
    assert res.counts() == {"accepted": 2}, res.entries
    assert C.check(res.machine, ctx) == []


# --------------------------------------------------------------------------- #
# 3. 文件修改：修改后读产出 → 派生标签 verify → 已验证终点
# --------------------------------------------------------------------------- #
@pytest.fixture
def filemod():
    reg = {"bash": spec("bash", ["command"], ["ok", "returncode", "stdout"], "returncode == 0", "stdout"),
           "write": spec("write", ["filePath", "content"], ["ok", "returncode", "stdout"], "returncode == 0",
                         "stdout", label="apply")}
    rules = parse_rules({
        "terminals": [{"id": "END_VERIFIED", "kind": "verified"}, {"id": "END_UNVERIFIED", "kind": "unverified"}],
        "labels": [{"label": "verify", "when": {"kind": "tool", "label": "probe", "args_contain": "${output_path}",
                                                 "after": {"label": "apply"}},
                    "quote": "Reopen the saved file after writing it."}],
        "terminal_conditions": [{"terminal": "END_VERIFIED",
                                 "required_evidence": [{"label": "verify", "success": True}],
                                 "invalidating_events": [{"label": "apply"}],
                                 "quote": "Reopen the saved file after writing it."}]})
    inputs = {"request": "add a line", "input_path": "/w/in.txt", "output_path": "/w/out.txt"}
    good = trace([tool(1, "bash", {"command": "cat /w/in.txt"}, {"returncode": 0, "stdout": "a"}),
                  tool(2, "write", {"filePath": "out.txt", "content": "a\nb"}, {"returncode": 0, "stdout": ""}),
                  tool(3, "bash", {"command": "cat out.txt"}, {"returncode": 0, "stdout": "a\nb"}),
                  model(4, "done"), end(5)], inputs, "good")
    unverified = trace([tool(1, "bash", {"command": "cat /w/in.txt"}, {"returncode": 0, "stdout": "a"}),
                        tool(2, "write", {"filePath": "/w/out.txt", "content": "x"}, {"returncode": 0, "stdout": ""}),
                        end(3)], inputs, "unverified")
    ctx = build_context("filemod", "# Edit\nReopen the saved file after writing it.", [],
                        [good, unverified], registry=reg, rules=rules)
    m0 = machine({
        "s1": ({"kind": "model", "prompt": "Write inspect command. Return {\"inspect_cmd\": ...}",
                "reads": ["request", "input_path"], "writes": ["inspect_cmd"]}, [("", "s2", None)]),
        "s2": ({"kind": "tool", "name": "bash", "phase": "probe", "input": {"command": "${inspect_cmd}"},
                "writes": ["returncode", "stdout"]}, [("returncode == 0", "s3", None), ("", "END_UNVERIFIED", None)]),
        "s3": ({"kind": "model", "prompt": "Write new content and check command. Return {\"content\": ..., \"check_cmd\": ...}",
                "reads": ["request", "stdout", "output_path"], "writes": ["content", "check_cmd"]}, [("", "s4", None)]),
        "s4": ({"kind": "tool", "name": "write", "phase": "apply", "input": {"filePath": "${output_path}", "content": "${content}"},
                "writes": ["returncode", "stdout"]}, [("returncode == 0", "s5", None), ("", "END_UNVERIFIED", None)]),
        "s5": ({"kind": "tool", "name": "bash", "phase": "probe", "labels": ["verify"], "input": {"command": "${check_cmd}"},
                "writes": ["returncode", "stdout"]}, [("returncode == 0", "END_VERIFIED", None), ("", "END_UNVERIFIED", None)]),
    }, [("request", {"init_from": "task.input.request"}), ("input_path", {"init_from": "task.input.input_path"}),
        ("output_path", {"init_from": "task.input.output_path"}), ("inspect_cmd", {}), ("content", {}),
        ("check_cmd", {}), ("returncode", {"type": "integer", "init": 0}), ("stdout", {})],
        [("END_VERIFIED", "verified"), ("END_UNVERIFIED", "unverified")], "s1")
    return ctx, m0, good, unverified


def test_filemod_derived_label_and_verified_terminal(filemod):
    ctx, m0, good, unverified = filemod
    assert C.check(m0, ctx) == []
    p = prepare(good, ctx)
    assert p.events[2].labels == {"verify"} and p.tau == "END_VERIFIED"
    assert prepare(unverified, ctx).tau == "END_UNVERIFIED"
    res = update(m0, [("good.jsonl", good, ""), ("unverified.jsonl", unverified, "")], ctx)
    assert res.counts() == {"accepted": 2}, res.entries
    assert C.check(res.machine, ctx) == []
    s4 = res.machine.states["s4"].action
    assert s4.input["filePath"] == "${output_path}"                     # 相对路径也认出是产出路径


def test_filemod_machine_without_verify_before_verified_end_fails_check(filemod):
    ctx, m0, _g, _u = filemod
    m0.states["s4"].transitions[0].to = "END_VERIFIED"                  # 修改后直接宣称已验证
    errs = C.check(m0, ctx)
    assert any("证据" in e and "END_VERIFIED" in e for e in errs), errs


# --------------------------------------------------------------------------- #
# 4. 自定义工具流程：query_db → aggregate → 总结。工具没有注册表定义，接口只从轨迹推断。
# --------------------------------------------------------------------------- #
@pytest.fixture
def custom():
    rules = parse_rules({
        "terminals": [{"id": "REPORTED", "kind": "reported"}, {"id": "ABORTED", "kind": "aborted"}],
        "requirements": [{"id": "R1", "kind": "before", "a": {"tool": "query_db"}, "b": {"tool": "aggregate"},
                          "quote": "Aggregate only rows returned by a query."}],
        "terminal_conditions": [{"terminal": "REPORTED",
                                 "required_evidence": [{"tool": "aggregate", "success": True}],
                                 "invalidating_events": [], "quote": "Report the aggregated figures."}]})
    inputs = {"question": "monthly totals"}
    good = trace([model(1, "plan"), tool(2, "query_db", {"sql": "select * from t"}, {"rows": "[1,2]"}),
                  tool(3, "aggregate", {"rows": "[1,2]", "op": "sum"}, {"value": "3"}),
                  model(4, "Total is 3"), end(5)], inputs, "good")
    ctx = build_context("dbreport", "# Report\nAggregate only rows returned by a query. Report the aggregated figures.",
                        [], [good], registry=None, rules=rules)
    m0 = machine({
        "s1": ({"kind": "model", "prompt": "Write sql. Return {\"sql\": ...}", "reads": ["question"], "writes": ["sql"]},
               [("", "s2", None)]),
        "s2": ({"kind": "tool", "name": "query_db", "input": {"sql": "${sql}"}, "writes": ["ok", "rows"]},
               [("ok == True", "s3", None), ("", "ABORTED", None)]),
        "s3": ({"kind": "model", "prompt": "Choose op. Return {\"op\": ...}", "reads": ["question", "rows"], "writes": ["op"]},
               [("", "s4", None)]),
        "s4": ({"kind": "tool", "name": "aggregate", "input": {"rows": "${rows}", "op": "${op}"}, "writes": ["ok", "value"]},
               [("ok == True", "s5", None), ("", "ABORTED", None)]),
        "s5": ({"kind": "model", "observable": True, "prompt": "Report. Return {\"report\": ...}",
                "reads": ["value"], "writes": ["report"]}, [("", "REPORTED", None)]),
    }, [("question", {"init_from": "task.input.question"}), ("sql", {}), ("rows", {}), ("op", {}),
        ("value", {}), ("report", {}), ("ok", {"type": "boolean"})],
        [("REPORTED", "reported"), ("ABORTED", "aborted")], "s1")
    return ctx, m0, good


def test_custom_tools_are_opaque_and_inferred(custom):
    ctx, m0, good = custom
    assert ctx.tools["query_db"].source == "inferred" and ctx.tools["query_db"].success == "ok == True"
    assert ctx.tools["aggregate"].primary == "value"
    assert any("没有注册表定义" in n for n in ctx.notes)
    assert C.check(m0, ctx) == []
    res = update(m0, [("good.jsonl", good, "")], ctx)
    assert res.counts() == {"accepted": 1}, res.entries
    assert res.entries[0]["anchors"] == ["s2", "s4", "s5", "REPORTED"]


def test_custom_new_tool_state_keeps_opaque_name_and_generates_params(custom):
    ctx, m0, good = custom
    extra = trace([tool(1, "query_db", {"sql": "select 1"}, {"rows": "[1]"}),
                   tool(2, "normalize", {"rows": "[1]", "mode": "z"}, {"rows2": "[0]"}),
                   tool(3, "aggregate", {"rows": "[0]", "op": "sum"}, {"value": "0"}),
                   model(4, "zero"), end(5)], {"question": "q"}, "extra")
    ctx2 = build_context("dbreport", ctx.skill_text, [], [good, extra], registry=None,
                         rules={"terminals": ctx.terminals, "label_rules": ctx.label_rules,
                                "requirements": ctx.requirements, "terminal_conditions": ctx.terminal_conditions})
    res = update(m0, [("good.jsonl", good, ""), ("extra.jsonl", extra, "")], ctx2)
    assert res.counts() == {"accepted": 2}, res.entries
    new = [st for st in res.machine.states.values() if st.action.kind == "tool" and st.action.name == "normalize"]
    assert len(new) == 1
    assert new[0].action.input == {"rows": "${t1_rows}", "mode": "${t1_mode}"}
    gate = res.machine.states[f"{new[0].id}_gate"]
    assert set(gate.action.writes) == {"t1_rows", "t1_mode"} and "question" in gate.action.reads


# --------------------------------------------------------------------------- #
# 通用性守卫
# --------------------------------------------------------------------------- #
def test_unknown_event_kind_is_reported_not_dropped(retrieval):
    ctx, m0, good, _bad = retrieval
    weird = trace([tool(1, "search", {"query": "x"}, {"results": "r"}),
                   Record(step=2, action={"kind": "telemetry"}, output={}),
                   tool(3, "open", {"url": "u"}, {"text": "t"}), model(4, "a"), end(5)], {"question": "q"}, "weird")
    res = update(m0, [("weird.jsonl", weird, "")], ctx)
    assert res.counts() == {"unsupported": 1}
    assert "telemetry" in res.entries[0]["why"]


@pytest.mark.parametrize("word", ["openpyxl", "workbook", "xlsx", "file_ops", "input_path", "output_path",
                                  "filePath", "returncode", "stdout", "bash"])
def test_compiler_has_no_domain_or_tool_names(word):
    hits = []
    for p in sorted(FSM_DIR.glob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(rf"\b{re.escape(word)}\b", line):
                hits.append(f"{p.name}:{i}")
    assert not hits, f"{word!r} 出现在编译器里：{hits}"
