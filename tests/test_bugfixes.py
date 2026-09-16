"""Regression tests for fixed bugs.

1. Rules written by ``compile`` (key ``label_rules``) lost their label rules when read back.
2. ``compile-stepwise`` identified traces by task id: several traces of one task collided.
3. An accepted trace that was no longer available was silently left out of the protection replay.
4. Endpoint failures were counted as failed drafting rounds, and failed rule extraction continued without rules.
5. JSON repair sent back only the first 6000 characters of a long reply.
"""
from __future__ import annotations

import json

import httpx
import pytest

from hexis.cli import main as cli_main
from hexis.compiler import stepwise as SW
from hexis.compiler.context import build_context, load_rules, parse_rules, rules_dict
from hexis.compiler.init import extract_rules, initialize
from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.llm.llm_client import LLMHTTPError, ModelAdapter, OpenAIClient
from hexis.llm.model_iface import ModelUnavailable
from hexis.machine.schema import Machine
from hexis.traces import judge

RULES = {
    "terminals": [{"id": "DONE", "kind": "done"}],
    "labels": [{"label": "verify", "when": {"kind": "tool", "label": "probe", "after": {"label": "apply"}},
                "quote": "check the output"}],
    "requirements": [{"id": "R1", "kind": "must_occur", "a": {"tool": "export"}, "quote": "export the table"}],
    "terminal_conditions": [{"terminal": "DONE", "required_evidence": [{"tool": "export", "success": True}],
                             "invalidating_events": [], "quote": "export the table"}],
}


# --------------------------------------------------------------------------- #
# 1. rules round trip
# --------------------------------------------------------------------------- #
def test_rules_written_with_label_rules_key_keep_their_labels(tmp_path):
    ctx = build_context("s", "doc", [], [], rules=parse_rules(RULES))
    written = {k: v for k, v in ctx.to_dict().items()
               if k in ("terminals", "label_rules", "requirements", "terminal_conditions")}
    path = tmp_path / "rules_extracted.json"
    path.write_text(json.dumps(written), encoding="utf-8")
    back = load_rules(path)
    assert [r.label for r in back["label_rules"]] == ["verify"]
    assert [r.id for r in back["requirements"]] == ["R1"]


def test_raw_rules_dict_with_label_rules_key_is_parsed():
    ctx = build_context("s", "doc", [], [], rules=parse_rules(RULES))
    raw = json.loads(json.dumps({k: v for k, v in ctx.to_dict().items()
                                 if k in ("terminals", "label_rules", "requirements", "terminal_conditions")}))
    ctx2 = build_context("s", "doc", [], [], rules=raw)
    assert [r.label for r in ctx2.label_rules] == ["verify"]


def test_rules_dict_round_trips(tmp_path):
    ctx = build_context("s", "doc", [], [], rules=parse_rules(RULES))
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules_dict(ctx)), encoding="utf-8")
    ctx2 = build_context("s", "doc", [], [], rules=load_rules(path))
    assert rules_dict(ctx2) == rules_dict(ctx)


# --------------------------------------------------------------------------- #
# 2 + 3. stepwise trace keys and protected traces
# --------------------------------------------------------------------------- #
def _write_example_inputs(tmp_path, traces):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(tc.SKILL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    tdir = tmp_path / "traces"
    tdir.mkdir()
    for name, t in traces:
        (tdir / f"{name}.jsonl").write_text(t.to_jsonl(), encoding="utf-8")
    tools = {"tools": {
        "read_csv": {"input_schema": {"path": {"type": "string", "required": True}},
                     "output_schema": {"ok": "boolean", "header_row": "string", "rows": "array"}, "success": "ok == True"},
        "fix_header": {"input_schema": {"header_row": {"type": "string", "required": True}},
                       "output_schema": {"ok": "boolean", "header_row": "string"}, "success": "ok == True"},
        "export": {"input_schema": {"header_row": {"type": "string"}, "rows": {"type": "array"},
                                    "output_path": {"type": "string"}, "source_path": {"type": "string"}},
                   "output_schema": {"ok": "boolean", "output_path": "string"}, "success": "ok == True",
                   "label": "apply"}}}
    (tmp_path / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    (tmp_path / "m0.json").write_text(tc.reference_machine().model_dump_json(by_alias=True), encoding="utf-8")
    return skill, tdir


def _same_task_traces(n):
    out = []
    ref = tc.reference_machine()
    for i, task in enumerate(tc.gen_tasks(n, seed=3)):
        task = dict(task, task_id="same-task")
        res = runtime.run_task(ref, task, model=tc.build_model(), tools=tc.build_registry(tc.MemFS(task["files"])),
                               doc=tc.skill_doc())
        out.append((f"run{i}", judge.judged(res.trace, lambda t, task=task: tc.verify(task, t), ref.prohibitions)))
    return out


def _stepwise(tmp_path, skill, tdir, *extra):
    return cli_main(["compile-stepwise", "--skill", str(skill), "--traces", str(tdir), "--out", str(tmp_path / "out"),
                     "--tools", str(tmp_path / "tools.json"), "--machine-init", str(tmp_path / "m0.json"), *extra])


def test_stepwise_keys_traces_by_file_name(tmp_path):
    traces = _same_task_traces(4)
    skill, tdir = _write_example_inputs(tmp_path, traces)
    decisions = tmp_path / "dec.json"
    decisions.write_text(json.dumps({name: {"accept_proposals": True} for name, _t in traces}), encoding="utf-8")
    assert _stepwise(tmp_path, skill, tdir, "--apply", str(decisions), "--continue-after-insert") == 0
    assert _stepwise(tmp_path, skill, tdir, "--apply", str(decisions), "--continue-after-insert", "--show", "2") == 0
    progress = json.loads((tmp_path / "out" / "progress.json").read_text(encoding="utf-8"))
    keys = [e["trace"] for e in progress["entries"]]
    assert sorted(keys) == sorted(name for name, _t in traces)
    assert len(keys) == len(set(keys))


def test_stepwise_refuses_to_continue_without_an_accepted_trace(tmp_path):
    traces = _same_task_traces(3)
    skill, tdir = _write_example_inputs(tmp_path, traces)
    decisions = tmp_path / "dec.json"
    decisions.write_text(json.dumps({name: {"accept_proposals": True} for name, _t in traces}), encoding="utf-8")
    assert _stepwise(tmp_path, skill, tdir, "--apply", str(decisions), "--continue-after-insert") == 0
    progress = json.loads((tmp_path / "out" / "progress.json").read_text(encoding="utf-8"))
    accepted = progress["accepted"][0]["trace"]
    (tdir / f"{accepted}.jsonl").unlink()
    with pytest.raises(SystemExit) as exc:
        _stepwise(tmp_path, skill, tdir, "--report")
    assert accepted in str(exc.value)


def test_update_with_decisions_rejects_an_unprepared_accepted_trace():
    traces = _same_task_traces(2)
    ctx = build_context("table-clean", tc.skill_doc(), [], [t for _n, t in traces])
    m = tc.reference_machine()
    missing = SW.Accepted(trace="gone", source="gone.jsonl", anchors=["s1"], ignore=[], prep=None)
    with pytest.raises(ValueError, match="gone"):
        SW.update_with_decisions(m, traces[0][1], ctx, source="run0.jsonl", spec={"accept_proposals": True},
                                 accepted=[missing])


# --------------------------------------------------------------------------- #
# 4. endpoint failures stop initialization
# --------------------------------------------------------------------------- #
class Refusing:
    def __init__(self):
        self.calls = 0

    def generate(self, *, prompt, values, history=()):
        self.calls += 1
        raise LLMHTTPError(401, "invalid key", "https://api.example/v1/chat/completions")


def test_initialize_stops_at_an_endpoint_failure():
    ctx = build_context("s", "# S\n## S1 do it", [("S1", "do it", "doc:2")], [], rules=parse_rules(RULES))
    model = Refusing()
    with pytest.raises(ModelUnavailable):
        initialize(ctx, model=model, rounds=3)
    assert model.calls == 1


def test_rule_extraction_does_not_continue_without_rules():
    ctx = build_context("s", "# S\n## S1 do it", [("S1", "do it", "doc:2")], [])
    with pytest.raises(ModelUnavailable):
        extract_rules(ctx, Refusing())


def test_format_errors_still_use_the_remaining_rounds():
    ctx = build_context("s", "# S\n## S1 do it", [("S1", "do it", "doc:2")], [], rules=parse_rules(RULES))

    class Garbage:
        calls = 0

        def generate(self, *, prompt, values, history=()):
            Garbage.calls += 1
            raise ValueError("the reply is not a JSON object")

    res = initialize(ctx, model=Garbage(), rounds=3)
    assert res.machine is None and Garbage.calls == 3


def test_compile_reports_an_endpoint_failure(tmp_path, monkeypatch, capsys):
    from contextlib import contextmanager

    from hexis.cli import _model

    @contextmanager
    def refusing(a, **kw):
        yield Refusing(), {"provider": "default", "model": "m", "base_url": "https://api.example/v1"}

    monkeypatch.setattr(_model, "open_model", refusing)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(tc.SKILL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (skill / "compile.json").write_text(json.dumps(RULES), encoding="utf-8")
    code = cli_main(["compile", "--skill", str(skill), "--out", str(tmp_path / "b"), "--backend", "none"])
    assert code == 2
    assert "model endpoint failed" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 5. JSON repair keeps long replies
# --------------------------------------------------------------------------- #
def _adapter(**kw):
    seen = []
    long_reply = "x" * 9000
    replies = [long_reply, '{"a": 1}']

    def handler(request):
        seen.append(json.loads(request.content.decode("utf-8")))
        content = replies[min(len(seen) - 1, 1)]
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    client = OpenAIClient("m", "https://api.example/v1", "sk-test-0000", transport=httpx.MockTransport(handler))
    return ModelAdapter(client, **kw), seen


def test_json_repair_truncates_by_default():
    adapter, seen = _adapter()
    assert adapter.generate(prompt="p", values={}) == {"a": 1}
    assert len(seen[1]["messages"][-2]["content"]) == 6000


def test_json_repair_can_send_the_whole_reply():
    adapter, seen = _adapter(repair_chars=None)
    assert adapter.generate(prompt="p", values={}) == {"a": 1}
    assert len(seen[1]["messages"][-2]["content"]) == 9000
