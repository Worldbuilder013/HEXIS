"""Build directories: ``compile`` writes one, ``update`` folds new traces into it with a decider."""
from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from hexis import updater as U
from hexis.builddir import BuildDir
from hexis.cli import _model
from hexis.cli import main as cli_main
from hexis.compiler.context import build_context
from hexis.compiler.decide import decide_trace
from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.llm.model_iface import ModelUnavailable
from hexis.machine.schema import Transition, Variable
from hexis.skill_loader import load_agent_skill, markdown_clauses
from hexis.step_judge import DecisionLog, ModelDecider
from hexis.traces import judge

TOOLS = {"tools": {
    "read_csv": {"description": "read a CSV file", "input_schema": {"path": {"type": "string", "required": True}},
                 "output_schema": {"ok": "boolean", "header_row": "string", "rows": "array"}, "success": "ok == True",
                 "primary": "header_row"},
    "fix_header": {"description": "fix one header field", "input_schema": {"header_row": {"type": "string", "required": True}},
                   "output_schema": {"ok": "boolean", "header_row": "string"}, "success": "ok == True",
                   "primary": "header_row"},
    "export": {"description": "write the table to a file",
               "input_schema": {"header_row": {"type": "string"}, "rows": {"type": "array"},
                                "output_path": {"type": "string"}, "source_path": {"type": "string"}},
               "output_schema": {"ok": "boolean", "output_path": "string"}, "success": "ok == True",
               "primary": "output_path", "label": "apply"}}}


def traces(n, seed, error_rate=0.0):
    ref = tc.reference_machine()
    out = []
    for task in tc.gen_tasks(n, seed=seed):
        res = runtime.run_task(ref, task, model=tc.build_model(error_rate=error_rate, seed=seed),
                               tools=tc.build_registry(tc.MemFS(task["files"])), doc=tc.skill_doc())
        out.append(judge.judged(res.trace, lambda t, task=task: tc.verify(task, t), ref.prohibitions))
    return out


def skeleton():
    """read_csv → export → end: the updates have to add the header check and the repair loop."""
    m = tc.reference_machine()
    del m.states["s2"], m.states["s3"]
    m.states["s1"].transitions = [Transition(cond="ok == True", to="s4"), Transition(to="FALLBACK")]
    m.states["s1"].action.writes = ["ok", "header_row", "rows"]
    m.variables = [v for v in m.variables if v.name not in ("header_ok", "fix_count")] + [Variable(name="ok", type="boolean")]
    return m


@pytest.fixture
def inputs(tmp_path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(tc.SKILL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "tools.json").write_text(json.dumps(TOOLS), encoding="utf-8")
    (tmp_path / "m0.json").write_text(skeleton().model_dump_json(by_alias=True), encoding="utf-8")
    for name, batch in (("batch1", traces(6, seed=1)), ("batch2", traces(6, seed=2))):
        d = tmp_path / name
        d.mkdir()
        for i, t in enumerate(batch):
            (d / f"{name}_{i:02d}.jsonl").write_text(t.to_jsonl(), encoding="utf-8")
    return tmp_path


class Decider:
    """A scripted model for step decisions: match the first reachable candidate, otherwise add a state."""

    def __init__(self, *, fail_after=None, invalid=False):
        self.calls = 0
        self.fail_after = fail_after
        self.invalid = invalid

    def generate(self, *, prompt, values, history=()):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ModelUnavailable("endpoint went away")
        if self.invalid:
            return {"decision": "maybe"}
        reachable = [c for c in values["candidates"] if c["reachable"]]
        if reachable:
            return {"decision": "match", "state": reachable[0]["id"], "purpose": "", "clause": ""}
        return {"decision": "new", "purpose": f"run {values['step'].get('tool', 'the step')}", "clause": "S3"}

    def usage(self):
        return {"llm_calls": self.calls, "prompt_tokens": 10 * self.calls, "completion_tokens": self.calls}


def use_model(monkeypatch, model):
    @contextmanager
    def fake(a, **kw):
        yield model, {"provider": "default", "model": "scripted", "base_url": "https://api.example/v1"}
    monkeypatch.setattr(_model, "open_model", fake)


def no_model(monkeypatch):
    @contextmanager
    def fake(a, **kw):
        raise AssertionError("no model should be opened")
        yield
    monkeypatch.setattr(_model, "open_model", fake)


def compile_build(d, *extra):
    return cli_main(["compile", "--skill", str(d / "skill"), "--traces", str(d / "batch1"), "--out", str(d / "build"),
                     "--tools", str(d / "tools.json"), "--machine-init", str(d / "m0.json"), *extra])


def update(d, *extra):
    return cli_main(["update", "--build", str(d / "build"), *extra])


def protected_replay_errors(build):
    bd = BuildDir.open(build)
    progress = bd.load_progress()
    stored = bd.stored_traces(progress)
    skill = load_agent_skill(bd.skill_dir)
    ctx = build_context(skill.slug, skill.body, markdown_clauses(skill.body), [i.trace for i in stored.values()],
                        registry=bd.registry(), rules=bd.rules())
    accepted = U.rebuild_accepted(progress, stored, ctx)
    errors, _warnings = U.baseline_check(bd.load_machine(), ctx, accepted)
    return errors, len(accepted)


# --------------------------------------------------------------------------- #
def test_compile_writes_a_build_directory(inputs):
    assert compile_build(inputs) == 0
    b = inputs / "build"
    for name in ("build.json", "skill/SKILL.md", "tools.json", "rules.json", "progress.json", "machine.json",
                 "machine_init.json", "update_log.json", "report.md", "GUIDE.md", "PROMPT.md"):
        assert (b / name).exists(), name
    manifest = json.loads((b / "build.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "hexis-build/1" and manifest["runs"][0]["status"] == "ok"
    progress = json.loads((b / "progress.json").read_text(encoding="utf-8"))
    assert len(progress["traces"]) == 6 and len(progress["entries"]) == 6
    assert len(list((b / "traces").glob("*.jsonl"))) == 6
    assert progress["accepted"]
    assert protected_replay_errors(b)[0] == []


def test_update_with_a_model_decides_steps_and_protects_earlier_traces(inputs, monkeypatch):
    assert compile_build(inputs) == 0
    accepted_before = json.loads((inputs / "build" / "progress.json").read_text(encoding="utf-8"))["accepted"]
    model = Decider()
    use_model(monkeypatch, model)
    assert update(inputs, "--traces", str(inputs / "batch2")) == 0
    b = inputs / "build"
    progress = json.loads((b / "progress.json").read_text(encoding="utf-8"))
    assert len(progress["entries"]) == 12
    assert {a["trace"] for a in accepted_before} <= {a["trace"] for a in progress["accepted"]}
    assert model.calls > 0
    log = [json.loads(line) for line in (b / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == model.calls and all(r["model"] == "scripted" for r in log)
    assert "sk-" not in (b / "decisions.jsonl").read_text(encoding="utf-8")
    errors, n_accepted = protected_replay_errors(b)
    assert errors == [] and n_accepted == len(progress["accepted"])
    manifest = json.loads((b / "build.json").read_text(encoding="utf-8"))
    assert manifest["runs"][-1]["model"]["model"] == "scripted"
    assert manifest["runs"][-1]["questions"]["asked"] == model.calls

    digest = progress["machine_sha256"]
    again = Decider()
    use_model(monkeypatch, again)
    assert update(inputs, "--traces", str(inputs / "batch2")) == 0
    progress2 = json.loads((b / "progress.json").read_text(encoding="utf-8"))
    assert again.calls == 0 and len(progress2["entries"]) == 12 and progress2["machine_sha256"] == digest


def test_update_resumes_after_an_endpoint_failure(inputs, monkeypatch):
    assert compile_build(inputs) == 0
    use_model(monkeypatch, Decider(fail_after=3))
    assert update(inputs, "--traces", str(inputs / "batch2")) == 3
    b = inputs / "build"
    manifest = json.loads((b / "build.json").read_text(encoding="utf-8"))
    assert manifest["runs"][-1]["status"] == "interrupted"
    model = Decider()
    use_model(monkeypatch, model)
    assert update(inputs) == 0
    progress = json.loads((b / "progress.json").read_text(encoding="utf-8"))
    assert len(progress["entries"]) == 12
    manifest = json.loads((b / "build.json").read_text(encoding="utf-8"))
    assert manifest["runs"][-1]["questions"]["cached"] >= 1
    assert protected_replay_errors(b)[0] == []


def test_show_previews_without_model_calls_or_writes(inputs, monkeypatch, capsys):
    assert compile_build(inputs) == 0
    b = inputs / "build"
    before = {p: p.read_bytes() for p in b.rglob("*") if p.is_file()}
    no_model(monkeypatch)
    capsys.readouterr()
    assert update(inputs, "--traces", str(inputs / "batch2"), "--show", "2") == 0
    assert "pending traces" in capsys.readouterr().out
    after = {p: p.read_bytes() for p in b.rglob("*") if p.is_file()}
    assert after == before


def test_update_needs_every_accepted_trace(inputs, monkeypatch):
    assert compile_build(inputs) == 0
    b = inputs / "build"
    progress = json.loads((b / "progress.json").read_text(encoding="utf-8"))
    rec = next(r for r in progress["traces"] if r["key"] == progress["accepted"][0]["trace"])
    (b / rec["file"]).unlink()
    no_model(monkeypatch)
    assert update(inputs, "--decider", "align") == 2


def test_update_refuses_a_machine_that_no_longer_replays(inputs, monkeypatch):
    assert compile_build(inputs) == 0
    b = inputs / "build"
    machine = json.loads((b / "machine.json").read_text(encoding="utf-8"))
    machine["states"]["s4"]["action"]["name"] = "read_csv"
    (b / "machine.json").write_text(json.dumps(machine), encoding="utf-8")
    no_model(monkeypatch)
    assert update(inputs, "--traces", str(inputs / "batch2"), "--decider", "align") == 2
    assert json.loads((b / "machine.json").read_text(encoding="utf-8")) == machine


def test_align_and_file_deciders(inputs, monkeypatch, tmp_path):
    assert compile_build(inputs) == 0
    no_model(monkeypatch)
    assert update(inputs, "--traces", str(inputs / "batch2"), "--decider", "align", "--max-traces", "2") == 0
    b = inputs / "build"
    progress = json.loads((b / "progress.json").read_text(encoding="utf-8"))
    assert len(progress["entries"]) == 8
    pending = [r["key"] for r in progress["traces"] if r["key"] not in {e["trace"] for e in progress["entries"]}]
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({k: {"accept_proposals": True} for k in pending}), encoding="utf-8")
    assert update(inputs, "--decisions", str(decisions), "--continue-after-insert") == 0
    progress = json.loads((b / "progress.json").read_text(encoding="utf-8"))
    assert len(progress["entries"]) == 12
    assert {e["decider"] for e in progress["entries"]} == {"align", "file"}
    assert protected_replay_errors(b)[0] == []


def test_invalid_answers_fall_back_to_the_proposal_and_are_cached(inputs, tmp_path):
    skill = load_agent_skill(inputs / "skill")
    batch = traces(3, seed=2)
    ctx = build_context(skill.slug, skill.body, markdown_clauses(skill.body), batch, rules=None)
    machine = skeleton()
    log = DecisionLog(tmp_path / "decisions.jsonl")
    bad = Decider(invalid=True)
    decider = ModelDecider(bad, model_id="scripted", log=log)
    spec = decider.spec_for(machine, ctx, batch[0], key="t0")
    assert bad.calls == 2 * decider.stats["asked"] and decider.stats["fallback"] == decider.stats["asked"]
    records = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records and all(r["source"] == "fallback" for r in records)
    again = Decider(invalid=True)
    decider2 = ModelDecider(again, model_id="scripted", log=DecisionLog(tmp_path / "decisions.jsonl"))
    assert decider2.spec_for(machine, ctx, batch[0], key="t0") == spec and again.calls == 0


def test_decide_trace_restarts_after_an_ignored_step_and_honours_exclude(inputs):
    skill = load_agent_skill(inputs / "skill")
    batch = traces(4, seed=5)
    ctx = build_context(skill.slug, skill.body, markdown_clauses(skill.body), batch, rules=None)
    machine = skeleton()
    trace = max(batch, key=lambda t: len(t.records))
    asked: list = []

    def ignore_first(view):
        asked.append(view.step)
        if view.step == 0 and not view.ignored:
            return {"decision": "ignore", "state": None, "purpose": "", "clause": ""}
        if view.tier1:
            return {"decision": "match", "state": view.tier1[0], "purpose": "", "clause": ""}
        return {"decision": "new", "state": None, "purpose": "step", "clause": ""}

    spec = decide_trace(machine, ctx, trace, ask=ignore_first, key="t")
    assert spec["steps"]["0"] == {"d": "ignore"}
    assert asked[0] == 0 and 0 not in asked[1:]
    excluded = decide_trace(machine, ctx, trace, key="t",
                            ask=lambda view: {"decision": "exclude", "state": None, "purpose": "not the skill", "clause": ""})
    assert excluded == {"exclude": "not the skill"}
