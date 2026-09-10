"""Native OpenCode bridge: result identity, real tools and runtime entry point."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from skill2fsm.opencode_tools import OpenCodeError, OpenCodeTools, tool_result


def test_result_preserves_native_error_and_exit_status():
    part = {"tool": "bash", "state": {"status": "completed", "input": {"command": "exit 2"},
            "output": "failed", "metadata": {"exit": 2}}}
    out = tool_result(part, "bash", {"command": "exit 2"})
    assert out["returncode"] == 2 and out["ok"] is False
    assert out["opencode_part"] == part
    bad = deepcopy(part)
    bad["state"]["status"] = "error"
    bad["state"]["error"] = "permission denied"
    assert tool_result(bad, "bash", {"command": "exit 2"})["stderr"] == "permission denied"
    with pytest.raises(OpenCodeError, match="different"):
        tool_result(part, "read", {"command": "exit 2"})
    with pytest.raises(OpenCodeError, match="different"):
        tool_result(part, "bash", {"command": "exit 0"})


@pytest.fixture
def native(tmp_path):
    if not shutil.which("opencode"):
        pytest.skip("OpenCode binary required for native integration tests")
    with OpenCodeTools(tmp_path, timeout_s=30) as tools:
        yield tools


def test_native_file_lifecycle_and_bash(native, tmp_path):
    target = str(tmp_path / "note.txt")
    calls = [
        ("write", {"filePath": target, "content": "before\n"}),
        ("read", {"filePath": target}),
        ("edit", {"filePath": target, "oldString": "before", "newString": "after", "replaceAll": False}),
        ("glob", {"pattern": "*.txt", "path": str(tmp_path)}),
        ("grep", {"pattern": "after", "path": str(tmp_path)}),
    ]
    for name, args in calls:
        before = deepcopy(args)
        out = native.call(name, args)
        assert out["ok"], out
        assert args == before
        assert out["opencode_part"]["state"]["input"] == args
    assert Path(target).read_text() == "after\n"
    out = native.call("bash", {"command": "pwd; exit 2", "description": "Check directory and exit code"})
    assert out["returncode"] == 2 and out["ok"] is False
    assert str(tmp_path.resolve()) in out["stdout"]
    with pytest.raises(OpenCodeError, match="Unsupported"):
        native.call("file_ops", {"op": "read", "path": target})
    if "list" not in native.available_tools:
        with pytest.raises(OpenCodeError, match="does not provide"):
            native.call("list", {"path": str(tmp_path)})


def test_native_read_failure_is_not_reported_as_success(native, tmp_path):
    out = native.call("read", {"filePath": str(tmp_path / "missing.txt")})
    assert not out["ok"] and out["stderr"]


def test_xlsx_entry_uses_native_tools_and_saves_trace(tmp_path, monkeypatch):
    if not shutil.which("opencode"):
        pytest.skip("OpenCode binary required")
    from skill2fsm.cli import run as run_fsm
    from skill2fsm.schema import Machine, State, ToolAction, EndAction, Transition, Variable, Terminal
    import json

    m = Machine(skill_id="native-smoke", initial="copy", fallback="FALLBACK", variables=[
        Variable(name="input_path", init_from="task.input.input_path"),
        Variable(name="output_path", init_from="task.input.output_path")],
        states={"copy": State(id="copy", action=ToolAction(name="bash", input={
            "command": 'cp "${input_path}" "${output_path}"', "description": "Copy input workbook"},
            writes=["stdout", "returncode"]), transitions=[Transition(to="end")]),
            "end": State(id="end", action=EndAction(terminal="done")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done"))},
        terminals=[Terminal(id="done")])
    machine = tmp_path / "machine.json"
    machine.write_text(m.model_dump_json(by_alias=True))
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"native execution smoke payload")
    (tmp_path / "SKILL.md").write_text("Copy the input file.")

    @contextmanager
    def no_inference(**_kwargs):
        yield SimpleNamespace(model="unused", base_url="local-test")

    monkeypatch.setattr(run_fsm, "client_from_env", no_inference)
    args = SimpleNamespace(machine=str(machine), fallback=False, skill=str(tmp_path), prompt="copy",
        workbook=str(source), model="", provider="default", max_steps=8, keep=str(tmp_path / "kept.xlsx"),
        golden=None, answer_position=None, quiet=True, json=str(tmp_path / "trace.jsonl"),
        opencode_bin="opencode", tool_timeout=30)
    assert run_fsm.run_machine(args) == 0
    assert Path(args.keep).read_bytes() == source.read_bytes()
    records = [json.loads(line) for line in Path(args.json).read_text().splitlines()]
    assert any(r.get("action", {}).get("name") == "bash" for r in records)


def test_timeout_aborts_native_tool_and_closes_backend(native, tmp_path):
    import time
    native.timeout_s = .5
    with pytest.raises(OpenCodeError, match="timed out"):
        native.call("bash", {"command": "sleep 2; touch late.txt", "description": "Timeout cleanup test"})
    assert native.process is None
    time.sleep(2)
    assert not (tmp_path / "late.txt").exists()
