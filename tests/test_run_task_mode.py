"""``run --mode task`` runs any machine on inputs given on the command line, in a kept working directory."""
from __future__ import annotations

import json
import shutil

import httpx
import pytest

from hexis.cli import main as cli_main
from hexis.llm.llm_client import OpenAIClient

MACHINE = {
    "format": "efsm-v1", "skill_id": "echo", "initial": "gen", "fallback": "FALLBACK", "max_steps": 10,
    "variables": [{"name": "text", "init_from": "task.input.text"}, {"name": "cmd"},
                  {"name": "returncode", "type": "integer"}, {"name": "stdout"}],
    "terminals": [{"id": "DONE", "kind": "done"}, {"id": "END_FALLBACK", "kind": "fallback"}],
    "states": {
        "gen": {"id": "gen", "action": {"kind": "model", "prompt": "Write cmd that saves text to out.txt.",
                                        "reads": ["text"], "writes": ["cmd"]},
                "transitions": [{"if": "", "to": "run"}]},
        "run": {"id": "run", "action": {"kind": "tool", "name": "bash", "input": {"command": "${cmd}"},
                                        "writes": ["returncode", "stdout"]},
                "transitions": [{"if": "returncode == 0", "to": "done"}, {"if": "", "to": "FALLBACK"}]},
        "done": {"id": "done", "action": {"kind": "end", "terminal": "DONE"}},
        "FALLBACK": {"id": "FALLBACK", "action": {"kind": "end", "terminal": "END_FALLBACK"}},
    },
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / "machine.json").write_text(json.dumps(MACHINE), encoding="utf-8")
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: echo\ndescription: save text\n---\n# Echo\n\n## S1 Save\nSave the text.\n",
                                    encoding="utf-8")
    reply = {"choices": [{"message": {"content": json.dumps({"cmd": "printf hello > out.txt"})}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 5}}

    def fake_client_from_env(**kw):
        return OpenAIClient("scripted", "https://api.example/v1", "sk-test-0000",
                            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply)))

    import hexis.cli.run as run_cli
    monkeypatch.setattr(run_cli, "client_from_env", fake_client_from_env)
    return tmp_path


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_task_mode_runs_in_the_working_directory(setup, capsys):
    work = setup / "work"
    code = cli_main(["run", "--mode", "task", "--machine", str(setup / "machine.json"), "--skill", str(setup / "skill"),
                     "--input", "text=hello", "--workdir", str(work), "--executor", "local",
                     "--json", str(setup / "trace.jsonl"), "--quiet"])
    assert code == 0
    assert (work / "out.txt").read_text(encoding="utf-8") == "hello"
    assert (setup / "trace.jsonl").is_file()
    assert "stopped: terminal" in capsys.readouterr().out


def test_task_mode_requires_the_declared_inputs(setup):
    with pytest.raises(SystemExit) as exc:
        cli_main(["run", "--mode", "task", "--machine", str(setup / "machine.json"), "--skill", str(setup / "skill"),
                  "--executor", "local"])
    assert exc.value.code == 2
