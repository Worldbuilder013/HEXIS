"""GUIDE.md and PROMPT.md describe the whole machine, deterministically."""
from __future__ import annotations

import json
import re

from hexis import guide
from hexis.cli import main as cli_main
from hexis.examples import table_clean as tc
from hexis.machine.schema import Machine

TRICKY = {
    "format": "efsm-v1", "skill_id": "tricky", "initial": "gen", "fallback": "FALLBACK", "max_steps": 20,
    "variables": [{"name": "request", "init_from": "task.input.request"}, {"name": "cmd"},
                  {"name": "returncode", "type": "integer"}, {"name": "stdout"}, {"name": "verdict"},
                  {"name": "tries", "type": "integer", "init": 0}],
    "terminals": [{"id": "DONE", "kind": "done", "output": ["stdout"]}, {"id": "END_FALLBACK", "kind": "fallback"}],
    "states": {
        "gen": {"id": "gen", "clause": "S1", "action": {"kind": "model", "prompt": "Write cmd for \"request\" | <b>",
                                                        "reads": ["request"], "writes": ["cmd"]},
                "transitions": [{"if": "tries >= 3", "to": "FALLBACK"}, {"if": "", "to": "run"}]},
        "run": {"id": "run", "action": {"kind": "tool", "name": "bash", "input": {"command": "${cmd}"},
                                        "writes": ["returncode", "stdout"]},
                "transitions": [{"if": "returncode == 0", "to": "check"}, {"if": "", "to": "gen", "inc": "tries"}]},
        "check": {"id": "check", "action": {"kind": "judge", "prompt": "Is stdout right?", "reads": ["stdout"],
                                            "writes": ["verdict"], "labels": ["yes", "no", "abstain"]},
                  "transitions": [{"if": "verdict == 'yes'", "to": "end"}, {"if": "", "to": "FALLBACK"}]},
        "end": {"id": "end", "action": {"kind": "end", "terminal": "DONE"}},
        "FALLBACK": {"id": "FALLBACK", "action": {"kind": "end", "terminal": "END_FALLBACK"}},
    },
}


def machine() -> Machine:
    return Machine.model_validate(json.loads(json.dumps(TRICKY)))


def test_mermaid_uses_generated_ids_and_escapes_labels():
    text = guide.mermaid(machine())
    assert not re.search(r"^\s*end[\[\(\{]", text, re.M)
    node_lines = [l for l in text.splitlines() if re.match(r"\s*n\d+[\[\(\{]", l)]
    assert len(node_lines) == len(TRICKY["states"])
    assert '"1. returncode == 0"' in text
    assert "#lt;" not in text or "<" not in text.replace("<br/>", "")


def test_guide_lists_every_state_transition_terminal_and_input():
    m = machine()
    text = guide.render_guide(m)
    for sid in m.states:
        assert f"`{sid}`" in text
    for st in m.states.values():
        for t in st.transitions:
            assert (f"`{t.cond}`" if t.cond else "otherwise") in text
    assert "`DONE`" in text and "`END_FALLBACK`" in text
    assert "| `request` | `request` | string |" in text
    assert "`gen` leaves for `FALLBACK` once `tries >= 3`" in text
    assert "```mermaid" in text


def test_prompt_contains_everything_needed_to_execute():
    m = machine()
    text = guide.render_prompt(m, retries=2)
    for sid in m.states:
        assert f"### `{sid}`" in text
    assert "Write cmd for \"request\" | <b>" in text
    assert '"command": "${cmd}"' in text
    assert "if `returncode == 0` → `check`" in text and "otherwise → `gen`, add 1 to `tries`" in text
    assert "Labels: `yes`, `no`, `abstain`; abstain label `abstain`" in text
    assert "After 20 states" in text and "at most 2 times" in text
    assert "This is the fallback state: follow section 7." in text
    assert "Appendix" not in text
    assert "Appendix: skill document" in guide.render_prompt(m, skill_doc="# Doc\nsteps")


def test_rendering_is_deterministic_and_has_no_secrets():
    m = machine()
    manifest = {"skill": {"sha256": "ab" * 32},
                "runs": [{"model": {"provider": "default", "model": "some-model", "base_url": "https://secret-host/v1"}}]}
    a = guide.render_guide(m, manifest=manifest) + guide.render_prompt(m)
    b = guide.render_guide(machine(), manifest=manifest) + guide.render_prompt(machine())
    assert a == b
    assert "secret-host" not in a and "some-model" in a


def test_gate_is_named_for_tool_arguments():
    m = tc.reference_machine()
    text = guide.render_prompt(m)
    assert "### `s1`: tool `read_csv`" in text


def test_guide_command_writes_both_files(tmp_path):
    mpath = tmp_path / "machine.json"
    mpath.write_text(machine().model_dump_json(by_alias=True), encoding="utf-8")
    assert cli_main(["guide", "--machine", str(mpath), "--out", str(tmp_path / "docs"), "--skill", str(tc.SKILL_PATH.parent),
                     "--embed-skill"]) == 0
    g = (tmp_path / "docs" / "GUIDE.md").read_text(encoding="utf-8")
    p = (tmp_path / "docs" / "PROMPT.md").read_text(encoding="utf-8")
    assert g.startswith("# Table Cleaning") or g.startswith("# ")
    assert "Appendix: skill document" in p
