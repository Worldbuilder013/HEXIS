"""The abstain label is ``abstain``; machines written with the earlier label ``弃权`` keep working unchanged."""
from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from hexis.compiler import common
from hexis.execution import runtime
from hexis.llm.llm_client import ModelAdapter, OpenAIClient
from hexis.llm.model_iface import ScriptedModel, ToolRegistry, _abstain
from hexis.machine import checks
from hexis.machine.schema import ABSTAIN, LEGACY_ABSTAIN, JudgeAction, Machine, abstain_label

LEGACY_MACHINE = {
    "format": "efsm-v1", "skill_id": "legacy", "initial": "j", "fallback": "FALLBACK",
    "variables": [{"name": "x", "init_from": "task.input.x"}, {"name": "verdict"}],
    "terminals": [{"id": "done", "kind": "done"}, {"id": "END_FALLBACK", "kind": "fallback"}],
    "states": {
        "j": {"id": "j", "action": {"kind": "judge", "prompt": "Is x fine?", "reads": ["x"], "writes": ["verdict"],
                                     "labels": ["yes", "no", "弃权"]},
              "transitions": [{"if": "verdict == 'yes'", "to": "done"}, {"if": "verdict == '弃权'", "to": "FALLBACK"},
                              {"if": "", "to": "FALLBACK"}]},
        "done": {"id": "done", "action": {"kind": "end", "terminal": "done"}},
        "FALLBACK": {"id": "FALLBACK", "action": {"kind": "end", "terminal": "END_FALLBACK"}},
    },
}


def test_constants():
    assert ABSTAIN == "abstain" and LEGACY_ABSTAIN == "弃权"
    assert common.ABSTAIN == ABSTAIN


@pytest.mark.parametrize("labels, expected", [
    (["a", "b", "abstain"], "abstain"),
    (["a", "b", "弃权"], "弃权"),
    (["弃权", "a", "abstain"], "abstain"),
    (["abstain", "a", "弃权"], "弃权"),
])
def test_missing_abstain_is_taken_from_the_labels(labels, expected):
    assert abstain_label(labels) == expected
    assert JudgeAction(prompt="q", reads=["x"], writes=["v"], labels=labels).abstain == expected


def test_explicit_abstain_is_kept_and_must_be_a_label():
    assert JudgeAction(prompt="q", reads=["x"], writes=["v"], labels=["a", "none"], abstain="none").abstain == "none"
    with pytest.raises(ValidationError):
        JudgeAction(prompt="q", reads=["x"], writes=["v"], labels=["a", "b"])


def test_legacy_machine_validates_round_trips_and_checks_clean():
    m = Machine.model_validate(json.loads(json.dumps(LEGACY_MACHINE)))
    assert m.states["j"].action.abstain == "弃权"
    again = Machine.model_validate(json.loads(m.model_dump_json(by_alias=True)))
    assert again.model_dump() == m.model_dump()
    assert checks.structural_findings(m) == []


def test_legacy_machine_routes_the_old_abstain_label():
    m = Machine.model_validate(json.loads(json.dumps(LEGACY_MACHINE)))
    model = ScriptedModel(judge=lambda prompt, values: "弃权")
    res = runtime.run_task(m, {"task_id": "t", "input": {"x": "1"}}, model=model, tools=ToolRegistry())
    assert res.path() == ["j", "FALLBACK"]


def test_abstain_lookup_keeps_its_order():
    assert _abstain(["yes", "弃权"]) == "弃权"
    assert _abstain(["yes", "abstain"]) == "abstain"
    assert _abstain(["弃权", "abstain"]) == "弃权"
    assert _abstain(["yes", "unknown"]) == "unknown"
    assert _abstain(["yes", "no"]) == ""


@pytest.mark.parametrize("labels, expected", [(["clean", "dirty", "弃权"], "弃权"), (["clean", "dirty", "abstain"], "abstain")])
def test_classify_falls_back_to_either_abstain_label(labels, expected):
    body = {"choices": [{"message": {"content": "purple"}, "finish_reason": "stop"}], "usage": {}}
    client = OpenAIClient("m", "https://api.example/v1", "sk-test-0000",
                          transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    assert ModelAdapter(client).classify(prompt="q", values={}, labels=labels) == expected
