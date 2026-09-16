"""回退态是重试枢纽，不是终结：回到最近工具步的入口重来，清零触发回退的计数；用完才解释执行或停机。"""
from __future__ import annotations

from hexis.execution import runtime
from hexis.machine.schema import Machine


def machine() -> Machine:
    return Machine.model_validate({
        "format": "efsm-v1", "skill_id": "t", "initial": "g", "fallback": "FALLBACK",
        "variables": [{"name": "x", "init_from": "task.input.x"}, {"name": "cmd"},
                      {"name": "returncode", "type": "integer", "init": 0}, {"name": "stdout"},
                      {"name": "n", "type": "integer", "init": 0}],
        "terminals": [{"id": "DONE", "kind": "done"}, {"id": "END_FALLBACK", "kind": "fallback"}],
        "states": {
            "g": {"id": "g", "action": {"kind": "model", "prompt": "cmd", "reads": ["x"], "writes": ["cmd"]},
                  "transitions": [{"if": "n >= 2", "to": "FALLBACK"}, {"if": "", "to": "t"}]},
            "t": {"id": "t", "action": {"kind": "tool", "name": "run", "input": {"c": "${cmd}"},
                                        "writes": ["returncode", "stdout"]},
                  "transitions": [{"if": "returncode == 0", "to": "DONE"}, {"if": "", "to": "g", "inc": "n"}]},
            "DONE": {"id": "DONE", "action": {"kind": "end", "terminal": "DONE"}},
            "FALLBACK": {"id": "FALLBACK", "action": {"kind": "end", "terminal": "END_FALLBACK"}},
        }})


class Model:
    def __init__(self, ok_after: int):
        self.calls = 0
        self.ok_after = ok_after

    def generate(self, *, prompt, values, history=()):
        self.calls += 1
        return {"cmd": f"try{self.calls}"}

    def classify(self, **kw):
        return "弃权"


class Tools:
    def __init__(self, ok_after: int):
        self.n = 0
        self.ok_after = ok_after

    def call(self, name, inp):
        self.n += 1
        ok = self.n >= self.ok_after
        return {"ok": ok, "returncode": 0 if ok else 1, "stdout": inp["c"]}


def test_retry_hub_resets_counter_and_reenters_at_gate():
    # 工具前 4 次都失败：机器自己的 n >= 2 出口进回退态两次，每次回到生成门重来，第 5 次成功
    m = machine()
    rr = runtime.run_task(m, {"input": {"x": 1}}, model=Model(0), tools=Tools(5), doc="", max_steps=60,
                          on_error="fallback", retries=3, interpret=False)
    assert rr.stopped == "terminal" and rr.retries == 2
    retries = [r for r in rr.trace.records if r.action.get("kind") == "retry"]
    assert [r.action["to"] for r in retries] == ["g", "g"]
    assert all("n" in r.action["reset"] for r in retries)
    assert rr.path()[-1] == "DONE"


def test_retries_exhausted_stops_without_interpret():
    m = machine()
    rr = runtime.run_task(m, {"input": {"x": 1}}, model=Model(0), tools=Tools(100), doc="", max_steps=80,
                          on_error="fallback", retries=2, interpret=False)
    assert rr.stopped == "fallback_exhausted" and rr.retries == 2
    assert rr.fallback_steps == 0


def test_default_keeps_old_fallback_semantics():
    # retries=0：回退态直接进解释段（旧行为不变）
    class Interp(Model):
        def generate(self, *, prompt, values, history=()):
            self.calls += 1
            if "cmd" in prompt:
                return {"cmd": "c"}
            return {"kind": "end", "terminal": "DONE"}
    m = machine()
    rr = runtime.run_task(m, {"input": {"x": 1}}, model=Interp(0), tools=Tools(100), doc="doc", max_steps=40,
                          on_error="fallback")
    assert rr.retries == 0 and rr.fallback_steps >= 1 and rr.stopped == "terminal"
