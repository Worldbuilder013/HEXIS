"""㉖ 解释器的执行侧账：每步的 meta、model 动作记什么、用量测不到就留空、回退的位置与步数。

这套测试全部密闭：模型是 :class:`~hexis.llm.model_iface.ScriptedModel`（查表、无网络），工具
是进程内纯函数。它盯的是四件**报告要读、但语义上无关紧要**的事——正因为无关紧要，写错了不会
让任何验收失败，只会让实验报告里的数字悄悄失真，所以得单独钉住：

1. 每步都有 ``Record.meta``：耗时、这步调了几次模型、花了多少 token；
2. ``model`` 动作记模板 id / 模板 / reads / 提示词摘要，**不记渲染后的全文**（题面不重复存）；
3. token 字段**测不到就是 ``None``**，不是 0、更不是按单价估的数；
4. 回退段跑了几步、从哪个状态退进去的（交付物指标 5「回退率与位置清单」）。

顺带跨检既有停机原因没被改动（与 test_01 / test_03 / test_11 的期望对齐），以及
:mod:`hexis.execution.interpreter` 这个别名模块确实只是别名。
"""

import json

import pytest

from hexis.execution import interpreter, runtime
from hexis.examples import table_clean as tc
from hexis.llm.model_iface import ScriptedModel, ToolRegistry
from hexis.machine.schema import (
    EndAction, FALLBACK, Machine, ModelAction, State, Terminal, ToolAction,
    Transition, Variable, empty_machine,
)

#: 题面：它该出现在 ``Record.vars`` 里（那是变量表），但**绝不该**出现在 model 动作的记录里。
PROBLEM = "求 2^10 的十进制各位数字之和，答案写成整数"
#: 状态 s2 的私有模板。它不含题面——题面是变量，运行时才拼进去。
TEMPLATE = "按文档 S3 求解：读 problem，只回一个 JSON 对象 {\"answer\": ...}"
#: FALLBACK 解释读的「文档」。够长，好验证它没被逐步抄进轨迹。
DOC = "# 玩具技能文档\n" + "解释段读的就是这份东西，逐步给出下一个动作。\n" * 40
#: 工具真正执行的命令行（照 sandbox.ExecResult 的 ``command`` 字段记）。
ARGV = ["python", "scripts/math_verify.py", "--json", "equiv", "1024", "1024"]


# --------------------------------------------------------------------------- #
# 手搭的小机器与桩
# --------------------------------------------------------------------------- #
def _tools() -> ToolRegistry:
    """一个工具：报 ``command``（真跑过的 argv），像 sandbox 那样。"""
    reg = ToolRegistry()
    reg.add("math_verify", lambda inp: {"ok": True, "verified": True,
                                        "command": list(ARGV)})
    return reg


def _task(problem: str = PROBLEM) -> dict:
    return {"task_id": "m1", "input": {"problem": problem}}


def _machine(*, after_s1: str = "s2") -> Machine:
    """读→（核验）→生成→结束。``after_s1`` 指到 ``FALLBACK`` 就是「s1 之后退下来」。"""
    return Machine(
        skill_id="t26", initial="s1", max_steps=8,
        variables=[Variable(name="problem", init_from="task.input.problem"),
                   Variable(name="answer")],
        states={
            "s1": State(id="s1", clause="S1",
                        action=ToolAction(name="math_verify",
                                          input={"expr": "${problem}"},
                                          reads=["problem"], writes=["verified"]),
                        transitions=[Transition(to=after_s1)]),
            "s2": State(id="s2", clause="S3",
                        action=ModelAction(prompt=TEMPLATE, reads=["problem"],
                                           writes=["answer"]),
                        transitions=[Transition(to="end")]),
            "end": State(id="end", action=EndAction(terminal="done")),
            FALLBACK: State(id=FALLBACK, action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done", output=["answer"])],
    )


def _gen(prompt: str, values: dict, history: tuple = ()) -> dict:
    """两种提示词各一套脚本：s2 的模板出答案，DOC 出「下一个动作」。"""
    if prompt == TEMPLATE:
        return {"answer": "7"}
    if "draft" not in values:                       # 解释段第一步：先写个草稿（model 动作）
        return {"kind": "model", "writes": ["draft"],
                "output": {"draft": "2^10 = 1024"}}
    return {"kind": "end", "terminal": "done", "answer": "7"}   # 第二步：交卷


def _model() -> ScriptedModel:
    return ScriptedModel(gen=_gen)


class _MeteredModel(ScriptedModel):
    """带 ``usage()`` 的桩，形状照 :class:`~hexis.llm.llm_client.ModelAdapter` 的累计计数器。

    ``unmeasured=True`` 模拟「端点根本没报 usage」：调用数照涨，token 停在 0——运行器必须把这种
    情况报成 ``None``，而不是照抄那个 0。
    """

    def __init__(self, *, per_call=(11, 7), unmeasured: bool = False, **kw):
        super().__init__(**kw)
        self._per_call = per_call
        self._unmeasured = bool(unmeasured)
        self.n = self.p = self.c = self.unm = 0

    def _tick(self) -> None:
        self.n += 1
        if self._unmeasured:
            self.unm += 1
            return
        self.p += self._per_call[0]
        self.c += self._per_call[1]

    def generate(self, **kw) -> dict:
        self._tick()
        return super().generate(**kw)

    def classify(self, **kw) -> str:
        self._tick()
        return super().classify(**kw)

    def usage(self) -> dict:
        return {"llm_calls": self.n, "prompt_tokens": self.p,
                "completion_tokens": self.c, "unmeasured_calls": self.unm}


def _run(machine=None, *, model=None, doc: str = DOC, max_steps=None,
         tools=None, task=None):
    return runtime.run_task(machine or _machine(), task or _task(),
                            model=model or _model(), tools=tools or _tools(),
                            doc=doc, max_steps=max_steps)


def _model_record(res):
    return next(r for r in res.trace.records if r.action.get("kind") == "model")


# --------------------------------------------------------------------------- #
# ① 每步都有 meta
# --------------------------------------------------------------------------- #
def test_every_step_records_meta_with_tokens_and_ms():
    """耗时与 token 两个键**每步都在**——报告要能分清「这步没花」与「这步没测到」。"""
    res = _run()
    assert res.stopped == runtime.STOP_TERMINAL, res.error
    assert res.path() == ["s1", "s2", "end"]
    for rec in res.trace.records:
        assert isinstance(rec.meta, dict) and rec.meta, rec.step
        assert isinstance(rec.meta["ms"], float) and rec.meta["ms"] >= 0.0
        assert isinstance(rec.meta["llm_calls"], int)
        assert "prompt_tokens" in rec.meta and "completion_tokens" in rec.meta


def test_meta_stays_out_of_action_and_output():
    """账不进 action/output：它不参与规范化、不参与评判，也就不该改变状态身份。"""
    res = _run()
    for rec in res.trace.records:
        assert "ms" not in rec.action and "ms" not in rec.output
        assert "llm_calls" not in rec.action and "llm_calls" not in rec.output


def test_tool_step_meta_carries_the_argv_actually_executed():
    """工具步记**真跑过的** argv，而不是 input 模板反推出来的命令行。"""
    rec = _run().trace.records[0]
    assert rec.action["kind"] == "tool"
    assert rec.meta["argv"] == ARGV
    # ``--json`` 在子命令之前——这正是「模板拼对了没有」与「实际怎么跑的」之间要能对账的地方
    assert rec.meta["argv"].index("--json") < rec.meta["argv"].index("equiv")


def test_steps_without_a_model_call_report_zero_not_none():
    """没调模型的一步，token 是 0：这是测量结论，不是「不知道」。"""
    tool_step, end_step = _run().trace.records[0], _run().trace.records[-1]
    for rec in (tool_step, end_step):
        assert rec.meta["llm_calls"] == 0
        assert rec.meta["prompt_tokens"] == 0 and rec.meta["completion_tokens"] == 0


# --------------------------------------------------------------------------- #
# ② model 动作记什么
# --------------------------------------------------------------------------- #
def test_model_record_carries_template_id_reads_and_digest():
    rec = _model_record(_run())
    assert rec.state == "s2" and rec.action["template_id"] == "s2"
    assert rec.action["prompt"] == TEMPLATE                  # 模板文本 = 状态身份
    assert rec.action["reads"] == ["problem"]                # 这一步真正消费的变量
    digest = rec.action["prompt_sha256"]
    assert len(digest) == 64 and int(digest, 16) >= 0        # 是个 sha256 十六进制串
    assert digest == runtime.prompt_digest(TEMPLATE, {"problem": PROBLEM})


def test_model_record_does_not_carry_the_rendered_prompt():
    """渲染后的提示词里嵌着整道题面。题面在 ``vars`` 里存**一次**就够，动作里不再存第二遍。"""
    rec = _model_record(_run())
    assert PROBLEM not in json.dumps(rec.action, ensure_ascii=False)
    assert rec.vars["problem"] == PROBLEM                    # 它合法的落脚点只有变量表
    assert rec.output == {"answer": "7"}                     # 产出照收，按 writes 白名单


def test_prompt_digest_is_stable_and_value_sensitive():
    """同模板同取值 ⇒ 同摘要；换了题面就换摘要——「这一步问了什么」因此可核对。"""
    a = runtime.prompt_digest(TEMPLATE, {"problem": PROBLEM})
    assert a == runtime.prompt_digest(TEMPLATE, {"problem": PROBLEM})
    assert a != runtime.prompt_digest(TEMPLATE, {"problem": "另一道题"})
    assert a != runtime.prompt_digest(TEMPLATE + "！", {"problem": PROBLEM})
    other = _run(task=_task("另一道题"))
    assert _model_record(other).action["prompt_sha256"] != a


# --------------------------------------------------------------------------- #
# ③ 用量：测不到就留 None
# --------------------------------------------------------------------------- #
def test_token_fields_are_none_when_the_model_reports_no_usage():
    """``ScriptedModel`` 没有 ``usage()``——那就是**不知道**，不是 0。"""
    assert not hasattr(_model(), "usage")                    # 鸭子类型的另一半确实缺席
    res = _run()
    assert res.llm_calls == 1                                # 调用数是自己数的，测得到
    assert res.prompt_tokens is None and res.completion_tokens is None
    assert res.unmeasured_calls == 1
    assert res.wall_s >= 0.0                                 # 墙钟永远测得到，是数不是 None
    meta = _model_record(res).meta
    assert meta["llm_calls"] == 1
    assert meta["prompt_tokens"] is None and meta["completion_tokens"] is None


def test_token_fields_are_read_from_the_model_when_it_reports_usage():
    """模型有 ``usage()`` 就用它的实测数——鸭子类型，不 import llm_client、不改接口。"""
    model = _MeteredModel(gen=_gen, per_call=(11, 7))
    res = _run(model=model)
    assert res.llm_calls == 1
    assert (res.prompt_tokens, res.completion_tokens) == (11, 7)
    assert res.unmeasured_calls == 0
    meta = _model_record(res).meta
    assert (meta["prompt_tokens"], meta["completion_tokens"]) == (11, 7)
    assert "unmeasured_calls" not in meta                    # 全测到了就不写这一栏


def test_unreported_usage_never_becomes_zero():
    """端点没报 usage 时累加出来的那个 0 不许冒充实测：整趟与单步都报 ``None``。"""
    model = _MeteredModel(gen=_gen, unmeasured=True)
    res = _run(model=model)
    assert model.usage()["prompt_tokens"] == 0               # 桩这边确实是 0
    assert res.prompt_tokens is None and res.completion_tokens is None
    assert res.unmeasured_calls == 1 == res.llm_calls
    assert _model_record(res).meta["unmeasured_calls"] == 1


def test_a_run_without_any_model_call_spends_zero_tokens():
    """一次模型都没调 ⇒ 0（这是测量）；与「没测到」区分开。"""
    m = _machine()
    m.states["s1"].transitions = [Transition(to="end")]      # 跳过生成那步
    res = _run(m)
    assert res.llm_calls == 0
    assert (res.prompt_tokens, res.completion_tokens) == (0, 0)
    assert res.unmeasured_calls == 0


# --------------------------------------------------------------------------- #
# ④ 回退：跑了几步、从哪退的
# --------------------------------------------------------------------------- #
def test_fallback_entry_state_and_step_count_are_recorded():
    """指标 5「回退率与位置清单」要的两样：从哪个状态退的、解释段跑了几步。"""
    res = _run(_machine(after_s1=FALLBACK))
    assert res.stopped == runtime.STOP_TERMINAL, res.error
    assert res.entered_fallback()
    assert res.fallback_entry == "s1"                        # 位置：从 s1 退下来
    assert res.fallback_entry_step == 2                      # 解释段第一条记录的步号
    assert res.fallback_steps == 2 == sum(1 for r in res.trace.records
                                          if r.state == FALLBACK)
    assert res.path() == ["s1", FALLBACK, FALLBACK]


def test_fallback_runs_a_model_action_and_submits_a_final_answer():
    """解释段能跑生成、能交卷——不然回退只是「优雅地放弃」。"""
    res = _run(_machine(after_s1=FALLBACK))
    steps = res.trace.records
    assert steps[1].action["kind"] == "model"
    assert steps[1].output == {"draft": "2^10 = 1024"}       # 产出按它声明的 writes 收
    assert res.values["draft"] == "2^10 = 1024"              # 也落进了变量表
    assert steps[2].action == {"kind": "end", "terminal": "done"}
    assert steps[2].output == {"answer": "7"}                # 终止那步带着最终答案
    assert res.values["answer"] == "7"                       # 评判据此打分


def test_fallback_history_does_not_carry_the_bookkeeping():
    """喂回给解释的历史里**没有 meta**：账是宿主的事，占不得模型那点上下文预算。

    ``ModelAdapter`` 把每条历史压到 800 字符，让 token/耗时/argv 去挤那点额度，等于拿宿主的
    实现细节换掉真正要看的动作与结果。
    """
    seen = []

    def gen(prompt, values, history):
        seen.append(history)
        return _gen(prompt, values, history)

    res = _run(_machine(after_s1=FALLBACK), model=ScriptedModel(gen=gen))
    assert res.stopped == runtime.STOP_TERMINAL, res.error
    last = seen[-1]
    assert last and all("meta" not in h for h in last)
    assert all("action" in h and "output" in h for h in last)   # 该看的都还在


def test_fallback_model_record_hides_the_document_but_keeps_the_digest():
    """FALLBACK 的「模板」是整份文档：只记 id 与摘要，逐步内联比记渲染全文还糟。"""
    res = _run(_machine(after_s1=FALLBACK))
    rec = res.trace.records[1]
    text = json.dumps(rec.action, ensure_ascii=False)
    assert rec.action["template_id"] == FALLBACK and "prompt" not in rec.action
    assert "解释段读的就是这份东西" not in text and len(text) < len(DOC)
    assert rec.action["prompt_sha256"] == runtime.prompt_digest(DOC,
                                                                res.trace.records[0].vars)
    # 解释一步把整张变量表都读进去了，reads 就该如实是那一整张表
    assert rec.action["reads"] == sorted(res.trace.records[0].vars)


def test_fallback_steps_and_entry_for_the_all_fallback_machine():
    """空机器起点即回退态：没有「从哪退下来」可言，记 FALLBACK 自己表示全程解释。

    与 test_01 / test_03 的期望对齐：整条轨迹都在 FALLBACK、停机原因是 terminal。
    """
    task = tc.gen_tasks(1, seed=3)[0]
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(empty_machine("table-clean"), task,
                           model=tc.build_model(), tools=tc.build_registry(fs),
                           doc=tc.skill_doc())
    assert res.stopped == runtime.STOP_TERMINAL, res.error
    assert res.fallback_entry == FALLBACK and res.fallback_entry_step == 1
    assert res.fallback_steps == len(res.trace.records) > 0
    assert all(r.state == FALLBACK for r in res.trace.records)


def test_abstain_route_is_reported_as_a_fallback_position():
    """判断弃权走兜底边进 FALLBACK（test_11 的场景），位置清单要说得出「从 s2 退的」。"""
    tasks = [t for t in tc.gen_tasks(12, seed=9)
             if tc.is_canonical(",".join(t["files"][t["input"]["path"]]["header"]))]
    task = tasks[0]
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(tc.reference_machine(), task,
                           model=tc.build_model(abstain_on=lambda values: True),
                           tools=tc.build_registry(fs), doc=tc.skill_doc())
    assert res.stopped == runtime.STOP_TERMINAL, res.error
    first_fb = next(r for r in res.trace.records if r.state == FALLBACK)
    before = [r.state for r in res.trace.records if r.step < first_fb.step]
    assert res.fallback_entry == before[-1]                  # 就是弃权那一步所在的状态
    assert res.fallback_entry_step == first_fb.step
    assert res.fallback_steps == sum(1 for r in res.trace.records
                                     if r.state == FALLBACK)


# --------------------------------------------------------------------------- #
# ⑤ 四种停机原因没被改动
# --------------------------------------------------------------------------- #
def test_terminal_and_stuck_and_max_steps_and_state_error_unchanged():
    assert (runtime.STOP_TERMINAL, runtime.STOP_STATE_ERROR, runtime.STOP_STUCK,
            runtime.STOP_MAX_STEPS) == ("terminal", "state_error", "stuck",
                                        "max_steps")
    # terminal：正常走完
    assert _run().stopped == runtime.STOP_TERMINAL

    # stuck：出边都不成立，也没有兜底边
    stuck = _machine()
    stuck.states["s1"].transitions = [Transition(cond="verified == 'nope'", to="s2")]
    r = _run(stuck)
    assert r.stopped == runtime.STOP_STUCK and "s1" in r.error

    # max_steps：自环转圈
    loop = _machine()
    loop.states["s1"].transitions = [Transition(to="s1")]
    r = _run(loop, max_steps=3)
    assert r.stopped == runtime.STOP_MAX_STEPS and len(r.trace.records) == 3

    # state_error：这一步炸了（工具没注册）
    boom = _machine()
    boom.states["s1"].action = ToolAction(name="没这个工具", writes=["verified"])
    r = _run(boom)
    assert r.stopped == runtime.STOP_STATE_ERROR and r.trace.records[-1].meta


def test_state_error_from_the_fallback_segment_still_counts_its_steps():
    """解释段里炸掉，账照记：错的那步也在 fallback_steps 里，位置清单不会漏。"""
    res = _run(_machine(after_s1=FALLBACK),
               model=ScriptedModel(gen=lambda p, v, h: {"kind": "user"}))
    assert res.stopped == runtime.STOP_STATE_ERROR
    assert "不认的动作类型" in res.error
    assert res.fallback_steps == 1 and res.fallback_entry == "s1"


def test_replaying_a_run_is_unaffected_by_the_bookkeeping():
    """meta 不参与回放：带账的轨迹照样被产出它的机器复述（回归 test_05 的地基）。"""
    from hexis.legacy import replay
    task = tc.gen_tasks(1, seed=3)[0]
    fs = tc.MemFS(task["files"])
    ref = tc.reference_machine()
    res = runtime.run_task(ref, task, model=tc.build_model(),
                           tools=tc.build_registry(fs), doc=tc.skill_doc())
    assert all(r.meta for r in res.trace.records)
    assert replay.reproduces(ref, res.trace)


# --------------------------------------------------------------------------- #
# ⑥ interpreter 只是别名
# --------------------------------------------------------------------------- #
def test_interpreter_is_the_same_module_not_a_second_implementation():
    for name in interpreter.__all__:
        assert getattr(interpreter, name) is getattr(runtime, name), name
    assert interpreter.run_task is runtime.run_task
    assert interpreter.RunResult is runtime.RunResult


@pytest.mark.parametrize("name", ["run_task", "interpret_step", "pick_edge",
                                  "prompt_digest", "RunResult"])
def test_interpreter_exports_the_public_surface(name):
    assert name in interpreter.__all__ and hasattr(interpreter, name)
