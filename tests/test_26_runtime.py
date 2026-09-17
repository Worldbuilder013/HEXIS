"""The interpreter's execution-side accounting: per-step meta, what a model action records, usage left empty when unmeasured, fallback position and step count.

These tests are fully sealed: the model is :class:`~hexis.llm.model_iface.ScriptedModel` (table lookup, no
network), and the tool is an in-process pure function. They watch four things that **the report reads but that
do not matter semantically** -- precisely because they do not matter, getting them wrong fails no acceptance
check and only quietly distorts the numbers in the experiment report, so they have to be pinned separately:

1. every step has ``Record.meta``: latency, how many model calls this step made, how many tokens it spent;
2. a ``model`` action records template id / template / reads / prompt digest, **not the rendered full text** (the problem is not stored twice);
3. token fields are **``None`` when unmeasured**, not 0, let alone a number estimated from unit prices;
4. how many steps the fallback segment ran and which state it fell back from (the fallback rate and position list of the report).

It also cross-checks that the existing stop reasons are unchanged (consistent with the expectations of test_01 / test_03 /
test_11), and that the alias module :mod:`hexis.execution.interpreter` really is only an alias.
"""

import json

import pytest

from hexis.examples import table_clean as tc
from hexis.execution import interpreter, runtime
from hexis.llm.model_iface import ScriptedModel, ToolRegistry
from hexis.machine.schema import (
    FALLBACK,
    EndAction,
    Machine,
    ModelAction,
    State,
    Terminal,
    ToolAction,
    Transition,
    Variable,
    empty_machine,
)

#: problem statement: it should appear in ``Record.vars`` (the variable table) but **never** in the record of a model action.
PROBLEM = "Find the sum of the decimal digits of 2^10 and write the answer as an integer"
#: private template of state s2. It does not contain the problem -- the problem is a variable, filled in only at run time.
TEMPLATE = "Solve per document S3: read problem and reply with only one JSON object {\"answer\": ...}"
#: the "document" the FALLBACK interpretation reads. Long enough to verify it is not copied into the trace on every step.
DOC = "# Toy skill document\n" + "This is what the interpreted segment reads to give the next action step by step.\n" * 40
#: the command line the tool actually executed (recorded like the ``command`` field of sandbox.ExecResult).
ARGV = ["python", "scripts/math_verify.py", "--json", "equiv", "1024", "1024"]


# --------------------------------------------------------------------------- #
# A small hand-built machine and stubs
# --------------------------------------------------------------------------- #
def _tools() -> ToolRegistry:
    """One tool: reports ``command`` (the argv that really ran), like sandbox does."""
    reg = ToolRegistry()
    reg.add("math_verify", lambda inp: {"ok": True, "verified": True,
                                        "command": list(ARGV)})
    return reg


def _task(problem: str = PROBLEM) -> dict:
    return {"task_id": "m1", "input": {"problem": problem}}


def _machine(*, after_s1: str = "s2") -> Machine:
    """read -> (verify) -> generate -> end. Pointing ``after_s1`` at ``FALLBACK`` means "fall back after s1"."""
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
    """One script per kind of prompt: the s2 template yields the answer, DOC yields "the next action"."""
    if prompt == TEMPLATE:
        return {"answer": "7"}
    if "draft" not in values:                       # first interpreted step: write a draft first (model action)
        return {"kind": "model", "writes": ["draft"],
                "output": {"draft": "2^10 = 1024"}}
    return {"kind": "end", "terminal": "done", "answer": "7"}   # second step: submit


def _model() -> ScriptedModel:
    return ScriptedModel(gen=_gen)


class _MeteredModel(ScriptedModel):
    """A stub with ``usage()``, shaped like the cumulative counters of :class:`~hexis.llm.llm_client.ModelAdapter`.

    ``unmeasured=True`` simulates "the endpoint reported no usage at all": the call count still grows, tokens stay
    at 0 -- the runner must report this as ``None`` instead of copying that 0.
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
# 1. every step has meta
# --------------------------------------------------------------------------- #
def test_every_step_records_meta_with_tokens_and_ms():
    """The latency and token keys are **present on every step** -- the report must tell "this step spent nothing" from "this step was not measured"."""
    res = _run()
    assert res.stopped == runtime.STOP_TERMINAL, res.error
    assert res.path() == ["s1", "s2", "end"]
    for rec in res.trace.records:
        assert isinstance(rec.meta, dict) and rec.meta, rec.step
        assert isinstance(rec.meta["ms"], float) and rec.meta["ms"] >= 0.0
        assert isinstance(rec.meta["llm_calls"], int)
        assert "prompt_tokens" in rec.meta and "completion_tokens" in rec.meta


def test_meta_stays_out_of_action_and_output():
    """Accounting stays out of action/output: it takes no part in normalization or judging, so it must not change state identity."""
    res = _run()
    for rec in res.trace.records:
        assert "ms" not in rec.action and "ms" not in rec.output
        assert "llm_calls" not in rec.action and "llm_calls" not in rec.output


def test_tool_step_meta_carries_the_argv_actually_executed():
    """A tool step records the argv that **really ran**, not a command line reconstructed from the input template."""
    rec = _run().trace.records[0]
    assert rec.action["kind"] == "tool"
    assert rec.meta["argv"] == ARGV
    # ``--json`` comes before the subcommand -- exactly where "was the template assembled right" and "how it actually ran" must reconcile
    assert rec.meta["argv"].index("--json") < rec.meta["argv"].index("equiv")


def test_steps_without_a_model_call_report_zero_not_none():
    """A step without a model call has 0 tokens: that is a measured result, not "unknown"."""
    tool_step, end_step = _run().trace.records[0], _run().trace.records[-1]
    for rec in (tool_step, end_step):
        assert rec.meta["llm_calls"] == 0
        assert rec.meta["prompt_tokens"] == 0 and rec.meta["completion_tokens"] == 0


# --------------------------------------------------------------------------- #
# 2. what a model action records
# --------------------------------------------------------------------------- #
def test_model_record_carries_template_id_reads_and_digest():
    rec = _model_record(_run())
    assert rec.state == "s2" and rec.action["template_id"] == "s2"
    assert rec.action["prompt"] == TEMPLATE                  # template text = state identity
    assert rec.action["reads"] == ["problem"]                # the variables this step actually consumed
    digest = rec.action["prompt_sha256"]
    assert len(digest) == 64 and int(digest, 16) >= 0        # a sha256 hex string
    assert digest == runtime.prompt_digest(TEMPLATE, {"problem": PROBLEM})


def test_model_record_does_not_carry_the_rendered_prompt():
    """The rendered prompt embeds the whole problem. Storing the problem **once** in ``vars`` is enough; the action does not store it a second time."""
    rec = _model_record(_run())
    assert PROBLEM not in json.dumps(rec.action, ensure_ascii=False)
    assert rec.vars["problem"] == PROBLEM                    # its only legitimate home is the variable table
    assert rec.output == {"answer": "7"}                     # output collected as usual, by the writes whitelist


def test_prompt_digest_is_stable_and_value_sensitive():
    """Same template and values => same digest; a different problem => a different digest -- so "what this step asked" is verifiable."""
    a = runtime.prompt_digest(TEMPLATE, {"problem": PROBLEM})
    assert a == runtime.prompt_digest(TEMPLATE, {"problem": PROBLEM})
    assert a != runtime.prompt_digest(TEMPLATE, {"problem": "another problem"})
    assert a != runtime.prompt_digest(TEMPLATE + "!", {"problem": PROBLEM})
    other = _run(task=_task("another problem"))
    assert _model_record(other).action["prompt_sha256"] != a


# --------------------------------------------------------------------------- #
# 3. usage: None when unmeasured
# --------------------------------------------------------------------------- #
def test_token_fields_are_none_when_the_model_reports_no_usage():
    """``ScriptedModel`` has no ``usage()`` -- that means **unknown**, not 0."""
    assert not hasattr(_model(), "usage")                    # the other half of the duck typing really is absent
    res = _run()
    assert res.llm_calls == 1                                # the call count is counted by the runner itself, so it is measured
    assert res.prompt_tokens is None and res.completion_tokens is None
    assert res.unmeasured_calls == 1
    assert res.wall_s >= 0.0                                 # wall-clock time is always measurable: a number, not None
    meta = _model_record(res).meta
    assert meta["llm_calls"] == 1
    assert meta["prompt_tokens"] is None and meta["completion_tokens"] is None


def test_token_fields_are_read_from_the_model_when_it_reports_usage():
    """If the model has ``usage()``, its measured numbers are used -- duck typing, without importing llm_client or changing the interface."""
    model = _MeteredModel(gen=_gen, per_call=(11, 7))
    res = _run(model=model)
    assert res.llm_calls == 1
    assert (res.prompt_tokens, res.completion_tokens) == (11, 7)
    assert res.unmeasured_calls == 0
    meta = _model_record(res).meta
    assert (meta["prompt_tokens"], meta["completion_tokens"]) == (11, 7)
    assert "unmeasured_calls" not in meta                    # when everything was measured, this field is not written


def test_unreported_usage_never_becomes_zero():
    """The 0 accumulated when the endpoint reports no usage must not pose as a measurement: both the run and the step report ``None``."""
    model = _MeteredModel(gen=_gen, unmeasured=True)
    res = _run(model=model)
    assert model.usage()["prompt_tokens"] == 0               # on the stub side it really is 0
    assert res.prompt_tokens is None and res.completion_tokens is None
    assert res.unmeasured_calls == 1 == res.llm_calls
    assert _model_record(res).meta["unmeasured_calls"] == 1


def test_a_run_without_any_model_call_spends_zero_tokens():
    """No model call at all => 0 (a measurement); kept distinct from "not measured"."""
    m = _machine()
    m.states["s1"].transitions = [Transition(to="end")]      # skip the generation step
    res = _run(m)
    assert res.llm_calls == 0
    assert (res.prompt_tokens, res.completion_tokens) == (0, 0)
    assert res.unmeasured_calls == 0


# --------------------------------------------------------------------------- #
# 4. fallback: how many steps, and where it fell back from
# --------------------------------------------------------------------------- #
def test_fallback_entry_state_and_step_count_are_recorded():
    """The two things the fallback rate and position list need: which state it fell back from, and how many steps the interpreted segment ran."""
    res = _run(_machine(after_s1=FALLBACK))
    assert res.stopped == runtime.STOP_TERMINAL, res.error
    assert res.entered_fallback()
    assert res.fallback_entry == "s1"                        # position: fell back from s1
    assert res.fallback_entry_step == 2                      # step number of the first record of the interpreted segment
    assert res.fallback_steps == 2 == sum(1 for r in res.trace.records
                                          if r.state == FALLBACK)
    assert res.path() == ["s1", FALLBACK, FALLBACK]


def test_fallback_runs_a_model_action_and_submits_a_final_answer():
    """The interpreted segment can run generation and submit an answer -- otherwise fallback would only be "giving up gracefully"."""
    res = _run(_machine(after_s1=FALLBACK))
    steps = res.trace.records
    assert steps[1].action["kind"] == "model"
    assert steps[1].output == {"draft": "2^10 = 1024"}       # output collected by the writes it declared
    assert res.values["draft"] == "2^10 = 1024"              # and it also landed in the variable table
    assert steps[2].action == {"kind": "end", "terminal": "done"}
    assert steps[2].output == {"answer": "7"}                # the end step carries the final answer
    assert res.values["answer"] == "7"                       # judging scores based on this


def test_fallback_history_does_not_carry_the_bookkeeping():
    """The history fed back to the interpreter has **no meta**: accounting is the host's business and must not take up the model's limited context.

    ``ModelAdapter`` truncates each history item to 800 characters; letting tokens/latency/argv eat into that
    allowance would trade the actions and results that matter for host implementation details.
    """
    seen = []

    def gen(prompt, values, history):
        seen.append(history)
        return _gen(prompt, values, history)

    res = _run(_machine(after_s1=FALLBACK), model=ScriptedModel(gen=gen))
    assert res.stopped == runtime.STOP_TERMINAL, res.error
    last = seen[-1]
    assert last and all("meta" not in h for h in last)
    assert all("action" in h and "output" in h for h in last)   # everything that matters is still there


def test_fallback_model_record_hides_the_document_but_keeps_the_digest():
    """The "template" of FALLBACK is the whole document: only the id and digest are recorded; inlining it on every step would be even worse than recording the rendered text."""
    res = _run(_machine(after_s1=FALLBACK))
    rec = res.trace.records[1]
    text = json.dumps(rec.action, ensure_ascii=False)
    assert rec.action["template_id"] == FALLBACK and "prompt" not in rec.action
    assert "This is what the interpreted segment reads" not in text and len(text) < len(DOC)
    assert rec.action["prompt_sha256"] == runtime.prompt_digest(DOC,
                                                                res.trace.records[0].vars)
    # an interpreted step reads the whole variable table, so reads must faithfully be that whole table
    assert rec.action["reads"] == sorted(res.trace.records[0].vars)


def test_fallback_steps_and_entry_for_the_all_fallback_machine():
    """For the empty machine the start is the fallback state: there is no "where it fell back from", so FALLBACK itself is recorded to mean interpreted throughout.

    Consistent with the expectations of test_01 / test_03: the whole trace is in FALLBACK and the stop reason is terminal.
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
    """A judge abstains and takes the default edge into FALLBACK (the test_11 scenario); the position list must be able to say "fell back from s2"."""
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
    assert res.fallback_entry == before[-1]                  # exactly the state of the abstaining step
    assert res.fallback_entry_step == first_fb.step
    assert res.fallback_steps == sum(1 for r in res.trace.records
                                     if r.state == FALLBACK)


# --------------------------------------------------------------------------- #
# 5. the four stop reasons are unchanged
# --------------------------------------------------------------------------- #
def test_terminal_and_stuck_and_max_steps_and_state_error_unchanged():
    assert (runtime.STOP_TERMINAL, runtime.STOP_STATE_ERROR, runtime.STOP_STUCK,
            runtime.STOP_MAX_STEPS) == ("terminal", "state_error", "stuck",
                                        "max_steps")
    # terminal: completes normally
    assert _run().stopped == runtime.STOP_TERMINAL

    # stuck: no outgoing edge holds, and there is no default edge
    stuck = _machine()
    stuck.states["s1"].transitions = [Transition(cond="verified == 'nope'", to="s2")]
    r = _run(stuck)
    assert r.stopped == runtime.STOP_STUCK and "s1" in r.error

    # max_steps: going in circles on a self-loop
    loop = _machine()
    loop.states["s1"].transitions = [Transition(to="s1")]
    r = _run(loop, max_steps=3)
    assert r.stopped == runtime.STOP_MAX_STEPS and len(r.trace.records) == 3

    # state_error: this step blew up (tool not registered)
    boom = _machine()
    boom.states["s1"].action = ToolAction(name="no_such_tool", writes=["verified"])
    r = _run(boom)
    assert r.stopped == runtime.STOP_STATE_ERROR and r.trace.records[-1].meta


def test_state_error_from_the_fallback_segment_still_counts_its_steps():
    """A blow-up inside the interpreted segment is still accounted for: the failing step is in fallback_steps too, so the position list misses nothing."""
    res = _run(_machine(after_s1=FALLBACK),
               model=ScriptedModel(gen=lambda p, v, h: {"kind": "user"}))
    assert res.stopped == runtime.STOP_STATE_ERROR
    assert "unknown action kind" in res.error
    assert res.fallback_steps == 1 and res.fallback_entry == "s1"


# --------------------------------------------------------------------------- #
# 6. interpreter is only an alias
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
