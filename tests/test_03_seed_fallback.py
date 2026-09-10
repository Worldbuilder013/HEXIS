"""③ 初始机器全回退：起点直接进 FALLBACK，任意接受轨迹都被平凡复述。"""

from skill2fsm import replay, runtime
from skill2fsm.examples import table_clean as tc
from skill2fsm.schema import FALLBACK, empty_machine


def test_empty_machine_starts_at_fallback():
    m = empty_machine("table-clean")
    assert m.initial == FALLBACK
    assert FALLBACK in m.states


def test_empty_machine_reproduces_any_trace():
    """空机器一进 FALLBACK 就是解释模式，对任何轨迹回放都平凡通过。"""
    ref = tc.reference_machine()
    em = empty_machine("table-clean")
    for task in tc.gen_tasks(6, seed=7):
        fs = tc.MemFS(task["files"])
        # 用一台真机器跑出一条真轨迹，再拿空机器回放它
        res = runtime.run_task(ref, task, model=tc.build_model(),
                               tools=tc.build_registry(fs), doc=tc.skill_doc())
        assert replay.reproduces(em, res.trace)


def test_empty_machine_run_is_terminal_and_correct():
    task = tc.gen_tasks(1, seed=7)[0]
    fs = tc.MemFS(task["files"])
    res = runtime.run_task(empty_machine("table-clean"), task,
                           model=tc.build_model(), tools=tc.build_registry(fs),
                           doc=tc.skill_doc())
    assert res.stopped == "terminal"
    assert tc.verify(task, res.trace)
