"""⑪ 判断弃权时路由到 FALLBACK，任务仍能完成。

判断动作拿不准就输出弃权标签；机器随即走兜底边进 FALLBACK（解释执行），而不是硬答一个
可能错的标签。这是「一条路径至少错一次概率 ≤ Σεᵢ」里弃权作为调节阀的运行时体现。
"""

from skill2fsm import runtime
from skill2fsm.examples import table_clean as tc
from skill2fsm.schema import FALLBACK


def _run_with_abstaining_judge(task):
    """判断永远弃权的模型。"""
    fs = tc.MemFS(task["files"])
    model = tc.build_model(abstain_on=lambda values: True)
    return runtime.run_task(tc.reference_machine(), task, model=model,
                            tools=tc.build_registry(fs), doc=tc.skill_doc())


def test_abstain_routes_into_fallback():
    # 用规范表头任务：判断本可直接答「规范」，但这里强制弃权
    tasks = [t for t in tc.gen_tasks(12, seed=9)
             if tc.is_canonical(",".join(t["files"][t["input"]["path"]]["header"]))]
    task = tasks[0]
    res = _run_with_abstaining_judge(task)
    assert FALLBACK in res.path(), f"弃权后应进 FALLBACK，实际 {res.path()}"


def test_task_still_completes_after_abstain():
    tasks = [t for t in tc.gen_tasks(12, seed=9)
             if tc.is_canonical(",".join(t["files"][t["input"]["path"]]["header"]))]
    task = tasks[0]
    res = _run_with_abstaining_judge(task)
    assert res.stopped == "terminal", res.error
    assert tc.verify(task, res.trace)              # 弃权走解释兜底，任务照样完成
