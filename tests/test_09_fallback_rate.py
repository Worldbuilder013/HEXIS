"""⑨ 模拟任务流下回退率随观测积累单调走低。

机器从轨迹学循环上限 K = ceil(loop_margin × 观测最大圈数)。早期只见过修一轮的任务，K 小，
遇到要修三轮的任务就计满落进 FALLBACK；见得越多 K 越大，同一条评测流上落入 FALLBACK 的
比例就降下来。这正是「违约率 = 未学得形态的质量 + 已学得区的弃权率」里第一项随任务流
缩小的样子。
"""

from skill2fsm import compiler, runtime
from skill2fsm.examples import table_clean as tc
from skill2fsm.schema import FALLBACK


def _fallback_rate(machine, tasks) -> float:
    hits = 0
    for task in tasks:
        fs = tc.MemFS(task["files"])
        res = runtime.run_task(machine, task, model=tc.build_model(),
                               tools=tc.build_registry(fs), doc=tc.skill_doc())
        if any(r.state == FALLBACK for r in res.trace.records):
            hits += 1
    return hits / len(tasks)


def test_fallback_rate_decreases_with_more_observed_traces(accepted, max_fix):
    pool = accepted(60, seed=2)
    eval_tasks = [t for t in tc.gen_tasks(30, seed=77)]      # 固定评测流（含要修三轮的）
    prohibitions = tc.reference_machine().prohibitions

    rates = []
    for cap in (1, 2, 3):                                    # 逐轮放宽观测：最多见过修 cap 轮的
        train = [t for t in pool if max_fix(t) <= cap]
        cr = compiler.compile(tc.skill_doc(), train, skill_id="tc",
                              prohibitions=prohibitions)
        assert cr.findings == []
        rates.append(_fallback_rate(cr.machine, eval_tasks))

    assert rates == sorted(rates, reverse=True), rates       # 单调不增
    assert rates[-1] < rates[0], rates                       # 且确实下降
    assert rates[-1] == 0.0                                  # 学满后评测流不再落回退
