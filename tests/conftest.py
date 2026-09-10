"""Shared fixtures: a fixed random seed and judged traces of the hermetic table_clean example skill."""

import random

import pytest


@pytest.fixture(autouse=True)
def _fixed_seed():
    """确定性是这套测试的地基：错误率注入、任务生成都吃固定种子。"""
    random.seed(0)


@pytest.fixture
def make_traces():
    """产一批 table_clean 轨迹（已评判）。可注入判断误差率。"""
    from skill2fsm import judge, runtime
    from skill2fsm.examples import table_clean as tc

    def _make(n, *, seed=0, error_rate=0.0):
        refm = tc.reference_machine()
        out = []
        for task in tc.gen_tasks(n, seed=seed):
            fs = tc.MemFS(task["files"])
            res = runtime.run_task(
                refm, task,
                model=tc.build_model(error_rate=error_rate, seed=seed),
                tools=tc.build_registry(fs), doc=tc.skill_doc())
            out.append(judge.judged(
                res.trace, (lambda t, task=task: tc.verify(task, t)),
                refm.prohibitions))
        return out

    return _make


@pytest.fixture
def accepted(make_traces):
    """只要接受轨迹（T+）。"""
    def _acc(n, *, seed=0, error_rate=0.0):
        return [t for t in make_traces(n, seed=seed, error_rate=error_rate)
                if t.verdict == "accepted"]
    return _acc


@pytest.fixture
def max_fix():
    """一条轨迹里 fix_header 出现的次数（= 修复了几轮）。"""
    def _mf(trace):
        return sum(1 for r in trace.records
                   if (r.action or {}).get("name") == "fix_header")
    return _mf
