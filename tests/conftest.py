"""Shared fixtures: a fixed random seed and judged traces of the hermetic table_clean example skill."""

import random

import pytest


@pytest.fixture(autouse=True)
def _fixed_seed():
    """Determinism is the foundation of this test suite: error rate injection and task generation both use fixed seeds."""
    random.seed(0)


@pytest.fixture
def make_traces():
    """Produce a batch of table_clean traces (already judged). A judge error rate can be injected."""
    from hexis.examples import table_clean as tc
    from hexis.execution import runtime
    from hexis.traces import judge

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
    """Accepted traces (T+) only."""
    def _acc(n, *, seed=0, error_rate=0.0):
        return [t for t in make_traces(n, seed=seed, error_rate=error_rate)
                if t.verdict == "accepted"]
    return _acc


@pytest.fixture
def max_fix():
    """Number of fix_header steps in a trace (= how many repair rounds ran)."""
    def _mf(trace):
        return sum(1 for r in trace.records
                   if (r.action or {}).get("name") == "fix_header")
    return _mf
