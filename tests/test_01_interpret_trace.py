"""① 解释模式产出合法轨迹 JSONL：字段齐全，工具结果由宿主填、不由模型编。"""

from skill2fsm import runtime
from skill2fsm.examples import table_clean as tc
from skill2fsm.schema import Trace, empty_machine


def _run_empty(task):
    fs = tc.MemFS(task["files"])
    return runtime.run_task(empty_machine("table-clean"), task,
                            model=tc.build_model(), tools=tc.build_registry(fs),
                            doc=tc.skill_doc()), fs


def test_interpret_produces_terminal_trace():
    task = tc.gen_tasks(1, seed=3)[0]
    res, _ = _run_empty(task)
    assert res.stopped == "terminal", res.error
    assert res.trace.records
    assert all(r.state == "FALLBACK" for r in res.trace.records)


def test_every_record_has_the_full_shape():
    task = tc.gen_tasks(1, seed=3)[0]
    res, _ = _run_empty(task)
    for r in res.trace.records:
        assert isinstance(r.step, int) and r.step >= 1
        assert r.state and isinstance(r.action, dict) and "kind" in r.action
        assert isinstance(r.vars, dict)          # 每步后的全部变量取值都在


def test_tool_output_is_filled_by_the_host_not_the_model():
    """read_csv 的 output 必须等于内存文件系统里的真实表头，而非模型自述。"""
    task = tc.gen_tasks(1, seed=3)[0]
    res, fs = _run_empty(task)
    reads = [r for r in res.trace.records if r.action.get("name") == "read_csv"]
    assert reads, "解释执行应该先读文件"
    path = task["input"]["path"]
    truth = ",".join(fs.files[path]["header"]) if path in fs.files else None
    # 文件已可能被 export 覆盖到别处；用任务原始表头核对
    original = ",".join(task["files"][path]["header"])
    assert reads[0].output.get("header_row") == original


def test_trace_jsonl_round_trip():
    task = tc.gen_tasks(1, seed=3)[0]
    res, _ = _run_empty(task)
    res.trace.verdict = "accepted"                # 补个 verdict 好序列化头部
    back = Trace.from_jsonl(res.trace.to_jsonl())
    assert back.verdict == "accepted"
    assert [r.action.get("name") for r in back.records] == \
           [r.action.get("name") for r in res.trace.records]
