"""(1) Interpretive mode produces valid trace JSONL: all fields present, tool results filled in by the host, not made up by the model."""

from hexis.examples import table_clean as tc
from hexis.execution import runtime
from hexis.machine.schema import Trace, empty_machine


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
        assert isinstance(r.vars, dict)          # all variable values after each step are present


def test_tool_output_is_filled_by_the_host_not_the_model():
    """The output of read_csv must equal the real header in the in-memory file system, not what the model claims."""
    task = tc.gen_tasks(1, seed=3)[0]
    res, fs = _run_empty(task)
    reads = [r for r in res.trace.records if r.action.get("name") == "read_csv"]
    assert reads, "interpretive execution should read the file first"
    path = task["input"]["path"]
    truth = ",".join(fs.files[path]["header"]) if path in fs.files else None
    # the file may already have been overwritten elsewhere by export; check against the task's original header
    original = ",".join(task["files"][path]["header"])
    assert reads[0].output.get("header_row") == original


def test_trace_jsonl_round_trip():
    task = tc.gen_tasks(1, seed=3)[0]
    res, _ = _run_empty(task)
    res.trace.verdict = "accepted"                # fill in a verdict so the header serializes
    back = Trace.from_jsonl(res.trace.to_jsonl())
    assert back.verdict == "accepted"
    assert [r.action.get("name") for r in back.records] == \
           [r.action.get("name") for r in res.trace.records]
