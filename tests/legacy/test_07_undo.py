"""Undo takes effect: inject a trace that makes validation fail, the extension is rolled back for the whole round, and the machine is unchanged."""

from hexis.examples import table_clean as tc
from hexis.legacy import compiler
from hexis.machine.schema import Record, Trace

_V = {"path": "z.csv", "output_path": "o.csv", "request": "r"}


def _ghost_trace() -> Trace:
    """An accepted trace that makes the write-before-read check fail: the judge action claims to read a ghost variable that is never written.

    It is made the **shortest** and placed first in the trace list, so that it is the one that sets
    the judge state's reads (compilation goes shortest trace first, and the first sighting fixes the
    shape).
    """
    hv = {"header_row": "name,quantity", "rows": [["a", "1"]]}
    return Trace(task={"input": dict(_V)}, verdict="accepted", records=[
        Record(step=1, state="a",
               action={"kind": "tool", "name": "read_csv", "input": {"path": "z.csv"}},
               output={"ok": True, **hv}, vars={**_V, **hv}),
        Record(step=2, state="b",
               action={"kind": "judge", "prompt": tc.JUDGE_Q,
                       "reads": ["header_row", "ghost"]},
               output={"header_ok": "well_formed"}, vars={**_V, **hv, "header_ok": "well_formed"}),
        Record(step=3, state="c",
               action={"kind": "tool", "name": "export",
                       "input": {"header_row": "name,quantity", "rows": [["a", "1"]],
                                 "output_path": "o.csv", "source_path": "z.csv"}},
               output={"ok": True, "output_path": "o.csv"},
               vars={**_V, **hv, "header_ok": "well_formed"}),
        Record(step=4, state="d", action={"kind": "end", "terminal": "done"},
               vars=dict(_V)),
    ])


def test_clean_round_applies(accepted):
    res, applied = compiler.compile_round(
        None, tc.skill_doc(), accepted(24, seed=2), skill_id="tc",
        prohibitions=tc.reference_machine().prohibitions)
    assert applied and res.findings == []


def test_failing_round_rolls_back_and_base_is_untouched(accepted):
    good = accepted(24, seed=2)
    base = compiler.compile(tc.skill_doc(), good, skill_id="tc",
                            prohibitions=tc.reference_machine().prohibitions).machine
    before = base.model_dump_json(by_alias=True)

    res, applied = compiler.compile_round(
        base, tc.skill_doc(), [_ghost_trace()] + good, skill_id="tc",
        prohibitions=tc.reference_machine().prohibitions)

    assert not applied
    assert res.machine is base                              # whole round rolled back
    assert base.model_dump_json(by_alias=True) == before    # machine unchanged byte for byte
    assert any("ghost" in f for f in res.findings)          # the rejection comes with a reason
