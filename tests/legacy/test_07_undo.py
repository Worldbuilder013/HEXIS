"""⑦ 撤销生效：注入使校验失败的轨迹，扩展被整轮回滚，机器不变。"""

from hexis.legacy import compiler
from hexis.examples import table_clean as tc
from hexis.machine.schema import Record, Trace

_V = {"path": "z.csv", "output_path": "o.csv", "request": "r"}


def _ghost_trace() -> Trace:
    """一条会让先写后读检查失败的接受轨迹：判断动作声称要读一个从没被写的 ghost 变量。

    它做成**最短**并放在轨迹列表最前，好让它先给判断状态定 reads（编译按短轨迹优先、
    首见定形）。
    """
    hv = {"header_row": "名称,数量", "rows": [["a", "1"]]}
    return Trace(task={"input": dict(_V)}, verdict="accepted", records=[
        Record(step=1, state="a",
               action={"kind": "tool", "name": "read_csv", "input": {"path": "z.csv"}},
               output={"ok": True, **hv}, vars={**_V, **hv}),
        Record(step=2, state="b",
               action={"kind": "judge", "prompt": tc.JUDGE_Q,
                       "reads": ["header_row", "ghost"]},
               output={"header_ok": "规范"}, vars={**_V, **hv, "header_ok": "规范"}),
        Record(step=3, state="c",
               action={"kind": "tool", "name": "export",
                       "input": {"header_row": "名称,数量", "rows": [["a", "1"]],
                                 "output_path": "o.csv", "source_path": "z.csv"}},
               output={"ok": True, "output_path": "o.csv"},
               vars={**_V, **hv, "header_ok": "规范"}),
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
    assert res.machine is base                              # 整轮回滚
    assert base.model_dump_json(by_alias=True) == before    # 机器逐字节不变
    assert any("ghost" in f for f in res.findings)          # 驳回附理由
