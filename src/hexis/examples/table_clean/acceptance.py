"""table_clean 的客观验收：只看轨迹，不看模型的自述。

「任务算完成」= 轨迹里有 ``export`` 动作、导出的表头规范、且执行走到了终止。这是确定性的
客观闸，和「有没有触犯禁止性要求 P1」是两条独立的轴——后者由 judge.evaluate 依 machine
的 prohibitions 判（触犯即拒，哪怕这里通过）。
"""

from __future__ import annotations

from hexis.machine.schema import Trace
from hexis.examples.table_clean.tools import is_canonical


def verify(task: dict, trace: Trace) -> bool:
    """结果对不对：有导出、导出的表头规范、且到达了终止动作。"""
    exports = [r for r in trace.records
               if (r.action or {}).get("name") == "export"]
    if not exports:
        return False
    header_row = (exports[-1].action.get("input") or {}).get("header_row", "")
    if not is_canonical(header_row):
        return False
    return any((r.action or {}).get("kind") == "end" for r in trace.records)
