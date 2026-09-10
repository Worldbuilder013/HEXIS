"""table_clean 的密闭脚本：判断桩、FALLBACK 解释脚本、任务生成器，以及一台手写参考机器。

这里没有任何随机的不确定：判断用 :func:`~.tools.is_canonical` 给「真值」，错误率由
:class:`~skill2fsm.model_iface.ScriptedModel` 按种子确定性注入；任务生成吃固定种子。参考
机器 :func:`reference_machine` 是一台**正确的** table_clean EFSM，用来喂运行器与回放做对照，
也是「编译出来的机器该长什么样」的标尺。
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from ...model_iface import ScriptedModel
from ...schema import (
    EndAction, JudgeAction, Machine, Prohibition, State, Terminal,
    ToolAction, Transition, Variable,
)
from .tools import is_canonical

JUDGE_Q = "当前 header_row 是否符合 SKILL.md S2.1 的规范判据"
LABELS = ["规范", "不规范", "弃权"]

_GOOD_HEADERS = [
    ["名称", "数量", "日期"],
    ["商品", "单价", "库存"],
    ["姓名", "年龄", "城市"],
    ["订单号", "金额", "状态"],
]


# --------------------------------------------------------------------------- #
# 判断桩与 FALLBACK 解释脚本
# --------------------------------------------------------------------------- #
def make_judge(abstain_on: Optional[Callable[[dict], bool]] = None):
    """判断函数：读 ``header_row``，规范/不规范二选一。``abstain_on`` 命中则弃权。"""

    def judge(prompt: str, values: dict) -> str:
        if abstain_on and abstain_on(values):
            return "弃权"
        return "规范" if is_canonical(values.get("header_row", "")) else "不规范"

    return judge


def interpret(prompt: str, values: dict, history: tuple = ()) -> dict:
    """FALLBACK 解释脚本：模型读文档+历史+变量，逐步走完 table_clean。

    完全由当前变量与历史决定下一动作，因此确定性、可复述。判断（表头规范吗）在这里是
    模型**内联**做的（解释模式不建独立判断状态），用 :func:`is_canonical` 模拟。
    """
    fixes = sum(1 for r in history
                if (r.get("action") or {}).get("name") == "fix_header")
    exported = any((r.get("action") or {}).get("name") == "export" for r in history)
    if exported:
        return {"kind": "end", "terminal": "done"}
    if "header_row" not in values:
        return {"kind": "tool", "name": "read_csv",
                "input": {"path": values["path"]},
                "writes": ["header_row", "rows"]}
    hr = values["header_row"]
    if not is_canonical(hr) and fixes < 3:
        return {"kind": "tool", "name": "fix_header",
                "input": {"header_row": hr}, "writes": ["header_row"]}
    return {"kind": "tool", "name": "export",
            "input": {"header_row": hr, "rows": values.get("rows", []),
                      "output_path": values["output_path"],
                      "source_path": values["path"]},
            "writes": ["output_path"]}


def build_model(*, error_rate: float = 0.0, seed: int = 0,
                abstain_on: Optional[Callable[[dict], bool]] = None) -> ScriptedModel:
    """构造密闭模型桩：judge 走 :func:`make_judge`，generate 走 :func:`interpret`。"""
    return ScriptedModel(judge=make_judge(abstain_on), gen=interpret,
                         error_rate=error_rate, seed=seed)


# --------------------------------------------------------------------------- #
# 任务生成器
# --------------------------------------------------------------------------- #
def gen_tasks(n: int, *, seed: int = 0, bad_ratio: float = 0.5) -> list[dict]:
    """产出 n 个任务，混合规范表头（直接导出）与不规范表头（触发 1..3 次修复）。

    每个任务自带内存文件（``files``），确定性由 ``seed`` 保证。``output_path`` 恒不等于
    ``path``（走 happy path）；触犯 P1 的任务由测试单独构造。
    """
    rng = random.Random(seed)
    tasks: list[dict] = []
    for i in range(n):
        good = list(rng.choice(_GOOD_HEADERS))
        make_bad = rng.random() < bad_ratio
        header = list(good)
        if make_bad:
            k = rng.randint(1, min(3, len(header)))
            for j in rng.sample(range(len(header)), k):
                header[j] = "" if rng.random() < 0.5 else f"Unnamed: {j}"
        rows = [[f"v{i}{c}" for c in range(len(good))]]
        path, out = f"in{i}.csv", f"out{i}.csv"
        tasks.append({
            "task_id": f"t{i}",
            "input": {"path": path, "output_path": out,
                      "request": "清洗这张表并导出"},
            "files": {path: {"header": header, "rows": rows}},
            "acceptance": {"kind": "script"},
            "_expected_header": good,   # 供验收比对（下划线前缀＝测试内部用）
        })
    return tasks


# --------------------------------------------------------------------------- #
# 手写参考机器（正确的 table_clean EFSM）
# --------------------------------------------------------------------------- #
def reference_machine() -> Machine:
    """一台手写的、正确的 table_clean 状态机。读→判→（修⟲）→导出，带 P1。"""
    return Machine(
        skill_id="table-clean",
        initial="s1",
        variables=[
            Variable(name="path", type="string", init_from="task.input.path"),
            Variable(name="output_path", type="string", init_from="task.input.output_path"),
            Variable(name="request", type="string", init_from="task.input.request"),
            Variable(name="header_row", type="string"),
            Variable(name="rows", type="array"),
            Variable(name="header_ok", type="string"),
            Variable(name="fix_count", type="integer", init=0),
        ],
        states={
            "s1": State(id="s1", clause="S1",
                        action=ToolAction(name="read_csv",
                                          input={"path": "${path}"},
                                          reads=["path"],
                                          writes=["header_row", "rows"]),
                        transitions=[Transition(to="s2")]),
            "s2": State(id="s2", clause="S2.1",
                        action=JudgeAction(prompt=JUDGE_Q,
                                           reads=["header_row"],
                                           writes=["header_ok"],
                                           labels=list(LABELS)),
                        transitions=[
                            Transition(cond="header_ok == '规范'", to="s4"),
                            Transition(cond="header_ok == '不规范' and fix_count < 3",
                                       to="s3"),
                            Transition(to="FALLBACK"),
                        ]),
            "s3": State(id="s3", clause="S3",
                        action=ToolAction(name="fix_header",
                                          input={"header_row": "${header_row}"},
                                          reads=["header_row"],
                                          writes=["header_row"]),
                        transitions=[Transition(to="s2", inc="fix_count")]),
            "s4": State(id="s4", clause="S4",
                        action=ToolAction(name="export",
                                          input={"header_row": "${header_row}",
                                                 "rows": "${rows}",
                                                 "output_path": "${output_path}",
                                                 "source_path": "${path}"},
                                          reads=["header_row", "rows",
                                                 "output_path", "path"],
                                          writes=["output_path"]),
                        transitions=[Transition(to="end")]),
            "end": State(id="end", action=EndAction(terminal="done")),
            "FALLBACK": State(id="FALLBACK", action=EndAction(terminal="done")),
        },
        terminals=[Terminal(id="done", output=["output_path"])],
        prohibitions=[Prohibition(
            id="P1", check="forbid_action",
            pattern={"name": "export",
                     "equal": ["input.output_path", "input.source_path"]})],
    )
