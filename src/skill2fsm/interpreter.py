"""``interpreter`` —— :mod:`skill2fsm.runtime` 的**别名模块**：同一份实现，两个名字。

为什么两个名字都留着：规格文档把这一层叫「解释器」（``interpreter``），照它写的代码会
``from skill2fsm import interpreter``；而这个包从第一天起就把它叫 ``runtime``，12 个测试文件
是按 ``from skill2fsm import runtime`` 导入的，轨迹与报告里也已经写着 ``runtime.run_task``
的出身。改名能让规格好看一点，代价是一次性打断既有导入与既有产物的可追溯性——不值。于是这里
只做一件事：把 runtime 的公开面**原样**再导出一遍，让两个名字都能解析。

**没有第二份实现。** 这里不新增行为、不包装、不改默认值：``interpreter.run_task is
runtime.run_task`` 恒真（``test_26`` 把这条钉死）。两份实现会各自漂移，而它们本来就是同一个
解释器——一台机器怎么跑、怎么记账、怎么进 FALLBACK，只该有一处说了算。

要改行为请改 :mod:`skill2fsm.runtime`，这里跟着动就好。
"""

from __future__ import annotations

from .runtime import (
    STOP_MAX_STEPS,
    STOP_STATE_ERROR,
    STOP_STUCK,
    STOP_TERMINAL,
    RunResult,
    fill_template,
    interpret_step,
    pick_edge,
    prompt_digest,
    rebuild,
    run_task,
)

__all__ = [
    "STOP_MAX_STEPS", "STOP_STATE_ERROR", "STOP_STUCK", "STOP_TERMINAL",
    "RunResult", "fill_template", "interpret_step", "pick_edge",
    "prompt_digest", "rebuild", "run_task",
]
