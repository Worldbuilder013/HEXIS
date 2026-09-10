"""table_clean —— 一个密闭、秒级的玩具技能，用来自测整套 skill2fsm 流程。

读一张 CSV、检查并（必要时逐处修复）表头、导出，带一条「不得覆盖原文件」的禁止性要求。
三个工具是内存纯函数，模型是脚本桩，任务由生成器造，全程无网络、可复现。
"""

from pathlib import Path

from .acceptance import verify
from .scripted import (
    JUDGE_Q, LABELS, build_model, gen_tasks, interpret, make_judge,
    reference_machine,
)
from .tools import MemFS, build_registry, is_canonical

SKILL_PATH = Path(__file__).with_name("SKILL.md")


def skill_doc() -> str:
    """SKILL.md 的私有正文（FALLBACK 解释执行时模型读的就是它）。"""
    return SKILL_PATH.read_text(encoding="utf-8")


__all__ = [
    "JUDGE_Q", "LABELS", "MemFS", "SKILL_PATH", "build_model", "build_registry",
    "gen_tasks", "interpret", "is_canonical", "make_judge", "reference_machine",
    "skill_doc", "verify",
]
