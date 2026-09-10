"""多选题判分：产出文件里的 ``\\boxed{X}`` 与参考答案字母比较。

只认最后一个 ``\\boxed{...}``；花括号里允许多余空白与括号（``\\boxed{(C)}``）。文件缺失、没有 boxed、
字母不在 A–E 都判失败并给出原因。
"""
from __future__ import annotations

import pathlib
import re

_BOX = re.compile(r"\\boxed\{\s*\(?\s*([A-Ea-e])\s*\)?\s*\}")


def extract_choice(text: str) -> str | None:
    hits = _BOX.findall(text or "")
    return hits[-1].upper() if hits else None


def grade_answer_file(path: pathlib.Path, expected: str) -> tuple[bool, str]:
    path = pathlib.Path(path)
    if not path.is_file():
        return False, "没有产出 answer.txt"
    text = path.read_text(encoding="utf-8", errors="replace")
    got = extract_choice(text)
    if got is None:
        return False, f"answer.txt 里没有 \\boxed{{X}}：{text.strip()[:60]!r}"
    exp = str(expected).strip().upper()
    return (got == exp), ("" if got == exp else f"参考答案 {exp}，产出 {got}")


__all__ = ["extract_choice", "grade_answer_file"]
