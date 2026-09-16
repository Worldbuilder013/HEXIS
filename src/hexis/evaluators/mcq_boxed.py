"""Multiple-choice grading: the ``\\boxed{X}`` in the output file is compared with the reference answer letter.

Only the last ``\\boxed{...}`` counts; extra whitespace and parentheses are allowed inside the braces
(``\\boxed{(C)}``). A missing file, no boxed answer, or a letter outside A-E all fail, with the reason given.
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
        return False, "no answer.txt produced"
    text = path.read_text(encoding="utf-8", errors="replace")
    got = extract_choice(text)
    if got is None:
        return False, f"no \\boxed{{X}} in answer.txt: {text.strip()[:60]!r}"
    exp = str(expected).strip().upper()
    return (got == exp), ("" if got == exp else f"reference answer {exp}, got {got}")


__all__ = ["extract_choice", "grade_answer_file"]
