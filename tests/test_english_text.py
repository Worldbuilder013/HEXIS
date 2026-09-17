"""User-visible text, model-visible text, comments and docs are English.

Every line of the package sources, package data, tests and project documents is scanned for CJK characters. The
only exceptions are compatibility identifiers listed in ``ALLOWED``: values that already exist in machines, traces
or task files produced by earlier versions and are still recognized.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CJK_RUN = re.compile(r"[⺀-⿟　-〿぀-ヿ㄀-ㄯ㆐-ㇿ㐀-䶿"
                     r"一-鿿가-힯豈-﫿︰-﹏＀-￯]+")

#: path (relative to the repository root) -> CJK strings that may appear in it, with the reason
ALLOWED: dict[str, dict[str, str]] = {
    "src/hexis/machine/schema.py": {
        "弃权": "abstain label written by earlier versions; still accepted",
    },
    "src/hexis/traces/trace_adapter.py": {
        "结束": "label prefix produced by the legacy labelers and stored in existing machines",
        "解答有错": "legacy labeler label stored in existing machines",
        "核验写错了": "legacy labeler label stored in existing machines",
    },
    "src/hexis/legacy/compiler.py": {
        "判据": "clause keyword recognized in skill documents written in Chinese",
        "规范": "clause keyword recognized in skill documents written in Chinese",
    },
    "src/hexis/legacy/checker.py": {
        "弱": "provenance origin value stored in existing provenance files",
        "待标定": "provenance origin value stored in existing provenance files",
        "已标定": "provenance origin value stored in existing provenance files",
    },
    "tests/test_34_doc_judge.py": {
        "解答有错": "asserts the legacy labeler labels",
        "核验写错了": "asserts the legacy labeler labels",
    },
    "tests/test_abstain_compat.py": {
        "弃权": "machines written by earlier versions keep working",
    },
}

TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".txt", ".cff", ".in"}
PROJECT_FILES = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md",
                 "CITATION.cff", "pyproject.toml", "MANIFEST.in", ".env.example"]


def _files() -> list[pathlib.Path]:
    out = []
    for base in (ROOT / "src" / "hexis", ROOT / "tests", ROOT / ".github"):
        if base.is_dir():
            out += [p for p in base.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES
                    and "__pycache__" not in p.parts]
    out += [ROOT / f for f in PROJECT_FILES if (ROOT / f).is_file()]
    this = pathlib.Path(__file__).resolve()
    return sorted(p for p in out if p.resolve() != this)


def test_no_cjk_outside_the_compatibility_allowlist():
    problems = []
    for path in _files():
        rel = path.relative_to(ROOT).as_posix()
        allowed = ALLOWED.get(rel, {})
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for run in CJK_RUN.findall(line):
                if not any(run in a for a in allowed):
                    problems.append(f"{rel}:{lineno}: {line.strip()[:120]}")
                    break
    assert not problems, "non-English text:\n" + "\n".join(problems[:50])


@pytest.mark.parametrize("rel", sorted(ALLOWED))
def test_allowlist_entries_are_still_needed(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    for literal in ALLOWED[rel]:
        assert literal in text, f"{rel}: allowlisted text {literal!r} no longer appears; remove the entry"
