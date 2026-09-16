"""Shared helpers for file-answer tasks (DABench / SealQA and the like).

Puts the task assets into the working directory and grades according to metadata.verifier.

Conventions for metadata in the task yaml:
  assets: [file paths relative to root, ...]   → each copied into the working directory (file name kept)
  assets_dir / assets_target: directory → copied as the assets_target subdirectory of the working directory
  verifier: dabench | sealqa_judge
  answers (dabench): [[name, value], ...]; answer (sealqa): reference answer text
"""
from __future__ import annotations

import pathlib
import shutil


def stage_assets(meta: dict, dest: pathlib.Path, root: str | pathlib.Path = ".") -> dict:
    """Copy the assets (paths relative to ``root``) and return machine input variables: data_path (first file), docs_dir."""
    base = pathlib.Path(root)
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    inputs: dict = {}
    for rel in meta.get("assets") or []:
        src = base / str(rel)
        shutil.copy2(src, dest / src.name)
        inputs.setdefault("data_path", str(dest / src.name))
    if meta.get("assets_dir"):
        src = base / str(meta["assets_dir"])
        target = dest / str(meta.get("assets_target") or "docs")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
        inputs["docs_dir"] = str(target)
    return inputs


def grade(meta: dict, answer_path: pathlib.Path, reply_text: str | None = None) -> tuple:
    """→ (passed | None, why, source). sealqa_judge is graded later by a (model) judge, so only None is recorded here."""
    kind = str(meta.get("verifier") or "")
    answer_path = pathlib.Path(answer_path)
    if kind == "dabench":
        from hexis.evaluators.dabench_answer import grade_answer_file, grade_text, parse_pairs
        if answer_path.is_file():
            ok, why = grade_answer_file(answer_path, meta["answers"])
            return ok, why, "answer.txt"
        if reply_text and parse_pairs(reply_text):
            ok, why = grade_text(reply_text, meta["answers"])
            return ok, why, "reply_text"
        return False, "no answer.txt produced", "none"
    if kind == "sealqa_judge":
        if answer_path.is_file():
            return None, "awaiting grading by the judge", "answer.txt"
        if reply_text and reply_text.strip():
            return None, "awaiting grading by the judge (taken from the reply text)", "reply_text"
        return False, "no answer.txt produced", "none"
    raise ValueError(f"unknown verifier: {kind!r}")


__all__ = ["stage_assets", "grade"]
