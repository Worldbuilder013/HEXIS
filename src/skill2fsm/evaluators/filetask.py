"""文件答案题（DABench / SealQA 这类）的通用工具：把题目资产放进作业目录、按 metadata.verifier 判分。

任务 yaml 的 metadata 约定：
  assets: [相对 root 的文件路径, ...]   → 逐个复制到作业目录（保留文件名）
  assets_dir / assets_target: 目录 → 复制成作业目录下的 assets_target 子目录
  verifier: dabench | sealqa_judge
  answers（dabench）: [[name, value], ...]；answer（sealqa）: 参考答案文本
"""
from __future__ import annotations

import pathlib
import shutil

def stage_assets(meta: dict, dest: pathlib.Path, root: str | pathlib.Path = ".") -> dict:
    """复制资产（路径相对 ``root``），返回机器可用的输入变量：data_path（第一个文件）、docs_dir（目录）。"""
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
    """→ (passed | None, why, source)。sealqa_judge 由人（模型判官）事后判分，这里只记 None。"""
    kind = str(meta.get("verifier") or "")
    answer_path = pathlib.Path(answer_path)
    if kind == "dabench":
        from .dabench_answer import grade_answer_file, grade_text, parse_pairs
        if answer_path.is_file():
            ok, why = grade_answer_file(answer_path, meta["answers"])
            return ok, why, "answer.txt"
        if reply_text and parse_pairs(reply_text):
            ok, why = grade_text(reply_text, meta["answers"])
            return ok, why, "reply_text"
        return False, "没有产出 answer.txt", "none"
    if kind == "sealqa_judge":
        if answer_path.is_file():
            return None, "待判官判分", "answer.txt"
        if reply_text and reply_text.strip():
            return None, "待判官判分（取自回复正文）", "reply_text"
        return False, "没有产出 answer.txt", "none"
    raise ValueError(f"不认识的 verifier: {kind!r}")


__all__ = ["stage_assets", "grade"]
