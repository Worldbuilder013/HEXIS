"""DABench 闭合式答案判分：产出文本里的 ``@name[value]`` 与参考 (name, value) 列表逐项比对。

数值按参考值的小数位数取容差（参考 34.65 → |差| ≤ 0.005 + 1e-9），整数参考也允许 1e-6 的浮点误差；
非数值按去引号、去首尾空白、大小写不敏感比较。逗号分隔的列表值逐项按同样规则比较。参考里的每一项都必须在产出里找到同名且相符的值。
"""
from __future__ import annotations

import pathlib
import re

_HEAD = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)\s*\[")


def _strip_list(v: str) -> str:
    """列表字面量去掉外层方括号与引号：'[]' → ''，'[1, 2]' → '1, 2'，\"['a']\" → 'a'。"""
    v = v.strip()
    while v.startswith("[") and v.endswith("]"):
        v = v[1:-1].strip()
    if v.startswith("[") and "]" not in v:      # 上游参考按第一个 ] 截断留下的半个列表：'[1, 2, 3'
        v = v[1:].strip()
    return ", ".join(x.strip().strip("'\"") for x in v.split(",")) if v else ""


def parse_pairs(text: str) -> dict[str, list[str]]:
    """按方括号深度取 @name[...] 的值，嵌套的列表字面量也能整段取出。"""
    out: dict[str, list[str]] = {}
    text = text or ""
    for m in _HEAD.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"[": 1, "]": -1}.get(text[i], 0)
            i += 1
        if depth:
            continue
        out.setdefault(m.group(1).lower(), []).append(_strip_list(text[m.end():i - 1]))
    return out


def _num(s: str):
    s = s.strip().strip("'\"").replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _match(expected: str, got: str) -> bool:
    expected, got = _strip_list(expected), _strip_list(got)
    if expected in ("[", "") and got in ("[", ""):      # 参考里空列表有时记成 '['（上游按第一个 ] 截断所致）
        return True
    if "," in expected.strip() and "," in got.strip():
        es, gs = [x for x in expected.split(",")], [x for x in got.split(",")]
        return len(es) == len(gs) and all(_match(a.strip(), b.strip()) for a, b in zip(es, gs))
    if ":" in expected and ":" in got:            # 字典项 'month_3': 5.9 —— 键按文本、值按数值比较
        ek, ev = expected.split(":", 1); gk, gv = got.split(":", 1)
        return _norm(ek.strip("{} ")) == _norm(gk.strip("{} ")) and _match(ev.strip().rstrip("}"), gv.strip().rstrip("}"))
    e, g = _num(expected), _num(got)
    if e is not None and g is not None:
        if e != e or g != g:                       # nan：两边都是 nan 才算一致
            return (e != e) and (g != g)
        dec = len(expected.split(".")[1]) if "." in expected.strip() else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9 if dec else 1e-6
        return abs(e - g) <= tol
    return _norm(expected) == _norm(got)


def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", x.strip().strip("'\"")).lower()


def grade_text(text: str, expected: list) -> tuple[bool, str]:
    pairs = parse_pairs(text)
    if not pairs:
        return False, f"没有找到 @name[value]：{text.strip()[:60]!r}"
    bad = []
    for name, val in expected:
        cands = pairs.get(str(name).lower())
        if not cands:
            bad.append(f"缺少 @{name}[...]")
        elif not any(_match(str(val), c) for c in cands):
            bad.append(f"@{name}: 参考 {val}，产出 {cands[-1]}")
    return (not bad), ("; ".join(bad)[:200] if bad else "")


def grade_answer_file(path: pathlib.Path, expected: list) -> tuple[bool, str]:
    path = pathlib.Path(path)
    if not path.is_file():
        return False, "没有产出 answer.txt"
    return grade_text(path.read_text(encoding="utf-8", errors="replace"), expected)


__all__ = ["parse_pairs", "grade_text", "grade_answer_file"]
