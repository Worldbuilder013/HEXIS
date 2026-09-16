"""DABench closed-form answer grading: ``@name[value]`` pairs in the output text are checked against a reference (name, value) list.

Numbers get a tolerance derived from the number of decimal places of the reference value (reference 34.65 →
|diff| ≤ 0.005 + 1e-9); integer references still allow a 1e-6 floating-point error. Non-numeric values are compared
after removing quotes and surrounding whitespace, case-insensitively. Comma-separated list values are compared item by
item with the same rules. Every item of the reference must be found in the output with the same name and a matching
value.
"""
from __future__ import annotations

import pathlib
import re

_HEAD = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)\s*\[")


def _strip_list(v: str) -> str:
    """Strip the outer square brackets and quotes of a list literal: '[]' → '', '[1, 2]' → '1, 2', \"['a']\" → 'a'."""
    v = v.strip()
    while v.startswith("[") and v.endswith("]"):
        v = v[1:-1].strip()
    if v.startswith("[") and "]" not in v:      # half list left by an upstream cut at the first ]: '[1, 2, 3'
        v = v[1:].strip()
    return ", ".join(x.strip().strip("'\"") for x in v.split(",")) if v else ""


def parse_pairs(text: str) -> dict[str, list[str]]:
    """Take the value of each @name[...] by bracket depth, so nested list literals are extracted whole."""
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
    if expected in ("[", "") and got in ("[", ""):      # empty reference lists are sometimes stored as '[' (upstream cut at the first ])
        return True
    if "," in expected.strip() and "," in got.strip():
        es, gs = [x for x in expected.split(",")], [x for x in got.split(",")]
        return len(es) == len(gs) and all(_match(a.strip(), b.strip()) for a, b in zip(es, gs))
    if ":" in expected and ":" in got:            # dict item 'month_3': 5.9 -- keys compared as text, values as numbers
        ek, ev = expected.split(":", 1); gk, gv = got.split(":", 1)
        return _norm(ek.strip("{} ")) == _norm(gk.strip("{} ")) and _match(ev.strip().rstrip("}"), gv.strip().rstrip("}"))
    e, g = _num(expected), _num(got)
    if e is not None and g is not None:
        if e != e or g != g:                       # nan: they agree only if both sides are nan
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
        return False, f"no @name[value] found: {text.strip()[:60]!r}"
    bad = []
    for name, val in expected:
        cands = pairs.get(str(name).lower())
        if not cands:
            bad.append(f"missing @{name}[...]")
        elif not any(_match(str(val), c) for c in cands):
            bad.append(f"@{name}: reference {val}, got {cands[-1]}")
    return (not bad), ("; ".join(bad)[:200] if bad else "")


def grade_answer_file(path: pathlib.Path, expected: list) -> tuple[bool, str]:
    path = pathlib.Path(path)
    if not path.is_file():
        return False, "no answer.txt produced"
    return grade_text(path.read_text(encoding="utf-8", errors="replace"), expected)


__all__ = ["parse_pairs", "grade_text", "grade_answer_file"]
