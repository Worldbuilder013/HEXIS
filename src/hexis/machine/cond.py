"""Syntax, evaluation and static analysis of transition guards.

A guard (``Transition.if``) is a **simple predicate over variables**: equality, comparison, emptiness tests on
collections, and boolean and/or/not. It is deliberately weak: weak enough to be decidable. That buys two things:

* **Determinism is checkable.** Checking that the guards on a state's out-edges are pairwise mutually exclusive and
  together exhaustive relies on compressing "infinitely many variable values" into "finitely many configurations" and
  enumerating them one by one. The compression is based on the atomic predicates extracted from the guards
  (:func:`atoms_of`): a numeric variable can change the truth of an edge only at the few thresholds it is compared
  against, so one representative value between thresholds is enough. See compiler.check.
* **Safety.** Evaluation goes through an allowlisting AST visitor, with **no eval/exec anywhere**. Only the node
  types accepted by ``_check`` below are let through; anything else is a :class:`CondError`. Guard expressions are
  part of the machine.json syntax, and a guard in an untrusted machine file cannot break out of the sandbox either.

An undefined variable raises :class:`CondError` during evaluation (it is not treated as false). This is the run-time
safety net for "write before read": if a path reads a variable before it is written, it fails right here instead of
quietly taking the wrong edge.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Union


class CondError(ValueError):
    """The guard expression is invalid, or its evaluation hit an undefined variable."""


#: Allowlisted predicate functions: applied to collections/strings, they test for empty/non-empty.
_PREDICATES = {
    "empty": lambda x: x is None or (hasattr(x, "__len__") and len(x) == 0),
    "nonempty": lambda x: x is not None and hasattr(x, "__len__") and len(x) > 0,
}

_CMP_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


@dataclass(frozen=True)
class Atom:
    """An atomic predicate. ``op`` is a comparison operator name or ``empty``/``nonempty`` (then ``const`` is None)."""

    var: str
    op: str
    const: Any = None


# --------------------------------------------------------------------------- #
# Parsing + allowlist validation (cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1024)
def parse(expr: str) -> ast.Expression:
    """Parse into an AST and check every node against the allowlist. Invalid (including empty) → :class:`CondError`."""
    if not expr or not expr.strip():
        raise CondError("empty guard (a default edge should never get here; default edges are detected by cond=='')")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CondError(f"guard syntax error: {expr!r} ({exc.msg})") from exc
    _check(tree)
    return tree


def _check(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _check(node.body)
    elif isinstance(node, ast.BoolOp):                     # and / or
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise CondError("only and/or are allowed")
        for v in node.values:
            _check(v)
    elif isinstance(node, ast.UnaryOp):                    # not
        if not isinstance(node.op, ast.Not):
            raise CondError("the only unary operator allowed is not")
        _check(node.operand)
    elif isinstance(node, ast.Compare):                    # ==, !=, <, in ...
        for op in node.ops:
            if type(op) not in _CMP_OPS:
                raise CondError(f"comparison operator not allowed: {type(op).__name__}")
        _check(node.left)
        for c in node.comparators:
            _check(c)
    elif isinstance(node, ast.Call):                       # empty(x) / nonempty(x)
        if not (isinstance(node.func, ast.Name) and node.func.id in _PREDICATES):
            raise CondError("only empty(x) / nonempty(x) are allowed")
        if len(node.args) != 1 or node.keywords:
            raise CondError(f"{node.func.id} takes exactly one positional argument")
        _check(node.args[0])
    elif isinstance(node, (ast.List, ast.Tuple)):          # [a,b] for in
        for e in node.elts:
            _check(e)
    elif isinstance(node, ast.Name):                       # variable
        return
    elif isinstance(node, ast.Constant):                   # constant
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            raise CondError(f"constant type not allowed: {type(node.value).__name__}")
    else:
        raise CondError(f"syntax node not allowed: {type(node).__name__}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(expr: str, env: dict) -> bool:
    """Evaluate under the variable values ``env`` and return a bool. Undefined variable → :class:`CondError`."""
    tree = parse(expr)
    return bool(_eval(tree.body, env))


def _eval(node: ast.AST, env: dict) -> Any:
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval(v, env) for v in node.values)
        return any(_eval(v, env) for v in node.values)
    if isinstance(node, ast.UnaryOp):                      # not
        return not _eval(node.operand, env)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            if not _CMP_OPS[type(op)](left, right):
                return False
            left = right                                   # supports chained a<b<c
        return True
    if isinstance(node, ast.Call):                         # empty / nonempty
        return _PREDICATES[node.func.id](_eval(node.args[0], env))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, env) for e in node.elts]
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise CondError(f"guard uses undefined variable {node.id!r} (write before read violated)")
        return env[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise CondError(f"evaluation hit an unexpected node {type(node).__name__}")


# --------------------------------------------------------------------------- #
# Static analysis: for structural checks
# --------------------------------------------------------------------------- #
def vars_of(expr: str) -> set[str]:
    """All free variable names referenced by the guard (for the write-before-read check)."""
    if not expr:
        return set()
    return {n.id for n in ast.walk(parse(expr)) if isinstance(n, ast.Name)
            and n.id not in _PREDICATES}


def atoms_of(expr: str) -> list[Atom]:
    """Extract all atomic predicates (so that check can enumerate the finite configurations).

    Comparisons of the form ``var op const`` (a variable on one side, a constant on the other) become :class:`Atom`;
    ``empty(x)``/``nonempty(x)`` become ``Atom(x, 'empty'/'nonempty')``. Variable-to-variable comparisons produce no
    atom (configuration enumeration cannot cover them; they are left to judge actions or FALLBACK).
    """
    out: list[Atom] = []
    for node in ast.walk(parse(expr)):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a = _atom_from_compare(node)
            if a:
                out.append(a)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                out.append(Atom(arg.id, node.func.id))
    return out


def _atom_from_compare(node: ast.Compare) -> Any:
    left, op, right = node.left, node.ops[0], node.comparators[0]
    opname = type(op).__name__ if type(op) in _CMP_OPS else None
    if opname is None:
        return None
    if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
        return Atom(left.id, opname, right.value)
    if isinstance(right, ast.Name) and isinstance(left, ast.Constant):
        # constant on the left: flip it (a < 3 ≡ 3 > a); in/not in are not symmetric and are not flipped
        flip = {"Lt": "Gt", "LtE": "GtE", "Gt": "Lt", "GtE": "LtE",
                "Eq": "Eq", "NotEq": "NotEq"}
        if opname in flip:
            return Atom(right.id, flip[opname], left.value)
    if isinstance(left, ast.Name) and isinstance(right, (ast.List, ast.Tuple)) \
            and opname in ("In", "NotIn"):
        vals = tuple(e.value for e in right.elts if isinstance(e, ast.Constant))
        return Atom(left.id, opname, vals)
    return None


def thresholds_of(expr: str, var: str) -> list[float]:
    """The threshold constants a numeric variable is compared against in the guard (for interval partitioning)."""
    out: list[float] = []
    for a in atoms_of(expr):
        if a.var == var and a.op in ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq") \
                and isinstance(a.const, (int, float)) and not isinstance(a.const, bool):
            out.append(float(a.const))
    return sorted(set(out))
