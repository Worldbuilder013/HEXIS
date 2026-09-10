"""跳转条件的语法、求值与静态分析。

一条跳转条件（``Transition.if``）是**变量上的简单谓词**：等值、比较、集合空判、布尔与或
非。它有意地弱——弱到可判定。这带来两样东西：

* **确定性可检查。** 同一状态各出边条件两两互斥且覆盖（定理2），靠的是把「无穷的变量
  取值」压成「有限的格局」再逐个枚举。压缩的依据就是从条件里抽出的原子谓词
  （:func:`atoms_of`）：一个数值变量只在它被比较的那几个阈值处才可能改变某条边的真假，
  阈值之间取一个代表值即可。见 compiler.check。
* **安全。** 求值走一个白名单 AST 访问器，**没有一处 eval/exec**。放行的节点就下面
  ``_ALLOWED`` 那几类，别的一律 :class:`CondError`。条件表达式是 machine.json 语法的
  一部分，一台不可信的机器文件里的条件也炸不出沙箱。

未定义变量在求值时抛 :class:`CondError`（不是当假）——「变量先写后读」的运行时兜底：一条
路径若在某变量写入前就读它，这里当场炸，而不是安静地走错边。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Union


class CondError(ValueError):
    """条件表达式非法，或求值时撞上未定义变量。"""


#: 白名单谓词函数：作用在集合/字符串上，判空/判非空。
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
    """一个原子谓词。``op`` ∈ 比较算子名 或 ``empty``/``nonempty``；后者 ``const`` 为 None。"""

    var: str
    op: str
    const: Any = None


# --------------------------------------------------------------------------- #
# 解析 + 白名单校验（缓存）
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1024)
def parse(expr: str) -> ast.Expression:
    """解析成 AST 并逐节点校验白名单。非法即 :class:`CondError`。空串非法。"""
    if not expr or not expr.strip():
        raise CondError("空条件（兜底边不该走到这里，兜底靠 cond=='' 判定）")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CondError(f"条件语法错误: {expr!r} ({exc.msg})") from exc
    _check(tree)
    return tree


def _check(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _check(node.body)
    elif isinstance(node, ast.BoolOp):                     # and / or
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise CondError("只允许 and/or")
        for v in node.values:
            _check(v)
    elif isinstance(node, ast.UnaryOp):                    # not
        if not isinstance(node.op, ast.Not):
            raise CondError("一元算子只允许 not")
        _check(node.operand)
    elif isinstance(node, ast.Compare):                    # ==, !=, <, in ...
        for op in node.ops:
            if type(op) not in _CMP_OPS:
                raise CondError(f"不允许的比较算子 {type(op).__name__}")
        _check(node.left)
        for c in node.comparators:
            _check(c)
    elif isinstance(node, ast.Call):                       # empty(x) / nonempty(x)
        if not (isinstance(node.func, ast.Name) and node.func.id in _PREDICATES):
            raise CondError("只允许 empty(x) / nonempty(x)")
        if len(node.args) != 1 or node.keywords:
            raise CondError(f"{node.func.id} 只接受一个位置参数")
        _check(node.args[0])
    elif isinstance(node, (ast.List, ast.Tuple)):          # [a,b] 供 in
        for e in node.elts:
            _check(e)
    elif isinstance(node, ast.Name):                       # 变量
        return
    elif isinstance(node, ast.Constant):                   # 常量
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            raise CondError(f"不允许的常量类型 {type(node.value).__name__}")
    else:
        raise CondError(f"不允许的语法节点 {type(node).__name__}")


# --------------------------------------------------------------------------- #
# 求值
# --------------------------------------------------------------------------- #
def evaluate(expr: str, env: dict) -> bool:
    """在变量取值 ``env`` 下求值，返回 bool。未定义变量 → :class:`CondError`。"""
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
            left = right                                   # 支持链式 a<b<c
        return True
    if isinstance(node, ast.Call):                         # empty / nonempty
        return _PREDICATES[node.func.id](_eval(node.args[0], env))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, env) for e in node.elts]
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise CondError(f"条件用到未定义变量 {node.id!r}（先写后读被破坏）")
        return env[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise CondError(f"求值撞到未预期节点 {type(node).__name__}")


# --------------------------------------------------------------------------- #
# 静态分析：供结构检查用
# --------------------------------------------------------------------------- #
def vars_of(expr: str) -> set[str]:
    """条件引用的全部自由变量名（供「先写后读」检查）。"""
    if not expr:
        return set()
    return {n.id for n in ast.walk(parse(expr)) if isinstance(n, ast.Name)
            and n.id not in _PREDICATES}


def atoms_of(expr: str) -> list[Atom]:
    """抽出全部原子谓词（供 check 枚举有限格局）。

    ``var op const`` 形式的比较（一侧是变量、另一侧是常量）落成 :class:`Atom`；
    ``empty(x)``/``nonempty(x)`` 落成 ``Atom(x, 'empty'/'nonempty')``。变量对变量的比较
    不产原子（格局枚举覆盖不了，留给判断动作或 FALLBACK）。
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
        # 常量在左：翻过来（a < 3 ≡ 3 > a）；in/not in 不对称，不翻
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
    """某数值变量在条件里被比较的阈值常量（供区间划分）。"""
    out: list[float] = []
    for a in atoms_of(expr):
        if a.var == var and a.op in ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq") \
                and isinstance(a.const, (int, float)) and not isinstance(a.const, bool):
            out.append(float(a.const))
    return sorted(set(out))
