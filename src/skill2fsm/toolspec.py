"""工具定义：编译器与执行后端共用的接口描述。

编译器把工具名当作不透明标识，只通过 :class:`ToolSpec` 了解一个工具：参数字段、产出字段、
成功判据、主要输出。定义有三个来源，按可信度记在 ``source`` 上：

* ``registry``  显式提供的工具注册表（JSON 文件），是接口保证；
* ``backend``   执行后端 ``describe_tools()`` 报出来的；
* ``inferred``  只从轨迹里观察到的：说明「出现过什么」，不是接口保证。

推断规则只有一条**状态字段惯例**（:data:`STATUS_CONVENTIONS`）：产出里有 ``returncode`` 就按
``returncode == 0`` 判成功，有 ``ok`` 就按 ``ok == True``；两者都没有就不判。这是关于字段名的
惯例，写在这里而不是编译器里，编译器只读 ``success`` 表达式。
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

#: 状态字段惯例：(字段名, 成功判据, 成功取值, 失败取值)。顺序即优先级。
STATUS_CONVENTIONS: tuple[tuple[str, str, Any, Any], ...] = (
    ("returncode", "returncode == 0", 0, 1),
    ("ok", "ok == True", True, False),
)
#: 这些字段描述调用状态而非内容，不作语义变量、不作判断状态的读取对象。
STATUS_KEYS = frozenset({"ok", "returncode", "stderr", "error", "error_text", "exists",
                         "metadata", "opencode_part"})


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)      # 字段 → {type, required?, constant?}
    output_schema: dict = field(default_factory=dict)     # 字段 → type
    success: str = ""                                     # 产出字段上的成功判据（条件表达式）
    primary: Optional[str] = None                         # 主要输出字段
    status_keys: list = field(default_factory=lambda: sorted(STATUS_KEYS))
    label: str = ""                                       # 注册表给定的固定阶段标签（可空）
    source: str = "inferred"
    observed_inputs: Counter = field(default_factory=Counter)
    observed_outputs: Counter = field(default_factory=Counter)
    calls: int = 0

    # ---- 派生 ---- #
    def input_keys(self) -> list[str]:
        keys = list(self.input_schema) + [k for k in self.observed_inputs if k not in self.input_schema]
        return list(dict.fromkeys(keys))

    def output_keys(self) -> list[str]:
        keys = list(self.output_schema) + [k for k in self.observed_outputs if k not in self.output_schema]
        return list(dict.fromkeys(keys))

    def sure_outputs(self) -> frozenset:
        """保证出现的产出字段：注册表声明的全部字段；推断时取每次调用都出现的字段。"""
        if self.source in ("registry", "backend") and self.output_schema:
            return frozenset(self.output_schema)
        if self.calls:
            return frozenset(k for k, n in self.observed_outputs.items() if n >= self.calls)
        return frozenset()

    def constant_keys(self) -> frozenset:
        return frozenset(k for k, v in self.input_schema.items()
                         if isinstance(v, Mapping) and v.get("constant"))

    def status_values(self, ok: Optional[bool]) -> dict:
        """调用成败已知时，成功判据里各字段的取值（三值护卫判定用）。未知返回空。"""
        if ok is None or not self.success:
            return {}
        out: dict = {}
        try:
            tree = ast.parse(self.success, mode="eval")
        except SyntaxError:
            return {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
                    and isinstance(node.left, ast.Name) and isinstance(node.comparators[0], ast.Constant)):
                c = node.comparators[0].value
                if ok:
                    out[node.left.id] = c
                elif isinstance(c, bool):
                    out[node.left.id] = not c
                elif isinstance(c, (int, float)):
                    out[node.left.id] = c + 1
                else:
                    out[node.left.id] = f"not-{c}"
        return out

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema,
                "output_schema": self.output_schema, "success": self.success, "primary": self.primary,
                "status_keys": list(self.status_keys), "label": self.label, "source": self.source,
                "observed_inputs": dict(self.observed_inputs),
                "observed_outputs": dict(self.observed_outputs), "calls": self.calls}

    @classmethod
    def from_dict(cls, name: str, d: Mapping, *, source: str = "registry") -> "ToolSpec":
        return cls(name=name, description=str(d.get("description") or ""),
                   input_schema=dict(d.get("input_schema") or d.get("input") or {}),
                   output_schema=dict(d.get("output_schema") or d.get("output") or {}),
                   success=str(d.get("success") or ""), primary=d.get("primary"),
                   status_keys=list(d.get("status_keys") or sorted(STATUS_KEYS)),
                   label=str(d.get("label") or ""), source=source)


def infer_success(output_keys: Iterable[str]) -> str:
    keys = set(output_keys)
    for key, expr, _s, _f in STATUS_CONVENTIONS:
        if key in keys:
            return expr
    return ""


def infer_primary(observed_outputs: Mapping[str, int]) -> Optional[str]:
    """主要输出：出现最多的非状态字段。"""
    cands = [(n, k) for k, n in observed_outputs.items() if k not in STATUS_KEYS]
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[0], x[1]))
    return cands[0][1]


def observe(spec: ToolSpec, inp: Mapping, out: Mapping) -> None:
    """把一次调用记进定义的观察计数；推断来源的定义据此补成功判据与主要输出。"""
    spec.calls += 1
    for k in (inp or {}):
        spec.observed_inputs[str(k)] += 1
    for k in (out or {}):
        spec.observed_outputs[str(k)] += 1
    if spec.source == "inferred":
        if not spec.success:
            spec.success = infer_success(spec.observed_outputs)
        spec.primary = infer_primary(spec.observed_outputs)


def load_registry(path: Any) -> dict[str, ToolSpec]:
    """读工具注册表 JSON：``{"tools": {name: {...}}}`` 或 ``{name: {...}}``。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tools = data.get("tools", data) if isinstance(data, dict) else {}
    return {name: ToolSpec.from_dict(name, d) for name, d in tools.items()}


__all__ = ["STATUS_CONVENTIONS", "STATUS_KEYS", "ToolSpec", "infer_primary", "infer_success",
           "load_registry", "observe"]
