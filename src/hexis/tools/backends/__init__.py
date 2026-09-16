"""执行后端：``describe_tools()`` 报工具定义，``call(name, arguments)`` 执行。

编译器不 import 这个包。它只读工具注册表（``opencode.json`` 这类文件）或后端报出来的
:class:`~hexis.tools.toolspec.ToolSpec`；运行时由入口脚本把机器接到一个具体后端上。
机器里出现的工具名后端必须能执行，缺了就报错，不会换成别的工具。
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from hexis.tools.toolspec import ToolSpec, load_registry

REGISTRY_DIR = Path(__file__).resolve().parent
REGISTRIES = {p.stem: p for p in REGISTRY_DIR.glob("*.json")}


@runtime_checkable
class ExecutionBackend(Protocol):
    def describe_tools(self) -> dict[str, ToolSpec]: ...
    def call(self, name: str, arguments: dict) -> dict: ...


def registry(name: str) -> dict[str, ToolSpec]:
    """按后端名取注册表（``opencode`` → ``backends/opencode.json``）。"""
    p = REGISTRIES.get(name)
    if p is None:
        raise KeyError(f"没有名为 {name!r} 的工具注册表；可用：{sorted(REGISTRIES)}")
    return load_registry(p)


__all__ = ["ExecutionBackend", "REGISTRIES", "REGISTRY_DIR", "registry"]
