"""Execution backends: ``describe_tools()`` reports tool definitions, ``call(name, arguments)`` executes.

The compiler does not import this package. It only reads tool registries (files like ``opencode.json``) or the
:class:`~hexis.tools.toolspec.ToolSpec` objects a backend reports; at run time the entry script connects the
machine to a concrete backend. Every tool name that appears in the machine must be executable by the backend;
a missing one is an error, and it is never replaced by another tool.
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
    """Get a registry by backend name (``opencode`` -> ``backends/opencode.json``)."""
    p = REGISTRIES.get(name)
    if p is None:
        raise KeyError(f"no tool registry named {name!r}; available: {sorted(REGISTRIES)}")
    return load_registry(p)


__all__ = ["ExecutionBackend", "REGISTRIES", "REGISTRY_DIR", "registry"]
