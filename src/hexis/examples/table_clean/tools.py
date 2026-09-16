"""Three deterministic fake tools for the table_clean toy skill, running on an in-memory file system.

The tools are purely functional: an ``input`` dict goes in and an ``output`` dict comes out, the
same input always gives the same output, and they never touch the real disk or the network and run
in milliseconds. A "file" is a ``dict[path -> {header, rows}]`` (:class:`MemFS`) pre-populated by
the task generator. The tools **execute faithfully**: ``export`` writes even when the target is the
source file; the prohibition "never overwrite the source file" is enforced on the judging side
(prohibition), not inside the tool. This division of labor is intentional: tools only do the work,
and discipline is kept by the machine's structure and by judging.
"""

from __future__ import annotations

from typing import Optional

from hexis.llm.model_iface import ToolRegistry

BAD_MARK = "Unnamed"


def is_canonical(header_row: str) -> bool:
    """Well-formedness criterion (matches SKILL.md S2.1): comma-separated, every field non-empty, no ``Unnamed``."""
    if not header_row:
        return False
    fields = header_row.split(",")
    return all(f.strip() and BAD_MARK not in f for f in fields)


def _fix_once(header_row: str) -> str:
    """Rename the **first** bad field (empty or containing Unnamed) to ``column{i}``. One fix at a
    time: a header with more bad fields needs more fixes, so the repair loop has a real iteration
    count (1..k) instead of finishing in one step. Deterministic: the same bad header is always
    fixed the same way.
    """
    fields = header_row.split(",")
    for i, f in enumerate(fields):
        if not f.strip() or BAD_MARK in f:
            fields[i] = f"column{i + 1}"
            break
    return ",".join(fields)


class MemFS:
    """In-memory file system: ``path -> {'header': [...], 'rows': [...]}``."""

    def __init__(self, files: Optional[dict] = None):
        self.files: dict[str, dict] = {}
        for path, tbl in (files or {}).items():
            self.files[path] = {"header": list(tbl["header"]),
                                "rows": [list(r) for r in tbl.get("rows", [])]}

    def snapshot(self) -> dict:
        """Deep-copy the current state, so acceptance can check whether the source file was touched."""
        return {p: {"header": list(t["header"]),
                    "rows": [list(r) for r in t["rows"]]}
                for p, t in self.files.items()}


def build_registry(fs: MemFS) -> ToolRegistry:
    """Bind the three tools to a :class:`MemFS` and return a :class:`ToolRegistry` that can be fed straight to runtime."""

    def read_csv(inp: dict) -> dict:
        path = inp["path"]
        if path not in fs.files:
            return {"error": f"file not found: {path}"}
        tbl = fs.files[path]
        return {"ok": True, "header_row": ",".join(tbl["header"]),
                "rows": [list(r) for r in tbl["rows"]]}

    def fix_header(inp: dict) -> dict:
        return {"ok": True, "header_row": _fix_once(inp["header_row"])}

    def export(inp: dict) -> dict:
        """Write header_row + rows to output_path. **Executes faithfully**, even if that overwrites source_path."""
        out = inp["output_path"]
        header = str(inp.get("header_row", "")).split(",")
        rows = inp.get("rows", [])
        fs.files[out] = {"header": header, "rows": [list(r) for r in rows]}
        return {"ok": True, "output_path": out}

    reg = ToolRegistry()
    reg.add("read_csv", read_csv)
    reg.add("fix_header", fix_header)
    reg.add("export", export)
    return reg
