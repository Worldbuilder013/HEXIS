"""table_clean 玩具技能的三个确定性假工具，跑在一个内存文件系统上。

工具是纯函数式的：``input`` dict 进、``output`` dict 出，同输入同输出，不碰真磁盘、不联网、
毫秒级。「文件」是一个 ``dict[path -> {header, rows}]``（:class:`MemFS`），由任务生成器
预置。工具**忠实执行**——``export`` 即便目标就是原文件也照写不误；「不得覆盖原文件」这条
禁止性要求由评判侧（prohibition）拦，不在工具里拦。这条分工是有意的：工具只管做，纪律
由机器的结构与评判来守。
"""

from __future__ import annotations

from typing import Optional

from ...model_iface import ToolRegistry

BAD_MARK = "Unnamed"


def is_canonical(header_row: str) -> bool:
    """规范判据（对应 SKILL.md S2.1）：逗号分隔、每个字段非空、不含 ``Unnamed``。"""
    if not header_row:
        return False
    fields = header_row.split(",")
    return all(f.strip() and BAD_MARK not in f for f in fields)


def _fix_once(header_row: str) -> str:
    """把**第一个**坏字段（空或含 Unnamed）改成 ``列{i}``。一次修一个：坏字段多就要多修
    几次，修复成环因此有真实的圈数（1..k），而不是一步到位。确定性：同一坏表头同一修法。
    """
    fields = header_row.split(",")
    for i, f in enumerate(fields):
        if not f.strip() or BAD_MARK in f:
            fields[i] = f"列{i + 1}"
            break
    return ",".join(fields)


class MemFS:
    """内存文件系统：``path -> {'header': [...], 'rows': [...]}``。"""

    def __init__(self, files: Optional[dict] = None):
        self.files: dict[str, dict] = {}
        for path, tbl in (files or {}).items():
            self.files[path] = {"header": list(tbl["header"]),
                                "rows": [list(r) for r in tbl.get("rows", [])]}

    def snapshot(self) -> dict:
        """深拷贝当前状态，供验收比对「原文件有没有被动过」。"""
        return {p: {"header": list(t["header"]),
                    "rows": [list(r) for r in t["rows"]]}
                for p, t in self.files.items()}


def build_registry(fs: MemFS) -> ToolRegistry:
    """把三个工具绑定到一个 :class:`MemFS`，返回可直接喂给 runtime 的 :class:`ToolRegistry`。"""

    def read_csv(inp: dict) -> dict:
        path = inp["path"]
        if path not in fs.files:
            return {"error": f"文件不存在: {path}"}
        tbl = fs.files[path]
        return {"ok": True, "header_row": ",".join(tbl["header"]),
                "rows": [list(r) for r in tbl["rows"]]}

    def fix_header(inp: dict) -> dict:
        return {"ok": True, "header_row": _fix_once(inp["header_row"])}

    def export(inp: dict) -> dict:
        """把 header_row + rows 写到 output_path。**忠实执行**——即便覆盖 source_path。"""
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
