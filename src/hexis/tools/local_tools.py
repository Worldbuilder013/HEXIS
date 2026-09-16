"""本机执行器：在作业目录里直接执行机器的原生工具调用，不经 OpenCode。

结果形状与 OpenCode 后端一致（ok / returncode / stdout / stderr），工具定义取注册表 ``backends/opencode.json``。
实现了注册表里的全部七个原生工具：``bash``（subprocess）、``read`` / ``write`` / ``edit``（文本文件）、
``list`` / ``glob`` / ``grep``（目录与检索）。相对路径一律相对作业目录解析。
``python_bin`` 所在目录会插到 PATH 最前面，命令里的 ``python3`` 因此解析到带 openpyxl 的解释器。
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from hexis.tools.backends import registry as _registry


def _ok(stdout: str = "", returncode: int = 0, stderr: str = "") -> dict:
    return {"ok": returncode == 0, "returncode": returncode, "stdout": stdout[-30000:], "stderr": stderr[-6000:]}


def _fail(msg: str, returncode: int = 1) -> dict:
    return {"ok": False, "returncode": returncode, "stdout": "", "stderr": msg[-6000:]}


class LocalTools:
    def __init__(self, workdir: Path, *, timeout_s: float = 120, python_bin: Optional[str] = None) -> None:
        self.workdir = Path(workdir).resolve()
        self.timeout_s = float(timeout_s)
        self.python_bin = python_bin
        self.calls: list[dict] = []
        self._reg = _registry("opencode")

    available_tools = frozenset({"bash", "read", "write", "edit", "list", "glob", "grep"})

    def __enter__(self) -> "LocalTools":
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def describe_tools(self) -> dict:
        return {n: self._reg[n] for n in self.available_tools if n in self._reg}

    # ---- 路径 ---- #
    def _path(self, p: Optional[str]) -> Path:
        s = str(p or "").strip()
        q = Path(s) if s else self.workdir
        return q if q.is_absolute() else (self.workdir / q)

    # ---- 调用 ---- #
    def call(self, name: str, inp: dict) -> dict:
        inp = dict(inp or {})
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            out = _fail(f"unknown tool: {name}", 127)
        else:
            try:
                out = fn(inp)
            except Exception as exc:                                  # noqa: BLE001
                out = _fail(f"{type(exc).__name__}: {exc}")
        self.calls.append({"name": name, "input": inp, "output": {k: v for k, v in out.items() if k != "stdout"}})
        return out

    def _t_bash(self, inp: dict) -> dict:
        cmd = str(inp.get("command") or "")
        env = dict(os.environ)
        if self.python_bin:
            env["PATH"] = str(Path(self.python_bin).resolve().parent) + os.pathsep + env.get("PATH", "")
        try:
            proc = subprocess.run(cmd, shell=True, cwd=str(self.workdir), env=env, capture_output=True,
                                  text=True, timeout=self.timeout_s, executable="/bin/bash")
            return _ok(proc.stdout or "", proc.returncode, proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            return {"ok": False, "returncode": 124, "stdout": str(exc.stdout or "")[-4000:],
                    "stderr": f"timed out after {self.timeout_s:.0f}s"}

    def _t_read(self, inp: dict) -> dict:
        p = self._path(inp.get("filePath"))
        if not p.is_file():
            return _fail(f"no such file: {p}")
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _fail(f"binary file, cannot read as text: {p} ({p.stat().st_size} bytes)")
        lines = text.splitlines()
        off = int(inp.get("offset") or 0)
        lim = int(inp.get("limit") or 2000)
        chunk = lines[off:off + lim]
        body = "\n".join(f"{i + off + 1:>6}| {ln}" for i, ln in enumerate(chunk))
        return _ok(f"<path>{p}</path>\n{body}\n" + (f"(showing {len(chunk)} of {len(lines)} lines)" if len(lines) > len(chunk) else ""))

    def _t_write(self, inp: dict) -> dict:
        p = self._path(inp.get("filePath"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(inp.get("content") or ""), encoding="utf-8")
        return _ok("Wrote file successfully.")

    def _t_edit(self, inp: dict) -> dict:
        p = self._path(inp.get("filePath"))
        if not p.is_file():
            return _fail(f"no such file: {p}")
        text = p.read_text(encoding="utf-8")
        old, new = str(inp.get("oldString") or ""), str(inp.get("newString") or "")
        if old not in text:
            return _fail("oldString not found in file")
        if inp.get("replaceAll"):
            text = text.replace(old, new)
        else:
            if text.count(old) > 1:
                return _fail("oldString matches more than once; provide more context or set replaceAll")
            text = text.replace(old, new, 1)
        p.write_text(text, encoding="utf-8")
        return _ok("Edit applied.")

    def _t_list(self, inp: dict) -> dict:
        p = self._path(inp.get("path"))
        if not p.is_dir():
            return _fail(f"no such directory: {p}")
        rows = []
        for child in sorted(p.iterdir()):
            rows.append(f"{child.name}{'/' if child.is_dir() else ''}")
        return _ok(f"{p}/\n" + "\n".join(rows))

    def _t_glob(self, inp: dict) -> dict:
        p = self._path(inp.get("path"))
        pat = str(inp.get("pattern") or "*")
        hits = sorted(str(x) for x in p.glob(pat))
        return _ok("\n".join(hits) if hits else "No files found")

    def _t_grep(self, inp: dict) -> dict:
        p = self._path(inp.get("path"))
        pat = re.compile(str(inp.get("pattern") or ""))
        include = str(inp.get("include") or "*")
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, include)]
        out: list[str] = []
        for f in files[:500]:
            try:
                for i, ln in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if pat.search(ln):
                        out.append(f"{f}:{i}: {ln[:300]}")
                        if len(out) >= 200:
                            break
            except OSError:
                continue
            if len(out) >= 200:
                break
        return _ok("\n".join(out) if out else "No matches found")


__all__ = ["LocalTools"]
