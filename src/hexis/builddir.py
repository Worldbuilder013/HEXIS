"""The build directory: everything needed to update a compiled machine later.

``hexis-agent compile --out BUILD`` writes the machine and its reports into ``BUILD`` together with the inputs
that produced it, so that ``hexis-agent update --build BUILD --traces NEW`` can fold in new traces without the
original skill directory or trace directory:

    build.json          manifest: skill, tool registry and rules sources, task inputs, one record per run
    skill/SKILL.md      snapshot of the skill document
    tools.json          the tool registry that was used ({"tools": {...}})
    rules.json          the effective skill rules (compile.json format)
    traces/<key>.jsonl  byte copies of every trace that was used; key = <file stem>-<first 8 hex of sha256>
    progress.json       processed traces, their outcome, and the accepted (protected) traces with their anchors
    machine.json        the current machine

The context of every run is rebuilt from the stored traces plus the new ones, and every previously accepted trace
must still replay on each candidate machine. Nothing in the build directory holds an API key.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from hexis import __version__
from hexis.compiler.stepwise import fingerprint, strip_tools
from hexis.machine.schema import Machine, Trace, load_machine
from hexis.traces.trace_adapter import load_any_trace

BUILD_FORMAT = "hexis-build/1"
PROGRESS_FORMAT = "hexis-progress/1"


class BuildError(RuntimeError):
    """The build directory is missing, incomplete or inconsistent."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def machine_digest(m: Machine) -> str:
    return sha256_bytes(m.model_dump_json(by_alias=True).encode("utf-8"))


def trace_key(stem: str, sha256: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "trace"
    return f"{safe}-{sha256[:8]}"


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temporary file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def dump_machine(m: Machine, path: Path) -> None:
    write_json(path, json.loads(m.model_dump_json(by_alias=True)))


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


@dataclass
class TraceItem:
    key: str
    sha256: str
    file: str            # path inside the build directory
    source: str          # path it was read from
    task_id: str
    trace: Trace


def load_trace_file(path: Path, ignore_tools: Sequence[str] = ()) -> Trace:
    """Read a trace the way the compiler reads trace directories, then drop harness marker tools."""
    t = load_any_trace(path, phase_rules="")
    if ignore_tools:
        strip_tools(t, list(ignore_tools))
    return t


class BuildDir:
    def __init__(self, root: Path, manifest: dict):
        self.root = Path(root)
        self.manifest = manifest

    # ---- creation / opening ---- #
    @classmethod
    def is_build(cls, root: Path) -> bool:
        return (Path(root) / "build.json").is_file()

    @classmethod
    def create(cls, root: Path, *, skill_dir: Path, skill: Any, tools: dict, tools_source: str,
               rules: Optional[dict], rules_source: str, task_inputs: Sequence[str] = (),
               ignore_tools: Sequence[str] = ()) -> "BuildDir":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        md = Path(skill_dir) / "SKILL.md"
        raw = md.read_bytes()
        (root / "skill").mkdir(exist_ok=True)
        (root / "skill" / "SKILL.md").write_bytes(raw)
        from hexis.tools.toolspec import registry_dict
        write_json(root / "tools.json", registry_dict(tools))
        if rules is not None:
            write_json(root / "rules.json", rules)
        manifest = {"format": BUILD_FORMAT, "hexis_version": __version__,
                    "skill": {"slug": skill.slug, "name": skill.name, "description": skill.description,
                              "sha256": sha256_bytes(raw), "source": str(skill_dir)},
                    "tools": {"source": tools_source}, "rules": {"source": rules_source},
                    "task_inputs": list(task_inputs), "ignore_tools": list(ignore_tools),
                    "guide": {"embed_skill": False, "retries": 3}, "runs": []}
        bd = cls(root, manifest)
        bd.save_manifest()
        return bd

    @classmethod
    def open(cls, root: Path) -> "BuildDir":
        root = Path(root)
        data = read_json(root / "build.json")
        if not isinstance(data, dict) or data.get("format") != BUILD_FORMAT:
            raise BuildError(f"{root} is not a hexis build directory (no {BUILD_FORMAT} build.json); "
                             "create one with `hexis-agent compile --out`")
        for need in ("machine.json", "skill/SKILL.md"):
            if not (root / need).is_file():
                raise BuildError(f"{root} is incomplete: {need} is missing")
        return cls(root, data)

    def save_manifest(self) -> None:
        write_json(self.root / "build.json", self.manifest)

    # ---- inputs ---- #
    @property
    def skill_dir(self) -> Path:
        return self.root / "skill"

    @property
    def ignore_tools(self) -> list:
        return list(self.manifest.get("ignore_tools") or [])

    def registry(self) -> dict:
        from hexis.tools.toolspec import load_registry
        p = self.root / "tools.json"
        return load_registry(p) if p.is_file() else {}

    def rules(self) -> Optional[dict]:
        from hexis.compiler.context import load_rules
        p = self.root / "rules.json"
        return load_rules(p) if p.is_file() else None

    def load_machine(self) -> Machine:
        return load_machine(self.root / "machine.json")

    # ---- progress ---- #
    def load_progress(self) -> dict:
        data = read_json(self.root / "progress.json")
        if data is None:
            return {"format": PROGRESS_FORMAT, "traces": [], "entries": [], "accepted": []}
        if data.get("format") != PROGRESS_FORMAT:
            raise BuildError(f"{self.root / 'progress.json'} has an unknown format")
        return data

    def save_progress(self, machine: Machine, progress: dict) -> None:
        progress["format"] = PROGRESS_FORMAT
        progress["machine_sha256"] = machine_digest(machine)
        progress["fingerprint"] = fingerprint(machine)
        progress["counts"] = counts(progress)
        dump_machine(machine, self.root / "machine.json")
        write_json(self.root / "progress.json", progress)

    def machine_changed_outside(self, machine: Machine, progress: dict) -> bool:
        want = progress.get("machine_sha256")
        return bool(want) and want != machine_digest(machine)

    # ---- traces ---- #
    def stored_traces(self, progress: dict) -> dict[str, TraceItem]:
        out: dict[str, TraceItem] = {}
        for rec in progress.get("traces") or []:
            p = self.root / rec["file"]
            if not p.is_file():
                raise BuildError(f"stored trace {rec['key']} is missing ({p})")
            out[rec["key"]] = TraceItem(key=rec["key"], sha256=rec["sha256"], file=rec["file"],
                                        source=rec.get("source", ""), task_id=rec.get("task_id", ""),
                                        trace=load_trace_file(p, self.ignore_tools))
        return out

    def stage(self, files: Iterable[Path], progress: dict, *, run: int, say=print) -> tuple[list[TraceItem], dict]:
        """Copy new trace files into the build directory. Files already staged (same content) are skipped."""
        known = {rec["sha256"]: rec for rec in progress.get("traces") or []}
        keys = {rec["key"] for rec in progress.get("traces") or []}
        added: list[TraceItem] = []
        stats = {"given": 0, "staged": 0, "duplicates": 0, "unreadable": 0}
        (self.root / "traces").mkdir(exist_ok=True)
        for f in files:
            f = Path(f)
            stats["given"] += 1
            raw = f.read_bytes()
            digest = sha256_bytes(raw)
            if digest in known:
                stats["duplicates"] += 1
                continue
            try:
                trace = load_trace_file(f, self.ignore_tools)
            except Exception as exc:                                  # noqa: BLE001
                stats["unreadable"] += 1
                say(f"unreadable trace {f.name}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            key = trace_key(f.stem, digest)
            n = 2
            while key in keys:
                key = f"{trace_key(f.stem, digest)}_{n}"
                n += 1
            rel = f"traces/{key}{f.suffix or '.jsonl'}"
            shutil.copyfile(f, self.root / rel)
            task_id = str((trace.task or {}).get("task_id") or "") if isinstance(trace.task, dict) else ""
            rec = {"key": key, "sha256": digest, "file": rel, "source": str(f), "task_id": task_id, "staged_run": run}
            progress.setdefault("traces", []).append(rec)
            known[digest] = rec
            keys.add(key)
            added.append(TraceItem(key=key, sha256=digest, file=rel, source=str(f), task_id=task_id, trace=trace))
            stats["staged"] += 1
        return added, stats

    # ---- runs ---- #
    def start_run(self, command: str, **info: Any) -> dict:
        run = {"n": len(self.manifest.get("runs") or []) + 1, "command": command, "started": now(),
               "status": "running", **info}
        self.manifest.setdefault("runs", []).append(run)
        self.save_manifest()
        return run

    def finish_run(self, run: dict, status: str = "ok", **info: Any) -> None:
        run.update(info)
        run["status"] = status
        run["finished"] = now()
        self.save_manifest()


def counts(progress: dict) -> dict:
    out: dict = {}
    for e in progress.get("entries") or []:
        out[e.get("status", "")] = out.get(e.get("status", ""), 0) + 1
    return out


def trace_files(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("*.jsonl"))


__all__ = ["BUILD_FORMAT", "BuildDir", "BuildError", "PROGRESS_FORMAT", "TraceItem", "counts", "dump_machine",
           "load_trace_file", "machine_digest", "read_json", "sha256_bytes", "trace_files", "trace_key", "write_json"]
