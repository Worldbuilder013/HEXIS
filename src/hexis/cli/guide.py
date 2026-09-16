"""Write a usage guide (GUIDE.md) and an agent prompt (PROMPT.md) for a compiled machine.

GUIDE.md explains the machine to people: inputs, tools, a diagram, every state and transition, limits, fallback
behaviour and how to run it. PROMPT.md is a system prompt that lets a tool-using agent execute the machine step by
step without the hexis runtime. PROMPT.md contains the whole machine, including prompts derived from the skill
document; the runtime never shows a model more than the prompt of the current state.

    hexis-agent guide --build BUILD_DIR
    hexis-agent guide --machine machine.json --skill SKILL_DIR --tools tools.json --out DOCS_DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib

from hexis.guide import write_docs
from hexis.machine.schema import load_machine
from hexis.skill_loader import load_agent_skill
from hexis.tools.toolspec import load_registry


def _read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def generate(*, machine_path: pathlib.Path, out: pathlib.Path, skill_dir: pathlib.Path | None = None,
             tools_path: pathlib.Path | None = None, embed_skill: bool = False, retries: int = 3,
             manifest: dict | None = None, progress: dict | None = None,
             clause_map: dict | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    m = load_machine(machine_path)
    name = desc = ""
    doc = None
    if skill_dir is not None and (skill_dir / "SKILL.md").is_file():
        skill = load_agent_skill(skill_dir)
        name = skill.name or skill.slug
        desc = skill.description or ""
        doc = skill.body
    if manifest:
        name = ((manifest.get("skill") or {}).get("name")) or name
        desc = ((manifest.get("skill") or {}).get("description")) or desc
    tools = load_registry(tools_path) if tools_path is not None and tools_path.is_file() else {}
    return write_docs(out, m, tools=tools, skill_name=name, skill_description=desc, skill_doc=doc,
                      embed_skill=embed_skill, retries=retries, manifest=manifest, progress=progress,
                      clause_map=clause_map)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--build", default=None, help="build directory written by `compile` or `update`")
    src.add_argument("--machine", default=None, help="machine JSON file, or a directory containing machine.json")
    ap.add_argument("--skill", default=None, help="skill directory (name, description and document); "
                                                  "default for --build: BUILD/skill")
    ap.add_argument("--tools", default=None, help="tool registry JSON (tool descriptions); default for --build: BUILD/tools.json")
    ap.add_argument("--out", default=None, help="output directory (default: the build directory, or the machine's directory)")
    ap.add_argument("--embed-skill", action="store_true",
                    help="append the skill document to PROMPT.md so the agent can finish the task after fallback")
    ap.add_argument("--retries", type=int, default=3, help="retries at the fallback state described in PROMPT.md")
    a = ap.parse_args(argv)

    manifest = progress = clause_map = None
    if a.build:
        build = pathlib.Path(a.build)
        if not (build / "machine.json").is_file():
            ap.error(f"{build} has no machine.json")
        machine_path = build / "machine.json"
        skill_dir = pathlib.Path(a.skill) if a.skill else build / "skill"
        tools_path = pathlib.Path(a.tools) if a.tools else build / "tools.json"
        out = pathlib.Path(a.out) if a.out else build
        manifest = _read_json(build / "build.json")
        progress = _read_json(build / "progress.json")
        init_log = _read_json(build / "init_log.json") or {}
        clause_map = init_log.get("clause_map") or None
    else:
        machine_path = pathlib.Path(a.machine)
        if not machine_path.exists():
            ap.error(f"{machine_path} does not exist")
        skill_dir = pathlib.Path(a.skill) if a.skill else None
        tools_path = pathlib.Path(a.tools) if a.tools else None
        out = pathlib.Path(a.out) if a.out else (machine_path if machine_path.is_dir() else machine_path.parent)
    guide, prompt = generate(machine_path=machine_path, out=out, skill_dir=skill_dir, tools_path=tools_path,
                             embed_skill=a.embed_skill, retries=a.retries, manifest=manifest, progress=progress,
                             clause_map=clause_map)
    print(f"wrote {guide} and {prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
