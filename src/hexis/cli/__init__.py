"""Command-line interface: ``hexis-agent <command> [options]``.

Each command lives in its own module with a ``main(argv, prog)`` entry point, so a command only
imports what it needs.
"""
from __future__ import annotations

import importlib
import sys

from hexis import __version__

#: command -> (module, one-line summary)
COMMANDS: dict[str, tuple[str, str]] = {
    "compile": ("hexis.cli.compile",
                "Build a machine from a skill document and traces (initialization + trace update)."),
    "update": ("hexis.cli.update",
               "Update a compiled machine with new traces; a model decides every trace step."),
    "compile-stepwise": ("hexis.cli.compile_stepwise",
                         "Update a machine trace by trace with externally supplied step decisions."),
    "guide": ("hexis.cli.guide", "Write GUIDE.md and PROMPT.md (a prompt for agents) for a compiled machine."),
    "run": ("hexis.cli.run", "Execute a machine on one task."),
    "collect": ("hexis.cli.collect", "Collect skill-execution traces with OpenCode on spreadsheet tasks."),
    "fold-traces": ("hexis.cli.fold_traces", "Fold the OpenCode event streams of a benchmark run into traces."),
    "bench": ("hexis.cli.bench", "Run machine and skill-execution arms on a task set in parallel."),
    "memory": ("hexis.cli.memory", "Build AWM workflows and ReasoningBank memories from trajectories."),
    "summarize": ("hexis.cli.summarize", "Compare arms on a shared task set: pass rate, cost, paired tests."),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: hexis-agent <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {summary}" for name, (_module, summary) in COMMANDS.items()]
    lines += ["", "Run `hexis-agent <command> --help` for the options of a command."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    if args[0] in ("-V", "--version"):
        print(f"hexis-agent {__version__}")
        return 0
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"hexis-agent: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    return int(module.main(rest, prog=f"hexis-agent {name}") or 0)


__all__ = ["COMMANDS", "main", "usage"]
