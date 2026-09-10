"""Command-line interface: ``skill2fsm <command> [options]``.

Each command lives in its own module with a ``main(argv, prog)`` entry point, so a command only
imports what it needs.
"""
from __future__ import annotations

import importlib
import sys

from .. import __version__

#: command -> (module, one-line summary)
COMMANDS: dict[str, tuple[str, str]] = {
    "compile": ("skill2fsm.cli.compile",
                "Build a machine from a skill document and traces (initialization + trace update)."),
    "compile-stepwise": ("skill2fsm.cli.compile_stepwise",
                         "Update a machine trace by trace with externally supplied step decisions."),
    "run": ("skill2fsm.cli.run", "Execute a machine on one task."),
    "collect": ("skill2fsm.cli.collect", "Collect skill-execution traces with OpenCode on spreadsheet tasks."),
    "fold-traces": ("skill2fsm.cli.fold_traces", "Fold the OpenCode event streams of a benchmark run into traces."),
    "bench": ("skill2fsm.cli.bench", "Run machine and skill-execution arms on a task set in parallel."),
    "memory": ("skill2fsm.cli.memory", "Build AWM workflows and ReasoningBank memories from trajectories."),
    "summarize": ("skill2fsm.cli.summarize", "Compare arms on a shared task set: pass rate, cost, paired tests."),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: skill2fsm <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {summary}" for name, (_module, summary) in COMMANDS.items()]
    lines += ["", "Run `skill2fsm <command> --help` for the options of a command."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    if args[0] in ("-V", "--version"):
        print(f"skill2fsm {__version__}")
        return 0
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"skill2fsm: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    return int(module.main(rest, prog=f"skill2fsm {name}") or 0)


__all__ = ["COMMANDS", "main", "usage"]
