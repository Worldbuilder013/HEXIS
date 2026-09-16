"""``interpreter`` -- an **alias module** for :mod:`hexis.execution.runtime`: one implementation, two names.

Why both names are kept: the design calls this layer the "interpreter" (``interpreter``), and code written
against that name imports ``interpreter``; the package, however, has called it ``runtime`` from day one, many
tests import ``runtime``, and existing traces and reports already record ``runtime.run_task`` as their origin.
Renaming would make the naming look tidier, at the cost of breaking existing imports and the traceability of
existing artifacts in one go -- not worth it. So this module does exactly one thing: re-export runtime's public
surface **unchanged**, so that both names resolve.

**There is no second implementation.** Nothing here adds behavior, wraps anything or changes defaults:
``interpreter.run_task is runtime.run_task`` always holds (``tests/test_26_runtime.py`` pins this). Two
implementations would drift apart, and they are one and the same interpreter -- how a machine runs, keeps its
accounts and enters FALLBACK should be decided in exactly one place.

To change behavior, change :mod:`hexis.execution.runtime`; this module follows along.
"""

from __future__ import annotations

from hexis.execution.runtime import (
    STOP_MAX_STEPS,
    STOP_STATE_ERROR,
    STOP_STUCK,
    STOP_TERMINAL,
    RunResult,
    fill_template,
    interpret_step,
    pick_edge,
    prompt_digest,
    rebuild,
    run_task,
)

__all__ = [
    "STOP_MAX_STEPS", "STOP_STATE_ERROR", "STOP_STUCK", "STOP_TERMINAL",
    "RunResult", "fill_template", "interpret_step", "pick_edge",
    "prompt_digest", "rebuild", "run_task",
]
