"""table_clean: a hermetic toy skill that runs in seconds, used to self-test the whole hexis pipeline.

It reads a CSV, checks the header (repairing it one spot at a time when needed) and exports the
table, with one prohibition: "never overwrite the source file". The three tools are in-memory pure
functions, the model is a scripted stub and tasks come from a generator, so every run is offline
and reproducible.
"""

from pathlib import Path

from hexis.examples.table_clean.acceptance import verify
from hexis.examples.table_clean.scripted import (
    JUDGE_Q,
    LABELS,
    build_model,
    gen_tasks,
    interpret,
    make_judge,
    reference_machine,
)
from hexis.examples.table_clean.tools import MemFS, build_registry, is_canonical

SKILL_PATH = Path(__file__).with_name("SKILL.md")


def skill_doc() -> str:
    """The private body of SKILL.md (this is what the model reads when interpreting under FALLBACK)."""
    return SKILL_PATH.read_text(encoding="utf-8")


__all__ = [
    "JUDGE_Q", "LABELS", "MemFS", "SKILL_PATH", "build_model", "build_registry",
    "gen_tasks", "interpret", "is_canonical", "make_judge", "reference_machine",
    "skill_doc", "verify",
]
