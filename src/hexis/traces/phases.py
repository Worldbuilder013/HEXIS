"""Phase classification: give steps that use "the same generic tool for different purposes" a decidable identity.

**Why this layer has to exist.** :func:`hexis.traces.normalize.canon_action` deliberately keeps
tool arguments out of the state identity -- "arguments live in variables and are not part of the
state identity". That rule is right for specialised tools like ``math_verify`` whose **name is
their purpose**; it collapses for **generic** tools like ``bash`` / ``run_python``: looking up
the table, writing the result and checking it back all fold into one
``('tool', 'name=run_python')``, the whole trace becomes one state looping on itself three times,
and no structure can be learned. All the information is in the command text, and by design the
command text does not enter the KEY.

**The three phases are skill-agnostic.** Every skill that "produces something" has the same
shape::

    probe   read the inputs, see the current state   -- does not touch the output
    apply   write the output                         -- has a write operation
    verify  read back what was just written          -- mentions the output but does not write

The only skill-specific part is **how to recognise a "write"**, and even that needs no per-skill
rules: write operations in Python and shell come in only a handful of shapes
(:data:`WRITE_SIGNALS`), and which file is the output is **declared by the task itself**
(``output_path`` and the like, recognised from the task input by :func:`outputs_of`). So the
default classifier :func:`default_phase` works directly for any skill; there is no need to write
one per skill.

If a skill really needs its own criteria (domain-specific write operations), register a function
with the same signature via :func:`register`; :attr:`hexis.machine.schema.Machine.phase_rules`
records which set was used.

**The criteria are pure functions, not a model.** Anyone can rerun them against the bodies in
``artifacts/<sha>.py`` to check.

**Classified at collection time, not at compile time.**
:func:`hexis.traces.trace_adapter._extract_code` moves code bodies out to artifacts and leaves
only ``code_sha256`` in the record -- by compile time the body can no longer be read. So
``to_trace`` computes the phase **before** the body is moved out and writes it into
``action["phase"]``, which travels with the record. Machine states carry ``phase`` as well (a
field of :class:`~hexis.machine.schema.ToolAction`), so ``canon_action`` symmetrically includes it
in the KEY on both sides, and replay needs no special handling -- unlike ``pred=``
(backward-looking context), which only applies on the compile side: the phase is a property of
**the action itself** and can be computed on both sides.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

#: Phase values. ``other`` is the fallback (empty or unrecognizable command) -- it is still a
#: decidable value; only the **empty string** means "this step does not take part in phase
#: refinement", with behaviour identical to before.
PHASE_PROBE = "probe"
PHASE_APPLY = "apply"
PHASE_VERIFY = "verify"
PHASE_OTHER = "other"

PHASES = (PHASE_PROBE, PHASE_APPLY, PHASE_VERIFY, PHASE_OTHER)

#: Argument keys the command body may live under. Aligned with trace_adapter.CODE_KEYS.
TEXT_KEYS = ("code", "command", "script", "cmd", "source", "argv")

#: **Generic** tools: the name says nothing about the purpose, so refinement is required.
#: For specialised tools (``math_verify``, ``apply_formula``) the ``name`` already is the
#: identity; they do not take part in refinement.
GENERIC_TOOLS = frozenset({
    "bash", "sh", "shell", "zsh", "run_python", "python", "python3",
    "run_script", "exec", "run", "execute", "run_command", "run_code",
    "file_ops", "file", "files", "fs", "filesystem",
})

#: Tools like ``file_ops`` have no command body; the purpose is in the ``op``/``action``/``mode`` key.
_OP_KEYS = ("op", "operation", "action", "mode", "command")
_WRITE_OPS = frozenset({"write", "append", "create", "delete", "remove", "move", "rename",
                        "copy", "mkdir", "touch", "save", "put", "edit", "replace", "insert"})
_READ_OPS = frozenset({"read", "list", "ls", "stat", "exists", "search", "grep", "find",
                       "cat", "head", "tail", "get", "open", "view"})

#: Shapes of "this command writes something". A few each for Python and shell, independent of any
#: particular skill. Every entry is a form actually observed; better to accept one too many than to
#: misclassify a write as a read -- that would leave the graph without an apply.
WRITE_SIGNALS: tuple[str, ...] = (
    ".save(", ".save (", ".write(", ".writelines(", ".to_csv(", ".to_excel(",
    ".to_json(", ".to_parquet(", "json.dump(", "pickle.dump(", "yaml.dump(",
    ".dump(", ".commit(", ".flush(",
    "write_text(", "write_bytes(", "makedirs(", "mkdir(",
    "shutil.copy", "shutil.move", "os.replace(", "os.rename(", "os.remove(",
)

#: Write operations in shell (matched on word boundaries, so ``cpu`` does not match ``cp``).
_SHELL_WRITE_RE = re.compile(
    r"(^|[;&|]|\s)(cp|mv|rm|tee|touch|mkdir|install|dd|git\s+commit|sed\s+-i)\s|>>?\s*\S")

#: ``open(..., "w")`` / ``'a'`` / ``'x'``: the second argument has a write mode.
_OPEN_WRITE_RE = re.compile(r"""open\s*\([^)]*,\s*['"][^'"]*[wax]""")

#: Key-name hints in the task input for "this is the output". The output is **declared by the
#: task**, not guessed from file names by the classifier.
_OUTPUT_KEY_HINTS = ("output", "out_path", "outpath", "dst", "dest", "target", "_out")

CLASSIFIERS: dict[str, Callable[..., str]] = {}


def register(name: str) -> Callable:
    """Register a classifier. Like prohibitions and labelers, it is a **reviewed pure function** that never calls a model.

    Signature ``(tool_name, input_dict, *, outputs=()) -> str``, returning one of :data:`PHASES`
    or the empty string.
    """
    def deco(fn: Callable[..., str]):
        CLASSIFIERS[name] = fn
        return fn
    return deco


def command_text(inp: Mapping) -> str:
    """Assemble the command body to inspect from the tool input. Empty string if there is none.

    **Does not read ``*_sha256`` / ``*_path``** -- those are fingerprints left behind after the body
    was moved out; classifying a phase from them would be guessing content from a hash.
    """
    if not isinstance(inp, Mapping):
        return ""
    parts: list[str] = []
    for k in TEXT_KEYS:
        v = inp.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
        elif isinstance(v, (list, tuple)) and v:
            parts.append(" ".join(str(x) for x in v))
    return "\n".join(parts)


def outputs_of(task_input: Mapping) -> tuple[str, ...]:
    """Recognise "what the output is" from the task input: the values of keys containing output/dst/target etc., plus the key names themselves.

    The output is **declared by the task**, not guessed by the classifier. Key names are included
    because executors often inject the path into the code as a variable (``OUTPUT_XLSX``), so the
    command mentions the variable name rather than the path.
    """
    out: list[str] = []
    if not isinstance(task_input, Mapping):
        return ()
    for k, v in task_input.items():
        kl = str(k).lower()
        if any(h in kl for h in _OUTPUT_KEY_HINTS):
            out.append(str(k))
            if isinstance(v, str) and v.strip():
                out.append(v)
    return tuple(dict.fromkeys(out))


def writes_something(command: str) -> bool:
    """Whether this command has a write operation. Skill-agnostic; looks only at the shape."""
    c = command or ""
    if any(sig in c for sig in WRITE_SIGNALS):
        return True
    if _OPEN_WRITE_RE.search(c):
        return True
    return bool(_SHELL_WRITE_RE.search(c))


def mentions(command: str, needles: Sequence[str]) -> bool:
    c = (command or "").lower()
    return any(str(n).lower() in c for n in needles if str(n).strip())


@register("default")
def default_phase(tool: str, inp: Mapping, *, outputs: Sequence[str] = ()) -> str:
    """Generic three phases. **Usable directly by any skill**; there is no need to write one per skill.

    Criteria from most to least specific; the order must not be reversed:

    1. has a write operation => **apply**;
    2. no write, but mentions the output declared by the task => **verify** (reading back what was
       just written);
    3. any other non-empty command => **probe**.

    Checking for writes before checking for the output matters: the write step almost always
    mentions the output path (``wb.save(OUTPUT)``); checking the other way round would misclassify
    every write as a verify, and the graph would then have only verify and no apply.

    When ``outputs`` is empty (the skill produces no file, e.g. a math problem only submits an
    answer string), rule 2 naturally drops out and classification degrades to a write / read
    split -- still decidable, just coarser.
    """
    if (tool or "").lower() not in GENERIC_TOOLS:
        return ""                                   # specialised tool: the name already is the identity
    c = command_text(inp)
    if not c.strip():
        return _phase_by_op(inp, outputs)
    if writes_something(c):
        return PHASE_APPLY
    if outputs and mentions(c, outputs):
        return PHASE_VERIFY
    return PHASE_PROBE


def _phase_by_op(inp: Mapping, outputs: Sequence[str]) -> str:
    """Generic tools without a command body (``file_ops``): classify by ``op`` and path. Same criteria as above, read differently."""
    op = ""
    for k in _OP_KEYS:
        v = inp.get(k) if isinstance(inp, Mapping) else None
        if isinstance(v, str) and v.strip():
            op = v.strip().lower()
            break
    if not op:
        return PHASE_OTHER
    if op in _WRITE_OPS:
        return PHASE_APPLY
    if op in _READ_OPS:
        paths = " ".join(str(v) for k, v in inp.items()
                         if isinstance(v, str) and k not in _OP_KEYS)
        return PHASE_VERIFY if (outputs and mentions(paths, outputs)) else PHASE_PROBE
    return PHASE_OTHER


def classify(tool: str, inp: Mapping, rules: str = "", *,
             outputs: Sequence[str] = ()) -> str:
    """Classify the phase with the classifier named by ``rules``. Empty or unregistered ``rules`` => empty string (no refinement)."""
    fn = CLASSIFIERS.get(rules or "")
    if fn is None:
        return ""
    try:
        p = fn(str(tool or ""), inp if isinstance(inp, Mapping) else {}, outputs=tuple(outputs))
    except TypeError:                                # older classifier that does not accept outputs
        try:
            p = fn(str(tool or ""), inp if isinstance(inp, Mapping) else {})
        except Exception:                            # noqa: BLE001
            return ""
    except Exception:                                # noqa: BLE001
        return ""
    return p if p in PHASES else ""


def phase_of(action: Any) -> str:
    """Read the recorded phase from an action (a bare dict in a record or an Action model in a machine)."""
    if isinstance(action, Mapping):
        return str(action.get("phase") or "")
    return str(getattr(action, "phase", "") or "")


__all__ = ["CLASSIFIERS", "GENERIC_TOOLS", "PHASES", "PHASE_APPLY", "PHASE_OTHER",
           "PHASE_PROBE", "PHASE_VERIFY", "TEXT_KEYS", "WRITE_SIGNALS", "classify",
           "command_text", "default_phase", "mentions", "outputs_of", "phase_of",
           "register", "writes_something"]
