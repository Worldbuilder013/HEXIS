"""Action normalization: fold a trace record (or a machine action) into a comparable **KEY**.

This is the least conspicuous part of the whole compilation, and the easiest one to get quietly
wrong. Two key decisions are built on top of it:

* **Compilation** has to decide "is this step the same step as that one" -- same => reconnect to
  the old state (forming loops, merging ≈-equivalent histories; Myhill-Nerode state minimality
  depends on it), different => open a new state. Fold too much, and two semantically different
  branches are merged into one state; fold too little, and every step is unique, so the machine
  degenerates into a straight line with no loops.
* **Replay** has to decide "is the step the machine picked the step the trace actually took".
  Folding too much wrongly reports a successful reproduction; folding too little makes every
  machine unable to reproduce any trace.

The two decisions need **deliberately different** notions of sameness:

* strict (compilation) -- tools are merged by name only, but judge actions
  also compare ``prompt`` and end actions compare ``terminal``. Compilation must be this strict:
  two judges asking different questions are two different semantic branches, and merging them
  would treat "is the header canonical" and "are the amounts right" as the same step.
* loose (replay) -- compares only ``kind``, plus the name
  for tools. Replay must be this loose: action arguments in a trace are **concrete values**
  (rendered prompt, filled-in input), while in the machine they are **templates** (``${var}``),
  so a literal comparison never matches; and the prompt carries the problem statement, so
  putting it into the KEY would make every step unique.

So this module does **not unify them**; it folds both into two named modes of one
function: ``strict=True`` is the compilation side, ``strict=False`` (the default) is the replay
side. Which mode to use is a semantic question, not an implementation detail, hence it is in the
signature.

Rules (``kind`` always participates):

===========  ==========================================  ==============================
kind         loose (replay side)                         strict (compile side, adds)
===========  ==========================================  ==============================
``tool``     kind + normalized tool name                 same (tools don't split on args)
``judge``    kind + writes                               + ``prompt``
``model``    kind + writes                               + ``prompt``
``user``     kind + writes                               + ``prompt``
``end``      kind + ``terminal``                         same (both modes compare terminal)
===========  ==========================================  ==============================

How the two modes differ from the two existing implementations: loose is **slightly finer** than
``_action_matches`` -- it additionally compares the terminal of ``end`` and the writes of
judge/model. This is intentional: an ``end``'s terminal is "which way the run ended", and merging
ok with give_up into one step breaks the rejection-set exclusion; writes is "which variable this
step writes to", which determines what later conditions can read, so it belongs to state identity
rather than to arguments. On real traces (one terminal per end state, one variable per question)
the two produce **exactly the same grouping** -- test_14 cross-checks this pair by pair on the
table_clean records, so that pointing replay here later does not change behaviour. Likewise,
strict is slightly finer than ``_sig`` (judge/model additionally compare writes).

``canon_action`` accepts both :class:`~hexis.machine.schema.Record` (whose ``.action`` is a **bare
dict**) and the schema's Action models (ToolAction/ModelAction/JudgeAction/UserAction/EndAction)
-- the compiler holds models, traces hold dicts, and the same step must fold into the same KEY,
otherwise every comparison between the two sides is wrong. Recognition is duck-typed and **does
not import** :mod:`hexis.machine.schema`: this module therefore has no dependencies and no import
cycles, so either side can import it freely.

⚠️ A pitfall: the ``writes`` of judge/model have **no** dedicated field in trace records (see
runtime._run_action: a judge records ``{kind,prompt,reads}``, a model records
``{kind,template_id,prompt,reads,prompt_sha256}``), so they can only be inferred from the keys of
``Record.output`` (a judge's output is exactly ``{written variable: label}``; for other actions,
status keys such as ``ok``/``error`` are removed -- the same convention as
compiler._infer_writes). So **pass the whole Record, not just ``rec.action``**: a bare action
dict has no output, and the judge's writes degrade to empty (strict then happens to fall back to
the behaviour of ``_sig`` -- no worse, but no more precise either).

Same-source comparability: a judge's ``prompt`` is present in traces, so the **strict mode** of
tool/judge/end still matches across sources (record x machine action). ``model`` joined this group
as of that runtime revision: ``runtime._model_action`` records the **raw template** (not the
rendered text) into ``rec.action["prompt"]``, so model records produced by the runtime also match
the machine's ``ModelAction`` in strict mode. But this only holds for records **that recorded a
prompt** -- ``user`` actions still record only ``{kind}`` (the runtime refuses to execute user
actions), and old or external traces may lack the field too; the strict key of such records
degrades to ``("model"/"user", "writes=…")``, which can never equal a machine action. When unsure
in a cross-source comparison, use the loose mode -- which is exactly what replay does. test_14 pins
this boundary so that it is not mistaken for an accident later.

The KEY is ``tuple[str, ...]``: hashable, ``json.dumps``-able, stable across processes -- no
:func:`hash`, no :func:`id`, no dependence on dict insertion order (everything that needs sorting
is sorted).

**No LaTeX / answer normalization here** (whether ``\\frac12`` and ``\\frac{1}{2}`` are the same
answer). That needs sympy and is equivalence of **results**, not of **actions**. This module uses
only the standard library, so that structural checks, the compiler and replay can import it
freely without taking on heavy dependencies.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

__all__ = [
    "STATUS_KEYS", "action_writes", "canon_action", "canon_output",
    "canon_tool_name", "same_action",
]

#: Reserved keys in a tool output that say "did this call succeed": they are status bits, not
#: written variables. Same convention as ``k != "ok"`` in compiler._infer_writes; ``error`` is
#: excluded here as well.
STATUS_KEYS = frozenset({"ok", "error"})

#: Reserved **begin tool** name: the step that compilation virtually prepends to every trace
#: (``kind="tool"``, no input, no output); a no-op at run time. It gives a machine exactly one
#: starting point (``initial`` is always this state), and the real first step becomes an ordinary
#: branch after it -- on a math skill, the ``start_mismatch`` caused by 10 of 54 accepted traces
#: (T+) having a different first action drops to zero this way. The name must be a fixed point of
#: :func:`canon_tool_name` (leading and trailing underscores are stripped, so it is not called
#: ``__begin__``) and must never collide with a skill's own tools.
BEGIN_TOOL = "skill2fsm_begin"

#: Characters treated as separators in tool names: hyphens and any whitespace.
_SEP_RE = re.compile(r"[-\s]+")
_UNDERSCORE_RE = re.compile(r"_+")

#: Action kinds that carry writes (+ a natural-language field in strict mode only) -> the name of
#: the field strict mode additionally compares.
_TEXT_FIELD = {"judge": "prompt", "model": "prompt", "user": "prompt"}


# --------------------------------------------------------------------------- #
# Tool names
# --------------------------------------------------------------------------- #
def canon_tool_name(name: str) -> str:
    """Fold a tool name into its canonical form: drop directories and ``.py``, lower-case, ``-``/whitespace -> ``_``.

    ``scripts/math_verify.py``, ``math-verify`` and ``MATH_VERIFY`` all fold into ``math_verify``.
    This is not pedantry: the same script written as a path in the docs, as a bare name in a
    trace, and with hyphens by the model is the normal case; without folding them together, one
    s3 (check) state splits into three states that never form a loop, and the compiled machine
    is useless.

    Runs of ``_`` collapse into one, leading/trailing ``_`` are removed (``math - verify`` and
    ``math__verify`` likewise converge to ``math_verify``). An empty name returns an empty string.
    """
    s = str(name or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/").rsplit("/", 1)[-1]     # keep only the last segment (drop directories)
    s = s.lower()
    if s.endswith(".py"):                           # already lower-cased, so .PY is stripped here too
        s = s[:-3]
    s = _SEP_RE.sub("_", s)
    return _UNDERSCORE_RE.sub("_", s).strip("_")


# --------------------------------------------------------------------------- #
# Unpacking: Record / bare record dict / bare action dict / Action model -> (action mapping, output mapping)
# --------------------------------------------------------------------------- #
def _model_fields(act: Any) -> dict:
    """Take the fields that participate in the KEY from an Action model (duck-typed, no schema import)."""
    d: dict = {"kind": getattr(act, "kind", "")}
    for f in ("name", "terminal", "prompt", "prompt", "phase"):
        v = getattr(act, f, None)
        if v is not None:
            d[f] = v
    w = getattr(act, "writes", None)
    if w is not None:
        d["writes"] = list(w)
    return d


def _unwrap(obj: Any) -> tuple[Mapping, Mapping]:
    """Normalize into ``(action mapping, output mapping)``. The output is non-empty only when a whole record is available."""
    act = getattr(obj, "action", None)                       # Record (pydantic model)
    if isinstance(act, Mapping):
        out = getattr(obj, "output", None)
        return act, out if isinstance(out, Mapping) else {}
    if isinstance(obj, Mapping):
        inner = obj.get("action")
        if isinstance(inner, Mapping):                       # bare record read straight from JSONL
            out = obj.get("output")
            return inner, out if isinstance(out, Mapping) else {}
        return obj, {}                                       # bare action dict: no output
    if getattr(obj, "kind", None) is not None:               # schema Action model
        return _model_fields(obj), {}
    raise TypeError(f"unrecognized action carrier {type(obj).__name__} (expected Record / dict / Action model)")


# --------------------------------------------------------------------------- #
# writes: half of the state identity (which variables this step writes to)
# --------------------------------------------------------------------------- #
def _writes_of(act: Mapping, out: Mapping) -> list[str]:
    """Use the declared writes if present; otherwise (the usual case for trace records) infer them from the output keys. Sorted and deduplicated."""
    declared = act.get("writes")
    if isinstance(declared, (list, tuple)):
        return sorted({str(w) for w in declared})
    kind = str(act.get("kind") or "")
    keys = {str(k) for k in out}
    if kind != "judge":                     # a judge's output is exactly {written variable: label}; keep all
        keys -= STATUS_KEYS
    return sorted(keys)


def action_writes(rec_or_action: Any) -> list[str]:
    """Names of the variables this step writes (sorted, deduplicated). Pass a Record to infer undeclared writes from its output."""
    act, out = _unwrap(rec_or_action)
    return _writes_of(act, out)


# --------------------------------------------------------------------------- #
# Action KEY
# --------------------------------------------------------------------------- #
def canon_action(rec_or_action: Any, *, strict: bool = False) -> tuple[str, ...]:
    """Fold one action into a KEY. ``strict=False`` is the replay mode, ``True`` the compile mode (differences in the module docs).

    Returns a tuple of ``str`` only: hashable, JSON-serializable, stable across processes. An
    unknown kind returns just ``(kind,)`` -- the same fallback as the strict signature; no
    structure is invented.
    """
    act, out = _unwrap(rec_or_action)
    kind = str(act.get("kind") or "")
    if kind == "tool":
        # Tools don't split on arguments: arguments live in variables and are not part of the
        # state identity. Same behaviour in both modes.
        # The only exception is the **recorded phase** (act["phase"]): generic tools like
        # ``bash``/``run_python`` share a name but serve different purposes, and without refinement
        # they fold into one state. The phase is determined at collection time by the pure
        # functions in hexis.traces.phases, and machine states carry the same field -- **both sides
        # are symmetric**, so it can enter the shared equivalence relation, unlike ``pred=``
        # (backward-looking context), which only applies on the compile side. Actions without a
        # phase behave exactly as before.
        key = ("tool", "name=" + canon_tool_name(act.get("name") or ""))
        ph = str(act.get("phase") or "")
        return key + (("phase=" + ph,) if ph else ())
    if kind == "end":
        # terminal is "which way the run ended"; both modes compare it: merging it breaks the
        # rejection-set exclusion check.
        return ("end", "terminal=" + str(act.get("terminal") or "done"))
    if kind in _TEXT_FIELD:
        key = [kind, "writes=" + ",".join(_writes_of(act, out))]
        if strict:
            # Only the compile mode includes the natural-language field: the prompt carries the
            # problem statement, and with it in the KEY every step is unique.
            field = _TEXT_FIELD[kind]
            key.append(f"{field}=" + str(act.get(field) or ""))
        return tuple(key)
    return (kind,)


def is_begin(rec_or_action: Any) -> bool:
    """Whether this step is the reserved begin tool :data:`BEGIN_TOOL`."""
    act, _out = _unwrap(rec_or_action)
    return (str(act.get("kind") or "") == "tool"
            and canon_tool_name(act.get("name") or "") == BEGIN_TOOL)


def context_key(rec_or_action: Any, preds: Sequence[Any] = (), *,
                k: int = 1) -> tuple[str, ...]:
    """Strict KEY + the loose KEYs of at most ``k`` **predecessors** -- the compile-side state identity, used only for candidate lookup.

    Rule: **identity may look backward, never forward.** The same action under different
    predecessor contexts can be different states (Myhill-Nerode: those two histories are in fact
    not equivalent -- on a math skill, folding every ``math_verify`` call into one state with 4
    successors that cannot be told apart is exactly what ``k=0`` produces); but splitting by
    successors would require knowing the future, which is the job of conditions and judges, not
    of identity.

    Replay **does not look** at this KEY: it only compares actions (loose mode), and a cloned state's
    action is a deep copy, so identity refinement is transparent to replay.
    ``k=0`` degenerates to ``canon_action(strict=True)``.
    """
    base = canon_action(rec_or_action, strict=True)
    if k <= 0 or not preds:
        return base
    tail = tuple("pred=" + "|".join(canon_action(p, strict=False))
                 for p in list(preds)[-k:])
    return base + tail


def same_action(a: Any, b: Any, *, strict: bool = False) -> bool:
    """Whether two actions are "the same step" in the given mode. One side may be a Record and the other an Action model."""
    return canon_action(a, strict=strict) == canon_action(b, strict=strict)


# --------------------------------------------------------------------------- #
# Output trimming
# --------------------------------------------------------------------------- #
def canon_output(out: Mapping, writes: Sequence[str]) -> dict:
    """Trim one output by the ``writes`` allow-list: undeclared keys are dropped, declared keys that are present are kept.

    "Collect outputs by the writes allow-list" is established runtime discipline
    (runtime.rebuild); this is the **offline** version of the same thing, used to remove noise such
    as ``ok`` before comparing two outputs. Keys are put back sorted by name, so the result of
    ``json.dumps`` is independent of the original insertion order of the two dicts. Empty
    ``writes`` => an empty dict (nothing declared, nothing collected).
    """
    src = out or {}
    return {k: src[k] for k in sorted({str(w) for w in (writes or ())}) if k in src}
