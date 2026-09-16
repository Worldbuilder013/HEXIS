"""Machine interpreter: run one machine (or a FALLBACK interpretation) and produce one trace.

A run is a deterministic loop: execute the current state's action -> evaluate the transition guards in
declaration order -> the first edge that holds wins -> until a terminal or the step limit. Three things must
be right here:

**Determinism.** The same input run twice takes the same path. Transitions are evaluated in declaration order
(the default edge always last); judge actions rely on the model stub/implementation to guarantee
``temperature=0`` itself. A nondeterministic interpreter makes it impossible to tell whether a regression came
from a bad change or from an unlucky draw.

**Stop reasons must be distinguishable.** ``terminal`` (normal) / ``state_error`` (a step blew up, or a guard
evaluation hit an undefined variable) / ``stuck`` (no edge to take) / ``max_steps`` (going in circles). All four
look the same as "the result is wrong", but each calls for a completely different fix.

**A trace is "about itself".** Each :class:`~hexis.machine.schema.Record` describes how this machine moved and
which variables it read and wrote; it carries no reference answer, so the whole trace can be shown to the
compiler agent. Tool results are filled into ``output`` by the host (here), not made up by the model.

The ``FALLBACK`` state is special: reaching it switches to "the model reads the whole document + history and
executes the procedure step by step" (interpreted execution), while the trace keeps being recorded as usual.
Runs end up here when guards do not cover a case, when a judge abstains, or while compilation is still shallow
-- it keeps the machine usable at any stage of learning. The interpreted segment still runs real actions: tools,
generation (``model``), and terminals that carry a final answer are all accepted.

**Execution-side accounting is kept separately.** Per-step tokens, latency and the argv actually executed go
into ``Record.meta``; whole-run usage and the fallback position go into :class:`RunResult`. Neither goes into
``action``/``output``, so they take no part in normalization or judging and do not change state identity. The
first rule of accounting is **if it cannot be measured, say so**: when the model interface reports no usage,
leave ``None``; never fill in a number estimated from unit prices -- an estimate will be read downstream as a
measurement.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from hexis.machine import cond
from hexis.machine.schema import FALLBACK, Machine, Record, Trace
from hexis.traces import normalize as _normalize
from hexis.traces import phases as _phases

STOP_TERMINAL = "terminal"
STOP_STATE_ERROR = "state_error"
STOP_STUCK = "stuck"
STOP_MAX_STEPS = "max_steps"
STOP_FALLBACK_EXHAUSTED = "fallback_exhausted"

_COUNTER_EXIT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*(\d+)\s*$")

_VAR_RE = re.compile(r"\$\{(\w+)\}")


@dataclass
class RunResult:
    """Everything observable about one run. ``trace`` is its trace (the verdict is filled in by judging).

    The first five fields say "how it stopped"; the ones after them are **the accounting the experiment report
    aggregates**, under two rules:

    * **If it cannot be measured, say so.** ``prompt_tokens``/``completion_tokens`` being ``None`` means the
      model interface reported no usage at all for this run (:class:`~hexis.llm.model_iface.ScriptedModel`
      has no ``usage()``, and real endpoints may not report it either); it does **not** mean "spent 0 tokens".
      ``0`` only appears when no model call was made at all. A number estimated from per-token prices would be
      read downstream as a measurement, which is worse than leaving it empty. ``unmeasured_calls`` states how
      many calls in this run were not measured.
    * **Fallbacks must be locatable.** The report needs a fallback rate and a list of fallback positions:
      ``fallback_steps`` is how many steps the interpreted segment ran, ``fallback_entry`` is **which state**
      it switched in from, and ``fallback_entry_step`` is the step number of the first record of the
      interpreted segment. When the start state is the fallback state (empty machine), ``fallback_entry``
      records ``FALLBACK`` itself, meaning "interpreted throughout" rather than "fell back from somewhere".

    ``wall_s`` is local wall-clock time, which is always measurable, so it is a number, not ``None``.
    """

    trace: Trace
    stopped: str = STOP_TERMINAL
    error: str = ""
    values: dict = field(default_factory=dict)
    llm_calls: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    unmeasured_calls: int = 0
    wall_s: float = 0.0
    fallback_steps: int = 0
    fallback_entry: Optional[str] = None
    fallback_entry_step: Optional[int] = None
    #: how many retries the fallback hub made (going back to the entry of the failing step to regenerate and re-execute).
    retries: int = 0

    def path(self) -> list[str]:
        """The sequence of visited states. Determinism tests compare this, not timings."""
        return [r.state for r in self.trace.records if r.state]

    def entered_fallback(self) -> bool:
        """Whether this run entered the interpreted segment (the numerator of the fallback rate)."""
        return self.fallback_steps > 0


# --------------------------------------------------------------------------- #
# Two pieces of pure logic: extract a JSON object from model output, surface the exception line
# --------------------------------------------------------------------------- #
def _first_json_object(text: str) -> Optional[dict]:
    """Extract the first complete JSON object from a piece of text. Models rarely reply with a bare object, so salvage it rather than fail."""
    s = text or ""
    start = s.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(s[start:i + 1])
                    except ValueError:
                        break
                    return v if isinstance(v, dict) else None
        start = s.find("{", start + 1)
    return None


def _exception_first(err: str, limit: int = 400) -> str:
    """Move the actual exception line of a traceback to the front (so truncating to the first N characters does not cut it off)."""
    text = (err or "").strip()
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("Traceback"):
        return text[:limit]
    tail = lines[-1]
    frame = next((ln.strip() for ln in reversed(lines[:-1])
                  if ln.strip().startswith("File ")), "")
    head = tail + (f"  ({frame.split('/')[-1]})" if frame else "")
    return (head + "\n" + text)[:limit]


# --------------------------------------------------------------------------- #
# Execution-side accounting: usage / latency / argv / prompt digest
# --------------------------------------------------------------------------- #
#: keys read from the model interface's ``usage()``. Extra keys are ignored (other implementations may add them freely).
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "llm_calls", "unmeasured_calls")


def _usage_of(model: Any) -> Optional[dict]:
    """A snapshot of the **cumulative** usage reported by the model interface; ``None`` without ``usage()``.

    Duck typing: ``llm_client.ModelAdapter`` has ``usage()``, the test stub
    :class:`~hexis.llm.model_iface.ScriptedModel` does not -- we neither add the interface to the stub nor invent
    numbers for it; missing is missing (the consequence is that the token fields stay ``None``, see
    :func:`_step_tokens`).
    """
    fn = getattr(model, "usage", None)
    if not callable(fn):
        return None
    try:
        u = fn()
    except Exception:                                       # noqa: BLE001
        return None                                         # a failure in accounting must not affect execution
    if not isinstance(u, Mapping):
        return None
    return {k: u.get(k) for k in _USAGE_KEYS}


def _grew(before: Optional[dict], after: Optional[dict], key: str) -> Optional[int]:
    """How much a counter grew between two snapshots. Returns ``None`` (not 0) if either side is not an integer."""
    if not before or not after:
        return None
    a, b = after.get(key), before.get(key)
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    if not isinstance(a, int) or not isinstance(b, int):
        return None
    return max(0, a - b)


def _step_tokens(before: Optional[dict], after: Optional[dict],
                 calls: int) -> tuple[Optional[int], Optional[int], int]:
    """``(prompt_tokens, completion_tokens, number of unmeasured calls)`` for one interval.

    Three cases are kept apart: **no model call at all** => ``(0, 0, 0)``, which is a measurement, not an
    estimate; **the interface reports no usage** => ``(None, None, calls)``; **usage is reported, but every call
    in the interval was counted as unmeasured** => ``None`` as well -- when the endpoint gives no usage,
    ``ModelAdapter`` adds 0 to its totals, and copying that 0 would amount to inventing a number.
    """
    if calls <= 0:
        return 0, 0, 0
    if before is None or after is None:
        return None, None, calls
    unmeasured = _grew(before, after, "unmeasured_calls") or 0
    prompt = _grew(before, after, "prompt_tokens")
    completion = _grew(before, after, "completion_tokens")
    if not prompt and not completion and unmeasured >= calls:
        return None, None, unmeasured
    return prompt, completion, unmeasured


def _argv_of(tools: Any, name: str, out: Any) -> Optional[list]:
    """The argv **actually executed** in this step. Taken from the tool output if present, otherwise asked of the tool object.

    :class:`~hexis.legacy.sandbox.ExecResult` records ``command``; other tools may call it ``argv``/``cmd``. If
    none is available, return ``None`` -- the report would rather leave it empty than pass off a command line
    reconstructed from the input template as the one that really ran. This column captures exactly the
    difference between the two: ``math_verify.py`` needs ``--json`` **before** the subcommand, and the executor
    also rewrites the ``.venv/bin/python3`` prefix to the real local interpreter (a control recorded during
    harness calibration). What the template says and what actually ran are not the same line.
    """
    src = out if isinstance(out, Mapping) else {}
    for key in ("argv", "command", "cmd"):
        v = src.get(key)
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
    getter = getattr(tools, "get", None)                     # ToolRegistry.get
    if callable(getter):
        try:
            tool = getter(name)
        except Exception:                                   # noqa: BLE001
            tool = None
        for attr in ("last_argv", "last_command"):
            v = getattr(tool, attr, None)
            if isinstance(v, (list, tuple)) and v:
                return [str(x) for x in v]
    return None


def _meta(ms: float, calls: int, before: Optional[dict], after: Optional[dict], *,
          argv: Optional[list] = None) -> dict:
    """One step's execution-side accounting, stored in ``Record.meta``.

    The ``prompt_tokens``/``completion_tokens`` keys are **present on every step** (their values may be
    ``None``): the report must be able to tell "this step spent nothing" from "this step was not measured", and
    keys that come and go would leave it guessing.
    """
    prompt, completion, unmeasured = _step_tokens(before, after, calls)
    meta: dict = {"ms": round(ms, 3), "llm_calls": calls,
                  "prompt_tokens": prompt, "completion_tokens": completion}
    if unmeasured:
        meta["unmeasured_calls"] = unmeasured
    if argv:
        meta["argv"] = list(argv)
    return meta


def prompt_digest(template: str, values: Mapping) -> str:
    """A stable digest of the rendered prompt: ``sha256(template + canonical JSON of the values read)``.

    Same template + same values => same digest, stable across processes (it does not use :func:`hash`, which is
    salted per process). It is a verifiable anchor for "what exactly this step asked"; the full text does not go
    into the trace (see :func:`_model_action` for why).
    """
    payload = json.dumps({str(k): values.get(k) for k in sorted(values or {}, key=str)},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256((template + "\x00" + payload).encode("utf-8")).hexdigest()


def _model_action(template: str, reads: Any, values_read: Mapping, *,
                  template_id: str, inline_template: bool = True) -> dict:
    """What a ``model`` action looks like in the trace: **template id + template text + reads + prompt sha256**,
    **without the rendered full text**.

    Why the rendered text is not recorded: the rendering embeds the whole problem statement (MATH-500 problems
    often run to hundreds of characters), and storing a copy on every step would inflate a trace to more than
    ten times the size of the problem, while that text is **already in the trace** -- it is in ``Record.vars``,
    so recording it again is pure duplication. The compiler needs a few other things, all provided here:
    ``template_id`` + the template text give **state identity** (the strict level of ``normalize`` compares
    exactly ``prompt``, so two sides from the same source line up), ``reads`` gives variable ownership for
    "which variables this step actually consumed", and ``prompt_sha256`` makes "same template, same values =>
    the same question" verifiable -- with the digest in hand, reproducing later what a step asked only takes
    recomputing it from the template and that step's ``vars``.

    ``inline_template=False`` is used for FALLBACK: there the "template" is the whole skill document (tens of
    KB), and inlining it on every step would be even worse than recording the rendered text, so only the id and
    the digest are kept.
    """
    act: dict = {"kind": "model", "template_id": template_id}
    if inline_template:
        act["prompt"] = template
    act["reads"] = [str(r) for r in (reads or [])]
    act["prompt_sha256"] = prompt_digest(template, values_read)
    return act


def bind_outputs(raw: Any, binds: Any) -> Any:
    """Rename tool outputs according to ``ToolAction.binds``: ``{"stdout": "workbook_content"}`` makes
    ``out["stdout"]`` also appear under the name ``workbook_content``. The original key is kept, so listing both
    ``returncode`` and the semantic name in ``writes`` collects both. ``raw`` is returned unchanged when it is
    not an object or there are no binds."""
    if not binds or not isinstance(raw, dict):
        return raw
    mapped = dict(raw)
    for src, dst in dict(binds).items():
        if src in raw and dst:
            mapped[dst] = raw[src]
    return mapped


def rebuild(raw: Any, names: list[str]) -> dict:
    """**Reconstruct** an object from a whitelist of names rather than filtering -- anything undeclared is dropped, and new keys are withheld by default.

    In-machine state outputs and session-side outputs share this
    policy, so the rule lives in one place. Returns an empty dict when ``raw`` is not an object.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    return {n: raw[n] for n in names if n in raw}


# --------------------------------------------------------------------------- #
# Template filling and edge selection
# --------------------------------------------------------------------------- #
def fill_template(obj: Any, values: dict) -> Any:
    """Fill the ``${var}`` placeholders in an action's input template from the variables. A value that is exactly ``${var}`` keeps its original type."""
    if isinstance(obj, str):
        m = _VAR_RE.fullmatch(obj)
        if m:
            return values.get(m.group(1))
        return _VAR_RE.sub(lambda x: str(values.get(x.group(1), "")), obj)
    if isinstance(obj, dict):
        return {k: fill_template(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill_template(v, values) for v in obj]
    return obj


def pick_edge(machine: Machine, sid: str, values: dict):
    """Pick the next edge: in declaration order (default edge last), the first whose guard holds wins.

    Returns ``(edge, error)``. When guard evaluation hits an undefined variable an error is returned (it is not
    treated as false) -- silently taking the default edge would disguise "the predicate is broken" as "the
    computation was wrong", the hardest kind of failure to track down.
    """
    for t in machine.out_edges(sid):
        if not t.cond:
            return t, ""
        try:
            if cond.evaluate(t.cond, values):
                return t, ""
        except cond.CondError as exc:
            return None, f"guard {t.cond!r} of {sid} failed to evaluate: {exc}"
    return None, ""


# --------------------------------------------------------------------------- #
# Execute a non-FALLBACK state
# --------------------------------------------------------------------------- #
def _phase_of_call(act, inp: dict, phase_rules: str, outputs: tuple) -> dict:
    """What this step is **actually doing**: classified on the spot from the rendered input body, not copied from the state's own declaration.

    The phase is a property of the action itself; the machine side has it, so the record side must have it too
    -- ``canon_action`` includes it in the KEY at both levels, and without it replay would mismatch on the very
    first step. But its **source** cannot be the state's declaration: the state says "what this step should be",
    while the record has to capture "what this step is". Copying would make compilation circular -- the machine
    would only relearn the labels it put on itself.

    The consequence has been observed: for a state declared as ``probe``, the model wrote a ``wb.save(...)``
    command, and the record copied ``probe``; another step that only read back the output workbook and should
    have been ``verify`` was also copied as ``probe``. Three of the four steps in one trace were mislabeled, and
    those labels are exactly what the next compilation round uses as identity.

    The criterion is shared with the external-log path (:func:`hexis.traces.phases.classify`), so traces from both
    sources are comparable. When the classifier gives no answer (a dedicated tool, no registered rule), the
    state's declaration is used, as before.
    """
    if phase_rules:
        got = _phases.classify(str(getattr(act, "name", "") or ""), inp, phase_rules,
                               outputs=outputs)
        if got:
            return {"phase": got}
    declared = str(getattr(act, "phase", "") or "")
    return {"phase": declared} if declared else {}


def _run_action(state, values: dict, *, model, tools, phase_rules: str = "",
                outputs: tuple = ()) -> tuple[dict, dict, str, int]:
    """Execute the state's action; returns (action for the record, output, error, llm_calls_delta). Updates values in place.

    ``phase_rules``/``outputs`` are used to classify **what this step is actually doing** (see :func:`_phase_of_call`).
    """
    act = state.action
    kind = act.kind
    if kind == "tool":
        inp = fill_template(act.input, values)
        ph = _phase_of_call(act, inp, phase_rules, outputs)
        if _normalize.is_begin(act):
            # begin tool: a no-op. Do not consult the tool table -- the compiler inserted it, and no tool table should know it.
            return {"kind": "tool", "name": act.name, "input": {}, **ph}, {}, "", 0
        try:
            out = tools.call(act.name, inp)
        except Exception as exc:                                # noqa: BLE001
            return {"kind": "tool", "name": act.name, "input": inp, **ph}, {}, \
                _exception_first(f"{type(exc).__name__}: {exc}"), 0
        if isinstance(out, dict) and set(out) == {"error"}:
            return {"kind": "tool", "name": act.name, "input": inp, **ph}, out, \
                str(out["error"])[:400], 0
        values.update(rebuild(bind_outputs(out, getattr(act, "binds", None)), act.writes))
        return {"kind": "tool", "name": act.name, "input": inp, **ph}, out, "", 0
    if kind == "judge":
        vread = {k: values.get(k) for k in act.reads}
        try:
            label = model.classify(prompt=act.prompt, values=vread,
                                   labels=act.labels,
                                   examples=tuple(e.model_dump() for e in act.examples))
        except Exception as exc:                                # noqa: BLE001
            return {"kind": "judge", "prompt": act.prompt}, {}, \
                f"judge action call failed: {type(exc).__name__}: {exc}", 1
        values[act.writes[0]] = label
        return ({"kind": "judge", "prompt": act.prompt, "reads": act.reads},
                {act.writes[0]: label}, "", 1)
    if kind == "model":
        vread = {k: values.get(k) for k in act.reads}
        # record template id + template + reads + digest, not the rendered text (the problem is already in vars). See _model_action.
        rec_act = _model_action(act.prompt, act.reads, vread, template_id=state.id)
        calls = 0
        try:
            try:
                out = model.generate(prompt=act.prompt, values=vread)
                calls += 1
            except Exception as first:                          # noqa: BLE001
                # one generation blew up (most likely the chain of thought used up the budget and the answer is empty): ask the same
                # state again in place, which is far cheaper than going to the fallback hub and rerunning the whole machine. Only a second failure is reported as before.
                calls += 1
                if "budget" not in str(first) and "JSON" not in str(first) and "timed out" not in str(first).lower():
                    raise
                out = model.generate(prompt=act.prompt, values=vread)
                calls += 1
            got = rebuild(out, act.writes)
            if act.writes and not any(str(got.get(w) or "").strip() for w in act.writes):
                # valid JSON came back but without any declared output key (or all empty): tell the model which keys are missing and ask again.
                # This is the interpreter repairing a format slip; it does not change the state's semantics or decide the content for the model.
                out = model.generate(prompt=act.prompt + f"\n\nYour previous answer was a JSON object without the required "
                                     f"keys {list(act.writes)} (or with empty values). Return exactly one JSON object whose keys "
                                     f"are exactly {list(act.writes)}, each with non-empty content.", values=vread)
                calls += 1
        except Exception as exc:                                # noqa: BLE001
            return rec_act, {}, \
                f"generate action call failed: {type(exc).__name__}: {exc}", max(calls, 1)
        values.update(rebuild(out, act.writes))
        return rec_act, out, "", calls
    if kind == "user":
        return {"kind": "user"}, {}, "the sealed environment has no user interface (user actions are not supported)", 0
    return {"kind": kind}, {}, f"unknown action kind {kind}", 0


# --------------------------------------------------------------------------- #
# FALLBACK: interpret one step
# --------------------------------------------------------------------------- #
#: keys of an interpreted-mode reply that are "control": they say **what** this step does, not the values it **produces**.
_CONTROL_KEYS = frozenset({"kind", "name", "input", "reads", "writes", "output",
                           "values", "terminal", "prompt", "note", "thought",
                           "reason", "why"})


def _payload(action: Mapping) -> dict:
    """The values the model itself produced in an interpretation reply: ``output``/``values`` first, otherwise the keys outside the control keys.

    Both forms are accepted because models write both under the "reply with one JSON object" constraint; the
    control-key blocklist lets a flat form such as ``{"kind":"end","terminal":"done","answer":"42"}`` still yield the answer.
    """
    for key in ("output", "values"):
        v = action.get(key)
        if isinstance(v, Mapping):
            return dict(v)
    return {k: v for k, v in action.items() if k not in _CONTROL_KEYS}


def _history_item(rec: Any) -> Any:
    """One history item fed to the interpreter: **with meta removed**.

    ``meta`` is execution-side accounting (tokens, latency, argv) and has nothing to do with "what has already
    been done"; ``ModelAdapter`` truncates each history item to 800 characters, and letting the accounting eat
    into that budget would trade the actions and results that matter for host implementation details.
    """
    if hasattr(rec, "model_dump"):
        d = rec.model_dump()
    elif isinstance(rec, Mapping):
        d = dict(rec)
    else:
        return rec
    d.pop("meta", None)
    return d


def interpret_step(doc: str, values: dict, history: list, step: int, *,
                   model, tools) -> tuple[Record, bool, str]:
    """One FALLBACK step: the model reads document + history + variables and gives the next action; the host executes and records it.

    Returns ``(record, done, error)``. ``done`` means the action was end (the machine should stop). Updates values in place.

    Four kinds of action are accepted: ``tool`` (executed by the host, result filled in by the host), ``model``
    (**this generate call itself** is that generation; its output is taken directly without asking a second
    time), ``end`` (may carry a final answer), and anything else is recorded as an "unknown action kind" and
    stops. The interpreted segment runs real actions and can submit an answer so that the machine can actually
    finish the task before its structure has been compiled -- otherwise fallback would only be "giving up
    gracefully", and the first arm of the three-arm experiment would be meaningless.

    Each step's tokens/latency go into ``Record.meta`` (one interpreted step = one model call).
    """
    hist = tuple(_history_item(r) for r in history)
    vread = dict(values)                    # snapshot of the variables fed to this interpretation (the prompt digest uses it)
    reads = sorted(vread, key=str)          # an interpreted step reads the whole variable table; record that faithfully
    u0 = _usage_of(model)
    t0 = time.perf_counter()

    def _rec(action: dict, output: Optional[dict] = None, *,
             argv: Optional[list] = None) -> Record:
        """Build a record from the current values and settle this step's accounting."""
        return Record(step=step, state=FALLBACK, action=action,
                      output=dict(output or {}), vars=dict(values),
                      meta=_meta((time.perf_counter() - t0) * 1000.0, 1, u0,
                                 _usage_of(model), argv=argv))

    try:
        action = model.generate(prompt=doc, values=vread, history=hist)
    except Exception as exc:                                    # noqa: BLE001
        rec = _rec(_model_action(doc, reads, vread, template_id=FALLBACK,
                                 inline_template=False))
        return rec, False, f"FALLBACK interpretation call failed: {type(exc).__name__}: {exc}"
    kind = action.get("kind")
    if kind == "end":
        # end may carry a final answer (``{"kind":"end","terminal":"done","answer":"42"}``): the fallback segment of a
        # math machine must be able to **submit**; the answer lands in values and output and is judged as usual. Without
        # one it is an empty dict, and the record shape is exactly as before.
        answer = _payload(action)
        values.update(answer)
        rec = _rec({"kind": "end", "terminal": action.get("terminal", "done")}, answer)
        return rec, True, ""
    if kind == "tool":
        inp = fill_template(action.get("input", {}), values)
        name = action.get("name")
        try:
            out = tools.call(name, inp)
        except Exception as exc:                                # noqa: BLE001
            rec = _rec({"kind": "tool", "name": name, "input": inp})
            return rec, False, _exception_first(f"{type(exc).__name__}: {exc}")
        argv = _argv_of(tools, name, out)
        if isinstance(out, dict) and set(out) == {"error"}:
            rec = _rec({"kind": "tool", "name": name, "input": inp}, out, argv=argv)
            return rec, False, str(out["error"])[:400]
        values.update(rebuild(out, action.get("writes", [])))
        rec = _rec({"kind": "tool", "name": name, "input": inp}, out, argv=argv)
        return rec, False, ""
    if kind == "model":
        produced = _payload(action)
        declared = action.get("writes")
        # the difference from the tool branch is intentional: tool output comes from the **host** and must be collected by the
        # writes the model declared, otherwise a tool could dump anything into the variable table; model output is written by the
        # model itself, so filtering it by its own declaration buys no constraint and only turns this step into a no-op when the
        # model forgets writes (spinning all the way to the step limit). So: collect by the declaration if there is one, otherwise take everything.
        names = ([str(w) for w in declared] if isinstance(declared, (list, tuple))
                 else list(produced))
        values.update(rebuild(produced, names))
        rec = _rec(_model_action(doc, action.get("reads") or reads, vread,
                                 template_id=FALLBACK, inline_template=False),
                   produced)
        return rec, False, ""
    rec = _rec({"kind": kind})
    return rec, False, f"FALLBACK interpretation gave an unknown action kind {kind!r}"


# --------------------------------------------------------------------------- #
# Run a task
# --------------------------------------------------------------------------- #
#: sentinel for :func:`halt_at_fallback`: a state name that **does not exist**. Lowercase with underscores, so it cannot
#: collide with real state names (``s1`` / ``FALLBACK`` / ``END_*``); if it ever did, a suffix is added to avoid it.
HALT_SENTINEL = "__halt_at_fallback__"


def halt_at_fallback(machine: Machine) -> Machine:
    """Return an equivalent copy that **stops when it reaches the fallback state** instead of switching to the :func:`interpret_step` loop.

    ``run_task`` only switches to its built-in interpreted execution when ``cur == machine.fallback``; pointing
    ``fallback`` at a non-existent state name lets control flow reach the fallback state **itself** normally. Its
    action is ``end`` (for the empty machine and compiled machines alike), so the machine stops cleanly there and
    control returns to the caller.

    Three places need this, for different reasons, but none of them wants runtime to interpret on its own: arm
    three needs the fallback segment to be continued by **the same** executor as the other two arms (otherwise
    "how much the fallback segment cost" is not comparable); the conformance check wants to see how far the
    machine gets **on its own**; and ``--no-fallback`` means "stop on entering fallback". So there is only this
    one implementation.

    The original machine is left byte-for-byte untouched; if the sentinel collides with a real state, a suffix is
    added -- colliding without avoiding it would make that state the fallback state, and the machine would start
    interpreted execution right there.
    """
    name = HALT_SENTINEL
    while name in machine.states:                           # never collides in practice, but a collision would be serious
        name += "_"
    return machine.model_copy(update={"fallback": name})


def entry_of(machine: Machine, sid: str) -> str:
    """The entry of a tool state: the model state that generates its arguments (its only successor is the tool state and it writes variables of the tool's template); otherwise the state itself."""
    st = machine.states.get(sid)
    if st is None or st.action.kind != "tool":
        return sid
    need = set(_VAR_RE.findall(json.dumps(st.action.input, ensure_ascii=False)))
    for gid, g in machine.states.items():
        if g.action.kind != "model" or getattr(g.action, "observable", False):
            continue
        # the generation gate may itself carry a counter-limit exit (cnt >= K -> fallback state); that is not a successor
        succ = [t.to for t in g.transitions
                if not (t.to == machine.fallback and _COUNTER_EXIT_RE.match(t.cond or ""))]
        if succ == [sid] and need & set(g.action.writes):
            return gid
    return sid


def run_task(machine: Machine, task: dict, *, model, tools, doc: str = "",
             max_steps: Optional[int] = None, on_error: str = "stop",
             retries: int = 0, interpret: bool = True) -> RunResult:
    """Run a machine to complete one task; returns the trace and the stop reason.

    With ``on_error="fallback"``, when a step of the machine segment blows up (tool error, failed model call) the
    run no longer stops on the spot but moves to ``machine.fallback`` and the interpreted segment carries on --
    this is what real data collection needs: while the machine segment is still incompletely learned, one bad
    command should not waste the whole run; the fallback segment can still finish the task, be scored and go into
    T+. The default ``"stop"`` keeps the old semantics (sealed tests and the three-arm experiment record
    state_error with it). If the interpreted segment itself blows up, the run always stops, under both settings.

    Initial working state = the task input fields (for the FALLBACK interpretation to read) overlaid with the
    variable table's init/init_from. Reaching ``machine.fallback`` switches to interpreted mode, and
    :class:`RunResult` records **which state it fell back from and how many steps the interpreted segment ran**
    (exactly the two things the fallback rate and position list need). Per-step tokens/latency/argv go into
    ``Record.meta`` and whole-run usage into :class:`RunResult` -- if it cannot be measured, it stays ``None``.

    **Fallback is not the end.** With ``retries > 0`` the fallback state first acts as a **retry hub**: on each
    entry it goes back to the entry of the most recently executed tool state (to the generation gate if there is
    one, so the arguments are regenerated) and resets the counter variable that triggered the fallback, at most
    ``retries`` times; only when the retries are used up does it hand over to the interpreted segment
    (``interpret=True``) or stop (``stopped="fallback_exhausted"``). With ``on_error="fallback"`` state errors go
    through this hub too.
    """
    task_input = task.get("input", {})
    task_outputs = _phases.outputs_of(task_input)     # outputs are declared by the task, not guessed by the classifier
    values: dict = dict(task_input)
    values.update(machine.initial_values(task_input))
    records: list[Record] = []
    limit = int(max_steps or machine.max_steps or 24)
    llm_calls = 0
    cur = machine.initial
    step = 0
    t_run = time.perf_counter()
    usage0 = _usage_of(model)
    fallback_steps = 0
    errors: list[str] = []
    # the start is the fallback state (empty machine): there is no "where it fell back from"; record itself to mean "interpreted throughout".
    fallback_entry: Optional[str] = (machine.initial
                                     if machine.initial == machine.fallback else None)
    fallback_entry_step: Optional[int] = None
    retry_used = 0
    last_tool: Optional[str] = None            # most recently executed tool state
    came_from: Optional[str] = None            # which state, and which edge, this entry into the fallback state came from
    came_edge = None

    def result(stopped: str, error: str = "") -> RunResult:
        trace = Trace(task=task, verdict="unknown", records=records)
        p_tok, c_tok, unmeasured = _step_tokens(usage0, _usage_of(model), llm_calls)
        return RunResult(trace=trace, stopped=stopped, error=error,
                         values=values, llm_calls=llm_calls,
                         prompt_tokens=p_tok, completion_tokens=c_tok,
                         unmeasured_calls=unmeasured,
                         wall_s=round(time.perf_counter() - t_run, 6),
                         fallback_steps=fallback_steps,
                         fallback_entry=fallback_entry,
                         fallback_entry_step=fallback_entry_step,
                         retries=retry_used)

    while step < limit:
        # ---- FALLBACK: a retry hub first; once the retries are used up, interpret / stop ---- #
        if cur == machine.fallback and records and retry_used < retries:
            retry_used += 1
            frm = last_tool or came_from or machine.initial
            reset: list[str] = []
            if came_edge is not None:
                mm = _COUNTER_EXIT_RE.match(came_edge.cond or "")
                if mm and mm.group(1) in values:
                    values[mm.group(1)] = 0
                    reset.append(mm.group(1))
            target = entry_of(machine, frm)
            step += 1
            records.append(Record(step=step, state=machine.fallback,
                                  action={"kind": "retry", "from": came_from or frm, "to": target,
                                          "reset": reset, "attempt": retry_used},
                                  vars=dict(values), meta=_meta(0.0, 0, usage0, usage0)))
            came_from = came_edge = None
            cur = target
            continue
        if cur == machine.fallback and not interpret:
            return result(STOP_FALLBACK_EXHAUSTED,
                          f"fallback used up {retry_used} retries, not handing over to the interpreted segment (from {came_from or fallback_entry})")
        if cur == machine.fallback:
            while step < limit:
                step += 1
                rec, done, err = interpret_step(doc, values, records, step,
                                                model=model, tools=tools)
                records.append(rec)
                llm_calls += 1
                fallback_steps += 1
                if fallback_entry_step is None:
                    fallback_entry_step = rec.step
                if err:
                    return result(STOP_STATE_ERROR, f"{FALLBACK}: {err}")
                if done:
                    return result(STOP_TERMINAL)
            return result(STOP_MAX_STEPS, "FALLBACK interpretation took too many steps without stopping")

        state = machine.states.get(cur)
        if state is None:
            return result(STOP_STATE_ERROR, f"jumped to a non-existent state {cur!r}")
        step += 1
        t_step = time.perf_counter()

        # ---- end action ---- #
        if state.action.kind == "end":
            records.append(Record(step=step, state=cur, clause=state.clause,
                                  action={"kind": "end",
                                          "terminal": state.action.terminal},
                                  vars=dict(values),
                                  meta=_meta((time.perf_counter() - t_step) * 1000.0,
                                             0, usage0, usage0)))
            return result(STOP_TERMINAL)

        # ---- ordinary action ---- #
        u_before = _usage_of(model)
        rec_action, output, err, dcalls = _run_action(
            state, values, model=model, tools=tools,
            phase_rules=machine.phase_rules, outputs=task_outputs)
        llm_calls += dcalls
        argv = (_argv_of(tools, getattr(state.action, "name", ""), output)
                if state.action.kind == "tool" else None)
        records.append(Record(step=step, state=cur, clause=state.clause,
                              action=rec_action, output=output, vars=dict(values),
                              meta=_meta((time.perf_counter() - t_step) * 1000.0,
                                         dcalls, u_before, _usage_of(model),
                                         argv=argv)))
        if state.action.kind == "tool":
            last_tool = cur
        if err:
            if on_error != "fallback" or cur == machine.fallback:
                return result(STOP_STATE_ERROR, f"{cur}: {err}")
            if fallback_entry is None:
                fallback_entry = cur
            errors.append(f"{cur}: {err}")
            came_from, came_edge = cur, None
            cur = machine.fallback
            continue

        # ---- edge selection ---- #
        edge, eerr = pick_edge(machine, cur, values)
        if eerr:
            return result(STOP_STATE_ERROR, eerr)
        if edge is None:
            return result(STOP_STUCK, f"none of the outgoing edges of {cur} holds, and there is no default edge")
        if edge.inc:
            values[edge.inc] = (values.get(edge.inc) or 0) + 1
        if edge.to == machine.fallback:
            came_from, came_edge = cur, edge
            if fallback_entry is None:
                fallback_entry = cur             # the position list needs exactly "which state it fell back from"
        cur = edge.to

    return result(STOP_MAX_STEPS, f"no stop after {limit} steps")
