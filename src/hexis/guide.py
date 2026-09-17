"""Usage guides for compiled machines.

:func:`render_guide` writes ``GUIDE.md`` for people: what the machine needs, what it does state by state, a
diagram, and how to run it. :func:`render_prompt` writes ``PROMPT.md``: a system prompt that lets a tool-using
agent execute the machine step by step without the hexis runtime, following the same rules as
:func:`hexis.execution.runtime.run_task`.

Both renderers are deterministic: the same machine and inputs always give the same text.

``PROMPT.md`` places the whole machine, including the prompts derived from the skill document, in the agent's
context. The hexis runtime never does that: it shows a model only the prompt of the state being executed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hexis import __version__
from hexis.compiler.common import counter_exit
from hexis.compiler.stepwise import fingerprint
from hexis.execution.runtime import entry_of
from hexis.machine.schema import Machine, State

KIND_WORD = {"tool": "tool", "model": "generate", "judge": "decide", "user": "ask user", "end": "end"}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def state_order(m: Machine) -> list[str]:
    """States in breadth-first order from the initial state (transitions in evaluation order), then the rest."""
    seen: list[str] = []
    queue = [m.initial] if m.initial in m.states else []
    while queue:
        sid = queue.pop(0)
        if sid in seen or sid not in m.states:
            continue
        seen.append(sid)
        queue += [t.to for t in m.states[sid].ordered_transitions()]
    rest = sorted(s for s in m.states if s not in seen)
    return seen + rest


def task_inputs(m: Machine) -> list[tuple[str, str, str]]:
    """(variable, task input key, type) for every variable initialized from the task."""
    return [(v.name, v.init_from.split(".")[-1], v.type) for v in m.variables if v.init_from]


def counter_limits(m: Machine) -> list[tuple[str, str, int, str]]:
    """(state, counter, bound, target) for every ``counter >= K`` guard."""
    out = []
    for sid in state_order(m):
        for t in m.states[sid].ordered_transitions():
            ce = counter_exit(t.cond)
            if ce:
                out.append((sid, ce[0], ce[1], t.to))
    return out


def gate_of(m: Machine, sid: str) -> Optional[str]:
    """The state that prepares the arguments of tool state ``sid``, if there is one."""
    e = entry_of(m, sid)
    return e if e != sid else None


def description(st: State) -> str:
    return str(getattr(st, "description", "") or "").strip()


def excerpt(text: str, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def edge_text(t, *, code: bool = True) -> str:
    q = (lambda s: f"`{s}`") if code else (lambda s: s)
    head = f"if {q(t.cond)}" if t.cond else "otherwise"
    tail = f", add 1 to {q(t.inc)}" if t.inc else ""
    return f"{head} → {q(t.to)}{tail}"


def fence(text: str, lang: str = "") -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text}\n{ticks}"


def md_cell(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def tool_names(m: Machine) -> list[str]:
    return sorted({st.action.name for st in m.states.values() if st.action.kind == "tool"})


def _spec_field(spec: Any, name: str, default: Any = None) -> Any:
    if spec is None:
        return default
    if isinstance(spec, Mapping):
        return spec.get(name, default)
    return getattr(spec, name, default)


# --------------------------------------------------------------------------- #
# Mermaid
# --------------------------------------------------------------------------- #
def _mm_escape(text: str) -> str:
    return (str(text).replace("&", "#amp;").replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")
            .replace("|", "#124;").replace("`", "#96;"))


def mermaid(m: Machine, *, label_chars: int = 48) -> str:
    """A Mermaid ``flowchart`` of the machine. Node ids are generated (``n0``, ``n1``…) because state ids may be
    Mermaid keywords such as ``end``; the real id is the node label."""
    order = state_order(m)
    ids = {sid: f"n{i}" for i, sid in enumerate(order)}
    lines = ["flowchart TD", '    n_start(["start"])']
    for sid in order:
        a = m.states[sid].action
        what = {"tool": f"tool {getattr(a, 'name', '')}", "model": "generate", "judge": "decide",
                "user": "ask user", "end": f"end {getattr(a, 'terminal', '')}"}.get(a.kind, a.kind)
        if a.kind == "model" and getattr(a, "observable", False):
            what = "deliverable"
        label = f'"{_mm_escape(excerpt(sid, label_chars))}<br/>{_mm_escape(excerpt(what, label_chars))}"'
        shape = {"tool": f"[{label}]", "model": f"({label})", "judge": f"{{{label}}}", "user": f"[/{label}/]",
                 "end": f"([{label}])"}.get(a.kind, f"[{label}]")
        if sid == m.fallback:
            shape = f"[[{label}]]"
        lines.append(f"    {ids[sid]}{shape}")
    if m.initial in ids:
        lines.append(f"    n_start --> {ids[m.initial]}")
    for sid in order:
        for k, t in enumerate(m.states[sid].ordered_transitions(), 1):
            if t.to not in ids:
                continue
            text = f"{k}. " + (t.cond if t.cond else "otherwise") + (f" [+1 {t.inc}]" if t.inc else "")
            lines.append(f'    {ids[sid]} -->|"{_mm_escape(excerpt(text, label_chars))}"| {ids[t.to]}')
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# GUIDE.md
# --------------------------------------------------------------------------- #
def _what_it_does(m: Machine, sid: str) -> str:
    st = m.states[sid]
    a = st.action
    desc = description(st)
    if sid == m.fallback:
        return "fallback: retries, then interpreted execution (see Fallback)"
    if a.kind == "tool":
        base = f"calls `{a.name}` with `{json.dumps(a.input, ensure_ascii=False, sort_keys=False)}`"
        gate = gate_of(m, sid)
        base += f" (arguments prepared by `{gate}`)" if gate else ""
    elif a.kind == "model":
        base = ("writes the deliverable: " if getattr(a, "observable", False) else "generates: ") + excerpt(a.prompt, 140)
    elif a.kind == "judge":
        base = f"decides {', '.join(f'`{x}`' for x in a.labels)}: " + excerpt(a.prompt, 120)
    elif a.kind == "user":
        base = "asks the user: " + excerpt(getattr(a, "prompt", ""), 120)
    else:
        base = f"ends with terminal `{a.terminal}`"
    return f"{desc} ({base})" if desc else base


def render_guide(m: Machine, *, tools: Optional[Mapping[str, Any]] = None, skill_name: str = "",
                 skill_description: str = "", manifest: Optional[Mapping] = None,
                 progress: Optional[Mapping] = None, clause_map: Optional[Mapping[str, str]] = None,
                 prompt_file: str = "PROMPT.md") -> str:
    tools = dict(tools or {})
    clause_map = dict(clause_map or {})
    order = state_order(m)
    title = skill_name or m.skill_id
    kinds: dict[str, int] = {}
    for st in m.states.values():
        kinds[st.action.kind] = kinds.get(st.action.kind, 0) + 1
    n_edges = sum(len(st.transitions) for st in m.states.values())
    out: list[str] = [f"# {title}: machine guide", ""]
    if skill_description:
        out += [skill_description.strip(), ""]
    prov = [f"skill `{m.skill_id}`", f"format `{m.format}`", f"machine version `{m.version}`",
            f"structure fingerprint `{fingerprint(m)}`", f"generated by hexis {__version__}"]
    sha = ((manifest or {}).get("skill") or {}).get("sha256") if manifest else None
    if sha:
        prov.append(f"skill document sha256 `{sha[:16]}`")
    out += ["_" + "; ".join(prov) + "_", ""]

    out += ["## Summary", ""]
    out.append("- States: " + ", ".join(f"{n} {KIND_WORD.get(k, k)}" for k, n in sorted(kinds.items()))
               + f" ({len(m.states)} in total)")
    out.append(f"- Transitions: {n_edges}; variables: {len(m.variables)}; step limit: {m.max_steps}")
    out.append("- Terminals: " + (", ".join(f"`{t.id}` ({t.kind or 'unspecified'})" for t in m.terminals) or "none"))
    out.append(f"- Start state: `{m.initial}`; fallback state: `{m.fallback}`")
    counts = (progress or {}).get("counts") if progress else None
    if counts:
        out.append("- Traces processed: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    models = sorted({(r.get("model") or {}).get("model", "") for r in (manifest or {}).get("runs", [])
                     if (r.get("model") or {}).get("model")}) if manifest else []
    if models:
        out.append("- Models used to build it: " + ", ".join(f"`{x}`" for x in models))
    out.append("")

    out += ["## Inputs", ""]
    ins = task_inputs(m)
    if ins:
        out += ["The task must provide these inputs:", "", "| Variable | Task input | Type |", "|---|---|---|"]
        out += [f"| `{v}` | `{k}` | {t} |" for v, k, t in ins]
    else:
        out.append("The machine declares no task inputs.")
    out.append("")

    out += ["## Tools", ""]
    names = tool_names(m)
    if names:
        out += ["| Tool | Description | Used in | Success when | Source |", "|---|---|---|---|---|"]
        for n in names:
            spec = tools.get(n)
            used = [sid for sid in order if m.states[sid].action.kind == "tool" and m.states[sid].action.name == n]
            out.append(f"| `{n}` | {md_cell(excerpt(_spec_field(spec, 'description', ''), 160))} | "
                       f"{', '.join(f'`{s}`' for s in used)} | "
                       f"{('`' + md_cell(_spec_field(spec, 'success')) + '`') if _spec_field(spec, 'success') else ''} | "
                       f"{md_cell(_spec_field(spec, 'source', 'not given' if spec is None else ''))} |")
    else:
        out.append("The machine calls no tools.")
    out.append("")

    out += ["## Flow", "", fence(mermaid(m), "mermaid"), ""]

    out += ["## States", "", "| State | Kind | What it does | Reads | Writes | Clause |", "|---|---|---|---|---|---|"]
    for sid in order:
        st = m.states[sid]
        a = st.action
        reads = getattr(a, "reads", []) or []
        writes = getattr(a, "writes", []) or []
        clause = st.clause or clause_map.get(sid, "")
        out.append(f"| `{sid}` | {KIND_WORD.get(a.kind, a.kind)} | {md_cell(_what_it_does(m, sid))} | "
                   f"{', '.join(f'`{x}`' for x in reads)} | {', '.join(f'`{x}`' for x in writes)} | {md_cell(clause)} |")
    out.append("")

    out += ["## Transitions", "", "After a state's action, its transitions are checked in this order and the first one "
            "that holds is taken.", ""]
    for sid in order:
        edges = m.states[sid].ordered_transitions()
        if not edges:
            continue
        out.append(f"- `{sid}`: " + "; ".join(f"{k}. {edge_text(t)}" for k, t in enumerate(edges, 1)))
    out.append("")

    out += ["## Terminals", "", "| Terminal | Kind | Outputs | End states |", "|---|---|---|---|"]
    for t in m.terminals:
        ends = [sid for sid in order if m.states[sid].action.kind == "end" and m.states[sid].action.terminal == t.id]
        out.append(f"| `{t.id}` | {t.kind or ''} | {', '.join(f'`{o}`' for o in t.output)} | "
                   f"{', '.join(f'`{e}`' for e in ends)} |")
    out.append("")

    out += ["## Loops and limits", ""]
    limits = counter_limits(m)
    if limits:
        out += [f"- `{sid}` leaves for `{to}` once `{cnt} >= {k}`" for sid, cnt, k, to in limits]
    else:
        out.append("- No counter limits.")
    out += [f"- A run stops after {m.max_steps} steps.", ""]

    into_fb = sorted({sid for sid in order for t in m.states[sid].transitions if t.to == m.fallback})
    out += ["## Fallback", "",
            f"`{m.fallback}` is reached through a transition or when a state's action fails. The hexis runtime then "
            "retries from the most recently executed tool state (from the state that prepares its arguments, if there "
            "is one) and resets the counter whose limit led there; when the retries are used up, it hands the task to "
            "interpreted execution of the skill document.",
            ""]
    if into_fb:
        out += ["States with a transition to the fallback state: " + ", ".join(f"`{s}`" for s in into_fb), ""]

    out += ["## Running the machine", "",
            "With the command-line interface (a build directory can be passed as `--machine`):", "",
            fence("hexis-agent run --machine BUILD_DIR "
                  + " ".join(f"--input {k}=..." for _v, k, _t in ins) + " --workdir WORK_DIR --executor local", "bash"),
            "",
            "From Python:", "",
            fence("from hexis.execution import runtime\n"
                  "from hexis.machine.schema import load_machine\n\n"
                  "machine = load_machine(\"machine.json\")\n"
                  "result = runtime.run_task(machine, {\"input\": {" + ", ".join(f'"{k}": ...' for _v, k, _t in ins)
                  + "}}, model=model, tools=tools, doc=skill_document, on_error=\"fallback\", retries=3)\n"
                  "print(result.stopped, result.path())", "python"),
            "",
            f"With an agent: give `{prompt_file}` to a tool-using agent as its system prompt (or paste it at the start "
            "of the conversation) together with the task inputs.", ""]

    out += ["## Privacy", "",
            f"`{prompt_file}` contains the whole machine, including prompts derived from the skill document. The hexis "
            "runtime shows a model only the prompt of the state it is executing. A build directory also contains "
            "copies of the traces and the questions sent to the model while updating the machine; review it before "
            "sharing.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# PROMPT.md
# --------------------------------------------------------------------------- #
def _state_block(m: Machine, sid: str) -> list[str]:
    st = m.states[sid]
    a = st.action
    desc = description(st)
    head = {"tool": f"tool `{getattr(a, 'name', '')}`", "model": "generate", "judge": "decide",
            "user": "ask user", "end": "end"}.get(a.kind, a.kind)
    if a.kind == "model" and getattr(a, "observable", False):
        head = "generate the deliverable"
    if sid == m.fallback:
        return [f"### `{sid}`: fallback", "", "This is the fallback state: follow section 7.", ""]
    lines = [f"### `{sid}`: {head}", ""]
    if desc:
        lines += [desc, ""]
    if a.kind == "tool":
        gate = gate_of(m, sid)
        if gate:
            lines += [f"Its arguments are prepared by `{gate}`.", ""]
        lines += ["Arguments:", "", fence(json.dumps(a.input, ensure_ascii=False, indent=2), "json"), ""]
        binds = dict(getattr(a, "binds", {}) or {})
        stores = [f"`{w}`" + (f" (the `{src}` result field)" if w in binds.values() and w not in binds else "")
                  for w in a.writes
                  for src in [next((s for s, d in binds.items() if d == w), w)]]
        lines += ["Store: " + (", ".join(stores) if stores else "nothing"), ""]
    elif a.kind == "model":
        lines += ["Instruction:", "", fence(a.prompt), ""]
        lines += ["Inputs: " + (", ".join(f"`{r}`" for r in a.reads) or "none") + "  ·  Store: "
                  + (", ".join(f"`{w}`" for w in a.writes) or "nothing"), ""]
    elif a.kind == "judge":
        lines += ["Question:", "", fence(a.prompt), ""]
        lines += ["Inputs: " + (", ".join(f"`{r}`" for r in a.reads) or "none"),
                  "Labels: " + ", ".join(f"`{x}`" for x in a.labels) + f"; abstain label `{a.abstain}`",
                  f"Store the label in `{a.writes[0]}`." if a.writes else ""]
        if a.examples:
            lines += ["Examples:", "", fence(json.dumps([e.model_dump() for e in a.examples][:3], ensure_ascii=False,
                                                        indent=2), "json")]
        lines.append("")
    elif a.kind == "user":
        lines += ["Question for the user:", "", fence(getattr(a, "prompt", "") or ""), "",
                  "Store the answer in: " + (", ".join(f"`{w}`" for w in getattr(a, "writes", [])) or "nothing"), ""]
    else:
        term = next((t for t in m.terminals if t.id == a.terminal), None)
        kind = f" ({term.kind})" if term is not None and term.kind else ""
        lines += [f"Stop: terminal `{a.terminal}`{kind}.", ""]
        return lines
    edges = st.ordered_transitions()
    lines.append("Transitions:")
    lines += [f"{k}. {edge_text(t)}" for k, t in enumerate(edges, 1)] if edges else ["(none: report `stuck`)"]
    lines.append("")
    return lines


def render_prompt(m: Machine, *, tools: Optional[Mapping[str, Any]] = None, skill_description: str = "",
                  skill_doc: Optional[str] = None, retries: int = 3) -> str:
    tools = dict(tools or {})
    order = state_order(m)
    ins = task_inputs(m)
    out: list[str] = [f"# Procedure: {m.skill_id}", ""]
    if skill_description:
        out += [skill_description.strip(), ""]
    out += ["You carry out this task by executing the state machine below exactly. It was compiled from a skill "
            f"document by hexis (structure fingerprint {fingerprint(m)}). At every moment you are in exactly one "
            "state: perform that state's action, record the results in variables, and choose the next state with "
            "its transitions. Do not skip, merge, reorder or add steps.",
            "",
            "This is a multi-step episode: after each tool call you receive its result and continue with the next "
            "step. Keep going until you reach an end state or these instructions tell you to stop.",
            "",
            "## 1. Execution loop", "",
            f"Start in state `{m.initial}` and repeat:", "",
            "1. Write `STATE <id>` on its own line.",
            "2. Perform the state's action (section 5) exactly once.",
            "3. Check the state's transitions in the listed order and follow the first one whose condition is true; "
            "`otherwise` applies when no earlier condition is true. If the transition says `add 1 to <counter>`, "
            "increase that counter by 1.",
            "4. Write `NEXT <id> because <condition or otherwise>`, followed by the variables you set in this step.",
            "",
            f"Count every executed state. After {m.max_steps} states, stop and report `max_steps`.",
            "",
            "Actions:", "",
            "- **tool**: call the named tool with the arguments shown. Replace every `${name}` with the current "
            "value of variable `name`: if the whole argument is `${name}`, pass the value unchanged; inside longer "
            "text, insert the value as text (an unset variable becomes empty text). Then store only the listed "
            "result fields. A result that reports a failure (for example a non-zero return code) is still a result: "
            "store it and follow the transitions. If the tool cannot be called at all, or it returns only an error "
            "message, go to the fallback state (section 7).",
            "- **generate**: write the listed variables yourself, following the instruction and using only the "
            "listed inputs. Do not call tools for this. If you cannot produce them, go to the fallback state.",
            "- **decide**: answer the question with exactly one of the listed labels and store it. If you cannot "
            "decide with confidence, use the abstain label.",
            "- **ask user**: ask the question and store the answer.",
            "- **end**: stop and report (section 6).",
            "",
            "## 2. Variables", ""]
    if ins:
        out += ["Task inputs (take them from the task; ask for any that are missing): "
                + ", ".join(f"`{k}`" for _v, k, _t in ins) + ". Every task input is also a variable with the same "
                "name.", ""]
    out += ["| Variable | Type | Initial value |", "|---|---|---|"]
    for v in m.variables:
        init = (f"task input `{v.init_from.split('.')[-1]}`" if v.init_from
                else (f"`{json.dumps(v.init, ensure_ascii=False)}`" if v.init is not None else "unset"))
        out.append(f"| `{v.name}` | {v.type} | {init} |")
    out += ["", "A variable without an initial value is unset until an action stores it.", "",
            "## 3. Conditions", "",
            "Conditions use `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, combined with `and`, `or` and `not`; "
            "`empty(x)` is true when `x` is unset or empty and `nonempty(x)` when it has content; strings are quoted; "
            "`True` and `False` are booleans. If a condition needs a variable that is unset, stop and report "
            "`state_error` with the condition. If no transition applies, stop and report `stuck`.", "",
            "## 4. Tools", ""]
    names = tool_names(m)
    if names:
        out += ["| Tool | Purpose | Result fields | Succeeds when |", "|---|---|---|---|"]
        for n in names:
            spec = tools.get(n)
            outs = _spec_field(spec, "output_schema") or {}
            fields = ", ".join(f"`{k}`" for k in (outs.keys() if isinstance(outs, Mapping) else outs))
            if not fields:
                fields = ", ".join(sorted({f"`{w}`" for st in m.states.values()
                                           if st.action.kind == "tool" and st.action.name == n
                                           for w in st.action.writes}))
            succ = _spec_field(spec, "success")
            out.append(f"| `{n}` | {md_cell(excerpt(_spec_field(spec, 'description', ''), 200))} | {fields} | "
                       f"{('`' + md_cell(succ) + '`') if succ else ''} |")
        out += ["", "Call tools by these names. If your environment offers the same capability under a different name "
                "(for example a shell tool for `bash`), use it with equivalent arguments and derive the result fields "
                "from its output.", ""]
    else:
        out += ["The machine calls no tools.", ""]
    out += ["## 5. States", ""]
    for sid in order:
        out += _state_block(m, sid)
    out += ["## 6. Finishing", "",
            "At an end state, stop and report the terminal (id and kind), the list of states you went through, and "
            "the terminal's outputs:", "",
            "| Terminal | Kind | Report |", "|---|---|---|"]
    out += [f"| `{t.id}` | {t.kind or ''} | {', '.join(f'`{o}`' for o in t.output)} |" for t in m.terminals]
    out += ["", "If you stopped for another reason (`max_steps`, `state_error`, `stuck`, `fallback`), report the reason, "
            "the state where it happened, and what was done.", "",
            "## 7. Fallback", "",
            f"You are in the fallback state `{m.fallback}` when a transition leads to it or when an action cannot be "
            "carried out.", "",
            f"1. Retry, at most {retries} times in total: go back to the most recently executed tool state (to the state "
            "that prepares its arguments, if one is listed; if no tool state has run yet, to the state you came from). "
            "If the transition that led here was a `counter >= K` limit, reset that counter to 0. Continue from there."]
    if skill_doc:
        out += ["2. When no retries are left, finish the task by following the skill document in the appendix, one "
                "action at a time, and report that execution fell back and from which state."]
    else:
        out += ["2. When no retries are left, stop and report `fallback` with the state you came from."]
    out.append("")
    if m.prohibitions:
        out += ["## 8. Prohibitions", "", "Never do the following, in any state:", ""]
        out += [f"- `{p.id}` ({p.check}): `{json.dumps(p.pattern, ensure_ascii=False)}`" for p in m.prohibitions]
        out.append("")
    if skill_doc:
        out += ["## Appendix: skill document", "", fence(skill_doc.strip(), "markdown"), ""]
    return "\n".join(out)


def write_docs(out_dir: Any, m: Machine, *, tools: Optional[Mapping[str, Any]] = None, skill_name: str = "",
               skill_description: str = "", skill_doc: Optional[str] = None, embed_skill: bool = False,
               retries: int = 3, manifest: Optional[Mapping] = None, progress: Optional[Mapping] = None,
               clause_map: Optional[Mapping[str, str]] = None) -> tuple[Path, Path]:
    """Write ``GUIDE.md`` and ``PROMPT.md`` into ``out_dir`` and return their paths."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    guide = d / "GUIDE.md"
    prompt = d / "PROMPT.md"
    guide.write_text(render_guide(m, tools=tools, skill_name=skill_name, skill_description=skill_description,
                                  manifest=manifest, progress=progress, clause_map=clause_map) + "\n",
                     encoding="utf-8")
    prompt.write_text(render_prompt(m, tools=tools, skill_description=skill_description,
                                    skill_doc=skill_doc if embed_skill else None, retries=retries) + "\n",
                      encoding="utf-8")
    return guide, prompt


__all__ = ["counter_limits", "mermaid", "render_guide", "render_prompt", "state_order", "task_inputs", "write_docs"]
