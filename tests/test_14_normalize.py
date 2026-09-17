"""(14) Action normalization: dicts in traces and models in machines fold into the same KEY; the loose and strict modes each keep their own boundary.
"""

import json

from hexis.examples import table_clean as tc
from hexis.machine.schema import (
    EndAction,
    JudgeAction,
    ModelAction,
    Record,
    ToolAction,
    UserAction,
)
from hexis.traces.normalize import (
    action_writes,
    canon_action,
    canon_output,
    canon_tool_name,
    same_action,
)

Q = "does the current header_row satisfy the canonical criterion of SKILL.md S2.1"
PROMPT = "rewrite the data in rows as a short summary"


# --------------------------------------------------------------------------- #
# trace dict <-> machine model: all five kinds must fold into the same KEY
# --------------------------------------------------------------------------- #
#: (name, Record in runtime shape, equivalent Action model). judge records question/reads, user
#: records only kind, and writes are always inferred from output. The model record here
#: **deliberately has only kind**: it stands for records that "did not record the prompt"
#: (external/old traces, and user); today's runtime._model_action does record the template,
#: which is covered by the rich group below.
_RUNTIME_PAIRS = [
    ("tool",
     Record(step=1, action={"kind": "tool", "name": "scripts/math_verify.py",
                            "input": {"expr": "1+1"}},
            output={"ok": True, "verified": True}),
     ToolAction(name="math-verify", input={"expr": "${expr}"},
                reads=["expr"], writes=["verified"])),
    ("judge",
     Record(step=2, action={"kind": "judge", "prompt": Q, "reads": ["header_row"]},
            output={"header_ok": "canonical"}),
     JudgeAction(prompt=Q, reads=["header_row"], writes=["header_ok"],
                 labels=list(tc.LABELS))),
    ("model",
     Record(step=3, action={"kind": "model"}, output={"ok": True, "summary": "omitted"}),
     ModelAction(prompt=PROMPT, reads=["rows"], writes=["summary"])),
    ("user",
     Record(step=4, action={"kind": "user"}, output={"answer": "42"}),
     UserAction(prompt=PROMPT, writes=["answer"])),
    ("end",
     Record(step=5, action={"kind": "end", "terminal": "done"}),
     EndAction(terminal="done")),
]


def test_record_and_action_model_agree_loose_for_all_five_kinds():
    """Loose mode: trace records and machine actions of all five kinds fold into the same KEY. Replay relies on exactly this."""
    for name, rec, act in _RUNTIME_PAIRS:
        assert canon_action(rec) == canon_action(act), name


def test_record_and_action_model_agree_strict_for_all_five_kinds():
    """Strict mode: as long as the trace records the distinguishing field (the prompt of judge and
    model), all five kinds agree in the compile mode as well."""
    rich = [
        ("tool", Record(step=1, action={"kind": "tool", "name": "MATH_VERIFY"}),
         ToolAction(name="scripts/math-verify.py")),
        ("judge", _RUNTIME_PAIRS[1][1], _RUNTIME_PAIRS[1][2]),
        ("model",
         Record(step=3, action={"kind": "model", "prompt": PROMPT},
                output={"ok": True, "summary": "omitted"}),
         ModelAction(prompt=PROMPT, reads=["rows"], writes=["summary"])),
        ("user",
         Record(step=4, action={"kind": "user", "prompt": PROMPT},
                output={"answer": "42"}),
         UserAction(prompt=PROMPT, writes=["answer"])),
        ("end", _RUNTIME_PAIRS[4][1], _RUNTIME_PAIRS[4][2]),
    ]
    for name, rec, act in rich:
        assert canon_action(rec, strict=True) == canon_action(act, strict=True), name


def test_a_record_without_a_prompt_field_is_loose_comparable_only():
    """When a record has **no** prompt field, the strict key of model/user cannot match across sources -- the loose key still matches.

    This is not a bug but a fact of the record format: a record that does not store the prompt
    has no distinguishing field, so its strict key can only degrade to ``(kind, writes=…)``,
    which never equals a machine action with a prompt. ``user`` records are exactly like that
    today (the runtime refuses to execute user actions and records only ``{"kind": "user"}``),
    and so are external traces and old traces from before that runtime revision. The runtime
    now records the raw template for model, so real model records take the path of
    ``test_..._strict_for_all_five_kinds`` above -- both cases are pinned so they are not
    mistaken for an accident later.
    """
    rec, act = _RUNTIME_PAIRS[2][1], _RUNTIME_PAIRS[2][2]
    assert canon_action(rec) == canon_action(act)
    assert canon_action(rec, strict=True) != canon_action(act, strict=True)


def test_raw_record_dict_and_bare_action_dict():
    """The KEY folded from a whole record (with output) equals the Record's; a bare action dict alone degrades writes to empty.

    That is the situation of code that only has ``rec.action``: the
    degraded strict key happens to equal the ``("judge", prompt)`` grouping -- no worse, but no
    more precise either: pass the whole Record for precision.
    """
    act = {"kind": "judge", "prompt": Q, "reads": ["header_row"]}
    out = {"header_ok": "canonical"}
    raw = {"step": 2, "action": act, "output": out}                 # one line of JSONL
    rec = Record(step=2, action=act, output=out)
    assert canon_action(raw) == canon_action(rec)
    assert canon_action(raw, strict=True) == canon_action(rec, strict=True)
    assert action_writes(act) == []                                 # a bare dict has no output
    assert canon_action(act, strict=True) == ("judge", "writes=", "prompt=" + Q)


def test_writes_are_recovered_from_the_record_output():
    """A judge's output is exactly {written variable: label}; ok/error of other actions are status bits, not writes."""
    assert action_writes(_RUNTIME_PAIRS[1][1]) == ["header_ok"]
    assert action_writes(_RUNTIME_PAIRS[2][1]) == ["summary"]      # ok is dropped
    assert action_writes(_RUNTIME_PAIRS[1][2]) == ["header_ok"]    # declared on the model side


# --------------------------------------------------------------------------- #
# Tool names
# --------------------------------------------------------------------------- #
def test_canon_tool_name_collapses_path_case_and_dashes():
    """The loop of the whole s3 (check) state hinges on this folding: the three spellings must converge to one name."""
    names = ["scripts/math_verify.py", "math-verify", "MATH_VERIFY"]
    assert {canon_tool_name(n) for n in names} == {"math_verify"}
    assert canon_tool_name("skills\\Math-Verify.PY") == "math_verify"
    assert canon_tool_name("  math   verify ") == "math_verify"
    assert canon_tool_name("") == ""
    assert canon_tool_name("export") != canon_tool_name("read_csv")


def test_tool_key_uses_the_canonical_name():
    a = Record(step=1, action={"kind": "tool", "name": "scripts/math_verify.py"})
    b = ToolAction(name="math-verify")
    assert same_action(a, b) and same_action(a, b, strict=True)


# --------------------------------------------------------------------------- #
# The boundary between the two modes
# --------------------------------------------------------------------------- #
def test_strict_splits_judges_by_question_loose_does_not():
    """Two questions that both write ok_label: the compile mode must keep them apart (two semantic branches), the replay mode need not."""
    j1 = JudgeAction(prompt="is the header canonical?", reads=["header_row"],
                     writes=["ok_label"], labels=list(tc.LABELS))
    j2 = JudgeAction(prompt="do the amounts match?", reads=["amount"],
                     writes=["ok_label"], labels=list(tc.LABELS))
    assert same_action(j1, j2)
    assert not same_action(j1, j2, strict=True)


def test_judge_writes_participate_in_both_modes():
    """Writing different variables = later conditions read different things = not the same step; both modes split them."""
    j1 = JudgeAction(prompt=Q, reads=["header_row"], writes=["header_ok"],
                     labels=list(tc.LABELS))
    j2 = JudgeAction(prompt=Q, reads=["header_row"], writes=["amount_ok"],
                     labels=list(tc.LABELS))
    assert not same_action(j1, j2)
    assert not same_action(j1, j2, strict=True)


def test_loose_splits_tools_but_never_their_arguments():
    """Arguments live in variables and are not part of the state identity -- two fix_header calls are one step, not two."""
    fix_a = Record(step=1, action={"kind": "tool", "name": "fix_header",
                                   "input": {"header_row": "name,,date"}},
                   output={"ok": True, "header_row": "name,col2,date"})
    fix_b = Record(step=2, action={"kind": "tool", "name": "fix_header",
                                   "input": {"header_row": "name,col2,Unnamed: 2"}},
                   output={"ok": True, "header_row": "name,col2,col3"})
    export = Record(step=3, action={"kind": "tool", "name": "export",
                                    "input": {"output_path": "out.csv"}},
                    output={"ok": True, "output_path": "out.csv"})
    assert same_action(fix_a, fix_b)
    assert same_action(fix_a, fix_b, strict=True)      # arguments take no part in strict mode either
    assert not same_action(fix_a, export)


def test_model_prompt_never_enters_the_loose_key():
    """A rendered prompt carries the problem statement: counting it in the loose KEY makes every step unique and breaks replay entirely."""
    m1 = ModelAction(prompt="solve this problem: 1+1=?", writes=["answer"])
    m2 = ModelAction(prompt="solve this problem: integral ∫x dx", writes=["answer"])
    assert same_action(m1, m2)
    assert not same_action(m1, m2, strict=True)
    assert all("1+1" not in part for part in canon_action(m1))


def test_end_terminal_participates_in_both_modes():
    """A different way of ending is not the same step -- loose mode keeps the terminal on purpose."""
    e1, e2 = EndAction(terminal="done"), EndAction(terminal="give_up")
    assert not same_action(e1, e2)
    assert not same_action(e1, e2, strict=True)


def test_different_kinds_never_collide():
    acts = [ToolAction(name="x"), ModelAction(prompt="p"), UserAction(prompt="p"),
            JudgeAction(prompt=Q, reads=["a"], writes=["b"], labels=list(tc.LABELS)),
            EndAction(terminal="done")]
    for mode in (False, True):
        keys = [canon_action(a, strict=mode) for a in acts]
        assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------- #
# Output trimming and KEY stability
# --------------------------------------------------------------------------- #
def test_canon_output_keeps_declared_drops_the_rest():
    out = {"header_row": "a,b", "rows": [["1"]], "ok": True, "debug": "noise"}
    assert canon_output(out, ["header_row", "rows"]) == {"header_row": "a,b",
                                                        "rows": [["1"]]}
    assert canon_output(out, ["missing"]) == {}          # declared but not produced
    assert canon_output(out, []) == {}                   # nothing declared, nothing collected
    assert canon_output({}, ["header_row"]) == {}


def test_canon_output_is_order_independent():
    o1 = {"a": 1, "b": 2, "ok": True}
    o2 = {"ok": True, "b": 2, "a": 1}
    assert json.dumps(canon_output(o1, ["b", "a"])) == \
           json.dumps(canon_output(o2, ["a", "b"]))


def test_keys_are_hashable_json_serialisable_and_stable():
    """Stable across processes: no hash()/id(), no dependence on dict insertion order. json.dumps of two constructions is byte-identical."""
    a = Record(step=1, action={"kind": "judge", "prompt": Q, "reads": ["x"]},
               output={"p": 1, "q": 2})
    b = Record(step=9, action={"reads": ["x"], "prompt": Q, "kind": "judge"},
               output={"q": 2, "p": 1})
    for mode in (False, True):
        ka, kb = canon_action(a, strict=mode), canon_action(b, strict=mode)
        assert ka == kb
        assert json.dumps(ka, ensure_ascii=False) == json.dumps(kb, ensure_ascii=False)
        assert isinstance(ka, tuple) and all(isinstance(p, str) for p in ka)
        assert len({ka, kb}) == 1                        # hashable, and the same key


# --------------------------------------------------------------------------- #
# Cross-check: compare the grouping of the two existing implementations on the table_clean records
# --------------------------------------------------------------------------- #
def _table_clean_records(make_traces):
    recs = [r for t in make_traces(6, seed=1) for r in t.records]
    kinds = {r.action.get("kind") for r in recs}
    assert recs and {"tool", "judge", "end"} <= kinds
    return recs


def test_records_of_the_same_step_land_in_one_class(make_traces):
    """Records produced by the same state land in the same KEY however far apart their arguments are; different states never merge."""
    recs = _table_clean_records(make_traces)
    by_key: dict = {}
    for r in recs:
        by_key.setdefault(canon_action(r), set()).add(r.state)
    assert all(len(states) == 1 for states in by_key.values()), by_key
    assert len(by_key) == len({s for ss in by_key.values() for s in ss})
