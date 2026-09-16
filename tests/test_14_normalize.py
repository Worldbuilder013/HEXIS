"""⑭ 动作规范化：轨迹里的 dict 与机器里的模型折成同一个 KEY，松紧两档各守各的边界。

这套断言是给日后「把 compiler._sig / replay._action_matches 指到 normalize 上」上的保险：
只要分组关系不变，重指就不会改变行为。
"""

import json

from hexis.legacy import compiler, replay
from hexis.examples import table_clean as tc
from hexis.traces.normalize import (
    action_writes, canon_action, canon_output, canon_tool_name, same_action,
)
from hexis.machine.schema import (
    EndAction, JudgeAction, ModelAction, Record, ToolAction, UserAction,
)

Q = "当前 header_row 是否符合 SKILL.md S2.1 的规范判据"
PROMPT = "把 rows 里的数据改写成一段摘要"


# --------------------------------------------------------------------------- #
# 轨迹 dict ←→ 机器模型：五种 kind 都得折出同一个 KEY
# --------------------------------------------------------------------------- #
#: (名字, 运行时形状的 Record, 等价的 Action 模型)。judge 记 question/reads、user 只记 kind，
#: writes 一律靠 output 反推。这里的 model 记录**刻意只写 kind**：它代表「没把 prompt 记下来」
#: 的那一类记录（外部/旧轨迹、以及 user），今天的 runtime._model_action 是会记模板的，
#: 那一档由下面 rich 那组覆盖。
_RUNTIME_PAIRS = [
    ("tool",
     Record(step=1, action={"kind": "tool", "name": "scripts/math_verify.py",
                            "input": {"expr": "1+1"}},
            output={"ok": True, "verified": True}),
     ToolAction(name="math-verify", input={"expr": "${expr}"},
                reads=["expr"], writes=["verified"])),
    ("judge",
     Record(step=2, action={"kind": "judge", "prompt": Q, "reads": ["header_row"]},
            output={"header_ok": "规范"}),
     JudgeAction(prompt=Q, reads=["header_row"], writes=["header_ok"],
                 labels=list(tc.LABELS))),
    ("model",
     Record(step=3, action={"kind": "model"}, output={"ok": True, "summary": "略"}),
     ModelAction(prompt=PROMPT, reads=["rows"], writes=["summary"])),
    ("user",
     Record(step=4, action={"kind": "user"}, output={"answer": "42"}),
     UserAction(prompt=PROMPT, writes=["answer"])),
    ("end",
     Record(step=5, action={"kind": "end", "terminal": "done"}),
     EndAction(terminal="done")),
]


def test_record_and_action_model_agree_loose_for_all_five_kinds():
    """松档：五种 kind 的轨迹记录与机器动作折出同一个 KEY。回放靠的就是这条。"""
    for name, rec, act in _RUNTIME_PAIRS:
        assert canon_action(rec) == canon_action(act), name


def test_record_and_action_model_agree_strict_for_all_five_kinds():
    """严档：只要轨迹把区分字段记下来（judge 与 model 的 prompt），五种
    kind 在编译档下同样一致。"""
    rich = [
        ("tool", Record(step=1, action={"kind": "tool", "name": "MATH_VERIFY"}),
         ToolAction(name="scripts/math-verify.py")),
        ("judge", _RUNTIME_PAIRS[1][1], _RUNTIME_PAIRS[1][2]),
        ("model",
         Record(step=3, action={"kind": "model", "prompt": PROMPT},
                output={"ok": True, "summary": "略"}),
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
    """记录里**没有** prompt 字段时，model/user 的严档就跨不了源——松档照常对上。

    这不是 bug，是记录格式的事实：一条不记提示词的记录拿不出区分字段，严档只能退化成
    ``(kind, writes=…)``，与带 prompt 的机器动作必然不等。今天 ``user`` 记录正是这样
    （runtime 拒绝执行 user 动作，只记 ``{"kind": "user"}``），外部轨迹与 runtime 那一轮
    之前的旧轨迹也一样。runtime 现在给 model 记模板原文，所以真实的 model 记录走的是上面
    ``test_..._strict_for_all_five_kinds`` 那条路——两种情形都钉住，免得日后当成偶然。
    """
    rec, act = _RUNTIME_PAIRS[2][1], _RUNTIME_PAIRS[2][2]
    assert canon_action(rec) == canon_action(act)
    assert canon_action(rec, strict=True) != canon_action(act, strict=True)


def test_raw_record_dict_and_bare_action_dict():
    """整条记录（含 output）折出的 KEY 与 Record 一致；只给裸 action dict 则 writes 退化成空。

    后者正是 compiler._sig 今天的处境（它只拿得到 ``rec.action``），所以退化后的严档恰好等于
    ``("judge", prompt)`` 那套分组——不会更错，但也别指望更准：要准就传整条 Record。
    """
    act = {"kind": "judge", "prompt": Q, "reads": ["header_row"]}
    out = {"header_ok": "规范"}
    raw = {"step": 2, "action": act, "output": out}                 # JSONL 里的一行
    rec = Record(step=2, action=act, output=out)
    assert canon_action(raw) == canon_action(rec)
    assert canon_action(raw, strict=True) == canon_action(rec, strict=True)
    assert action_writes(act) == []                                 # 裸 dict 没有 output
    assert canon_action(act, strict=True) == ("judge", "writes=", "prompt=" + Q)


def test_writes_are_recovered_from_the_record_output():
    """judge 的 output 就是 {写入变量: 标签}；其余动作的 ok/error 是状态位、不算写入。"""
    assert action_writes(_RUNTIME_PAIRS[1][1]) == ["header_ok"]
    assert action_writes(_RUNTIME_PAIRS[2][1]) == ["summary"]      # ok 被剔掉
    assert action_writes(_RUNTIME_PAIRS[1][2]) == ["header_ok"]    # 模型侧是声明的


# --------------------------------------------------------------------------- #
# 工具名
# --------------------------------------------------------------------------- #
def test_canon_tool_name_collapses_path_case_and_dashes():
    """整个 s3（校验）状态的成环都押在这次折叠上：三种写法必须收敛成一个名字。"""
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
# 两档的分界
# --------------------------------------------------------------------------- #
def test_strict_splits_judges_by_question_loose_does_not():
    """同样写 ok_label 的两个提问：编译档必须分开（两个语义分岔），回放档不必。"""
    j1 = JudgeAction(prompt="表头规范吗", reads=["header_row"],
                     writes=["ok_label"], labels=list(tc.LABELS))
    j2 = JudgeAction(prompt="金额对得上吗", reads=["amount"],
                     writes=["ok_label"], labels=list(tc.LABELS))
    assert same_action(j1, j2)
    assert not same_action(j1, j2, strict=True)


def test_judge_writes_participate_in_both_modes():
    """写不同变量 = 后续条件读到的东西不同 = 不是同一步，两档都分。"""
    j1 = JudgeAction(prompt=Q, reads=["header_row"], writes=["header_ok"],
                     labels=list(tc.LABELS))
    j2 = JudgeAction(prompt=Q, reads=["header_row"], writes=["amount_ok"],
                     labels=list(tc.LABELS))
    assert not same_action(j1, j2)
    assert not same_action(j1, j2, strict=True)


def test_loose_splits_tools_but_never_their_arguments():
    """参数活在变量里，不属于状态身份——两次 fix_header 是同一步，不是两步。"""
    fix_a = Record(step=1, action={"kind": "tool", "name": "fix_header",
                                   "input": {"header_row": "名称,,日期"}},
                   output={"ok": True, "header_row": "名称,列2,日期"})
    fix_b = Record(step=2, action={"kind": "tool", "name": "fix_header",
                                   "input": {"header_row": "名称,列2,Unnamed: 2"}},
                   output={"ok": True, "header_row": "名称,列2,列3"})
    export = Record(step=3, action={"kind": "tool", "name": "export",
                                    "input": {"output_path": "out.csv"}},
                    output={"ok": True, "output_path": "out.csv"})
    assert same_action(fix_a, fix_b)
    assert same_action(fix_a, fix_b, strict=True)      # 参数在严档同样不参与
    assert not same_action(fix_a, export)


def test_model_prompt_never_enters_the_loose_key():
    """渲染过的 prompt 里带着题面：计进松档 KEY，每一步都独一无二，回放全废。"""
    m1 = ModelAction(prompt="解这道题：1+1=?", writes=["answer"])
    m2 = ModelAction(prompt="解这道题：积分 ∫x dx", writes=["answer"])
    assert same_action(m1, m2)
    assert not same_action(m1, m2, strict=True)
    assert all("1+1" not in part for part in canon_action(m1))


def test_end_terminal_participates_in_both_modes():
    """结束方式不同就不是同一步——这是松档比 replay._action_matches 细的地方，有意为之。"""
    e1, e2 = EndAction(terminal="done"), EndAction(terminal="give_up")
    assert not same_action(e1, e2)
    assert not same_action(e1, e2, strict=True)
    assert replay._action_matches(e1, {"kind": "end", "terminal": "give_up"})


def test_different_kinds_never_collide():
    acts = [ToolAction(name="x"), ModelAction(prompt="p"), UserAction(prompt="p"),
            JudgeAction(prompt=Q, reads=["a"], writes=["b"], labels=list(tc.LABELS)),
            EndAction(terminal="done")]
    for mode in (False, True):
        keys = [canon_action(a, strict=mode) for a in acts]
        assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------- #
# 产出裁剪与 KEY 的稳定性
# --------------------------------------------------------------------------- #
def test_canon_output_keeps_declared_drops_the_rest():
    out = {"header_row": "a,b", "rows": [["1"]], "ok": True, "debug": "噪声"}
    assert canon_output(out, ["header_row", "rows"]) == {"header_row": "a,b",
                                                        "rows": [["1"]]}
    assert canon_output(out, ["missing"]) == {}          # 声明了但没产出
    assert canon_output(out, []) == {}                   # 什么都没声明就什么都不收
    assert canon_output({}, ["header_row"]) == {}


def test_canon_output_is_order_independent():
    o1 = {"a": 1, "b": 2, "ok": True}
    o2 = {"ok": True, "b": 2, "a": 1}
    assert json.dumps(canon_output(o1, ["b", "a"])) == \
           json.dumps(canon_output(o2, ["a", "b"]))


def test_keys_are_hashable_json_serialisable_and_stable():
    """跨进程稳定：不用 hash()/id()，不吃 dict 插入顺序。两次构造 json.dumps 逐字相同。"""
    a = Record(step=1, action={"kind": "judge", "prompt": Q, "reads": ["x"]},
               output={"p": 1, "q": 2})
    b = Record(step=9, action={"reads": ["x"], "prompt": Q, "kind": "judge"},
               output={"q": 2, "p": 1})
    for mode in (False, True):
        ka, kb = canon_action(a, strict=mode), canon_action(b, strict=mode)
        assert ka == kb
        assert json.dumps(ka, ensure_ascii=False) == json.dumps(kb, ensure_ascii=False)
        assert isinstance(ka, tuple) and all(isinstance(p, str) for p in ka)
        assert len({ka, kb}) == 1                        # 可哈希、且是同一个键


# --------------------------------------------------------------------------- #
# 交叉验证：拿 table_clean 的记录表，比对既有两份实现的分组
# --------------------------------------------------------------------------- #
def _table_clean_records(make_traces):
    recs = [r for t in make_traces(6, seed=1) for r in t.records]
    kinds = {r.action.get("kind") for r in recs}
    assert recs and {"tool", "judge", "end"} <= kinds
    return recs


def test_loose_grouping_equals_replay_action_matches(make_traces):
    """松档在 table_clean 上与 replay._action_matches 的分组**逐对相同**。

    这是日后把 replay 指到 normalize 上的保险：比较的正是 replay 真实的用法——机器某状态的
    动作 × 轨迹某步的记录。
    """
    recs = _table_clean_records(make_traces)
    machine = tc.reference_machine()
    pairs = 0
    for state in machine.states.values():
        for rec in recs:
            pairs += 1
            assert replay._action_matches(state.action, rec.action) == \
                same_action(state.action, rec), (state.id, rec.step, rec.action)
    assert pairs > 100                                   # 真跑了一张表，不是空循环


def test_strict_grouping_equals_compiler_sig(make_traces):
    """严档在 table_clean 上与 compiler._sig 的分组**逐对相同**（重指编译器的保险）。"""
    recs = _table_clean_records(make_traces)
    for x in recs:
        for y in recs:
            assert (compiler._sig(x.action) == compiler._sig(y.action)) == \
                same_action(x, y, strict=True), (x.action, y.action)


def test_records_of_the_same_step_land_in_one_class(make_traces):
    """同一状态跑出来的记录，无论参数差多远，都归到同一个 KEY；不同状态互不相并。"""
    recs = _table_clean_records(make_traces)
    by_key: dict = {}
    for r in recs:
        by_key.setdefault(canon_action(r), set()).add(r.state)
    assert all(len(states) == 1 for states in by_key.values()), by_key
    assert len(by_key) == len({s for ss in by_key.values() for s in ss})
