"""动作规范化：把一条轨迹记录（或一台机器的动作）折成可比较的 **KEY**。

这是整套编译里最不起眼、也最容易悄悄错的一块。两处关键判断都建在它上面：

* **编译**要判「这一步和那一步是不是同一步」——同 ⇒ 接回旧状态（成环、合并 ≈-等价的历史，
  Myhill-Nerode 的状态最小性靠它），异 ⇒ 开一个新状态。折过头，两条语义不同的分岔被并成
  一个状态；折不够，每一步都独一无二，机器退化成一条没有环的直线。
* **回放**要判「机器挑的这一步，是不是轨迹实际走的那一步」。折过头会误判复述成功；折不够
  会让任何机器都复述不了任何轨迹。

仓库里本来就有两份实现，而且它们**故意不一样**：

* :func:`hexis.legacy.compiler._sig`（严）——工具只按名字并，但判断动作要比 ``prompt``、终止
  动作要比 ``terminal``。编译期必须这么严：两个提问不同的判断是两个不同的语义分岔，并掉就
  等于把「表头规不规范」和「金额对不对」当成同一步。
* :func:`hexis.legacy.replay._action_matches`（松）——只比 ``kind``，工具再加名字。回放期必须
  这么松：轨迹里的动作参数是**具体值**（渲染过的 prompt、填好的 input），机器里的是**模板**
  （``${var}``），逐字比一定不等；而 prompt 里带着题面，把它计进 KEY 会让每一步都独一无二。

所以这里**不统一它们**，而是把两套并成一个函数的两个具名档位：``strict=True`` 对应编译侧，
``strict=False``（默认）对应回放侧。谁该用哪档是语义问题，不是实现细节，因此写在签名里。

规则（``kind`` 恒参与）：

===========  ==========================================  ==============================
kind         loose（回放侧）                              strict（编译侧，额外再加）
===========  ==========================================  ==============================
``tool``     kind + 规范化后的工具名                       同左（工具不因参数分裂）
``judge``    kind + writes                               + ``prompt``
``model``    kind + writes                               + ``prompt``
``user``     kind + writes                               + ``prompt``
``end``      kind + ``terminal``                          同左（两档都比 terminal）
===========  ==========================================  ==============================

两档与既有两份实现的差：loose 比 ``_action_matches`` **略细**——多比了 ``end`` 的 terminal
与 judge/model 的 writes。这是有意的：``end`` 的 terminal 是「以哪种方式结束」，把 ok 与
give_up 并成一步会让拒绝集排除失效；而 writes 是「这一步往哪个变量写」，它决定后续条件读得
到什么，属于状态身份而非参数。在真实轨迹上（一个 end 状态一种 terminal、一个提问写一个变量）
两者的**分组完全一致**——test_14 拿 table_clean 的记录表逐对交叉验证了这一点，好让日后把
replay 指过来不改变行为。strict 同理比 ``_sig`` 略细（judge/model 多比 writes）。

``canon_action`` 同时吃 :class:`~hexis.machine.schema.Record`（``.action`` 是**裸 dict**）和
schema 的 Action 模型（ToolAction/ModelAction/JudgeAction/UserAction/EndAction）——编译器手上
是模型、轨迹里是 dict，同一步必须折出同一个 KEY，否则两侧一比就全错。识别走鸭子类型，**不
import** :mod:`hexis.machine.schema`：本模块因此零依赖、无环，任何一侧都能自由 import。

⚠️ 一个坑：judge/model 的 ``writes`` 在轨迹记录里**没有**单独字段（见 runtime._run_action：
judge 记的是 ``{kind,prompt,reads}``、model 记的是
``{kind,template_id,prompt,reads,prompt_sha256}``），只能从 ``Record.output``
的键反推（judge 的 output 就是 ``{写入变量: 标签}``，其余动作去掉 ``ok``/``error`` 这类状态
键——与 compiler._infer_writes 同一套约定）。所以**要传整条 Record，别只传 ``rec.action``**：
裸 action dict 拿不到 output，judge 的 writes 会退化成空（此时 strict 恰好退回 ``_sig`` 的
行为，不会更错，但也不会更准）。

同源性：judge 的 ``prompt`` 轨迹里有，所以 tool/judge/end 的**严档**跨源（记录 × 机器动作）
照样对得上。``model`` 从 runtime 那一轮起也进了这一档：``runtime._model_action`` 把**模板原文**
（不是渲染后的全文）记进 ``rec.action["prompt"]``，所以 runtime 产出的 model 记录与机器的
``ModelAction`` 严档也对得上。但这只对**记了 prompt 的**记录成立——``user`` 动作至今只记
``{kind}``（runtime 拒绝执行 user 动作），旧轨迹与外部轨迹也可能没这个字段；这类记录的严档
退化成 ``("model"/"user", "writes=…")``，与机器动作必然不等。跨源比较拿不准就用松档——回放正是
这么做的。test_14 把这条边界钉死了，免得日后当成偶然。

KEY 是 ``tuple[str, ...]``：可哈希、可 ``json.dumps``、跨进程稳定——不用 :func:`hash`、不用
:func:`id`、不吃 dict 的插入顺序（该排的都排过）。

**这里不做 LaTeX / 答案的规范化**（``\\frac12`` 与 ``\\frac{1}{2}`` 是否同一个答案）。那件事
要 sympy，住在 ``grader.py``，是**结果**的等价而不是**动作**的等价。本模块只用标准库，好让
结构检查、编译器、回放随便 import 而不背上重依赖。
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

__all__ = [
    "STATUS_KEYS", "action_writes", "canon_action", "canon_output",
    "canon_tool_name", "same_action",
]

#: 工具产出里表「这次调用成不成」的保留键：它们是状态位，不是写入变量。
#: 与 compiler._infer_writes 的 ``k != "ok"`` 同一套约定，这里连 ``error`` 一并排除。
STATUS_KEYS = frozenset({"ok", "error"})

#: 保留的**开局工具**名：编译时给每条轨迹虚拟地垫在最前面的那一步（``kind="tool"``、
#: 无入参、无产出），运行时是空操作。它让一台机器永远只有一个起点（``initial`` 恒为这个
#: 状态），真实的第一步变成它之后的普通分岔——math-skill 那 54 条 T+ 里 10 条开局动作不同
#: 导致的 ``start_mismatch`` 由此归零。名字必须是 :func:`canon_tool_name` 的不动点（下划线
#: 开头结尾会被削掉，所以不叫 ``__begin__``），且不可能与技能自己的工具撞名。
BEGIN_TOOL = "skill2fsm_begin"

#: 工具名里当作分隔符的字符：连字符与各种空白。
_SEP_RE = re.compile(r"[-\s]+")
_UNDERSCORE_RE = re.compile(r"_+")

#: 带 writes（+ 严格档才带自然语言字段）的动作类型 → 严格档要额外比的那个字段名。
_TEXT_FIELD = {"judge": "prompt", "model": "prompt", "user": "prompt"}


# --------------------------------------------------------------------------- #
# 工具名
# --------------------------------------------------------------------------- #
def canon_tool_name(name: str) -> str:
    """把一个工具名折成规范形：去目录、去 ``.py``、小写、``-``/空白 → ``_``。

    ``scripts/math_verify.py``、``math-verify``、``MATH_VERIFY`` 折成同一个 ``math_verify``。
    这不是洁癖：同一个脚本在文档里写路径、在轨迹里写裸名、在模型嘴里写连字符，是常态；不折
    到一起，一个 s3（校验）状态就会裂成三个互不成环的状态，编译出来的机器直接废掉。

    连续的 ``_`` 折成一个、首尾的 ``_`` 去掉（``math - verify`` 与 ``math__verify`` 同样收敛
    到 ``math_verify``）。空名返回空串。
    """
    s = str(name or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/").rsplit("/", 1)[-1]     # 只留最后一段（去掉目录部分）
    s = s.lower()
    if s.endswith(".py"):                           # 已小写，所以 .PY 也在这里被削掉
        s = s[:-3]
    s = _SEP_RE.sub("_", s)
    return _UNDERSCORE_RE.sub("_", s).strip("_")


# --------------------------------------------------------------------------- #
# 拆包：Record / 裸 record dict / 裸 action dict / Action 模型 → (action映射, output映射)
# --------------------------------------------------------------------------- #
def _model_fields(act: Any) -> dict:
    """从一个 Action 模型上取出参与 KEY 的那几个字段（鸭子类型，不 import schema）。"""
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
    """归一成 ``(动作映射, 产出映射)``。产出只有在能拿到整条记录时才非空。"""
    act = getattr(obj, "action", None)                       # Record（pydantic 模型）
    if isinstance(act, Mapping):
        out = getattr(obj, "output", None)
        return act, out if isinstance(out, Mapping) else {}
    if isinstance(obj, Mapping):
        inner = obj.get("action")
        if isinstance(inner, Mapping):                       # JSONL 直接读出来的裸记录
            out = obj.get("output")
            return inner, out if isinstance(out, Mapping) else {}
        return obj, {}                                       # 裸 action dict：没有 output
    if getattr(obj, "kind", None) is not None:               # schema 的 Action 模型
        return _model_fields(obj), {}
    raise TypeError(f"不认的动作载体 {type(obj).__name__}（要 Record / dict / Action 模型）")


# --------------------------------------------------------------------------- #
# writes：状态身份的一半（这一步往哪些变量写）
# --------------------------------------------------------------------------- #
def _writes_of(act: Mapping, out: Mapping) -> list[str]:
    """声明了就用声明的；没声明（轨迹记录的常态）就从 output 的键反推。已排序去重。"""
    declared = act.get("writes")
    if isinstance(declared, (list, tuple)):
        return sorted({str(w) for w in declared})
    kind = str(act.get("kind") or "")
    keys = {str(k) for k in out}
    if kind != "judge":                     # judge 的 output 就是 {写入变量: 标签}，全留
        keys -= STATUS_KEYS
    return sorted(keys)


def action_writes(rec_or_action: Any) -> list[str]:
    """这一步写入的变量名（排序去重）。传 Record 才能从 output 反推出未声明的 writes。"""
    act, out = _unwrap(rec_or_action)
    return _writes_of(act, out)


# --------------------------------------------------------------------------- #
# 动作 KEY
# --------------------------------------------------------------------------- #
def canon_action(rec_or_action: Any, *, strict: bool = False) -> tuple[str, ...]:
    """把一步动作折成 KEY。``strict=False`` 为回放档、``True`` 为编译档（差异见模块文档）。

    返回全是 ``str`` 的元组：可哈希、可 JSON、跨进程稳定。不认的 kind 只回 ``(kind,)``——与
    compiler._sig 的兜底一致，不臆造结构。
    """
    act, out = _unwrap(rec_or_action)
    kind = str(act.get("kind") or "")
    if kind == "tool":
        # 工具不因参数分裂：参数活在变量里，不属于状态身份。两档同一行为。
        # 唯一的例外是**已记下的阶段**（act["phase"]）：``bash``/``run_python`` 这种通用工具
        # 名字一样、用途不同，不细化就折成一个状态。阶段由 hexis.traces.phases 的纯函数在采集时
        # 判出，机器状态那侧也带同一个字段——**两侧对称**，所以它可以进共用的等价关系，
        # 与只在编译侧生效的 ``pred=``（往回看的语境）不同。没有阶段的动作行为完全照旧。
        key = ("tool", "name=" + canon_tool_name(act.get("name") or ""))
        ph = str(act.get("phase") or "")
        return key + (("phase=" + ph,) if ph else ())
    if kind == "end":
        # terminal 是「以哪种方式结束」，两档都比：并掉它，拒绝集的排除检查会失效。
        return ("end", "terminal=" + str(act.get("terminal") or "done"))
    if kind in _TEXT_FIELD:
        key = [kind, "writes=" + ",".join(_writes_of(act, out))]
        if strict:
            # 只有编译档带自然语言字段：prompt 里有题面，进了 KEY 每一步都独一无二。
            field = _TEXT_FIELD[kind]
            key.append(f"{field}=" + str(act.get(field) or ""))
        return tuple(key)
    return (kind,)


def is_begin(rec_or_action: Any) -> bool:
    """这一步是不是保留的开局工具 :data:`BEGIN_TOOL`。"""
    act, _out = _unwrap(rec_or_action)
    return (str(act.get("kind") or "") == "tool"
            and canon_tool_name(act.get("name") or "") == BEGIN_TOOL)


def context_key(rec_or_action: Any, preds: Sequence[Any] = (), *,
                k: int = 1) -> tuple[str, ...]:
    """严格档 KEY + 至多 ``k`` 个**前驱**的松档 KEY——编译侧的状态身份，只用于候选查找。

    规则：**身份可以往回看，不能往前看。** 同一个动作在不同前驱语境下可以是不同的状态
    （Myhill-Nerode：这两段历史其实不等价——math-skill 里所有 ``math_verify`` 调用折成一个
    状态、4 个后继分不开，就是 ``k=0`` 的后果）；但按后继分需要预知未来，那是条件与判断的
    活，不是身份的活。

    回放**不看**这个 KEY：:func:`hexis.legacy.replay._action_matches` 只比动作（松档），克隆
    状态的动作是深拷贝，所以身份细化对回放透明——test_08 的分裂早已证明这一点。
    ``k=0`` 就退化成 ``canon_action(strict=True)``。
    """
    base = canon_action(rec_or_action, strict=True)
    if k <= 0 or not preds:
        return base
    tail = tuple("pred=" + "|".join(canon_action(p, strict=False))
                 for p in list(preds)[-k:])
    return base + tail


def same_action(a: Any, b: Any, *, strict: bool = False) -> bool:
    """两步动作在给定档位下是不是「同一步」。两侧可以一边是 Record、一边是 Action 模型。"""
    return canon_action(a, strict=strict) == canon_action(b, strict=strict)


# --------------------------------------------------------------------------- #
# 产出裁剪
# --------------------------------------------------------------------------- #
def canon_output(out: Mapping, writes: Sequence[str]) -> dict:
    """按 ``writes`` 白名单裁剪一次产出：没声明的键一律丢掉，声明了又确实有的留下。

    「出参按 writes 白名单收」是运行时的既定纪律（runtime.rebuild）；这里做的是同一件事的
    **离线**版本，用来在比较两次产出前先去掉 ``ok`` 这类噪声。键按名字排序装回，因此
    ``json.dumps`` 的结果与两个 dict 当初的插入顺序无关。``writes`` 为空 ⇒ 返回空 dict
    （什么都没声明，就什么都不收）。
    """
    src = out or {}
    return {k: src[k] for k in sorted({str(w) for w in (writes or ())}) if k in src}
