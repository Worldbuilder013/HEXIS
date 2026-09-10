"""算法 1「顺序转写编译」的**编译智能体**那一半。确定性守门程序是 :mod:`skill2fsm.checker`。

:mod:`skill2fsm.compiler` 是一台一次成型的编译器：整轮算完、整轮检查、不过就整轮撤销。
一个**智能体式**的编译过程不是「一轮」，而是一串小提议——加个状态、接条边、把这个分岔收成
判断动作、给这个环配个计数器。所以本模块不再自己动 :class:`~skill2fsm.schema.Machine`：它
**只**通过 :class:`skill2fsm.checker.Checker` 的八个受票接口改机器，一条提议一条回执，被拒
的提议不牵连它之前被接受的提议。本模块因此**不 import**、也不需要 ``save_machine``。

模型可以出现在哪里，以及不能出现在哪里
--------------------------------------
方法的核心主张是：**编译产物是确定性的，智能体的不确定性在编译期一次性付清**。所以模型只
许出现在四处（:data:`MODEL_TOUCHPOINTS`）：

(a) **新步 vs 重复**的语义判定（L6）；
(b) **条款归属**（这一步落实文档哪一句）；
(c) 分岔学不出确定条件时，**起草判断动作**的提问与标签集（L10，:func:`draft_judge`）；
(d) 给那个判断动作**标定误差率**（:func:`skill2fsm.fit.calibrate`）。

除此之外一律不碰模型：对齐、建状态、接边、成环、学分岔条件、算循环上限、支持度裁剪、验收
——全是可复算的确定性计算。**每一次模型回复都过一遍模式检查**；解析不出来或不合模式的回复
是一次 **REJECT，不是猜**。同一个点上连续两次被拒（模型回复不合模式，或守门程序拒了提议）
就 :meth:`~skill2fsm.checker.Checker.demote_to_fallback` 然后往下走——**宁可少编，不编错**。

``model=None`` 必须能跑（密闭自测走的就是这条路），此时退到确定性启发式：

* 新步 vs 重复 —— 按规范化动作 KEY（:func:`skill2fsm.normalize.canon_action`，严档）判；
* 条款归属 —— **一律留空**（归属是语义判断，没有模型就不假装有）；
* 判断动作 —— **不起草**（轨迹里本来就有的判断步照常转写，那不是起草）。

两趟，以及为什么必须是两趟
--------------------------
守门程序的每个建边接口（``add_state`` 的 ``from_support``、``add_transition`` /
``close_loop`` 的 ``support``）都要求**建边时就给出支持度**——八个接口里没有一个能事后给
一条已有的边补记支持度。而 :func:`skill2fsm.verify.verify_machine` 又要求每条非回退边的
支持度 ≥ ``min_support``。于是「一边顺序走一边建边」在第一条轨迹上就会把所有边钉死在
support=1，验收必挂、整批回滚。所以本模块把算法 1 拆成两趟，**决策的顺序仍然是轨迹的顺序**：

* **第一趟 转写（**:func:`transcribe`**）** —— 按「步数少的优先」逐条轨迹、逐个动作走，
  在一份*台账*上做算法 1 的 L3–L11 的全部**决策**（新步/重复/分岔、条款归属、判断标签、
  访问次数、支持度）。这一趟**不改任何机器**，因此也不需要守门程序：它是智能体的思考。
* **第二趟 落账（**:func:`apply_plan`**）** —— 把决策序列按原顺序翻成受票提议，带上最终的
  支持度，逐条交给守门程序裁决。每条提议一张回执；连拒两次就在那个点退回解释执行。

代价说在明处：第二趟里被拒的提议无法回过头去改第一趟的决策（例如互斥冲突要到落账才暴露）。
这时走的是同一条退路——连拒两次 ⇒ ``demote_to_fallback``，那一段退回解释执行。

回边为什么一定带条件
--------------------
``Checker.close_loop`` 在源状态已有兜底边时**拒绝无条件回边**，而 ``add_state`` 建出来的
状态天然带一条通往 FALLBACK 的兜底边——也就是说，受票接口下**建不出无条件回边**。这不是绕
过去的坑，是它想要的形状：回边必须自带「什么时候该再绕一圈」的谓词，兜底位留给
FALLBACK（绕不动了就退回解释执行）。所以本模块给每条回边学一个在该状态**所有观测快照上恒
真**的谓词（:func:`skill2fsm.fit.separating`，``others`` 为空），学不出就把这个环整个放弃。

判断动作只在原地改写
--------------------
L10 的「起草一个判断动作」在真实轨迹上有一条硬边界：回放
（:func:`skill2fsm.replay._action_matches`）逐步比动作，**凭空插一个判断状态**会让机器比
轨迹多走一步，那条轨迹立刻复述不出来。所以本模块只在**这一步本来就是判断步**（``kind ==
"judge"``）时才起草——改写它的提问与标签集，``writes`` 保持不动，松档 KEY 因此不变、回放照旧
对得上。分岔落在工具步上而条件又学不出来，就是学不出来：整个分岔退回 FALLBACK。

交付物
------
:class:`CompileResult` 除机器外还交出**覆盖报告**（``coverage``）：逐条款的支持/单薄/无轨迹
触达，哪些结构来自文档、哪些是编译器自己加的（**循环上限 K 是编译器加的——SKILL.md 没写过
任何圈数上限**），回退面有多大，以及再补哪些轨迹最值钱。``coverage`` 是结构化数据，
:func:`skill2fsm.report.render` 直接渲染得了。
"""

from __future__ import annotations

import json as _json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from . import checker as _checker
from . import cond as _cond
from . import compiler as _compiler
from . import fit as _fit
from . import replay as _replay
from . import report as _report
from . import runtime as _runtime
from . import verify as _verify
from .checks import structural_findings
from .normalize import canon_action, canon_tool_name
from .schema import (
    FALLBACK, JudgeAction, Machine, Thresholds, Trace, Variable,
)

__all__ = [
    "ABSTAIN", "HARNESS_PRIMITIVES", "MAX_STRIKES", "MODEL_TOUCHPOINTS", "SKELETON_EXAMPLE",
    "SKELETON_FORMAT",
    "ClauseRow", "CompileResult", "Plan", "PlanResult", "Proposal", "apply_plan",
    "clause_rows", "compile_skill", "draft_judge", "markdown_clauses", "transcribe",
]

#: 判断动作的弃权标签。全仓一致（见 :class:`skill2fsm.schema.JudgeAction`）。
ABSTAIN = "弃权"

#: 同一个点上连续多少次被拒就退回解释执行。**宁可少编，不编错。**
MAX_STRIKES = 2

#: 模型**唯一**允许出现的几处。别处出现模型调用就是这套方法的自我否定。
#: 前四处是单智能体编译器的；后三处是多智能体编译（skill2fsm/agents）加的——同样过模式
#: 检查、同样 REJECT 不猜，契约全部登记在 :data:`REGISTRY`，智能体只能经
#: :class:`TouchpointGuard` 按登记的契约问模型。
MODEL_TOUCHPOINTS: tuple[str, ...] = (
    "new_or_repeat",        # (a) L6 新步 vs 重复
    "clause_attribution",   # (b) 这一步落实文档哪一条条款
    "draft_judge",          # (c) L10 起草判断动作的提问与标签集
    "calibrate_judge",      # (d) 给那个判断动作标定误差率
    "introduce_judge",      # (e) 从文档条款引入一个判断动作（轨迹里本没有这一步）
    "split_context",        # (f) 同一动作在两种前驱语境下是不是同一步
    "annotate_judge",       # (g) 在线采集探针：按判断的问题给当前快照打标签
    "draft_skeleton",       # (h) 文档 → 骨架机器（文档先行编译的第一步；轨迹随后在线标定）
    "classify_clauses",     # (i) 流水线起草：一节条款逐条判「步骤 / 约束 / 跳过」与「谁做、哪个阶段」
)

#: 问条款归属时最多摆多少个候选标签（269 条条款全塞进标签集没有意义）。
_CLAUSE_LABEL_CAP = 60

#: 验收不过时最多修几轮（每轮把「肇事状态」退回解释执行再验一次）。
_MAX_REPAIR = 4

_Q_NEW_OR_REPEAT = "这一步是流程里新的一步，还是回到之前已经走过的某一步？"
_Q_CLAUSE = "这一步在落实技能文档的哪一条条款？拿不准就弃权。"

#: ``draft_skeleton`` 触点里给模型看的 **efsm-v1 格式说明**。原来的说明只说了规矩没说形状，
#: 模型只能凭「efsm-v1」四个字猜字段名，猜错一个就过不了 Machine 校验，而这个触点只给一次
#: 机会。样例是一台真机器：test_39 钉住它能过 Machine 校验与结构检查，说明因此不会与 schema
#: 漂移。
#: 机器里 tool 状态允许的**全部**名字：harness 的两个原语。不是工具库，起草时不接任何工具清单。
HARNESS_PRIMITIVES: tuple[str, ...] = ("bash", "file_ops")

SKELETON_EXAMPLE: dict = {
    "format": "efsm-v1", "skill_id": "example", "initial": "s1", "fallback": "FALLBACK",
    "max_steps": 24,
    "variables": [
        {"name": "input_path", "type": "string", "init_from": "task.input.input_path"},
        {"name": "output_path", "type": "string", "init_from": "task.input.output_path"},
        {"name": "content", "type": "string", "init": ""},
        {"name": "plan", "type": "string", "init": ""},
        {"name": "apply_cmd", "type": "string", "init": ""},
        {"name": "verify_cmd", "type": "string", "init": ""},
        {"name": "plan_conf", "type": "string", "init": ""},
        {"name": "returncode", "type": "integer", "init": 0},
        {"name": "repair_count", "type": "integer", "init": 0},
    ],
    "states": {
        "s1": {"id": "s1", "clause": "S1",
               "action": {"kind": "tool", "name": "file_ops", "phase": "probe",
                          "input": {"op": "read", "path": "${input_path}"},
                          "reads": ["input_path"], "writes": ["content"]},
               "transitions": [{"if": "empty(content)", "to": "FALLBACK"},
                               {"to": "s2"}]},
        "s2": {"id": "s2", "clause": "S2",
               "action": {"kind": "model",
                          "prompt": "按文档拟一份最小改动计划，并写出执行它的 shell 命令与回读核对的 shell 命令",
                          "reads": ["content", "input_path", "output_path"],
                          "writes": ["plan", "apply_cmd", "verify_cmd"]},
               "transitions": [{"if": "repair_count >= 3", "to": "s7"},
                               {"to": "s3"}]},
        "s3": {"id": "s3", "clause": "S2",
               "action": {"kind": "judge", "prompt": "这份计划的证据充分吗？拿不准就弃权。",
                          "reads": ["plan"], "writes": ["plan_conf"],
                          "labels": ["充分", "不足", "弃权"], "abstain": "弃权"},
               "transitions": [{"if": "plan_conf == '充分'", "to": "s4"},
                               {"if": "plan_conf == '不足'", "to": "s2", "inc": "repair_count"},
                               {"to": "FALLBACK"}]},
        "s4": {"id": "s4", "clause": "S2",
               "action": {"kind": "tool", "name": "bash", "phase": "apply",
                          "input": {"command": "${apply_cmd}"},
                          "reads": ["apply_cmd"], "writes": ["returncode"]},
               "transitions": [{"if": "returncode != 0", "to": "s2", "inc": "repair_count"},
                               {"to": "s5"}]},
        "s5": {"id": "s5", "clause": "S3",
               "action": {"kind": "tool", "name": "bash", "phase": "verify",
                          "input": {"command": "${verify_cmd}"},
                          "reads": ["verify_cmd"], "writes": ["returncode"]},
               "transitions": [{"if": "returncode == 0", "to": "s6"},
                               {"if": "returncode != 0", "to": "s2", "inc": "repair_count"},
                               {"to": "FALLBACK"}]},
        "s6": {"id": "s6", "clause": "S3", "action": {"kind": "end", "terminal": "END_VERIFIED"}},
        "s7": {"id": "s7", "clause": "S2", "action": {"kind": "end", "terminal": "END_UNVERIFIED"}},
        "FALLBACK": {"id": "FALLBACK", "action": {"kind": "end", "terminal": "END_FALLBACK"}},
    },
    "terminals": [{"id": "END_VERIFIED", "kind": "verified"},
                  {"id": "END_UNVERIFIED", "kind": "unverified"},
                  {"id": "END_FALLBACK", "kind": "fallback"}],
    "audit_tools": ["bash"],
    "phase_rules": "default",
}

SKELETON_FORMAT = """\
efsm-v1 的形状（JSON 对象）：
- 顶层：format="efsm-v1"、skill_id、initial（起点状态 id）、fallback="FALLBACK"、max_steps、
  states（id → 状态）、variables、terminals、audit_tools。
- 状态：{id, clause, action, transitions, origin}。id 用 s1、s2…；clause 是它落实的条款 id。
  origin 标这个状态**凭什么存在**："document" = 文档明确要求的一步（如"修改后必须回读核对"），
  "compiler" = 你为了把流程编成机器而做的实现选择（如插一个"计划够不够"的判断、设一个重试上限）。
  transitions 里每条边也可以带 origin，同一口径。后续用真实轨迹修正机器时，document 的部分是约束、
  不许被绕过；compiler 的部分允许被改。拿不准就写 document。
- action 四种，按 kind 区分：
  tool  {name, phase, input, reads, writes}  执行一步。name 只能是 bash 或 file_ops：
        bash     input={"command": "..."}，产出键 returncode / stdout；
        file_ops input={"op": "read"|"write"|"list", "path": "...", "content"?: "..."}，读时产出键 content；
        writes 里可以用**语义名**（如 workbook_content），这时加 binds={"stdout": "workbook_content"}
        说明它由哪个产出键承载；不加 binds 的 writes 名必须就是产出键本身；
        phase 是这一步的用途：probe（读输入看现状）/ apply（写产出）/ verify（回读自己刚写的核对）。
        命令与路径里随任务变的部分写成 "${变量}"，由前面某个 model 状态产出；
  model {prompt, reads, writes}       模型写一段内容，出参按 writes 收；
  judge {prompt, reads, writes, labels, abstain}  固定提问、答案锁在 labels 里，labels 必含 "弃权"，
        答案写进 writes 的那个变量，之后的分岔只读它；
  end   {terminal}                    停机，terminal 指向 terminals 里的一项。
- transitions：[{if, to, inc}]。if 是变量上的谓词，**只有**这几种写法：
  x == 'A'、x != 'A'、n >= 3、n < 3、empty(x)、nonempty(x)，用中缀 and / or / not 连接，
  例如 "returncode == 0 and empty(stdout)"。不是函数调用：and(...) 、&&、|| 一律非法；
  只能引用 variables 里声明过的名字。没有 if 的是兜底边，一个状态至多一条、最后求值。
  同一状态各出边必须两两互斥且覆盖全部取值。inc 是走这条边时 +1 的计数变量。
- 回边（能绕回来的边）必须带 inc，且被绕回的那个状态要有一条 "计数 >= K" 的出口边。
- variables：[{name, type, init | init_from}]，type ∈ string/integer/number/boolean/array/object；
  任务输入用 init_from="task.input.<键>"，计数变量 init=0。每个状态读的变量在通往它的每条
  路径上都要先被写过。
- terminals：[{id, kind}]，kind ∈ verified / unverified / fallback。必须有 id 为 FALLBACK 的状态，
  action 是 end、指向 kind=fallback 的终点；任何状态拿不准时都可以有一条边去 FALLBACK。
- audit_tools：["bash"]。kind=verified 的终点只能经 phase=verify 的 bash 状态到达，且那条边的 if
  要读它写出的 returncode。
- phase_rules："default"。
最小样例：
"""

_JUDGE_PROMPT = """\
你在把一份技能文档编译成状态机。状态 {state} 之后出现了一个分岔：同样的一步之后，
执行有时走向 {targets}，而这些分支**无法**用现有变量上的确定性谓词分开。

请起草一次**固定提问**，让它的答案能决定该走哪一支。要求：
1. 只读这些变量：{reads}；
2. 答案锁死在一个有限标签集里，每个分支一个标签，另加一个弃权标签 {abstain}；
3. 回一个 JSON 对象：{{"prompt": "...", "labels": ["...", ...], "abstain": "{abstain}"}}。

分岔两侧的变量快照样例：
{samples}
"""


# --------------------------------------------------------------------------- #
# 触点登记表：每个触点的提问、reads 白名单、回复模式、REJECT 规则、model=None 退路
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Touchpoint:
    """一个模型触点的**全部**契约。智能体只能通过 :class:`TouchpointGuard` 按它问模型。

    ``reads`` 是允许传给模型的值键白名单（交集规则，同 draft_judge 的 ``want = [...]``）；
    ``parse(raw, values) -> dict | None`` 做模式检查，``None`` 即 REJECT；``fallback(values)``
    是 ``model=None`` 时的确定性退路（``None`` 表示「不做」）；``allowed_ops`` 限定这个触点
    的提案批里能出现哪些受票接口（空 = 不发提案，只写字段）。
    """

    id: str
    prompt: str
    reads: frozenset
    reply_kind: str                         # "classify" | "generate"
    parse: Any
    reject_rules: tuple = ()
    fallback: Any = None
    max_strikes: int = MAX_STRIKES
    allowed_ops: frozenset = frozenset()


def _parse_in_labels(raw: Any, values: Mapping) -> Optional[dict]:
    labels = list(values.get("labels") or ())
    if not isinstance(raw, str) or raw not in labels or raw == ABSTAIN:
        return None
    return {"label": raw}


def _parse_draft_judge(raw: Any, _values: Mapping) -> Optional[dict]:
    if not isinstance(raw, Mapping):
        return None
    if not isinstance(raw.get("prompt"), str) or not raw["prompt"].strip():
        return None
    if not isinstance(raw.get("labels"), (list, tuple)) or len(raw["labels"]) < 2:
        return None
    return dict(raw)


def _parse_introduce_judge(raw: Any, values: Mapping) -> Optional[dict]:
    """``{clause_id, prompt, labels[2..9], abstain?, reads?, label_locators{label: locator}}``。

    REJECT：非 Mapping；prompt 空；标签少于 2 或多于 9、有重复；``clause_id`` 不在给的
    条款表里；给了 ``label_locators`` 却有标签没定位；``reads`` 越出白名单的部分被**丢弃**
    （同 draft_judge：模型回一个白名单外的变量名，就是在给判断偷偷加上下文）。
    """
    if not isinstance(raw, Mapping):
        return None
    q = raw.get("prompt")
    labs = raw.get("labels")
    cid = str(raw.get("clause_id") or raw.get("clause") or "")
    if not isinstance(q, str) or not q.strip() or not isinstance(labs, (list, tuple)):
        return None
    labs = list(dict.fromkeys(str(x) for x in labs
                              if isinstance(x, (str, int, float)) and not isinstance(x, bool)))
    labs = [l for l in labs if l != ABSTAIN]
    if len(labs) < 2 or len(labs) > 9:
        return None
    rows = set(values.get("clause_ids") or ())
    if rows and cid not in rows:
        return None
    locs = raw.get("label_locators")
    if locs is not None and not isinstance(locs, Mapping):
        return None
    if isinstance(locs, Mapping) and any(l not in locs for l in labs):
        return None
    want = list(values.get("reads") or ())
    reads = raw.get("reads")
    if reads is not None:
        if not isinstance(reads, (list, tuple)):
            return None
        reads = [r for r in reads if r in set(want)]
    return {"clause": cid, "prompt": q.strip(), "labels": labs,
            "abstain": ABSTAIN, "reads": list(reads) if reads else want,
            "label_locators": dict(locs or {})}


_ROLES = ("step", "constraint", "skip")
_KINDS = ("tool", "model", "judge")
_PHASES = ("probe", "apply", "verify")


def _parse_classify_clauses(raw: Any, values: Mapping) -> Optional[dict]:
    """``{"items": [{clause_id, role, kind?, phase?, summary?}, ...]}``。

    不在条款表里的 id 丢弃；role 不在三值里丢弃；role=step 却缺 kind、或 kind=tool 却缺
    phase 的条目丢弃。一条合法的都没有 ⇒ REJECT。没提到的条款由流水线按「约束」补。
    """
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except ValueError:
            return None
    if not isinstance(raw, Mapping):
        return None
    items = raw.get("items")
    if not isinstance(items, (list, tuple)):
        return None
    ids = set(values.get("clause_ids") or ())
    out = []
    for it in items:
        if not isinstance(it, Mapping):
            continue
        cid = str(it.get("clause_id") or it.get("id") or "")
        role = str(it.get("role") or "").strip().lower()
        kind = str(it.get("kind") or "").strip().lower()
        phase = str(it.get("phase") or "").strip().lower()
        if (ids and cid not in ids) or role not in _ROLES:
            continue
        if role == "step":
            if kind not in _KINDS:
                continue
            if kind == "tool" and phase not in _PHASES:
                continue
        out.append({"clause_id": cid, "role": role, "kind": kind if role == "step" else "",
                    "phase": phase if (role == "step" and kind == "tool") else "",
                    "summary": str(it.get("summary") or "")[:200]})
    return {"items": out} if out else None


def _parse_split_context(raw: Any, _values: Mapping) -> Optional[dict]:
    if not isinstance(raw, str) or raw not in ("同一步", "不同步", ABSTAIN):
        return None
    return {"label": raw}


REGISTRY: dict[str, Touchpoint] = {
    "new_or_repeat": Touchpoint(
        "new_or_repeat", _Q_NEW_OR_REPEAT,
        frozenset({"当前位置", "这一步", "同型的已有状态", "labels"}), "classify",
        _parse_in_labels, ("回复不在标签集", "弃权"), None, MAX_STRIKES,
        frozenset({"add_state", "add_transition", "close_loop", "set_terminal"})),
    "clause_attribution": Touchpoint(
        "clause_attribution", _Q_CLAUSE, frozenset({"这一步", "候选条款", "labels"}),
        "classify", _parse_in_labels, ("回复不在候选条款里",),
        lambda v: {"label": ""}, MAX_STRIKES, frozenset()),
    "draft_judge": Touchpoint(
        "draft_judge", _JUDGE_PROMPT,
        frozenset({"state", "reads", "targets", "abstain", "samples", "writes", "clause"}),
        "generate", _parse_draft_judge, ("非 Mapping / prompt 空 / labels 少于 2",),
        None, MAX_STRIKES, frozenset({"add_judge"})),
    "calibrate_judge": Touchpoint(
        "calibrate_judge", "（判断动作自己的提问）", frozenset(), "classify",
        lambda raw, v: ({"label": raw} if isinstance(raw, str) else None), (), None,
        MAX_STRIKES, frozenset()),
    "introduce_judge": Touchpoint(
        "introduce_judge",
        "文档里哪一句要求在这一步做一次判定？给出固定提问、有限标签集（每个标签在该条款原文"
        "里的定位）、只读这些变量。回一个 JSON 对象："
        "{\"clause_id\": ..., \"prompt\": ..., \"labels\": [...], \"abstain\": \"弃权\", "
        "\"reads\": [...], \"label_locators\": {label: locator}}",
        frozenset({"state", "clause_text", "clause_ids", "reads", "prev_action",
                   "next_actions", "samples"}),
        "generate", _parse_introduce_judge,
        ("clause_id 不在条款表", "标签 <2 或 >9", "标签无原文定位", "reads 越出白名单"),
        None, 1, frozenset({"add_judge"})),
    "split_context": Touchpoint(
        "split_context",
        "同一动作在这两种前驱语境下是不是同一步？回「同一步」/「不同步」/「弃权」。",
        frozenset({"action", "pred_a", "pred_b", "sample_vars", "labels"}), "classify",
        _parse_split_context, ("不在三个标签里",), None, MAX_STRIKES,
        frozenset({"split_state"})),
    "annotate_judge": Touchpoint(
        "annotate_judge", "（判断动作自己的提问）", frozenset(), "classify",
        _parse_in_labels, ("不在标签集 ⇒ 写弃权",), None, MAX_STRIKES, frozenset()),
    "draft_skeleton": Touchpoint(
        "draft_skeleton",
        "把这份技能文档转写成一台 efsm-v1 状态机（JSON）。规矩：每个状态挂一条条款（clause 用给定"
        "的条款 id）；工具入参里凡是随任务变化的值一律写成 ${变量}，并让前面某个 model 状态"
        "写出那个变量；分岔用变量条件或判断动作；回边配计数变量与上限；verified 终点只能经审计"
        "工具到达。只回 JSON 对象。\n"
        "不给你任何工具清单：tool 状态的 name 只能是 bash 或 file_ops 这两个原语，具体命令由"
        "前面的 model 状态按文档现写。VARIABLES 里 clauses 是条款表（id → 原文）：每个状态的 clause"
        "填它最贴近的那一条 id；一个状态可以覆盖多条条款，不必一条一个状态，也不要照抄样例的"
        "状态数。不要在推理里逐条推演，想清主干就直接写 JSON。\n"
        "若 VARIABLES 里带 previous 与 errors：previous 是你上一版机器，errors 是确定性检查对它"
        "报的错；只修这些错，其余保持不动，仍回**完整**的机器 JSON。\n\n" + SKELETON_FORMAT
        + _json.dumps(SKELETON_EXAMPLE, ensure_ascii=False, indent=1),
        frozenset({"doc", "clause_ids", "clauses", "tool_names", "input_keys", "audit_tools",
                   "skill_id", "previous", "errors"}),
        "generate", None, ("不是合法 efsm-v1", "结构检查有 error", "工具名不在允许集"),
        None, 1, frozenset({"open_machine", "add_state", "add_transition", "close_loop",
                            "add_judge", "set_terminal"})),
    "classify_clauses": Touchpoint(
        "classify_clauses",
        "下面是技能文档的一节，逐条条款判定。role 三选一：step（这一条要求执行一个动作）、"
        "constraint（对怎么做的限定，不单独成一步）、skip（标题、背景、与执行无关）。role=step 时"
        "再给 kind：tool（要跑命令或读写文件；再给 phase：probe=读输入看现状 / apply=写产出 / "
        "verify=回读自己刚写的核对）、model（要想、要写内容、要拟计划）、judge（要在有限几种情形里"
        "判定一种，之后走法不同）。按条款在文档里的顺序回。只回一个 JSON 对象："
        "{\"items\": [{\"clause_id\": ..., \"role\": ..., \"kind\": ..., \"phase\": ..., "
        "\"summary\": \"十字以内\"}]}",
        frozenset({"section", "clauses", "clause_ids", "primitives"}),
        "generate", _parse_classify_clauses, ("items 不是列表", "没有一条合法条目"),
        None, MAX_STRIKES, frozenset()),
}
# draft_skeleton 的 parse 需要 Machine 校验，定义在 agents/seed.py 里再回填，避免循环 import。
assert tuple(REGISTRY) == MODEL_TOUCHPOINTS, "登记表与 MODEL_TOUCHPOINTS 必须逐项一致"


class TouchpointGuard:
    """智能体拿到的**唯一**模型句柄。每次提问都过白名单与模式检查，不合就记振、不猜。

    ``ask`` 返回解析后的 dict，REJECT 返回 ``None``（并已在 ``ctx`` 上对 ``point`` 记一振）；
    ``model=None`` 时走触点的 ``fallback``（没有退路就 ``None``，什么都不做）。
    """

    def __init__(self, model: Any, ctx: "_Ctx", registry: Optional[Mapping] = None) -> None:
        self._model = model
        self.ctx = ctx
        self.registry = dict(registry or REGISTRY)
        self.calls = 0
        self.rejects: list[dict] = []

    @property
    def has_model(self) -> bool:
        return self._model is not None

    def ask(self, tp_id: str, values: Mapping, *, point: str = "",
            labels: Sequence[str] = (), history: tuple = ()) -> Optional[dict]:
        tp = self.registry.get(tp_id)
        if tp is None:
            self.ctx.strike(point, f"触点 {tp_id!r} 没登记")
            return None
        vals = dict(values)
        if labels:
            vals["labels"] = list(labels)
        extra = set(vals) - set(tp.reads) - {"labels"}
        if tp.reads and extra:
            self.ctx.strike(point, f"[E_READS_WHITELIST] 触点 {tp_id} 传了白名单外的值 "
                                   f"{sorted(extra)}")
            self.rejects.append({"touchpoint": tp_id, "why": "reads", "extra": sorted(extra)})
            return None
        if self._model is None:
            return tp.fallback(vals) if tp.fallback is not None else None
        self.calls += 1
        self.ctx.model_calls += 1
        try:
            if tp.reply_kind == "classify":
                labs = list(vals.get("labels") or [])
                raw = self._model.classify(
                    prompt=tp.prompt,
                    values={k: v for k, v in vals.items() if k != "labels"}, labels=labs)
            else:
                raw = self._model.generate(prompt=tp.prompt, values=vals, history=history)
        except Exception as exc:                                    # noqa: BLE001
            self.ctx.strike(point, f"触点 {tp_id} 调用失败：{type(exc).__name__}: {exc}")
            return None
        parsed = tp.parse(raw, vals)
        if parsed is None:
            self.ctx.strike(point, f"触点 {tp_id} 的回复不合模式：{str(raw)[:120]!r}")
            self.rejects.append({"touchpoint": tp_id, "why": "schema", "raw": str(raw)[:200]})
            return None
        return parsed


# --------------------------------------------------------------------------- #
# 对外的数据形状
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proposal:
    """编译智能体的一条改写提议。

    ``op`` 是守门程序八个受票接口之一，``payload`` 是它的关键字参数，``rationale`` 是**这
    条提议凭什么**（人读），``trace_id``/``step`` 指回它是从哪条轨迹的第几步学出来的——
    审计一台机器时，这三样合起来回答「这条边凭哪条轨迹、哪条文档条款存在」。

    本类与 :class:`skill2fsm.checker.Proposal` 是两回事：那个是守门程序的入参形状（只有
    ``op``/``args``），这个多带溯源。:meth:`to_checker` 做转换。
    """

    op: str
    payload: dict = field(default_factory=dict)
    rationale: str = ""
    trace_id: str = ""
    step: int = 0
    #: 溯源行（:class:`skill2fsm.batch.Provenance` 或同形 dict）。多智能体模式下必带；
    #: 单智能体旧路径不带，守门程序也不要求。
    prov: Any = None

    def to_checker(self) -> _checker.Proposal:
        """翻成守门程序认的提议形状。带溯源就一并交给受票接口的 ``prov=``。"""
        args = dict(self.payload)
        if self.prov is not None and self.op not in ("commit", "mark", "rewind"):
            args["prov"] = (self.prov.to_dict() if hasattr(self.prov, "to_dict")
                            else dict(self.prov))
        return _checker.Proposal(op=self.op, args=args)

    @property
    def point(self) -> str:
        """这条提议**作用在哪个点**——连拒两次时退回解释执行的就是它。

        建边类提议的点是**边的源状态**（在那里编不下去了），没有源状态时才退而取被建的
        状态本身。``open_machine``/``commit`` 不落在任何状态上，返回空串。
        """
        p = self.payload
        return str(p.get("from_state") or p.get("state_id") or "")


@dataclass
class PlanResult:
    """一趟落账的结果：回执、接受/拒绝计数、退回解释执行的点、被跳过的提议。"""

    receipts: list = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    demoted: list[str] = field(default_factory=list)
    skipped: list[Proposal] = field(default_factory=list)


@dataclass
class CompileResult:
    """一次编译的全部交付物。``machine`` 是唯一的产物，其余都是它的账。"""

    machine: Machine
    receipts: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    judges: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    diff_vs_reference: Optional[dict] = None


@dataclass(frozen=True)
class ClauseRow:
    """条款表的一行。``locator`` 形如 ``SKILL.md:12-20``，没有就是空串。"""

    id: str
    title: str
    text: str
    locator: str = ""


# --------------------------------------------------------------------------- #
# 第一趟的台账
# --------------------------------------------------------------------------- #
@dataclass
class _PState:
    """台账里的一个状态：一步动作 + 它的出身。"""

    sid: str
    key: tuple
    kind: str                                   # tool / judge / model / end
    payload: dict = field(default_factory=dict)  # tool/model 的动作 dict
    clause: str = ""
    terminal: str = ""
    prompt: str = ""
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    labels_seen: list = field(default_factory=list)
    examples: list = field(default_factory=list)   # [(label, {read: value})]
    drafted: bool = False                          # 判断动作是不是模型起草的
    error_rate: float = 0.0
    support: int = 0
    origin: tuple = ("", 0)                        # (trace_id, step)
    # ---- 多智能体编译加的溯源与引入判断字段（默认空：单智能体产物逐字节不变） ---- #
    introduced: bool = False                       # 从文档引入的判断（轨迹里本没有这一步）
    gold_from: str = ""                            # 程序打标器名（trace_adapter.LABELERS）
    origin_kind: str = ""                          # document / trace / compiler …
    locator: str = ""                              # 条款原句位置


@dataclass
class _PEdge:
    """台账里的一条边。``cond`` 在第二段「定条件」时才填上。"""

    src: str
    dst: str
    support: int = 0
    back: bool = False
    creating: bool = False                          # 这条边同时把目标状态建出来
    cond: str = ""
    counter: str = ""
    bound: int = 0
    origin: tuple = ("", 0)


@dataclass
class Plan:
    """第一趟转写的产物：一份**还没落到任何机器上**的编译台账。

    它是智能体的思考痕迹——状态、边、支持度、分岔处的变量快照、判断动作观测到的标签、每个
    状态在单条轨迹里被进入的最大次数（定循环上限用）。第二趟据它生成受票提议。
    """

    skill_id: str = "compiled"
    states: dict = field(default_factory=dict)          # sid -> _PState
    order: list = field(default_factory=list)           # 状态创建顺序
    by_key: dict = field(default_factory=lambda: defaultdict(list))
    edges: dict = field(default_factory=dict)           # (src,dst) -> _PEdge
    edge_order: list = field(default_factory=list)
    out: dict = field(default_factory=lambda: defaultdict(list))
    initial: str = ""
    obs: dict = field(default_factory=lambda: defaultdict(list))   # sid -> [(dst, snap)]
    #: 与 obs 平行的语境账：sid -> [(dst, 前驱 sid, 进入 sid 前它在本条轨迹里已被访问的次数)]。
    #: 分裂智能体靠它算「按前驱 / 按访问次数能不能把分岔分开」的列联表。单独一份而不是把
    #: obs 的二元组撑成四元组：obs 的消费者（fit_guards/_unlink/_judge_branch）都按二元组解包。
    obs_ctx: dict = field(default_factory=lambda: defaultdict(list))
    #: 分岔被退回 FALLBACK 时它原本的去处（_block 记下）：分裂 / 引入判断的智能体要知道
    #: 「这个分岔本来分向哪几个状态」，而 _block 之后 out[p] 已经空了。
    blocked_targets: dict = field(default_factory=dict)
    #: sid -> 变量名：定这个状态的分岔条件时**只**看这个变量（引入的判断状态用它，让
    #: learn_cond 学出的谓词一定是「判断变量 == 标签」，而不是碰巧也能分开的别的变量）。
    guard_vars: dict = field(default_factory=dict)
    visits: dict = field(default_factory=lambda: defaultdict(int))
    var_types: dict = field(default_factory=dict)
    input_keys: set = field(default_factory=set)
    trace_states: dict = field(default_factory=lambda: defaultdict(list))
    blocked: set = field(default_factory=set)           # 分岔编不出来、只留兜底边的状态
    thin: list = field(default_factory=list)            # 支持度不足、被裁掉的边
    pruned: list = field(default_factory=list)          # 因此走不到、被剪掉的状态
    loop_bounds: list = field(default_factory=list)     # K 的取值台账
    notes: list = field(default_factory=list)           # 转写过程中说不下去的地方
    max_records: int = 0
    _n: int = 0

    # ---- 便捷 ---- #
    def new_sid(self) -> str:
        self._n += 1
        return f"s{self._n}"

    def reaches(self, src: str, dst: str) -> bool:
        """台账的图上 ``src`` 能不能走到 ``dst``（含 ``src is dst``）。"""
        seen, stack = {src}, [src]
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            for nxt in self.out.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return dst in seen

    def add_edge(self, src: str, dst: str, *, back: bool, creating: bool,
                 origin: tuple) -> _PEdge:
        e = self.edges.get((src, dst))
        if e is None:
            e = _PEdge(src=src, dst=dst, back=back, creating=creating, origin=origin)
            self.edges[(src, dst)] = e
            self.edge_order.append((src, dst))
            self.out[src].append(dst)
        return e


# --------------------------------------------------------------------------- #
# 编译上下文（模型触点 + 计数）
# --------------------------------------------------------------------------- #
@dataclass
class _Ctx:
    model: Any = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    rows: list = field(default_factory=list)
    doc: str = ""
    progress: bool = False
    model_calls: int = 0
    strikes: dict = field(default_factory=lambda: defaultdict(int))
    rejects: list = field(default_factory=list)
    #: 状态身份的钩子：``key_of(rec, prev_rec, counts) -> tuple``。默认 ``None`` =
    #: ``canon_action(rec, strict=True)``。多智能体的转写智能体用它把分裂智能体的
    #: ``Refinement`` 落成「同一动作、不同前驱 / 不同访问次数 ⇒ 不同身份」，而不改转写循环。
    #: ``counts`` 是本条轨迹内按基础 KEY 计数的 dict，钩子自己维护。
    key_of: Any = None

    def say(self, msg: str) -> None:
        if self.progress:
            print(f"[compile_agent] {msg}", file=sys.stderr)

    def strike(self, point: str, why: str) -> bool:
        """在一个点上记一次拒绝。返回**是否已经连拒到上限**（该退回解释执行了）。"""
        self.strikes[point] += 1
        self.rejects.append({"point": point, "why": why, "n": self.strikes[point]})
        self.say(f"点 {point} 第 {self.strikes[point]} 次被拒：{why}")
        return self.strikes[point] >= MAX_STRIKES

    def clear(self, point: str) -> None:
        self.strikes[point] = 0


# --------------------------------------------------------------------------- #
# 模型触点（四处，每处都过模式检查；不合模式 = REJECT，不是猜）
# --------------------------------------------------------------------------- #
def _brief(action: Mapping) -> str:
    """一步动作的短摘要，喂给模型看。私有面的 prompt 全文不进这里。"""
    kind = str(action.get("kind") or "")
    if kind == "tool":
        return f"tool:{canon_tool_name(action.get('name') or '')}"
    if kind == "judge":
        return f"judge:{str(action.get('prompt') or '')[:60]}"
    if kind == "end":
        return f"end:{action.get('terminal') or 'done'}"
    return kind or "?"


def _ask_new_or_repeat(ctx: _Ctx, point: str, rec: Any,
                       cands: Sequence[str]) -> Optional[str]:
    """(a) 新步 vs 重复。返回 ``"new"`` / 某个已有状态 id；不合模式返回 ``None``。"""
    labels = ["新步"] + [f"重复:{s}" for s in cands] + [ABSTAIN]
    values = {"当前位置": point, "这一步": _brief(rec.action),
              "同型的已有状态": ",".join(cands) or "（没有）"}
    ctx.model_calls += 1
    try:
        ans = ctx.model.classify(prompt=_Q_NEW_OR_REPEAT, values=values,
                                 labels=list(labels))
    except Exception:                                       # noqa: BLE001
        return None
    if not isinstance(ans, str) or ans not in labels or ans == ABSTAIN:
        return None
    if ans == "新步":
        return "new"
    return ans.split(":", 1)[1]


def _clause_candidates(rec: Any, rows: Sequence[ClauseRow]) -> list[ClauseRow]:
    """把候选条款先缩到一个能摆进标签集的范围。**这是检索，不是归属**——归属是模型的事。"""
    act = rec.action if isinstance(rec.action, Mapping) else {}
    name = canon_tool_name(act.get("name") or "")
    hits = [r for r in rows if name and (name in r.text.lower()
                                         or name.replace("_", "-") in r.text.lower()
                                         or name.replace("_", " ") in r.text.lower())]
    pool = hits or list(rows)
    return pool[:_CLAUSE_LABEL_CAP]


def _ask_clause(ctx: _Ctx, rec: Any) -> str:
    """(b) 条款归属。``model=None`` 或不合模式一律返回空串（**不假装归属**）。"""
    if ctx.model is None or not ctx.rows:
        return ""
    pool = _clause_candidates(rec, ctx.rows)
    if not pool:
        return ""
    labels = [r.id for r in pool] + [ABSTAIN]
    values = {"这一步": _brief(rec.action),
              "候选条款": " | ".join(f"{r.id} {r.title}" for r in pool)}
    ctx.model_calls += 1
    try:
        ans = ctx.model.classify(prompt=_Q_CLAUSE, values=values, labels=list(labels))
    except Exception:                                       # noqa: BLE001
        return ""
    if not isinstance(ans, str) or ans not in labels or ans == ABSTAIN:
        return ""
    return ans


def draft_judge(question_ctx: Mapping, *, model: Any) -> Optional[JudgeAction]:
    """(c) 分岔学不出确定条件时，**智能体起草**一个判断动作；守门程序随后校验。

    ``question_ctx`` 至少要给 ``reads``（允许读的变量白名单）与 ``targets``
    （``{目标状态: [变量快照]}``），可给 ``state``/``writes``/``clause``。

    返回 ``None`` 而不是抛异常的三种情形，都是**同一件事**——这次起草作废，调用方按一次
    拒绝处理：没有模型；模型调用炸了；回复解析不出来或不合模式（缺 ``prompt``、标签少于
    两个、标签不是标量）。**绝不拿半个回复凑一个判断动作**。

    ``reads`` 只认白名单里的变量：模型回一个白名单外的变量名，就是在给这个判断动作偷偷加
    上下文，而窄读正是这套编译要守住的东西。
    """
    if model is None:
        return None
    ctx = dict(question_ctx or {})
    reads = [str(r) for r in (ctx.get("reads") or [])]
    if not reads:
        return None
    writes = [str(w) for w in (ctx.get("writes") or [])]
    if not writes:
        writes = [f"{ctx.get('state') or 'branch'}_verdict"]
    targets = list((ctx.get("targets") or {}))
    abstain = str(ctx.get("abstain") or ABSTAIN)
    prompt = _JUDGE_PROMPT.format(
        state=ctx.get("state") or "?", targets=", ".join(targets) or "?",
        reads=", ".join(reads), abstain=abstain,
        samples=_sample_text(ctx.get("targets") or {}, reads))
    try:
        raw = model.generate(prompt=prompt, values={"reads": reads, "targets": targets})
    except Exception:                                       # noqa: BLE001
        return None
    if not isinstance(raw, Mapping):
        return None
    prompt = raw.get("prompt")
    labels = raw.get("labels")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if not isinstance(labels, (list, tuple)):
        return None
    labs: list[str] = []
    for x in labels:
        if isinstance(x, (str, int, float)) and not isinstance(x, bool):
            s = str(x).strip()
            if s and s not in labs:
                labs.append(s)
    if len(labs) < 2:
        return None
    abstain = str(raw.get("abstain") or abstain)
    if abstain not in labs:
        labs.append(abstain)
    want = [r for r in (raw.get("reads") or reads) if r in reads] or reads
    try:
        return JudgeAction(prompt=prompt.strip(), reads=want, writes=writes,
                           labels=labs, abstain=abstain)
    except (ValidationError, ValueError):
        return None


def _sample_text(targets: Mapping, reads: Sequence[str]) -> str:
    lines = []
    for tgt in list(targets)[:4]:
        for snap in list(targets[tgt])[:2]:
            vals = ", ".join(f"{k}={snap.get(k)!r}" for k in reads)
            lines.append(f"  → {tgt}: {vals}")
    return "\n".join(lines) or "  （无）"


def _calibrate(ctx: _Ctx, judge: JudgeAction,
               samples: Sequence[tuple]) -> Optional[float]:
    """(d) 标定判断动作的误差率。标定不了返回 ``None``（调用方按拒绝处理）。"""
    if ctx.model is None or not samples:
        return None
    ctx.model_calls += len(samples)
    try:
        return float(_fit.calibrate(judge, list(samples), model=ctx.model))
    except Exception:                                       # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# 第一趟：顺序转写（算法 1 L2–L11 的决策）
# --------------------------------------------------------------------------- #
def _tid(trace: Trace) -> str:
    task = trace.task if isinstance(trace.task, dict) else {}
    return str(task.get("task_id") or "")


def _type_of(v: Any) -> Optional[str]:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, (list, tuple)):
        return "array"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, str):
        return "string"
    return None


def _order_traces(t_plus: Sequence[Trace]) -> list[tuple[int, Trace]]:
    """L2 的「步数少的优先」。同长度按 task_id、再按原下标，保证**顺序完全确定**。"""
    return sorted(enumerate(t_plus),
                  key=lambda it: (len(it[1].records), _tid(it[1]), it[0]))


def _create_state(plan: Plan, sid: str, rec: Any, key: tuple, prev_vars: dict,
                  ctx: _Ctx, tid: str) -> _PState:
    """L7：把一条记录的动作装成一个新状态（含条款归属、reads/writes）。"""
    act = rec.action if isinstance(rec.action, Mapping) else {}
    kind = str(act.get("kind") or "")
    clause = _ask_clause(ctx, rec)
    st = _PState(sid=sid, key=key, kind=kind, clause=clause,
                 origin=(tid, int(getattr(rec, "step", 0) or 0)))
    if kind == "end":
        st.terminal = str(act.get("terminal") or "done")
    elif kind == "judge":
        st.prompt = str(act.get("prompt") or "")
        st.reads = [str(r) for r in (act.get("reads") or [])]
        st.writes = _compiler._infer_writes(dict(act), dict(rec.output or {}))
        if not st.reads:
            st.reads = sorted(k for k in prev_vars if k not in st.writes)
        if not st.writes:
            st.writes = [f"{sid}_verdict"]
    elif kind == "model":
        st.reads = ([str(r) for r in (act.get("reads") or [])]
                    or _compiler._infer_reads(dict(act), prev_vars))
        st.writes = _compiler._infer_writes(dict(act), dict(rec.output or {}))
        st.payload = {"kind": "model",
                      "prompt": str(act.get("prompt") or act.get("template") or ""),
                      "reads": list(st.reads), "writes": list(st.writes)}
    else:                                              # tool（认不出的 kind 也当工具处理）
        st.reads = _compiler._infer_reads(dict(act), prev_vars)
        st.writes = _compiler._infer_writes(dict(act), dict(rec.output or {}))
        st.payload = {"kind": "tool", "name": str(act.get("name") or ""),
                      "input": _compiler._templatize(dict(act.get("input") or {}),
                                                     prev_vars),
                      "reads": list(st.reads), "writes": list(st.writes)}
        if act.get("phase"):            # 阶段随记录来，进状态、进 KEY（两侧对称）
            st.payload["phase"] = str(act["phase"])
    plan.states[sid] = st
    plan.order.append(sid)
    plan.by_key[key].append(sid)
    return st


def _resolve_target(plan: Plan, p: str, rec: Any, key: tuple, prev_vars: dict,
                    ctx: _Ctx, tid: str) -> tuple[Optional[str], bool]:
    """L6–L9 的那个决定：这一步是**新的一步**，还是**回到已有的某一步**。

    返回 ``(目标状态, 这次是不是新建的)``；这个点编不下去时返回 ``(None, False)``。

    确定性档：规范化动作 KEY 撞上已有状态 ⇒ 重复，否则新步。给了模型就问模型
    （:data:`MODEL_TOUCHPOINTS` 的 (a)），回复不合模式记一次拒绝、并退回确定性档；同一个点
    连拒到上限就编不下去了。
    """
    # 只按后继对齐、不做全局查重的 KEY（文档骨架里的 model 状态折成 ("model",)：轨迹里
    # 任何位置的模型步都"像"它，全局查重会把第一个模型步对到骨架里错的位置上；它们只在
    # _step 的 L5（当前状态的后继）里对齐，这里一律当新步）。
    local_only = set(getattr(ctx, "local_only_keys", ()) or ())
    cands = [] if key in local_only else list(plan.by_key.get(key, []))
    exact = _aligned_by_state(plan, p, rec, key)
    if exact is not None:
        cands = [exact] + [c for c in cands if c != exact]  # 自报的状态优先
    choice = cands[0] if cands else "new"
    if ctx.model is not None:
        ans = _ask_new_or_repeat(ctx, p, rec, cands)
        if ans is None:
            if ctx.strike(p, "新步/重复的回复不合模式"):
                return None, False
        elif ans == "new" or ans in cands:
            choice = ans
            ctx.clear(p)
        elif ctx.strike(p, f"模型指向不存在的状态 {ans!r}"):
            return None, False
    if choice == "new":
        sid = plan.new_sid()
        _create_state(plan, sid, rec, key, prev_vars, ctx, tid)
        return sid, True
    return choice, False


def _observe(plan: Plan, sid: str, rec: Any, tid: str) -> None:
    """把这一步的观测累进台账：判断动作的标签与样例、变量类型、轨迹归属。"""
    st = plan.states[sid]
    if tid and tid not in plan.trace_states[sid]:
        plan.trace_states[sid].append(tid)
    out = dict(rec.output or {})
    for w in st.writes:
        t = _type_of(out.get(w))
        if t and w not in plan.var_types:
            plan.var_types[w] = t
    if st.kind != "judge" or not st.writes:
        return
    label = out.get(st.writes[0])
    if isinstance(label, str) and label and label not in st.labels_seen:
        st.labels_seen.append(label)
        snap = {k: dict(rec.vars).get(k) for k in st.reads if k != "label"}
        st.examples.append((label, snap))


def transcribe(t_plus: Sequence[Trace], *, ctx: Optional[_Ctx] = None,
               skill_id: str = "compiled", plan: Optional[Plan] = None,
               on_trace: Any = None, ordered: bool = True) -> Plan:
    """**第一趟**：按 L2 的顺序（步数少的优先）逐条轨迹走完，出一份 :class:`Plan`。

    这一趟一个字节的机器都不改——它做的全是决策：对齐、新步/重复、分岔在哪、支持度多少、
    判断动作见过哪些标签、每个状态在单条轨迹里最多被进入几次。

    ``plan`` 给了就在它上面**继续**转写（文档骨架做种子、轨迹在线更新走的就是这条）；
    ``on_trace(plan, trace, before, after)`` 每转写完一条轨迹调一次，``before/after`` 是转写
    前后的 ``(状态数, 边数, 各边支持度之和)``——在线更新的逐条账靠它；``ordered=False`` 按给定
    顺序喂（在线到达的顺序），不按步数重排。
    """
    ctx = ctx or _Ctx()
    plan = plan if plan is not None else Plan(skill_id=skill_id)
    seq = _order_traces(t_plus) if ordered else list(enumerate(t_plus))
    for _, trace in seq:
        task = trace.task if isinstance(trace.task, dict) else {}
        inp = task.get("input", {}) if isinstance(task.get("input", {}), dict) else {}
        plan.input_keys |= set(inp)
        for k, v in inp.items():
            t = _type_of(v)
            if t and k not in plan.var_types:
                plan.var_types[k] = t
        plan.max_records = max(plan.max_records, len(trace.records))
        before = _plan_size(plan)
        _transcribe_one(plan, trace, ctx)
        if on_trace is not None:
            on_trace(plan, trace, before, _plan_size(plan))
    return plan


def _plan_size(plan: Plan) -> tuple[int, int, int]:
    return (len(plan.states), len(plan.edges), sum(e.support for e in plan.edges.values()))


def _transcribe_one(plan: Plan, trace: Trace, ctx: _Ctx) -> None:
    """L3–L11：沿一条接受轨迹逐动作走。"""
    recs = list(trace.records)
    if not recs:
        return
    tid = _tid(trace)
    task = trace.task if isinstance(trace.task, dict) else {}
    task_input = task.get("input", {}) if isinstance(task.get("input", {}), dict) else {}
    visits: dict = defaultdict(int)
    prev_sid, prev_rec = "", None
    pprev_sid = ""
    counts: dict = defaultdict(int)                 # 本条轨迹内按基础 KEY 的计数（供 key_of）

    try:
        for i, rec in enumerate(recs):
            key = (ctx.key_of(rec, prev_rec, counts) if ctx.key_of is not None
                   else canon_action(rec, strict=True))
            prev_vars = dict(prev_rec.vars) if prev_rec is not None else dict(task_input)
            if not prev_sid:                               # L3：起点
                if not plan.initial:
                    sid = plan.new_sid()
                    _create_state(plan, sid, rec, key, prev_vars, ctx, tid)
                    plan.initial = sid
                elif plan.states[plan.initial].key == key:
                    sid = plan.initial
                else:
                    plan.notes.append({
                        "trace": tid, "step": i, "kind": "start_mismatch",
                        "why": f"这条轨迹的起手动作 {_brief(rec.action)} 与已编译的起点 "
                               f"{plan.initial} 不是同一步；一台机器只有一个起点，这套"
                               "形状表达不了开局就分岔，整条轨迹不转写"})
                    return
            else:
                sid = _step(plan, prev_sid, rec, key, prev_vars, ctx, tid, i)
                if sid is None:
                    return
                e = plan.edges[(prev_sid, sid)]
                e.support += 1
                plan.obs[prev_sid].append((sid, dict(prev_rec.vars)))
                # 第 4 元是轨迹**对象**而不是 task_id：同一道题有多次运行，task_id 分不开它们，
                # 打标器拿错运行就会给错标签（实测过：标签与后继对不上）。
                plan.obs_ctx[prev_sid].append((sid, pprev_sid, visits[prev_sid], trace, i - 1))
            visits[sid] += 1
            _observe(plan, sid, rec, tid)
            pprev_sid, prev_sid, prev_rec = prev_sid, sid, rec
    finally:
        # 半路走不下去的轨迹，**它已经走过的那一段照样算数**：访问次数是循环上限 K 的
        # 唯一依据，丢掉半条会让 K 偏小、把本来绕得完的环提前踢进 FALLBACK。
        for s, c in visits.items():
            plan.visits[s] = max(plan.visits[s], c)


def _aligned_by_state(plan: Plan, p: str, rec: Any, key: tuple) -> Optional[str]:
    """轨迹记录自报的状态 id，如果它在台账里且身份对得上，就是**精确**的对齐目标。

    折叠过的身份会撞车：种子里的 model 状态一律折成 ``("model",)``（文档的私有 prompt 与
    轨迹里模型说的话不可比，见 agents/seed.py），于是一个状态的回边与主干边如果都指向 model
    状态，L5 按出边顺序取第一个就可能取到回边。实测：一条 17 步、走了三圈修复环的轨迹，整条
    链塌回起点自环，分岔学不出条件、被堵，23 状态的骨架编成 2 状态。

    ``Record.state`` 是**执行侧如实记下的**「当时在哪个状态」，比折叠键精确。只在两个条件都
    满足时用它：那个 id 在台账里存在，且它的身份与这一步算出来的身份一致——不然就是另一台
    机器的状态名，一律忽略。外部 agent 日志没有这个字段（``to_trace`` 一律记 FALLBACK），
    这条路自然不生效。
    """
    sid = str(getattr(rec, "state", "") or "")
    if not sid or sid == FALLBACK:
        return None
    st = plan.states.get(sid)
    if st is None or st.key != key:
        return None
    return sid


def _step(plan: Plan, p: str, rec: Any, key: tuple, prev_vars: dict,
          ctx: _Ctx, tid: str, step: int) -> Optional[str]:
    """从 ``p`` 走一步：L5 对齐 / L6-L8 新步或成环 / L9 分岔。编不下去返回 ``None``。"""
    if p in plan.blocked:
        return None
    exact = _aligned_by_state(plan, p, rec, key)
    if exact is not None and exact in plan.out.get(p, []):
        return exact                                       # 记录自报的状态，且确实是 p 的后继
    for dst in plan.out.get(p, []):                        # L5：同 KEY 的出边，直接前移
        if plan.states[dst].key == key:
            return dst
    target, created = _resolve_target(plan, p, rec, key, prev_vars, ctx, tid)
    if target is None:
        plan.blocked.add(p)
        plan.notes.append({"trace": tid, "step": step, "kind": "blocked",
                           "state": p, "why": "同一个点连拒到上限，这一段退回解释执行"})
        return None
    back = (not created) and plan.reaches(target, p)        # L8：成环
    plan.add_edge(p, target, back=back, creating=created, origin=(tid, step))
    return target


# --------------------------------------------------------------------------- #
# 第一趟半：定条件（L9 的 fit.learn_cond / L10 的判断动作）
# --------------------------------------------------------------------------- #
def _plan_variables(plan: Plan) -> list[Variable]:
    """台账推出来的变量表。三个来源：

    * **被某个状态写过的**；
    * **任务输入的字段**（带 ``init_from``）；
    * **计数变量**——边上的 ``inc`` 递增它、条件读它，但没有任何状态"写"它。文档骨架里的
      ``repair_count`` 就是这样：只有初值，靠回边加一。漏了它，凡是读它的条件都会被守门程序
      判「用到未声明的变量」，整批提案连坐（实测：不摘文档边之后，三批因此被拒）。

    轨迹的 ``vars`` 里还有别的东西（产出这批轨迹的那台机器自己的计数变量之类），一律**不
    收**：机器只认自己写得出来的变量，收了它们条件就会读一个永远没人写的名字。
    """
    names: set[str] = set(plan.input_keys)
    for st in plan.states.values():
        names |= set(st.writes)
    counters = {e.counter for e in plan.edges.values() if getattr(e, "counter", None)}
    names |= counters
    out: list[Variable] = []
    for n in sorted(names):
        if n in counters and n not in plan.input_keys:
            out.append(Variable(name=n, type="integer", init=0))
            continue
        out.append(Variable(name=n, type=plan.var_types.get(n, "string"),
                            init_from=f"task.input.{n}" if n in plan.input_keys else None))
    return out


def _tighten(conds: dict, snaps: Mapping, variables: Sequence[Variable]) -> dict:
    """把学出的条件**收紧到观测支持的最小形式**：``x != 'b'`` ⇒ ``x == 'a'``（若本支的
    ``x`` 恒为 ``'a'``）。

    两个理由，缺一不可：

    * **安全**。分岔外的格局（判断动作弃权那一格）本该落到兜底边、也就是 FALLBACK；留着
      ``!=`` 形式会把「没见过的取值」也吞进某一支。宁可少编。
    * **确定**。:func:`skill2fsm.fit.candidate_atoms` 枚举字符串字面量时走的是一个
      ``set``，同一批快照在不同进程里可能先给 ``==`` 也可能先给 ``!=``，学出的条件因此**跨
      进程不稳定**。收紧到等值形式让两种搜索顺序收敛到同一个答案。
    """
    out = dict(conds)
    for tgt in list(out):
        expr = out[tgt]
        try:
            used = sorted(_cond.vars_of(expr))
        except _cond.CondError:
            continue
        if len(used) != 1:
            continue
        v = used[0]
        vals = {s.get(v) for s in snaps.get(tgt, []) if v in s}
        if len(vals) != 1:
            continue
        val = next(iter(vals))
        if isinstance(val, bool) or not isinstance(val, (str, int, float)):
            continue
        eq = f"{v} == {val!r}"
        if eq == expr:
            continue
        mine = list(snaps.get(tgt, []))
        others = [s for o in snaps if o != tgt for s in snaps[o]]
        if not _fit._separates(eq, mine, others):
            continue
        trial = {**out, tgt: eq}
        if _fit.mutually_exclusive(list(trial.values()), snaps, variables):
            out = trial
    return out


def _always_guard(snaps: Sequence[dict], variables: Sequence[Variable]) -> Optional[str]:
    """给一条**回边**学一个「在这个状态的全部观测快照上恒真」的谓词。学不出返回 ``None``。

    为什么回边非要有条件：见模块文档「回边为什么一定带条件」。
    """
    if not snaps:
        return None
    atoms = _fit.candidate_atoms(list(snaps), variables)
    expr = _fit.separating(atoms, list(snaps), (), max_atoms=1)
    if expr is None:
        return None
    return _tighten({"__loop__": expr}, {"__loop__": list(snaps)}, variables)["__loop__"]


def _judge_branch(plan: Plan, p: str, snaps: Mapping, ctx: _Ctx) -> Optional[dict]:
    """L10：分岔学不出确定条件时，起草一个判断动作，用它的裁决变量当条件。

    **只在这一步本来就是判断步时才做**（见模块文档「判断动作只在原地改写」）：凭空插一个
    判断状态会让机器比轨迹多走一步，那条轨迹立刻复述不出来，得不偿失。
    """
    st = plan.states.get(p)
    if st is None or st.kind != "judge" or ctx.model is None:
        return None
    reads = list(st.reads) or sorted({k for s in snaps.values() for x in s for k in x})
    judge = draft_judge({"state": p, "reads": reads, "writes": list(st.writes),
                         "targets": {t: list(v) for t, v in snaps.items()},
                         "clause": st.clause}, model=ctx.model)
    if judge is None:
        ctx.strike(p, "判断动作起草失败或回复不合模式")
        return None
    usable = [l for l in judge.labels if l != judge.abstain]
    targets = list(snaps)
    if len(usable) < len(targets):
        ctx.strike(p, f"起草的标签集只有 {len(usable)} 个非弃权标签，盖不住 "
                      f"{len(targets)} 个分支")
        return None
    assign = {t: usable[i] for i, t in enumerate(targets)}
    samples = [(dict(s), assign[t]) for t in targets for s in snaps[t]]
    rate = _calibrate(ctx, judge, samples)
    if rate is None:
        ctx.strike(p, "判断动作标定不了误差率")
        return None
    if rate > ctx.thresholds.judge_err_max:
        ctx.strike(p, f"标定误差率 {rate} > 上限 {ctx.thresholds.judge_err_max}")
        return None
    st.prompt = judge.prompt
    st.labels_seen = [l for l in judge.labels if l != judge.abstain]
    st.reads = list(judge.reads)
    st.error_rate = rate
    st.support = len(samples)
    st.drafted = True
    st.examples = [(assign[t], {k: s.get(k) for k in judge.reads})
                   for t in targets for s in snaps[t][:1]]
    w = st.writes[0]
    return {t: f"{w} == {assign[t]!r}" for t in targets}


def _drop_thin(plan: Plan, min_support: int) -> None:
    """L14 的前半：支持度不足的边整条拿掉，只被它们够得着的状态一并剪掉——退回解释执行。"""
    for kk in list(plan.edge_order):
        e = plan.edges[kk]
        if e.support >= min_support:
            continue
        if e.origin and e.origin[0] == "document":
            # 文档骨架的边、轨迹没走到：**留着**。它不是「把偶然当规律」，是「文档要求、轨迹
            # 尚未观测」——两者的处置必须不同。摘掉等于让机器只会做已经见过的那几条路，文档
            # 写了而这批轨迹恰好没走到的分支（错误处理、边界情形）会在产物里整段消失。
            #
            # 代价是支持度门槛：这条边 support=0，拿它去过 min_support 必然不过，而验收不过会
            # 让 _settle 从**源状态**退回 FALLBACK，把已经对上轨迹的主干一起带走（实测：整台
            # 机器塌成 begin→FALLBACK）。所以门槛那侧要一起豁免——见 checker._support_rows：
            # 支持度要求的是「被编译下来的那条路有证据」，文档边本来就不是从轨迹编下来的。
            plan.notes.append({"kind": "doc_unobserved", "edge": f"{e.src}->{e.dst}",
                               "cond": e.cond, "support": e.support,
                               "why": "文档骨架里的这条边在本批轨迹里没被走过：**照样留在产物里**"
                                      "（文档的主张不因没被观测到就消失），但记在这一栏里——"
                                      "再补轨迹时它是最值钱的目标之一"})
            continue
        plan.thin.append({"edge": f"{e.src}->{e.dst}", "support": e.support,
                          "min_support": min_support,
                          "why": "只凭这么少的轨迹就编译下来，是在把偶然当规律；"
                                 "这条分支退回解释执行"})
        _unlink(plan, kk)
    _prune(plan)
    # 转写时的「回边」是按当时图上能不能绕回来判的。文档骨架带着一批 support=0 的回路
    # （s4→s2 修复环），轨迹从 s2 直走 s4 时看起来是在成环，于是被记成回边；上面把那些文档边
    # 摘掉之后它根本绕不回来了，再当回边去学「什么时候该再绕一圈」只会学不出、把 s2 整个堵死
    # （实测：文档骨架 + 4 条直线轨迹编出一台只有起点的机器）。只降不升：真回路照旧。
    for e in plan.edges.values():
        if e.back and not plan.reaches(e.dst, e.src):
            e.back = False
            plan.notes.append({"kind": "back_edge_demoted", "edge": f"{e.src}->{e.dst}",
                               "why": "转写时靠文档边才成环；那些边没被轨迹走过、已摘掉，"
                                      "这条边现在是前向边"})
    # 同一个来源的第二个坑：落账只给「有创建边或是起点」的状态发提案。骨架里 s4 的创建边是
    # 文档边 s3→s4；轨迹从 s2 直接走到 s4 时 s4 已存在，那条边记成 creating=False。文档边摘掉
    # 后 s4 就没有创建边，整个状态连同它后面的主干都落不下去（实测：state:s5 被拒「引用了不
    # 存在的状态 s4」）。补法：还留着入边的状态若没有创建边，把它最早的那条前向入边升为创建边。
    creators = {kk[1] for kk in plan.edge_order if plan.edges[kk].creating}
    for sid in plan.order:
        if sid == plan.initial or sid in creators:
            continue
        incoming = [kk for kk in plan.edge_order if kk[1] == sid]
        pick = next((kk for kk in incoming if not plan.edges[kk].back), None) or \
            (incoming[0] if incoming else None)
        if pick is None:
            continue
        plan.edges[pick].creating = True
        creators.add(sid)
        plan.notes.append({"kind": "creator_reanchored", "state": sid,
                           "edge": f"{pick[0]}->{pick[1]}",
                           "why": "原来的创建边是文档边、没被轨迹走过、已摘掉；改由这条有轨迹"
                                  "支持的入边创建它"})


def _unlink(plan: Plan, kk: tuple) -> None:
    e = plan.edges.pop(kk, None)
    if e is None:
        return
    plan.edge_order = [x for x in plan.edge_order if x != kk]
    plan.out[e.src] = [d for d in plan.out.get(e.src, []) if d != e.dst]
    plan.obs[e.src] = [(d, s) for d, s in plan.obs.get(e.src, []) if d != e.dst]


def _prune(plan: Plan) -> None:
    """从起点顺着**还留着的**边重算可达，走不到的状态与边一起剪掉。"""
    if not plan.initial:
        return
    seen, stack = {plan.initial}, [plan.initial]
    while stack:
        cur = stack.pop()
        if cur in plan.blocked:
            continue
        for nxt in plan.out.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    dead = [s for s in plan.order if s not in seen]
    for s in dead:
        plan.pruned.append(s)
        plan.order.remove(s)
        st = plan.states.pop(s)
        plan.by_key[st.key] = [x for x in plan.by_key.get(st.key, []) if x != s]
        plan.out.pop(s, None)
        plan.obs.pop(s, None)
    if dead:
        for kk in list(plan.edge_order):
            if kk[0] in dead or kk[1] in dead:
                _unlink(plan, kk)


def _avail_map(plan: Plan) -> dict:
    """每个状态**进入时**一定已被写过的变量（按路径取交集）。

    与 :func:`skill2fsm.checks._write_before_read` 同一条判据：并集问「有没有一条路让它
    存在」，交集问「是不是每条路都让它存在」。只有后者能保证运行时不撞未定义变量。
    """
    seed = set(plan.input_keys)
    universe = set(seed)
    for st in plan.states.values():
        universe |= set(st.writes)
    avail = {sid: set(universe) for sid in plan.states}
    if plan.initial in avail:
        avail[plan.initial] = set(seed)
    incoming: dict = defaultdict(list)
    for src, dst in plan.edge_order:
        incoming[dst].append(src)
    for _ in range(len(plan.states) + 2):
        changed = False
        for sid in plan.order:
            if sid == plan.initial:
                continue
            srcs = incoming.get(sid, [])
            new = set() if not srcs else set(universe)
            for src in srcs:
                st = plan.states.get(src)
                new &= avail.get(src, set()) | (set(st.writes) if st else set())
            if new != avail.get(sid):
                avail[sid] = new
                changed = True
        if not changed:
            break
    return avail


def _template_vars(payload: Any) -> list[str]:
    """台账里某个 tool 状态的入参模板引用了哪些变量。判据与 checks.template_vars 同源。"""
    if not isinstance(payload, Mapping):
        return []
    from .checks import template_vars
    return template_vars(type("A", (), {"input": payload.get("input") or {}})())


def _prune_reads(plan: Plan, avail: dict) -> None:
    """把每个状态的 ``reads`` 收到「走到它时一定已写过」的范围内。

    ``compiler._infer_reads`` 是**值相等**的启发式：input 里某个值恰好等于某个变量当前的值，
    就算读了它。实测它会读出根本无关的变量（提交步读 ``verify_exit``），而那个变量在别的
    路径上没人写——守门程序据此判 ``E_READ_BEFORE_WRITE``，整处编译决定被丢掉。与其让一条
    臆测出来的读把一整段图拖下水，不如在这里如实收窄：**读不到的就不算读**，并把删掉的记
    进台账（覆盖报告能说出「这一步本来疑似还读了什么，因为某条路径上没人写而作罢」）。
    """
    for sid in plan.order:
        st = plan.states[sid]
        if not st.reads:
            continue
        ok = avail.get(sid, set())
        # 入参模板引用的变量**不能删**：那不是 _infer_reads 猜出来的，是这一步真要拿来渲染
        # 入参的。删掉读声明、留下模板，机器真跑时那一格渲染成空（实测：bash 拿到空命令）。
        # 它没人写就该是一处 E_READ_BEFORE_WRITE，由 params.bind_unbound_templates 去补生产者。
        pinned = set(_template_vars(st.payload))
        dropped = [r for r in st.reads if r not in ok and r not in pinned]
        if not dropped:
            continue
        st.reads = [r for r in st.reads if r in ok]
        if isinstance(st.payload, dict) and "reads" in st.payload:
            st.payload["reads"] = list(st.reads)
        plan.notes.append({
            "kind": "read_pruned", "state": sid, "dropped": dropped,
            "why": "这些变量在通往本状态的某条路径上没人写过（按路径取交集），"
                   "留着会让整处编译决定过不了先写后读检查——如实删掉"})


def _fix_doc_branch(plan: Plan, p: str, targets: list, fixed: dict, snaps: dict,
                    variables: list) -> bool:
    """文档骨架那个状态的出边：文档定的条件照用，轨迹长出的新去处各配一条与它们互斥的条件。

    骨架的分岔长这样：``if repair_count >= 3 → s_done`` 加一条**默认边** ``→ s2``。原来的判据
    要求「每条边都有非空条件」才复用文档条件——默认边的条件是空串，判据当场不成立，于是去
    重学；文档说了而轨迹没走过的那些去处没有快照，学不出，整个分岔被堵。以前这个毛病被裁剪
    掩盖着：文档边都摘了，状态只剩一个去处，走的是主干那条路。不摘之后 23 个分岔全堵。

    正确的读法是把一个分岔看成「若干带条件的边 + 至多一条兜底边」：带条件的彼此互斥即可，
    兜底边接住其余。所以：

    * 文档的带条件边 —— 原样保留；
    * 文档的默认边 —— 留在兜底位；
    * 轨迹长出的新去处 —— 从它自己的快照上学一条谓词，**再合取上文档各条件的否定**，
      保证与文档的边两两互斥（:func:`skill2fsm.checks._determinism` 要的是两两互斥，不看顺序）。

    学不出就返回 ``False``，让调用方走原来那条重学 / 引入判断 / 堵的路。
    """
    doc_guarded = [t for t in targets if fixed.get(t)]
    doc_default = [t for t in targets if t in fixed and not fixed.get(t)]
    fresh = [t for t in targets if t not in fixed]
    if len(doc_default) > 1:
        return False                                   # 骨架自己就不合法，交给下游报错
    if not fresh:
        for t in doc_guarded:
            plan.edges[(p, t)].cond = fixed[t]
        for t in doc_default:
            plan.edges[(p, t)].cond = ""
        return True
    if len(fresh) > 1 or not doc_default:
        # 多个新去处彼此也要互斥，或没有兜底位可留：这两种情形交给正常的学条件那条路
        return False
    t = fresh[0]
    g = _always_guard(snaps.get(t) or [], variables)
    if g is None:
        return False
    negs = " and ".join(f"not ({fixed[d]})" for d in doc_guarded)
    plan.edges[(p, t)].cond = f"({g}) and {negs}" if negs else g
    for d in doc_guarded:
        plan.edges[(p, d)].cond = fixed[d]
    plan.edges[(p, doc_default[0])].cond = ""
    plan.notes.append({"kind": "doc_branch_extended", "state": p, "new_target": t,
                       "cond": plan.edges[(p, t)].cond,
                       "why": "文档的分岔上长出一个新去处：文档那几条边原样保留，新边配一条"
                              "只在自己观测上为真、且与文档各条件互斥的谓词，文档的默认边"
                              "留在兜底位"})
    return True


def fit_guards(plan: Plan, ctx: _Ctx) -> None:
    """给每条边定条件（L9/L10），并给每条回边算计数上限（L8 的 K）。

    * 一个状态只有**一个**去处 ⇒ 主干边，无条件（回边除外，见 :func:`_always_guard`）。
    * 有**两个及以上**去处 ⇒ :func:`skill2fsm.fit.learn_cond` 学一组两两互斥的谓词；学不出
      就试 L10 的判断动作；再不行整个分岔退回解释执行（该状态进 ``plan.blocked``，出边一条
      都不落，只剩守门程序给的那条兜底边）。
    * 所有分岔边**一律带条件**，兜底位留给 FALLBACK：没见过的格局落回解释执行，而不是被
      某一支吞掉。
    """
    avail = _avail_map(plan)
    _prune_reads(plan, avail)
    variables = _plan_variables(plan)
    declared = {v.name for v in variables}
    thr = ctx.thresholds
    for p in list(plan.order):
        if p in plan.blocked:
            # 转写期就连拒到上限的点：它的出边一条都不落，兜底位留给 FALLBACK。
            for kk in [k for k in plan.edge_order if k[0] == p]:
                _unlink(plan, kk)
            continue
        targets = list(plan.out.get(p, []))
        if not targets:
            continue
        snaps: dict = {t: [] for t in targets}
        only = plan.guard_vars.get(p)                  # 引入的判断状态：只看判断变量
        # 出边条件在**本状态执行之后**求值，所以可用集是「进入时可用 ∪ 本状态写的」。
        # 不过滤的话会学出一条读「某条路径上没人写的变量」的谓词，守门程序当场判先写后读。
        usable = (avail.get(p, set()) | set(plan.states[p].writes)) & declared
        for dst, snap in plan.obs.get(p, []):
            if dst in snaps:
                snaps[dst].append({k: v for k, v in snap.items()
                                   if k in usable and (only is None or k == only)})
        # ---- 文档骨架的分岔：条件是文档定的，不重学 ---- #
        fixed = getattr(plan, "fixed_conds", {}).get(p) or {}
        if fixed and _fix_doc_branch(plan, p, targets, fixed, snaps, variables):
            continue
        if len(targets) == 1:
            e = plan.edges[(p, targets[0])]
            if not e.back:
                e.cond = ""                                # 主干
                continue
            g = _always_guard(snaps[targets[0]], variables)
            if g is None:
                _block(plan, p, "回边学不出「什么时候该再绕一圈」的谓词，这个环整个放弃")
                continue
            e.cond = g
            continue
        learned = _fit.learn_cond(snaps, variables, min_support=thr.min_support,
                                  holdout_ratio=thr.holdout_ratio,
                                  acc_thr=thr.acc_thr)
        if learned is None:
            learned = _judge_branch(plan, p, snaps, ctx)
        if learned is None:
            _block(plan, p, "分岔学不出两两互斥的条件，也起不了判断动作："
                            "整个分岔退回解释执行")
            continue
        learned = _tighten(learned, snaps, variables)
        for t in targets:
            plan.edges[(p, t)].cond = learned.get(t, "")
        if any(not plan.edges[(p, t)].cond for t in targets):
            _block(plan, p, "有分支没拿到条件（学出的条件不全）")
    _prune(plan)
    _install_bounds(plan, ctx)


def _block(plan: Plan, p: str, why: str) -> None:
    plan.blocked.add(p)
    plan.blocked_targets[p] = list(plan.out.get(p, []))
    plan.notes.append({"kind": "blocked", "state": p, "why": why,
                       "targets": list(plan.out.get(p, []))})
    for kk in list(plan.edge_order):
        if kk[0] == p:
            _unlink(plan, kk)


def _install_bounds(plan: Plan, ctx: _Ctx) -> None:
    """给每条回边配计数变量与上限 K，并**把 K 是谁定的记下来**。

    ``doc_bound`` 恒为 ``None``：本模块不从文档里抠圈数上限（抠出来的数会被当成文档要求，
    而它其实是我读文档读出来的）。所以 :func:`skill2fsm.fit.loop_bound_detail` 一律给出
    ``source="compiler"``——覆盖报告因此**说得出**「这条上限是编译器为了停机补的，不是文档
    要求」。
    """
    for kk in plan.edge_order:
        e = plan.edges[kk]
        if not e.back:
            continue
        lb = _fit.loop_bound_detail(plan.visits.get(e.dst, 1),
                                    margin=ctx.thresholds.loop_margin)
        # 文档骨架给这条回边起过名（``inc: repair_count``）就沿用它，别改名：骨架的条件里
        # 写的是那个名字（``repair_count >= 3``），改成 ``sN_count`` 会让那些条件读一个没人
        # 声明的变量，整批提案被守门程序驳回（实测：不摘文档边之后三批因此被拒）。
        # 只有轨迹自己长出来的环才由 harness 取名。
        e.counter = e.counter or f"{e.dst}_count"
        e.bound = e.bound or lb.k
        plan.loop_bounds.append({
            "back_edge": f"{e.src}->{e.dst}", "var": e.counter, "k": lb.k,
            "source": lb.source, "observed_max": lb.observed_max, "margin": lb.margin,
            "why": "技能文档没有写任何圈数上限；K = ceil(margin × 单条轨迹里该状态被进入的"
                   "最大次数)，是编译器为了「环一定停得下来」补上的"})


# --------------------------------------------------------------------------- #
# 第二趟：落账（把决策翻成受票提议）
# --------------------------------------------------------------------------- #
#: 五个批的名字与顺序：多智能体的合并阶段序（skill2fsm/batch.py）与它一致。
BATCH_NAMES: tuple[str, ...] = ("entry", "trunk", "judges", "transitions", "loops")


def build_batches(plan: Plan, ctx: _Ctx, *, prohibitions: Sequence = (),
                  audit_tools: Sequence[str] = ()) -> dict[str, list[Proposal]]:
    """把台账翻成五个**批**：``entry``（open_machine）/ ``trunk``（建状态，含带创建边的
    判断与终止）/ ``judges``（原地改写的判断，单智能体路径为空）/ ``transitions``（前向
    分岔边）/ ``loops``（回边）。:func:`build_proposals` 是它们按序的拼接。

    回边排最后不是偷懒：``close_loop`` 会调 :func:`skill2fsm.fit.install_counter`，它要把
    环的**目标状态当时已有的每条条件出边**都 ``and`` 上 ``count < K``，再插一条
    ``count >= K → FALLBACK``。收完环再往那个状态上接新的条件边，新边身上没有 ``count < K``，
    与上限出口在计满那一格同时成立——结构检查当场判条件重叠。所以环必须最后收。
    """
    out: dict[str, list[Proposal]] = {name: [] for name in BATCH_NAMES}
    variables = _plan_variables(plan)
    open_payload = {"variables": [v.model_dump() for v in variables],
                    "prohibitions": [dict(p) if isinstance(p, Mapping) else p
                                     for p in (prohibitions or [])],
                    "max_steps": max(24, plan.max_records * 2)}
    if audit_tools:
        open_payload["audit_tools"] = list(audit_tools)
    out["entry"].append(Proposal(
        "open_machine", open_payload,
        rationale=f"L1：打开机器 {plan.skill_id}，声明 {len(variables)} 个变量"
                  f"（任务输入 {sorted(plan.input_keys)} 走 init_from）"))

    creator = {kk[1]: plan.edges[kk] for kk in plan.edge_order if plan.edges[kk].creating}
    for sid in plan.order:
        st = plan.states[sid]
        e = creator.get(sid)
        attach: dict = {}
        if e is not None:
            attach = {"from_state": e.src, "from_cond": e.cond,
                      "from_support": e.support}
        elif sid != plan.initial:
            continue                       # 没有创建边也不是起点：这个状态落不下去
        out["trunk"].append(_state_proposal(plan, st, attach, sid == plan.initial))

    for kk in plan.edge_order:
        e = plan.edges[kk]
        if e.creating or e.back:
            continue
        out["transitions"].append(Proposal(
            "add_transition",
            {"from_state": e.src, "to": e.dst, "cond": e.cond, "support": e.support},
            rationale=f"L9：{e.src} 上的分岔，条件 {e.cond or '（兜底）'} 由 "
                      f"fit.learn_cond 在 {e.support} 次观测的变量快照上学出",
            trace_id=e.origin[0], step=e.origin[1]))

    for kk in plan.edge_order:
        e = plan.edges[kk]
        if not e.back:
            continue
        out["loops"].append(Proposal(
            "close_loop",
            {"from_state": e.src, "to": e.dst, "cond": e.cond,
             "counter": e.counter, "bound": e.bound, "support": e.support},
            rationale=f"L8：{e.src}→{e.dst} 是回边，收成有上限的环；计数变量 {e.counter}"
                      f" 上限 {e.bound}（**上限是编译器加的，文档没写**）",
            trace_id=e.origin[0], step=e.origin[1]))
    return out


def build_proposals(plan: Plan, ctx: _Ctx, *, prohibitions: Sequence = ()) -> list[Proposal]:
    """把台账翻成一串受票提议。顺序 = 决策顺序，唯一的例外是**回边排在最后**——
    即 :func:`build_batches` 五个批按 :data:`BATCH_NAMES` 的拼接。"""
    batches = build_batches(plan, ctx, prohibitions=prohibitions)
    return [p for name in BATCH_NAMES for p in batches[name]]


def _state_proposal(plan: Plan, st: _PState, attach: dict, is_initial: bool) -> Proposal:
    src = attach.get("from_state") or "（起点）"
    if st.kind == "end":
        payload = {"state_id": st.sid, "terminal": st.terminal, "clause": st.clause}
        seed_t = (getattr(plan, "seed_terminals", {}) or {}).get(st.terminal)
        if seed_t is not None:                     # 文档骨架声明的终点类别（verified/unverified）
            payload["kind"] = seed_t.kind
            payload["output"] = list(seed_t.output)
        if st.origin_kind:
            payload["origin"] = st.origin_kind
        if st.locator:
            payload["locator"] = st.locator
        payload.update(attach)
        return Proposal("set_terminal", payload,
                        rationale=f"L11：{src} 之后这条轨迹结束，结束方式 {st.terminal}",
                        trace_id=st.origin[0], step=st.origin[1])
    if st.kind == "judge":
        labels = sorted(st.labels_seen)
        if ABSTAIN not in labels:
            labels.append(ABSTAIN)
        examples = []
        for lbl, snap in st.examples:
            ex = {k: v for k, v in snap.items() if k != "label"}
            ex["label"] = lbl
            examples.append(ex)
        payload = {"state_id": st.sid, "prompt": st.prompt,
                   "reads": list(st.reads), "writes": list(st.writes),
                   "labels": labels, "abstain": ABSTAIN, "examples": examples,
                   "error_rate": st.error_rate, "support": st.support,
                   "clause": st.clause}
        if st.introduced:
            payload.update({"introduced": True, "gold_from": st.gold_from})
        if st.origin_kind:
            payload["origin"] = st.origin_kind
        if st.locator:
            payload["locator"] = st.locator
        payload.update(attach)
        how = ("从文档引入（introduce_judge）" if st.introduced else
               "模型起草（L10）" if st.drafted else "轨迹里本来就有的判断步，照原样转写")
        return Proposal("add_judge", payload,
                        rationale=f"L7：{src} 之后是一次判断，{how}；标签 {labels} 取自"
                                  f"{'起草' if st.drafted else '轨迹观测'}",
                        trace_id=st.origin[0], step=st.origin[1])
    payload = {"state_id": st.sid, "action": dict(st.payload), "clause": st.clause}
    if st.origin_kind:
        payload["origin"] = st.origin_kind
    if st.locator:
        payload["locator"] = st.locator
    if is_initial and not attach:
        payload["initial"] = True
    payload.update(attach)
    return Proposal("add_state", payload,
                    rationale=f"L7：{src} 之后是新的一步 {_brief(st.payload)}，"
                              f"读 {st.reads} → 写 {st.writes}，条款 "
                              f"{st.clause or '（未归属：条款归属要模型）'}",
                    trace_id=st.origin[0], step=st.origin[1])


def apply_plan(ck: _checker.Checker, proposals: Sequence[Proposal], *,
               max_strikes: int = MAX_STRIKES, progress: bool = False) -> PlanResult:
    """**第二趟**：把提议一条一条交给守门程序裁决。

    规矩两条，都直接对应算法 1 的要求：

    * **被拒的提议不牵连它之前被接受的提议**——这是守门程序本来就有的性质，这里只是不去
      破坏它：一条被拒，接着提下一条。
    * **同一个点上连拒 ``max_strikes`` 次 ⇒ ``demote_to_fallback`` 那个点，然后往下走。**
      退回解释执行会连带删掉只能经过那个点到达的状态，所以之后凡是碰到已死状态的提议一律
      跳过（不再去撞一次必然的拒绝）。

    ``ck`` 必须还没 ``open_machine``（提议列表的第一条就是它），或者已经打开——两种都行，
    重复打开会被守门程序自己拒掉并记一张回执。
    """
    res = PlanResult()
    strikes: dict = defaultdict(int)
    dead: set[str] = set()
    alive: set[str] = set()
    for prop in proposals:
        point = prop.point
        touched = {str(prop.payload.get(k) or "")
                   for k in ("from_state", "to", "state_id")} - {""}
        if touched & dead:
            res.skipped.append(prop)
            continue
        if alive and prop.op != "open_machine":
            need = {str(prop.payload.get(k) or "") for k in ("from_state", "to")} - {""}
            if need - alive:
                res.skipped.append(prop)
                continue
        receipt = ck.apply(prop.to_checker())
        res.receipts.append(receipt)
        if progress:
            print(f"[compile_agent] {prop.op} "
                  f"{'✓' if receipt.accepted else '✗'} {receipt.reason[:110]}",
                  file=sys.stderr)
        if receipt.accepted:
            res.accepted += 1
            strikes[point] = 0
            if ck.opened:
                alive = set(ck.machine.states)
            continue
        res.rejected += 1
        if not ck.opened:
            continue
        strikes[point] += 1
        if point and strikes[point] >= max_strikes:
            dr = ck.demote_to_fallback(
                point, note=f"同一个点连拒 {strikes[point]} 次（最后一次："
                            f"{receipt.reason[:80]}）——宁可少编，不编错")
            res.receipts.append(dr)
            strikes[point] = 0
            if dr.accepted:
                res.demoted.append(point)
                dead.add(point)
                alive = set(ck.machine.states)
                dead |= {s for s in touched if s not in alive}
    return res


# --------------------------------------------------------------------------- #
# L12/L13：拒绝集与验收
# --------------------------------------------------------------------------- #
def _machine_walk(machine: Machine, trace: Trace) -> tuple[list, Optional[int]]:
    """沿轨迹推机器，返回 ``([(记录下标, 状态)], 偏离处的记录下标或 None)``。

    驱动逻辑与 :func:`skill2fsm.replay.replay` 一致（用轨迹记录的 output 按 writes 白名单
    推变量、回边自己 inc、``pick_edge`` 选边）；这里要的是**位置**，好知道该退谁。
    """
    r = _replay.walk(machine, trace)
    # 虚拟开局步与零宽判断记的下标可能是 -1 / 前一条：定位「该退谁」只看真实消费了记录的状态
    seq = [(i, sid) for i, sid in r.seq if i >= 0]
    return seq, (None if r.ok else r.diverged_at)


def _offenders(machine: Machine, rep: Any, t_plus: Sequence[Trace],
               t_minus: Sequence[Trace]) -> list[str]:
    """验收挂了，该把哪些状态退回解释执行。保序去重，不含 FALLBACK 与不存在的状态。"""
    out: list[str] = []

    def push(sid: str) -> None:
        if sid and sid != machine.fallback and sid in machine.states and sid not in out:
            out.append(sid)

    for sid, _c in rep.weak_edges:
        push(sid)
    for sid, _r in rep.hot_judges:
        push(sid)
    for i in rep.unreproduced:
        seq, _d = _machine_walk(machine, t_plus[i])
        push(seq[-1][1] if seq else machine.initial)
    for i in rep.unexcluded:
        neg = t_minus[i]
        cut = neg.error_step if neg.error_step is not None else 10 ** 9
        seq, _d = _machine_walk(machine, neg)
        # 退掉哪个状态？退 ``p`` 会让机器在 **p 之后那一步** 就进 FALLBACK，所以要挑
        # 「下一条记录的 step 仍 ≤ 出错位置」的最后一个状态——退它，回退段才盖得住出错处，
        # 这条反例才从「已编译区段里漏掉的」变成「还没编译到、尚不可排除的」。
        at = [sid for idx, sid in seq
              if idx + 1 < len(neg.records) and neg.records[idx + 1].step <= cut]
        push(at[-1] if at else (seq[-1][1] if seq else machine.initial))
    for f in rep.findings:
        if f.severity == "error":
            push(f.state_id)
    return out


def _settle(ck: _checker.Checker, t_plus: Sequence[Trace], t_minus: Sequence[Trace],
            thr: Thresholds, ctx: _Ctx, *, commit: bool = True) -> tuple[Any, list, bool]:
    """L12+L13：先自己跑一遍验收，挂了就把肇事状态退回解释执行，通过了才 ``commit``。

    为什么不直接 ``commit`` 让它挂：``Checker.commit`` 是**全有全无**的——不过就整批回滚到
    上一次提交（这里是 ``open_machine``，也就是一台空机器）。真挂一次，前面所有被接受的改写
    连同它们的回执一起白做，而且**回滚之后没有任何受票接口能把它们再放回去**。所以这里先用
    :func:`skill2fsm.verify.verify_machine`（只读、不改机器）看一眼，按报告退掉肇事的那几个
    点，能过了再提交。修不动就**不提交**：机器停在「已通过全部结构检查、但没盖验收章」的状态，
    比回滚成一台空机器诚实得多，账记在 ``stats["commit"]`` 里。
    """
    receipts: list = []
    rep = _verify.verify_machine(ck.machine, t_plus, t_minus, thresholds=thr)
    seen_bad: list = []
    for _ in range(_MAX_REPAIR):
        if rep.ok:
            break
        bad = _offenders(ck.machine, rep, t_plus, t_minus)
        if not bad or bad == seen_bad:
            break                    # 退不动了（比如起手动作就与起点对不上）：别空转
        seen_bad = list(bad)
        moved = False
        for sid in bad:
            r = ck.demote_to_fallback(
                sid, note="L12/L13：验收指着这里说不过——" + _verify.summary(rep)[:120])
            receipts.append(r)
            moved = moved or r.accepted
            ctx.say(f"退回解释执行 {sid}：{'✓' if r.accepted else '✗'}")
        if not moved:
            break
        rep = _verify.verify_machine(ck.machine, t_plus, t_minus, thresholds=thr)
    if not rep.ok:
        return rep, receipts, False
    if not commit:
        # 多智能体回合：验收过了也不在这里提交——只有 orchestrator 对 incumbent 提交一次
        return rep, receipts, True
    receipts.append(ck.commit(t_plus=t_plus, t_minus=t_minus))
    return rep, receipts, receipts[-1].accepted


# --------------------------------------------------------------------------- #
# 技能与条款表
# --------------------------------------------------------------------------- #
def _resolve_skill(skill: Any) -> tuple[str, str, list, Optional[Machine]]:
    """从 ``skill`` 里取出 ``(skill_id, 文档正文, 禁止项, 参考机器或 None)``。

    认四种形状：:class:`~skill2fsm.skill_loader.AgentSkill`、带 ``skill_doc()`` /
    ``reference_machine()`` 的模块或对象（``examples.table_clean`` 就是）、``dict``、以及
    一段正文字符串或一个技能目录路径。

    **参考机器只作对差用**（``CompileResult.diff_vs_reference``），一个字节都不进编译产物；
    禁止项则是人工标出的、编不进图的东西，只有 ``skill`` 自己带 ``prohibitions`` 时才收
    ——从参考机器上顺手拿是把靶子当箭。
    """
    if skill is None:
        return "compiled", "", [], None
    if isinstance(skill, Mapping):
        doc = str(skill.get("doc") or skill.get("body") or skill.get("text") or "")
        sid = str(skill.get("skill_id") or skill.get("slug") or skill.get("name")
                  or "compiled")
        ref = skill.get("reference_machine")
        return sid, doc, list(skill.get("prohibitions") or []), \
            (ref if isinstance(ref, Machine) else None)
    if isinstance(skill, (str, Path)):
        text = str(skill)
        p = Path(text) if (isinstance(skill, Path)
                           or ("\n" not in text and len(text) < 4096)) else None
        if p is not None and p.is_dir():
            try:
                from .skill_loader import load_agent_skill
                loaded = load_agent_skill(p)
                return loaded.slug, loaded.body, [], None
            except Exception:                               # noqa: BLE001
                pass
        return "compiled", text, [], None

    doc = ""
    for attr in ("skill_doc", "full_text"):
        fn = getattr(skill, attr, None)
        if callable(fn):
            try:
                doc = str(fn())
                break
            except Exception:                               # noqa: BLE001
                doc = ""
    if not doc:
        for attr in ("body", "doc", "text"):
            v = getattr(skill, attr, None)
            if isinstance(v, str) and v:
                doc = v
                break
    ref: Optional[Machine] = None
    rm = getattr(skill, "reference_machine", None) or \
        getattr(skill, "build_reference_machine", None)
    if isinstance(rm, Machine):
        ref = rm
    elif callable(rm):
        try:
            got = rm()
            ref = got if isinstance(got, Machine) else None
        except Exception:                                   # noqa: BLE001
            ref = None
    sid = ""
    for attr in ("skill_id", "slug", "name"):
        v = getattr(skill, attr, None)
        if isinstance(v, str) and v:
            sid = v
            break
    if not sid and ref is not None:
        sid = ref.skill_id
    proh = list(getattr(skill, "prohibitions", ()) or [])
    return sid or "compiled", doc, proh, ref


def markdown_clauses(doc: str) -> list[tuple[str, str, str]]:
    """对**任何** SKILL.md 都能用的条款切法：``(id, 原文, 定位)``。

    编号按结构：H1 是标题不编号；H2 依次 S1、S2…；H3 在其下 S2.1、S2.2…；标题下的每个
    列表项再下一级 S2.1.1…。这与手写参考机台账里按位置编的号一致（见
    ``out/reference/machine_xlsx.notes.md``）。没有标题的文档整份是 S1。
    定位是 ``doc:<行号>``（相对传入文本），不冒充文件行号。
    """
    lines = (doc or "").splitlines()
    out: list[tuple[str, str, str]] = []
    h2 = h3 = item = 0
    cur_head: Optional[str] = None
    buf: list[str] = []
    buf_line = 0

    def flush() -> None:
        # 同一标题下、列表之后又出现的正文并回该标题那一行：id 必须唯一，条款不能有两个 S2。
        nonlocal buf
        if cur_head and buf:
            text = "\n".join(buf).strip()
            for k, (cid, old, loc) in enumerate(out):
                if cid == cur_head:
                    out[k] = (cid, old + "\n" + text, loc)
                    break
            else:
                out.append((cur_head, text, f"doc:{buf_line}"))
        buf = []

    for i, ln in enumerate(lines, 1):
        st = ln.strip()
        if st.startswith("## ") and not st.startswith("### "):
            flush(); h2 += 1; h3 = 0; item = 0
            cur_head, buf, buf_line = f"S{h2}", [st], i
        elif st.startswith("### "):
            flush(); h3 += 1; item = 0
            cur_head, buf, buf_line = f"S{max(h2, 1)}.{h3}", [st], i
        elif st[:2] in ("- ", "* ") or (st[:3].rstrip(".").isdigit() and ". " in st[:4]):
            flush(); item += 1
            base = f"S{max(h2, 1)}" + (f".{h3}" if h3 else "")
            out.append((f"{base}.{item}", st, f"doc:{i}"))
            buf, buf_line = [], 0
            if cur_head is None:
                cur_head = base
        elif st.startswith("# "):
            flush(); cur_head = None
        elif st:
            if cur_head is None:
                cur_head, buf_line = "S1", i
            buf.append(st)
    flush()
    return out


def clause_rows(doc: str, clauses: Sequence = ()) -> list[ClauseRow]:
    """L1 的「列出条款表」。``clauses`` 给了就用（:class:`Clause` 记录或 ``(id, text)``
    对都认）；没给先试 :func:`skill2fsm.compiler.partition`（``## Sx`` 式编号标题），切不出
    再用 :func:`markdown_clauses` 按结构切——任何 SKILL.md 都有条款表。"""
    rows: list[ClauseRow] = []
    src: Sequence = clauses if clauses else _compiler.partition(doc or "")
    if not src and not clauses:
        src = markdown_clauses(doc or "")
    for item in src:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            cid, text = str(item[0]), str(item[1])
            title = next((l.strip("# ").strip() for l in text.splitlines() if l.strip()),
                         cid)
            rows.append(ClauseRow(id=cid, title=title, text=text,
                                  locator=str(item[2]) if len(item) > 2 else ""))
            continue
        cid = str(getattr(item, "id", "") or "")
        if not cid:
            continue
        loc = ""
        fn = getattr(item, "locator", None)
        if callable(fn):
            try:
                loc = str(fn())
            except Exception:                               # noqa: BLE001
                loc = ""
        rows.append(ClauseRow(id=cid, title=str(getattr(item, "title", "") or cid),
                              text=str(getattr(item, "text", "") or ""), locator=loc))
    return rows


# --------------------------------------------------------------------------- #
# 覆盖报告（交付物之一）
# --------------------------------------------------------------------------- #
def _coverage(machine: Machine, plan: Plan, rows: Sequence[ClauseRow],
              t_plus: Sequence[Trace], t_minus: Sequence[Trace], ctx: _Ctx,
              rep: Any, committed: bool, demoted: Sequence[str]) -> dict:
    """逐条款的支持/单薄/无轨迹触达 + 结构出身 + 回退面 + 「再补哪些轨迹最值钱」。

    底座直接用 :func:`skill2fsm.report.cover_report`，所以
    :func:`skill2fsm.report.render` 拿这份字典就能渲染；本函数只往上加账。
    """
    thr = ctx.thresholds
    base = _report.cover_report(machine, [(r.id, r.text) for r in rows],
                                t_plus=t_plus, t_minus=t_minus)

    cite: dict = defaultdict(list)
    for sid, st in sorted(machine.states.items()):
        if st.clause:
            cite[st.clause].append(sid)
    in_support: dict = defaultdict(int)
    for src, t in machine.transitions_all():
        if t.to != machine.fallback:
            in_support[t.to] += t.support

    table: list[dict] = []
    supported, thin, untouched = [], [], []
    for r in rows:
        states = cite.get(r.id, [])
        traces = sorted({tid for s in states for tid in plan.trace_states.get(s, [])})
        # 支持度按**走过这些状态的轨迹条数**算，不按入边支持度：起点没有入边，拿入边去衡量
        # 它会把「每条轨迹都走了的第一步」判成单薄。入边支持度另存一栏，两个数各说各的。
        sup = len(traces)
        if not states:
            status = "untouched"
            untouched.append(r.id)
        elif sup < thr.min_support:
            status = "thin"
            thin.append(r.id)
        else:
            status = "supported"
            supported.append(r.id)
        table.append({"id": r.id, "title": r.title, "locator": r.locator,
                      "status": status, "states": states, "traces": traces,
                      "support": sup,
                      "edge_support": sum(in_support.get(s, 0) for s in states)})

    # K 的台账按「这条回边现在还在不在机器上」标一个 live：验收阶段退回解释执行会连带删掉
    # 一整段，那一段的 K 只是历史，不该再算成产物里的编译器引入结构。
    live_edges = {f"{sid}->{t.to}" for sid, t in machine.transitions_all()}
    bounds = [{**lb, "live": lb["back_edge"] in live_edges} for lb in plan.loop_bounds]

    introduced: list[dict] = []
    for lb in bounds:
        if not lb["live"]:
            continue
        introduced.append({"kind": "loop_bound", **lb})
        introduced.append({
            "kind": "counter_variable", "var": lb["var"],
            "why": "计数变量是编译器为了给环配上限而引入的，文档里没有这个变量"})
    for sid, t in machine.transitions_all():
        if t.cond and sid in plan.states and not t.inc:
            introduced.append({
                "kind": "branch_guard", "state": sid, "cond": t.cond, "to": t.to,
                "why": "分岔条件由 fit.learn_cond 在变量快照上学出，文档没有明写这个谓词"})
    fb_states = sorted({sid for sid, t in machine.transitions_all()
                        if t.to == machine.fallback and sid != machine.fallback})
    introduced.append({
        "kind": "fallback_surface", "states": fb_states,
        "why": "通往 FALLBACK 的兜底边是编译器留的逃生口：条件盖不到、判断弃权、"
               "环绕满了都从这里退回「模型读整份文档解释执行」"})

    from_doc = [{"kind": "state", "id": sid, "clause": st.clause,
                 "action": st.action.kind,
                 "locator": next((r.locator for r in rows if r.id == st.clause), "")}
                for sid, st in sorted(machine.states.items()) if st.clause]

    next_traces: list[dict] = []
    for cid in untouched:
        next_traces.append({"kind": "clause", "target": cid,
                            "why": "没有任何轨迹触达这条条款，它现在整条落在 FALLBACK 里"})
    for row in plan.thin:
        next_traces.append({"kind": "edge", "target": row["edge"],
                            "why": f"支持度 {row['support']} < {thr.min_support}，"
                                   "被裁掉了；再多几条走这条分支的轨迹就能编下来"})
    for note in plan.notes:
        if note.get("kind") == "blocked":
            next_traces.append({"kind": "branch", "target": note.get("state", ""),
                                "why": note.get("why", "")})
    starts = [n for n in plan.notes if n.get("kind") == "start_mismatch"]
    if starts:
        next_traces.append({
            "kind": "start", "target": machine.initial,
            "why": f"{len(starts)} 条接受轨迹的**起手动作**与已编译的起点不是同一步，整条"
                   "没有转写。一台机器只有一个起点，这套形状表达不了「开局就分岔」；要么"
                   "按起手动作把轨迹分组各编一台，要么补一个统一的开局步骤再采一轮轨迹"})
    for sid in fb_states:
        next_traces.append({"kind": "fallback", "target": sid,
                            "why": "这个状态还留着一条通往解释执行的兜底边：走过它之后的"
                                   "轨迹越多，越有机会把兜底那一支也编出来"})

    judge_states = [s for s in machine.states.values() if s.action.kind == "judge"]
    base.update({
        "skill_id": machine.skill_id,
        "clause_table": table,
        "supported": supported, "thin": thin, "untouched": untouched,
        "clause_attribution": "model" if ctx.model is not None else "none",
        "structures": {"from_document": from_doc, "compiler_introduced": introduced},
        "loop_bounds": bounds,
        "fallback_surface": {
            "states_with_fallback_edge": fb_states,
            "n_edges_to_fallback": sum(1 for _s, t in machine.transitions_all()
                                       if t.to == machine.fallback),
            "demoted": list(demoted),
            "blocked_branches": sorted(plan.blocked),
            "thin_edges_dropped": list(plan.thin),
            "pruned_states": list(plan.pruned),
        },
        "next_traces": next_traces,
        "notes": list(plan.notes),
        "model_dependence": {
            "model_used": ctx.model is not None,
            "touchpoints": list(MODEL_TOUCHPOINTS),
            "model_free": [
                "状态与主干（按规范化动作 KEY 对齐轨迹）",
                "重复与成环（KEY 撞上已有状态即重复）",
                "分岔条件（fit.learn_cond：支持度 / 留出正确率 / 互斥可证 三关）",
                "回边的计数变量与上限 K（fit.loop_bound_detail）",
                "支持度裁剪与结构检查、验收（checker / verify）",
                "终止态与结束方式（照轨迹的 end 记录）",
            ],
            "needs_model": [
                f"条款归属（本次：{'模型归属' if ctx.model is not None else '全部留空'}）",
                f"新步 vs 重复的语义判定（本次："
                f"{'问模型' if ctx.model is not None else '结构启发式，按动作 KEY'}）",
                "分岔学不出条件时起草判断动作（本次："
                f"{'可起草' if ctx.model is not None else '不起草，分岔整个退回 FALLBACK'}）",
                "判断动作的误差率标定（没有标定就没有 Σεᵢ 这个上界）",
                "从文档条款引入判断动作（多智能体：introduce_judge；model=None 走技能包的 judges 库）",
                "同一动作在两种前驱语境下是否同一步（多智能体：split_context；model=None 按能否分别学出条件）",
                "在线采集探针给快照打标签（多智能体：annotate_judge；不跑就没有判断记录）",
            ],
        },
        "verify": _verify.report_dict(rep),
        "committed": committed,
        "judge_states": {s.id: {"error_rate": s.action.error_rate,
                                "support": s.action.support} for s in judge_states},
    })
    return base


def _diff(machine: Machine, ref: Optional[Machine]) -> Optional[dict]:
    """编译产物与手写目标机器的逐项对差。没有参考机器就返回 ``None``。"""
    if ref is None:
        return None

    def keys(m: Machine) -> dict:
        return {sid: canon_action(st.action, strict=False)
                for sid, st in m.states.items() if sid != m.fallback}

    got, want = keys(machine), keys(ref)
    gset, wset = set(got.values()), set(want.values())
    return {
        "reference_skill_id": ref.skill_id,
        "n_states_reference": ref.n_states(), "n_states_compiled": machine.n_states(),
        "n_transitions_reference": len(ref.transitions_all()),
        "n_transitions_compiled": len(machine.transitions_all()),
        "actions_matched": sorted("|".join(k) for k in (gset & wset)),
        "actions_missing": sorted("|".join(k) for k in (wset - gset)),
        "actions_extra": sorted("|".join(k) for k in (gset - wset)),
        "judges_reference": sorted(s.id for s in ref.states.values()
                                   if s.action.kind == "judge"),
        "judges_compiled": sorted(s.id for s in machine.states.values()
                                  if s.action.kind == "judge"),
        "prohibitions_reference": [p.id for p in ref.prohibitions],
        "prohibitions_compiled": [p.id for p in machine.prohibitions],
        "note": "参考机器是**目标形状**，不是交付物；这张表只用来说明编译产物差在哪里。",
    }


# --------------------------------------------------------------------------- #
# 顶层
# --------------------------------------------------------------------------- #
def compile_skill(skill: Any, t_plus: Sequence[Trace], t_minus: Sequence[Trace] = (), *,
                  model: Any = None, thresholds: Optional[Thresholds] = None,
                  clauses: Sequence = (), progress: bool = False,
                  begin: bool = False) -> CompileResult:
    """算法 1：从真实执行轨迹顺序转写编译出一台机器。**唯一的写机器通道是守门程序。**

    ``skill`` 认技能对象/模块/``dict``/正文/目录路径（见 :func:`_resolve_skill`）；
    ``t_plus`` 是接受轨迹，``t_minus`` 是拒绝轨迹（L12 用它验「机器会不会在出错处或更早偏
    离」）；``clauses`` 不给就从正文切。``model=None`` 走确定性启发式，全程无网络。

    返回的 :class:`CompileResult` 里，``machine`` 是产物，``receipts`` 是它每一步怎么长出来
    的审计痕迹，``coverage`` 是覆盖报告（:func:`skill2fsm.report.render` 直接渲染），
    ``judges`` 列出机器里每个判断动作及其出身，``stats`` 是计数，``diff_vs_reference`` 在能
    拿到手写目标机器时给出逐项对差。
    """
    thr = thresholds or Thresholds()
    skill_id, doc, prohibitions, ref = _resolve_skill(skill)
    rows = clause_rows(doc, clauses)
    ctx = _Ctx(model=model, thresholds=thr, rows=rows, doc=doc, progress=progress)
    ctx.say(f"L1：技能 {skill_id}，条款表 {len(rows)} 条，"
            f"T+ {len(t_plus)} 条、T- {len(t_minus)} 条，"
            f"模型 {'有' if model is not None else '无（走确定性启发式）'}")
    if begin:
        # 开局工具：每条轨迹前垫一步 BEGIN_TOOL，机器因此只有一个起点，真实首步成为它之后
        # 的分岔——start_mismatch 归零。回放对此透明（replay.walk 会虚拟地补上同一步），
        # 所以 T+/T- 的排除与复述检查照常拿原轨迹跑。
        from .trace_adapter import with_begin
        t_plus = [with_begin(t) for t in t_plus]
        t_minus = [with_begin(t) for t in t_minus]

    # ---- 第一趟：转写（L2–L11 的决策） ---- #
    plan = transcribe(t_plus, ctx=ctx, skill_id=skill_id)
    _drop_thin(plan, thr.min_support)                       # L14 前半
    fit_guards(plan, ctx)                                   # L9/L10 定条件 + L8 定 K
    ctx.say(f"转写完成：{len(plan.order)} 个状态、{len(plan.edge_order)} 条边、"
            f"{len(plan.blocked)} 处分岔编不出来")

    # ---- 第二趟：落账（每条改写一张回执） ---- #
    ck = _checker.Checker(skill_id, doc=doc, thresholds=thr)
    proposals = build_proposals(plan, ctx, prohibitions=prohibitions)
    filed = apply_plan(ck, proposals, progress=progress)
    if not ck.opened:                                       # 连机器都没打开：交空手
        from .schema import empty_machine
        return CompileResult(machine=empty_machine(skill_id), receipts=filed.receipts,
                             coverage={}, judges=[],
                             stats={"committed": False,
                                    "commit": "open_machine 都没通过，什么都没编"},
                             diff_vs_reference=None)

    # ---- L12 + L13：拒绝集与验收 ---- #
    rep, settle_receipts, committed = _settle(ck, t_plus, t_minus, thr, ctx)
    machine = ck.machine
    receipts = filed.receipts + settle_receipts
    demoted = list(filed.demoted) + [r.detail.get("state_id", "")
                                     for r in settle_receipts
                                     if r.op == "demote_to_fallback" and r.accepted]

    judges = [{
        "state": sid, "prompt": st.action.prompt, "reads": list(st.action.reads),
        "writes": list(st.action.writes), "labels": list(st.action.labels),
        "abstain": st.action.abstain, "error_rate": st.action.error_rate,
        "support": st.action.support, "clause": st.clause,
        "source": ("drafted" if (sid in plan.states and plan.states[sid].drafted)
                   else "trace"),
    } for sid, st in sorted(machine.states.items()) if st.action.kind == "judge"]

    coverage = _coverage(machine, plan, rows, t_plus, t_minus, ctx, rep, committed,
                         demoted)
    stats = {
        "traces_in": len(t_plus), "traces_negative": len(t_minus),
        "clauses": len(rows),
        "states_added": sum(1 for r in receipts
                            if r.op in ("add_state", "add_judge", "set_terminal")
                            and r.accepted),
        # 只加边的那两个接口。建状态的接口**自带入边**（add_state 的 from_state），所以
        # 机器上的边比这个数多——机器的实际条数看 n_transitions。
        "transitions_added": sum(1 for r in receipts
                                 if r.op in ("add_transition", "close_loop")
                                 and r.accepted),
        "n_states": machine.n_states(),
        "n_transitions": len(machine.transitions_all()),
        "judges_calibrated": sum(1 for j in judges if j["support"] > 0),
        "judges_added": sum(1 for r in receipts if r.op == "add_judge" and r.accepted),
        "judges_drafted": sum(1 for j in judges if j["source"] == "drafted"),
        "fallback_demotions": len([d for d in demoted if d]),
        "proposals": len(proposals), "accepted": filed.accepted,
        "rejected": filed.rejected, "skipped": len(filed.skipped),
        "model_calls": ctx.model_calls,
        "model_rejects": len(ctx.rejects),
        "prompt_tokens": None, "completion_tokens": None,
        "blocked_branches": sorted(plan.blocked),
        "thin_edges_dropped": len(plan.thin),
        # 起手动作与起点对不上、整条没转写的接受轨迹。一台机器只有一个起点，这套形状表达
        # 不了「开局就分岔」——这类轨迹注定复述不出来，验收因此会挂，得如实报出来。
        "traces_not_transcribed": sum(1 for n in plan.notes
                                      if n.get("kind") == "start_mismatch"),
        "committed": committed,
        "commit": ("验收通过并已提交" if committed
                   else "验收没过、**没有提交**（提交是全有全无的，挂一次就把已接受的改写"
                        "整批回滚成空机器）：" + _verify.summary(rep)),
        "structural_findings": structural_findings(machine),
    }
    usage = _runtime._usage_of(model)
    if usage:
        stats["prompt_tokens"] = usage.get("prompt_tokens")
        stats["completion_tokens"] = usage.get("completion_tokens")
    ctx.say(f"L13：{stats['commit']}")
    return CompileResult(machine=machine, receipts=receipts, coverage=coverage,
                         judges=judges, stats=stats,
                         diff_vs_reference=_diff(machine, ref))
