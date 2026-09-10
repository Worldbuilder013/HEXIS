"""把一次真实的 agent 循环折成 :class:`~skill2fsm.schema.Trace`，并解析核验工具的输出。

这个模块坐在**采集**与**编译**之间，是三臂实验里唯一一处「原始运行 → 训练数据」的转换。
它做三件事，每一件错了都会让下游安静地失去意义：

1. **解析核验结果**（:func:`parse_verify`）。编译器学的分岔条件几乎全都读 ``verify_status``：
   核验过了就提交、没过就回去修。把 ``INCONCLUSIVE`` 读成 ``FAIL``、把崩溃读成「答案错」、
   把 ``counterexample`` 的反极性拉平成 ``ok = (rc == 0)``——机器都会照着学出一条错的边，
   而且**跑起来毫无异常**：它只是在错误的时候修、在该修的时候提交。
2. **维护变量快照**（:func:`to_trace`）。每条记录的 ``vars`` 是编译器拟合条件用的样本点。
   变量的更新规则就是这批训练数据的生成规则，所以逐条写在下面的「变量表」里。
3. **落盘与读回**（:func:`write_jsonl` / :func:`read_jsonl`）。

跨臂可比性是硬约束
------------------
三臂比较的前提是同一件事在三臂里叫同一个名字。所以：

* 工具名一律过 :func:`~skill2fsm.normalize.canon_tool_name`——``scripts/math_verify.py``、
  ``math-verify``、``MATH_VERIFY`` 折成同一个 ``math_verify``；
* 最终提交那一步在**每一条臂**里都叫 ``submit_answer``（:data:`SUBMIT_TOOL`）。P1 那条禁止项
  是 ``require_before: submit_answer 之前必须先跑过核验``，被守卫的动作名一旦在某条臂里叫别的，
  这条臂的违规率就恒为 0——不是因为它守规矩，而是因为检查根本没开火。

这两条由 :func:`check_canonical` 在 :func:`to_trace` 结束时逐条校验，不合就抛
:class:`TraceAdapterError`，绝不让一条命名不齐的轨迹悄悄进数据集。

变量表（``Record.vars`` 里每条记录都带全，初值见括号）
--------------------------------------------------
============== ================================================================
变量            更新规则
============== ================================================================
``repair_count`` （0）**一次核验开始执行时**，若上一次核验的 ``verify_status`` 既非空
                 也非 ``PASS``，则先 +1。即「在一次没过的核验之后又核验了一次」= 修了
                 一轮。第一次核验永远不 +1（它前面没有失败）。
``verify_status``（``""``）最近一次核验的状态，取自 :func:`parse_verify`：
                 ``PASS``/``FAIL``/``INCONCLUSIVE``/``ERROR``/``TIMEOUT``。``""`` = 还没核验过。
``verify_exit``  （-1）最近一次核验的进程退出码。-1 是哨兵（还没核验过），真实退出码 ≥ 0，
                 沙箱的超时/启动失败用负码（见 sandbox.TIMEOUT_RC/ERROR_RC）。
``verify_stdout``（``""``）最近一次核验的 stdout，截到 :data:`VERIFY_STDOUT_MAX` 字符。
                 截断是必要的：vars 逐条记录快照，不截断等于把同一段 stdout 抄十几遍。
                 状态行在第一行，截断不会伤到要读的那部分。
``candidate``    （``""``）当前摆在桌面上的候选答案。任何带答案参数的步骤都会更新它
                 （核验步、提交步都算）。**这一条比规格多走了一步**：只在提交时才填的话，
                 提交之前每条记录的 candidate 都是空串，编译器就拟合不出「已有候选答案 ⇒
                 该去核验」这条边——而这正是数学技能的主循环。
``answer``       （``""``）**只**由提交步写入的最终答案。它与 ``candidate`` 分开，是为了让
                 「提交了」这件事在变量上留下痕迹。
============== ================================================================

刻意不做的事
------------
**不把任务输入铺进 ``vars``。** ``runtime.run_task`` 会做 ``values = dict(task["input"])``，
在玩具技能上无害；但 MATH-500 的任务字典里**带着参考答案**。铺进去，参考答案就出现在每一条
记录的变量快照里，编译器完全可能拟合出一条读金标准的分岔条件——机器于是「学会」了看答案，
三臂比较立刻作废。任务输入照常写进轨迹头部（复现要用），但不进变量。

代码正文离线存放
----------------
模型现写的 python（``run_python`` 的 ``code`` 参数）不留在记录里：记录只留
``code_sha256`` 与 ``code_path``，正文写成 ``artifacts/<sha256>.py``。给了 ``artifacts_dir``
就当场落盘；没给就先寄存在 ``Record.meta`` 里（``meta`` 不参与规范化、不参与评判），由
:func:`write_jsonl` 落盘时一并搬出去——两条路径写出的 JSONL 逐字相同，且**评判永远在搬完
之后做**，所以从 JSONL 读回来重判一定得到同一个 verdict。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from . import judge as _judge
from . import phases as _phases
from .normalize import BEGIN_TOOL, canon_action, canon_tool_name, is_begin
from .sandbox import ERROR_RC, TIMEOUT_RC
from .schema import FALLBACK, Prohibition, Record, Trace

__all__ = [
    "branch_key", "branch_label", "next_action_label",
    "ensure_phases", "load_any_trace", "read_raw_jsonl",
    "ANSWER_ARG_KEYS", "ARTIFACTS_DIRNAME", "CODE_KEYS", "ERROR", "FAIL",
    "INCONCLUSIVE", "INVERTED_SUBCOMMANDS", "PASS", "REPLY_MAX", "RawRun",
    "RawStep", "SUBMIT_ALIASES", "SUBMIT_TOOL", "TIMEOUT", "TOOL_STDOUT_MAX",
    "TraceAdapterError", "VERIFY_STDOUT_MAX", "VERIFY_SUBCOMMANDS",
    "VERIFY_TOOLS", "VerifyOutcome", "check_canonical", "initial_vars",
    "parse_verify", "read_jsonl", "to_trace", "write_jsonl",
]


class TraceAdapterError(RuntimeError):
    """轨迹转换里**不能容忍**的错误：命名不齐、步号不连续。宁可炸也不要脏数据。"""


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
ERROR = "ERROR"
TIMEOUT = "TIMEOUT"

#: 五种状态的全集。前三种是工具**自己**报的，后两种是本模块对「没有可信裁决」的分类。
STATUSES = (PASS, FAIL, INCONCLUSIVE, ERROR, TIMEOUT)

#: ``math_verify.py`` 的全部子命令（逐字取自 vendored 脚本的 ``build_parser``）。
VERIFY_SUBCOMMANDS = frozenset({
    "equiv", "derivative", "antiderivative", "definite-integral", "substitute",
    "satisfies", "limit", "solve", "system", "counterexample",
})

#: 极性**相反**的子命令：它要找的就是反例，找到了才算「搜索成功」。见 :func:`parse_verify`。
INVERTED_SUBCOMMANDS = frozenset({"counterexample"})

#: 算「核验代码跑过了」的工具（规范名）。``run_python``/``run_script`` 与技能自带的
#: ``math_verify`` 同等对待——标定实测模型用 ``run_python`` 比用 ``math_verify.py`` 更频繁，
#: 只认后者会把真的核验过的运行误判成违规（docs/HARNESS_CALIBRATION.md 第 4 节第 2 条）。
VERIFY_TOOLS = frozenset({"math_verify", "run_python", "run_script"})

#: 最终提交那一步的**唯一**规范名。
SUBMIT_TOOL = "submit_answer"

#: 折叠后会被改写成 :data:`SUBMIT_TOOL` 的别名。模型嘴里的提交动作五花八门，跨臂对不齐
#: 就等于关掉 P1 检查，所以在这里一次性收口。``done``/``finish`` 这类**不**收——它们是
#: 「结束」不是「交答案」，混进来会把一次弃权当成一次提交。
SUBMIT_ALIASES = frozenset({
    "submit_answer", "submit", "submit_final_answer", "final_answer",
    "finalize_answer", "give_answer", "answer",
})

#: 步骤参数里可能装着答案的键，按优先级。
ANSWER_ARG_KEYS = ("answer", "final_answer", "candidate", "result", "value")

#: 参数里装着「代码正文」的键：这些值搬去 ``artifacts/``，记录只留哈希。
CODE_KEYS = ("code", "script", "source")

#: 离线正文的目录名（相对轨迹目录）。
ARTIFACTS_DIRNAME = "artifacts"

VERIFY_STDOUT_MAX = 400
TOOL_STDOUT_MAX = 2000
REPLY_MAX = 4000

#: 墙钟守卫在各平台留下的退出码：124（coreutils ``timeout``）、137/-9（SIGKILL）、
#: 143/-15（SIGTERM）、以及本仓库沙箱的 :data:`~skill2fsm.sandbox.TIMEOUT_RC`。
_TIMEOUT_RCS = frozenset({124, 137, 143, -9, -15, TIMEOUT_RC})

_STATUS_LINE_RE = re.compile(r"^\s*status\s*:\s*([A-Za-z_]+)\s*$")


# --------------------------------------------------------------------------- #
# 原始运行
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawStep:
    """agent 循环里的一步，**未加工**。

    ``kind`` 四选一：``model``（一次模型回复）/ ``tool``（一次工具调用）/ ``judge``
    （一次语义判断）/ ``end``（停机）。``text`` 是已经剥掉 ``<think>`` 的模型正文——
    端点把推理块一律内联在 ``content`` 里（docs/HARNESS_CALIBRATION.md 第 2 节），不剥
    就会把草稿当答案。``meta`` 装 token 数、耗时、模型 id 这类执行侧账；``meta["state"]``
    可以覆盖记录的状态名，``meta["timed_out"]`` 告诉解析器这次是被墙钟杀掉的。
    """

    kind: str
    name: str = ""
    args: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    text: str = ""
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RawRun:
    """一次完整运行：任务、步骤序列，加上「谁跑的」这组出身字段。"""

    task: dict
    steps: list[RawStep]
    arm: str = ""
    run: int = 0
    model: str = ""
    harness: str = ""


# --------------------------------------------------------------------------- #
# 核验输出解析
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VerifyOutcome:
    """一次核验的裁决。

    ``status`` 是**工具报的原始状态**（或本模块给出的 ``ERROR``/``TIMEOUT``），它是编译器
    拟合条件时读的那个值。``ok`` 是**带子命令语义的读法**，两者在 ``counterexample`` 上
    故意不一致，理由见 :func:`parse_verify`。``detail`` 是审计用的旁证，不进变量。
    """

    status: str
    ok: bool
    detail: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """状态就是 ``PASS``（不看子命令极性）。想问「答案对不对」用这个。"""
        return self.status == PASS


def parse_verify(stdout: str, returncode: Optional[int], *,
                 argv: Sequence[str] = (),
                 timed_out: Optional[bool] = None) -> VerifyOutcome:
    """把核验工具的一次输出读成 :class:`VerifyOutcome`。

    规则全部来自对 vendored ``scripts/math_verify.py`` 的实测（见
    ``tests/test_17_verify_parse.py`` 里从 ``run_validation_suite.py`` 抬过来的用例表）：

    * ``--json`` 模式：stdout 是一个 JSON 对象，键按字母序回来（脚本用 ``sort_keys=True``），
      读 ``["status"]``；
    * 普通模式：stdout 的**第一行**恒为 ``status: PASS|FAIL|INCONCLUSIVE``；
    * **退出码 0 ⟺ ``status: PASS``；``FAIL`` 与 ``INCONCLUSIVE`` 都退 1**。所以单看退出码
      永远分不开「答案错」和「工具没结论」，更分不开「答案错」和「工具崩了」——三者都是 1。
      因此本函数**先要一个 ``status``**，要不到就判 ``ERROR``；
    * 空 stdout + stderr 上的 traceback + rc 1 ⇒ ``ERROR``（喂 LaTeX 就是这个形状）；
    * rc 2（argparse 用法错，例如 ``--json`` 放到了子命令后面）⇒ ``ERROR``；
    * rc 124 / 被墙钟守卫杀掉 ⇒ ``TIMEOUT``（``equiv "9^9^9" "1"`` 永不返回，靠这条兜住）；
    * ``rc not in {0, 1}`` 一律 ``ERROR``：退出码越界说明这不是核验协议的输出，此时哪怕
      stdout 里真有一行 ``status:`` 也不该信。

    **``counterexample`` 的极性是反的。** 它的活儿是「找一个反例」：``status: PASS`` 表示
    「等式被符号地证明了，因此不存在反例」（搜索一无所获），``status: FAIL`` 表示**找到了
    见证点**（搜索成功，claim 是假的）。所以 ``ok`` 在这个子命令上等于 ``status == FAIL``。
    ``ok`` 读作「这次调用拿到了它要找的那个肯定答案」，**不是**「答案对」——想问后者请读
    ``status`` 或 :attr:`VerifyOutcome.passed`。:func:`to_trace` 维护的 ``verify_status``
    走的正是 ``status``，所以这处反转不会污染修复循环。

    找到状态行时**信状态行**，即使它与退出码不符（``detail["exit_agrees"]`` 如实记下这件事）：
    模型自己写的核验脚本完全可能 ``print("status: FAIL")`` 之后正常退出，把它判成 ERROR 等于
    把一个读得懂的裁决扔掉。真正不可信的情况（rc 越界、没有状态行）已经在前面拦掉了。
    """
    text = stdout or ""
    rc = returncode
    check = _subcommand_of(argv)
    inverted = check in INVERTED_SUBCOMMANDS
    detail: dict[str, Any] = {"rc": rc, "check": check, "inverted": inverted}

    label, mode, line_no, payload = _read_status(text)
    detail["mode"] = mode
    if payload is not None:
        detail["payload"] = payload
        if "witness" in payload:
            detail["witness"] = payload["witness"]
    if line_no is not None:
        detail["status_line"] = line_no
        detail["status_first_line"] = (line_no == 0)

    # ---- 先判「根本没拿到可信裁决」的几种 ---- #
    if timed_out or (isinstance(rc, int) and rc in _TIMEOUT_RCS):
        detail["reason"] = "timeout"
        return VerifyOutcome(TIMEOUT, False, detail)
    if not isinstance(rc, int):
        detail["reason"] = "no_returncode"
        return VerifyOutcome(ERROR, False, detail)
    if rc not in (0, 1):
        detail["reason"] = ("usage_error" if rc == 2 else
                            "spawn_error" if rc == ERROR_RC else "bad_exit")
        return VerifyOutcome(ERROR, False, detail)
    if label is None:
        detail["reason"] = "crash" if not text.strip() else "no_status"
        return VerifyOutcome(ERROR, False, detail)
    if label not in (PASS, FAIL, INCONCLUSIVE):
        detail["reason"] = "unknown_status"
        detail["raw_status"] = label
        return VerifyOutcome(ERROR, False, detail)

    # ---- 有裁决 ---- #
    detail["exit_agrees"] = ((rc == 0) == (label == PASS))
    if inverted:
        detail["witness_found"] = (label == FAIL)
        ok = (label == FAIL)
    else:
        ok = (label == PASS)
    return VerifyOutcome(label, ok, detail)


def _subcommand_of(argv: Sequence[str]) -> str:
    """从 argv 里认出 ``math_verify.py`` 的子命令，认不出返回空串。

    只认白名单里的词，因此不会把 ``--var``、脚本路径、或者某个恰好同名的参数值当成子命令。
    """
    for tok in argv or ():
        s = str(tok)
        if s in VERIFY_SUBCOMMANDS:
            return s
    return ""


def _read_status(text: str) -> tuple[Optional[str], str, Optional[int], Optional[dict]]:
    """从 stdout 里取状态：返回 ``(状态标签, 模式, 行号, JSON载荷)``。

    先试 JSON（``--json`` 模式，整份 stdout 是一个对象）；再逐行找**整行**形如
    ``status: XXX`` 的那一行。找不到返回 ``(None, ...)``——调用方据此判 ERROR。

    普通模式下状态恒在第一行，但这里仍然扫完全部行并记下行号：模型自己写的核验脚本可能
    先打印两句别的。要求「整行匹配」把 ``note: ... status: ...`` 这种正文里的字样挡在外面。
    """
    stripped = (text or "").strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            st = payload.get("status")
            label = str(st).strip().upper() if isinstance(st, str) else None
            return label, "json", None, payload
    for i, line in enumerate(stripped.splitlines()):
        m = _STATUS_LINE_RE.match(line)
        if m:
            return m.group(1).strip().upper(), "plain", i, None
    return None, ("json" if stripped.startswith("{") else "plain"), None, None


# --------------------------------------------------------------------------- #
# 变量
# --------------------------------------------------------------------------- #
def initial_vars() -> dict:
    """一次运行开始时的变量快照。初值的含义见模块文档的「变量表」。"""
    return {
        "repair_count": 0,
        "verify_status": "",
        "verify_exit": -1,
        "verify_stdout": "",
        "candidate": "",
        "answer": "",
    }


#: **模型侧**变量：臂一臂二只暴露工具调用，模型脑内的东西（它上一句说了什么、上一步调了
#: 什么工具、成没成）在轨迹里原本没有快照。从文档引入的判断动作要读的正是这类东西——没有
#: 快照，``fit.calibrate`` 就没有样本可标定。所以每条记录的 ``vars`` 多记这三个；它们不在
#: :func:`initial_vars` 里（那是被 MathTask 奇偶测试钉住的工具侧变量表），也不会被编译器
#: 当成机器变量——除非某个判断动作声明读它。
MODEL_VARS = ("last_reply", "last_tool", "last_tool_ok")


def model_vars() -> dict:
    return {"last_reply": "", "last_tool": "", "last_tool_ok": None}


# --------------------------------------------------------------------------- #
# 开局工具：编译期的轨迹视图
# --------------------------------------------------------------------------- #
def with_begin(trace: Trace) -> Trace:
    """在轨迹最前面垫一条 :data:`~skill2fsm.normalize.BEGIN_TOOL` 记录（step = 首条 -1，
    通常是 0）。幂等：已经垫过的原样返回。**不写回磁盘**——这是编译器与回放看的视图，
    采集下来的轨迹一个字节不动（test_17 钉着 ``records[0]`` 是第一条真实的工具步）。

    ``error_step`` 不用平移：真实记录的 step 号照旧。
    """
    recs = list(trace.records or ())
    if recs and is_begin(recs[0].action):
        return trace
    task = trace.task if isinstance(trace.task, dict) else {}
    inp = task.get("input", {}) if isinstance(task.get("input", {}), dict) else {}
    first = (recs[0].step - 1) if recs else 0
    begin = Record(step=first, state=FALLBACK, clause="",
                   action={"kind": "tool", "name": BEGIN_TOOL, "input": {}},
                   output={}, vars=dict(inp))
    return trace.model_copy(update={"records": [begin] + recs})


def with_input(trace: Trace, keys: Sequence[str]) -> Trace:
    """把任务字典顶层的 ``keys`` 镜像进 ``task["input"]``（编译视图，不写回磁盘）。

    旧轨迹（collect.py 采的那 80 条）头部只有 ``task_id/problem/answer_ref/…``，没有
    ``input``——于是 ``problem`` 不是可声明的变量，从文档引入的判断动作读不到题面。参考答案
    （``answer_ref``/``answer``）**永远不镜像**：``run_task`` 把 ``task["input"]`` 摊进变量表，
    金标进了变量就可能被学成一条读答案的条件（test_17 钉着这条）。幂等。
    """
    task = trace.task if isinstance(trace.task, dict) else {}
    inp = dict(task.get("input") or {}) if isinstance(task.get("input"), dict) else {}
    added = False
    for k in keys:
        if k in ("answer", "answer_ref", "solution", "golden"):
            continue
        if k not in inp and k in task:
            inp[k] = task[k]
            added = True
    if not added and "input" in task:
        return trace
    return trace.model_copy(update={"task": {**task, "input": inp}})


# --------------------------------------------------------------------------- #
# 程序打标器：给从文档引入的判断动作打金标
# --------------------------------------------------------------------------- #
#: ``gold_from`` 名 → ``(trace, i) -> label | None``。``i`` 是该判断在轨迹里**之前**的那条
#: 记录的下标（零宽判断不消费记录）。打标器是纯函数、像禁止项一样受审查；**模型永不产
#: 金标**。回放（replay.walk）与标定（fit.calibrate 的样本）用同一张表，两边构造上一致。
LABELERS: dict[str, Callable[[Trace, int], Optional[str]]] = {}


def register_labeler(name: str) -> Callable:
    """装饰器：把一个 ``(trace, i) -> label | None`` 登记成打标器。"""
    def deco(fn: Callable[[Trace, int], Optional[str]]):
        LABELERS[name] = fn
        return fn
    return deco


def _verify_records_after(trace: Trace, i: int) -> list:
    """``i`` 之后的核验记录。判断动作的程序打标要看它们的结果。"""
    return [r for r in trace.records[i + 1:]
            if (r.action or {}).get("kind") == "tool"
            and (r.action or {}).get("name") in VERIFY_TOOLS]


@register_labeler("next_action")
def next_action_label(trace: Trace, i: int) -> Optional[str]:
    """下一步的**动作标签**：``tool:bash/apply`` / ``model`` / ``judge`` / ``end:done``。

    这是「先分开、合并要验」那条路的基石打标器（见 :mod:`skill2fsm.merge`）。分岔处插的判断
    动作，它的金标就是「轨迹接下来那一步是什么」——由程序从轨迹里现读，模型永不产金标。
    有了它，分岔的出边条件一律是 ``v == '<标签>'``，两两互斥、由构造成立，等同性因此可证。

    ``i`` 按打标器的统一约定，是这个判断**之前**那条记录的下标；要标的是 ``i + 1``。
    """
    recs = list(trace.records or ())
    j = i + 1
    if j < 0 or j >= len(recs):
        return None
    return branch_label(recs[j])          # 传**记录**：裸 action 看不到 output，writes 会算空，指纹跟建树那侧对不上


def branch_key(rec_or_action: Any) -> tuple:
    """一步的身份键，**不含提示词与提问**。

    规范化的严格档会把模型步的提示词与判断步的提问算进键。那两段文字来自产出这条轨迹的那台
    机器，同一个步骤在不同机器版本下因此拿到不同的键。身份不该依赖产出轨迹的那一方，所以在
    这里去掉。剩下的分量是动作类别、工具名、阶段与写出的变量，它们都是这一步自身的属性。
    """
    key = canon_action(rec_or_action, strict=True)
    return tuple(x for x in key if not (x.startswith("prompt=") or x.startswith("question=")))


def branch_label(rec_or_action: Any) -> str:
    """一步的分支标签，由 :func:`branch_key` 直接翻成可读形式。

    标签要满足三条。它只依赖这一步本身，因为读标签的程序只看得到轨迹。它对不同的键给出不同
    的值，否则两个后继共用一个标签，其中一个成为死代码。它要能被模型读懂，因为运行时由模型
    在这些标签里选一个。
    """
    key = branch_key(rec_or_action)
    parts = {k.split("=", 1)[0]: k.split("=", 1)[1] for k in key[1:] if "=" in k}
    kind = key[0] if key else "?"
    if kind == "tool":
        name, ph = parts.get("name", ""), parts.get("phase", "")
        return f"{name}/{ph}" if ph else name
    if kind == "end":
        return f"结束:{parts.get('terminal', 'done')}"
    writes = parts.get("writes", "")
    return f"{kind}→{writes}" if writes else kind


@register_labeler("next_action_after")
def next_action_after(trace: Trace, i: int) -> Optional[str]:
    """``i`` 之后的第一步是哪一类动作（:func:`branch_label`）——「文档说这时该判一下」的
    判断最通用的程序金标：金标是轨迹自己的未来，不看参考答案。``i`` 之后没有记录 ⇒ ``None``。
    """
    recs = trace.records
    if i + 1 >= len(recs) or i < -1:
        return None
    return branch_label(recs[i + 1])


@register_labeler("fail_attr_from_trace")
def fail_attr_from_trace(trace: Trace, i: int) -> Optional[str]:
    """核验 FAIL 之后的失败归因（参考机结构④的两个标签），从**之后发生的事**倒推：

    * 下一次核验时 ``candidate`` 没变、结果却 PASS ⇒ 答案本来就对，是 **核验写错了**；
    * 下一次核验时 ``candidate`` 变了 ⇒ 模型改了答案，是 **解答有错**；
    * 没有下一次核验，或看不出 ⇒ ``None``（弃权）。

    只在紧邻的前一条记录是 ``verify_status != PASS`` 的核验步时才有意义；否则 ``None``。
    """
    recs = trace.records
    if i < 0 or i >= len(recs):
        return None
    here = recs[i]
    if (here.action or {}).get("name") not in VERIFY_TOOLS:
        return None
    if (here.vars or {}).get("verify_status") in ("", PASS):
        return None
    cand_before = (here.vars or {}).get("candidate")
    later = _verify_records_after(trace, i)
    if not later:
        return None
    nxt = later[0]
    cand_after = (nxt.vars or {}).get("candidate")
    if cand_after != cand_before:
        return "解答有错"
    if (nxt.vars or {}).get("verify_status") == PASS:
        return "核验写错了"
    return None


def _clip(text: str, limit: int) -> str:
    """截断并**留下截断的痕迹**：省略号后面写清原长，免得日后把截断当成工具真的没输出。"""
    s = text or ""
    if len(s) <= limit:
        return s
    return s[:limit] + f"…[截断，共 {len(s)} 字符]"


def _first_answer(args: dict) -> Optional[str]:
    """从一步的参数里取答案文本，取不到返回 None。"""
    for key in ANSWER_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, (str, int, float)) and str(v).strip():
            return str(v)
    return None


def _canon_step_name(raw_name: str) -> str:
    """工具名的规范形：先过 :func:`canon_tool_name`，再把提交类别名收口到 ``submit_answer``。"""
    name = canon_tool_name(raw_name)
    return SUBMIT_TOOL if name in SUBMIT_ALIASES else name


# --------------------------------------------------------------------------- #
# 代码正文离线存放
# --------------------------------------------------------------------------- #
def _extract_code(inp: dict) -> tuple[dict, dict]:
    """把参数里的代码正文换成哈希，返回 ``(新参数, {sha: 正文})``。"""
    out = dict(inp)
    bodies: dict[str, str] = {}
    for key in CODE_KEYS:
        v = out.get(key)
        if not isinstance(v, str) or not v.strip():
            continue
        sha = hashlib.sha256(v.encode("utf-8")).hexdigest()
        out.pop(key)
        out[f"{key}_sha256"] = sha
        out[f"{key}_path"] = f"{ARTIFACTS_DIRNAME}/{sha}.py"
        out[f"{key}_bytes"] = len(v.encode("utf-8"))
        bodies[sha] = v
    return out, bodies


def _spill(bodies: dict, artifacts_dir: Any) -> None:
    """把正文写成 ``<artifacts_dir>/<sha>.py``。同名即同内容（哈希即文件名），已存在就不重写。"""
    if not bodies:
        return
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    for sha, body in bodies.items():
        p = root / f"{sha}.py"
        if not p.exists():
            p.write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# 单步转换
# --------------------------------------------------------------------------- #
def _record_for(step: RawStep, idx: int, values: dict, *,
                artifacts_dir: Any, phase_rules: str = "",
                outputs: tuple = ()) -> tuple[Record, Optional[VerifyOutcome]]:
    """把一个 :class:`RawStep` 折成一条 :class:`~skill2fsm.schema.Record`，顺带更新 ``values``。

    ``output`` 的键**就是这一步写入的变量名**：``normalize.action_writes`` 从 output 的键反推
    writes（``ok``/``error`` 除外），编译器再据此给状态标 writes。所以核验步的 output 用的是
    ``verify_status``/``verify_exit``/``verify_stdout`` 这组全名，而不是 ``status``——否则状态
    声明写了 ``status``、条件却读 ``verify_status``，两边对不上。
    """
    kind = (step.kind or "").strip().lower()
    meta = dict(step.meta or {})
    state = str(meta.pop("state", "") or FALLBACK)
    outcome: Optional[VerifyOutcome] = None

    if kind == "tool":
        name = _canon_step_name(step.name)
        raw_args = dict(step.args or {})
        # 阶段要在**抽走正文之前**判：_extract_code 之后记录里只剩 code_sha256，
        # 拿哈希判不出这一步在干什么（见 skill2fsm.phases 的模块文档）。
        phase = (_phases.classify(name, raw_args, phase_rules, outputs=outputs)
                 if phase_rules else "")
        inp, bodies = _extract_code(raw_args)
        if bodies:
            if artifacts_dir is not None:
                _spill(bodies, artifacts_dir)
            else:
                meta["code_bodies"] = dict(bodies)   # 由 write_jsonl 落盘时搬走
        action = {"kind": "tool", "name": name, "input": inp}
        if phase:
            action["phase"] = phase

        if name == SUBMIT_TOOL:
            output = _submit_output(step, values)
        elif name in VERIFY_TOOLS:
            output, outcome = _verify_output(step, values, meta)
        else:
            output = {"ok": (step.returncode in (None, 0)),
                      "stdout": _clip(step.stdout, TOOL_STDOUT_MAX)}
            # 桌面上的候选答案：核验之外的工具也可能带着它（例如 solve 的 --expected）
            cand = _first_answer(inp)
            if cand is not None:
                values["candidate"] = cand
        if step.returncode is not None:
            meta.setdefault("returncode", step.returncode)
        if (step.stderr or "").strip():
            meta.setdefault("stderr", _clip(step.stderr, TOOL_STDOUT_MAX))
        if "last_tool" in values:                # 只在 to_trace 开了模型侧快照时记
            values["last_tool"] = name
            values["last_tool_ok"] = bool(output.get("ok", step.returncode in (None, 0)))
    elif kind == "model":
        # 模型正文进 output 而不是 meta：``absent``/``regex`` 这类文本禁止项只扫
        # action + output（见 judge._record_text），藏进 meta 就等于永远查不到模型说了什么。
        action = {"kind": "model"}
        output = {"reply": _clip(step.text or step.stdout, REPLY_MAX)}
        if "last_reply" in values:
            values["last_reply"] = output["reply"]
    elif kind == "judge":
        args = dict(step.args or {})
        writes = [str(w) for w in (args.get("writes") or [])]
        label = args.get("label", step.text or "")
        action = {"kind": "judge",
                  "prompt": str(args.get("prompt", "")),
                  "reads": [str(r) for r in (args.get("reads") or [])]}
        output = {w: label for w in writes}
        values.update(output)
    elif kind == "end":
        terminal = str((step.args or {}).get("terminal") or step.name or "done")
        action = {"kind": "end", "terminal": terminal}
        output = {}
    elif kind == "user":
        # 用户输入：问了什么在 action.prompt，用户答了什么在 output.answer
        args = dict(step.args or {})
        action = {"kind": "user", "prompt": str(args.get("prompt") or step.text or "")}
        output = {"answer": _clip(str(args.get("answer") or step.stdout or ""), REPLY_MAX)}
    else:
        raise TraceAdapterError(
            f"无法识别的事件：第 {idx} 步 kind={step.kind!r}（认 model / tool / judge / user / end）。"
            f"这份日志需要自己的适配器，不能丢掉事件后当作已编译")

    return Record(step=idx, state=state, clause="", action=action,
                  output=output, vars=dict(values), meta=meta), outcome


def _verify_output(step: RawStep, values: dict, meta: dict
                   ) -> tuple[dict, VerifyOutcome]:
    """核验步：解析裁决、按变量表更新 ``repair_count`` 与三个 ``verify_*``。"""
    args = dict(step.args or {})
    argv = args.get("argv") or (step.meta or {}).get("argv") or ()
    if isinstance(argv, str):
        argv = argv.split()
    outcome = parse_verify(step.stdout, step.returncode, argv=argv,
                           timed_out=(step.meta or {}).get("timed_out"))
    prev = values.get("verify_status") or ""
    if prev and prev != PASS:
        # 上一次核验没过，这次又核验了一次 ⇒ 中间修了一轮。
        values["repair_count"] = int(values.get("repair_count") or 0) + 1
    cand = _first_answer(args)
    if cand is not None:
        values["candidate"] = cand
    output = {
        "ok": outcome.ok,
        "verify_status": outcome.status,
        "verify_exit": (step.returncode if isinstance(step.returncode, int) else -1),
        "verify_stdout": _clip(step.stdout, VERIFY_STDOUT_MAX),
    }
    values["verify_status"] = output["verify_status"]
    values["verify_exit"] = output["verify_exit"]
    values["verify_stdout"] = output["verify_stdout"]
    meta["verify"] = dict(outcome.detail)
    return output, outcome


def _submit_output(step: RawStep, values: dict) -> dict:
    """提交步：写 ``answer``（与 ``candidate``），并把自报的 ``verified`` 标记带进 output。

    ``verified`` 是 :func:`skill2fsm.judge.terminal_kind` 认得的三种线索之一：预算耗尽被
    强制提交的那次运行标 ``False``，P1 的 ``only_when: {terminal_kind: verified}`` 因此
    不会在它头上开火。**判不出类别时 P1 照查**（judge._kind_matches 的保守方向），所以
    这个标记只在采集侧确实知道时才写，不猜。
    """
    args = dict(step.args or {})
    text = _first_answer(args)
    if text is None:
        text = (step.text or "").strip()
    output: dict[str, Any] = {"ok": True, "answer": text}
    verified = args.get("verified", (step.meta or {}).get("verified"))
    if isinstance(verified, bool):
        output["verified"] = verified
    values["answer"] = text
    values["candidate"] = text
    return output


# --------------------------------------------------------------------------- #
# 整条运行
# --------------------------------------------------------------------------- #
def to_trace(raw: RawRun, *, acceptance: Optional[Callable[[Any], bool]] = None,
             prohibitions: Iterable[Prohibition] = (),
             artifacts_dir: Any = None,
             snapshot_model_vars: bool = False,
             phase_rules: str = "") -> Trace:
    """把一次原始运行折成一条已评判的 :class:`~skill2fsm.schema.Trace`。

    * 步号从 1 起连续，与 ``runtime.run_task`` 一致；
    * ``state`` 缺省是 :data:`~skill2fsm.schema.FALLBACK`——臂一臂二整条都是解释执行，每一步
      都是回退步；臂三由 ``runtime`` 自己写状态名，走不到这里。采集侧要覆盖就写
      ``step.meta["state"]``；
    * ``clause`` 一律留空：条款归属由编译 agent 在转录时补，采集时猜一个只会造出假溯源；
    * ``vars`` 的更新规则见模块文档的变量表。

    **verdict 的定线**：

    1. 任一禁止项被触犯 ⇒ ``rejected``，``error_step`` = 违规发生的那一步（由
       :func:`skill2fsm.judge.evaluate` 给出）。禁止项优先，哪怕答案是对的；
    2. 否则若注入了 ``acceptance`` 且它判否 ⇒ ``rejected``，``error_step`` 按下面的优先级；
    3. 否则若注入了 ``acceptance`` 且它判是 ⇒ ``accepted``；
    4. **没注入 ``acceptance``** ⇒ ``unknown``。没做客观验收就写 ``accepted``，等于往接受集
       里掺没验过的轨迹——这与 :func:`skill2fsm.judge.judged` 的缺省不同，是刻意的。

    **``error_step`` 的优先级**（``rejected`` 必须带上它，否则 ``schema.Trace`` 直接抛）：

    1. 禁止项违规的那一步——位置精确，且它就是拒绝集排除检查要盯的锚；
    2. **第一处偏离**：最早一次 ``verify_status != PASS`` 的核验步。这是运行内部第一次
       出现「事情不对」的可观察证据；
    3. 最后一步——什么线索都没有时的兜底（从头错到尾，只能指向结局）；
    4. 一步都没有的空运行记 ``0``（第 1 步之前），因为 ``schema`` 不允许 rejected 缺 error_step。
    """
    task_in = (raw.task or {}).get("input") if isinstance(raw.task, dict) else None
    outputs = _phases.outputs_of(task_in or {})     # 产出由**任务声明**，不由分类器猜文件名
    values = initial_vars()
    if snapshot_model_vars:
        # 模型侧快照（MODEL_VARS）：从文档引入的判断动作要读它们，标定要它们做样本。
        # 默认关着：test_17 钉着「每条记录恰好带 initial_vars 的六个变量」，那是单智能体
        # 路径与既有 80 条轨迹的契约；多智能体的采集路径显式打开。
        values.update(model_vars())
    records: list[Record] = []
    first_divergence: Optional[int] = None

    for idx, step in enumerate(raw.steps or (), start=1):
        rec, outcome = _record_for(step, idx, values, artifacts_dir=artifacts_dir,
                                   phase_rules=phase_rules, outputs=outputs)
        records.append(rec)
        if outcome is not None and outcome.status != PASS and first_divergence is None:
            first_divergence = idx

    draft = Trace(task=dict(raw.task or {}), arm=raw.arm, run=raw.run,
                  model=raw.model, harness=raw.harness,
                  verdict="unknown", records=records)
    check_canonical(draft)

    plist = list(prohibitions or ())
    banned = _judge.evaluate(draft, None, plist)      # 只跑禁止项：acceptance 留到下一步
    if banned.verdict == "rejected":
        verdict, error_step = "rejected", banned.error_step
    elif acceptance is None:
        verdict, error_step = "unknown", None
    elif acceptance(draft):
        verdict, error_step = "accepted", None
    else:
        last = records[-1].step if records else 0
        verdict, error_step = "rejected", (first_divergence
                                           if first_divergence is not None else last)

    return Trace(task=draft.task, arm=draft.arm, run=draft.run, model=draft.model,
                 harness=draft.harness, verdict=verdict, error_step=error_step,
                 records=records)


def check_canonical(trace: Trace) -> None:
    """校验跨臂可比性的两条硬约束，不合抛 :class:`TraceAdapterError`。

    1. 每个工具名都已是规范形（``canon_tool_name`` 的不动点），且没有任何提交别名漏网：
       ``math_verify.py`` 与 ``math_verify`` 在两条臂里各写各的，编译器会开出两个互不成环的
       状态，回放与路径一致率立刻失真；
    2. 提交那一步只叫 :data:`SUBMIT_TOOL`。P1 的 ``require_before`` 守的就是这个名字，改名
       等于把这条臂的违规检查关掉——而报告里会显示「违规率 0%」，看不出是关掉了。

    顺带校验步号从 1 起连续（编译器与回放都按下标定位 ``error_step``）。经
    :func:`with_begin` 垫过开局步的视图（首条是 step 0 的 BEGIN_TOOL）同样合法。
    """
    recs = list(trace.records or ())
    if recs and is_begin(recs[0].action) and recs[0].step == 0:
        recs = recs[1:]
    for i, rec in enumerate(recs, start=1):
        if rec.step != i:
            raise TraceAdapterError(
                f"步号不连续：第 {i} 条记录的 step={rec.step}（要 1 起连续）")
        act = rec.action or {}
        if act.get("kind") != "tool":
            continue
        name = str(act.get("name") or "")
        if name != canon_tool_name(name):
            raise TraceAdapterError(
                f"第 {i} 步的工具名 {name!r} 不是规范形"
                f"（要 {canon_tool_name(name)!r}）——跨臂就对不齐了")
        if name in SUBMIT_ALIASES and name != SUBMIT_TOOL:
            raise TraceAdapterError(
                f"第 {i} 步的提交动作叫 {name!r}，必须统一成 {SUBMIT_TOOL!r}，"
                f"否则 P1 的 require_before 在这条臂上永远不开火")


# --------------------------------------------------------------------------- #
# 落盘与读回
# --------------------------------------------------------------------------- #
_UNSAFE_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _stem(trace: Trace, index: int) -> str:
    """轨迹文件名：``<task_id>__<arm>__run<NN>``，缺的字段跳过，非法字符折成 ``_``。"""
    task = trace.task if isinstance(trace.task, dict) else {}
    parts = [str(task.get("task_id") or f"trace{index:04d}")]
    if trace.arm:
        parts.append(str(trace.arm))
    parts.append(f"run{int(trace.run):02d}")
    return _UNSAFE_RE.sub("_", "__".join(parts)).strip("_") or f"trace{index:04d}"


def write_jsonl(traces: Iterable[Trace], out_dir: Any) -> list[Path]:
    """把若干轨迹逐条写成 ``out_dir/<stem>.jsonl``，返回写出的路径（与入参同序）。

    还没搬走的代码正文（寄存在 ``Record.meta["code_bodies"]``）在这里一并搬到
    ``out_dir/artifacts/<sha>.py``，写出的 JSONL 里只剩哈希。``meta`` 不参与规范化也不参与
    评判，所以搬与不搬**不会改变任何 verdict**——落盘的那份和内存里那份判出来一样。

    **不改动入参**：搬运在深拷贝上做。同名（同任务同臂同轮次）时给文件加 ``-2``、``-3``
    后缀，绝不静默覆盖——覆盖掉的是一条采集不回来的轨迹。
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    used: set[str] = set()
    for i, trace in enumerate(traces, start=1):
        copy = trace.model_copy(deep=True)
        bodies: dict[str, str] = {}
        for rec in copy.records:
            got = rec.meta.pop("code_bodies", None)
            if isinstance(got, dict):
                bodies.update({str(k): str(v) for k, v in got.items()})
        _spill(bodies, root / ARTIFACTS_DIRNAME)

        stem = _stem(copy, i)
        name, n = stem, 1
        while name in used:
            n += 1
            name = f"{stem}-{n}"
        used.add(name)
        path = root / f"{name}.jsonl"
        path.write_text(copy.to_jsonl(), encoding="utf-8")
        paths.append(path)
    return paths


def read_jsonl(path: Any) -> Trace:
    """读回一条轨迹。文件不存在时报清楚是**哪个路径**，别让调用方去猜。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"轨迹文件不存在: {p}")
    return Trace.from_jsonl(p)


# --------------------------------------------------------------------------- #
# 通用 agent 轨迹：任何 harness 跑出来的记录都能进编译器
# --------------------------------------------------------------------------- #
def ensure_phases(trace: Trace, rules: str = "default") -> Trace:
    """按**命令正文**给工具步定阶段（probe / apply / verify / other）。就地改 ``action``，幂等。

    阶段是内容的函数，所以只要正文还在就**现算**，不管记录里原来写的是什么。这一条是有代价
    换来的：``runtime`` 曾经把状态自己的声明抄进记录，于是轨迹说的是「机器认为这一步该干
    什么」而不是「这一步干了什么」，编译因此是循环的——机器重新学到的只是它自己贴的标签。
    实测一条 4 步轨迹里三步标错（声明 probe 的状态里命令有 ``wb.save``，实际是 apply；声明
    apply 的状态只回读产出，实际是 verify）。

    正文取不到时（``_extract_code`` 把它抽走了、只剩 ``code_sha256``）保留原有的标注——
    拿哈希判不出这一步在干什么，宁可留着旧值也不要瞎改。专用工具（名字即用途）分类器返回
    空串，同样不动。
    """
    task = trace.task if isinstance(trace.task, dict) else {}
    outputs = _phases.outputs_of(task.get("input") or {})
    for rec in trace.records:
        act = rec.action
        if not isinstance(act, dict) or act.get("kind") != "tool":
            continue
        inp = dict(act.get("input") or {})
        if not _phases.command_text(inp).strip() and act.get("phase"):
            continue                       # 正文没了，旧标注是唯一的线索
        p = _phases.classify(str(act.get("name") or ""), inp, rules, outputs=outputs)
        if p:
            act["phase"] = p
    return trace


def read_raw_jsonl(path: Any) -> tuple[RawRun, Optional[bool]]:
    """读一份**原始 agent 事件日志**（不是本仓库的 Trace 格式），折成 :class:`RawRun`。

    首行是头部：``{"task": {...}, "verdict": "accepted"|"rejected"}``（也认 ``"ok": true/false``，
    或 ``"input": {...}`` 直接当任务输入）。其余每行一步，字段与 :class:`RawStep` 同名::

        {"kind": "tool",  "name": "bash",     "args": {"command": "ls"}, "stdout": "...", "returncode": 0}
        {"kind": "tool",  "name": "file_ops", "args": {"op": "read", "path": "a.xlsx"}, "stdout": "..."}
        {"kind": "model", "text": "..."}
        {"kind": "end"}

    返回 ``(RawRun, 判决)``；判决 ``None`` 表示日志没说对错（进不了 T+ 也进不了 T−）。
    """
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise TraceAdapterError(f"空日志: {p}")
    head = json.loads(lines[0])
    if not isinstance(head, dict):
        raise TraceAdapterError(f"{p} 首行不是 JSON 对象")
    task = dict(head.get("task") or {})
    if "input" in head and "input" not in task:
        task["input"] = dict(head["input"] or {})
    task.setdefault("task_id", str(head.get("task_id") or p.stem))
    verdict: Optional[bool] = None
    if "verdict" in head:
        verdict = str(head["verdict"]).lower() == "accepted"
    elif "ok" in head:
        verdict = bool(head["ok"])
    steps: list[RawStep] = []
    for i, ln in enumerate(lines[1:], start=1):
        d = json.loads(ln)
        if not isinstance(d, dict) or not d.get("kind"):
            raise TraceAdapterError(f"{p} 第 {i} 步缺 kind")
        steps.append(RawStep(kind=str(d["kind"]), name=str(d.get("name") or ""),
                             args=dict(d.get("args") or d.get("input") or {}),
                             stdout=str(d.get("stdout") or d.get("output") or ""),
                             stderr=str(d.get("stderr") or ""),
                             returncode=d.get("returncode"),
                             text=str(d.get("text") or ""), meta=dict(d.get("meta") or {})))
    return RawRun(task=task, steps=steps, arm=str(head.get("arm") or "agent"),
                  run=int(head.get("run") or 0), model=str(head.get("model") or ""),
                  harness=str(head.get("harness") or "")), verdict


def tool_output(rec: Any) -> dict:
    """一条工具记录的完整产出：``output`` 加上采集时放进 ``meta`` 的状态字段（返回码、stderr）。

    这是轨迹**格式**的知识，只住在适配器里：编译器只看合并后的产出字典，不知道哪些字段来自 meta。
    """
    out = dict(getattr(rec, "output", None) or {})
    meta = getattr(rec, "meta", None) or {}
    for k in ("returncode", "stderr"):
        if k in meta and k not in out:
            out[k] = meta[k]
    return out


def load_any_trace(path: Any, *, phase_rules: str = "default",
                   artifacts_dir: Any = None) -> Trace:
    """读一份轨迹文件，本仓库 Trace 格式或原始 agent 日志都认，并补齐阶段。

    判法看第二行：Trace 的记录行有 ``step``/``action``，原始日志的步骤行有 ``kind``。
    """
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    second = json.loads(lines[1]) if len(lines) > 1 else {}
    if isinstance(second, dict) and "action" in second and "step" in second:
        tr = read_jsonl(p)
    else:
        raw, ok = read_raw_jsonl(p)
        tr = to_trace(raw, acceptance=(None if ok is None else (lambda _t, _ok=ok: _ok)),
                      artifacts_dir=artifacts_dir, phase_rules=phase_rules)
    return ensure_phases(tr, phase_rules) if phase_rules else tr
