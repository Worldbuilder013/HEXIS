"""阶段分类：给「同一个通用工具、不同用途」的步骤一个可判定的身份。

**这一层为什么必须有。** :func:`hexis.traces.normalize.canon_action` 故意不把工具参数算进状态
身份——「参数活在变量里，不属于状态身份」。这条规矩对 ``math_verify`` 这种**名字即用途**的
专用工具是对的；对 ``bash`` / ``run_python`` 这种**通用**工具就塌了：查表、写结果、回头核对
三步全折成一个 ``('tool', 'name=run_python')``，整条轨迹变成一个状态自环三次，什么结构也学
不出来。信息全在命令文本里，而命令文本按设计不进 KEY。

**三阶段是技能无关的。** 任何「产出一个东西」的技能都是同一个形状::

    probe   读输入、看清楚现状          —— 不碰产出
    apply   写产出                      —— 有写操作
    verify  回读自己刚写的东西核对       —— 提到产出但不写

技能相关的只有**怎么认出「写」**，而这一层也不必逐技能写规则：写操作在 Python 与 shell 里
就那么几种形态（:data:`WRITE_SIGNALS`），产出是哪个文件由**任务自己声明**（``output_path``
之类，:func:`outputs_of` 从任务输入里认）。所以默认分类器 :func:`default_phase` 对任何技能
都直接可用，不需要为每个技能各写一份。

真有技能需要自己的判据（领域特有的写操作），用 :func:`register` 登记一个同签名的函数即可，
:attr:`hexis.machine.schema.Machine.phase_rules` 记着用的是哪一套。

**判据是纯函数，不是模型。** 谁都能拿 ``out/artifacts/<sha>.py`` 里的正文重跑一遍核对。

**在采集时判，不在编译时判。** :func:`hexis.traces.trace_adapter._extract_code` 会把代码正文抽到
artifacts、记录里只留 ``code_sha256``——编译时已经读不到正文了。所以 ``to_trace`` 在抽走正文
**之前**算出阶段，写进 ``action["phase"]`` 随记录走。机器状态那侧同样带 ``phase``
（:class:`~hexis.machine.schema.ToolAction` 的字段），于是 ``canon_action`` 两侧对称地把它算进
KEY，回放不需要任何特殊处理——这与只在编译侧生效的 ``pred=``（往回看的语境）不同：阶段是
**动作自身**的属性，两侧都算得出。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

#: 阶段取值。``other`` 是兜底（命令为空、认不出）——它照样是一个可判定的值；
#: **空串**才表示「这一步不参与阶段细化」，行为与从前逐字相同。
PHASE_PROBE = "probe"
PHASE_APPLY = "apply"
PHASE_VERIFY = "verify"
PHASE_OTHER = "other"

PHASES = (PHASE_PROBE, PHASE_APPLY, PHASE_VERIFY, PHASE_OTHER)

#: 命令正文可能落在参数的哪些键上。与 trace_adapter.CODE_KEYS 对齐。
TEXT_KEYS = ("code", "command", "script", "cmd", "source", "argv")

#: **通用**工具：名字不说明用途，非细化不可。专用工具（``math_verify``、``apply_formula``）
#: 的 ``name`` 已经是身份了，不参与细化。
GENERIC_TOOLS = frozenset({
    "bash", "sh", "shell", "zsh", "run_python", "python", "python3",
    "run_script", "exec", "run", "execute", "run_command", "run_code",
    "file_ops", "file", "files", "fs", "filesystem",
})

#: ``file_ops`` 这类工具没有命令正文，用途写在 ``op``/``action``/``mode`` 这个键上。
_OP_KEYS = ("op", "operation", "action", "mode", "command")
_WRITE_OPS = frozenset({"write", "append", "create", "delete", "remove", "move", "rename",
                        "copy", "mkdir", "touch", "save", "put", "edit", "replace", "insert"})
_READ_OPS = frozenset({"read", "list", "ls", "stat", "exists", "search", "grep", "find",
                       "cat", "head", "tail", "get", "open", "view"})

#: 「这条命令在写东西」的形态。Python 与 shell 各几种，与具体技能无关。
#: 逐条都是实测见过的写法；宁可多收一条，也不要把一次写错分成读——那会让图上没有 apply。
WRITE_SIGNALS: tuple[str, ...] = (
    ".save(", ".save (", ".write(", ".writelines(", ".to_csv(", ".to_excel(",
    ".to_json(", ".to_parquet(", "json.dump(", "pickle.dump(", "yaml.dump(",
    ".dump(", ".commit(", ".flush(",
    "write_text(", "write_bytes(", "makedirs(", "mkdir(",
    "shutil.copy", "shutil.move", "os.replace(", "os.rename(", "os.remove(",
)

#: shell 里的写操作（按词边界匹配，避免 ``cpu`` 命中 ``cp``）。
_SHELL_WRITE_RE = re.compile(
    r"(^|[;&|]|\s)(cp|mv|rm|tee|touch|mkdir|install|dd|git\s+commit|sed\s+-i)\s|>>?\s*\S")

#: ``open(..., "w")`` / ``'a'`` / ``'x'``：第二个参数带写模式。
_OPEN_WRITE_RE = re.compile(r"""open\s*\([^)]*,\s*['"][^'"]*[wax]""")

#: 任务输入里「这是产出」的键名线索。产出**由任务声明**，不由分类器猜文件名。
_OUTPUT_KEY_HINTS = ("output", "out_path", "outpath", "dst", "dest", "target", "_out")

CLASSIFIERS: dict[str, Callable[..., str]] = {}


def register(name: str) -> Callable:
    """登记一个分类器。像禁止项与打标器一样是**受审查的纯函数**，不调模型。

    签名 ``(tool_name, input_dict, *, outputs=()) -> str``，返回 :data:`PHASES` 之一或空串。
    """
    def deco(fn: Callable[..., str]):
        CLASSIFIERS[name] = fn
        return fn
    return deco


def command_text(inp: Mapping) -> str:
    """从工具入参里拼出可供判读的命令正文。取不到就空串。

    **不读 ``*_sha256`` / ``*_path``**——那是正文被抽走之后留下的指纹，拿它判阶段等于拿
    哈希猜内容。
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
    """从任务输入里认出「产出是什么」：键名带 output/dst/target 之类的那些值 + 键名本身。

    产出**由任务声明**，不由分类器猜。带上键名是因为执行器往往把路径作为变量注入代码
    （``OUTPUT_XLSX``），命令里出现的是变量名而不是路径。
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
    """这条命令有没有写操作。技能无关，只看形态。"""
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
    """通用三阶段。**任何技能都能直接用**，不需要为每个技能各写一份。

    判据按特异性从高到低，顺序不能反：

    1. 有写操作 ⇒ **apply**；
    2. 没写、但提到任务声明的产出 ⇒ **verify**（在回读自己刚写的东西）；
    3. 其余非空命令 ⇒ **probe**。

    先判写、再判产出是要紧的：写那一步几乎总会提到产出路径（``wb.save(OUTPUT)``），反过来
    判会把每一次写都错分成校验，图上于是只有校验、没有应用。

    ``outputs`` 为空时（技能不产出文件，比如数学题只交一个答案字符串）第 2 条自然失效，
    退化成「写 / 读」两分——仍然可判定，只是分得粗。
    """
    if (tool or "").lower() not in GENERIC_TOOLS:
        return ""                                   # 专用工具：name 已经是身份
    c = command_text(inp)
    if not c.strip():
        return _phase_by_op(inp, outputs)
    if writes_something(c):
        return PHASE_APPLY
    if outputs and mentions(c, outputs):
        return PHASE_VERIFY
    return PHASE_PROBE


def _phase_by_op(inp: Mapping, outputs: Sequence[str]) -> str:
    """没有命令正文的通用工具（``file_ops``）：按 ``op`` 与路径判。判据同上，只是换了读法。"""
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
    """按 ``rules`` 指名的分类器判阶段。``rules`` 为空或未登记 ⇒ 空串（不细化）。"""
    fn = CLASSIFIERS.get(rules or "")
    if fn is None:
        return ""
    try:
        p = fn(str(tool or ""), inp if isinstance(inp, Mapping) else {}, outputs=tuple(outputs))
    except TypeError:                                # 不吃 outputs 的老分类器
        try:
            p = fn(str(tool or ""), inp if isinstance(inp, Mapping) else {})
        except Exception:                            # noqa: BLE001
            return ""
    except Exception:                                # noqa: BLE001
        return ""
    return p if p in PHASES else ""


def phase_of(action: Any) -> str:
    """从一个动作（记录里的裸 dict 或机器里的 Action 模型）读出已记下的阶段。"""
    if isinstance(action, Mapping):
        return str(action.get("phase") or "")
    return str(getattr(action, "phase", "") or "")


__all__ = ["CLASSIFIERS", "GENERIC_TOOLS", "PHASES", "PHASE_APPLY", "PHASE_OTHER",
           "PHASE_PROBE", "PHASE_VERIFY", "TEXT_KEYS", "WRITE_SIGNALS", "classify",
           "command_text", "default_phase", "mentions", "outputs_of", "phase_of",
           "register", "writes_something"]
