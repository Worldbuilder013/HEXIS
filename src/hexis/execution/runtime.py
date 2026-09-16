"""状态机解释器：跑一台机器（或一段 FALLBACK 解释）出一条轨迹。

一次运行是一个确定性循环：跑当前状态的动作 → 按声明顺序求跳转条件 → 第一个为真的边胜出 →
到终止或撞步数上限。三件事这里必须做对：

**确定性。** 同一份输入跑两次走同一条路。转移按声明顺序求值（兜底边永远最后），判断动作
交给模型的桩/实现自行保证 ``temperature=0``。不确定的解释器会让「这次退步是改坏了还是抽签
抽差了」变得无法分辨。

**停机原因要能分辨。** ``terminal``（正常）/ ``state_error``（某步炸了或条件求值撞未定义
变量）/ ``stuck``（没有边可走）/ ``max_steps``（转圈）。这四种在「结果不对」上一模一样，
但要改的东西完全不同。

**轨迹是「关于它自己」的。** 每条 :class:`~hexis.machine.schema.Record` 描述这台机器怎么走、
读写了哪些变量，不含参照答案，因此整条都能交给编译器 agent 看。工具结果由宿主（这里）
填进 ``output``，不由模型编。

``FALLBACK`` 状态特殊：到达它就切「模型读整份文档 + 历史，逐步解释执行」，同时照常记录
轨迹。条件覆盖不到、判断弃权、编译尚浅时都会走到这里——它让机器在任何学习程度下都能用。
解释段照样跑真动作：工具、生成（``model``）、带最终答案的终止都认。

**执行侧的账另开一路。** 每步的 token、耗时、真正执行的 argv 进 ``Record.meta``，整趟的用量
与回退位置进 :class:`RunResult`；两者都不进 ``action``/``output``，因此不参与规范化、不参与
评判、不改变状态身份。账的第一纪律是**测不到就说测不到**：模型接口没报 usage 时留 ``None``，
绝不按单价估一个数填进去——估出来的数字会被下游当实测读。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from hexis.machine import cond
from hexis.traces import normalize as _normalize
from hexis.traces import phases as _phases
from hexis.machine.schema import FALLBACK, Machine, Record, Trace

STOP_TERMINAL = "terminal"
STOP_STATE_ERROR = "state_error"
STOP_STUCK = "stuck"
STOP_MAX_STEPS = "max_steps"
STOP_FALLBACK_EXHAUSTED = "fallback_exhausted"

_COUNTER_EXIT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*(\d+)\s*$")

_VAR_RE = re.compile(r"\$\{(\w+)\}")


@dataclass
class RunResult:
    """一次运行的全部可观察结果。``trace`` 是那条轨迹（verdict 待评判补）。

    前五个字段是「怎么停的」，后面几个是**实验报告要统计的账**，两条纪律：

    * **测不到就说测不到。** ``prompt_tokens``/``completion_tokens`` 为 ``None`` 表示模型接口
      这一趟根本没报用量（:class:`~hexis.llm.model_iface.ScriptedModel` 没有 ``usage()``，真实
      端点也可能不报），**不是**「花了 0 个 token」。``0`` 只在确实一次模型都没调时出现。按
      token 单价估出来的数字会被下游当实测读，那比空着更糟。``unmeasured_calls`` 明说这趟有
      几次调用没测到。
    * **回退要能定位。** 交付物的指标 5 是「回退率与位置清单」：``fallback_steps`` 是解释段跑
      了几步，``fallback_entry`` 是**从哪个状态**切进去的，``fallback_entry_step`` 是解释段第一
      条记录的步号。起点即回退态（空机器）时 ``fallback_entry`` 记 ``FALLBACK`` 自己，表示
      「全程解释」而不是「从某处退下来」。

    ``wall_s`` 是本机墙钟，永远测得到，所以是数不是 ``None``。
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
    #: 回退枢纽里重试了几次（回到出错那一步的入口重新生成、重新执行）。
    retries: int = 0

    def path(self) -> list[str]:
        """走过的状态序列。确定性测试比它，不比耗时。"""
        return [r.state for r in self.trace.records if r.state]

    def entered_fallback(self) -> bool:
        """这趟有没有落进解释段（回退率的分子）。"""
        return self.fallback_steps > 0


# --------------------------------------------------------------------------- #
# 两段纯逻辑：从模型输出里取 JSON 对象、截取异常首行
# --------------------------------------------------------------------------- #
def _first_json_object(text: str) -> Optional[dict]:
    """从一段文本里取第一个完整的 JSON 对象。模型很少只回一个裸对象，能救回就别退回失败。"""
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
    """把 traceback 里真正那句异常提到前面（截断取前 N 字符时别把它截掉）。"""
    text = (err or "").strip()
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("Traceback"):
        return text[:limit]
    tail = lines[-1]
    frame = next((ln.strip() for ln in reversed(lines[:-1])
                  if ln.strip().startswith("File ")), "")
    head = tail + (f"  （{frame.split('/')[-1]}）" if frame else "")
    return (head + "\n" + text)[:limit]


# --------------------------------------------------------------------------- #
# 执行侧的账：用量 / 耗时 / argv / 提示词摘要
# --------------------------------------------------------------------------- #
#: 从模型接口的 ``usage()`` 里认的键。多出来的键一律不管（别的实现可以自由加）。
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "llm_calls", "unmeasured_calls")


def _usage_of(model: Any) -> Optional[dict]:
    """模型接口报的**累计**用量快照；没有 ``usage()`` 就返回 ``None``。

    鸭子类型：``llm_client.ModelAdapter`` 有 ``usage()``，测试桩
    :class:`~hexis.llm.model_iface.ScriptedModel` 没有——不为桩补接口、也不给桩编数，缺就是
    缺（缺的后果是 token 字段留 ``None``，见 :func:`_step_tokens`）。
    """
    fn = getattr(model, "usage", None)
    if not callable(fn):
        return None
    try:
        u = fn()
    except Exception:                                       # noqa: BLE001
        return None                                         # 记账炸了不许影响执行
    if not isinstance(u, Mapping):
        return None
    return {k: u.get(k) for k in _USAGE_KEYS}


def _grew(before: Optional[dict], after: Optional[dict], key: str) -> Optional[int]:
    """两次快照之间某个计数涨了多少。任一侧不是整数就返回 ``None``（不当 0）。"""
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
    """一段区间的 ``(prompt_tokens, completion_tokens, 没测到的调用数)``。

    三种情形分清楚：**一次模型都没调** ⇒ ``(0, 0, 0)``，这是测量不是估算；**接口不报用量**
    ⇒ ``(None, None, calls)``；**报了、但这段的调用全被它记成 unmeasured** ⇒ 同样 ``None``
    ——端点没给 usage 时 ``ModelAdapter`` 会把 0 累加进去，照抄那个 0 就等于编数。
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
    """这一步**真正执行的** argv。工具产出里带就用产出里的，否则问工具对象自己。

    :class:`~hexis.legacy.sandbox.ExecResult` 记的是 ``command``，别的工具可能叫 ``argv``/``cmd``。
    都拿不到就返回 ``None``——报告里宁可空着，也不要把 input 模板反推出来的命令行冒充成真跑过
    的那条。这一栏抓的正是两者的差：``math_verify.py`` 的 ``--json`` 必须在子命令**之前**，而
    执行器还会按标定记录在案的那条控制把 ``.venv/bin/python3`` 前缀改写成本机真实解释器
    （见 ``docs/HARNESS_CALIBRATION.md`` 第 4 节第 3 条）。模板里写的和真跑的不是同一行。
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
    """一步的执行侧账，进 ``Record.meta``。

    ``prompt_tokens``/``completion_tokens`` 两个键**每步都在**（值可能是 ``None``）：报告要能
    分清「这步没花」与「这步没测到」，键时有时无就只能猜。
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
    """渲染后提示词的稳定摘要：``sha256(模板 + 读到的变量取值的规范 JSON)``。

    同模板 + 同取值 ⇒ 同摘要，且跨进程稳定（不用 :func:`hash`，它每进程加盐）。它是「这一步
    到底问了什么」的可核对锚，全文则不进轨迹（理由见 :func:`_model_action`）。
    """
    payload = json.dumps({str(k): values.get(k) for k in sorted(values or {}, key=str)},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256((template + "\x00" + payload).encode("utf-8")).hexdigest()


def _model_action(template: str, reads: Any, values_read: Mapping, *,
                  template_id: str, inline_template: bool = True) -> dict:
    """``model`` 动作在轨迹里的样子：**模板 id + 模板文本 + reads + 提示词 sha256**，
    **不含渲染后的全文**。

    为什么不记渲染全文：渲染结果里嵌着整道题面（MATH-500 的题干动辄几百字），每步存一份会把
    一条轨迹撑到题面的十几倍，而那段文本**轨迹里已经有了**——它在 ``Record.vars`` 里，再记一遍
    只是重复。编译器要的是另外几样，这里都给：``template_id`` + 模板文本给**状态身份**
    （``normalize`` 的严格档比的就是 ``prompt``，同源两侧因此对得上）、``reads`` 给「这一步真正
    消费了哪些变量」的变量归属、``prompt_sha256`` 给「同模板同取值 ⇒ 同一次提问」的可核对性
    ——摘要在手，日后要复现某一步问了什么，拿模板与那步的 ``vars`` 重算一遍就能自证。

    ``inline_template=False`` 用于 FALLBACK：那里的「模板」是整份技能文档（几十 KB），逐步内联
    比记渲染全文还糟，所以只留 id 与摘要。
    """
    act: dict = {"kind": "model", "template_id": template_id}
    if inline_template:
        act["prompt"] = template
    act["reads"] = [str(r) for r in (reads or [])]
    act["prompt_sha256"] = prompt_digest(template, values_read)
    return act


def bind_outputs(raw: Any, binds: Any) -> Any:
    """按 ``ToolAction.binds`` 把工具产出改名：``{"stdout": "workbook_content"}`` 让
    ``out["stdout"]`` 同时以 ``workbook_content`` 之名出现。原键保留，所以 ``writes`` 里
    同时列 ``returncode`` 与语义名都收得到。``raw`` 不是对象或没有绑定时原样返回。"""
    if not binds or not isinstance(raw, dict):
        return raw
    mapped = dict(raw)
    for src, dst in dict(binds).items():
        if src in raw and dst:
            mapped[dst] = raw[src]
    return mapped


def rebuild(raw: Any, names: list[str]) -> dict:
    """按名字白名单**重新构造**一个对象，不是过滤——声明之外的一律丢弃，新键默认扣下。

    机内状态出参与会话侧出参共用这条
    政策，判据放一处。``raw`` 不是对象时返回空 dict。
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
# 模板填充与选边
# --------------------------------------------------------------------------- #
def fill_template(obj: Any, values: dict) -> Any:
    """把动作 input 模板里的 ``${var}`` 用变量填掉。整体是 ``${var}`` 时保留原类型。"""
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
    """选下一条边：按声明顺序（兜底边最后）第一个条件为真的胜出。

    返回 ``(edge, error)``。条件求值撞未定义变量时返回 error（不当假）——静默走兜底会把
    「谓词写崩」伪装成「算错了」，那是最难查的失败。
    """
    for t in machine.out_edges(sid):
        if not t.cond:
            return t, ""
        try:
            if cond.evaluate(t.cond, values):
                return t, ""
        except cond.CondError as exc:
            return None, f"{sid} 的条件 {t.cond!r} 求值失败: {exc}"
    return None, ""


# --------------------------------------------------------------------------- #
# 执行一个非 FALLBACK 状态
# --------------------------------------------------------------------------- #
def _phase_of_call(act, inp: dict, phase_rules: str, outputs: tuple) -> dict:
    """这一步**实际在干什么**：从渲染后的入参正文现判，而不是照抄状态自己的声明。

    阶段是动作自身的属性，机器那侧有、记录这侧也必须有——``canon_action`` 两档都把它算进 KEY，
    丢了回放第一步就对不上。但**来源**不能是状态的声明：状态说的是「这一步该是什么」，记录
    要记的是「这一步是什么」。照抄会让编译变成循环——机器重新学到的只是它自己贴的标签。

    实测过后果：一个声明为 ``probe`` 的状态，模型给它写了一条 ``wb.save(...)`` 的命令，记录
    照抄成 ``probe``；另一条只回读产出簿、本该是 ``verify``，也照抄成 ``probe``。一条 4 步的
    轨迹里三步标错，而这些标签正是下一轮编译的身份依据。

    判据与外部日志那条路同源（:func:`hexis.traces.phases.classify`），所以两种来源的轨迹可比。
    分类器给不出（专用工具、没登记规则）时退回状态的声明，行为与从前一致。
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
    """执行状态动作，返回 (记录用的 action, output, error, llm_calls_delta)。就地更新 values。

    ``phase_rules``/``outputs`` 用来判**这一步实际在干什么**（见 :func:`_phase_of_call`）。
    """
    act = state.action
    kind = act.kind
    if kind == "tool":
        inp = fill_template(act.input, values)
        ph = _phase_of_call(act, inp, phase_rules, outputs)
        if _normalize.is_begin(act):
            # 开局工具：空操作。不问工具表——它是编译器垫的，任何工具表都不该认识它。
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
                f"判断动作调用失败: {type(exc).__name__}: {exc}", 1
        values[act.writes[0]] = label
        return ({"kind": "judge", "prompt": act.prompt, "reads": act.reads},
                {act.writes[0]: label}, "", 1)
    if kind == "model":
        vread = {k: values.get(k) for k in act.reads}
        # 记模板 id + 模板 + reads + 摘要，不记渲染全文（题面已经在 vars 里）。见 _model_action。
        rec_act = _model_action(act.prompt, act.reads, vread, template_id=state.id)
        calls = 0
        try:
            try:
                out = model.generate(prompt=act.prompt, values=vread)
                calls += 1
            except Exception as first:                          # noqa: BLE001
                # 一次生成炸了（多半是思维链把预算吃光、答案为空）：同一状态原地再问一次，
                # 比转回退枢纽从头重跑整台机器便宜得多。第二次还炸才按原逻辑报错。
                calls += 1
                if "预算" not in str(first) and "JSON" not in str(first) and "timed out" not in str(first).lower():
                    raise
                out = model.generate(prompt=act.prompt, values=vread)
                calls += 1
            got = rebuild(out, act.writes)
            if act.writes and not any(str(got.get(w) or "").strip() for w in act.writes):
                # 回了合法 JSON 但没有一个声明的产出键（或全空）：把「缺哪些键」告诉模型再问一次。
                # 这是解释器对格式失误的补救，不改状态的语义，也不替模型决定内容。
                out = model.generate(prompt=act.prompt + f"\n\nYour previous answer was a JSON object without the required "
                                     f"keys {list(act.writes)} (or with empty values). Return exactly one JSON object whose keys "
                                     f"are exactly {list(act.writes)}, each with non-empty content.", values=vread)
                calls += 1
        except Exception as exc:                                # noqa: BLE001
            return rec_act, {}, \
                f"生成动作调用失败: {type(exc).__name__}: {exc}", max(calls, 1)
        values.update(rebuild(out, act.writes))
        return rec_act, out, "", calls
    if kind == "user":
        return {"kind": "user"}, {}, "密闭环境没有用户接口（user 动作不支持）", 0
    return {"kind": kind}, {}, f"不认的动作类型 {kind}", 0


# --------------------------------------------------------------------------- #
# FALLBACK：解释执行一步
# --------------------------------------------------------------------------- #
#: 解释模式回复里属于「控制」的键：它们说这一步**做什么**，不是这一步**产出**的值。
_CONTROL_KEYS = frozenset({"kind", "name", "input", "reads", "writes", "output",
                           "values", "terminal", "prompt", "note", "thought",
                           "reason", "why"})


def _payload(action: Mapping) -> dict:
    """解释回复里模型自己产出的那部分值：优先 ``output``/``values``，否则取控制键之外的键。

    两种写法都认，是因为「回一个 JSON 对象」这条约束下模型两种都会写；控制键的黑名单让
    ``{"kind":"end","terminal":"done","answer":"42"}`` 这种扁平写法也能把答案摘出来。
    """
    for key in ("output", "values"):
        v = action.get(key)
        if isinstance(v, Mapping):
            return dict(v)
    return {k: v for k, v in action.items() if k not in _CONTROL_KEYS}


def _history_item(rec: Any) -> Any:
    """喂给解释的一条历史：**去掉 meta**。

    ``meta`` 是执行侧的账（token、耗时、argv），与「已经做过什么」无关；``ModelAdapter`` 把
    每条历史压到 800 字符，让记账去挤那点预算，等于拿宿主的实现细节换掉真正要看的动作与结果。
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
    """FALLBACK 一步：模型读文档+历史+变量给下一动作，宿主执行并记录。

    返回 ``(record, done, error)``。``done`` 表示动作是 end（该停机）。就地更新 values。

    认四种动作：``tool``（宿主执行，结果由宿主填）、``model``（**这一趟 generate 本身**就是
    那次生成，产出直接收下，不再问第二次）、``end``（可带最终答案）、其余一律记成「不认的动作
    类型」并停。解释段能跑真动作、能交出答案，是为了让机器在结构还没编译出来时也能把题真的
    做完——否则回退只是「优雅地放弃」，三臂实验里第一臂就没意义了。

    每步的 token/耗时进 ``Record.meta``（一步解释 = 一次模型调用）。
    """
    hist = tuple(_history_item(r) for r in history)
    vread = dict(values)                    # 喂给这次解释的变量快照（提示词摘要按它算）
    reads = sorted(vread, key=str)          # 解释一步把整张变量表都读进去了，如实记
    u0 = _usage_of(model)
    t0 = time.perf_counter()

    def _rec(action: dict, output: Optional[dict] = None, *,
             argv: Optional[list] = None) -> Record:
        """按当前 values 收一条记录，顺带结这一步的账。"""
        return Record(step=step, state=FALLBACK, action=action,
                      output=dict(output or {}), vars=dict(values),
                      meta=_meta((time.perf_counter() - t0) * 1000.0, 1, u0,
                                 _usage_of(model), argv=argv))

    try:
        action = model.generate(prompt=doc, values=vread, history=hist)
    except Exception as exc:                                    # noqa: BLE001
        rec = _rec(_model_action(doc, reads, vread, template_id=FALLBACK,
                                 inline_template=False))
        return rec, False, f"FALLBACK 解释调用失败: {type(exc).__name__}: {exc}"
    kind = action.get("kind")
    if kind == "end":
        # 终止可以带最终答案（``{"kind":"end","terminal":"done","answer":"42"}``）：数学机器的
        # 回退段必须能**交卷**，答案落进 values 与 output，之后照常评判。没带就是空 dict，
        # 记录形状与从前一模一样。
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
        # 与 tool 分支的差别是有意的：tool 的产出来自**宿主**，必须按模型声明的 writes 收，
        # 否则工具能把一堆东西倒进变量表；model 的产出本来就是模型自己写的，再拿它自己的声明
        # 去过滤它自己的产出买不到任何约束，只会在它忘了写 writes 时把这一步变成空转（然后
        # 一路空转到步数上限）。所以：声明了按声明收，没声明就照单全收。
        names = ([str(w) for w in declared] if isinstance(declared, (list, tuple))
                 else list(produced))
        values.update(rebuild(produced, names))
        rec = _rec(_model_action(doc, action.get("reads") or reads, vread,
                                 template_id=FALLBACK, inline_template=False),
                   produced)
        return rec, False, ""
    rec = _rec({"kind": kind})
    return rec, False, f"FALLBACK 解释给出不认的动作类型 {kind!r}"


# --------------------------------------------------------------------------- #
# 跑一个任务
# --------------------------------------------------------------------------- #
#: :func:`halt_at_fallback` 的哨兵：一个**不存在的**状态名。小写下划线，与真实状态名
#: （``s1`` / ``FALLBACK`` / ``END_*``）不可能撞；真撞上会自动加后缀避开。
HALT_SENTINEL = "__halt_at_fallback__"


def halt_at_fallback(machine: Machine) -> Machine:
    """返回一台等价副本，它**走到回退态就停机**，不切 :func:`interpret_step` 的解释循环。

    ``run_task`` 只在 ``cur == machine.fallback`` 时切自带的解释执行；把 ``fallback`` 指到一个
    不存在的状态名，控制流就正常落到那个回退状态**本身**，它的动作是 ``end``（空机器与编译
    产物都如此），机器于是在那里干净停机，控制权回到调用方手上。

    三处要这件事，理由各不相同但都不希望 runtime 自己解释：臂三要回退段由与另两臂**同一个**
    执行器续跑（否则「回退段花了多少」不可比）；一致性检查要看机器**自己**走到哪为止；
    ``--no-fallback`` 要的是「进回退就停」。所以实现只留这一份。

    原机器一个字节不动；哨兵撞上真状态就加后缀——撞上而不避，那个状态会被当成回退态，
    机器在那儿就地开始解释执行。
    """
    name = HALT_SENTINEL
    while name in machine.states:                           # 实际撞不上，但撞上后果严重
        name += "_"
    return machine.model_copy(update={"fallback": name})


def entry_of(machine: Machine, sid: str) -> str:
    """一个工具状态的入口：专门给它生成参数的模型状态（唯一后继是它、写它模板里的变量），没有就是它自己。"""
    st = machine.states.get(sid)
    if st is None or st.action.kind != "tool":
        return sid
    need = set(_VAR_RE.findall(json.dumps(st.action.input, ensure_ascii=False)))
    for gid, g in machine.states.items():
        if g.action.kind != "model" or getattr(g.action, "observable", False):
            continue
        # 生成门自己可能挂着计数上限出口（cnt >= K → 回退态），那不算它的后继
        succ = [t.to for t in g.transitions
                if not (t.to == machine.fallback and _COUNTER_EXIT_RE.match(t.cond or ""))]
        if succ == [sid] and need & set(g.action.writes):
            return gid
    return sid


def run_task(machine: Machine, task: dict, *, model, tools, doc: str = "",
             max_steps: Optional[int] = None, on_error: str = "stop",
             retries: int = 0, interpret: bool = True) -> RunResult:
    """跑一台机器完成一个任务，返回轨迹与停机原因。

    ``on_error="fallback"`` 时机器段某一步炸了（工具报错、模型调用失败）不再当场停机，而是转到
    ``machine.fallback`` 交给解释段接着做——真实采集要的是这个：机器段学得还不全时，一次坏命令
    不该让整趟运行报废，回退段照样能把题做完、判分、进 T+。默认 ``"stop"`` 保持旧语义
    （密闭测试与三臂实验按它记 state_error）。解释段自己炸了一律停机，两档都一样。

    工作状态初值 = 任务输入的字段（供 FALLBACK 解释读取）叠加变量表的 init/init_from。
    到达 ``machine.fallback`` 即切解释模式，并在 :class:`RunResult` 上记下**从哪个状态退进去
    的、解释段跑了几步**（回退率与位置清单要的就是这两样）。每步的 token/耗时/argv 记进
    ``Record.meta``，整趟的用量记在 :class:`RunResult` 上——测不到就留 ``None``。

    **回退不是终结。** ``retries > 0`` 时回退态先是一个**重试枢纽**：每次进入，回到最近执行的
    工具状态的入口（有生成门就回生成门，重新生成参数），把触发回退的计数变量清零，最多 ``retries``
    次；重试用完才交给解释段（``interpret=True``）或停机（``stopped="fallback_exhausted"``）。
    ``on_error="fallback"`` 时状态出错也走这个枢纽。
    """
    task_input = task.get("input", {})
    task_outputs = _phases.outputs_of(task_input)     # 产出由任务声明，不由分类器猜
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
    # 起点就是回退态（空机器）：没有「从哪退下来」可言，记它自己表示「全程解释」。
    fallback_entry: Optional[str] = (machine.initial
                                     if machine.initial == machine.fallback else None)
    fallback_entry_step: Optional[int] = None
    retry_used = 0
    last_tool: Optional[str] = None            # 最近执行过的工具状态
    came_from: Optional[str] = None            # 这次进回退态是从哪个状态、经哪条边
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
        # ---- FALLBACK：先当重试枢纽，重试用完再解释执行 / 停机 ---- #
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
                          f"回退 {retry_used} 次重试用完，不交解释段（来自 {came_from or fallback_entry})")
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
            return result(STOP_MAX_STEPS, "FALLBACK 解释走了太多步还没停机")

        state = machine.states.get(cur)
        if state is None:
            return result(STOP_STATE_ERROR, f"跳到不存在的状态 {cur!r}")
        step += 1
        t_step = time.perf_counter()

        # ---- 终止动作 ---- #
        if state.action.kind == "end":
            records.append(Record(step=step, state=cur, clause=state.clause,
                                  action={"kind": "end",
                                          "terminal": state.action.terminal},
                                  vars=dict(values),
                                  meta=_meta((time.perf_counter() - t_step) * 1000.0,
                                             0, usage0, usage0)))
            return result(STOP_TERMINAL)

        # ---- 普通动作 ---- #
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

        # ---- 选边 ---- #
        edge, eerr = pick_edge(machine, cur, values)
        if eerr:
            return result(STOP_STATE_ERROR, eerr)
        if edge is None:
            return result(STOP_STUCK, f"{cur} 的出边一条都没成立，也没有兜底边")
        if edge.inc:
            values[edge.inc] = (values.get(edge.inc) or 0) + 1
        if edge.to == machine.fallback:
            came_from, came_edge = cur, edge
            if fallback_entry is None:
                fallback_entry = cur             # 位置清单要的就是「从哪一状态退的」
        cur = edge.to

    return result(STOP_MAX_STEPS, f"走了 {limit} 步还没停机")
