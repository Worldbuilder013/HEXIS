"""执行级一致性：**机器真跑出来的东西必须等同于轨迹**。

:mod:`skill2fsm.replay` 做的是推演——沿轨迹推机器，逐步比动作身份。它证明不了「把这台机器
交给 :func:`skill2fsm.runtime.run_task` 真跑一遍，走出来的还是这串动作」。两者之间隔着一整层
运行时语义：``${var}`` 模板要有人填、``writes`` 白名单要收得住产出、``phase`` 要两侧都记、
条件要在真实变量上求值。实测过的三个洞全长在这一层，而回放全都看不见：

* 机器状态带 ``phase``、运行时记录不带 ⇒ 身份两侧不对称；
* model 状态声明 ``writes=[s2_cmd]``，但没人告诉模型要回这个键 ⇒ 变量空 ⇒ 工具拿到空入参；
* 工具入参模板引用的变量根本没有状态写它 ⇒ 渲染成空串。

所以这里把「等同」做成可执行的：用**轨迹**当模型与工具的替身去驱动真正的 ``run_task``，
跑完之后把产出的记录序列与原轨迹逐步比对。替身只做一件事——把轨迹里那一步实际发生的东西
交给机器：

* 机器要执行 ``tool`` ⇒ 取轨迹里下一条工具记录的 ``output``，并核对工具名与阶段；
* 机器要执行 ``model`` ⇒ 轨迹里下一条就是模型步就用它的产出；若下一条是工具步（说明这是
  **零宽生成**：入参门插的状态，轨迹里本没有这一步），就按后继工具状态的 ``${var}`` 模板从
  那条工具记录的**真实入参**反推出变量值——与 :func:`skill2fsm.replay.walk` 同一条规则；
* 机器要执行 ``judge`` ⇒ 轨迹里有判断步就用它的标签，否则用程序打标器（``gold_from``），
  再不行弃权。

判据（三条全要满足才算等同）：

1. **步数与顺序**：产出记录去掉零宽状态后，与轨迹逐条对齐，长度相同；
2. **动作身份**：每一步的 kind、规范工具名、阶段都相同；
3. **入参逐字相同**：工具步渲染出来的 ``input`` 与轨迹记录的 ``input`` 相等。第 3 条是
   「入参烤死」的检测器：把别的任务的字面量烤进模板的机器，回放能过，这里过不了。

中途退回解释执行也算不等同——机器没能自己走完。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from . import runtime as _runtime
from .normalize import canon_tool_name, is_begin
from .schema import Machine, Trace


@dataclass
class Divergence:
    """一处偏离。``step`` 是轨迹的步号（对不上时是产出侧的步号）。"""

    step: int
    why: str
    machine: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

    def located(self) -> str:
        return f"第 {self.step} 步：{self.why}"


@dataclass
class ConformanceResult:
    ok: bool
    trace_id: str = ""
    matched: int = 0
    total: int = 0
    divergences: list = field(default_factory=list)
    stopped: str = ""
    error: str = ""
    llm_calls: int = 0
    #: 机器在这一步停在回退态（把剩下的交回解释执行）。``None`` = 自己走完了。
    deferred_at: Optional[int] = None

    def summary(self) -> str:
        head = f"{self.trace_id or '(无题号)'}：对齐 {self.matched}/{self.total} 步，停机 {self.stopped}"
        if self.ok:
            return head + ("，等同" if self.deferred_at is None
                           else f"，前 {self.matched} 步等同，第 {self.deferred_at} 步起交回退段")
        first = self.divergences[0].located() if self.divergences else self.error
        return head + f"，**不等同**——{first}"


def _is_zero_width(machine: Machine, sid: str) -> bool:
    """零宽状态：轨迹里本来就没有这一步，比对时投影掉。三类：

    * 编译器**引入**的判断与生成（``introduced``）——见 :func:`skill2fsm.replay.walk` 的同名规则；
    * 保留的**开局工具**（:data:`skill2fsm.normalize.BEGIN_TOOL`）——它是编译器为「一台机器只有
      一个起点」垫的空操作，回放同样把它当虚拟步。
    """
    st = machine.states.get(sid)
    if st is None:
        return False
    a = st.action
    if is_begin(a):
        return True
    return bool(getattr(a, "introduced", False)) and a.kind in ("judge", "model")


def _tid(trace: Trace) -> str:
    task = trace.task if isinstance(trace.task, dict) else {}
    return str(task.get("task_id") or "")


class TraceDriver:
    """既当 Model 又当 Tool：机器每要一步，就把**轨迹里那一步**交给它。

    它不发明任何东西：模型该产的值取自轨迹的模型记录或紧随其后那条工具记录的真实入参，
    工具结果取自轨迹的工具记录。取不到就如实记一处偏离并交空产出——让机器当场露馅，
    而不是替它圆场。
    """

    def __init__(self, machine: Machine, trace: Trace, labelers: Optional[dict] = None) -> None:
        self.machine = machine
        self.records = list(trace.records)
        self.trace = trace
        self.i = 0
        self.divergences: list[Divergence] = []
        self.by_prompt = {st.action.prompt: st for st in machine.states.values()
                          if st.action.kind == "model"}
        self.by_question = {st.action.prompt: st for st in machine.states.values()
                            if st.action.kind == "judge"}
        # 变量被谁消费：工具入参模板 var -> [(状态, 入参键)]；以及出边条件里出现的变量。
        # **只有被消费的变量才影响这趟运行**，因此也只有它们才是「轨迹必须供得出」的东西；
        # 没人读的产出（一句结论）供不出也不算偏离——它左右不了机器走哪条路。
        self.consumed: dict = {}
        for st in machine.states.values():
            if st.action.kind != "tool":
                continue
            for k, v in (getattr(st.action, "input", {}) or {}).items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    self.consumed.setdefault(v[2:-1], []).append((st, k))
        self.cond_vars: set = set()
        for _sid, t in machine.transitions_all():
            if t.cond:
                try:
                    from .cond import vars_of
                    self.cond_vars |= set(vars_of(t.cond))
                except Exception:                                   # noqa: BLE001
                    pass
        if labelers is None:
            from .trace_adapter import LABELERS as labelers          # 延迟 import，避免环
        self.labelers = dict(labelers or {})

    # ---- 轨迹游标 ---- #
    def _peek(self):
        """下一条**该被机器消费的**记录。开局步跳过：机器那侧的开局状态是空操作，
        根本不会调工具，游标停在它上面就会让真正的第一步比错对象（实测：每条轨迹都报
        「工具名不符：机器 read_csv，轨迹 skill2fsm_begin」）。下标仍按原轨迹计，
        打标器要的就是原下标。"""
        while self.i < len(self.records) and is_begin(self.records[self.i].action):
            self.i += 1
        return self.records[self.i] if self.i < len(self.records) else None

    def _next_tool_record(self):
        for rec in self.records[self.i:]:
            if (rec.action or {}).get("kind") == "tool":
                return rec
        return None

    def _note(self, why: str, machine: Any = None, trace: Any = None) -> None:
        step = self.records[self.i].step if self.i < len(self.records) else len(self.records) + 1
        self.divergences.append(Divergence(step, why, dict(machine or {}), dict(trace or {})))

    # ---- Model 协议 ---- #
    def generate(self, *, prompt: str, values: dict, history: tuple = ()) -> dict:
        st = self.by_prompt.get(prompt)
        rec = self._peek()
        out: dict = {}
        if rec is not None and (rec.action or {}).get("kind") == "model":
            self.i += 1                                   # 轨迹里确实有这一步
            out.update(rec.output or {})
        writes = list(getattr(st.action, "writes", []) if st else [])
        # 只补**被消费**的变量：工具入参模板要它、或某条边的条件读它。别的产出左右不了走向。
        need = [w for w in writes if w not in out and (w in self.consumed or w in self.cond_vars)]
        out.update(self._supply(need))
        still = [w for w in need if w not in out]
        if still:
            self._note(f"model 状态 {st.id if st else '?'} 要写 {still}，"
                       f"而轨迹供不出这些值——机器真跑时这几个变量会是空的",
                       {"writes": writes, "unsatisfied": still},
                       dict((rec.action or {}) if rec else {}))
        return out

    def _supply(self, want: Sequence[str]) -> dict:
        """从轨迹里把这些变量的值取出来。规则只有一条，与回放同源：

        变量 ``v`` 被某个工具状态的入参键 ``k`` 消费 ⇒ 它的值就是**接下来那条同名同阶段的
        工具记录**里 ``input[k]`` 的实际取值。取不到就是取不到，不编。
        """
        out: dict = {}
        for var in want:
            for st, key in self.consumed.get(var, []):
                rec = self._find_tool(st)
                if rec is None:
                    continue
                real = dict((rec.action or {}).get("input") or {})
                if key in real:
                    out[var] = real[key]
                    break
        return out

    def _find_tool(self, st: Any):
        """游标之后第一条与状态 ``st`` 同名同阶段的工具记录。"""
        name = canon_tool_name(str(getattr(st.action, "name", "") or ""))
        phase = str(getattr(st.action, "phase", "") or "")
        for rec in self.records[self.i:]:
            act = rec.action or {}
            if act.get("kind") != "tool":
                continue
            if canon_tool_name(str(act.get("name") or "")) != name:
                continue
            if phase and str(act.get("phase") or "") != phase:
                continue
            return rec
        return None

    def classify(self, *, prompt: str, values: dict, labels: list, examples: tuple = ()) -> str:
        st = self.by_question.get(prompt)
        abstain = getattr(st.action, "abstain", "弃权") if st else (labels[-1] if labels else "")
        introduced = bool(getattr(st.action, "introduced", False)) if st else False
        rec = self._peek()
        # **引入的判断按定义零宽**：轨迹里本来没有这一步，它的标签由程序打标器现算。不先判这
        # 一条的话，轨迹里恰好有一个真 judge 步时会被它吃掉——真 judge 的裁决（"action"）不在
        # 分岔标签集里，判断只好弃权、走 FALLBACK（实测：前缀树上三条轨迹因此断在判断步）。
        if introduced:
            gold = getattr(st.action, "gold_from", "")
            fn = self.labelers.get(gold) if gold else None
            if fn is not None:
                try:
                    label = fn(self.trace, self.i - 1)
                except Exception:                                   # noqa: BLE001
                    label = None
                if label in labels:
                    return label
            return abstain
        if rec is not None and (rec.action or {}).get("kind") == "judge":
            self.i += 1
            for v in (rec.output or {}).values():
                if isinstance(v, str) and v in labels:
                    return v
            return abstain
        gold = getattr(st.action, "gold_from", "") if st else ""
        fn = self.labelers.get(gold) if gold else None
        if fn is not None:
            try:
                label = fn(self.trace, self.i - 1)
            except Exception:                                       # noqa: BLE001
                label = None
            if label in labels:
                return label
        return abstain

    # ---- Tool 协议 ---- #
    def call(self, name: str, inp: dict) -> dict:
        rec = self._peek()
        if rec is None:
            self._note(f"机器还要调 {name}，轨迹已经走完", {"name": name, "input": dict(inp or {})})
            return {"error": "轨迹已经走完"}
        act = rec.action or {}
        if act.get("kind") != "tool":
            self._note(f"机器要调 {name}，轨迹这一步是 {act.get('kind')}",
                       {"name": name, "input": dict(inp or {})}, dict(act))
            return {"error": "轨迹这一步不是工具调用"}
        if canon_tool_name(name) != canon_tool_name(str(act.get("name") or "")):
            self._note(f"工具名不符：机器 {name}，轨迹 {act.get('name')}",
                       {"name": name}, {"name": act.get("name")})
        self.i += 1
        return dict(rec.output or {})


def check_trace(machine: Machine, trace: Trace, *, max_steps: Optional[int] = None,
                compare_input: bool = True) -> ConformanceResult:
    """跑一遍机器（模型与工具都由轨迹驱动），再逐步比对。判据见模块文档。"""
    driver = TraceDriver(machine, trace)
    task = trace.task if isinstance(trace.task, dict) else {}
    # 走到回退态就停机，不切 runtime 自带的解释循环：这里要看的是**机器自己**走到哪为止。
    halted = _runtime.halt_at_fallback(machine)
    try:
        rr = _runtime.run_task(halted, task, model=driver, tools=driver, doc="",
                               max_steps=max_steps or (len(trace.records) * 3 + 16))
    except Exception as exc:                                        # noqa: BLE001
        return ConformanceResult(False, _tid(trace), 0, len(trace.records),
                                 [Divergence(0, f"跑不起来：{type(exc).__name__}: {exc}")],
                                 "exception", str(exc))
    # 开局步两侧都要去掉：它是编译器垫的空操作，编译视图（``with_begin``）的轨迹里有，
    # 别处来的轨迹里没有。只投影一侧就会整体错位一格（实测：每条轨迹都报「第 0 步动作类型
    # 不符」，产物因此一台都提交不出去）。
    got = [r for r in rr.trace.records
           if not _is_zero_width(machine, r.state) and not is_begin(r.action)]
    want = [r for r in trace.records if not is_begin(r.action)]
    divs = list(driver.divergences)
    # 停在回退态**不是**偏离：那是机器如实说「这一段我还没编译到」，与
    # :func:`skill2fsm.replay.walk` 同一条判定（走到 FALLBACK 即返回接受）。要求的是：
    # 交出去之前的那一段必须逐步等同。
    deferred_at: Optional[int] = None
    if got and got[-1].state == machine.fallback:
        deferred_at = got[-1].step
        got = got[:-1]
    matched = 0
    for k, w in enumerate(want):
        if k >= len(got):
            if deferred_at is None:
                divs.append(Divergence(w.step, f"机器只走了 {len(got)} 步，轨迹有 {len(want)} 步",
                                       {}, dict(w.action or {})))
            break                                   # 停在回退态：后面这段本就没承诺
        g, wa, ga = got[k], dict(w.action or {}), dict(got[k].action or {})
        if wa.get("kind") != ga.get("kind"):
            divs.append(Divergence(w.step, f"动作类型不符：机器 {ga.get('kind')}，轨迹 {wa.get('kind')}",
                                   ga, wa))
            break
        if wa.get("kind") == "tool":
            if canon_tool_name(str(ga.get("name") or "")) != canon_tool_name(str(wa.get("name") or "")):
                divs.append(Divergence(w.step, f"工具名不符：机器 {ga.get('name')}，轨迹 {wa.get('name')}",
                                       ga, wa))
                break
            if str(ga.get("phase") or "") != str(wa.get("phase") or ""):
                divs.append(Divergence(w.step, f"阶段不符：机器 {ga.get('phase') or '（无）'}，"
                                               f"轨迹 {wa.get('phase') or '（无）'}", ga, wa))
                break
            if compare_input and dict(ga.get("input") or {}) != dict(wa.get("input") or {}):
                divs.append(Divergence(w.step, "工具入参不同——机器真跑时执行的不是轨迹里那一条",
                                       dict(ga.get("input") or {}), dict(wa.get("input") or {})))
                break
        matched += 1
    if len(got) > len(want) and matched == len(want):
        divs.append(Divergence(want[-1].step if want else 0,
                               f"机器多走了 {len(got) - len(want)} 步（轨迹到此为止）"))
    ok = not divs and (matched == len(want) or deferred_at is not None)
    if rr.stopped != _runtime.STOP_TERMINAL and not divs:
        divs.append(Divergence(len(want), f"机器没有正常停机：{rr.stopped} {rr.error[:120]}"))
        ok = False
    return ConformanceResult(ok, _tid(trace), matched, len(want), divs, rr.stopped,
                             rr.error, rr.llm_calls, deferred_at)


def check(machine: Machine, traces: Sequence[Trace], **kw) -> tuple[bool, list]:
    """对一批轨迹逐条检查。返回 ``(全部等同, [ConformanceResult])``。"""
    out = [check_trace(machine, t, **kw) for t in traces]
    return all(r.ok for r in out), out


__all__ = ["ConformanceResult", "Divergence", "TraceDriver", "check", "check_trace"]
