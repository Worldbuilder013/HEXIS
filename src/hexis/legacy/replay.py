"""回放与排除：确定性地核对一台机器与一条轨迹的关系。**不调模型。**

定理2 说：给定结果序列，机器的状态与变量序列是它的函数。所以回放不必真执行动作——用轨迹
里记录的每步 ``vars`` 就能驱动选边，逐条核对机器会不会走出同一条路。

* :func:`walk` —— **唯一的**沿轨迹推机器的实现。回放、回退位置、编译期定位「该退谁」
  原来是三份长得一样的循环（本模块、verify.fallback_entry、compile_agent._machine_walk），
  三份各自漂移过一次就够了；现在它们都是它的薄包装。
* :func:`replay` —— 机器能否复述这条轨迹。走到 ``FALLBACK`` 即无条件接受剩余（解释模式
  不承诺具体路径，这正是「全回退机器平凡复述任何轨迹」的由来）。
* :func:`reproduces` —— 接受轨迹（T+）该被复述。
* :func:`excludes` —— 拒绝轨迹（T-）该在其出错位置或更早被机器**偏离**。这是「正例只能
  证实不能证伪」的代码化：没有 T-，状态合并过头也能静默通过。

两样东西只在 :func:`walk` 里处理，别处不再各自发明：

* **开局工具** :data:`~hexis.traces.normalize.BEGIN_TOOL`：机器的起点是它、而轨迹第一条记录
  不是（臂一臂二的轨迹从不记它）时，虚拟地垫一条进去，索引照旧按**原轨迹**报——调用方拿
  ``diverged_at`` 去查 ``trace.records`` 不会错位。
* **从文档引入的判断动作**（``JudgeAction.introduced``）：轨迹里没有这一步，它是**零宽**
  的——不消费记录，标签由 ``gold_from`` 指名的程序打标器现算（:data:`trace_adapter.LABELERS`），
  算不出就弃权，然后照常选边。标定（``fit.calibrate``）与回放用同一个打标器，所以两边构造上
  一致。臂三跑出来的轨迹里若真有一条匹配的判断记录，就按普通判断步消费它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from hexis.traces.normalize import BEGIN_TOOL, canon_action, is_begin
from hexis.execution.runtime import pick_edge
from hexis.machine.schema import Machine, Record, Trace


@dataclass
class ReplayResult:
    ok: bool
    diverged_at: Optional[int] = None      # 轨迹里第几条记录（0基）开始偏离
    reason: str = ""


@dataclass
class WalkResult:
    """一次推演的全部可观察结果。索引一律按**原轨迹**的 ``records`` 计。

    ``seq`` 是 ``[(记录下标, 状态 id)]``：消费了记录的状态记它消费的那条；零宽判断记它
    **之前**的那条记录的下标；虚拟开局步记 ``-1``。``fallback_at`` 是进 FALLBACK 时的
    ``Record.step``（记录走完仍停在回退态则是最后一步 +1），没进过是 ``None``。
    """

    ok: bool
    seq: list = field(default_factory=list)
    diverged_at: Optional[int] = None
    reason: str = ""
    fallback_at: Optional[int] = None
    values: dict = field(default_factory=dict)
    ended: bool = False


def _is_introduced_judge(action: Any) -> bool:
    """零宽状态：引入的判断，或入参门引入的生成——回放里都不消费记录。"""
    return action.kind in ("judge", "model") and bool(getattr(action, "introduced", False))


def _begin_record(step: int, vars_: dict) -> Record:
    return Record(step=step, action={"kind": "tool", "name": BEGIN_TOOL, "input": {}},
                  output={}, vars=dict(vars_))


def walk(machine: Machine, trace: Trace, *,
         labelers: Optional[Mapping[str, Callable[[Trace, int], Optional[str]]]] = None
         ) -> WalkResult:
    """沿轨迹推机器。变量由机器自己演算（轨迹 output 按各状态 writes 白名单更新、回边
    自己 inc），机器自造的计数变量因此能正确参与选边——定理2。"""
    task = trace.task if isinstance(trace.task, dict) else {}
    inp = task.get("input", {}) if isinstance(task.get("input", {}), dict) else {}
    values: dict = dict(inp)
    values.update(machine.initial_values(inp))

    records = list(trace.records)
    offset = 0
    init = machine.states.get(machine.initial)
    if (init is not None and is_begin(init.action)
            and not (records and is_begin(records[0].action))):
        first_step = records[0].step - 1 if records else 0
        records = [_begin_record(first_step, values)] + records
        offset = 1                                  # 报索引时减掉它

    if labelers is None:
        from hexis.traces.trace_adapter import LABELERS as labelers   # 延迟：trace_adapter 依赖本模块之外的一圈

    def idx(i: int) -> int:
        return i - offset

    cur = machine.initial
    i = 0
    seq: list = []
    guard = 0
    limit = int(machine.max_steps or 24) + len(records) + 8
    while True:
        if cur == machine.fallback:
            if i < len(records):
                fb = records[i].step
            else:
                fb = (records[-1].step + 1) if records else 0
            return WalkResult(True, seq, None, "", fb, values, False)
        st = machine.states.get(cur)
        if st is None:
            return WalkResult(False, seq, idx(i), f"机器没有状态 {cur!r}", None, values)
        rec = records[i] if i < len(records) else None

        # ---- 零宽判断：轨迹里没有这一步，标签由程序打标器现算 ---- #
        if st.action.kind == "judge" and _is_introduced_judge(st.action) and (
                rec is None or not _action_matches(st.action, rec.action)):
            guard += 1
            if guard > limit:
                return WalkResult(False, seq, idx(i), f"零宽判断 {cur} 绕不出去（{limit} 次）",
                                  None, values)
            fn = labelers.get(st.action.gold_from) if st.action.gold_from else None
            label: Optional[str] = None
            if fn is not None:
                try:
                    # 打标器的约定是「``i`` 是这个判断**之前**那条记录的下标」。零宽判断
                    # 停在「即将消费 records[idx(i)]」的位置上，所以之前那条是 idx(i)-1。
                    # 传 idx(i) 会让每个判断都预测**再下一步**——实测 54 条 T+ 只复述出 13 条。
                    label = fn(trace, idx(i) - 1)
                except Exception:                                   # noqa: BLE001
                    label = None
            if label not in st.action.labels:
                label = st.action.abstain
            values[st.action.writes[0]] = label
            seq.append((idx(i) - 1, cur))
            edge, eerr = pick_edge(machine, cur, values)
            if eerr:
                return WalkResult(False, seq, idx(i), eerr, None, values)
            if edge is None:
                return WalkResult(False, seq, idx(i), f"{cur} 在这一步无边可走", None, values)
            if edge.inc:
                values[edge.inc] = (values.get(edge.inc) or 0) + 1
            cur = edge.to
            continue

        # ---- 零宽生成：入参门插的 model 状态，轨迹里没有这一步；它写出的变量就是紧接着
        #      那条工具记录的真实入参（按后继工具状态的 ${var} 模板对回去）---- #
        if st.action.kind == "model" and getattr(st.action, "introduced", False) and (
                rec is None or not _action_matches(st.action, rec.action)):
            guard += 1
            if guard > limit:
                return WalkResult(False, seq, idx(i), f"零宽生成 {cur} 绕不出去（{limit} 次）",
                                  None, values)
            dflt = next((t for t in st.transitions if not t.cond), None)
            nxt_state = machine.states.get(dflt.to) if dflt else None
            if (rec is not None and nxt_state is not None and nxt_state.action.kind == "tool"
                    and (rec.action or {}).get("kind") == "tool"):
                tmpl = getattr(nxt_state.action, "input", {}) or {}
                real = (rec.action or {}).get("input") or {}
                for k, v in tmpl.items():
                    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                        var = v[2:-1]
                        if var in st.action.writes and k in real:
                            values[var] = real[k]
            for w in st.action.writes:
                values.setdefault(w, None)
            seq.append((idx(i) - 1, cur))
            edge, eerr = pick_edge(machine, cur, values)
            if eerr:
                return WalkResult(False, seq, idx(i), eerr, None, values)
            if edge is None:
                return WalkResult(False, seq, idx(i), f"{cur} 在这一步无边可走", None, values)
            if edge.inc:
                values[edge.inc] = (values.get(edge.inc) or 0) + 1
            cur = edge.to
            continue

        if rec is None:
            return WalkResult(True, seq, None, "", None, values, False)   # 记录走完
        if not _action_matches(st.action, rec.action):
            return WalkResult(
                False, seq, idx(i),
                f"第{idx(i)}步动作不符：机器在 {cur} 要 {st.action.kind}"
                f"/{getattr(st.action, 'name', '')}，轨迹是 "
                f"{rec.action.get('kind')}/{rec.action.get('name', '')}", None, values)
        seq.append((idx(i), cur))
        if st.action.kind == "end":
            return WalkResult(True, seq, None, "", None, values, True)
        for w in getattr(st.action, "writes", []) or []:
            if w in rec.output:
                values[w] = rec.output[w]
        edge, eerr = pick_edge(machine, cur, values)
        if eerr:
            return WalkResult(False, seq, idx(i), eerr, None, values)
        if edge is None:
            return WalkResult(False, seq, idx(i), f"{cur} 在这一步无边可走", None, values)
        if edge.inc:
            values[edge.inc] = (values.get(edge.inc) or 0) + 1
        nxt = edge.to
        if i + 1 < len(records) and nxt != machine.fallback:
            ns = machine.states.get(nxt)
            if ns is None or (not _is_introduced_judge(ns.action)
                              and not _action_matches(ns.action, records[i + 1].action)):
                return WalkResult(
                    False, seq, idx(i + 1),
                    f"第{idx(i + 1)}步机器走到 {nxt}，其动作与轨迹下一步对不上", None, values)
        cur = nxt
        i += 1


def replay(machine: Machine, trace: Trace) -> ReplayResult:
    """逐条核对机器能否走出这条轨迹。不比状态名（轨迹可能来自另一台机器），比**动作**。"""
    r = walk(machine, trace)
    return ReplayResult(r.ok, r.diverged_at, r.reason)


def reproduces(machine: Machine, trace: Trace) -> bool:
    return replay(machine, trace).ok


def excludes(machine: Machine, neg: Trace, *,
             evaluate: Optional[Callable[..., Any]] = None) -> bool:
    """机器是否排除这条拒绝轨迹——在出错位置或更早**偏离**，或被机器的禁止项**拦下**。

    两类反例要分别对待：走错顺序/条件判错的反例，机器结构会在出错处偏离；禁止性违规（覆盖
    原文件那种）在结构上和正常执行一模一样，靠机器带的 prohibitions 拦。任一成立即算排除。

    ``evaluate`` 是**注入的评判**，签名同 :func:`hexis.traces.judge.evaluate`
    ``(trace, acceptance, prohibitions) -> Verdict``；不给就用它。做成注入是为了让客观验收
    换得掉：MATH 那条线的验收是 :func:`hexis.grader.acceptance_for`（答案与参考等价），
    直接组进来即可——

    .. code-block:: python

        from hexis import grader, judge
        acc = grader.acceptance_for(gold)               # 答案与参考等价即算通过
        excludes(machine, neg, evaluate=lambda t, _a, p: judge.evaluate(t, acc, p))

    而回放本身**照旧不 import grader**（sympy 是重依赖），也不在模块级 import judge：默认
    实现按需取，import 图上仍是单向的。
    """
    cutoff = neg.error_step if neg.error_step is not None else len(neg.records)

    r = replay(machine, neg)
    if not r.ok and r.diverged_at is not None:
        d = r.diverged_at
        if d < 0:
            diverge_step = neg.records[0].step - 1 if neg.records else 0
        elif d < len(neg.records):
            diverge_step = neg.records[d].step
        else:
            diverge_step = 10 ** 9
        if diverge_step <= cutoff:
            return True

    if machine.prohibitions:
        if evaluate is None:
            from hexis.traces.judge import evaluate as evaluate       # 默认实现：judge.evaluate
        v = evaluate(neg, None, machine.prohibitions)
        if v.verdict == "rejected" and (v.error_step is None or v.error_step <= cutoff):
            return True
    return False


def _action_matches(m_act, r_act: dict) -> bool:
    """机器状态的动作与轨迹记录的动作是否「同一步」——**回放档**，折叠交给
    :func:`hexis.traces.normalize.canon_action`（``strict=False``）。

    比较是**跨源**的：一边是机器里的 Action 模型（input 是 ``${var}`` 模板、prompt 属私有
    面），一边是轨迹里的**裸 action dict**（都是渲染过的具体值）。松档正是为这种比较定的：
    只按 kind + 工具名 / 写入变量分组，参数与 prompt 一律不进 KEY。

    松档的 KEY 有两个分量在这里**刻意不参与**比较——它们不是「同一步」的判据，而是轨迹侧
    给不出、或跨源不可比的东西：

    * ``writes``：轨迹记录的 judge/model/user 动作**没有** writes 字段，只能从 ``output``
      的键反推，而回放拿到的是 ``rec.action``（裸 dict，没有 output），一侧必然为空。
      normalize 的模块文档把这条退化写在明处；这里照它办：一侧未知就不比这一项。
    * ``terminal``：终止的命名是**产出这条轨迹的那台机器**的私事，用它判偏离会把「换个名字
      收尾」误判成走错路。test_14 ``test_end_terminal_participates_in_both_modes`` 把这条差异
      钉成了断言（松档比本函数细），所以显式留着，而不是让它随委托悄悄消失。

    两处放宽之外，折叠规则（工具名归一、kind 不相并）一律走 normalize，不再自己判。
    """
    a = canon_action(m_act, strict=False)
    b = canon_action(r_act, strict=False)
    if len(a) != len(b) or a[0] != b[0]:
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        if x.startswith("writes=") and y.startswith("writes=") and "writes=" in (x, y):
            continue                    # 一侧反推不出 writes（裸 action dict 没有 output）
        if x.startswith("terminal=") and y.startswith("terminal="):
            continue                    # 终止命名是各机器私事，跨源不比
        return False
    return True


__all__ = ["ReplayResult", "WalkResult", "excludes", "replay", "reproduces", "walk"]
