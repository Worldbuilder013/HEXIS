"""评判：给一条轨迹定 accepted / rejected。

评判有两条独立的轴，任一不过即拒：

* **客观验收**（``acceptance``）—— 结果对不对。技能自带的确定性闸，只看轨迹。
* **禁止性要求**（``prohibitions``）—— 有没有做不该做的事。**即使结果对，触犯即拒**，
  并标出违规发生的那一步作为 ``error_step``（拒绝集排除检查的锚）。

评判是确定性的、不调模型：它是「确定性守门程序」的一半，编译器拿它把轨迹分成接受集与
拒绝集。

禁止项有两类形状。``absent``/``present``/``regex`` 看的是**文本**（某个串出没出现过），
``forbid_action``/``require_before`` 看的是**事件流**（谁在谁之前、哪两个参数相等）。数学
技能的 P1「任何非平凡结果都要至少跑一次独立核验」属于后者：它不是「别说某句话」，而是
「提交之前必须先跑过核验」，只有把轨迹当事件序列扫一遍才判得出来。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .normalize import canon_action, canon_tool_name
from .schema import Prohibition, Trace


@dataclass
class Verdict:
    verdict: str                       # "accepted" / "rejected"
    error_step: Optional[int] = None
    reason: str = ""


def evaluate(trace: Trace, acceptance: Optional[Callable[[Trace], bool]],
             prohibitions: list[Prohibition]) -> Verdict:
    """定 verdict。禁止项优先（触犯即拒，哪怕验收通过），其次看客观验收。"""
    for p in prohibitions or []:
        step = _violation_step(p, trace)
        if step is not None:
            return Verdict("rejected", error_step=step,
                           reason=f"触犯禁止性要求 {p.id}")
    ok = acceptance(trace) if acceptance else True
    if ok:
        return Verdict("accepted")
    return Verdict("rejected", error_step=_last_step(trace), reason="未通过客观验收")


def judged(trace: Trace, acceptance, prohibitions) -> Trace:
    """返回补好 verdict/error_step 的**新** trace（原 trace 不动），方便组接受/拒绝集。

    运行级出身（arm/run/model/harness）原样带过去：评判不该把「这条轨迹是谁跑的」擦掉。
    """
    v = evaluate(trace, acceptance, prohibitions)
    return Trace(task=trace.task, arm=trace.arm, run=trace.run, model=trace.model,
                 harness=trace.harness, verdict=v.verdict, error_step=v.error_step,
                 records=trace.records)


# --------------------------------------------------------------------------- #
# 禁止项检查
# --------------------------------------------------------------------------- #
def _violation_step(p: Prohibition, trace: Trace) -> Optional[int]:
    """返回违规发生的 step，没违规返回 None。"""
    if p.check == "forbid_action":
        return _forbid_action(p.pattern, trace)
    if p.check == "require_before":
        return _require_before(p.pattern, trace)
    if p.check == "regex":
        pat = re.compile(str(p.pattern))
        for r in trace.records:
            if pat.search(_record_text(r)):
                return r.step
        return None
    if p.check == "absent":
        for r in trace.records:
            if str(p.pattern) in _record_text(r):
                return r.step
        return None
    if p.check == "present":
        if not any(str(p.pattern) in _record_text(r) for r in trace.records):
            return _last_step(trace)
        return None
    return None


def _forbid_action(pattern: Any, trace: Trace) -> Optional[int]:
    """结构化禁止项：某动作 + 变量关系成立即违规。

    ``pattern`` = ``{"name": 工具名, "equal": ["input.a", "input.b"]}``：该工具动作里
    两个路径的取值相等即违规（例如导出目标 == 源文件 → 覆盖原文件）。
    """
    if not isinstance(pattern, dict):
        return None
    name = pattern.get("name")
    equal = pattern.get("equal")
    for r in trace.records:
        if name and (r.action or {}).get("name") != name:
            continue
        if equal:
            vals = [_resolve(r.action, path) for path in equal]
            if all(v is not None for v in vals) and len({str(v) for v in vals}) == 1:
                return r.step
    return None


def _resolve(action: dict, path: str) -> Any:
    """按 ``input.output_path`` 这样的点路径从动作 dict 取值。"""
    cur: Any = action
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


# --------------------------------------------------------------------------- #
# require_before：事件流上的先后要求（数学技能的 P1）
# --------------------------------------------------------------------------- #
#: 终点 id 里**不携带类别**的那几个通用名。``done`` 只说「结束了」，没说以什么方式结束，
#: 拿它当类别会把 only_when 变成一个碰巧永远不匹配的过滤器。
_GENERIC_TERMINALS = frozenset({"done", "end", "stop", "ok"})


def _require_before(pattern: Any, trace: Trace) -> Optional[int]:
    """「``action`` 出现之前必须先出现过 ``requires`` 里的任一个」，否则违规。

    ``pattern``::

        {"action": "submit_answer",                  # 被守卫的动作
         "requires": ["math_verify", "run_python"],  # 任一个即可（按规范化工具名比）
         "only_when": {"terminal_kind": "verified"}, # 可选；省略 = 所有轨迹都管
         "clause": "RV.0.1", "quote": "<技能文档原句>"}

    扫描 ``trace.records``，**按顺序**、只看第一次：先撞上 ``requires`` 里的动作 ⇒ 这次运行
    确实先核验过，不违规；先撞上被守卫的动作 ⇒ 违规，``error_step`` 就是这条记录的 ``step``。
    两个都没出现（例如预算耗尽、根本没提交）⇒ 不违规——P1 管的是「提交时有没有核验过」，不是
    「必须提交」。

    动作比较一律走 :func:`~skill2fsm.normalize.canon_action`，工具名因此过
    :func:`~skill2fsm.normalize.canon_tool_name`：``scripts/math_verify.py``、``math-verify``、
    ``MATH_VERIFY`` 折成同一个名字。**这里不写第二套名字匹配规则**——写了就会和编译/回放两侧
    的折叠口径分家。``action``/``requires`` 的元素既可以是裸工具名（字符串），也可以是完整的
    动作 dict（必须带 ``kind``，例如 ``{"kind": "end", "terminal": "END_VERIFIED"}``）。

    ``only_when.terminal_kind`` 按 :func:`terminal_kind` 判：一次**标注了未验证**地结束的运行
    什么都没声称，P1 不该在它头上开火。类别取值可以是一个字符串或一组字符串。
    """
    if not isinstance(pattern, Mapping):
        return None
    guarded = pattern.get("action")
    if guarded is None or guarded == "":
        return None
    only = pattern.get("only_when")
    if isinstance(only, Mapping) and "terminal_kind" in only:
        if not _kind_matches(terminal_kind(trace), only["terminal_kind"]):
            return None
    want = _action_key(guarded)
    required = {_action_key(x) for x in _as_list(pattern.get("requires"))}
    for r in trace.records:
        key = canon_action(r)
        if key in required:
            return None                 # 核验先跑过了
        if key == want:
            return r.step               # 守卫动作先到：之前一次核验都没有
    return None


def _action_key(spec: Any) -> tuple[str, ...]:
    """把 pattern 里的一项折成动作 KEY。裸字符串按工具名理解（P1 的两侧都是工具）。"""
    if isinstance(spec, Mapping):
        return canon_action(spec)
    return canon_action({"kind": "tool", "name": str(spec)})


def _as_list(v: Any) -> list:
    """None → []；单个字符串 → 单元素；其余序列原样摊平一层。"""
    if v is None:
        return []
    if isinstance(v, str) or isinstance(v, Mapping):
        return [v]
    if isinstance(v, Sequence):
        return list(v)
    return [v]


def _fold_kind(value: Any) -> str:
    """类别名的规范形。复用工具名那套折叠（小写、``-``/空白 → ``_``），再削掉 ``end_`` 前缀，
    好让终点 id ``END_UNVERIFIED`` 与类别 ``unverified`` 是同一个东西。"""
    s = canon_tool_name(str(value or ""))
    return s[4:] if s.startswith("end_") else s


def _kind_matches(actual: str, want: Any) -> bool:
    """轨迹实际的终点类别是不是 ``want`` 之一。

    **判不出类别时（``actual`` 为空）算匹配**，即 only_when 过滤器对不表态的轨迹不生效、
    禁止项照查。这是有意的保守方向：P1 要抓的是「没核验就提交」，豁免必须由运行**显式**
    标注（机器的 ``END_UNVERIFIED`` 终点、或提交动作自报的 ``verified: false``）才给；
    反过来把判不出的轨迹一律放行，会让一条头部残缺的轨迹悄悄绕过检查，违规率被系统性低估。
    """
    wants = {_fold_kind(w) for w in _as_list(want)}
    wants.discard("")
    if not wants:
        return True
    return not actual or actual in wants


def terminal_kind(trace: Trace) -> str:
    """从轨迹判断这次运行**以哪种方式**结束，判不出返回空串。

    评判只拿得到轨迹、拿不到机器，所以 :class:`~skill2fsm.schema.Terminal` 上声明的 ``kind``
    必须由执行侧落进轨迹。按下面的优先级读（前面的赢）：

    1. 结束那条记录的 ``action``/``action.input``/``output``/``vars`` 里显式写着的
       ``terminal_kind``——最直接，执行器想说清楚就写这个键；
    2. 结束记录的终点 id（``{"kind": "end", "terminal": ...}``）折成类别：``END_UNVERIFIED``
       → ``unverified``。``done``/``end``/``stop``/``ok`` 这几个通用 id 不算类别，跳过；
    3. 提交动作自报的布尔 ``verified``（预算耗尽时被强制标成 ``False`` 的那个标记）；
    4. 都没有 ⇒ 空串（不表态，见 :func:`_kind_matches` 对空串的处理）。

    「结束那条记录」= 最后一条 ``kind == "end"`` 的记录；没有 end 记录（解释执行的轨迹常常
    以一次 ``submit_answer`` 收尾）就看最后一条记录。两者不是同一条时两条都看。
    """
    recs = list(trace.records or ())
    if not recs:
        return ""
    last = recs[-1]
    end = next((r for r in reversed(recs) if _rec_kind(r) == "end"), None)
    tails = [end, last] if (end is not None and end is not last) else [end or last]

    for r in tails:                                     # ① 显式标注
        for src in _rec_sources(r):
            if "terminal_kind" in src:
                return _fold_kind(src["terminal_kind"])
    if end is not None:                                 # ② 终点 id
        k = _fold_kind(_rec_action(end).get("terminal"))
        if k and k not in _GENERIC_TERMINALS:
            return k
    for r in tails:                                     # ③ 自报的 verified 标记
        for src in _rec_sources(r):
            v = src.get("verified")
            if isinstance(v, bool):
                return "verified" if v else "unverified"
    return ""


def _rec_action(r) -> Mapping:
    a = getattr(r, "action", None)
    return a if isinstance(a, Mapping) else {}


def _rec_kind(r) -> str:
    return str(_rec_action(r).get("kind") or "")


def _rec_sources(r) -> list[Mapping]:
    """一条记录里可能藏着结束标记的几个位置，按查找顺序。"""
    act = _rec_action(r)
    out = [act]
    for m in (act.get("input"), getattr(r, "output", None), getattr(r, "vars", None)):
        if isinstance(m, Mapping):
            out.append(m)
    return out


def _record_text(r) -> str:
    import json
    return json.dumps({"action": r.action, "output": r.output},
                      ensure_ascii=False, default=str)


def _last_step(trace: Trace) -> Optional[int]:
    return trace.records[-1].step if trace.records else None
