"""EFSM 产物与轨迹的数据定义。

一份技能被理想化为函数 ``f: E* → A∪Z``：读一段历史（已发生的动作与结果），给出下一个
动作，或以某种方式结束。把这个函数落成「人能读、程序能跑」的东西，就是一台**扩展有限
状态机**：有限的控制（状态 ``q``）承载「走到哪一步」，带类型的变量（``ν``）承载数据
（修复次数 0..∞ 这种，普通自动机装不下，所以是 *extended*）。

这份文件只放数据模型，不放执行、不放模型调用。字段的形状对齐两份规格文档的
machine.json / 轨迹 JSONL——`clause`（条款归属）、`action`（这一步谁来做）、状态内嵌
`transitions`（带条件的出边）、`variables`（带 init_from）、`FALLBACK` 回退态、判断动作
的 `error_rate`/`support`。所有这些字段都是**可审计**的：一台机器整份 dump 出来，人能
逐条核对它凭哪条轨迹、哪条条款学出了每一步。

条件表达式（``Transition.if``）的语法与求值在 :mod:`skill2fsm.cond`，此处只存字符串。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

MACHINE_FILE = "machine.json"

#: 回退状态的保留标识。进入它 = 放弃已编译路径，改回「模型读整份文档解释执行」，同时
#: 照常记录轨迹。条件覆盖不到、判断弃权、校验反复失败都走这里。见 runtime.run_task。
FALLBACK = "FALLBACK"

VarType = Literal["string", "integer", "number", "boolean", "array", "object"]


# --------------------------------------------------------------------------- #
# 变量
# --------------------------------------------------------------------------- #
class Variable(BaseModel):
    """一个带类型的变量。初值来自 ``init``（字面量）或 ``init_from``（任务输入的某字段）。

    ``init_from`` 形如 ``"task.input.path"``：机器启动时从任务输入里取该字段填进来。
    计数类变量（修复次数）用 ``init: 0`` + 回边上的 ``inc`` 递增。
    """

    name: str
    type: VarType = "string"
    init: Optional[Any] = None
    init_from: Optional[str] = None

    @model_validator(mode="after")
    def _init_xor(self) -> "Variable":
        if self.init is not None and self.init_from is not None:
            raise ValueError(f"变量 {self.name} 的 init 与 init_from 只能给一个")
        return self


# --------------------------------------------------------------------------- #
# 动作：一个状态「这一步由谁执行」
# --------------------------------------------------------------------------- #
class ToolAction(BaseModel):
    """调一次工具。``input`` 是参数模板，值里可含 ``${var}`` 占位、运行时用变量填。"""

    kind: Literal["tool"] = "tool"
    name: str
    input: dict = Field(default_factory=dict)
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    #: **阶段**（probe / apply / verify / other）。只给 ``bash``/``run_python`` 这种
    #: **通用**工具用：它们名字一样、用途不同，不细化就会折成同一个状态。由
    #: :mod:`skill2fsm.phases` 的纯函数在**采集时**从命令正文判出（编译时正文已被抽到
    #: artifacts，读不到了）。非空时参与 ``canon_action`` 的 KEY，两侧对称，回放照旧。
    #: 专用工具（名字即用途）留空，行为与从前完全一致。
    phase: str = ""
    #: 派生标签：技能规则在轨迹事件上打出的标签（如「修改之后读产出」），随状态保存，
    #: 静态检查按它匹配规则模式。``phase`` 是基础标签，这里是其余的。
    labels: list[str] = Field(default_factory=list)
    #: **数据绑定**：工具产出键 → 语义变量名，如 ``{"stdout": "workbook_content"}``。
    #: 运行时先按它把产出改名，再按 ``writes`` 白名单收进变量表。没有它，文档说的
    #: 「把工作簿读进 workbook_content」与 bash 实际吐出的 ``stdout`` 永远接不上——
    #: ``rebuild`` 只按名字取值，写着 ``writes=["workbook_content"]`` 的状态从
    #: ``{ok, stdout, returncode}`` 里一个字都拿不到（实测八道题四轮全断在这）。
    #: 绑定由对齐阶段从轨迹里定，是它的一等产物，不是事后补丁。
    binds: dict[str, str] = Field(default_factory=dict)


class ModelAction(BaseModel):
    """生成内容的一步：一段**私有** prompt、一次无工具的模型调用，出参按 writes 白名单收。

    prompt 活在私有面（``exec/prompts/`` 或编译台账），永不进会话序列。
    """

    kind: Literal["model"] = "model"
    prompt: str
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    #: 由编译器**引入**的生成状态（入参门：某个工具入参随题变化、又没有变量可代，就在它前面
    #: 插一步生成）。轨迹里没有这一步，回放时它是**零宽**的：不消费记录，写出的变量直接取
    #: 紧接着那条工具记录的真实入参（见 replay.walk）。运行时它是一次真实的模型调用。
    introduced: bool = False
    #: **可观察**的模型状态：它写出的是交付内容（总结、答案、报告），轨迹里对应一条模型产出
    #: 事件，对齐与回放把它当作主要状态。缺省 False = 中间生成（生成入参、拟方案），零宽。
    observable: bool = False
    #: 派生标签（与 ToolAction.phase 同一口径），技能规则用它匹配模型状态。
    labels: list[str] = Field(default_factory=list)


class Example(BaseModel):
    """判断动作的一条标定样例：``reads`` 各键的取值（extra 承载）+ 它的真实 ``label``。

    样例**来自轨迹**——分岔处两侧轨迹的变量快照，不是人编的。
    """

    model_config = ConfigDict(extra="allow")
    label: str


class JudgeAction(BaseModel):
    """判断动作：编不成确定条件的语义判断落在这里，例如这个表头规不规范。

    一次固定提问、答案锁定在 ``labels`` 里（**必含弃权**），写进 ``writes`` 声明的变量，
    之后的跳转条件只认这个变量。误差率 ``error_rate`` 可标定（见 compiler.calibrate）：
    在分岔处的变量快照上离线跑这次判断、与实际走向对照。弃权是「一条路径至少错一次概率
    ≤ Σεᵢ」这条不等式的调节阀——拿不准就弃权走 FALLBACK，不硬答。
    """

    kind: Literal["judge"] = "judge"
    #: 发给模型的提示。与 :class:`ModelAction` 的同名字段一致：两者都是「这一步发给模型的话」。
    #: 旧文件里叫 ``question``，读入时仍然认。
    prompt: str = Field(validation_alias=AliasChoices("prompt", "question"))
    reads: list[str]
    writes: list[str]
    labels: list[str]
    abstain: str = "弃权"
    examples: list[Example] = Field(default_factory=list)
    error_rate: float = 0.0
    support: int = 0
    #: 由编译器**从文档引入**（轨迹里本来没有这一判断步）。回放时它是零宽的：不消费轨迹
    #: 记录，标签由 ``gold_from`` 指名的程序打标器现算（见 replay.walk）。
    introduced: bool = False
    #: 登记在 trace_adapter.LABELERS 里的**程序**打标器名。空串 = 没有程序能给它金标，
    #: 那它就标定不了、``support`` 恒 0，只能带一条去 FALLBACK 的默认边。
    gold_from: str = ""

    @model_validator(mode="after")
    def _abstain_in_labels(self) -> "JudgeAction":
        if self.abstain not in self.labels:
            raise ValueError(f"判断动作的弃权标签 {self.abstain!r} 必须在 labels 里")
        if not self.reads or not self.writes:
            raise ValueError("判断动作的 reads/writes 都不能为空")
        return self


class UserAction(BaseModel):
    """问一次用户。出参按 writes 白名单收。"""

    kind: Literal["user"] = "user"
    prompt: str = ""
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class EndAction(BaseModel):
    """终止动作：到达它这台机器停机，``terminal`` 指向 machine.terminals 里的一项。"""

    kind: Literal["end"] = "end"
    terminal: str


Action = Annotated[
    Union[ToolAction, ModelAction, JudgeAction, UserAction, EndAction],
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- #
# 状态、转移、机器
# --------------------------------------------------------------------------- #
class Transition(BaseModel):
    """一条带条件的出边。``if`` 为空 = 兜底边（状态内**最后**求值）。

    JSON 键是 ``if``（Python 关键字，字段名叫 ``cond`` 并用别名对上）；``to`` 是目标状态。
    ``inc`` 非空表示走这条边时对该计数变量 +1（修复成环靠它）。``support`` 记这条边被
    多少条轨迹走过。
    """

    model_config = ConfigDict(populate_by_name=True)

    cond: str = Field(default="", alias="if")
    to: str
    inc: Optional[str] = None
    support: int = 0
    #: 这条边凭什么存在：document / trace / compiler / harness(待标定) …（空串 = 未记）。
    #: 完整的溯源行在 checker 的 provenance 表里，这里只留一个可随机器 dump 的短标。
    origin: str = ""


class State(BaseModel):
    """一个状态：装**一个**动作 + 若干带条件的出边。``clause`` 是它归属的文档条款号。

    状态 = 历史的等价类（Myhill-Nerode）。``id`` 由编译器分配（``s1``/``s2``），不携带
    语义——语义在 ``clause`` 溯源到的文档原句里。
    """

    id: str
    clause: str = ""
    action: Action
    transitions: list[Transition] = Field(default_factory=list)
    #: 这个状态凭什么存在（同 Transition.origin）；``locator`` 是条款原句的位置
    #: （``SKILL.md:14``），从条款表抄来，永远不是自由文本。
    origin: str = ""
    locator: str = ""

    def ordered_transitions(self) -> list[Transition]:
        """带条件的在前、兜底边（``cond`` 为空）永远在最后。顺序即优先级。"""
        guarded = [t for t in self.transitions if t.cond]
        fallback = [t for t in self.transitions if not t.cond]
        return guarded + fallback


class Terminal(BaseModel):
    """一种结束方式。``output`` 是到达时按接口声明过滤出的键。

    ``kind`` 是这个终点的**类别**，默认空串（不表态）。``id`` 只是个标识符（``done``、
    ``END_UNVERIFIED``），类别才是可以被评判程序读的语义：数学机器要区分「核验通过后提交」
    （``kind="verified"``）与「预算耗尽、标注未验证地提交」（``kind="unverified"``）——两者
    都是「结束了」，但只有前者**声称**结果经过核验，P1 那类「提交前必须先跑核验」的禁止项
    因此只该管前者（见 judge._require_before 的 ``only_when``）。
    """

    id: str
    kind: str = ""
    output: list[str] = Field(default_factory=list)


class Prohibition(BaseModel):
    """一条禁止性要求（人工标出）。评判时在轨迹上检查，触犯即判拒——即使结果对。

    ``check`` / ``pattern`` 的五种形态：

    * ``absent`` —— ``pattern``（字符串）不得出现在任何动作的 input/output 文本里。
    * ``present`` —— 必须出现。
    * ``regex`` —— 正则匹配任一动作文本即违规。
    * ``forbid_action`` —— ``pattern`` 是 dict，匹配「某个动作 + 变量关系」即违规，例如
      ``{"name":"export","equal":["input.output_path","input.source_path"]}`` 表示
      「导出目标等于源文件（覆盖原文件）」。这对应规格里 P1 那种结构化禁止项。
    * ``require_before`` —— ``pattern`` 是 dict，**事件流**上的先后要求：某个动作出现之前，
      必须先出现过 ``requires`` 里的任一个动作，否则违规。数学技能的 P1（「任何非平凡结果
      都要至少跑一次独立核验」）就是这个形状：

      .. code-block:: python

          {"action": "submit_answer",
           "requires": ["math_verify", "run_python"],
           "only_when": {"terminal_kind": "verified"},
           "clause": "RV.0.1", "quote": "<技能文档原句>"}

      语义与终点类别的关系见 :func:`skill2fsm.judge._require_before`。``clause``/``quote``
      是溯源用的额外键，检查本身不读它们（pattern 是 ``Any``，多余的键一律不管）。
    """

    id: str
    check: Literal["absent", "present", "regex", "forbid_action", "require_before"]
    pattern: Any


class Thresholds(BaseModel):
    """编译期的一组阈值。默认值见字段。

    ``judge_err_max`` 是**判断动作标定误差率的上限**：标定（compiler.calibrate）算出的
    ``JudgeAction.error_rate`` 超过它，这个判断就不该留在机器里（改写提问、或退回
    FALLBACK）。它是「一条路径至少错一次的概率 ≤ Σεᵢ」那条不等式里每个 εᵢ 的天花板。
    """

    min_support: int = 2
    holdout_ratio: float = 0.2
    acc_thr: float = 0.9
    retry_budget: int = 3
    loop_margin: float = 1.5
    fallback_rate_target: float = 0.15
    judge_rewrite_max: int = 2
    judge_err_max: float = 0.2


class Machine(BaseModel):
    """一台编译出来的扩展有限状态机。**整份是私有面**，不进任何模型上下文。

    ``format`` 自描述，装载时按它与旧 ``toolize`` 的同名 machine.json 分派。``fallback``
    指向那个保留的回退状态。``initial`` 是起点。
    """

    format: Literal["efsm-v1"] = "efsm-v1"
    skill_id: str
    version: str = "0.1.0"
    initial: str
    fallback: str = FALLBACK
    max_steps: int = 24
    states: dict[str, State] = Field(default_factory=dict)
    variables: list[Variable] = Field(default_factory=list)
    terminals: list[Terminal] = Field(default_factory=list)
    prohibitions: list[Prohibition] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    #: **程序执行的审计工具**的规范名（``math_verify`` / ``audit_workbook``）。任何
    #: ``kind="verified"`` 的终点都必须经过其中之一才可达（checker 的 E_VERIFIED_UNAUDITED），
    #: 运行时这些名字必须在工具表里有真实执行器（runtime 的 E_AUDIT_TOOL_UNREGISTERED）。
    #: 来源是技能配置 / P1 的 ``requires``，永远不来自模型。空列表 = 这台机器不声称任何
    #: 「已验证」的结束方式。
    audit_tools: list[str] = Field(default_factory=list)
    #: 这台机器的 tool 状态用哪套**阶段分类规则**判身份（:data:`skill2fsm.phases.CLASSIFIERS`
    #: 的键）。自描述：换了规则就是另一台机器，不会出现「拿 A 规则编、用 B 规则跑」的静默错位。
    #: 空串 = 不做阶段细化（专用工具的技能不需要）。
    phase_rules: str = ""

    # ---- 便捷访问 ---- #
    def state(self, sid: str) -> Optional[State]:
        return self.states.get(sid)

    def out_edges(self, sid: str) -> list[Transition]:
        st = self.states.get(sid)
        return st.ordered_transitions() if st else []

    def transitions_all(self) -> list[tuple[str, Transition]]:
        """全部边，作 ``(源状态 id, 边)`` 对。供图算法遍历。"""
        return [(sid, t) for sid, s in self.states.items() for t in s.transitions]

    def var(self, name: str) -> Optional[Variable]:
        return next((v for v in self.variables if v.name == name), None)

    def terminal_ids(self) -> set[str]:
        return {t.id for t in self.terminals}

    def is_counter(self, name: str) -> bool:
        v = self.var(name)
        return bool(v and v.type == "integer")

    def n_states(self) -> int:
        """复杂度用的状态数：不含终止（end）状态。"""
        return sum(1 for s in self.states.values() if s.action.kind != "end")

    def initial_values(self, task_input: dict) -> dict:
        """按变量表算出机器启动时的工作状态。``init_from`` 从 ``task_input`` 取。"""
        vals: dict = {}
        for v in self.variables:
            if v.init_from:
                # 形如 "task.input.path"：取 input 之后的路径
                key = v.init_from.split(".")[-1]
                if key in task_input:
                    vals[v.name] = task_input[key]
            elif v.init is not None:
                vals[v.name] = v.init
        return vals


# --------------------------------------------------------------------------- #
# 轨迹：一次执行的完整记录
# --------------------------------------------------------------------------- #
class Record(BaseModel):
    """轨迹里的一步。``action`` 是执行的那个动作（``{kind,name?,input?,prompt?...}``）,
    ``output`` 是它的产出（工具结果由宿主填），``vars`` 是这一步执行后的全部变量取值。

    ``meta`` 装**与语义无关的执行侧账**：这一步的 token 数、延迟、真正执行的 argv、模型 id。
    它不参与规范化（normalize 只看 action/output）、不参与评判，纯粹是实验报告要统计的东西。
    单开一个字段而不是往 ``output`` 里塞，是因为 ``output`` 会被 writes 白名单收、会进
    canon_output，混进去就会污染状态身份。
    """

    step: int
    state: str = ""
    clause: str = ""
    action: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    vars: dict = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)


class Trace(BaseModel):
    """一次执行的轨迹 + 头部（任务输入、评判结果、这次运行的出身）。

    ``verdict`` 由评判补上（accepted/rejected）；rejected 必带 ``error_step``（第一处
    偏离正确的位置），它是拒绝集排除检查的锚。

    ``arm``/``run``/``model``/``harness`` 是**运行级出身**，全部可缺省：三臂实验的报告要能
    对每条轨迹回答「哪条臂、第几次重复、哪个模型端点、哪套执行器跑出来的」。它们不影响编译
    与评判（编译只看 records 与 task），只在报告与复现时被读。
    """

    task: dict = Field(default_factory=dict)
    arm: str = ""
    run: int = 0
    model: str = ""
    harness: str = ""
    verdict: Literal["accepted", "rejected", "unknown"] = "unknown"
    error_step: Optional[int] = None
    records: list[Record] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rejected_has_error_step(self) -> "Trace":
        if self.verdict == "rejected" and self.error_step is None:
            raise ValueError("被拒绝的轨迹必须标出 error_step（第一处偏离位置）")
        return self

    def to_jsonl(self) -> str:
        """首行头部，其余每行一条记录。

        头部按规格的键序写：``{"header": true, "task_id", "arm", "run", "input", "task",
        "model", "harness", "verdict", "error_step"}``。两处刻意的取舍：

        * ``task_id``/``input`` 是从 ``task`` 里**镜像**出来的（规格把它们摊在头部顶层），
          而 ``task`` 仍整份写出——它可能还装着 ``files`` 这类只镜像会丢的键。读回时以
          ``task`` 为准，缺了才用镜像拼。
        * ``arm``/``run``/``model``/``harness`` 取默认值时不写，头部因此不会被一堆空串撑大；
          读回时 ``.get`` 补回同样的默认值。
        """
        head: dict[str, Any] = {"header": True}
        task = self.task if isinstance(self.task, dict) else {}
        if "task_id" in task:
            head["task_id"] = task["task_id"]
        if self.arm:
            head["arm"] = self.arm
        if self.run:
            head["run"] = self.run
        if "input" in task:
            head["input"] = task["input"]
        head["task"] = self.task
        if self.model:
            head["model"] = self.model
        if self.harness:
            head["harness"] = self.harness
        head["verdict"] = self.verdict
        if self.error_step is not None:
            head["error_step"] = self.error_step
        lines = [json.dumps(head, ensure_ascii=False)]
        for r in self.records:
            lines.append(json.dumps(r.model_dump(), ensure_ascii=False))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, text_or_path: Any) -> "Trace":
        """从 JSONL 文本或文件路径读回。首行是头部，其余是记录。

        对**老头部**（只有 ``task``/``verdict``/``error_step``、没有 ``header`` 标记、没有
        出身字段）完全兼容：缺的键一律走默认值。对**按规格写的头部**（``task_id``/``input``
        摊在顶层、没有 ``task``）也认：拿这两个键拼回 ``task``。

        只有不含换行的短字符串才试着当路径——多行 JSONL 内容不会被误当文件名。
        """
        text = str(text_or_path)
        if isinstance(text_or_path, Path) or ("\n" not in text and len(text) < 4096):
            try:
                p = Path(text)
                if p.is_file():
                    text = p.read_text(encoding="utf-8")
            except OSError:
                pass
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        if not rows:
            raise ValueError("空轨迹")
        head, body = rows[0], rows[1:]
        task = head.get("task")
        if not isinstance(task, dict):                  # 规格形状：顶层的 task_id/input
            task = {k: head[k] for k in ("task_id", "input") if k in head}
        return cls(
            task=task,
            arm=head.get("arm") or "",
            run=head.get("run") or 0,
            model=head.get("model") or "",
            harness=head.get("harness") or "",
            verdict=head.get("verdict", "unknown"),
            error_step=head.get("error_step"),
            records=[Record(**r) for r in body],
        )


# --------------------------------------------------------------------------- #
# 空机器：增量构造的地基（对应验收 ③「初始全回退」）
# --------------------------------------------------------------------------- #
def empty_machine(skill_id: str) -> Machine:
    """一台合法但什么都不学的机器：起点直接进 FALLBACK（解释执行），能跑、过结构检查。

    编译从这里开始：每学一条轨迹就在它上面做一次可校验的小改。此时任意轨迹都被它「平凡
    复述」（因为一切都交给 FALLBACK 解释），这正是验收 ③ 要的地基状态。
    """
    return Machine(
        skill_id=skill_id,
        initial=FALLBACK,
        states={FALLBACK: State(id=FALLBACK, action=EndAction(terminal="done"))},
        terminals=[Terminal(id="done")],
    )


# --------------------------------------------------------------------------- #
# 存取
# --------------------------------------------------------------------------- #
def save_machine(machine: Machine, root: Any) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MACHINE_FILE
    path.write_text(
        json.dumps(json.loads(machine.model_dump_json(by_alias=True)),
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return path


def load_machine(root: Any) -> Machine:
    p = Path(root)
    if p.is_dir():
        p = p / MACHINE_FILE
    if not p.is_file():
        raise FileNotFoundError(f"没有状态机定义: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("format") != "efsm-v1":
        raise ValueError(f"{p} 不是 efsm-v1 机器（format={data.get('format')!r}）")
    return Machine.model_validate(data)
