"""技能状态机的初始化与更新（技能无关）。

实现 ``out/# 技能状态机的初始化与更新算法.md``，编译输入全部来自
:class:`~skill2fsm.fsm.context.CompileContext`（文档、工具定义、轨迹、技能规则）：

* :mod:`.context` 编译上下文：任务输入、工具定义、标签规则、要求、终点条件
* :mod:`.init`    第 2 节  初始状态机生成与规则抽取（模型触点）
* :mod:`.traces`  第 3 节  轨迹事件化（模型生成 / 工具调用 / 判断 / 用户输入 / 结束）
* :mod:`.align`   第 4 节  固定代价表上的动态规划对齐
* :mod:`.modify`  第 5 节  按步骤合同构造候选机器
* :mod:`.check`   第 6 节  变量 / 证据 / 要求检查与路径回放
* :mod:`.update`  第 7 节  接受规则与逐条更新

更新阶段不调模型，全程确定性。加一个新技能只需文档、轨迹和必要的工具定义。
"""
from .align import Alignment, align
from .check import analyze, replay
from .check import check as check_machine
from .context import CompileContext, EventPattern, Requirement, TerminalCondition, build_context, load_rules
from .init import InitResult, extract_rules, initialize, install_rules, normalize
from .modify import Build, StepContract, build_candidate
from .traces import Event, Prepared, Segment, load_traces, prepare, segment_trace
from .update import UpdateResult, update

__all__ = ["Alignment", "Build", "CompileContext", "Event", "EventPattern", "InitResult", "Prepared",
           "Requirement", "Segment", "StepContract", "TerminalCondition", "UpdateResult", "align",
           "analyze", "build_candidate", "build_context", "check_machine", "extract_rules",
           "initialize", "install_rules", "load_rules", "load_traces", "normalize", "prepare",
           "replay", "segment_trace", "update"]
