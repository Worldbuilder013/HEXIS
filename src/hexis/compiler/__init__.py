"""Initialization and update of skill state machines (skill independent).

All compile inputs come from :class:`~hexis.compiler.context.CompileContext` (document, tool definitions, traces,
skill rules):

* :mod:`.context` compile context: task inputs, tool definitions, label rules, requirements, terminal conditions
* :mod:`.init`    initial machine generation and rule extraction (model touchpoint)
* :mod:`.traces`  turning traces into events (model generation / tool call / judge / user input / end)
* :mod:`.align`   dynamic programming alignment over a fixed cost table
* :mod:`.modify`  candidate machine construction from step contracts
* :mod:`.check`   variable / evidence / requirement checks and path replay
* :mod:`.update`  acceptance rule and trace-by-trace update

The update stage calls no model and is fully deterministic. Adding a new skill needs only its document, traces and
any necessary tool definitions.
"""
from hexis.compiler.align import Alignment, align
from hexis.compiler.check import analyze, replay
from hexis.compiler.check import check as check_machine
from hexis.compiler.context import (
           CompileContext,
           EventPattern,
           Requirement,
           TerminalCondition,
           build_context,
           load_rules,
)
from hexis.compiler.init import InitResult, extract_rules, initialize, install_rules, normalize
from hexis.compiler.modify import Build, StepContract, build_candidate
from hexis.compiler.traces import Event, Prepared, Segment, load_traces, prepare, segment_trace
from hexis.compiler.update import UpdateResult, update

__all__ = ["Alignment", "Build", "CompileContext", "Event", "EventPattern", "InitResult", "Prepared",
           "Requirement", "Segment", "StepContract", "TerminalCondition", "UpdateResult", "align",
           "analyze", "build_candidate", "build_context", "check_machine", "extract_rules",
           "initialize", "install_rules", "load_rules", "load_traces", "normalize", "prepare",
           "replay", "segment_trace", "update"]
