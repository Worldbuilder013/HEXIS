"""hexis: compile agent skills into extended finite state machines.

A machine keeps the current state and a set of variables, executes the operation assigned to the current
state and evaluates guards over the variables to choose the next state; models are only called inside
states. Subpackages:

* :mod:`hexis.machine`: the efsm-v1 machine and trace model, the guard language, structural checks;
* :mod:`hexis.compiler`: compile context, initialization, trace update, stepwise decisions;
* :mod:`hexis.execution`: the runtime interpreter with retries and fallback to interpreted execution;
* :mod:`hexis.llm`, :mod:`hexis.tools`, :mod:`hexis.traces`: model access, tool backends, trace formats;
* :mod:`hexis.builddir`, :mod:`hexis.updater`, :mod:`hexis.step_judge`, :mod:`hexis.guide`: build directories,
  model-driven updates and usage guides;
* :mod:`hexis.cli`: the ``hexis-agent`` command;
* :mod:`hexis.legacy`: modules of an earlier compiler iteration, not used by the command-line interface.

The hermetic example skill :mod:`hexis.examples.table_clean` runs without network access.
"""

__version__ = "0.1.0"
