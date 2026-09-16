"""hexis: compile agent skills into extended finite state machines.

A machine keeps the current state and a set of variables, executes the operation assigned to the current
state and evaluates guards over the variables to choose the next state; models are only called inside
states. Main modules:

* :mod:`hexis.machine.schema`: machine and trace data model;
* :mod:`hexis.execution.runtime`: interpreter with retries and fallback to interpreted execution;
* :mod:`hexis.compiler`: compiler (compile context, initialization, trace update, stepwise update);
* :mod:`hexis.llm.llm_client`, :mod:`hexis.tools.opencode_tools`, :mod:`hexis.tools.local_tools`: model and tool backends;
* :mod:`hexis.evaluators`: graders;
* :mod:`hexis.cli`: the ``hexis`` command.

The hermetic example skill :mod:`hexis.examples.table_clean` runs without network access.
"""

__version__ = "0.1.0"
