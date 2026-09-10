"""skill2fsm: compile agent skills into extended finite state machines.

A machine keeps the current state and a set of variables, executes the operation assigned to the current
state and evaluates guards over the variables to choose the next state; models are only called inside
states. Main modules:

* :mod:`skill2fsm.schema`: machine and trace data model;
* :mod:`skill2fsm.runtime`: interpreter with retries and fallback to interpreted execution;
* :mod:`skill2fsm.fsm`: compiler (compile context, initialization, trace update, stepwise update);
* :mod:`skill2fsm.llm_client`, :mod:`skill2fsm.opencode_tools`, :mod:`skill2fsm.local_tools`: model and tool backends;
* :mod:`skill2fsm.evaluators`: graders;
* :mod:`skill2fsm.cli`: the ``skill2fsm`` command.

The hermetic example skill :mod:`skill2fsm.examples.table_clean` runs without network access.
"""

__version__ = "0.1.0"
