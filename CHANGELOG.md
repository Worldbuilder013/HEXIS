# Changelog

## 0.1.0

Initial release.

- Extended finite state machine model (`efsm-v1`): typed variables, tool / model / judge / user / end
  actions, ordered guarded transitions with loop counters, terminals and a fallback state.
- Interpreter with per-step accounting, retries at the fallback state and handover to interpreted
  execution of the skill document.
- Compiler: compile context from the skill document, tool registry and traces; model-based
  initialization with check feedback; model-free trace update (normalization, alignment,
  modification, static checks, replay of accepted traces, acceptance); stepwise update driven by
  external decisions.
- Tool backends: OpenCode native tools, a local `bash` subprocess, and model-realized tools defined in
  a registry.
- Graders: SpreadsheetBench golden workbooks, multiple-choice answers, DABench `@name[value]` answers
  and file-answer tasks.
- Command-line interface: `compile`, `compile-stepwise`, `run`, `collect`, `fold-traces`, `bench`,
  `memory` and `summarize`.
