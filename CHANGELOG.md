# Changelog

## 0.1.0

Initial release.

- Extended finite state machine model (`efsm-v1`): typed variables, tool / model / judge / user / end actions,
  ordered guarded transitions with loop counters, terminals and a fallback state.
- Interpreter with per-step accounting, retries at the fallback state and handover to interpreted execution of the
  skill document.
- Compiler: compile context from the skill document, tool registry and traces; model-based initialization with
  check feedback; trace update by deterministic alignment or by step decisions, with static checks and replay of
  every accepted trace before a change is accepted.
- `hexis-agent compile` with model endpoint options (`--model`, `--base-url`, `--api-key-env`, `--temperature`,
  `--extra-body`, ...); `--traces` is optional; writes a build directory (`build.json`, skill snapshot, tool
  registry, rules, trace copies, progress).
- `hexis-agent update`: folds new traces into a build directory; a model decides every trace step (match / new /
  ignore / exclude) with answers cached in `decisions.jsonl`, resumable runs, `--show` previews, a baseline check,
  and protection of every previously accepted trace. Decisions can also come from a file or from deterministic
  alignment.
- `hexis-agent guide`: `GUIDE.md` (inputs, tools, Mermaid diagram, states, transitions, limits, fallback) and
  `PROMPT.md` (a system prompt that lets a tool-using agent execute the machine); written by `compile` and
  `update` as well.
- `hexis-agent run --mode task`: run any machine on inputs given with `--input KEY=VALUE` in a kept `--workdir`;
  `--machine` accepts a build directory.
- Tool backends: OpenCode native tools, a local `bash` subprocess, and model-realized tools defined in a registry.
- Graders: SpreadsheetBench golden workbooks, multiple-choice answers, DABench `@name[value]` answers and file-answer
  tasks.
- Benchmark harness: `collect`, `fold-traces`, `bench`, `memory` and `summarize`.
- All messages, reports, prompts and documentation are in English. The abstain label of judge actions is
  `abstain`; machines that use the label of earlier versions still load and run. Recompiling a skill therefore
  shows models English text where earlier versions did not.
- Modules of an earlier compiler iteration are kept, unchanged, in `hexis.legacy`; the command-line interface does
  not use them.
- Fixes: rules written by `compile` keep their label rules when read back; `compile-stepwise` identifies traces by
  file name, so several traces of one task no longer collide; a missing accepted trace is an error instead of being
  left out of the protection replay; endpoint failures stop initialization and rule extraction instead of being
  counted as failed drafts; JSON repair during compilation sends back the whole reply.
