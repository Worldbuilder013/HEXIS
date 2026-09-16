# Contributing

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,xlsx]"
pytest
ruff check src tests
```

The test suite is hermetic: it needs no network, model endpoint or API key. Tests that exercise OpenCode's native
tools are skipped when the `opencode` executable is not on `PATH`.

## Layout

The package is `hexis` (distribution and command `hexis-agent`):

- `hexis.machine`: the efsm-v1 schema, the guard language and structural checks.
- `hexis.compiler`: compile context, initialization, trace normalization, alignment, candidate construction,
  checks and replay, update, stepwise decisions.
- `hexis.execution`: the runtime interpreter.
- `hexis.llm`, `hexis.tools`, `hexis.traces`, `hexis.evaluators`: model access, tool backends, trace formats,
  graders.
- `hexis.builddir`, `hexis.updater`, `hexis.step_judge`, `hexis.guide`: build directories, the update loop, the
  model decider and the usage guides.
- `hexis.legacy`: modules of an earlier compiler iteration. The command-line interface does not import them; do
  not use them in new code.

## Guidelines

- Keep the compiler independent of particular skills. Tool names, field names and skill names must not appear in
  `src/hexis/compiler/`; `tests/test_46_generic_skills.py` enforces this.
- `hexis.compiler` never calls a model directly: models are passed in (initialization, rule extraction) or decide
  through a callback (`hexis.compiler.decide.decide_trace`). The deterministic parts (alignment, candidate
  construction, checks, replay) must stay deterministic.
- All text that users or models see is English: messages, reports, prompts, registry descriptions, example
  skills, comments and docstrings. `tests/test_english_text.py` enforces this; its allowlist is only for
  identifiers that must match files written by earlier versions.
- Never put API keys in logs, build directories or command lines.
- Add or update tests with every change in behavior, and keep them hermetic.
- Record user-visible changes in `CHANGELOG.md`.
- Before a release, build and check the distribution: `python -m build && twine check --strict dist/*`.

## Releases

The version is defined in `src/hexis/__init__.py`. Publishing a GitHub release runs
`.github/workflows/publish.yml`, which uploads the distribution to PyPI through trusted publishing.
