# Contributing

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,xlsx]"
pytest
```

The test suite is hermetic: it needs no network, model endpoint or API key. Tests that exercise
OpenCode's native tools are skipped when the `opencode` executable is not on `PATH`.

## Guidelines

- Keep the compiler independent of particular skills. Tool names, field names and skill names must not
  appear in `src/skill2fsm/fsm/`; `tests/test_46_generic_skills.py` enforces this.
- The trace update stage of the compiler does not call a model; keep it deterministic.
- Add or update tests with every change in behavior, and keep them hermetic.
- Record user-visible changes in `CHANGELOG.md`.
- Before a release, build and check the distribution: `python -m build && twine check --strict dist/*`.

## Releases

The version is defined in `src/skill2fsm/__init__.py`. Publishing a GitHub release runs
`.github/workflows/publish.yml`, which uploads the distribution to PyPI through trusted publishing.
