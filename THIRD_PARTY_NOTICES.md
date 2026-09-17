# Third-party notices

hexis is released under the MIT License (see `LICENSE`). It bundles no third-party source code.

## Programs used at run time (not distributed)

- OpenCode, <https://opencode.ai> (MIT License): executes native tool calls.
  `src/hexis/tools/backends/opencode.json` lists its tool and argument names for interoperability.

## Python dependencies (installed from PyPI, not bundled)

| Package | License |
|---|---|
| pydantic, PyYAML | MIT |
| httpx | BSD-3-Clause (its dependency certifi is MPL-2.0) |
| pytest, build, ruff (extra `dev`) | MIT |
| twine (extra `dev`) | Apache-2.0 |
