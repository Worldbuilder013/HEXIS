# Security

## Execution model

hexis runs agents. `hexis-agent run` executes tool calls whose
arguments are produced by a language model, including arbitrary shell commands (`bash`) and file writes.
Treat machines, skills and task files as code.

- Run hexis in a disposable environment (container or virtual machine) that holds no credentials
  or data you need to protect.
- `--executor local` runs `bash` tool calls with `subprocess` and `shell=True` in the job directory, with
  the permissions of the current user and no further isolation. The default executor, OpenCode, also runs
  tools with the permissions of the current user.
- Tool calls can read and write files outside the job directory and reach the network unless the
  environment prevents it.
- Guards in machine files are evaluated by a whitelisted expression evaluator (`hexis.machine.cond`, no
  `eval`), but tool arguments rendered from a machine are executed as given. Only run machines and task
  files from sources you trust.
- Model endpoints are configured through environment variables or a `.env` file; keep `.env` out of
  version control (it is listed in `.gitignore`). Keys are read from the environment only (`--api-key-env`
  names the variable) and are never written to build directories or logs.
- `hexis-agent update` sends trace steps (tool arguments and shortened results) and the skill's clauses to the
  configured model endpoint. Build directories keep copies of the traces and of every question sent, and
  `PROMPT.md` contains the whole machine; review them before sharing.

## Reporting a vulnerability

Please report vulnerabilities privately through the repository's security advisory feature instead of a
public issue, with steps to reproduce and the affected version.
