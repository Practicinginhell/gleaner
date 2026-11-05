# scripts/ — Uniform Command Verbs

One wrapper per verb, so agents and CI never guess the stack's commands
(["scripts to rule them all"](https://github.blog/engineering/scripts-to-rule-them-all/)).
`AGENTS.md` and CI both call these.

| Script | Verb | What it runs | Contract |
| --- | --- | --- | --- |
| `setup` | Install deps | `uv pip install -e ".[dev,llama]"` (or `pip install` without uv) | Idempotent, non-interactive |
| `dev` | Run the app | `python -m gleaner "$@"` | Foreground, logs to stdout |
| `test` | Test suite | `python -m pytest -q "$@"` | Forwards args; exit 0 = green |
| `lint` | Static analysis | `python -m ruff check gleaner tests` | Read-only; exit 0 = clean |
| `format` | Autofix in place | `ruff check --fix` (safe fixes only, no `ruff format`) | Idempotent |
| `tangle` | Literate modules | `python scripts/tangle.py "$@"` (`--check` = drift check) | Exit 0 = in sync (ADR-0006) |

## Rules

- Every script runs from anywhere (each `cd`s to the repo root).
- `PYTHON` picks the interpreter (default: `python` on PATH). For example,
  `PYTHON=~/miniconda3/envs/gleaner/bin/python ./scripts/test` uses the project's 3.15 env.
- Never interactive.
- Exit codes are the API: `0` success, non-zero failure.
- Adding a verb? Add the wrapper here **and** a row to the Commands table in `AGENTS.md`.
