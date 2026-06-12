# Contributing

Thanks for your interest! Bug reports, feature ideas, and pull requests are all welcome.

## Development setup

The repo uses [mise](https://mise.jdx.dev) to pin Python and uv and to expose tasks:

```bash
git clone https://github.com/adrienkohlbecker/ostinote && cd ostinote
mise install        # Python + uv, creates .venv
mise run setup      # locked editable install with dev deps (uv sync)
mise run test       # pytest suite
mise run lint       # ruff check + format check (same as CI)
mise run fix        # auto-fix lint findings and reformat
```

Without mise: any Python 3.11+, `pip install -e . --group dev` (pip ≥ 25.1), then `pytest`, `ruff check .`, `ruff format .`. Note that CI installs the exact versions in `uv.lock`; if a lint or format check disagrees with CI, that's the first thing to look at.

Useful while hacking:

```bash
ostinote save --dry --agent claude   # show what the parser extracts, no model call
ostinote status                      # inspect memory state for this project
```

## Guidelines

- **Tests:** every behavior change needs a test in `tests/`. The suite is fast and offline — model calls are mocked, transcripts are synthesized fixtures.
- **Lint:** CI runs `ruff check` and `ruff format --check`; `mise run fix` keeps you clean.
- **No new dependencies** without prior discussion — the package is deliberately stdlib-only at runtime.
- **Portability:** code must work on macOS, Linux, and Windows (no bash, no POSIX-only APIs without an `os.name == "nt"` branch).
- **Adding an agent:** see "Adding another agent" in the README — a transcript parser in `ostinote/agents/` plus an installer entry is all it takes.

## Pull requests

Keep PRs focused on one change. Describe the user-visible behavior in the description; link the issue if one exists. CI (lint + tests on Linux/macOS/Windows) must pass.

## Reporting bugs

Use the bug report issue template. Hook crashes land in `~/.ostinote/hook-errors.log` and pipeline activity in `logs/memory-<date>.log` inside the data dir (default `~/.ostinote/projects/<slug>/`) — including the relevant lines makes most bugs trivially diagnosable. Review and redact log lines before posting: they can contain summaries of your sessions and local paths.
