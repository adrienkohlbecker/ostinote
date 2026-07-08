# Contributing

Thanks for your interest! Bug reports, feature ideas, and pull requests are all welcome.

## Development setup

The repo uses [mise](https://mise.jdx.dev) to pin Python and uv and to expose tasks:

```bash
git clone https://github.com/adrienkohlbecker/ostinote && cd ostinote
mise install                         # Python + uv, creates .venv
mise run setup                       # locked editable install with dev deps
mise run test                        # pytest suite
mise run test -- -m "not functional" # fast loop without subprocess tests
mise run lint                        # ruff check + format check (same as CI)
mise run fix                         # auto-fix lint findings and reformat
```

Use `mise tasks ls` to discover the full task list. Without mise: any Python 3.11+, `pip install -e . --group dev` (pip ≥ 25.1), then `pytest`, `ruff check .`, `ruff format .`. Note that CI installs the exact versions in `uv.lock`; if a lint or format check disagrees with CI, that's the first thing to look at.

Useful while hacking:

```bash
mise run dry:claude   # show what the Claude parser extracts, no model call
mise run dry:codex    # show what the Codex parser extracts, no model call
ostinote status       # inspect memory state for this project
```

Dry-run output can contain private transcript material; redact it before
sharing.

## Guidelines

- **Tests:** every behavior change needs a test in `tests/`. The suite is fast and offline — model calls are mocked, transcripts are synthesized fixtures.
- **Lint:** CI runs `ruff check` and `ruff format --check`; `mise run fix` keeps you clean.
- **No new dependencies** without prior discussion — the package is deliberately stdlib-only at runtime.
- **Portability:** code must work on macOS, Linux, and Windows (no bash, no POSIX-only APIs without an `os.name == "nt"` branch).
- **Adding an agent:** see "Adding another agent" in the README — a transcript parser in `ostinote/agents/` plus an installer entry is all it takes.

## Pull requests

Keep PRs focused on one change. Describe the user-visible behavior in the description; link the issue if one exists. CI (lint + tests on Linux/macOS/Windows) must pass.

## Releasing (maintainers)

`mise run release [patch|minor|major|X.Y.Z]` (default `patch`) runs the full test and lint suites, bumps the version in `pyproject.toml`/`uv.lock`, prepends a section to `CHANGELOG.md` built from the commit subjects since the last tag, commits, and creates a signed `vX.Y.Z` tag carrying that section. It requires a clean worktree on `main` and pushes nothing; it prints the `git push` and `gh release create` follow-up commands. Publishing the GitHub release triggers the PyPI publish workflow, which re-validates that the tag matches the project version and that the commit is on `main`.

## Reporting bugs

Use the bug report issue template. Hook crashes land in `~/.ostinote/hook-errors.log`, pipeline activity in `logs/memory-<date>.log`, and detached save subprocess output in `logs/background.log` inside the data dir (default `~/.ostinote/projects/<slug>/`) — including the relevant lines makes most bugs trivially diagnosable. Review and redact log lines before posting: they can contain summaries of your sessions and local paths.
