# Repository Guidelines

## Project Context

`ostinote` is a Python package and CLI for persistent, shared memory across coding agents. It installs hooks for Claude Code and Codex, saves summarized session history, and injects project memory into fresh sessions.

Treat this as infrastructure for agent memory: keep behavior predictable, preserve transcript/privacy boundaries, and avoid noisy repo-local state. By default memory lives outside the repo in `~/.ostinote/projects/<project-slug>/`; do not add generated memory files, logs, or hook state to the repository.

## Development

- Runtime minimum: Python 3.11 (`requires-python`); local development uses the Python 3.14 toolchain pinned by `mise`.
- Local toolchain is managed by `mise`; the repo venv is auto-created at `.venv`.
- Use `mise install`, then `mise run setup`, after a fresh checkout.
- Run tests with `mise run test`.
- Run lint/format checks with `mise run lint`; use `mise run fix` for mechanical fixes.
- CLI entrypoint is `ostinote = ostinote.cli:main`.

## Code Conventions

- Keep changes small and behavior-focused; this project is intentionally a compact single-package CLI.
- Prefer standard-library APIs and structured parsers over shell snippets or ad hoc string manipulation.
- Preserve compatibility with existing on-disk memory formats and legacy config keys unless the user explicitly asks for a breaking migration.
- Be careful around hook installers: validate config writes, avoid clobbering unrelated user hooks, and fail closed on malformed config.
- When touching transcript parsing, state tracking, recovery, consolidation, or installer behavior, add or update regression tests in the relevant `tests/test_<area>.py` module.
- Document functions with docstrings. Write them for an intermediate Python reader who has only surface-level knowledge of this tool: say what the function does, why its contract matters, and any non-obvious inputs, outputs, side effects, or failure modes.
- Keep code comments sparse and useful. Add comments for non-obvious code paths, especially tricky concurrency, atomic-write, hook-shape, parsing, or compatibility logic; do not narrate obvious assignments.

## Verification

Before finishing code changes, run the narrowest useful check plus the full checks when practical:

```bash
mise run test
mise run lint
```

For hook/parser work, also consider:

```bash
mise run dry:claude
mise run dry:codex
mise run status
```

If a command fails because the local environment lacks an installed agent CLI or live summarizer access, report that clearly and include the exact failing command.
