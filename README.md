# ostinote — continuous memory for coding agents

[![PyPI](https://img.shields.io/pypi/v/ostinote)](https://pypi.org/project/ostinote/)
[![CI](https://github.com/adrienkohlbecker/ostinote/actions/workflows/ci.yml/badge.svg)](https://github.com/adrienkohlbecker/ostinote/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/ostinote/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE.md)

**Shared, persistent memory for your coding agents.** Works with both **Claude Code** and **Codex CLI** at the same time, in the same project — every session starts already knowing what you worked on yesterday, what decisions were made, and what's next.

Coding agents start every session blank. You re-explain the project, the conventions, the thing that broke last Tuesday. `ostinote` fixes that: it quietly captures your sessions as you work, compresses them into short daily summaries using Claude Haiku (a few cents per day), and injects them back into context whenever a new session starts — no matter which agent you open.

> **The name:** an *ostinato* is a musical motif that persists, repeating underneath everything else in a piece. Cross it with *note* — a written one — and you get **ostinote**: a persistent note running under all your sessions.

```
You, yesterday in Codex:        "the deploy failed because of the zram unit"
Your agent, today in Claude Code:  already knows.
```

## Quick start

```bash
# 1. Install the CLI (any Python 3.11+; `pipx install ostinote` works too)
uv tool install ostinote

# If your shell cannot find `ostinote` after install:
uv tool update-shell

# 2. Hook it into your agents (one-time, global)
ostinote install claude
ostinote install codex

# 3. Done. Work normally — memory accumulates in ~/.ostinote/projects/<project-slug>/
```

**Requirements:** Python 3.11+ and the `claude` CLI (used for summarization, with Haiku). Works on macOS and Linux; Windows is supported in the code (no bash needed) but has not seen much real-world testing yet.

By default your memory lives **outside** your repos, in `~/.ostinote/projects/<project-slug>/` (one folder per project), so it does not show up in repo diffs or your agent's review UI. Prefer it in-repo? Set `"data_dir": ".ostinote"` (see [Configuration](#configuration)), keep that directory untracked, and treat anything inside it as future prompt input.

Codex will ask you to trust the new hooks once on its next start.

> **Migrating from the `remember` Claude Code marketplace plugin?** Disable it first (`/plugin`), or both will save the same sessions twice. Formats and config keys are compatible — see [Migrating from the `remember` plugin](#migrating-from-the-remember-plugin).

## Lifecycle

`ostinote` has three kinds of state, and they are removed separately:

| Goal | Command | What remains |
|---|---|---|
| Update the CLI | `uv tool upgrade ostinote` | Hooks and memory stay as-is |
| Remove hooks and skills | `ostinote uninstall all` | The CLI, memory files, logs, and Codex writable-root grants remain |
| Remove the CLI | `uv tool uninstall ostinote` | Hooks already registered in agent configs are not removed automatically |

Use `ostinote status` to locate the memory directory before deleting old data manually. Codex installs add the memory directory to `sandbox_workspace_write.writable_roots`; uninstall leaves that grant in place intentionally so existing memory remains usable until you remove the entry yourself.

## Privacy and data flow

`ostinote` reads agent transcripts, extracts the new human/assistant conversation, and sends that extract to the configured summarizer. The default summarizer is the `claude` CLI with Haiku when it is installed, falling back to the `codex` CLI on codex-only machines, so extracted conversation content leaves your machine for Anthropic's or OpenAI's API. The orchestration is local, but the default model call is not.

Generated memory and logs are plaintext markdown/text files under the memory directory. They can contain session summaries, local paths, command output, and other sensitive context. `ostinote save --dry` avoids model calls and writes, but it still prints extracted transcript content; redact it before sharing.

Memory files are injected verbatim into future fresh agent sessions. If you choose in-repo memory (`"data_dir": ".ostinote"`) or migrate an old `.remember/` directory, inspect it first, keep it untracked/private, and remember that edits to memory files can influence later sessions.

Only trusted user config (`~/.ostinote/config.json`) may set `summarizer.command` or `summarizer.engine`, because that command runs with your user privileges and receives extracted transcript content on stdin. Prefer direct, trusted executables over shell wrappers such as `sh -c`. Per-project config is treated as untrusted and cannot choose the executable or which vendor receives the extract.

See [SECURITY.md](SECURITY.md) for the vulnerability-reporting scope.

## How it works

The mental model is simple:

1. **Capture:** hooks notice when a Claude Code or Codex transcript has enough new conversation.
2. **Summarize:** a detached background process asks the summarizer for a concise entry and appends it to `now.md`.
3. **Compress:** periodic jobs merge `now.md` into `today-YYYY-MM-DD.md`, then older days into `recent.md` and `archive.md`.
4. **Inject:** fresh sessions receive a `=== MEMORY ===` block assembled from the useful layers.

Both agents use the same storage, locks, cooldowns, recovery logic, and consolidation pipeline. Parallel sessions are safe: each session tracks its own transcript position, shared writes are serialized by a lock, and git worktrees share the main checkout's memory by default (`share_worktrees`).

Optional curated memory files live alongside the automatic layers. `identity.md` is written by you and injected first. `core-memories.md` is append-only durable memory; the installed `/ostinote` skill in Claude Code and `$ostinote` skill in Codex append one dated line there when you want a fact to survive normal aging.

## Check and inspect

You rarely need these, but they are useful when something looks off:

```bash
ostinote status          # memory location, tracked sessions, last saves
ostinote status --costs  # per-day token usage and cost, summed from logs
ostinote doctor          # check hooks, config, summarizer, and writable data dir
ostinote doctor --live   # include one real summarizer call
ostinote save --dry      # print extracted conversation; no model call, no writes
ostinote save --force    # save right now, skipping cooldowns
ostinote consolidate     # run daily consolidation right now
```

Development tasks include `mise run test`, `mise run test -- -m "not functional"`, `mise run lint`, `mise run dry:claude`, and `mise run dry:codex`.

Troubleshooting notes:

- **Nothing is being saved.** Check that `claude` is on the PATH of the shell your agent launches hooks from, and that the session has at least `min_human_messages` human messages.
- **What is it capturing, exactly?** `ostinote save --dry` prints the extracted conversation without calling the model or writing anything. Treat the output as private transcript material.
- **Where are logs?** Pipeline activity is logged to `logs/memory-<date>.log` inside the data dir, detached save subprocess output goes to `logs/background.log`, and hook crashes land in `~/.ostinote/hook-errors.log`. Redact before sharing.
- **Codex says hooks need approval.** Expected on first run — Codex pins a hash of each hook command and asks once.

## The memory files

Everything lives in your per-project memory folder, created automatically:

| File | What it is |
|---|---|
| `now.md` | Current buffer — most recent activity |
| `today-*.md` | One file per day of compressed history |
| `recent.md` | The last ~7 days |
| `archive.md` | Older history |
| `identity.md` | Optional user-written persona/instructions, injected first |
| `core-memories.md` | Optional durable facts, appended by `/ostinote` / `$ostinote` |
| `state/`, `logs/` | Bookkeeping: transcript positions, locks, pipeline logs, background output |

## Configuration

Optional. Create `~/.ostinote/config.json` (trusted user config, applies everywhere) or `<project>/.ostinote/config.json` (untrusted project config, overrides safe project-local settings). The common settings are:

```json
{
  "timezone": "Europe/Paris",
  "data_dir": "~/.ostinote/projects/{slug}",
  "summarizer": { "timeout": 120 }
}
```

| Setting | Default | What it does |
|---|---|---|
| `data_dir` | `~/.ostinote/projects/{slug}` | Where memory lives. `{slug}` expands to the project's path. User config may point anywhere you trust. Project config may use an in-project relative path like `".ostinote"` or the validated per-project default external path; arbitrary external paths belong in user config. |
| `timezone` | system local | IANA zone name (e.g. `America/New_York`) for timestamps and day boundaries |
| `time_format` | `24h` | `24h` or `12h` timestamps |
| `share_worktrees` | `true` | Sessions in git worktrees share the main checkout's memory |
| `summarizer.engine` | auto | Trusted user-config-only choice of built-in summarizer: `claude`, `codex`, or auto (prefer claude, fall back to codex on codex-only machines) |
| `summarizer.command` | per engine | Trusted user-config-only argv array for a summarizer that reads a prompt on stdin and prints a response; overrides `engine` |
| `summarizer.timeout` | `120` | Summarizer timeout, seconds |

Less common tuning keys include `cooldowns.save_seconds`, `cooldowns.compress_seconds`, `thresholds.min_human_messages`, `thresholds.delta_lines_trigger`, `features.hourly_compression`, `features.recovery`, `features.consolidation`, and `debug`.

Because a per-project config file arrives with a cloned repo, it is treated as untrusted: `summarizer.command` and `summarizer.engine` are ignored there, and a project-set `data_dir` is contained before use.

`ostinote install` also takes `--project` to register hooks for a single project (in `<project>/.claude/settings.json` / `<project>/.codex/hooks.json`) instead of globally. Treat project hook files like executable agent config, review them before committing, and uninstall with the same `--project` scope.

## Adding another agent

The agent-specific surface is deliberately small: a transcript parser (`ostinote/agents/<name>.py`, ~80 lines — turn the agent's session log into a list of human/assistant messages) and an installer entry that writes its hook config. Everything else — storage, locking, summarization, consolidation, recovery — is shared.

## Migrating from the `remember` plugin

Coming from the `remember` Claude Code marketplace plugin? Disable it first (`/plugin`), or both will save the same sessions twice. The on-disk formats, project slugs, and config keys are compatible: old in-repo `.remember/` folders keep working with `"data_dir": ".remember"`, and external-mode trees can be copied into `~/.ostinote/projects/<slug>/`. Copy config to ostinote's locations (`~/.remember/config.json` -> `~/.ostinote/config.json`, per-project `.remember/config.json` -> `.ostinote/config.json`), and inspect any migrated memory before injecting it into fresh sessions.

## Relationship to claude-remember

This project is an independent implementation (entirely new code and originally written prompts, single Python package, no bash) inspired by the layered-memory design of [claude-remember](https://github.com/Digital-Process-Tools/claude-remember) by Digital Process Tools, extended to serve multiple agents and parallel sessions. See [NOTICE.md](NOTICE.md) for the full acknowledgment.

## License

[MIT](LICENSE.md)
