# ostinote — continuous memory for coding agents

**Shared, persistent memory for your coding agents.** Works with both **Claude Code** and **Codex CLI** at the same time, in the same project — every session starts already knowing what you worked on yesterday, what decisions were made, and what's next.

Coding agents start every session blank. You re-explain the project, the conventions, the thing that broke last Tuesday. `ostinote` fixes that: it quietly captures your sessions as you work, compresses them into short daily summaries using Claude Haiku (a few cents per day), and injects them back into context whenever a new session starts — no matter which agent you open.

> **The name:** an *ostinato* is a musical motif that persists, repeating underneath everything else in a piece. Cross it with *note* — a written one — and you get **ostinote**: a persistent note running under all your sessions.

```
You, yesterday in Codex:    "the deploy failed because of the zram unit"
Your agent, today in Claude Code:  already knows.
```

## Quick start

```bash
# 1. Install the CLI (any Python 3.9+; pipx works too)
uv tool install --editable /path/to/this/repo

# 2. Hook it into your agents (one-time, global)
ostinote install claude
ostinote install codex

# 3. Done. Work normally — memory accumulates in ~/.ostinote/projects/<project-slug>/
```

By default your memory lives **outside** your repos, in `~/.ostinote/projects/<project-slug>/` (one folder per project) — so it never shows up in your repo's diffs or your agent's review UI. Prefer it in-repo? Set `"data_dir": ".ostinote"` (see [Configuration](#configuration)).

Codex will ask you to trust the new hooks once on its next start. To stop using it: `ostinote uninstall all`.

> **Migrating from the `remember` Claude Code marketplace plugin?** Disable it first (`/plugin`), or both will save the same sessions twice. The on-disk formats, project slugs, and config keys are all compatible: old in-repo `.remember/` folders keep working with `"data_dir": ".remember"`, external-mode trees keep working with `"data_dir": "~/.remember/{slug}"` (same folder names), or copy their contents into `~/.ostinote/projects/<slug>/`. Old config files can be reused as-is — renamed keys like `ndc_seconds` and `ndc_compression` are still understood — but copy them to ostinote's locations: `~/.remember/config.json` → `~/.ostinote/config.json`, and a per-project `.remember/config.json` → `.ostinote/config.json` (ostinote never reads config from inside the data dir).

**Requirements:** Python 3.9+ and the `claude` CLI (used for summarization, with Haiku). Works on macOS and Linux; Windows is supported in the code (no bash needed) but hasn't seen real-world testing yet.

## What you get

- **Automatic capture.** While you work, new conversation is periodically summarized into one-line entries — no prompting, no copy-pasting.
- **Memory on session start.** Every new session (Claude or Codex) begins with your project's memory injected: today's activity, the last 7 days, older history, and an optional identity file.
- **One memory, every agent.** Both agents read and write the same per-project memory folder. Run five sessions in parallel across both tools — including in git worktrees — and they all feed the same memory without stepping on each other.
- **Handoff notes.** Type `/ostinote` before ending a session and the agent writes a short "here's where I left off" note that the next session (either agent) picks up and clears.

## How it works

Your raw transcripts are huge; context windows are not. So memory is stored in layers, each one a compressed version of the layer above:

```
your session transcript
   │   (summarized every few minutes during work)
   ▼
now.md                — running buffer of one-line entries for the current stretch
   │   (compressed about once an hour)
   ▼
today-2026-06-10.md   — today's activity, deduplicated and merged
   │   (consolidated once the day is over)
   ▼
recent.md             — the last ~7 days, one short entry per day
archive.md            — everything older, one short entry per week
```

All summarization and compression is done by Claude Haiku in the background — a typical save costs well under a cent.

Under the hood, both agents expose the same hook system (Codex's hooks are Claude-compatible), so one CLI serves both:

| When | What `ostinote` does |
|---|---|
| Session starts | Injects memory into context; rescues sessions that ended without a final save; kicks off daily consolidation if needed |
| After tool calls | Once enough new conversation accumulates, saves a summary in the background |
| Session ends (Claude) | Saves the session's final stretch in the background — the hook returns instantly, summarization finishes after the session is gone. Codex has no session-end hook event; recovery covers it instead |

## Features in detail

### Automatic capture

While you work, `ostinote` watches the session transcript (via the agents' post-tool-use hook). Once enough new conversation has accumulated (`thresholds.delta_lines_trigger`, default 50 transcript lines) and the per-session cooldown has passed (`cooldowns.save_seconds`, default 2 minutes), it launches a background save: the new exchanges are extracted, sent to Haiku, and appended to `now.md` as a one-line entry like:

```
## 15:05 | master
Built ostinote: multi-agent adapters, parallel-session state, CLI, tests.
```

Very short conversations are skipped (`thresholds.min_human_messages`, default 3 human messages), and if nothing meaningfully new happened since the last entry, the summarizer answers "SKIP" and nothing is written. Saves never block your agent — they run as detached background processes.

### Hourly compression

`now.md` grows one line per save. About once an hour (`cooldowns.compress_seconds`), the whole buffer is compressed into `today-YYYY-MM-DD.md`: entries about the same piece of work are merged into one time-blocked entry, filler is dropped, facts are kept. The buffer is then emptied. Disable with `features.hourly_compression: false` if you'd rather keep the raw entries all day.

### Daily consolidation

When a session starts and there are `today-*.md` files from *previous* days lying around, they're consolidated in the background: each past day becomes one short entry in `recent.md`, and entries older than a few days are rotated into `archive.md`, grouped by week. Processed daily files are renamed to `*.done.md` (kept, in case you ever want the originals). This is what keeps memory size roughly constant no matter how long you've been using it.

### Memory injection at session start

Every new session begins with a `=== MEMORY ===` block injected into the agent's context, assembled from whichever of these files exist and are non-empty, most personal first:

1. `identity.md` — who the agent is (see below)
2. `core-memories.md` — key moments (see below)
3. `ostinote.md` — the handoff note from the last session
4. `today-<date>.md` and `now.md` — what happened today
5. `recent.md` — the last ~7 days
6. `archive.md` — everything older

It also tells the agent where to write its next handoff note, so `/ostinote` works without any setup.

Injection only happens on fresh starts (`startup`/`clear`). Resumed and compacted sessions saw the memory already — both agents report how the session started via the hook's `source` field, and ostinote skips re-injection (and keeps the handoff intact) for `resume` and `compact`.

### Handoff notes (`/ostinote`)

The automatic pipeline captures *what happened*; the handoff captures *what matters next*. Before ending a session (or when context is getting full), type `/ostinote` — installed as a skill in Claude Code and a custom prompt in Codex. The agent writes a short structured note (state / next steps / gotchas) to `ostinote.md` in your memory folder. The next session — from either agent — gets it injected, then the file is cleared: it's a one-shot briefing, not accumulating history.

### Identity (`identity.md`)

Optional, written by you, never modified by the tool. If an `identity.md` file exists in your memory folder, its content is injected first in every session — use it to give your agent a persistent persona, values, or standing instructions that should survive across sessions and across agents. Example:

```markdown
# Identity
You are the long-running maintainer of this homelab. You prefer boring,
debuggable solutions; you treat the operator as the on-call.
```

This differs from `CLAUDE.md`/`AGENTS.md`: those are per-agent and usually committed to the repo; `identity.md` is agent-neutral, private (the data directory lives outside the repo by default), and travels with the memory.

### Core memories (`core-memories.md`)

Yours to curate, never compressed. While the compression layers deliberately shed detail over time, anything in `core-memories.md` (in your memory folder) is injected verbatim in every session, forever. Use it for the handful of moments or facts that should never age out — a hard-won debugging lesson, a decision and its rationale. Tip: when something like that happens, just tell your agent "add this to core memories" — it knows the path from the session-start injection.

The daily consolidation can also promote on its own: when a day's entries contain a clearly durable fact (a final decision, a gotcha that will bite again), it appends a dated one-liner. It's instructed to be very conservative — most days promote nothing — and it only ever appends; pruning the file stays yours.

### Recovery of missed sessions

Claude Code sessions get a final save from the session-end hook. But Codex has no session-end event, and any session can die without one (the laptop sleeps, the agent crashes, the machine shuts down mid-save). Every session leaves a small bookkeeping record, and at the next session start `ostinote` checks for transcripts that grew after their last save, have been idle for at least 5 minutes (so live parallel sessions are left alone), and are less than a week old — and saves up to 3 of them in the background. Disable with `features.recovery: false`.

### Parallel sessions, two agents, worktrees

Everything is built for concurrency: each session tracks its own resume position in the memory folder's `state/sessions/`, writes to the shared memory files are serialized by an atomic lock, and save cooldowns are per-session. So you can run Claude Code and Codex side by side — or several of each — and they'll all feed one memory without duplicate or interleaved entries. Sessions started inside a git worktree write to the main checkout's memory by default (`share_worktrees`), so a fix made in a worktree is remembered in the main checkout tomorrow.

## Everyday commands

You rarely need any of these — the hooks do everything — but they're useful for checking on things:

```bash
ostinote status          # where's my memory, what's in it, which sessions are tracked
ostinote status --costs  # per-day token usage and cost, summed from the logs
ostinote doctor          # check every link of the pipeline, loudly (--live: one real model call)
ostinote save --dry      # show what would be captured from the latest session (no cost)
ostinote save --force    # save right now, skipping cooldowns
ostinote consolidate     # run the daily consolidation right now
```

For development there are mise tasks: `mise run test`, `mise run lint`, `mise run dry:claude`, `mise run dry:codex`, etc.

## The memory files

Everything lives in your per-project memory folder — by default `~/.ostinote/projects/<project-slug>/`, created automatically (or `<project>/.ostinote/` if you set `data_dir` to a relative path):

| File | What it is |
|---|---|
| `now.md` | Current buffer — most recent activity |
| `today-*.md` | One file per day of compressed history |
| `recent.md` | The last ~7 days |
| `archive.md` | Older history |
| `ostinote.md` | The handoff note written by `/ostinote` (read once, then cleared) |
| `identity.md` | Optional — who your agent is. Write this yourself; it's injected first |
| `core-memories.md` | Optional — key moments worth keeping verbatim |
| `state/`, `logs/` | Bookkeeping: per-session positions, locks, pipeline logs |

## Configuration

Optional. Create `~/.ostinote/config.json` (applies everywhere) or `<project>/.ostinote/config.json` (overrides per project). Everything has sensible defaults:

```json
{
  "timezone": "Europe/Paris",
  "cooldowns": { "save_seconds": 120 },
  "thresholds": { "delta_lines_trigger": 50 }
}
```

| Setting | Default | What it does |
|---|---|---|
| `data_dir` | `~/.ostinote/projects/{slug}` | Where memory lives. `{slug}` expands to the project's path (non-alphanumerics dashed). The default keeps memory in your home directory, out of every repo. Set a relative value like `".ostinote"` to store it in-repo instead. (Per-project *config* is always read from `<project>/.ostinote/config.json`, wherever the data lives.) |
| `timezone` | system local | IANA zone name (e.g. `America/New_York`) for timestamps and day boundaries — set it on servers whose clock is UTC |
| `time_format` | `24h` | `24h` or `12h` timestamps |
| `share_worktrees` | `true` | Sessions in git worktrees share the main checkout's memory |
| `cooldowns.save_seconds` | `120` | Minimum time between saves of the same session |
| `cooldowns.compress_seconds` | `3600` | How often `now.md` is compressed into the daily file |
| `thresholds.min_human_messages` | `3` | Don't bother saving conversations shorter than this |
| `thresholds.delta_lines_trigger` | `50` | How much new transcript triggers an automatic save |
| `features.hourly_compression` | `true` | The hourly compression step |
| `features.recovery` | `true` | Rescue unsaved sessions on the next session start |
| `features.consolidation` | `true` | The daily recent/archive consolidation |
| `summarizer.command` | claude + haiku | Swap in a different summarizer — any command that reads a prompt on stdin and prints a response |
| `summarizer.timeout` | `120` | Summarizer timeout, seconds |
| `debug` | `false` | Chatty logs about locks and cooldowns |

`ostinote install` also takes `--project` to register hooks for a single project (in `<project>/.claude/settings.json` / `<project>/.codex/hooks.json`) instead of globally.

## Troubleshooting

- **Is it running?** `ostinote doctor` checks the whole chain (hooks registered, config readable, summarizer present, data dir writable) and `ostinote status` shows tracked sessions and when they last saved. Pipeline activity is logged to `logs/memory-<date>.log` inside the data dir; hook crashes land in `~/.ostinote/hook-errors.log`.
- **Nothing is being saved.** Check that `claude` is on the PATH of the shell your agent launches hooks from, and that the session has at least `min_human_messages` human messages.
- **What is it capturing, exactly?** `ostinote save --dry` prints the extracted conversation without calling the model or writing anything.
- **Codex says hooks need approval.** Expected on first run — Codex pins a hash of each hook command and asks once.

## Adding another agent

The agent-specific surface is deliberately small: a transcript parser (`ostinote/agents/<name>.py`, ~80 lines — turn the agent's session log into a list of human/assistant messages) and an installer entry that writes its hook config. Everything else — storage, locking, summarization, consolidation, recovery — is shared.

## Relationship to claude-remember

This project is an independent implementation (entirely new code and originally written prompts, single Python package, no bash) inspired by the layered-memory design of [claude-remember](https://github.com/Digital-Process-Tools/claude-remember) by Digital Process Tools, extended to serve multiple agents and parallel sessions. See [NOTICE.md](NOTICE.md) for the full acknowledgment.

## License

[MIT](LICENSE.md)
