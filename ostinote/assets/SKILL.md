---
name: ostinote
description: Save session state for clean continuation next session.
allowed-tools: Read, Write
---

Write a handoff note so the next session can continue cleanly. Use your knowledge of the current session — you were here. Write in first person ("I").

**Path:** the file named on the `Write next handoff to:` line of the `=== HANDOFF ===` section injected at session start. If that line is absent, use `<project root>/.ostinote/ostinote.md`. Overwrite the file.

**If the file already exists, Read it first before Writing** — the Write tool requires a prior Read for existing files; a 1-line Read is enough.

Format:

```
# Handoff

## State
{What's done, what's not. Files, PRs, decisions. 2-4 lines max.}

## Next
{What to pick up. Priority order. 1-3 items.}

## Context
{Non-obvious gotchas, blockers, preferences from this session. Skip if nothing.}
```

Rules:

- Under 20 lines total
- Specific: file paths, PR numbers, branch names
- Forward-looking — the next session doesn't care about the journey
- If nothing meaningful to hand off, write: "No active work."

Say "Saved." when done — nothing else.
