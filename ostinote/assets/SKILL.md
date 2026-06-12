---
name: ostinote
description: Append a core memory — a fact that should never age out.
allowed-tools: Read, Write, Bash
---

Append a core memory to this project's persistent memory. Core memories are injected verbatim at the start of every fresh session, forever — they are for facts that must never age out: a final decision and its rationale, a hard-won gotcha that will bite again, a standing preference.

**What to save:** if the user provided text with the command, that is the memory — tighten it, don't change its meaning. If they didn't, distill the most durable lesson or decision from the current session (usually one; never more than two). If nothing qualifies, say so and write nothing.

**Where:** `core-memories.md` in the memory folder named on the `Persistent memory in ...` line of the `=== OSTINOTE ===` section injected at session start. Before writing, verify that folder exists and looks like an ostinote data dir (it contains `now.md` or other memory files). If the section isn't in context or the path doesn't check out, run `ostinote status` and use the `data dir` it prints instead.

**Format:** append one dated one-liner per memory:

```
- YYYY-MM-DD: {the fact, in one line}
```

If the file exists, Read it first (the Write tool requires a prior Read), keep all existing content, and skip anything it already covers. If it doesn't exist, create it starting with a `# Core Memories` heading and a blank line.

Confirm by echoing the line(s) you appended — nothing else. Every line in this file costs context in every future session; if it has grown past a few dozen entries, also tell the user it is due for a manual prune.
