---
description: Append a core memory — a fact that should never age out.
---

Append a core memory to this project's persistent memory. Core memories are injected verbatim at the start of every fresh session, forever — they are for facts that must never age out: a final decision and its rationale, a hard-won gotcha that will bite again, a standing preference.

User input (may be empty): $ARGUMENTS

What to save: if the user input above is non-empty, that is the memory — tighten it, don't change its meaning. If it is empty, distill the most durable lesson or decision from the current session (usually one; never more than two). If nothing qualifies, say so and write nothing.

Where: `core-memories.md` in the memory folder named on the `Persistent memory in ...` line of the `=== OSTINOTE ===` section injected at session start. If that section isn't in context, run `ostinote status` and use the `data dir` it prints.

Format: append one dated one-liner per memory:

```
- YYYY-MM-DD: {the fact, in one line}
```

Keep all existing content of the file, and skip anything it already covers. If the file doesn't exist, create it starting with a `# Core Memories` heading and a blank line.

Confirm by echoing the line(s) you appended — nothing else.
