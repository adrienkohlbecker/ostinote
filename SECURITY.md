# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

- Use [GitHub's private vulnerability reporting](https://github.com/adrienkohlbecker/ostinote/security/advisories/new), or
- email **adrien.kohlbecker@gmail.com** with details and reproduction steps.

Please include any sanitized evidence that helps reproduce the boundary:

- `ostinote --version`, OS, Python version, and agent versions;
- whether hooks were installed globally or with `--project`;
- redacted hook config snippets, if the issue involves install/uninstall;
- redacted `logs/memory-<date>.log`, `logs/background.log`, or
  `~/.ostinote/hook-errors.log` lines, when relevant;
- whether `ostinote save --dry` or `ostinote doctor` reproduces the issue.

You can expect an acknowledgement within a few days. Please give a reasonable
window to address the issue before any public disclosure.

## Supported versions

Only the latest release receives security fixes.

## Expected behavior

`ostinote` runs a summarizer (by default the `claude` CLI, or the `codex` CLI
when only codex is installed) over your session transcripts and writes memory
files under `~/.ostinote/projects/<slug>/` by default (or
`<project>/.ostinote/` if configured in-repo); hook crashes are logged to
`~/.ostinote/hook-errors.log`. It executes no remote code and opens no network
listeners of its own, but note that the default summarizer sends the extracted
conversation content to the Anthropic API (or the OpenAI API via codex) — only
the orchestration is local.

These are not vulnerabilities by themselves:

- transcript content leaving the machine through the default `claude` or
  `codex` summarizer;
- a trusted user-configured `summarizer.command` running with the user's
  privileges and receiving extracted transcript content on stdin;
- user-chosen in-repo memory directories that are private and untracked;
- uninstall preserving existing memory files, logs, and Codex writable-root
  grants.

## In scope

The most sensitive surfaces, where reports are especially welcome:

- the spawned summarizer subprocess and the hook commands registered in your
  agent config;
- installer-written config and skill files, including Claude settings, Codex
  hook files, Codex `sandbox_workspace_write.writable_roots`, and user/project
  skill install paths;
- the memory-injection chain: untrusted transcript content flows through the
  summarizer into memory files that are injected verbatim into every future
  session of both agents, so anything that lets a poisoned session or a write
  to the memory directory persist instructions across sessions matters;
- per-project `.ostinote/config.json`, which arrives with a cloned repo and is
  therefore treated as untrusted (e.g. it may not set `summarizer.command` or
  `summarizer.engine`);
- path traversal, symlink following, hook clobbering, secret leakage, or unsafe
  log permissions across any of the files above.
