# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

- Use [GitHub's private vulnerability reporting](https://github.com/adrienkohlbecker/ostinote/security/advisories/new), or
- email **adrien.kohlbecker@gmail.com** with details and reproduction steps.

You can expect an acknowledgement within a few days. Please give a reasonable
window to address the issue before any public disclosure.

## Supported versions

Only the latest release receives security fixes.

## Scope notes

`ostinote` runs a summarizer (by default the `claude` CLI) over your session
transcripts and writes memory files under `~/.ostinote/projects/<slug>/` by
default (or `<project>/.ostinote/` if configured in-repo); hook crashes are
logged to `~/.ostinote/hook-errors.log`. It executes no remote code and opens
no network listeners of its own, but note that the default summarizer sends
the extracted conversation content to the Anthropic API — only the
orchestration is local.

The most sensitive surfaces, where reports are especially welcome:

- the spawned summarizer subprocess and the hook commands registered in your
  agent config;
- the memory-injection chain: untrusted transcript content flows through the
  summarizer into memory files that are injected verbatim into every future
  session of both agents, so anything that lets a poisoned session or a write
  to the memory directory persist instructions across sessions matters;
- per-project `.ostinote/config.json`, which arrives with a cloned repo and is
  therefore treated as untrusted (e.g. it may not set `summarizer.command`).
