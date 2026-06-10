# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

- Use [GitHub's private vulnerability reporting](https://github.com/adrienkohlbecker/ostinote/security/advisories/new), or
- email **adrien.kohlbecker@gmail.com** with details and reproduction steps.

You can expect an acknowledgement within a few days. Please give a reasonable
window to address the issue before any public disclosure.

## Scope notes

`ostinote` runs a local summarizer (by default the `claude` CLI) over your
session transcripts and writes memory files under `<project>/.ostinote/`. It
executes no remote code and opens no network listeners of its own. The most
sensitive surfaces are the spawned summarizer subprocess and the hook commands
registered in your agent config — reports touching those are especially welcome.
