"""Claude Code transcript adapter.

Claude Code stores sessions as JSONL under
``~/.claude/projects/<cwd-slug>/<session-id>.jsonl``. Each line is a message
object; we keep user/assistant messages, drop metadata and injected
system-reminder content, and condense tool_use blocks into short markers.
"""

from __future__ import annotations

import glob
import os
import re

from .base import Agent, Message

_SKIP_MARKERS = ("<system-reminder>", "<command-name>", "<local-command")


class ClaudeAgent(Agent):
    """Adapter for Claude Code session transcripts."""

    name = "claude"

    def _extract_messages(self, obj: dict) -> list[Message]:
        if obj.get("type") not in ("user", "assistant") or obj.get("isMeta"):
            return []
        texts = _extract_texts(obj.get("message", {}).get("content", ""))
        if not texts:
            return []
        role = "HUMAN" if obj["type"] == "user" else "AGENT"
        return [(role, "\n".join(texts))]

    def find_latest_transcript(self, cwd: str) -> str | None:
        """Return the most recently modified session file for ``cwd``, or None.

        Reproduces Claude Code's project-path slugging to find the session
        directory under ``~/.claude/projects``; picks by mtime, not filename.
        """
        # Claude Code slugs the session directory from the project path; on
        # Windows it lowercases the drive letter (D:\proj -> d--proj).
        if os.name == "nt" and re.match(r"^[A-Za-z]:", cwd):
            cwd = cwd[0].lower() + cwd[1:]
        slug = re.sub(r"[^a-zA-Z0-9]", "-", cwd)
        # Join from components: expanduser would leave the literal's forward
        # slashes intact on Windows, producing a mixed-separator path.
        sdir = os.path.join(os.path.expanduser("~"), ".claude", "projects", slug)
        files = glob.glob(os.path.join(sdir, "*.jsonl"))
        return max(files, key=os.path.getmtime) if files else None


def _extract_texts(content) -> list[str]:
    texts: list[str] = []
    if isinstance(content, str):
        if any(m in content for m in _SKIP_MARKERS):
            return texts
        stripped = content.strip()
        if stripped:
            texts.append(stripped)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "").strip()
                if text and not any(m in text for m in _SKIP_MARKERS):
                    texts.append(text)
            elif btype == "tool_use":
                texts.append(_format_tool_use(block))
    return texts


def _format_tool_use(block: dict) -> str:
    name = block.get("name", "?")
    inp = block.get("input", {}) or {}
    if name in ("Edit", "Read", "Write", "NotebookEdit"):
        filename = str(inp.get("file_path", "?")).rsplit("/", 1)[-1]
        return "[TOOL: %s %s]" % (name, filename)
    if name == "Bash":
        return "[TOOL: Bash `%s`]" % Agent._truncate(str(inp.get("command", "?")))
    if name in ("Grep", "Glob"):
        return "[TOOL: %s '%s']" % (name, inp.get("pattern", "?"))
    return "[TOOL: %s]" % name
