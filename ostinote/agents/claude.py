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

_SKIP_PREFIXES = ("<system-reminder>", "<command-name>", "<local-command")


class ClaudeAgent(Agent):
    """Adapter for Claude Code session transcripts."""

    name = "claude"

    def _extract_messages(self, obj: dict) -> list[Message]:
        if obj.get("type") not in ("user", "assistant") or obj.get("isMeta"):
            return []
        message = obj.get("message")
        if not isinstance(message, dict):
            return []
        texts = _extract_texts(message.get("content", ""))
        if not texts:
            return []
        role = "HUMAN" if obj["type"] == "user" else "AGENT"
        if role == "HUMAN" and all(text.startswith("[TOOL_RESULT:") for text in texts):
            role = "AGENT"
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

    def accepts_transcript_path(self, transcript_path: str, cwd: str) -> bool:
        """Accept hook transcripts only from Claude's project session dir."""
        expected = os.path.realpath(_session_dir(cwd))
        real_path = os.path.realpath(transcript_path)
        try:
            inside = os.path.commonpath([expected, real_path]) == expected
        except ValueError:
            inside = False
        return inside and os.path.isfile(real_path) and real_path.endswith(".jsonl")


def _extract_texts(content) -> list[str]:
    texts: list[str] = []
    if isinstance(content, str):
        if content.lstrip().startswith(_SKIP_PREFIXES):
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
                text = _string_content(block.get("text")).strip()
                if text and not text.lstrip().startswith(_SKIP_PREFIXES):
                    texts.append(text)
            elif btype == "tool_use":
                texts.append(_format_tool_use(block))
            elif btype == "tool_result":
                text = _string_content(block.get("content")).strip()
                if text:
                    detail = Agent._truncate(Agent._redact_sensitive(text), 200)
                    texts.append("[TOOL_RESULT: %s]" % detail)
    return texts


def _format_tool_use(block: dict) -> str:
    name = block.get("name", "?")
    inp = block.get("input", {}) or {}
    if not isinstance(inp, dict):
        inp = {}
    if name in ("Edit", "Read", "Write", "NotebookEdit"):
        filename = str(inp.get("file_path", "?")).rsplit("/", 1)[-1]
        return "[TOOL: %s %s]" % (name, filename)
    if name == "Bash":
        command = Agent._redact_sensitive(str(inp.get("command", "?")))
        return "[TOOL: Bash `%s`]" % Agent._truncate(command)
    if name in ("Grep", "Glob"):
        return "[TOOL: %s '%s']" % (name, inp.get("pattern", "?"))
    return "[TOOL: %s]" % name


def _session_dir(cwd: str) -> str:
    """Claude's project transcript directory for ``cwd``."""
    if os.name == "nt" and re.match(r"^[A-Za-z]:", cwd):
        cwd = cwd[0].lower() + cwd[1:]
    slug = re.sub(r"[^a-zA-Z0-9]", "-", cwd)
    return os.path.join(os.path.expanduser("~"), ".claude", "projects", slug)


def _string_content(value) -> str:
    """Flatten transcript text fields that may be strings or nested blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_string_content(item.get("text") or item.get("content")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return ""
