"""Codex CLI transcript adapter.

Codex stores sessions ("rollouts") as JSONL under
``~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<id>.jsonl``. Lines have a
``type`` field; conversation content lives in ``response_item`` payloads:

    message        role user/assistant, content blocks input_text/output_text
    function_call  name + JSON-string arguments (exec_command, apply_patch, …)
    custom_tool_call  name + input string
    reasoning, *_output, web_search_call  — skipped

User messages also carry injected context (AGENTS.md instructions,
environment context, hook output); those are filtered by prefix markers.
"""

from __future__ import annotations

import json
import os
import time

from .base import Agent, Message

# Injected (non-human) user message prefixes to drop.
_INJECTED_PREFIXES = (
    "# AGENTS.md",
    "<INSTRUCTIONS>",
    "<user_instructions>",
    "<environment_context>",
    "<turn_context>",
    "<system-reminder>",
    "<permissions",
    "<hook",
    "# Hook output",
    "Caveat:",
)

SESSIONS_ROOT = os.path.join(os.path.expanduser("~"), ".codex", "sessions")


class CodexAgent(Agent):
    """Adapter for Codex CLI rollout transcripts."""

    name = "codex"

    def _extract_messages(self, obj: dict) -> list[Message]:
        if obj.get("type") != "response_item":
            return []
        payload = obj.get("payload", {})
        if not isinstance(payload, dict):
            return []
        ptype = payload.get("type")

        if ptype == "message":
            role = payload.get("role")
            # "developer" carries injected context (permissions, hook
            # output) — not conversation.
            if role not in ("user", "assistant"):
                return []
            texts = _message_texts(payload.get("content", []), role)
            if not texts:
                return []
            return [("HUMAN" if role == "user" else "AGENT", "\n".join(texts))]
        if ptype == "function_call":
            return [("AGENT", _format_function_call(payload))]
        if ptype == "custom_tool_call":
            name = payload.get("name", "?")
            detail = Agent._truncate(Agent._redact_sensitive(str(payload.get("input", ""))))
            return [("AGENT", "[TOOL: %s %s]" % (name, detail))]
        if ptype == "function_call_output":
            output = payload.get("output", "")
            if not isinstance(output, str) or not output.strip():
                return []
            detail = Agent._truncate(Agent._redact_sensitive(output), 200)
            return [("AGENT", "[TOOL_OUTPUT: %s]" % detail)]
        return []

    def find_latest_transcript(self, cwd: str) -> str | None:
        """Scan the last few day-directories for a matching rollout.

        Returns the newest (by mtime) rollout whose first-line session_meta
        cwd equals ``cwd``, or None if no session from the last 3 days matches.
        """
        candidates = []
        for days_back in range(3):
            t = time.localtime(time.time() - days_back * 86400)
            # Join the date parts as components: a "%Y/%m/%d" literal would
            # embed forward slashes into a backslash path on Windows.
            day_dir = os.path.join(SESSIONS_ROOT, *time.strftime("%Y %m %d", t).split(" "))
            if not os.path.isdir(day_dir):
                continue
            for name in os.listdir(day_dir):
                if name.endswith(".jsonl"):
                    candidates.append(os.path.join(day_dir, name))
        for path in sorted(candidates, key=os.path.getmtime, reverse=True):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    meta = json.loads(f.readline())
                payload = meta.get("payload", {}) if isinstance(meta, dict) else {}
                if (
                    isinstance(meta, dict)
                    and meta.get("type") == "session_meta"
                    and isinstance(payload, dict)
                    and payload.get("cwd") == cwd
                ):
                    return path
            except (OSError, AttributeError, json.JSONDecodeError):
                continue
        return None

    def accepts_transcript_path(self, transcript_path: str, cwd: str) -> bool:
        """Accept hook transcripts only from Codex rollouts for this cwd."""
        real_root = os.path.realpath(SESSIONS_ROOT)
        real_path = os.path.realpath(transcript_path)
        try:
            inside = os.path.commonpath([real_root, real_path]) == real_root
        except ValueError:
            inside = False
        if not inside or not os.path.isfile(real_path) or not real_path.endswith(".jsonl"):
            return False
        try:
            with open(real_path, encoding="utf-8", errors="replace") as f:
                meta = json.loads(f.readline())
        except (OSError, json.JSONDecodeError):
            return False
        payload = meta.get("payload", {}) if isinstance(meta, dict) else {}
        return (
            isinstance(meta, dict)
            and meta.get("type") == "session_meta"
            and isinstance(payload, dict)
            and payload.get("cwd") == cwd
        )


def _message_texts(content, role: str) -> list[str]:
    texts: list[str] = []
    if not isinstance(content, list):
        return texts
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in ("input_text", "output_text", "text"):
            continue
        raw_text = block.get("text", "")
        if not isinstance(raw_text, str):
            continue
        text = raw_text.strip()
        if not text:
            continue
        if role == "user" and text.startswith(_INJECTED_PREFIXES):
            continue
        texts.append(text)
    return texts


def _format_function_call(payload: dict) -> str:
    name = payload.get("name", "?")
    try:
        args = json.loads(payload.get("arguments", "") or "{}")
    except json.JSONDecodeError:
        args = {}
    detail = ""
    if isinstance(args, dict):
        detail = (
            args.get("cmd")
            or args.get("command")
            or args.get("path")
            or args.get("file_path")
            or args.get("pattern")
            or ""
        )
        if isinstance(detail, list):
            detail = " ".join(str(p) for p in detail)
    if detail:
        detail = Agent._redact_sensitive(str(detail))
        return "[TOOL: %s `%s`]" % (name, Agent._truncate(detail))
    return "[TOOL: %s]" % name
