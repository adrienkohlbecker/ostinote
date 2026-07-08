"""Summarizer-model wrapper.

Two built-in engines, both sandboxed with cwd in the system temp dir and the
prompt piped on stdin: the Claude CLI with Haiku (no tools, one turn, no
session persistence) and the Codex CLI (read-only sandbox, ephemeral, low
reasoning effort). The ``summarizer.engine`` config key picks one explicitly;
the default auto-detects, preferring claude and falling back to codex for
codex-only setups. ``summarizer.model`` overrides the built-in command's
model. The full argv can also be overridden via the
``summarizer.command`` config key, as long as the command prints the response
text (claude-style JSON and codex ``--json`` event lines are also understood).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from .agents.base import Agent

DEFAULT_COMMAND = [
    "claude",
    "-p",
    "--model",
    "haiku",
    "--allowedTools",
    "",
    "--max-turns",
    "1",
    "--output-format",
    "json",
    "--no-session-persistence",
    "--mcp-config",
    '{"mcpServers":{}}',
    "--strict-mcp-config",
]

DEFAULT_COMMAND_CODEX = [
    "codex",
    "exec",
    "--json",
    "--sandbox",
    "read-only",
    "--skip-git-repo-check",
    "--ephemeral",
    "--color",
    "never",
    "-c",
    "model_reasoning_effort=low",
    "-",
]


@dataclass
class TokenUsage:
    """Token counts and cost reported by the summarizer engine; all zero when the engine reports none."""

    input: int = 0
    output: int = 0
    cache: int = 0
    cost_usd: float = 0.0


@dataclass
class ModelResult:
    """Parsed summarizer response: the text, its token usage, and whether the model answered SKIP."""

    text: str
    tokens: TokenUsage
    is_skip: bool


def resolve_command(cfg: dict) -> list[str]:
    """Return the summarizer argv this config resolves to.

    An explicit ``summarizer.command`` always wins (``summarizer.model`` is
    ignored with it — put the model in the argv). Otherwise
    ``summarizer.engine`` picks a built-in command; the auto default prefers
    claude and falls back to codex, so codex-only machines work without any
    config. When neither CLI is installed the claude command is returned so
    the eventual "not found" error names the primary engine.
    ``summarizer.model`` then overrides the built-in command's model — the pin
    lives in user config rather than here, because codex model slugs are
    versioned and rotate out (there is no stable family alias like claude's
    ``haiku``).
    """
    command = cfg["summarizer"]["command"]
    if command:
        return list(command)
    engine = cfg["summarizer"].get("engine", "")
    if engine == "claude":
        command = list(DEFAULT_COMMAND)
    elif engine == "codex":
        command = list(DEFAULT_COMMAND_CODEX)
    elif not shutil.which("claude") and shutil.which("codex"):
        command = list(DEFAULT_COMMAND_CODEX)
    else:
        command = list(DEFAULT_COMMAND)
    model = cfg["summarizer"].get("model", "")
    if model:
        command = _override_model(command, model)
    return command


def _override_model(command: list[str], model: str) -> list[str]:
    """Return ``command`` with its model replaced (or added) as ``model``."""
    command = list(command)
    for flag in ("--model", "-m"):
        if flag in command:
            value_at = command.index(flag) + 1
            if value_at < len(command):
                command[value_at] = model
                return command
    # No model flag in the command: insert one before a trailing stdin "-"
    # positional so the option/positional split stays unambiguous.
    insert_at = len(command) - 1 if command and command[-1] == "-" else len(command)
    command[insert_at:insert_at] = ["--model", model]
    return command


def call_model(prompt: str, cfg: dict) -> ModelResult:
    """Run the summarizer and return its parsed response.

    Raises RuntimeError on timeout, non-zero exit, or unparseable output.
    """
    command = resolve_command(cfg)
    timeout = cfg["summarizer"]["timeout"]

    # Resolve the executable explicitly — on Windows, npm shims like
    # claude.cmd are not found by CreateProcess without PATHEXT resolution.
    resolved = shutil.which(command[0])
    if resolved:
        command[0] = resolved

    # Safety net for custom summarizer.command values: the summarizer prompt
    # contains extracted transcript content, so a claude invocation must never
    # persist it into ~/.claude/projects/, and a codex invocation must never
    # persist it into ~/.codex/sessions/.
    executable = os.path.basename(command[0])
    if executable.startswith("claude") and "--no-session-persistence" not in command:
        command.append("--no-session-persistence")
    if executable.startswith("codex") and "--ephemeral" not in command:
        command.append("--ephemeral")

    # CLAUDECODE blocks nested claude sessions when run from a Claude hook.
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("summarizer timed out after %ss" % timeout) from None
    except OSError as e:
        raise RuntimeError("summarizer failed to start: %s" % e) from e

    if result.returncode != 0:
        stderr = Agent._redact_sensitive(result.stderr.strip())[:300]
        raise RuntimeError("summarizer exited %d: %s" % (result.returncode, stderr))

    return parse_response(result.stdout)


def parse_response(raw: str) -> ModelResult:
    """Parse summarizer stdout — claude JSON (dict or list), codex ``--json`` event lines, or plain text."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        jsonl = _parse_jsonl(raw)
        if jsonl is not None:
            return jsonl
        # Plain-text engine: take stdout verbatim.
        return ModelResult(text=raw, tokens=TokenUsage(), is_skip=raw.upper().startswith("SKIP"))

    if isinstance(data, list):
        # claude CLI v2.1.86+ may emit a list of message objects.
        text = ""
        for msg in reversed(data):
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "result":
                text = msg.get("result", "") or ""
                break
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                text = content
                break
            if isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                if parts:
                    text = "\n".join(parts)
                    break
        tokens = _extract_tokens(data[-1] if data and isinstance(data[-1], dict) else {})
    elif isinstance(data, dict):
        text = data.get("result") or ""
        tokens = _extract_tokens(data)
    else:
        text, tokens = str(data), TokenUsage()

    return ModelResult(text=text, tokens=tokens, is_skip=text.strip().upper().startswith("SKIP"))


def _parse_jsonl(raw: str) -> ModelResult | None:
    """Parse codex ``exec --json`` output: one event object per line.

    The response text comes from the last ``item.completed`` event holding an
    ``agent_message`` item; token usage from the ``turn.completed`` event.
    Returns None when the input is not a stream of JSON objects, so
    ``parse_response`` can fall back to treating stdout as plain text.
    """
    text = ""
    tokens = TokenUsage()
    saw_event = False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(event, dict):
            return None
        saw_event = True
        item = event.get("item")
        if event.get("type") == "item.completed" and isinstance(item, dict) and item.get("type") == "agent_message":
            if isinstance(item.get("text"), str):
                text = item["text"]
        usage = event.get("usage")
        if event.get("type") == "turn.completed" and isinstance(usage, dict):
            tokens = TokenUsage(
                input=usage.get("input_tokens", 0) or 0,
                output=usage.get("output_tokens", 0) or 0,
                cache=usage.get("cached_input_tokens", 0) or 0,
            )
    if not saw_event:
        return None
    return ModelResult(text=text, tokens=tokens, is_skip=text.strip().upper().startswith("SKIP"))


def _extract_tokens(data: dict) -> TokenUsage:
    usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
    tk_in = usage.get("input_tokens", 0) or data.get("input_tokens", 0) or 0
    tk_out = usage.get("output_tokens", 0) or data.get("output_tokens", 0) or 0
    tk_cache = usage.get("cache_read_input_tokens", 0) or data.get("cache_read_input_tokens", 0) or 0
    # Cost only when the engine reports it — no fallback price table to rot.
    cost = data.get("total_cost_usd") or 0.0
    return TokenUsage(input=tk_in, output=tk_out, cache=tk_cache, cost_usd=cost)
