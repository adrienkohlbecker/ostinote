"""Summarizer-model wrapper.

Default engine is the Claude CLI with Haiku, sandboxed: no tools, one turn,
no session persistence, cwd in the system temp dir, prompt piped on stdin.
The full argv can be overridden via the ``summarizer.command`` config key,
so a different engine (e.g. ``codex exec``) can be swapped in as long as it
prints the response text (claude-style JSON is also understood).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

# Haiku pricing (USD per token), fallback when the CLI omits total_cost_usd.
_INPUT_PRICE = 0.80 / 1_000_000
_OUTPUT_PRICE = 4.00 / 1_000_000
_CACHE_PRICE = 0.08 / 1_000_000

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


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    cache: int = 0
    cost_usd: float = 0.0


@dataclass
class ModelResult:
    text: str
    tokens: TokenUsage
    is_skip: bool


def call_model(prompt: str, cfg: dict) -> ModelResult:
    """Run the summarizer and return its parsed response.

    Raises RuntimeError on timeout, non-zero exit, or unparseable output.
    """
    command = list(cfg["summarizer"]["command"] or DEFAULT_COMMAND)
    timeout = cfg["summarizer"]["timeout"]

    # Resolve the executable explicitly — on Windows, npm shims like
    # claude.cmd are not found by CreateProcess without PATHEXT resolution.
    resolved = shutil.which(command[0])
    if resolved:
        command[0] = resolved

    # Safety net for custom summarizer.command values: a claude invocation
    # must never persist its transcript into ~/.claude/projects/.
    if (
        os.path.basename(command[0]).startswith("claude")
        and "--no-session-persistence" not in command
    ):
        command.append("--no-session-persistence")

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
        raise RuntimeError(
            "summarizer exited %d: %s" % (result.returncode, result.stderr.strip()[:300])
        )

    return parse_response(result.stdout)


def parse_response(raw: str) -> ModelResult:
    """Parse summarizer stdout — claude JSON (dict or list) or plain text."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Plain-text engine: take stdout verbatim.
        return ModelResult(
            text=raw, tokens=TokenUsage(), is_skip=raw.upper().startswith("SKIP")
        )

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
                parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                if parts:
                    text = "\n".join(parts)
                    break
        tokens = _extract_tokens(data[-1] if data and isinstance(data[-1], dict) else {})
    elif isinstance(data, dict):
        text = data.get("result") or ""
        tokens = _extract_tokens(data)
    else:
        text, tokens = str(data), TokenUsage()

    return ModelResult(
        text=text, tokens=tokens, is_skip=text.strip().upper().startswith("SKIP")
    )


def _extract_tokens(data: dict) -> TokenUsage:
    usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
    tk_in = usage.get("input_tokens", 0) or data.get("input_tokens", 0) or 0
    tk_out = usage.get("output_tokens", 0) or data.get("output_tokens", 0) or 0
    tk_cache = (
        usage.get("cache_read_input_tokens", 0) or data.get("cache_read_input_tokens", 0) or 0
    )
    cost = data.get("total_cost_usd") or (
        (tk_in - tk_cache) * _INPUT_PRICE + tk_out * _OUTPUT_PRICE + tk_cache * _CACHE_PRICE
    )
    return TokenUsage(input=tk_in, output=tk_out, cache=tk_cache, cost_usd=cost)
