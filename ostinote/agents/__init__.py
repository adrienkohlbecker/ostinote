"""Agent adapters — one per supported coding agent.

Each adapter knows how to parse its agent's transcript format into a common
``(role, text)`` message stream and how to locate transcripts on disk.
Adding a new agent means adding one module here and an install template.
"""

from __future__ import annotations

from .base import Agent
from .claude import ClaudeAgent
from .codex import CodexAgent

_AGENTS = {
    "claude": ClaudeAgent(),
    "codex": CodexAgent(),
}


def get_agent(name: str) -> Agent:
    try:
        return _AGENTS[name]
    except KeyError:
        raise ValueError(
            "unknown agent %r (known: %s)" % (name, ", ".join(sorted(_AGENTS)))
        ) from None


def agent_names() -> list:
    return sorted(_AGENTS)
