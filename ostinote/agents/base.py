"""Common agent-adapter interface."""

from __future__ import annotations

# A message is ("HUMAN" | "AGENT", text).
Message = tuple[str, str]


class Agent:
    """Base class for agent transcript adapters."""

    name = "base"

    def parse(self, transcript_path: str, skip_lines: int = 0) -> tuple[list[Message], int]:
        """Parse a transcript file.

        Args:
            transcript_path: Path to the agent's session transcript (JSONL).
            skip_lines: Raw line offset to resume from (incremental extraction).

        Returns:
            (messages, total_lines) — messages extracted after the offset and
            the file's current total raw line count (the next resume offset).
        """
        raise NotImplementedError

    def find_latest_transcript(self, cwd: str) -> str | None:
        """Best-effort: locate the most recent transcript for a project cwd.

        Used only for manual CLI invocations — hooks receive transcript_path
        on stdin and never need this.
        """
        raise NotImplementedError

    @staticmethod
    def _count_lines(path: str) -> int:
        count = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for _ in f:
                count += 1
        return count

    @staticmethod
    def _truncate(text: str, limit: int = 80) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"
