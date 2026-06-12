"""Common agent-adapter interface."""

from __future__ import annotations

import json
import sys
from typing import TypeAlias

# A message is ("HUMAN" | "AGENT", text).
Message: TypeAlias = tuple[str, str]


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

        Malformed JSONL lines are skipped (transcripts can be truncated
        mid-write); the skip count is reported on stderr, which lands in
        background.log when the save runs as a spawned subprocess.
        """
        messages: list[Message] = []
        total = 0
        malformed = 0
        try:
            f = open(transcript_path, encoding="utf-8", errors="replace")
        except OSError:
            return messages, 0

        with f:
            for line_num, line in enumerate(f):
                total = line_num + 1
                if line_num < skip_lines:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                messages.extend(self._extract_messages(obj))

        if malformed:
            print(
                "ostinote: %s: skipped %d malformed transcript line(s)" % (transcript_path, malformed),
                file=sys.stderr,
            )
        return messages, total

    def _extract_messages(self, obj: dict) -> list[Message]:
        """Messages contributed by one parsed transcript line (often none)."""
        raise NotImplementedError

    def find_latest_transcript(self, cwd: str) -> str | None:
        """Best-effort: locate the most recent transcript for a project cwd.

        Used only for manual CLI invocations — hooks receive transcript_path
        on stdin and never need this.
        """
        raise NotImplementedError

    @staticmethod
    def _truncate(text: str, limit: int = 80) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"
