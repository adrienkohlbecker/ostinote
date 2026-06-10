"""Token/cost accounting, parsed back out of the daily pipeline logs.

``Env.log_tokens`` writes one line per model call into
``logs/memory-YYYY-MM-DD.log``; this module aggregates those lines per day.
Cost is only present when the summarizer engine reported it, so the cost
column can undercount — token counts are always complete. The window is
bounded by log rotation (30 days).
"""

from __future__ import annotations

import os
import re

_TOKEN_LINE = re.compile(r"\[(\w+)\] tokens: (\d+)\+(\d+)cache→(\d+)out(?: \(\$([0-9.]+)\))?")
_LOG_NAME = re.compile(r"^memory-(\d{4}-\d{2}-\d{2})\.log$")


def parse_log(path: str) -> dict:
    """Sum the token lines of one daily log."""
    totals = {"calls": 0, "input": 0, "cache": 0, "output": 0, "cost": 0.0}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = _TOKEN_LINE.search(line)
                if not m:
                    continue
                totals["calls"] += 1
                totals["input"] += int(m.group(2))
                totals["cache"] += int(m.group(3))
                totals["output"] += int(m.group(4))
                if m.group(5):
                    totals["cost"] += float(m.group(5))
    except OSError:
        pass
    return totals


def day_totals(logs_dir: str) -> list[tuple[str, dict]]:
    """(date, totals) per daily log with model calls, oldest first."""
    out = []
    try:
        names = sorted(os.listdir(logs_dir))
    except OSError:
        return out
    for name in names:
        m = _LOG_NAME.match(name)
        if not m:
            continue
        totals = parse_log(os.path.join(logs_dir, name))
        if totals["calls"]:
            out.append((m.group(1), totals))
    return out
