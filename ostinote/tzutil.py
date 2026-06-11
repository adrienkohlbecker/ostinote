"""Timezone-aware date/time helpers.

A configured IANA zone (``timezone`` config key) wins; otherwise the system
local zone is used. An empty/invalid zone never silently becomes UTC.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo


def get_tz(name: str) -> ZoneInfo | None:
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def now(tz: ZoneInfo | None) -> datetime.datetime:
    # tz=None -> naive local time, which is what we want for "system local".
    return datetime.datetime.now(tz)


def today_str(tz: ZoneInfo | None) -> str:
    return now(tz).strftime("%Y-%m-%d")


def time_str(tz: ZoneInfo | None, time_format: str = "24h") -> str:
    if time_format == "12h":
        s = now(tz).strftime("%I:%M %p")
        return s.lstrip("0")
    return now(tz).strftime("%H:%M")
