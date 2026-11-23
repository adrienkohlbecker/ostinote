"""Timezone-aware date/time helpers.

A configured IANA zone (``timezone`` config key) wins; otherwise the system
local zone is used. An empty/invalid zone never silently becomes UTC.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo


def get_tz(name: str) -> ZoneInfo | None:
    """Resolve an IANA zone name; empty or invalid names fall back to None (system local)."""
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def now(tz: ZoneInfo | None) -> datetime.datetime:
    """Return the current time in ``tz``, or naive system-local time when tz is None."""
    # tz=None -> naive local time, which is what we want for "system local".
    return datetime.datetime.now(tz)


def today_str(tz: ZoneInfo | None) -> str:
    """Return today's date as ``YYYY-MM-DD`` in the given zone."""
    return now(tz).strftime("%Y-%m-%d")


def clock(moment: datetime.datetime, time_format: str = "24h") -> str:
    """Format ``moment`` as ``HH:MM`` (24h) or ``H:MM AM/PM`` (12h)."""
    if time_format == "12h":
        return moment.strftime("%I:%M %p").lstrip("0")
    return moment.strftime("%H:%M")


def local(moment: datetime.datetime, tz: ZoneInfo | None) -> datetime.datetime:
    """Convert an aware ``moment`` to ``tz``, or to the system local zone when tz is None."""
    return moment.astimezone(tz)
