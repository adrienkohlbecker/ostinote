"""Configuration loading with user/project layering.

Two config files, project overriding user:

- ``~/.ostinote/config.json``        (user defaults, all projects)
- ``<project>/.ostinote/config.json`` (per-project overrides)
"""

from __future__ import annotations

import copy
import json
import os

DEFAULTS: dict = {
    # Where memory files live. Absolute paths and ~ are honored; a relative
    # value is resolved against the project root. The "{slug}" placeholder
    # expands to the project path with non-alphanumerics dashed, so the
    # default keeps every project's memory in your home directory (out of the
    # repo, so it never shows up in diffs or the agent's review UI) while
    # staying per-project. Set "data_dir": ".ostinote" to store it in-repo.
    "data_dir": "~/.ostinote/projects/{slug}",
    # IANA timezone name for timestamps and daily file boundaries.
    # Empty = system local zone (never silently UTC).
    "timezone": "",
    # "24h" or "12h" — timestamp format in memory entries and logs.
    "time_format": "24h",
    # Sessions started in a git worktree share the main checkout's memory.
    "share_worktrees": True,
    "debug": False,
    "cooldowns": {
        # Minimum seconds between saves of the same session.
        "save_seconds": 120,
        # Minimum seconds between now.md -> today-*.md compressions.
        "compress_seconds": 3600,
    },
    "thresholds": {
        # Minimum human messages in the extract before a save happens.
        "min_human_messages": 3,
        # New transcript lines since last save that trigger an auto-save.
        "delta_lines_trigger": 50,
    },
    "features": {
        "hourly_compression": True,
        "recovery": True,
        "consolidation": True,
    },
    "summarizer": {
        # Override the summarizer invocation entirely (list of argv strings).
        # The prompt is always piped on stdin. Empty = built-in claude/haiku.
        "command": [],
        "timeout": 120,
    },
}

USER_CONFIG_PATH = os.path.expanduser("~/.ostinote/config.json")

# claude-remember's names for settings ostinote renamed, accepted in any
# config layer so old config files keep working as-is. A file that also sets
# the new name keeps the new name's value.
LEGACY_KEYS = {
    ("cooldowns", "ndc_seconds"): ("cooldowns", "compress_seconds"),
    ("features", "ndc_compression"): ("features", "hourly_compression"),
}


def _normalize_legacy(cfg: dict) -> dict:
    for (old_section, old_key), (new_section, new_key) in LEGACY_KEYS.items():
        section = cfg.get(old_section)
        if isinstance(section, dict) and old_key in section:
            value = section.pop(old_key)
            new = cfg.setdefault(new_section, {})
            if isinstance(new, dict):
                new.setdefault(new_key, value)
    return cfg


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load(project_root: str) -> dict:
    cfg = _merge(DEFAULTS, _normalize_legacy(_read_json(USER_CONFIG_PATH)))
    project = _read_json(os.path.join(project_root, ".ostinote", "config.json"))
    cfg = _merge(cfg, _normalize_legacy(project))
    return cfg
