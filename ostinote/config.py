"""Configuration loading with user/project layering.

Two config files, project overriding user:

- ``~/.ostinote/config.json``        (user defaults, all projects)
- ``<project>/.ostinote/config.json`` (per-project overrides)
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

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

# The project layer (``<project>/.ostinote/config.json``) is attacker-controlled
# the moment a repo is cloned, yet a session's hooks load it automatically. A
# few keys are too dangerous to honor from there:
#   - ``summarizer.command`` is executed as a subprocess on every save, so a
#     cloned repo setting it is straightforward remote code execution. It is
#     stripped from the project layer outright (no legitimate per-repo use).
#   - ``data_dir`` decides where memory is written and what the Codex installer
#     grants sandbox write access to. It stays settable per-project (the in-repo
#     ``.ostinote`` workflow needs it) but is flagged so ``Env`` can reject a
#     value that escapes the repo or the user's ostinote home.
# User-level config is trusted and may set either freely.
_PROJECT_DENIED = (("summarizer", "command"),)
_PROJECT_GUARDED = (("data_dir",),)
_PROJECT_MINIMUMS = {
    ("cooldowns", "save_seconds"): DEFAULTS["cooldowns"]["save_seconds"],
    ("cooldowns", "compress_seconds"): DEFAULTS["cooldowns"]["compress_seconds"],
    ("thresholds", "min_human_messages"): DEFAULTS["thresholds"]["min_human_messages"],
    ("thresholds", "delta_lines_trigger"): DEFAULTS["thresholds"]["delta_lines_trigger"],
}
_PROJECT_MAXIMUMS = {
    ("summarizer", "timeout"): DEFAULTS["summarizer"]["timeout"],
}


@dataclass(frozen=True)
class ConfigIssue:
    """One config problem found while loading user or project settings.

    ``level`` is formatted like the doctor output levels (``FAIL`` or
    ``warn``). ``path`` is the config file involved, and ``key`` is the dotted
    setting path when the issue belongs to a specific option.
    """

    level: str
    path: str
    key: str
    message: str


@dataclass(frozen=True)
class ConfigReport:
    """Merged config plus diagnostics and project-guard metadata."""

    config: dict
    guarded: set[tuple[str, ...]]
    issues: list[ConfigIssue]


def _walk(cfg: dict, path: tuple) -> dict | None:
    """Return the parent dict holding ``path[-1]``.

    Returns None if any ancestor is missing or not a dict. Used to
    inspect/strip nested keys by path tuple.
    """
    section = cfg
    for key in path[:-1]:
        section = section.get(key) if isinstance(section, dict) else None
        if not isinstance(section, dict):
            return None
    return section if isinstance(section, dict) else None


def _strip_path(cfg: dict, path: tuple) -> None:
    """Delete the nested key at ``path`` from ``cfg`` in place, if present."""
    parent = _walk(cfg, path)
    if parent is not None:
        parent.pop(path[-1], None)


def _has_path(cfg: dict, path: tuple) -> bool:
    """Return True if the nested key at ``path`` is set in ``cfg``."""
    parent = _walk(cfg, path)
    return parent is not None and path[-1] in parent


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


def _reject_constant(value: str) -> None:
    """Reject JSON's non-standard NaN/Infinity extensions."""
    raise ValueError("invalid JSON constant: %s" % value)


def _object_no_duplicates(pairs: Sequence[tuple[str, object]]) -> dict:
    """Build an object dict while rejecting duplicate keys."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: %s" % key)
        out[key] = value
    return out


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=_object_no_duplicates, parse_constant=_reject_constant)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _read_layer(path: str, source: str, issues: list[ConfigIssue]) -> dict:
    """Read one config layer, recording parse/read problems for doctor."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=_object_no_duplicates, parse_constant=_reject_constant)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        issues.append(
            ConfigIssue(
                "FAIL",
                path,
                "",
                "%s config invalid JSON at line %d column %d: %s" % (source, e.lineno, e.colno, e.msg),
            )
        )
        return {}
    except (OSError, ValueError) as e:
        issues.append(ConfigIssue("FAIL", path, "", "%s config unreadable or invalid: %s" % (source, e)))
        return {}
    if not isinstance(data, dict):
        issues.append(ConfigIssue("FAIL", path, "", "%s config must be a JSON object" % source))
        return {}
    return data


def _dotted(path: tuple[str, ...]) -> str:
    """Return a human-readable dotted config key path."""
    return ".".join(path)


def _matches_type(value, default) -> bool:
    """Return whether ``value`` has the safe shape implied by ``default``."""
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int) and not isinstance(default, bool):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, list):
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    return isinstance(value, type(default))


def _validate_layer(cfg: dict, defaults: dict, source: str, file_path: str, issues: list[ConfigIssue], path=()) -> dict:
    """Drop settings whose shape or project trust policy would break runtime."""
    clean = {}
    for key, value in cfg.items():
        key_path = (*path, key)
        if source == "project" and key_path in _PROJECT_DENIED:
            issues.append(
                ConfigIssue(
                    "warn",
                    file_path,
                    _dotted(key_path),
                    "project config setting ignored because it can execute commands",
                )
            )
            continue
        default = defaults.get(key) if isinstance(defaults, dict) else None
        if isinstance(default, dict):
            if not isinstance(value, dict):
                issues.append(
                    ConfigIssue(
                        "FAIL",
                        file_path,
                        _dotted(key_path),
                        "expected object, using default value",
                    )
                )
                continue
            clean[key] = _validate_layer(value, default, source, file_path, issues, key_path)
            continue
        if key in defaults and not _matches_type(value, default):
            issues.append(
                ConfigIssue(
                    "FAIL",
                    file_path,
                    _dotted(key_path),
                    "expected %s, using default value" % type(default).__name__,
                )
            )
            continue
        if key_path == ("time_format",) and value not in ("24h", "12h"):
            issues.append(ConfigIssue("FAIL", file_path, "time_format", "expected 24h or 12h, using default value"))
            continue
        if isinstance(value, (int, float)) and key_path in _PROJECT_MINIMUMS and source == "project":
            minimum = _PROJECT_MINIMUMS[key_path]
            if value < minimum:
                issues.append(
                    ConfigIssue(
                        "warn",
                        file_path,
                        _dotted(key_path),
                        "project value %s is below safe default %s, using default value" % (value, minimum),
                    )
                )
                continue
        if isinstance(value, (int, float)) and key_path in _PROJECT_MAXIMUMS and source == "project":
            maximum = _PROJECT_MAXIMUMS[key_path]
            if value > maximum:
                issues.append(
                    ConfigIssue(
                        "warn",
                        file_path,
                        _dotted(key_path),
                        "project value %s is above safe default %s, using default value" % (value, maximum),
                    )
                )
                continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            issues.append(ConfigIssue("FAIL", file_path, _dotted(key_path), "expected non-negative value"))
            continue
        clean[key] = copy.deepcopy(value)
    return clean


def load(project_root: str) -> dict:
    """Return the merged config for a project.

    Layers DEFAULTS, then the user config, then the project config (each later
    layer overriding earlier ones; nested dicts merge key-by-key). Legacy
    claude-remember key names are normalized in every layer. This never raises:
    a missing, unreadable, or malformed config file loads as empty, so a broken
    file silently falls back to defaults (``ostinote doctor`` surfaces the
    breakage out loud). Untrusted project-layer keys are filtered per
    ``load_trusted``; use that variant when you also need to know which guarded
    keys the project tried to set.
    """
    cfg, _ = load_trusted(project_root)
    return cfg


def load_trusted(project_root: str) -> tuple[dict, set]:
    """Load layered config and report guarded keys the project layer set.

    Same merge as ``load``, but the project layer is treated as untrusted:
    ``_PROJECT_DENIED`` keys are dropped from it before merging, and the
    returned set names the ``_PROJECT_GUARDED`` key paths the project config
    tried to set (e.g. ``{("data_dir",)}``) so ``Env`` can validate them
    against the trusted default. The user layer is never filtered.
    """
    report = load_report(project_root)
    return report.config, report.guarded


def load_report(project_root: str) -> ConfigReport:
    """Load config layers and keep diagnostics for ``ostinote doctor``.

    Runtime callers still get forgiving behavior: invalid or unsafe values are
    ignored in favor of defaults or trusted user settings. The report preserves
    why a value was ignored so doctor can make those silent fallbacks visible.
    """
    issues: list[ConfigIssue] = []
    user = _normalize_legacy(_read_layer(USER_CONFIG_PATH, "user", issues))
    user = _validate_layer(user, DEFAULTS, "user", USER_CONFIG_PATH, issues)
    cfg = _merge(DEFAULTS, user)
    project_path = os.path.join(project_root, ".ostinote", "config.json")
    project = _normalize_legacy(_read_layer(project_path, "project", issues))
    project = _validate_layer(project, DEFAULTS, "project", project_path, issues)
    guarded = {path for path in _PROJECT_GUARDED if _has_path(project, path)}
    cfg = _merge(cfg, project)
    return ConfigReport(cfg, guarded, issues)
