"""Hook registration for the supported agents.

Both agents use the same hooks JSON schema; only the file location differs:

- Claude Code: ``hooks`` key inside ``settings.json``
  (``~/.claude/settings.json`` or ``<project>/.claude/settings.json``)
- Codex: dedicated ``hooks.json``
  (``~/.codex/hooks.json`` or ``<project>/.codex/hooks.json``)

Entries managed by this tool are recognized by their command string
(contains ``ostinote`` and ``--agent``), making install/uninstall idempotent
and safe alongside user-defined hooks.

Also installs the ``/ostinote`` handoff command where it can match the
requested scope: a skill for Claude Code, and a user-scoped custom prompt for
Codex.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil

from .hooks import self_command

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_EVENTS = {
    "SessionStart": "session-start",
    "PostToolUse": "post-tool",
}
# Codex has no session-exit hook event, so its turn-scoped Stop stands in:
# every turn end is treated as a potential session end (cheap when nothing
# new happened).
_AGENT_EVENTS = {
    "claude": {"SessionEnd": "session-end"},
    "codex": {"Stop": "session-end"},
}


def _events_for(agent: str) -> dict:
    return {**_EVENTS, **_AGENT_EVENTS.get(agent, {})}


def _quote(part: str) -> str:
    if os.name == "nt":
        # Hook commands run through cmd.exe on Windows — POSIX single quotes
        # would be passed literally.
        return '"%s"' % part if (" " in part or "\t" in part) else part
    return shlex.quote(part)


def _command_str(subcommand: str, agent: str) -> str:
    base = " ".join(_quote(part) for part in self_command())
    return "%s hook %s --agent %s" % (base, subcommand, agent)


def _is_ours(command: str) -> bool:
    return "ostinote" in command and "--agent" in command


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _update_hooks(settings: dict, agent: str, remove_only: bool = False) -> dict:
    hooks = settings.setdefault("hooks", {})
    for event, subcommand in _events_for(agent).items():
        groups = hooks.get(event, [])
        # Strip our hooks from every matcher group, drop emptied groups.
        kept_groups = []
        for group in groups:
            inner = [h for h in group.get("hooks", []) if not _is_ours(h.get("command", ""))]
            if inner:
                group = dict(group)
                group["hooks"] = inner
                kept_groups.append(group)
        if not remove_only:
            kept_groups.append(
                {"hooks": [{"type": "command", "command": _command_str(subcommand, agent)}]}
            )
        if kept_groups:
            hooks[event] = kept_groups
        else:
            hooks.pop(event, None)
    if not hooks:
        settings.pop("hooks", None)
    return settings


def _hooks_file_for(agent: str, scope: str, project_root: str) -> str:
    if agent == "claude":
        base = (
            os.path.expanduser("~/.claude")
            if scope == "user"
            else os.path.join(project_root, ".claude")
        )
        return os.path.join(base, "settings.json")
    base = (
        os.path.expanduser("~/.codex")
        if scope == "user"
        else os.path.join(project_root, ".codex")
    )
    return os.path.join(base, "hooks.json")


def install(agent: str, scope: str, project_root: str, remove: bool = False) -> list[str]:
    """(Un)register hooks and the /ostinote command. Returns report lines."""
    report = []
    path = _hooks_file_for(agent, scope, project_root)
    settings = _read_json(path)
    settings = _update_hooks(settings, agent, remove_only=remove)
    _write_json(path, settings)
    report.append(
        "%s hooks %s: %s" % (agent, "removed from" if remove else "registered in", path)
    )

    # /ostinote handoff command
    if agent == "claude":
        skill_dir = (
            os.path.expanduser("~/.claude/skills/ostinote")
            if scope == "user"
            else os.path.join(project_root, ".claude", "skills", "ostinote")
        )
        target = os.path.join(skill_dir, "SKILL.md")
        if remove:
            if os.path.exists(target):
                shutil.rmtree(skill_dir, ignore_errors=True)
                report.append("claude /ostinote skill removed: %s" % skill_dir)
        else:
            os.makedirs(skill_dir, exist_ok=True)
            shutil.copyfile(os.path.join(ASSETS_DIR, "SKILL.md"), target)
            report.append("claude /ostinote skill installed: %s" % target)
    else:
        if scope != "user":
            if remove:
                report.append("codex /ostinote prompt is user-scoped; left unchanged")
            else:
                report.append(
                    "codex /ostinote prompt is user-scoped; run "
                    "`ostinote install codex --user` to install it"
                )
            report.extend(_warnings(agent, remove))
            return report
        target = os.path.expanduser("~/.codex/prompts/ostinote.md")
        if remove:
            if os.path.exists(target):
                os.remove(target)
                report.append("codex /ostinote prompt removed: %s" % target)
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(os.path.join(ASSETS_DIR, "codex-prompt.md"), target)
            report.append("codex /ostinote prompt installed: %s" % target)

    report.extend(_warnings(agent, remove))
    return report


def _warnings(agent: str, remove: bool) -> list[str]:
    warnings = []
    if remove:
        return warnings
    if agent == "claude":
        settings = _read_json(os.path.expanduser("~/.claude/settings.json"))
        enabled = settings.get("enabledPlugins", {})
        for key, value in enabled.items():
            if key.startswith("remember@") and value:
                warnings.append(
                    "WARNING: the '%s' plugin is still enabled in "
                    "~/.claude/settings.json — disable it (/plugin) or you "
                    "will get double saves and double memory injection." % key
                )
    else:
        warnings.append(
            "NOTE: Codex asks you to trust new hooks on first use — accept "
            "the prompt in your next codex session."
        )
    return warnings
