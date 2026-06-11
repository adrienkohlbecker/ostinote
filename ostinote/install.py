"""Hook registration for the supported agents.

Both agents use the same hooks JSON schema; only the file location differs:

- Claude Code: ``hooks`` key inside ``settings.json``
  (``~/.claude/settings.json`` or ``<project>/.claude/settings.json``)
- Codex: dedicated ``hooks.json``
  (``~/.codex/hooks.json`` or ``<project>/.codex/hooks.json``)

Entries managed by this tool are recognized by their command string
(contains ``ostinote`` and ``--agent``), making install/uninstall idempotent
and safe alongside user-defined hooks.

Also installs the ostinote core-memory command as a skill for both agents —
the same ``SKILL.md``, invoked as ``/ostinote`` in Claude Code and
``$ostinote`` in Codex (Codex deprecated custom prompts in favor of skills).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import tomllib

from .env import Env
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
# Where each agent looks for skills, and how the user invokes one.
_SKILLS = {
    "claude": (".claude", "/ostinote"),
    "codex": (".agents", "$ostinote"),
}
_MANAGED_SUBCOMMANDS = set(_EVENTS.values())
for _agent_events in _AGENT_EVENTS.values():
    _MANAGED_SUBCOMMANDS.update(_agent_events.values())
_WINDOWS_CMD_METACHARS = set("&|<>^")


class _ConfigError(RuntimeError):
    pass


def _events_for(agent: str) -> dict:
    return {**_EVENTS, **_AGENT_EVENTS.get(agent, {})}


def _format_command(argv: list[str]) -> str:
    if os.name == "nt":
        bad = [part for part in argv if any(char in part for char in _WINDOWS_CMD_METACHARS)]
        if bad:
            raise _ConfigError(
                "cannot safely render Windows hook command part with cmd.exe metacharacter: %s"
                % bad[0]
            )
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _command_str(subcommand: str, agent: str) -> str:
    return _format_command(self_command() + ["hook", subcommand, "--agent", agent])


def _is_ours(command: str) -> bool:
    try:
        parts = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return False
    for idx, part in enumerate(parts):
        if part != "hook":
            continue
        suffix = parts[idx:]
        if (
            len(suffix) == 4
            and suffix[1] in _MANAGED_SUBCOMMANDS
            and suffix[2] == "--agent"
            and suffix[3] in _SKILLS
            and _is_ostinote_invocation(parts[:idx])
        ):
            return True
    return False


def _is_ostinote_invocation(parts: list[str]) -> bool:
    if len(parts) == 1:
        name = os.path.basename(parts[0]).lower()
        stem, _ext = os.path.splitext(name)
        return stem == "ostinote"
    return len(parts) >= 3 and parts[-2:] == ["-m", "ostinote"]


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise _ConfigError(
            "invalid JSON in %s at line %d column %d: %s" % (path, e.lineno, e.colno, e.msg)
        ) from e
    except OSError as e:
        raise _ConfigError("cannot read %s: %s" % (path, e)) from e
    if not isinstance(data, dict):
        raise _ConfigError("invalid JSON in %s: expected an object" % path)
    return data


def _write_json(path: str, data: dict) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _write_text_atomic(path, text)


def _write_text_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".ostinote-", suffix=".tmp", dir=os.path.dirname(path), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _update_hooks(settings: dict, agent: str, remove_only: bool = False) -> dict:
    hooks = settings.get("hooks")
    if hooks is None:
        hooks = {}
        settings["hooks"] = hooks
    elif not isinstance(hooks, dict):
        raise _ConfigError("invalid hooks schema: expected hooks to be an object")
    for event, subcommand in _events_for(agent).items():
        kept_groups = _strip_managed_hooks(hooks.get(event, []))
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


def _strip_managed_hooks(groups) -> list:
    if not isinstance(groups, list):
        return [groups]
    kept_groups = []
    for group in groups:
        if not isinstance(group, dict):
            kept_groups.append(group)
            continue
        group_hooks = group.get("hooks")
        if not isinstance(group_hooks, list):
            kept_groups.append(group)
            continue
        inner = []
        for hook in group_hooks:
            command = hook.get("command") if isinstance(hook, dict) else None
            if isinstance(command, str) and _is_ours(command):
                continue
            inner.append(hook)
        if inner:
            group = dict(group)
            group["hooks"] = inner
            kept_groups.append(group)
    return kept_groups


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
    if not remove or os.path.exists(path):
        try:
            settings = _read_json(path)
            settings = _update_hooks(settings, agent, remove_only=remove)
        except _ConfigError as e:
            report.append("ERROR: %s" % e)
            return report
        _write_json(path, settings)
        report.append(
            "%s hooks %s: %s" % (agent, "removed from" if remove else "registered in", path)
        )

    # The ostinote core-memory command, as a skill (same file for both agents).
    base, invoke = _SKILLS[agent]
    skill_dir = (
        os.path.join(os.path.expanduser("~"), base, "skills", "ostinote")
        if scope == "user"
        else os.path.join(project_root, base, "skills", "ostinote")
    )
    target = os.path.join(skill_dir, "SKILL.md")
    if remove:
        if os.path.exists(target):
            shutil.rmtree(skill_dir, ignore_errors=True)
            report.append("%s %s skill removed: %s" % (agent, invoke, skill_dir))
    else:
        os.makedirs(skill_dir, exist_ok=True)
        shutil.copyfile(os.path.join(ASSETS_DIR, "SKILL.md"), target)
        report.append("%s %s skill installed: %s" % (agent, invoke, target))
        if agent == "codex":
            try:
                report.append(_ensure_codex_writable_root(project_root))
            except (OSError, _ConfigError) as e:
                report.append("ERROR: could not update Codex writable roots: %s" % e)

    report.extend(_warnings(agent, remove))
    return report


def _ensure_codex_writable_root(project_root: str) -> str:
    config_path = os.path.expanduser("~/.codex/config.toml")
    root = _home_relative(Env(project_root).data_dir)
    config = _read_toml(config_path)
    sandbox = config.get("sandbox_workspace_write")
    if sandbox is None:
        sandbox = {}
        config["sandbox_workspace_write"] = sandbox
    elif not isinstance(sandbox, dict):
        raise _ConfigError(
            "invalid TOML in %s: sandbox_workspace_write must be a table" % config_path
        )

    roots = sandbox.get("writable_roots")
    if roots is None:
        roots = []
        sandbox["writable_roots"] = roots
    elif not isinstance(roots, list) or not all(isinstance(item, str) for item in roots):
        raise _ConfigError(
            "invalid TOML in %s: sandbox_workspace_write.writable_roots "
            "must be an array of strings" % config_path
        )

    if _root_list_has_value(roots, root):
        return "codex writable root already present: %s" % root

    roots.append(root)
    _write_text_atomic(config_path, _format_toml(config))
    return "codex writable root added: %s" % root


def _read_toml(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise _ConfigError("invalid TOML in %s: %s" % (path, e)) from e


def _format_toml(data: dict) -> str:
    lines: list[str] = []
    _append_toml_lines(lines, data, ())
    return "\n".join(lines).rstrip() + "\n"


def _append_toml_lines(lines: list[str], table: dict, path: tuple[str, ...]) -> None:
    scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
    subtables = [(key, value) for key, value in table.items() if isinstance(value, dict)]
    if path:
        if lines:
            lines.append("")
        lines.append("[%s]" % ".".join(_toml_key(part) for part in path))
    for key, value in scalars:
        lines.extend(_format_toml_assignment(key, value))
    for key, value in subtables:
        _append_toml_lines(lines, value, (*path, key))


def _format_toml_assignment(key: str, value) -> list[str]:
    if isinstance(value, list):
        lines = ["%s = [" % _toml_key(key)]
        for item in value:
            lines.append("  %s," % _toml_value(item))
        lines.append("]")
        return lines
    return ["%s = %s" % (_toml_key(key), _toml_value(value))]


def _toml_value(value) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value).lower()
    if isinstance(value, list):
        return "[%s]" % ", ".join(_toml_value(item) for item in value)
    raise _ConfigError("cannot write unsupported TOML value: %r" % (value,))


def _toml_key(key: str) -> str:
    if key and all(char.isalnum() or char in "-_" for char in key):
        return key
    return json.dumps(key, ensure_ascii=False)


def _root_list_has_value(values: list[str], value: str) -> bool:
    target = _normalized_root(value)
    for existing in values:
        if existing == value or _normalized_root(existing) == target:
            return True
    return False


def _normalized_root(value: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(value)))


def _home_relative(path: str) -> str:
    home = os.path.abspath(os.path.expanduser("~"))
    absolute = os.path.abspath(path)
    try:
        if os.path.commonpath([home, absolute]) == home:
            rel = os.path.relpath(absolute, home)
            return "~" if rel == "." else os.path.join("~", rel)
    except ValueError:
        pass
    return path


def _warnings(agent: str, remove: bool) -> list[str]:
    warnings = []
    if remove:
        return warnings
    if agent == "claude":
        try:
            settings = _read_json(os.path.expanduser("~/.claude/settings.json"))
        except _ConfigError as e:
            return ["WARNING: could not inspect legacy Claude plugin setting: %s" % e]
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
