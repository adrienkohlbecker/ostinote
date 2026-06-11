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
import re
import shlex
import shutil
import subprocess
import tempfile

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
            except OSError as e:
                report.append("ERROR: could not update Codex writable roots: %s" % e)

    report.extend(_warnings(agent, remove))
    return report


def _ensure_codex_writable_root(project_root: str) -> str:
    config_path = os.path.expanduser("~/.codex/config.toml")
    root = _home_relative(Env(project_root).data_dir)
    try:
        with open(config_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    new_content, changed = _append_toml_array_value(
        content, "sandbox_workspace_write", "writable_roots", root
    )
    if changed:
        _write_text_atomic(config_path, new_content)
        return "codex writable root added: %s" % root
    return "codex writable root already present: %s" % root


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


def _append_toml_array_value(
    content: str, section: str, key: str, value: str
) -> tuple[str, bool]:
    values = _toml_array_values(content, section, key)
    if _toml_has_value(values, value):
        return content, False
    values.append(value)
    lines = content.splitlines(True)
    start, end = _toml_section_bounds(lines, section)
    array = _format_toml_array(key, values, "")
    if start is None:
        prefix = content
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if prefix:
            prefix += "\n"
        return prefix + "[%s]\n%s" % (section, array), True
    key_idx = _toml_key_index(lines, start + 1, end, key)
    if key_idx is None:
        new_lines = lines[:end] + [array] + lines[end:]
        return "".join(new_lines), True
    close_idx = _toml_array_close_index(lines, key_idx)
    indent = lines[key_idx][: len(lines[key_idx]) - len(lines[key_idx].lstrip())]
    array = _format_toml_array(key, values, indent)
    return "".join(lines[:key_idx] + [array] + lines[close_idx + 1 :]), True


def _toml_has_value(values: list[str], value: str) -> bool:
    target = _normalized_root(value)
    for existing in values:
        if existing == value or _normalized_root(existing) == target:
            return True
    return False


def _normalized_root(value: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(value)))


def _toml_array_values(content: str, section: str, key: str) -> list[str]:
    try:
        import tomllib  # type: ignore[import-not-found]

        table = tomllib.loads(content).get(section, {})
        values = table.get(key, []) if isinstance(table, dict) else []
        if isinstance(values, list):
            return [value for value in values if isinstance(value, str)]
    except Exception:
        pass
    lines = content.splitlines(True)
    start, end = _toml_section_bounds(lines, section)
    if start is None:
        return []
    key_idx = _toml_key_index(lines, start + 1, end, key)
    if key_idx is None:
        return []
    close_idx = _toml_array_close_index(lines, key_idx)
    text = "".join(lines[key_idx : close_idx + 1])
    values = []
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"|\'([^\']*)\'', text):
        token = match.group(0)
        if token.startswith("'"):
            values.append(token[1:-1])
            continue
        try:
            values.append(json.loads(token))
        except json.JSONDecodeError:
            pass
    return values


def _toml_section_bounds(lines: list[str], section: str):
    start = None
    for idx, line in enumerate(lines):
        name = _toml_section_name(line)
        if name is None:
            continue
        if start is not None:
            return start, idx
        if name == section:
            start = idx
    return start, len(lines)


def _toml_section_name(line: str):
    head = line.split("#", 1)[0].strip()
    if head.startswith("[[") or not (head.startswith("[") and head.endswith("]")):
        return None
    return head[1:-1].strip()


def _toml_key_index(lines: list[str], start: int, end: int, key: str):
    for idx in range(start, end):
        if lines[idx].split("#", 1)[0].split("=", 1)[0].strip() == key:
            return idx
    return None


def _toml_array_close_index(lines: list[str], start: int) -> int:
    depth = 0
    for idx in range(start, len(lines)):
        depth += lines[idx].count("[") - lines[idx].count("]")
        if depth <= 0 and "[" in lines[start]:
            return idx
    return start


def _format_toml_array(key: str, values: list[str], indent: str) -> str:
    lines = ["%s%s = [\n" % (indent, key)]
    for value in values:
        lines.append("%s  %s,\n" % (indent, json.dumps(value, ensure_ascii=False)))
    lines.append("%s]\n" % indent)
    return "".join(lines)


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
