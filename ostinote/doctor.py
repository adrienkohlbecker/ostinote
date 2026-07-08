"""``ostinote doctor``: loud checks for a pipeline that fails silently.

Hooks swallow every error by design (they must never break the agent), so
when something is misconfigured the only symptom is memory quietly not
accumulating. This command checks each link in the chain and says so out
loud. ``--live`` additionally runs one real (paid) summarizer call.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

from . import config as config_mod
from . import install as install_mod
from . import summarize
from .agents import agent_names
from .env import HOOK_ERRORS_PATH, Env
from .install import _events_for, registered_events
from .pipeline import staging_files
from .state import _pid_alive, all_states


def run(env: Env, live: bool = False) -> int:
    """Print a check report; exit code 1 if anything failed."""
    results: list[tuple[str, str]] = []

    _check_config(env, results)
    _check_data_dir(env, results)
    for agent in agent_names():
        registered_scopes = _check_hooks(agent, env, results)
        _check_skills(agent, registered_scopes, env, results)
        if agent == "codex":
            _check_codex_writable_roots(registered_scopes, env, results)
    _check_summarizer(env, results, live)
    _check_locks(env, results)
    _check_sessions(env, results)
    _check_hook_errors(results)

    failed = False
    for level, message in results:
        print(" %-4s  %s" % (level, message))
        failed = failed or level == "FAIL"
    return 1 if failed else 0


def _check_config(env: Env, results: list) -> None:
    report = config_mod.load_report(env.project_root)
    paths = [
        config_mod.USER_CONFIG_PATH,
        os.path.join(env.project_root, ".ostinote", "config.json"),
    ]
    issue_paths = {issue.path for issue in report.issues}
    for path in paths:
        if not os.path.exists(path):
            continue
        if path not in issue_paths:
            results.append(("ok", "config readable: %s" % path))
    for issue in report.issues:
        key = " (%s)" % issue.key if issue.key else ""
        results.append((issue.level, "config issue%s: %s — %s" % (key, issue.path, issue.message)))
    if ("data_dir",) in report.guarded:
        from . import env as env_mod

        requested = env_mod._resolve_data_dir(report.config, env.project_root)
        safe = env_mod.safe_data_dir(report.config, env.project_root, report.guarded)
        if requested != safe:
            results.append(
                (
                    "FAIL",
                    "project data_dir ignored because it escapes this project memory: %s -> %s" % (requested, safe),
                )
            )
        else:
            results.append(("warn", "project data_dir accepted after containment check: %s" % safe))


def _check_data_dir(env: Env, results: list) -> None:
    probe = ""
    try:
        env.ensure_dirs()
        fd, probe = tempfile.mkstemp(prefix=".doctor-", dir=env.state_dir, text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        results.append(("ok", "data dir writable: %s" % env.data_dir))
    except OSError as e:
        if probe:
            try:
                os.remove(probe)
            except OSError:
                pass
        results.append(("FAIL", "data dir not writable: %s — %s" % (env.data_dir, e)))


def _check_hooks(agent: str, env: Env, results: list) -> set[str]:
    expected = set(_events_for(agent))
    registered: set = set()
    registered_scopes: set[str] = set()
    for scope in ("user", "project"):
        for hooks_file in install_mod.hook_files_for(agent, scope, env.project_root):
            if not os.path.exists(hooks_file):
                continue
            try:
                with open(hooks_file, encoding="utf-8") as f:
                    settings = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                results.append(("FAIL", "%s: hook config unreadable: %s — %s" % (agent, hooks_file, e)))
                continue
            if not isinstance(settings, dict):
                results.append(("FAIL", "%s: hook config is not a JSON object: %s" % (agent, hooks_file)))
                continue
            any_agent_events = registered_events(settings)
            events = registered_events(settings, agent)
            wrong_agent = any_agent_events - events
            if wrong_agent:
                results.append(
                    (
                        "FAIL",
                        "%s: hook config contains ostinote hooks for the wrong agent in %s (%s)"
                        % (agent, hooks_file, ", ".join(sorted(wrong_agent))),
                    )
                )
            if events:
                registered_scopes.add(scope)
                registered |= events
    missing = expected - registered
    if not registered:
        results.append(("warn", "%s: no hooks registered — run `ostinote install %s`" % (agent, agent)))
    elif missing:
        results.append(
            (
                "FAIL",
                "%s: hooks missing %s — rerun `ostinote install %s`" % (agent, ", ".join(sorted(missing)), agent),
            )
        )
    else:
        events = ", ".join(sorted(expected))
        results.append(("ok", "%s: hooks registered (%s)" % (agent, events)))
    return registered_scopes


def _check_skills(agent: str, scopes: set[str], env: Env, results: list) -> None:
    """Report missing core-memory skill files for scopes with hooks."""
    for scope in sorted(scopes):
        path = install_mod._skill_path_for(agent, scope, env.project_root)
        if os.path.exists(path):
            results.append(("ok", "%s: skill installed (%s): %s" % (agent, scope, path)))
        else:
            results.append(("FAIL", "%s: skill missing (%s): %s" % (agent, scope, path)))


def _check_codex_writable_roots(scopes: set[str], env: Env, results: list) -> None:
    """Verify Codex sandbox access for scopes with Codex hooks."""
    if not scopes:
        return
    config_path = os.path.expanduser("~/.codex/config.toml")
    try:
        data = install_mod._parse_toml(config_path, install_mod._read_text_or_empty(config_path))
    except (OSError, install_mod._ConfigError) as e:
        results.append(("FAIL", "codex writable roots unreadable: %s" % e))
        return
    sandbox = data.get("sandbox_workspace_write")
    roots = sandbox.get("writable_roots") if isinstance(sandbox, dict) else None
    if not isinstance(roots, list) or not all(isinstance(item, str) for item in roots):
        results.append(("FAIL", "codex writable roots missing or invalid: %s" % config_path))
        return
    for scope in sorted(scopes):
        expected = install_mod.codex_writable_root_for(env.project_root, scope)
        if install_mod._root_list_has_value(roots, expected):
            results.append(("ok", "codex writable root present (%s): %s" % (scope, expected)))
        else:
            results.append(("FAIL", "codex writable root missing (%s): %s" % (scope, expected)))


def _check_summarizer(env: Env, results: list, live: bool) -> None:
    command = summarize.resolve_command(env.cfg)
    if shutil.which(command[0]):
        results.append(("ok", "summarizer executable found: %s" % command[0]))
    else:
        results.append(("FAIL", "summarizer executable not found: %s" % command[0]))
        return
    if not live:
        return
    started = time.time()
    try:
        result = summarize.call_model("Reply with exactly: OK", env.cfg)
        elapsed = time.time() - started
        if result.text.strip():
            results.append(("ok", "live summarizer call succeeded (%.1fs)" % elapsed))
        else:
            results.append(("FAIL", "live summarizer call returned empty text"))
    except RuntimeError as e:
        results.append(("FAIL", "live summarizer call failed: %s" % e))


def _check_locks(env: Env, results: list) -> None:
    stale = []
    try:
        names = os.listdir(env.state_dir)
    except OSError:
        return
    for name in names:
        if not name.endswith(".lock"):
            continue
        path = os.path.join(env.state_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                pid = int(f.read().strip() or "0")
        except (OSError, ValueError):
            pid = 0
        if not pid or not _pid_alive(pid):
            stale.append(name)
    if stale:
        # Harmless — stale locks are stolen on the next acquire.
        results.append(("warn", "stale lock(s), will be taken over: %s" % ", ".join(stale)))


def _check_sessions(env: Env, results: list) -> None:
    states = all_states(env.sessions_dir)
    gone = sum(1 for s in states if s.transcript_path and not os.path.exists(s.transcript_path))
    note = " (%d with deleted transcripts)" % gone if gone else ""
    results.append(("ok", "%d tracked session(s)%s" % (len(states), note)))
    staging = staging_files(env)
    if staging:
        results.append(("ok", "%d past day(s) awaiting consolidation" % len(staging)))


def _check_hook_errors(results: list) -> None:
    try:
        size = os.path.getsize(HOOK_ERRORS_PATH)
    except OSError:
        return
    if size:
        message = "hook crashes recorded — inspect %s (%d bytes)" % (HOOK_ERRORS_PATH, size)
        results.append(("warn", message))
