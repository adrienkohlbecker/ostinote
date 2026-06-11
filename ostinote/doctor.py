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
import time

from . import config as config_mod
from . import summarize
from .agents import agent_names
from .env import HOOK_ERRORS_PATH, Env
from .install import _events_for, _hooks_file_for, _is_ours
from .pipeline import staging_files
from .state import _pid_alive, all_states


def run(env: Env, live: bool = False) -> int:
    """Print a check report; exit code 1 if anything failed."""
    results: list[tuple[str, str]] = []

    _check_config(env, results)
    _check_data_dir(env, results)
    for agent in agent_names():
        _check_hooks(agent, env, results)
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
    paths = [
        config_mod.USER_CONFIG_PATH,
        os.path.join(env.project_root, ".ostinote", "config.json"),
    ]
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                json.load(f)
            results.append(("ok", "config readable: %s" % path))
        except (OSError, json.JSONDecodeError) as e:
            # The config loader silently ignores broken files — say it here.
            results.append(("FAIL", "config unreadable (silently ignored!): %s — %s" % (path, e)))


def _check_data_dir(env: Env, results: list) -> None:
    probe = os.path.join(env.state_dir, ".doctor-probe")
    try:
        env.ensure_dirs()
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        results.append(("ok", "data dir writable: %s" % env.data_dir))
    except OSError as e:
        results.append(("FAIL", "data dir not writable: %s — %s" % (env.data_dir, e)))


def _check_hooks(agent: str, env: Env, results: list) -> None:
    expected = set(_events_for(agent))
    registered: set = set()
    for scope in ("user", "project"):
        hooks_file = _hooks_file_for(agent, scope, env.project_root)
        try:
            with open(hooks_file, encoding="utf-8") as f:
                settings = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for event, groups in settings.get("hooks", {}).items():
            for group in groups:
                for hook in group.get("hooks", []):
                    if _is_ours(hook.get("command", "")):
                        registered.add(event)
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


def _check_summarizer(env: Env, results: list, live: bool) -> None:
    command = list(env.cfg["summarizer"]["command"] or summarize.DEFAULT_COMMAND)
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
