"""Lifecycle hook handlers, shared by Claude Code and Codex.

Both agents deliver the same stdin JSON to hooks (``session_id``,
``transcript_path``, ``cwd``, ``hook_event_name``), so one implementation
serves both; only the output envelope differs (Claude injects plain stdout,
Codex expects a ``hookSpecificOutput`` JSON object).

Hook handlers must never block or break the agent: heavy work (saves,
consolidation) is spawned as detached ``ostinote`` subprocesses, and the CLI
wraps these handlers to swallow all exceptions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

from .env import Env
from .pipeline import staging_files
from .state import SessionState, all_states

# Sessions whose transcript changed in the last N seconds are presumed live —
# their own post-tool hook will save them; recovery leaves them alone.
_RECOVERY_ACTIVE_WINDOW = 300
# Recover at most this many missed sessions per session start (cost bound).
_RECOVERY_MAX = 3


def read_hook_input() -> dict:
    try:
        data = json.load(sys.stdin)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def env_from_hook(data: dict) -> Env:
    cwd = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    return Env(cwd)


def self_command() -> list[str]:
    """argv prefix that re-invokes this tool in a subprocess."""
    exe = shutil.which("ostinote")
    argv0 = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if argv0.endswith("ostinote") and os.access(argv0, os.X_OK):
        return [argv0]
    if exe:
        return [exe]
    return [sys.executable, "-m", "ostinote"]


def spawn(env: Env, args: list[str]) -> None:
    """Launch a fully detached background subprocess, output to the log."""
    env.ensure_dirs()
    log_path = os.path.join(env.logs_dir, "background.log")
    detach: dict = {}
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        detach["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        detach["start_new_session"] = True
    with open(log_path, "ab") as log:
        subprocess.Popen(
            self_command() + args,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd=env.cwd,
            **detach,
        )


def emit(agent_name: str, event_name: str, text: str) -> None:
    """Print hook output in the right envelope for the agent."""
    if not text:
        return
    if agent_name == "codex":
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": event_name,
                        "additionalContext": text,
                    }
                }
            )
        )
    else:
        print(text)


# --- SessionStart -------------------------------------------------------------


def session_start(agent_name: str) -> None:
    data = read_hook_input()
    env = env_from_hook(data)
    env.ensure_dirs()
    source = data.get("source") or "startup"
    env.log("hook", "session-start (%s, %s): root=%s" % (agent_name, source, env.project_root))

    # Recovery: background-save sessions that ended without a final save
    # (crashes/sleeps that killed the session-end save).
    if env.cfg["features"]["recovery"]:
        recovered = _recover_missed(env)
        if recovered:
            env.log("hook", "recovery: %d session(s) queued" % recovered)

    # Consolidation of past-day staging files (silent — the agent can't act
    # on it, and session start is when injected context is already largest).
    if env.cfg["features"]["consolidation"] and staging_files(env):
        env.log("hook", "consolidation queued: %d staging file(s)" % len(staging_files(env)))
        spawn(env, ["consolidate", "--cwd", env.cwd])

    # A resumed or compacted session already saw the memory once — injecting
    # it again would duplicate context.
    if source in ("resume", "compact"):
        return

    sections: list[str] = []

    # Standing instructions: what memory exists and where.
    command = "$ostinote" if agent_name == "codex" else "/ostinote"
    sections.append(
        "=== OSTINOTE ===\n"
        "Persistent memory in %s: now.md (session buffer), today-*.md (daily), "
        "recent.md (last 7d), archive.md (older), core-memories.md (key moments; "
        "%s appends to it). Search them on user request." % (env.data_dir, command)
    )

    # Memory files, most specific first.
    memory_files = [
        env.identity_file,
        env.core_memories_file,
        env.today_file(),
        env.now_file,
        env.recent_file,
        env.archive_file,
    ]
    blocks = []
    for path in memory_files:
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read().strip()
        except OSError:
            continue
        if content:
            blocks.append("--- %s ---\n%s" % (os.path.basename(path), content))
    if blocks:
        sections.append("=== MEMORY ===\n" + "\n\n".join(blocks))

    emit(agent_name, "SessionStart", "\n\n".join(sections))


def _recover_missed(env: Env) -> int:
    """Queue --force saves for transcripts with unsaved content.

    A session is recoverable when its transcript still exists, has content
    beyond the saved line marker, and has been idle long enough that it's not
    an active parallel session.
    """
    now = time.time()
    candidates = []
    for state in all_states(env.sessions_dir):
        path = state.transcript_path
        if not path or not os.path.exists(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if _count_lines(path) <= state.line:
            continue
        if now - mtime < _RECOVERY_ACTIVE_WINDOW:
            continue  # probably a live parallel session
        if now - mtime > 7 * 86400:
            continue
        candidates.append((mtime, state))

    candidates.sort(reverse=True, key=lambda pair: pair[0])
    for _, state in candidates[:_RECOVERY_MAX]:
        spawn(
            env,
            [
                "save",
                "--agent",
                state.agent,
                "--session",
                state.session_id,
                "--transcript",
                state.transcript_path,
                "--cwd",
                env.cwd,
                "--force",
            ],
        )
    return len(candidates[:_RECOVERY_MAX])


# --- SessionEnd ----------------------------------------------------------------


def session_end(agent_name: str) -> None:
    """Queue a final save of anything not yet captured.

    Fired by Claude's SessionEnd, and by Codex's turn-scoped Stop — Codex
    has no session-exit event, so every turn end is treated as a potential
    session end. Cheap when nothing is new; the --final save keeps the
    min-human-messages gate, so it costs a model call at most once per few
    exchanges.
    """
    data = read_hook_input()
    env = env_from_hook(data)

    transcript_path = data.get("transcript_path") or ""
    session_id = data.get("session_id") or ""
    if not transcript_path or not os.path.exists(transcript_path):
        return
    if not session_id:
        session_id = os.path.basename(transcript_path).rsplit(".", 1)[0]

    state = SessionState.load(env.sessions_dir, agent_name, session_id)
    if _count_lines(transcript_path) <= state.line:
        return  # nothing new since the last save

    env.log("hook", "session-end (%s): queueing final save of %s" % (agent_name, session_id))
    spawn(
        env,
        [
            "save",
            "--agent",
            agent_name,
            "--session",
            session_id,
            "--transcript",
            transcript_path,
            "--cwd",
            env.cwd,
            "--final",
        ],
    )


# --- PostToolUse ---------------------------------------------------------------


def post_tool(agent_name: str) -> None:
    data = read_hook_input()
    env = env_from_hook(data)

    transcript_path = data.get("transcript_path") or ""
    session_id = data.get("session_id") or ""
    if not transcript_path or not os.path.exists(transcript_path):
        return
    if not session_id:
        session_id = os.path.basename(transcript_path).rsplit(".", 1)[0]

    current_lines = _count_lines(transcript_path)
    state = SessionState.load(env.sessions_dir, agent_name, session_id)

    # Register the session so recovery can find it even if no save ever ran.
    if state.transcript_path != transcript_path:
        env.ensure_dirs()
        state.transcript_path = transcript_path
        state.save()

    delta = current_lines - state.line
    if delta <= env.cfg["thresholds"]["delta_lines_trigger"]:
        return
    # Cheap pre-check; run_save re-checks both under the lock.
    if time.time() - state.last_attempt_ts < env.cfg["cooldowns"]["save_seconds"]:
        return

    env.log(
        "hook",
        "post-tool (%s): delta %d lines, queueing save of %s" % (agent_name, delta, session_id),
    )
    spawn(
        env,
        [
            "save",
            "--agent",
            agent_name,
            "--session",
            session_id,
            "--transcript",
            transcript_path,
            "--cwd",
            env.cwd,
        ],
    )


def _count_lines(path: str) -> int:
    """Count transcript lines, returning 0 on any I/O error.

    Text mode with ``errors="replace"`` mirrors how ``Agent.parse()``
    enumerates lines, so saved line markers and these delta checks always
    agree. The never-raise contract matters: hooks call this on transcripts
    that may vanish at any moment.
    """
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _ in f:
                count += 1
    except OSError:
        return 0
    return count
