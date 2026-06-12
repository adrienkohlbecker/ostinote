import os
import time
import types

import pytest

from ostinote import hooks as hooks_mod
from ostinote.state import SessionState
from tests.helpers import hook_stdin, project_env

# --- SessionStart ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,injected",
    [("startup", True), ("clear", True), ("", True), ("resume", False), ("compact", False)],
)
def test_session_start_source_filter(tmp_path, monkeypatch, capsys, source, injected):
    """Inject memory only for fresh starts, not resumes or compactions.

    Expected: startup, clear, and missing-source hook payloads print memory
    context; resume and compact payloads print nothing to avoid duplicated
    context in an already-running conversation.
    """
    env = project_env(tmp_path, monkeypatch, {"features": {"recovery": False, "consolidation": False}})
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nsomething happened\n")

    payload = {"cwd": env.cwd}
    if source:
        payload["source"] = source
    hook_stdin(monkeypatch, payload)
    hooks_mod.session_start("claude")
    out = capsys.readouterr().out
    if injected:
        assert "=== MEMORY ===" in out
        assert "something happened" in out
    else:
        assert out == ""


def test_session_start_queues_consolidation_without_injecting_on_resume(tmp_path, monkeypatch, capsys):
    """Start consolidation on resume without re-injecting memory context.

    Expected: a past `today-*.md` file queues `ostinote consolidate`, but because
    the source is `resume`, stdout stays empty.
    """
    env = project_env(tmp_path, monkeypatch, {"features": {"recovery": False}})
    env.ensure_dirs()
    (tmp_path / "data" / "today-2000-01-01.md").write_text("old", encoding="utf-8")
    queued = []
    # Capture the consolidation request while still letting `session_start`
    # decide whether memory should be printed for this source.
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    hook_stdin(monkeypatch, {"cwd": env.cwd, "source": "resume"})

    hooks_mod.session_start("codex")

    assert queued == [["consolidate", "--cwd", env.cwd]]
    assert capsys.readouterr().out == ""


def test_session_start_truncates_oversized_memory(tmp_path, monkeypatch, capsys):
    """Cap per-file memory injection so a runaway file cannot flood context.

    Expected: an oversized recent.md is injected tail-first with a truncation
    marker, keeping the newest entries and dropping the oldest.
    """
    env = project_env(tmp_path, monkeypatch, {"features": {"recovery": False, "consolidation": False}})
    env.ensure_dirs()
    body = "OLD-HEAD\n" + ("x" * 200_000) + "\nNEW-TAIL"
    (tmp_path / "data" / "recent.md").write_text(body, encoding="utf-8")
    hook_stdin(monkeypatch, {"cwd": env.cwd})

    hooks_mod.session_start("claude")

    out = capsys.readouterr().out
    assert "[earlier content truncated]" in out
    assert "NEW-TAIL" in out
    assert "OLD-HEAD" not in out


# --- PostToolUse -----------------------------------------------------------------------


def test_post_tool_registers_session_and_queues_save(tmp_path, monkeypatch):
    """Queue a background save after enough new transcript lines appear.

    Expected: `post_tool` records the transcript path for recovery and queues
    the exact `save --agent codex ...` command instead of running the heavy save
    inline inside the hook.
    """
    env = project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n{}\n", encoding="utf-8")
    queued = []
    # Hooks should enqueue background work, not run summarization synchronously.
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": str(transcript), "session_id": "s1"})

    hooks_mod.post_tool("codex")

    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert state.transcript_path == str(transcript)
    assert queued == [
        [
            "save",
            "--agent",
            "codex",
            "--session",
            "s1",
            "--transcript",
            str(transcript),
            "--cwd",
            env.cwd,
        ]
    ]


def test_hooks_canonicalize_transcript_path(tmp_path, monkeypatch):
    """Resolve dot-dot segments before paths reach state or subprocesses.

    Expected: `post_tool` stores and queues the canonical transcript path, so
    a symlinked or `..`-laden path cannot alias one transcript into two
    session identities.
    """
    env = project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n{}\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    indirect = str(tmp_path / "sub" / ".." / "session.jsonl")
    queued = []
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": indirect, "session_id": "s1"})

    hooks_mod.post_tool("codex")

    canonical = os.path.realpath(str(transcript))
    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert state.transcript_path == canonical
    assert queued and queued[0][queued[0].index("--transcript") + 1] == canonical


def test_malformed_hook_input_is_logged(tmp_path, monkeypatch):
    """Record undecodable hook payloads instead of failing silently.

    Expected: garbage on stdin leaves `post_tool` a no-op but writes an
    `unreadable hook input` diagnostic to the project log.
    """
    env = project_env(tmp_path, monkeypatch)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.chdir(tmp_path / "proj")  # the fallback env resolves from cwd
    hook_stdin(monkeypatch, "not json")

    hooks_mod.post_tool("codex")

    logs = list((tmp_path / "data" / "logs").glob("memory-*.log"))
    assert logs and "unreadable hook input" in logs[0].read_text(encoding="utf-8")
    assert env.cwd  # env fixture used; no save state should exist
    assert not os.path.exists(env.sessions_dir) or os.listdir(env.sessions_dir) == []


# --- SessionEnd ------------------------------------------------------------------------


def test_session_end_queues_final_save_from_transcript_basename(tmp_path, monkeypatch):
    """Queue a final save when the hook payload omits `session_id`.

    Expected: `session_end` derives the session id from the transcript filename
    and queues a `save ... --final` command so session close captures remaining
    transcript content.
    """
    env = project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session-abc.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    queued = []
    # No session id in stdin exercises the filename-derived fallback.
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": str(transcript)})

    hooks_mod.session_end("claude")

    assert queued == [
        [
            "save",
            "--agent",
            "claude",
            "--session",
            "session-abc",
            "--transcript",
            str(transcript),
            "--cwd",
            env.cwd,
            "--final",
        ]
    ]


# --- Spawn -----------------------------------------------------------------------------


def test_spawn_failure_is_logged_not_raised(tmp_path, monkeypatch):
    """Keep hooks alive when the background subprocess cannot start.

    Expected: a missing ostinote executable produces a `spawn failed` line in
    the daily log instead of raising into the hook handler.
    """
    env = project_env(tmp_path, monkeypatch)
    monkeypatch.setattr(hooks_mod, "self_command", lambda: [str(tmp_path / "missing-exe")])

    hooks_mod.spawn(env, ["save", "--agent", "codex"])

    logs = list((tmp_path / "data" / "logs").glob("memory-*.log"))
    assert logs and "spawn failed (save)" in logs[0].read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_spawn_creates_owner_only_background_log(tmp_path, monkeypatch):
    """Keep the background log private — it can quote transcript content.

    Expected: a successful spawn creates `logs/background.log` with mode 0o600,
    so summarized transcript fragments are not world-readable on shared
    machines.
    """
    env = project_env(tmp_path, monkeypatch)
    # Stub process creation: a real detached child would be reported by pytest
    # as an unreaped Popen, and the chmod under test happens before Popen runs.
    launched = []
    monkeypatch.setattr(hooks_mod.subprocess, "Popen", lambda *args, **kwargs: launched.append(args))

    hooks_mod.spawn(env, ["save"])

    assert launched
    log = tmp_path / "data" / "logs" / "background.log"
    assert log.stat().st_mode & 0o777 == 0o600


# --- Recovery --------------------------------------------------------------------------


def _recovery_env(tmp_path):
    """Env stand-in for `_recover_missed`, which only reads these attributes.

    A full Env would drag in config loading and data-dir resolution that the
    recovery-selection tests don't exercise.
    """
    return types.SimpleNamespace(sessions_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))


def test_recovery_uses_saved_line_not_failed_attempt_time(tmp_path, monkeypatch):
    """Recover missed transcript lines even after a failed save attempt.

    Expected: recovery compares transcript line count to the saved line marker,
    not `last_attempt_ts`; unsaved idle content queues a forced background save.
    """
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n{}\n")
    # Recovery deliberately ignores transcripts that still look active.
    idle_mtime = time.time() - hooks_mod._RECOVERY_ACTIVE_WINDOW - 1
    os.utime(transcript, (idle_mtime, idle_mtime))

    env = _recovery_env(tmp_path)
    state = SessionState.load(env.sessions_dir, "codex", "session")
    state.transcript_path = str(transcript)
    state.line = 1
    state.last_attempt_ts = time.time()
    state.save()

    queued = []
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))

    assert hooks_mod._recover_missed(env) == 1
    assert queued
    assert "--force" in queued[0]


def test_recovery_skips_fully_saved_transcripts(tmp_path, monkeypatch):
    """Avoid recovery work for transcripts already saved through their end.

    Expected: if the saved line marker equals the transcript line count, no
    background save is queued.
    """
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n")
    # Make the transcript old enough to pass the "not actively changing" gate.
    idle_mtime = time.time() - hooks_mod._RECOVERY_ACTIVE_WINDOW - 1
    os.utime(transcript, (idle_mtime, idle_mtime))

    env = _recovery_env(tmp_path)
    state = SessionState.load(env.sessions_dir, "codex", "session")
    state.transcript_path = str(transcript)
    state.line = 2
    state.save()

    queued = []
    # Intercept background process creation so the assertion can inspect intent.
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))

    assert hooks_mod._recover_missed(env) == 0
    assert queued == []
