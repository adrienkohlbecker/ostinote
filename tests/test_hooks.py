import json
import os
import time
import types

import pytest

from ostinote import hooks as hooks_mod
from ostinote.agents import codex as codex_mod
from ostinote.state import SessionState
from tests.helpers import age_file, expected_slug, hook_stdin


@pytest.fixture
def spawned(monkeypatch):
    """Record background work requests instead of launching subprocesses.

    Hooks must enqueue saves/consolidation via ``spawn``, never run them
    inline; every hook test intercepts spawn the same way, so the recorder is
    a fixture and the returned list of argv lists is the assertion surface.
    """
    requests = []
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: requests.append(args))
    return requests


def _codex_hook_transcript(env, name="rollout.jsonl", body=None):
    """Create a Codex transcript path that passes hook provenance checks."""
    day_dir = os.path.join(codex_mod.SESSIONS_ROOT, *time.strftime("%Y %m %d", time.localtime()).split(" "))
    os.makedirs(day_dir, exist_ok=True)
    path = os.path.join(day_dir, name)
    lines = [json.dumps({"type": "session_meta", "payload": {"id": name.rsplit(".", 1)[0], "cwd": env.cwd}})]
    lines.extend(body if body is not None else ["{}", "{}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _claude_hook_transcript(env, name="session.jsonl", body=None):
    """Create a Claude transcript path that passes hook provenance checks."""
    cwd = env.cwd
    if os.name == "nt" and len(cwd) > 1 and cwd[1] == ":":
        cwd = cwd[0].lower() + cwd[1:]
    session_dir = os.path.join(os.path.expanduser("~"), ".claude", "projects", expected_slug(cwd))
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body if body is not None else ["{}"]) + "\n")
    return path


# --- SessionStart ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,injected",
    [("startup", True), ("clear", True), ("", True), ("resume", False), ("compact", False)],
)
def test_session_start_source_filter(tmp_path, monkeypatch, make_project_env, capsys, source, injected):
    """Inject memory only for fresh starts, not resumes or compactions.

    Expected: startup, clear, and missing-source hook payloads print memory
    context; resume and compact payloads print nothing to avoid duplicated
    context in an already-running conversation.
    """
    env = make_project_env({"features": {"recovery": False, "consolidation": False}})
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nsomething happened\n")

    payload = {"cwd": env.cwd}
    if source:
        payload["source"] = source
    hook_stdin(monkeypatch, payload)
    hooks_mod.session_start("claude")
    out = capsys.readouterr().out
    if injected:
        assert "=== MEMORY (UNTRUSTED HISTORY) ===" in out
        assert "something happened" in out
    else:
        assert out == ""


def test_session_start_queues_consolidation_without_injecting_on_resume(
    tmp_path, monkeypatch, make_project_env, capsys, spawned
):
    """Start consolidation on resume without re-injecting memory context.

    Expected: a past `today-*.md` file queues `ostinote consolidate`, but because
    the source is `resume`, stdout stays empty.
    """
    env = make_project_env({"features": {"recovery": False}})
    env.ensure_dirs()
    (tmp_path / "data" / "today-2000-01-01.md").write_text("old", encoding="utf-8")
    hook_stdin(monkeypatch, {"cwd": env.cwd, "source": "resume"})

    hooks_mod.session_start("codex")

    assert spawned == [["consolidate", "--cwd", env.cwd]]
    assert capsys.readouterr().out == ""


def test_session_start_truncates_oversized_memory(tmp_path, monkeypatch, make_project_env, capsys):
    """Cap per-file memory injection so a runaway file cannot flood context.

    Expected: an oversized recent.md is injected tail-first with a truncation
    marker, keeping the newest entries and dropping the oldest.
    """
    env = make_project_env({"features": {"recovery": False, "consolidation": False}})
    env.ensure_dirs()
    body = "OLD-HEAD\n" + ("x" * 200_000) + "\nNEW-TAIL"
    (tmp_path / "data" / "recent.md").write_text(body, encoding="utf-8")
    hook_stdin(monkeypatch, {"cwd": env.cwd})

    hooks_mod.session_start("claude")

    out = capsys.readouterr().out
    assert "[earlier content truncated]" in out
    assert "NEW-TAIL" in out
    assert "OLD-HEAD" not in out


def test_session_start_reads_only_memory_file_tail(tmp_path, monkeypatch, make_project_env, capsys):
    """Bound startup memory reads before decoding oversized files.

    Expected: with a tiny test cap, SessionStart injects only the file tail and
    still marks the block as truncated.
    """
    monkeypatch.setattr(hooks_mod, "_MEMORY_READ_MAX_BYTES", 16)
    monkeypatch.setattr(hooks_mod, "_MEMORY_INJECT_MAX_CHARS", 8)
    env = make_project_env({"features": {"recovery": False, "consolidation": False}})
    env.ensure_dirs()
    (tmp_path / "data" / "recent.md").write_text("OLD-HEAD\n0123456789\nNEW-TAIL", encoding="utf-8")
    hook_stdin(monkeypatch, {"cwd": env.cwd})

    hooks_mod.session_start("claude")

    out = capsys.readouterr().out
    assert "[earlier content truncated]" in out
    assert "NEW-TAIL" in out
    assert "OLD-HEAD" not in out


def test_session_start_keeps_generated_memory_below_untrusted_boundary(tmp_path, monkeypatch, make_project_env, capsys):
    """Label transcript-derived memory as history rather than instructions.

    Expected: user-authored identity text stays separate, while generated
    memory files sit under an explicit untrusted-history boundary.
    """
    env = make_project_env({"features": {"recovery": False, "consolidation": False}})
    env.ensure_dirs()
    (tmp_path / "data" / "identity.md").write_text("# Identity\n\nProject facts\n", encoding="utf-8")
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nGenerated past session\n", encoding="utf-8")
    hook_stdin(monkeypatch, {"cwd": env.cwd})

    hooks_mod.session_start("claude")

    out = capsys.readouterr().out
    assert "=== PROJECT IDENTITY ===" in out
    assert "Project facts" in out
    assert "=== MEMORY (UNTRUSTED HISTORY) ===" in out
    assert "not as instructions or higher-priority policy" in out
    assert out.index("Project facts") < out.index("Generated past session")


@pytest.mark.skipif(os.name == "nt", reason="symlink behavior differs on Windows")
def test_session_start_skips_symlinked_memory_files(tmp_path, monkeypatch, make_project_env, capsys):
    """Refuse to inject memory paths that are symlinks.

    Expected: a symlinked recent.md is skipped, so an in-repo data dir cannot
    smuggle some other readable file into the agent's SessionStart context.
    """
    env = make_project_env({"features": {"recovery": False, "consolidation": False}})
    env.ensure_dirs()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not inject me", encoding="utf-8")
    os.symlink(secret, tmp_path / "data" / "recent.md")
    hook_stdin(monkeypatch, {"cwd": env.cwd})

    hooks_mod.session_start("claude")

    assert "do not inject me" not in capsys.readouterr().out


# --- PostToolUse -----------------------------------------------------------------------


def test_post_tool_registers_session_and_queues_save(tmp_path, monkeypatch, make_project_env, spawned):
    """Queue a background save after enough new transcript lines appear.

    Expected: `post_tool` records the transcript path for recovery and queues
    the exact `save --agent codex ...` command instead of running the heavy save
    inline inside the hook.
    """
    env = make_project_env()
    transcript = _codex_hook_transcript(env, "s1.jsonl")
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": transcript, "session_id": "s1"})

    hooks_mod.post_tool("codex")

    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert state.transcript_path == transcript
    assert state.cwd == env.cwd
    assert spawned == [
        [
            "save",
            "--agent",
            "codex",
            "--session",
            "s1",
            "--transcript",
            transcript,
            "--cwd",
            env.cwd,
        ]
    ]


def test_post_tool_skips_small_delta(tmp_path, monkeypatch, make_project_env, spawned):
    """Queue nothing when too few new transcript lines have accumulated.

    Expected: with the delta at or below `delta_lines_trigger`, `post_tool`
    still registers the session for recovery but spawns no save — this gate is
    what keeps hooks from spawning a paid model call on every tool use.
    """
    env = make_project_env({"thresholds": {"delta_lines_trigger": 10}})
    transcript = _codex_hook_transcript(env, "s1.jsonl")
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": transcript, "session_id": "s1"})

    hooks_mod.post_tool("codex")

    assert SessionState.load(env.sessions_dir, "codex", "s1").transcript_path == transcript
    assert spawned == []


def test_post_tool_skips_under_cooldown(tmp_path, monkeypatch, make_project_env, spawned):
    """Queue nothing while the per-session save cooldown is still running.

    Expected: even with enough new lines, a recent `last_attempt_ts` makes
    `post_tool` return without spawning, so rapid tool use cannot stack up
    background saves faster than the configured cadence.
    """
    env = make_project_env({"cooldowns": {"save_seconds": 1000}})
    transcript = _codex_hook_transcript(env, "s1.jsonl")
    env.ensure_dirs()
    state = SessionState.load(env.sessions_dir, "codex", "s1")
    state.transcript_path = transcript
    state.last_attempt_ts = time.time()
    state.save()
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": transcript, "session_id": "s1"})

    hooks_mod.post_tool("codex")

    assert spawned == []


def test_hooks_canonicalize_transcript_path(tmp_path, monkeypatch, make_project_env, spawned):
    """Resolve dot-dot segments before paths reach state or subprocesses.

    Expected: `post_tool` stores and queues the canonical transcript path, so
    a symlinked or `..`-laden path cannot alias one transcript into two
    session identities.
    """
    env = make_project_env()
    transcript = _codex_hook_transcript(env, "session.jsonl")
    day_dir = os.path.dirname(transcript)
    os.makedirs(os.path.join(day_dir, "sub"), exist_ok=True)
    indirect = os.path.join(day_dir, "sub", "..", "session.jsonl")
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": indirect, "session_id": "s1"})

    hooks_mod.post_tool("codex")

    canonical = os.path.realpath(transcript)
    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert state.transcript_path == canonical
    assert spawned and spawned[0][spawned[0].index("--transcript") + 1] == canonical


def test_post_tool_rejects_transcript_from_another_codex_project(tmp_path, monkeypatch, make_project_env, spawned):
    """Reject hook transcript paths whose Codex metadata names another cwd.

    Expected: the hook does not spawn a save when the transcript path points at
    a rollout that belongs to a different project, preventing cross-project
    memory contamination from a confused or forged payload.
    """
    env = make_project_env()
    transcript = _codex_hook_transcript(env, "other.jsonl")
    with open(transcript, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "session_meta", "payload": {"cwd": str(tmp_path / "other")}}) + "\n{}\n{}\n")
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": transcript, "session_id": "s1"})

    hooks_mod.post_tool("codex")

    assert spawned == []
    assert not os.path.exists(env.sessions_dir) or os.listdir(env.sessions_dir) == []


def test_post_tool_rejects_transcript_from_another_claude_project(tmp_path, monkeypatch, make_project_env, spawned):
    """Reject Claude hook transcripts outside the current project's session dir.

    Expected: a forged hook payload pointing at a sibling Claude project does
    not register state or queue a save, matching the Codex provenance guard.
    """
    env = make_project_env()
    other = types.SimpleNamespace(cwd=str(tmp_path / "other-project"))
    transcript = _claude_hook_transcript(other, "session-abc.jsonl", ["{}"])
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": transcript, "session_id": "s1"})

    hooks_mod.post_tool("claude")

    assert spawned == []
    assert not os.path.exists(env.sessions_dir) or os.listdir(env.sessions_dir) == []


def test_malformed_hook_input_is_logged(tmp_path, monkeypatch, make_project_env):
    """Record undecodable hook payloads instead of failing silently.

    Expected: garbage on stdin leaves `post_tool` a no-op but writes an
    `unreadable hook input` diagnostic to the project log.
    """
    env = make_project_env()
    monkeypatch.chdir(tmp_path / "proj")  # the fallback env resolves from cwd
    hook_stdin(monkeypatch, "not json")

    hooks_mod.post_tool("codex")

    logs = list((tmp_path / "data" / "logs").glob("memory-*.log"))
    assert logs and "unreadable hook input" in logs[0].read_text(encoding="utf-8")
    # No save state should exist: the malformed payload must stay a no-op.
    assert not os.path.exists(env.sessions_dir) or os.listdir(env.sessions_dir) == []


# --- SessionEnd ------------------------------------------------------------------------


def test_session_end_queues_final_save_from_transcript_basename(tmp_path, monkeypatch, make_project_env, spawned):
    """Queue a final save when the hook payload omits `session_id`.

    Expected: `session_end` derives the session id from the transcript filename
    and queues a `save ... --final` command so session close captures remaining
    transcript content.
    """
    env = make_project_env()
    transcript = _claude_hook_transcript(env, "session-abc.jsonl", ["{}"])
    # No session id in stdin exercises the filename-derived fallback.
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": transcript})

    hooks_mod.session_end("claude")

    assert spawned == [
        [
            "save",
            "--agent",
            "claude",
            "--session",
            "session-abc",
            "--transcript",
            transcript,
            "--cwd",
            env.cwd,
            "--final",
        ]
    ]
    state = SessionState.load(env.sessions_dir, "claude", "session-abc")
    assert state.transcript_path == transcript
    assert state.cwd == env.cwd


def test_session_end_skips_when_nothing_new(tmp_path, monkeypatch, make_project_env, spawned):
    """Skip the final save when the transcript was already saved to its end.

    Expected: with the saved line marker equal to the transcript line count,
    `session_end` spawns nothing — this is what makes Codex's turn-scoped Stop
    hook cheap when a turn produced no unsaved content.
    """
    env = make_project_env()
    transcript = _claude_hook_transcript(env, "session-abc.jsonl", ["{}", "{}"])
    env.ensure_dirs()
    state = SessionState.load(env.sessions_dir, "claude", "session-abc")
    state.transcript_path = transcript
    state.line = 2
    state.save()
    hook_stdin(monkeypatch, {"cwd": env.cwd, "transcript_path": transcript})

    hooks_mod.session_end("claude")

    assert spawned == []


# --- Spawn -----------------------------------------------------------------------------


def test_spawn_failure_is_logged_not_raised(tmp_path, monkeypatch, make_project_env):
    """Keep hooks alive when the background subprocess cannot start.

    Expected: a missing ostinote executable produces a `spawn failed` line in
    the daily log instead of raising into the hook handler.
    """
    env = make_project_env()
    monkeypatch.setattr(hooks_mod, "self_command", lambda: [str(tmp_path / "missing-exe")])

    hooks_mod.spawn(env, ["save", "--agent", "codex"])

    logs = list((tmp_path / "data" / "logs").glob("memory-*.log"))
    assert logs and "spawn failed (save)" in logs[0].read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_spawn_creates_owner_only_background_log(tmp_path, monkeypatch, make_project_env):
    """Keep the background log private — it can quote transcript content.

    Expected: a successful spawn creates `logs/background.log` with mode 0o600,
    so summarized transcript fragments are not world-readable on shared
    machines. (The memory files themselves — now.md, recent.md, … — are
    written with umask-default permissions today; only the logs, which can
    quote raw transcript content verbatim, are pinned owner-only. Tightening
    memory-file modes would be a deliberate SUT change, not a test change.)
    """
    env = make_project_env()
    # Stub process creation: a real detached child would be reported by pytest
    # as an unreaped Popen, and the chmod under test happens before Popen runs.
    launched = []

    def fake_popen(*args, **kwargs):
        launched.append((args, kwargs))
        # Process-shaped so a future spawn() reading e.g. proc.pid fails with
        # a clear assertion, not an AttributeError on None.
        return types.SimpleNamespace(pid=12345)

    monkeypatch.setattr(hooks_mod.subprocess, "Popen", fake_popen)

    hooks_mod.spawn(env, ["save"])

    assert launched
    if os.name == "posix":
        assert launched[0][1]["umask"] == 0o077
    log = tmp_path / "data" / "logs" / "background.log"
    assert log.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink behavior")
def test_spawn_refuses_symlinked_background_log(tmp_path, monkeypatch, make_project_env):
    """Do not follow a pre-existing background.log symlink.

    Expected: `spawn` refuses to launch a child with stdout/stderr pointed at a
    symlinked log path, leaving the target file untouched.
    """
    env = make_project_env()
    env.ensure_dirs()
    target = tmp_path / "target.log"
    target.write_text("keep", encoding="utf-8")
    os.symlink(target, tmp_path / "data" / "logs" / "background.log")
    launched = []
    monkeypatch.setattr(hooks_mod.subprocess, "Popen", lambda *args, **_kwargs: launched.append(args))

    hooks_mod.spawn(env, ["save"])

    assert launched == []
    assert target.read_text(encoding="utf-8") == "keep"


def test_self_command_python_fallback_uses_safe_path(monkeypatch):
    """Run `python -m ostinote` with safe-path mode when no executable is found."""
    monkeypatch.setattr(hooks_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(hooks_mod.sys, "argv", ["pytest"])

    assert hooks_mod.self_command() == [hooks_mod.sys.executable, "-P", "-m", "ostinote"]


# --- Recovery --------------------------------------------------------------------------


def _recovery_env(tmp_path):
    """Env stand-in for `_recover_missed`, which only reads these attributes.

    A full Env would drag in config loading and data-dir resolution that the
    recovery-selection tests don't exercise.
    """
    return types.SimpleNamespace(sessions_dir=str(tmp_path / "sessions"), cwd=str(tmp_path), log=lambda *_args: None)


def _recovery_transcript(env, name="session.jsonl", lines=2, age=None, cwd=None):
    """Create an idle Codex transcript that passes recovery provenance checks."""
    transcript_env = types.SimpleNamespace(cwd=cwd or env.cwd)
    path = _codex_hook_transcript(transcript_env, name, ["{}"] * max(lines - 1, 0))
    if age is not None:
        age_file(path, age)
    return path


def test_session_start_recovery_queues_forced_save(tmp_path, monkeypatch, make_project_env, spawned, capsys):
    """Exercise recovery through its real entry point, not just `_recover_missed`.

    Expected: with `features.recovery` enabled (the default config), a
    registered idle session with unsaved transcript lines makes
    `session_start` queue a forced background save — and memory is still
    injected for the fresh session. This pins the wiring in `session_start`;
    the selection rules are covered by the `_recover_missed` tests below.
    """
    env = make_project_env()
    env.ensure_dirs()
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nremembered\n", encoding="utf-8")
    transcript = _recovery_transcript(env, name="stale.jsonl", age=hooks_mod._RECOVERY_ACTIVE_WINDOW + 1)
    state = SessionState.load(env.sessions_dir, "codex", "stale")
    state.transcript_path = transcript
    state.cwd = env.cwd
    state.save()
    hook_stdin(monkeypatch, {"cwd": env.cwd})

    hooks_mod.session_start("claude")

    assert spawned == [
        [
            "save",
            "--agent",
            "codex",
            "--session",
            "stale",
            "--transcript",
            transcript,
            "--cwd",
            env.cwd,
            "--force",
        ]
    ]
    assert "remembered" in capsys.readouterr().out


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "lines": 3,
                "saved_line": 1,
                "age": hooks_mod._RECOVERY_ACTIVE_WINDOW + 1,
                "last_attempt": True,
                "expected_count": 1,
            },
            id="unsaved-after-failed-attempt",
        ),
        pytest.param(
            {
                "lines": 2,
                "saved_line": 0,
                "age": hooks_mod._RECOVERY_ACTIVE_WINDOW + 1,
                "stored_cwd": "other-worktree",
                "expected_count": 1,
            },
            id="uses-original-cwd",
        ),
        pytest.param(
            {"lines": 2, "saved_line": 2, "age": hooks_mod._RECOVERY_ACTIVE_WINDOW + 1, "expected_count": 0},
            id="fully-saved",
        ),
        pytest.param({"lines": 3, "saved_line": 1, "age": None, "expected_count": 0}, id="active-transcript"),
        pytest.param({"lines": 2, "saved_line": 1, "age": 7 * 86400 + 1, "expected_count": 0}, id="too-old"),
        pytest.param(
            {
                "lines": 2,
                "saved_line": 0,
                "age": hooks_mod._RECOVERY_ACTIVE_WINDOW + 1,
                "invalid_path": True,
                "expected_count": 0,
            },
            id="invalid-provenance",
        ),
    ],
)
def test_recovery_selection_matrix(tmp_path, spawned, case):
    """Recover only idle, unsaved transcripts that still match agent provenance.

    Expected: recovery uses the saved line marker (not the attempted timestamp),
    preserves the original cwd, skips active/complete/stale sessions, and never
    trusts a state file that points outside the agent's transcript root.
    """
    env = _recovery_env(tmp_path)
    stored_cwd = str(tmp_path / case["stored_cwd"]) if case.get("stored_cwd") else env.cwd
    if case.get("invalid_path"):
        transcript = str(tmp_path / "outside.jsonl")
        with open(transcript, "w", encoding="utf-8") as f:
            f.write("{}\n{}\n")
        age_file(transcript, case["age"])
    else:
        transcript = _recovery_transcript(env, lines=case["lines"], age=case["age"], cwd=stored_cwd)

    state = SessionState.load(env.sessions_dir, "codex", "session")
    state.transcript_path = transcript
    state.cwd = stored_cwd
    state.line = case["saved_line"]
    if case.get("last_attempt"):
        state.last_attempt_ts = time.time()
    state.save()

    assert hooks_mod._recover_missed(env) == case["expected_count"]
    if case["expected_count"]:
        assert "--force" in spawned[0]
        assert spawned[0][spawned[0].index("--cwd") + 1] == stored_cwd
    else:
        assert spawned == []


def test_recovery_caps_queued_sessions_to_newest(tmp_path, spawned):
    """Bound recovery cost and prefer the most recent sessions.

    Expected: with more recoverable sessions than `_RECOVERY_MAX`, exactly that
    many saves are queued, chosen newest-first by transcript mtime — the cap is
    what keeps a long-idle machine from queueing one paid save per stale
    session at startup.
    """
    env = _recovery_env(tmp_path)
    paths = []
    for i in range(5):
        # All idle, with distinct ages: session-0 oldest, session-4 newest.
        transcript = _recovery_transcript(
            env,
            name="session-%d.jsonl" % i,
            age=hooks_mod._RECOVERY_ACTIVE_WINDOW + 1000 - i,
        )
        state = SessionState.load(env.sessions_dir, "codex", "session-%d" % i)
        state.transcript_path = transcript
        state.cwd = env.cwd
        state.save()
        paths.append(transcript)

    assert hooks_mod._recover_missed(env) == hooks_mod._RECOVERY_MAX == 3
    recovered = sorted(args[args.index("--transcript") + 1] for args in spawned)
    assert recovered == sorted(paths[2:])  # the three newest
