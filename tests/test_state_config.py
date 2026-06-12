import json
import os
import time

import pytest

from ostinote import config as config_mod
from ostinote.state import PidLock, SessionState

# --- State and locks -------------------------------------------------------------------


def test_session_state_roundtrip(tmp_path):
    """Persist and reload one session's save position.

    Expected: line number, transcript path, and save timestamp survive a disk
    round trip, and another agent using the same session id gets a separate
    state file.
    """
    sessions = str(tmp_path / "sessions")
    s = SessionState.load(sessions, "codex", "id-1")
    assert s.line == 0
    s.line = 42
    s.transcript_path = "/t.jsonl"
    s.last_save_ts = 123.0
    s.save()
    s2 = SessionState.load(sessions, "codex", "id-1")
    assert (s2.line, s2.transcript_path, s2.last_save_ts) == (42, "/t.jsonl", 123.0)
    # Parallel sessions don't collide.
    other = SessionState.load(sessions, "claude", "id-1")
    assert other.line == 0


def test_session_state_canonicalizes_transcript_path(tmp_path):
    """Resolve stored transcript paths on load.

    Expected: a `..`-laden path written by an older version (or a tampered
    state file) comes back canonicalized, so the hooks' path comparisons see
    one identity per transcript.
    """
    sessions = str(tmp_path / "sessions")
    s = SessionState.load(sessions, "codex", "id-2")
    s.transcript_path = str(tmp_path / "sub" / ".." / "t.jsonl")
    s.save()
    s2 = SessionState.load(sessions, "codex", "id-2")
    assert s2.transcript_path == os.path.realpath(str(tmp_path / "t.jsonl"))


def test_pid_lock(tmp_path):
    """Allow only one holder for a PID lock at a time.

    Expected: the first lock acquire succeeds, a second concurrent acquire
    fails, and the second lock can acquire after the first releases.
    """
    path = str(tmp_path / "x.lock")
    a, b = PidLock(path), PidLock(path)
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_pid_lock_stale_takeover(tmp_path):
    """Recover from a lock file whose recorded PID is not alive.

    Expected: a stale lock containing a certainly-dead PID can be stolen, so a
    crashed previous process does not block future saves forever.
    """
    path = str(tmp_path / "x.lock")
    with open(path, "w") as f:
        f.write("999999999")  # certainly dead
    assert PidLock(path).acquire()


def test_recovery_uses_saved_line_not_failed_attempt_time(tmp_path, monkeypatch):
    """Recover missed transcript lines even after a failed save attempt.

    Expected: recovery compares transcript line count to the saved line marker,
    not `last_attempt_ts`; unsaved idle content queues a forced background save.
    """
    from ostinote import hooks

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n{}\n")
    # Recovery deliberately ignores transcripts that still look active.
    idle_mtime = time.time() - hooks._RECOVERY_ACTIVE_WINDOW - 1
    os.utime(transcript, (idle_mtime, idle_mtime))

    sessions = tmp_path / "sessions"
    state = SessionState.load(str(sessions), "codex", "session")
    state.transcript_path = str(transcript)
    state.line = 1
    state.last_attempt_ts = time.time()
    state.save()

    queued = []
    # `_recover_missed` only needs the Env attributes below; using a tiny stub
    # keeps the test focused on recovery selection, not full Env construction.
    env = type("EnvStub", (), {"sessions_dir": str(sessions), "cwd": str(tmp_path)})()
    monkeypatch.setattr(hooks, "spawn", lambda _env, args: queued.append(args))

    assert hooks._recover_missed(env) == 1
    assert queued
    assert "--force" in queued[0]


def test_recovery_skips_fully_saved_transcripts(tmp_path, monkeypatch):
    """Avoid recovery work for transcripts already saved through their end.

    Expected: if the saved line marker equals the transcript line count, no
    background save is queued.
    """
    from ostinote import hooks

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n")
    # Make the transcript old enough to pass the "not actively changing" gate.
    idle_mtime = time.time() - hooks._RECOVERY_ACTIVE_WINDOW - 1
    os.utime(transcript, (idle_mtime, idle_mtime))

    sessions = tmp_path / "sessions"
    state = SessionState.load(str(sessions), "codex", "session")
    state.transcript_path = str(transcript)
    state.line = 2
    state.save()

    queued = []
    # Intercept background process creation so the assertion can inspect intent.
    env = type("EnvStub", (), {"sessions_dir": str(sessions), "cwd": str(tmp_path)})()
    monkeypatch.setattr(hooks, "spawn", lambda _env, args: queued.append(args))

    assert hooks._recover_missed(env) == 0
    assert queued == []


# --- Config ---------------------------------------------------------------------------


def test_config_project_overrides(tmp_path, monkeypatch):
    """Merge default, user, and project config layers in the right order.

    Expected: project values override user values, user values still apply when
    project config omits them, and nested defaults survive partial overrides.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(json.dumps({"cooldowns": {"save_seconds": 60}, "timezone": "UTC"}))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(json.dumps({"cooldowns": {"save_seconds": 30}}))
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["save_seconds"] == 30
    assert cfg["cooldowns"]["compress_seconds"] == 3600  # default survives
    assert cfg["timezone"] == "UTC"  # user layer survives


def test_data_dir_slug_placeholder(tmp_path, monkeypatch):
    """Expand `{slug}` in `data_dir` using the project-root slug scheme.

    Expected: the resolved data directory uses the configured store directory
    plus a sanitized project path, matching the Claude/claude-remember style
    slug including the leading dash on Unix paths.
    """
    from ostinote.env import Env

    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    store = tmp_path / "store"
    # data_dir is honored only from the trusted user layer; a project-layer
    # value pointing outside the repo would be rejected as an untrusted redirect.
    user_cfg = tmp_path / "user-config.json"
    user_cfg.write_text(json.dumps({"data_dir": str(store / "{slug}")}))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(user_cfg))
    (proj / ".ostinote" / "config.json").write_text(json.dumps({"share_worktrees": False}))
    env = Env(str(proj))
    import re

    # claude-remember / Claude Code slug scheme: leading dash kept.
    # Windows paths start with drive letters instead.
    expected_slug = re.sub(r"[^a-zA-Z0-9]", "-", str(proj))
    if os.name != "nt":
        assert expected_slug.startswith("-")
    assert env.data_dir == str(store / expected_slug)


def test_config_legacy_remember_keys(tmp_path, monkeypatch):
    """Normalize old claude-remember config keys to Ostinote's current names.

    Expected: legacy `ndc_*` keys populate the new compression settings, legacy
    keys disappear from the loaded config, and an explicit new key wins if both
    old and new names are present.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(
        json.dumps(
            {
                "cooldowns": {"ndc_seconds": 1800, "git_backup_seconds": 900},
                "features": {"ndc_compression": False},
            }
        )
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["compress_seconds"] == 1800
    assert "ndc_seconds" not in cfg["cooldowns"]
    assert cfg["features"]["hourly_compression"] is False
    assert "ndc_compression" not in cfg["features"]

    # a file that sets both names keeps the new name's value
    (proj / ".ostinote").mkdir()
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps({"cooldowns": {"ndc_seconds": 60, "compress_seconds": 7200}})
    )
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["compress_seconds"] == 7200


def test_project_config_cannot_set_summarizer_command(tmp_path, monkeypatch):
    """Ignore an untrusted project config that tries to set the summarizer.

    Expected: a cloned repo's `.ostinote/config.json` cannot inject
    `summarizer.command` (which runs as a subprocess on every save); the loaded
    value falls back to the default while a user-layer command is honored.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(json.dumps({"summarizer": {"command": ["trusted-engine"]}}))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps({"summarizer": {"command": ["rm", "-rf", "/"], "timeout": 1}})
    )

    cfg, guarded = config_mod.load_trusted(str(proj))
    # The malicious project command is dropped; the trusted user command wins,
    # and a sibling key the project set (timeout) still merges normally.
    assert cfg["summarizer"]["command"] == ["trusted-engine"]
    assert cfg["summarizer"]["timeout"] == 1
    assert ("data_dir",) not in guarded


def test_project_config_data_dir_is_flagged_guarded(tmp_path, monkeypatch):
    """Report when the project layer sets the guarded `data_dir` key.

    Expected: `load_trusted` keeps the value (the in-repo `.ostinote` workflow
    needs it) but flags it so `Env` can containment-check it; an unset key is
    not flagged.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(json.dumps({"data_dir": ".ostinote"}))

    cfg, guarded = config_mod.load_trusted(str(proj))
    assert cfg["data_dir"] == ".ostinote"
    assert ("data_dir",) in guarded


def test_env_rejects_project_data_dir_escaping_repo(tmp_path, monkeypatch):
    """Refuse a project-supplied data_dir that escapes the repo and home.

    Expected: a cloned repo pointing data_dir outside both the project root and
    ~/.ostinote (an arbitrary-write / Codex-sandbox-escape attempt) is ignored,
    and Env falls back to the default `~/.ostinote/projects/<slug>` layout.
    """
    from ostinote.env import Env, _slugify

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    escape = tmp_path / "evil"
    (proj / ".ostinote" / "config.json").write_text(json.dumps({"data_dir": str(escape), "share_worktrees": False}))

    env = Env(str(proj))
    expected = os.path.realpath(os.path.expanduser("~/.ostinote/projects/%s" % _slugify(str(proj))))
    assert env.data_dir == expected
    assert str(escape) not in env.data_dir


def test_env_allows_project_data_dir_inside_repo(tmp_path, monkeypatch):
    """Honor an in-repo project data_dir, the documented `.ostinote` workflow.

    Expected: a relative `.ostinote` data_dir resolves under the project root
    and is kept, because it does not escape the repo.
    """
    from ostinote.env import Env

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(json.dumps({"data_dir": ".ostinote", "share_worktrees": False}))

    env = Env(str(proj))
    assert env.data_dir == os.path.realpath(str(proj / ".ostinote"))


def test_env_trusts_user_data_dir_anywhere(tmp_path, monkeypatch):
    """Let the trusted user layer place memory wherever it likes.

    Expected: a user-config data_dir outside the repo and ~/.ostinote is used
    as-is — the containment check only applies to the untrusted project layer.
    """
    from ostinote.env import Env

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    user_cfg = tmp_path / "user.json"
    user_cfg.write_text(json.dumps({"data_dir": str(tmp_path / "elsewhere")}))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(user_cfg))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(json.dumps({"share_worktrees": False}))

    env = Env(str(proj))
    assert env.data_dir == os.path.realpath(str(tmp_path / "elsewhere"))


def test_costs_day_totals(tmp_path):
    """Summarize token and cost lines from daily memory logs.

    Expected: only `memory-YYYY-MM-DD.log` files with token lines count; totals
    aggregate calls, input, cache, output, and only the cost values actually
    reported by the model engine.
    """
    from ostinote import costs

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "memory-2026-06-09.log").write_text(
        "12:00:00 [save] tokens: 100+50cache→20out ($0.000123)\n"
        "12:30:00 [compress] tokens: 200+0cache→40out\n"
        "12:31:00 [hook] not a token line\n",
        encoding="utf-8",
    )
    (logs / "memory-2026-06-10.log").write_text("09:00:00 [hook] no calls today\n", encoding="utf-8")
    (logs / "background.log").write_text("[save] tokens: 9+9cache→9out ($9)\n", encoding="utf-8")

    days = costs.day_totals(str(logs))
    assert [d for d, _ in days] == ["2026-06-09"]  # only daily logs with calls
    totals = days[0][1]
    assert totals["calls"] == 2
    assert totals["input"] == 300
    assert totals["cache"] == 50
    assert totals["output"] == 60
    assert totals["cost"] == pytest.approx(0.000123)  # unreported cost not invented
