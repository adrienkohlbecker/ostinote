import json
import os
import subprocess

import pytest

from ostinote import config as config_mod
from ostinote.env import Env, _slugify
from ostinote.state import PidLock, SessionState
from tests.helpers import expected_slug


def _project(tmp_path, cfg):
    """Create a project directory holding the given .ostinote/config.json.

    The config dict is the one input that distinguishes the Env/config tests
    from each other; this keeps it visible at the call site.
    """
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return proj


def _git(cwd, *args):
    """Run git in ``cwd`` with identity flags so commits work in the bare test HOME."""
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.email=test@test", "-c", "user.name=test", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


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


def test_pid_lock_steals_garbage_content(tmp_path):
    """Recover from a lock file corrupted by a crash mid-write.

    Expected: unparseable lock content is treated as stale rather than as a
    live holder, so a corrupted lock cannot block all future saves.
    """
    path = str(tmp_path / "x.lock")
    with open(path, "w") as f:
        f.write("not-a-pid")
    assert PidLock(path).acquire()


def test_pid_lock_does_not_steal_from_live_process(tmp_path):
    """Never steal a lock whose recorded PID is alive.

    Expected: a lock naming this very test process is refused — stealing from
    a live holder would let two background saves interleave writes to the
    shared memory files.
    """
    path = str(tmp_path / "x.lock")
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    assert not PidLock(path).acquire()


# --- Config ---------------------------------------------------------------------------


def test_config_project_overrides(tmp_path, monkeypatch):
    """Merge default, user, and project config layers in the right order.

    Expected: project values override user values, user values still apply when
    project config omits them, and nested defaults survive partial overrides.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(json.dumps({"cooldowns": {"save_seconds": 60}, "timezone": "UTC"}))
    proj = _project(tmp_path, {"cooldowns": {"save_seconds": 30}})
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
    proj = _project(tmp_path, {"share_worktrees": False})
    store = tmp_path / "store"
    # data_dir is honored only from the trusted user layer; a project-layer
    # value pointing outside the repo would be rejected as an untrusted redirect.
    user_cfg = tmp_path / "user-config.json"
    user_cfg.write_text(json.dumps({"data_dir": str(store / "{slug}")}))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(user_cfg))
    env = Env(str(proj))

    # claude-remember / Claude Code slug scheme: leading dash kept.
    # Windows paths start with drive letters instead.
    slug = expected_slug(str(proj))
    if os.name != "nt":
        assert slug.startswith("-")
    assert env.data_dir == str(store / slug)


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
    proj = _project(tmp_path, {"summarizer": {"command": ["rm", "-rf", "/"], "timeout": 1}})

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
    proj = _project(tmp_path, {"data_dir": ".ostinote"})

    cfg, guarded = config_mod.load_trusted(str(proj))
    assert cfg["data_dir"] == ".ostinote"
    assert ("data_dir",) in guarded


def test_env_rejects_project_data_dir_escaping_repo(tmp_path, monkeypatch):
    """Refuse a project-supplied data_dir that escapes the repo and home.

    Expected: a cloned repo pointing data_dir outside both the project root and
    ~/.ostinote (an arbitrary-write / Codex-sandbox-escape attempt) is ignored,
    and Env falls back to the default `~/.ostinote/projects/<slug>` layout.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    escape = tmp_path / "evil"
    proj = _project(tmp_path, {"data_dir": str(escape), "share_worktrees": False})

    env = Env(str(proj))
    expected = os.path.realpath(os.path.expanduser("~/.ostinote/projects/%s" % _slugify(str(proj))))
    assert env.data_dir == expected
    assert str(escape) not in env.data_dir


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require privileges on Windows")
def test_env_rejects_project_data_dir_symlink_escape(tmp_path, monkeypatch):
    """Refuse a project data_dir that escapes the repo through a symlink.

    Expected: `data_dir: "memdir"` where `memdir` is a repo-relative symlink to
    a directory outside the repo (git preserves absolute symlink targets in
    clones) is caught by the realpath-based containment check, and Env falls
    back to the default slug layout — a check on the unresolved path would
    grant the cloned repo an arbitrary write target.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    proj = _project(tmp_path, {"data_dir": "memdir", "share_worktrees": False})
    escape = tmp_path / "evil"
    escape.mkdir()
    os.symlink(str(escape), str(proj / "memdir"))

    env = Env(str(proj))
    expected = os.path.realpath(os.path.expanduser("~/.ostinote/projects/%s" % _slugify(str(proj))))
    assert env.data_dir == expected
    assert os.path.realpath(str(escape)) != env.data_dir


def test_env_allows_project_data_dir_inside_repo(tmp_path, monkeypatch):
    """Honor an in-repo project data_dir, the documented `.ostinote` workflow.

    Expected: a relative `.ostinote` data_dir resolves under the project root
    and is kept, because it does not escape the repo.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    proj = _project(tmp_path, {"data_dir": ".ostinote", "share_worktrees": False})

    env = Env(str(proj))
    assert env.data_dir == os.path.realpath(str(proj / ".ostinote"))


def test_env_trusts_user_data_dir_anywhere(tmp_path, monkeypatch):
    """Let the trusted user layer place memory wherever it likes.

    Expected: a user-config data_dir outside the repo and ~/.ostinote is used
    as-is — the containment check only applies to the untrusted project layer.
    """
    user_cfg = tmp_path / "user.json"
    user_cfg.write_text(json.dumps({"data_dir": str(tmp_path / "elsewhere")}))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(user_cfg))
    proj = _project(tmp_path, {"share_worktrees": False})

    env = Env(str(proj))
    assert env.data_dir == os.path.realpath(str(tmp_path / "elsewhere"))


# --- Worktree root resolution ------------------------------------------------------------


def test_env_shares_worktree_memory_with_main_checkout(tmp_path):
    """Collapse a linked git worktree onto the main checkout's memory.

    Expected: with `share_worktrees: true` (the default), an Env built from a
    linked worktree resolves project_root to the main checkout, so parallel
    worktree sessions read and write one shared memory directory instead of
    silently splitting the project's memory per worktree.
    """
    main = tmp_path / "main"
    (main / ".ostinote").mkdir(parents=True)
    (main / ".ostinote" / "config.json").write_text(json.dumps({"share_worktrees": True}), encoding="utf-8")
    _git(main, "init", "-q")
    _git(main, "add", ".")
    _git(main, "commit", "-q", "-m", "init")
    _git(main, "worktree", "add", "-q", str(tmp_path / "wt"))

    env = Env(str(tmp_path / "wt"))

    assert env.project_root == str(main)
    assert env.data_dir == Env(str(main)).data_dir


def test_env_worktree_opt_out_uses_own_root(tmp_path):
    """Honor `share_worktrees: false` by re-resolving against the worktree.

    Expected: when the project config opts out of shared worktree memory, the
    worktree's own checkout becomes the project root, giving it an independent
    memory directory.
    """
    main = tmp_path / "main"
    (main / ".ostinote").mkdir(parents=True)
    (main / ".ostinote" / "config.json").write_text(json.dumps({"share_worktrees": False}), encoding="utf-8")
    _git(main, "init", "-q")
    _git(main, "add", ".")
    _git(main, "commit", "-q", "-m", "init")
    _git(main, "worktree", "add", "-q", str(tmp_path / "wt"))

    env = Env(str(tmp_path / "wt"))

    assert env.project_root == str(tmp_path / "wt")
    assert env.data_dir != Env(str(main)).data_dir
