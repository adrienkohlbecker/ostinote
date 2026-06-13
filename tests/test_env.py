import json
import os

import pytest

from ostinote import config as config_mod
from ostinote.env import Env, _slugify
from tests.helpers import project_with_config


def test_env_rejects_project_data_dir_escaping_repo(tmp_path, monkeypatch):
    """Refuse a project-supplied data_dir that escapes the repo and home.

    Expected: a cloned repo pointing data_dir outside both the project root and
    ~/.ostinote (an arbitrary-write / Codex-sandbox-escape attempt) is ignored,
    and Env falls back to the default `~/.ostinote/projects/<slug>` layout.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    escape = tmp_path / "evil"
    proj = project_with_config(tmp_path, {"data_dir": str(escape), "share_worktrees": False})

    env = Env(str(proj))
    expected = os.path.realpath(os.path.expanduser("~/.ostinote/projects/%s" % _slugify(str(proj))))
    assert env.data_dir == expected
    assert str(escape) not in env.data_dir


def test_env_rejects_project_data_dir_pointing_at_other_memory_slug(tmp_path, monkeypatch):
    """Refuse a project data_dir under another ~/.ostinote project slug.

    Expected: cloned repos may use their own default external memory directory,
    but cannot point at a sibling project's memory inside the same ostinote
    home, because that would leak and contaminate cross-project memory.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    victim = home / ".ostinote" / "projects" / "victim"
    proj = project_with_config(tmp_path, {"data_dir": str(victim), "share_worktrees": False})

    env = Env(str(proj))

    expected = os.path.realpath(os.path.expanduser("~/.ostinote/projects/%s" % _slugify(str(proj))))
    assert env.data_dir == expected
    assert env.data_dir != os.path.realpath(str(victim))


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
    proj = project_with_config(tmp_path, {"data_dir": "memdir", "share_worktrees": False})
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
    proj = project_with_config(tmp_path, {"data_dir": ".ostinote", "share_worktrees": False})

    env = Env(str(proj))
    assert env.data_dir == os.path.realpath(str(proj / ".ostinote"))


@pytest.mark.skipif(os.name == "nt", reason="symlink behavior differs on Windows")
def test_env_logging_refuses_symlinked_log_file(tmp_path, monkeypatch):
    """Do not append hook logs through symlinks in an in-repo data dir."""
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    proj = project_with_config(tmp_path, {"data_dir": ".ostinote", "share_worktrees": False})
    logs = proj / ".ostinote" / "logs"
    logs.mkdir()
    target = tmp_path / "target.log"
    target.write_text("secret", encoding="utf-8")
    today = Env(str(proj)).today()
    os.symlink(target, logs / ("memory-%s.log" % today))

    Env(str(proj)).log("test", "hello")

    assert target.read_text(encoding="utf-8") == "secret"


@pytest.mark.skipif(os.name == "nt", reason="symlink behavior differs on Windows")
@pytest.mark.parametrize("child", ["state", "logs"])
def test_env_refuses_symlinked_state_or_logs_dirs(tmp_path, monkeypatch, child):
    """Do not create sessions or logs under symlinked data-dir children."""
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    proj = project_with_config(tmp_path, {"data_dir": ".ostinote", "share_worktrees": False})
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, proj / ".ostinote" / child)

    with pytest.raises(OSError, match="symlinked directory"):
        Env(str(proj)).ensure_dirs()


def test_env_trusts_user_data_dir_anywhere(tmp_path, monkeypatch):
    """Let the trusted user layer place memory wherever it likes.

    Expected: a user-config data_dir outside the repo and ~/.ostinote is used
    as-is — the containment check only applies to the untrusted project layer.
    """
    user_cfg = tmp_path / "user.json"
    user_cfg.write_text(json.dumps({"data_dir": str(tmp_path / "elsewhere")}))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(user_cfg))
    proj = project_with_config(tmp_path, {"share_worktrees": False})

    env = Env(str(proj))
    assert env.data_dir == os.path.realpath(str(tmp_path / "elsewhere"))
