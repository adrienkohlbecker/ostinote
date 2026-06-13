import json
import os
import subprocess

from ostinote.env import Env
from tests.helpers import project_with_config, run_git


def test_git_main_root_normalizes_git_output(tmp_path, monkeypatch):
    """Normalize the path git reports before deriving the project root.

    Expected: git prints forward-slash, possibly non-canonical paths (on
    Windows too), so _git_main_root must normpath them — otherwise the slug
    derived from a worktree's resolved root differs from the main checkout's
    own, silently splitting the shared memory directory.
    """
    from ostinote import env as env_mod

    # Forward slashes and a redundant segment, as git may emit them.
    reported = "%s/repo/sub/../.git" % str(tmp_path).replace(os.sep, "/")
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=reported + "\n", stderr="")
    monkeypatch.setattr(env_mod.subprocess, "run", lambda *a, **k: fake)

    assert env_mod._git_main_root(str(tmp_path)) == os.path.normpath(str(tmp_path / "repo"))


def test_env_shares_worktree_memory_with_main_checkout(tmp_path):
    """Collapse a linked git worktree onto the main checkout's memory.

    Expected: with `share_worktrees: true` (the default), an Env built from a
    linked worktree resolves project_root to the main checkout, so parallel
    worktree sessions read and write one shared memory directory instead of
    silently splitting the project's memory per worktree.
    """
    main = project_with_config(tmp_path, {"share_worktrees": True}, name="main")
    run_git(main, "init", "-q")
    run_git(main, "add", ".")
    run_git(main, "commit", "-q", "-m", "init")
    run_git(main, "worktree", "add", "--detach", "-q", str(tmp_path / "wt"))

    env = Env(str(tmp_path / "wt"))

    assert env.project_root == str(main)
    assert env.data_dir == Env(str(main)).data_dir


def test_env_started_from_subdir_uses_worktree_root(tmp_path):
    """Do not derive a separate project identity from a subdirectory cwd."""
    main = project_with_config(tmp_path, {"share_worktrees": True}, name="main")
    subdir = main / "src" / "pkg"
    subdir.mkdir(parents=True)
    run_git(main, "init", "-q")
    run_git(main, "add", ".")
    run_git(main, "commit", "-q", "-m", "init")

    env = Env(str(subdir))

    assert env.project_root == str(main)
    assert env.data_dir == Env(str(main)).data_dir


def test_env_worktree_opt_out_uses_own_root(tmp_path):
    """Honor `share_worktrees: false` by re-resolving against the worktree.

    Expected: when the project config opts out of shared worktree memory, the
    worktree's own checkout becomes the project root, giving it an independent
    memory directory.
    """
    main = project_with_config(tmp_path, {"share_worktrees": False}, name="main")
    run_git(main, "init", "-q")
    run_git(main, "add", ".")
    run_git(main, "commit", "-q", "-m", "init")
    run_git(main, "worktree", "add", "--detach", "-q", str(tmp_path / "wt"))

    env = Env(str(tmp_path / "wt"))

    assert env.project_root == str(tmp_path / "wt")
    assert env.data_dir != Env(str(main)).data_dir


def test_env_worktree_local_config_can_opt_out(tmp_path):
    """Let a linked worktree's own config opt out of shared memory."""
    main = project_with_config(tmp_path, {"share_worktrees": True}, name="main")
    worktree = tmp_path / "wt"
    run_git(main, "init", "-q")
    run_git(main, "add", ".")
    run_git(main, "commit", "-q", "-m", "init")
    run_git(main, "worktree", "add", "--detach", "-q", str(worktree))
    (worktree / ".ostinote" / "config.json").write_text(json.dumps({"share_worktrees": False}), encoding="utf-8")

    env = Env(str(worktree))

    assert env.project_root == str(worktree)
    assert env.data_dir != Env(str(main)).data_dir
