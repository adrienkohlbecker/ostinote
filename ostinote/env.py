"""Runtime environment: project root, data directory, config, logging.

``Env`` is the one object threaded through the pipeline. It resolves the
project root from the session cwd (collapsing git worktrees onto the main
checkout so parallel worktree sessions share one memory), loads layered
config, and owns the data-directory layout:

    <data>/
        now.md  today-YYYY-MM-DD.md  recent.md  archive.md
        ostinote.md (handoff)  identity.md  core-memories.md
        state/sessions/<agent>--<session>.json   per-session positions
        state/*.lock  state/last-compress.ts
        logs/memory-YYYY-MM-DD.log
"""

from __future__ import annotations

import os
import re
import subprocess
import time

from . import config as config_mod
from . import tzutil

# Hook crash log. Outside any data dir — written when Env construction
# itself may have failed — and never inside a project.
HOOK_ERRORS_PATH = os.path.expanduser("~/.ostinote/hook-errors.log")


def _slugify(path: str) -> str:
    # Same scheme as Claude Code's ~/.claude/projects/<slug> and
    # claude-remember's ~/.remember/<slug>, leading dash included, so
    # external-mode memory folders carry over between the tools by name.
    return re.sub(r"[^a-zA-Z0-9]", "-", path)


def _git_main_root(cwd: str) -> str:
    """Return the main checkout root for cwd, collapsing worktrees.

    Falls back to cwd when not in a git repo or git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            common = out.stdout.strip()
            if os.path.basename(common) == ".git":
                return os.path.dirname(common)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return cwd


class Env:
    def __init__(self, cwd: str):
        self.cwd = os.path.abspath(cwd)
        root = _git_main_root(self.cwd)
        cfg = config_mod.load(root)
        if not cfg["share_worktrees"]:
            root = self.cwd
            cfg = config_mod.load(root)
        self.project_root = root
        self.cfg = cfg
        self.tz = tzutil.get_tz(cfg["timezone"])

        # "{slug}" lets a single user-level setting keep every project's
        # memory outside its repo, e.g. "~/.ostinote/projects/{slug}".
        data_dir = os.path.expanduser(cfg["data_dir"].replace("{slug}", _slugify(root)))
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(root, data_dir)
        self.data_dir = data_dir

    # --- layout -------------------------------------------------------------

    @property
    def state_dir(self) -> str:
        return os.path.join(self.data_dir, "state")

    @property
    def sessions_dir(self) -> str:
        return os.path.join(self.state_dir, "sessions")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.data_dir, "logs")

    @property
    def now_file(self) -> str:
        return os.path.join(self.data_dir, "now.md")

    def today_file(self, date: str = "") -> str:
        return os.path.join(self.data_dir, "today-%s.md" % (date or self.today()))

    @property
    def recent_file(self) -> str:
        return os.path.join(self.data_dir, "recent.md")

    @property
    def archive_file(self) -> str:
        return os.path.join(self.data_dir, "archive.md")

    @property
    def handoff_file(self) -> str:
        return os.path.join(self.data_dir, "ostinote.md")

    @property
    def identity_file(self) -> str:
        return os.path.join(self.data_dir, "identity.md")

    @property
    def core_memories_file(self) -> str:
        return os.path.join(self.data_dir, "core-memories.md")

    def ensure_dirs(self) -> None:
        for d in (self.sessions_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)
        gitignore = os.path.join(self.data_dir, ".gitignore")
        if not os.path.exists(gitignore):
            try:
                with open(gitignore, "w", encoding="utf-8") as f:
                    f.write("*\n")
            except OSError:
                pass

    # --- time ----------------------------------------------------------------

    def today(self) -> str:
        return tzutil.today_str(self.tz)

    def time_now(self) -> str:
        return tzutil.time_str(self.tz, self.cfg["time_format"])

    # --- logging ---------------------------------------------------------------

    def log(self, component: str, message: str) -> None:
        self.ensure_dirs()
        ts = tzutil.now(self.tz).strftime("%H:%M:%S")
        line = "%s [%s] %s\n" % (ts, component, message)
        path = os.path.join(self.logs_dir, "memory-%s.log" % self.today())
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass

    def log_tokens(
        self, component: str, tk_in: int, tk_out: int, tk_cache: int, cost: float
    ) -> None:
        detail = "tokens: %d+%dcache→%dout" % (tk_in, tk_cache, tk_out)
        if cost:
            detail += " ($%.6f)" % cost
        self.log(component, detail)

    def rotate_logs(self) -> None:
        """Delete daily logs older than 30 days; cap background.log size."""
        try:
            cutoff = time.time() - 30 * 86400
            removed = 0
            for name in sorted(os.listdir(self.logs_dir)):
                path = os.path.join(self.logs_dir, name)
                if (
                    name.startswith("memory-")
                    and name.endswith(".log")
                    and os.path.getmtime(path) < cutoff
                ):
                    os.remove(path)
                    removed += 1
            if removed:
                self.log("rotate", "deleted %d logs older than 30 days" % removed)
            background = os.path.join(self.logs_dir, "background.log")
            if os.path.exists(background) and os.path.getsize(background) > 512 * 1024:
                with open(background, "w", encoding="utf-8"):
                    pass
        except OSError:
            return

    # branch of the *session* cwd (worktrees may be on different branches)
    def git_branch(self) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", self.cwd, "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            branch = out.stdout.strip()
            return branch if out.returncode == 0 and branch else "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
