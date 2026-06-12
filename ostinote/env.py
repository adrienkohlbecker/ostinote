"""Runtime environment: project root, data directory, config, logging.

``Env`` is the one object threaded through the pipeline. It resolves the
project root from the session cwd (collapsing git worktrees onto the main
checkout so parallel worktree sessions share one memory), loads layered
config, and owns the data-directory layout:

    <data>/
        now.md  today-YYYY-MM-DD.md  recent.md  archive.md
        identity.md  core-memories.md
        state/sessions/<agent>--<session>.json   per-session positions
        state/*.lock  state/last-compress.ts
        logs/memory-YYYY-MM-DD.log
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import time

from . import config as config_mod
from . import tzutil

# Hook crash log. Outside any data dir — written when Env construction
# itself may have failed — and never inside a project. Canonicalized so a
# symlink planted at this path cannot redirect the append elsewhere.
HOOK_ERRORS_PATH = os.path.realpath(os.path.expanduser("~/.ostinote/hook-errors.log"))


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
            # git prints forward-slash paths even on Windows; normalize so the
            # root (and the slug derived from it) matches an os.path-built cwd,
            # otherwise worktree sessions would split off their own memory dir.
            common = os.path.normpath(out.stdout.strip())
            if os.path.basename(common) == ".git":
                return os.path.dirname(common)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return cwd


def _resolve_root_config(cwd: str) -> tuple[str, dict, set]:
    """Resolve (project_root, cfg, guarded) for a session cwd.

    The root is the main checkout (worktrees collapsed) so parallel worktree
    sessions share one memory — unless the resolved project opts out via
    ``share_worktrees: false``, in which case the worktree's own checkout is
    used instead. ``guarded`` is the set of untrusted project keys that need
    validation (see ``config.load_trusted``).
    """
    cwd = os.path.abspath(cwd)
    root = _git_main_root(cwd)
    cfg, guarded = config_mod.load_trusted(root)
    if not cfg["share_worktrees"]:
        # Opted out of shared memory: re-resolve against this worktree's own
        # checkout, which may carry a different project config (including this
        # flag itself). Reloading is required, not redundant — the project
        # layer can differ between the two roots; the user layer cannot.
        root = cwd
        cfg, guarded = config_mod.load_trusted(root)
    return root, cfg, guarded


def _resolve_data_dir(cfg: dict, project_root: str) -> str:
    """Expand ``data_dir`` to an absolute, symlink-resolved path.

    ``{slug}`` becomes the dashed project path and ``~`` is expanded; a relative
    value resolves against the project root. The result is canonicalized with
    ``realpath`` so a session reached through a symlinked storage path still
    writes to one identity. The project *slug* itself is deliberately not
    realpath'd (see ``_slugify``) — only the final storage path is.
    """
    data_dir = os.path.expanduser(cfg["data_dir"].replace("{slug}", _slugify(project_root)))
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)
    return os.path.realpath(data_dir)


def _within(parent: str, child: str) -> bool:
    """Return True if ``child`` is ``parent`` or nested under it.

    Both paths should already be absolute/realpath'd. Returns False across
    Windows drives.
    """
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:
        return False


def safe_data_dir(cfg: dict, project_root: str, guarded: set) -> str:
    """Resolve the memory directory, refusing an untrusted redirect.

    When the *project* layer set ``data_dir`` and the resolved path escapes both
    the project root and ``~/.ostinote``, it is treated as a cloned-repo attempt
    to redirect writes (or to widen the Codex sandbox via the installer) and the
    built-in default layout is used instead. A ``data_dir`` from the trusted
    user layer is honored wherever it points.
    """
    resolved = _resolve_data_dir(cfg, project_root)
    if ("data_dir",) in guarded:
        safe_roots = (
            os.path.realpath(project_root),
            os.path.realpath(os.path.expanduser("~/.ostinote")),
        )
        if not any(_within(root, resolved) for root in safe_roots):
            return _resolve_data_dir(config_mod.DEFAULTS, project_root)
    return resolved


def data_dir_for(cwd: str) -> str:
    """Compute a project's validated memory directory without building an Env.

    Mirrors ``Env``'s root resolution and containment check so the Codex
    installer grants sandbox access to exactly the directory ``Env`` will write
    to — never an attacker-redirected one — without paying for tz/logging setup.
    """
    root, cfg, guarded = _resolve_root_config(cwd)
    return safe_data_dir(cfg, root, guarded)


class Env:
    """Resolved per-session environment: project root, config, and data layout.

    Construct one per hook invocation from the session cwd; everything
    downstream (saving, compression, recovery, logging) reads paths and
    settings from it rather than re-resolving them.
    """

    def __init__(self, cwd: str):
        """Resolve project root, validated data dir, config, and timezone for ``cwd``.

        Runs git to collapse worktrees onto the main checkout and reads both
        config layers, so construction touches the filesystem and may spawn a
        subprocess; it does not create any directories.
        """
        self.cwd = os.path.abspath(cwd)
        root, cfg, guarded = _resolve_root_config(self.cwd)
        self.project_root = root
        self.cfg = cfg
        self.tz = tzutil.get_tz(cfg["timezone"])
        self.data_dir = safe_data_dir(cfg, root, guarded)

    # --- layout -------------------------------------------------------------

    @property
    def state_dir(self) -> str:
        """Directory for locks, timestamps, and per-session state."""
        return os.path.join(self.data_dir, "state")

    @property
    def sessions_dir(self) -> str:
        """Directory of per-(agent, session) resume-position files."""
        return os.path.join(self.state_dir, "sessions")

    @property
    def logs_dir(self) -> str:
        """Directory for daily activity logs and background.log."""
        return os.path.join(self.data_dir, "logs")

    @property
    def now_file(self) -> str:
        """Path of now.md, the raw per-save notes awaiting compression."""
        return os.path.join(self.data_dir, "now.md")

    def today_file(self, date: str = "") -> str:
        """Path of the daily summary file for ``date`` (default: today in the configured tz)."""
        return os.path.join(self.data_dir, "today-%s.md" % (date or self.today()))

    @property
    def recent_file(self) -> str:
        """Path of recent.md, consolidated summaries of the last few days."""
        return os.path.join(self.data_dir, "recent.md")

    @property
    def archive_file(self) -> str:
        """Path of archive.md, long-term history aged out of recent.md."""
        return os.path.join(self.data_dir, "archive.md")

    @property
    def identity_file(self) -> str:
        """Path of identity.md, the standing project description injected into sessions."""
        return os.path.join(self.data_dir, "identity.md")

    @property
    def core_memories_file(self) -> str:
        """Path of core-memories.md, pinned facts that never age out."""
        return os.path.join(self.data_dir, "core-memories.md")

    def ensure_dirs(self) -> None:
        """Create the data-directory tree and drop a catch-all .gitignore.

        Idempotent; the .gitignore keeps an in-repo data dir (``data_dir:
        ".ostinote"``) out of version control. Failure to write it is ignored.
        """
        for d in (self.sessions_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)
        gitignore = os.path.join(self.data_dir, ".gitignore")
        if not os.path.exists(gitignore):
            with contextlib.suppress(OSError):
                with open(gitignore, "w", encoding="utf-8") as f:
                    f.write("*\n")

    # --- time ----------------------------------------------------------------

    def today(self) -> str:
        """Return today's date as YYYY-MM-DD in the configured timezone."""
        return tzutil.today_str(self.tz)

    def time_now(self) -> str:
        """Return the current clock time formatted per the ``time_format`` config."""
        return tzutil.time_str(self.tz, self.cfg["time_format"])

    # --- logging ---------------------------------------------------------------

    def log(self, component: str, message: str) -> None:
        """Append a timestamped line to today's memory-YYYY-MM-DD.log.

        Creates the data directories on first use; write errors are swallowed
        so logging can never break a hook.
        """
        self.ensure_dirs()
        ts = tzutil.now(self.tz).strftime("%H:%M:%S")
        line = "%s [%s] %s\n" % (ts, component, message)
        path = os.path.join(self.logs_dir, "memory-%s.log" % self.today())
        with contextlib.suppress(OSError):
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    def log_tokens(self, component: str, tk_in: int, tk_out: int, tk_cache: int, cost: float) -> None:
        """Log a summarizer call's token counts (and dollar cost, when known)."""
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
                if name.startswith("memory-") and name.endswith(".log") and os.path.getmtime(path) < cutoff:
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
        """Return the current git branch of the session cwd, or "unknown".

        Uses the session cwd, not the project root, because a worktree session
        may be on a different branch than the main checkout. "unknown" covers
        detached HEAD, non-repos, and git failures alike.
        """
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
