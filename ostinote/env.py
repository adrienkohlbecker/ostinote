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
import stat
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
            ["git", "--no-optional-locks", "-C", cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"],
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


def worktree_root_for(cwd: str) -> str:
    """Return the current git worktree top-level for ``cwd``.

    Falls back to the absolute cwd outside git or when git is unavailable. This
    is the root where project-scoped agent config should be installed.
    """
    cwd = os.path.abspath(cwd)
    try:
        out = subprocess.run(
            ["git", "--no-optional-locks", "-C", cwd, "rev-parse", "--path-format=absolute", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return os.path.normpath(out.stdout.strip())
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
    worktree_root = worktree_root_for(cwd)
    main_root = _git_main_root(cwd)
    main_cfg, main_guarded = config_mod.load_trusted(main_root)
    worktree_cfg, worktree_guarded = config_mod.load_trusted(worktree_root)
    if not worktree_cfg["share_worktrees"] or not main_cfg["share_worktrees"]:
        return worktree_root, worktree_cfg, worktree_guarded
    return main_root, main_cfg, main_guarded


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
        default_dir = _resolve_data_dir(config_mod.DEFAULTS, project_root)
        safe_roots = (
            os.path.realpath(project_root),
            default_dir,
        )
        if not any(_within(root, resolved) for root in safe_roots):
            return default_dir
    return resolved


def data_dir_for(cwd: str) -> str:
    """Compute a project's validated memory directory without building an Env.

    Mirrors ``Env``'s root resolution and containment check so the Codex
    installer grants sandbox access to exactly the directory ``Env`` will write
    to — never an attacker-redirected one — without paying for tz/logging setup.
    """
    root, cfg, guarded = _resolve_root_config(cwd)
    return safe_data_dir(cfg, root, guarded)


def _logical_child(parent: str, child: str) -> bool:
    """Return whether ``child`` is lexically nested under ``parent``."""
    try:
        return os.path.commonpath([os.path.abspath(parent), os.path.abspath(child)]) == os.path.abspath(parent)
    except ValueError:
        return False


def ensure_child_dir(root: str, path: str) -> None:
    """Create ``path`` under ``root`` without traversing existing symlinks.

    Project-owned memory dirs may live inside a cloned repository, so existing
    path components are treated as untrusted. A symlinked state/log directory
    is refused instead of followed.
    """
    root_abs = os.path.abspath(root)
    path_abs = os.path.abspath(path)
    if not _logical_child(root_abs, path_abs):
        raise OSError("path escapes root: %s" % path)
    rel = os.path.relpath(path_abs, root_abs)
    current = root_abs
    if rel != ".":
        for part in rel.split(os.sep):
            current = os.path.join(current, part)
            if os.path.lexists(current):
                if os.path.islink(current):
                    raise OSError("refusing symlinked directory: %s" % current)
                if not os.path.isdir(current):
                    raise OSError("refusing non-directory path component: %s" % current)
                if not _within(os.path.realpath(root_abs), os.path.realpath(current)):
                    raise OSError("directory escapes root: %s" % current)
    os.makedirs(path_abs, exist_ok=True)


def validate_child_file(root: str, path: str) -> None:
    """Ensure ``path`` can be written as a regular child of ``root``."""
    ensure_child_dir(root, os.path.dirname(path))
    if os.path.lexists(path):
        if os.path.islink(path):
            raise OSError("refusing symlinked file: %s" % path)
        if not os.path.isfile(path):
            raise OSError("refusing non-regular file: %s" % path)
        if not _within(os.path.realpath(root), os.path.realpath(path)):
            raise OSError("file escapes root: %s" % path)


def open_regular_child(path: str, root: str, flags: int, mode: int = 0o600) -> int:
    """Open a regular file under ``root`` without following final symlinks."""
    if flags & os.O_CREAT:
        ensure_child_dir(root, os.path.dirname(path))
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if not _within(real_root, real_path):
        raise OSError("path escapes root: %s" % path)
    if os.path.islink(path):
        raise OSError("refusing symlink: %s" % path)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError("refusing non-regular file: %s" % path)
    except Exception:
        os.close(fd)
        raise
    return fd


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
        for d in (self.state_dir, self.sessions_dir, self.logs_dir):
            ensure_child_dir(self.data_dir, d)
        gitignore = os.path.join(self.data_dir, ".gitignore")
        if not os.path.exists(gitignore):
            with contextlib.suppress(OSError):
                fd = open_regular_child(gitignore, self.data_dir, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
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
        with contextlib.suppress(OSError):
            self.ensure_dirs()
            ts = tzutil.now(self.tz).strftime("%H:%M:%S")
            line = "%s [%s] %s\n" % (ts, component, message)
            path = os.path.join(self.logs_dir, "memory-%s.log" % self.today())
            fd = open_regular_child(path, self.data_dir, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
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
            self.ensure_dirs()
            cutoff = time.time() - 30 * 86400
            removed = 0
            for name in sorted(os.listdir(self.logs_dir)):
                path = os.path.join(self.logs_dir, name)
                if os.path.islink(path):
                    continue
                if name.startswith("memory-") and name.endswith(".log") and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            if removed:
                self.log("rotate", "deleted %d logs older than 30 days" % removed)
            background = os.path.join(self.logs_dir, "background.log")
            oversized_background = (
                os.path.exists(background)
                and not os.path.islink(background)
                and os.path.getsize(background) > 512 * 1024
            )
            if oversized_background:
                fd = open_regular_child(background, self.data_dir, os.O_WRONLY | os.O_TRUNC)
                os.close(fd)
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
                ["git", "--no-optional-locks", "-C", self.cwd, "symbolic-ref", "-q", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            branch = out.stdout.strip()
            return branch if out.returncode == 0 and branch else "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
