"""Per-session save positions and cross-process locks.

Every (agent, session) pair gets its own state file under
``state/sessions/``, so any number of parallel sessions — across both
agents — track their own resume positions independently. Mutations of the
shared memory files (now.md, today-*.md) are serialized by an atomic
pid-lock instead.
"""

from __future__ import annotations

import json
import os
import re
import time


def _sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "-", value)[:120]


class SessionState:
    """Resume position and save timestamps for one (agent, session)."""

    def __init__(self, path: str, agent: str, session_id: str):
        """Build a zeroed state for ``path``; use ``load`` to read what's on disk."""
        self.path = path
        self.agent = agent
        self.session_id = session_id
        self.transcript_path = ""
        self.line = 0  # raw transcript line offset already saved
        self.last_attempt_ts = 0.0  # last time a save got past the cooldown
        self.last_save_ts = 0.0  # last time content was actually written

    @classmethod
    def load(cls, sessions_dir: str, agent: str, session_id: str) -> SessionState:
        """Load the state file for (agent, session), or a fresh zeroed state.

        Never raises: a missing or corrupt file yields default values (line 0,
        empty transcript path), which makes the next save start from the top of
        the transcript. The stored transcript path is realpath'd defensively.
        """
        path = os.path.join(sessions_dir, "%s--%s.json" % (_sanitize(agent), _sanitize(session_id)))
        state = cls(path, agent, session_id)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Canonicalize stored paths defensively; "" must stay "" because
            # realpath("") resolves to the process cwd.
            raw_path = data.get("transcript_path", "")
            state.transcript_path = os.path.realpath(raw_path) if raw_path else ""
            state.line = int(data.get("line", 0))
            state.last_attempt_ts = float(data.get("last_attempt_ts", 0))
            state.last_save_ts = float(data.get("last_save_ts", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return state

    def save(self) -> None:
        """Write the state atomically (tmp file + ``os.replace``).

        Concurrent writers of the same session are safe: each pid uses its own
        tmp file, so the last writer wins without corrupting the JSON.
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Per-pid tmp name: a hook and a background save may write the same
        # session state concurrently; a shared tmp path makes one of them
        # crash when the other's os.replace consumes it. Last writer wins.
        tmp = "%s.%d.tmp" % (self.path, os.getpid())
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "agent": self.agent,
                    "session_id": self.session_id,
                    "transcript_path": self.transcript_path,
                    "line": self.line,
                    "last_attempt_ts": self.last_attempt_ts,
                    "last_save_ts": self.last_save_ts,
                },
                f,
            )
        os.replace(tmp, self.path)


def all_states(sessions_dir: str) -> list[SessionState]:
    """Return every readable session state under ``sessions_dir``, sorted by filename.

    Unreadable or malformed files are skipped, and a missing directory yields
    an empty list, so callers (recovery, status) always get a usable list.
    """
    states = []
    try:
        names = os.listdir(sessions_dir)
    except OSError:
        return states
    for name in sorted(names):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(sessions_dir, name), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        state = SessionState.load(sessions_dir, data.get("agent", "?"), data.get("session_id", "?"))
        states.append(state)
    return states


class PidLock:
    """Atomic create-or-fail lock; stale locks (dead pid) are taken over."""

    def __init__(self, path: str):
        """Wrap the lock file at ``path``; nothing is acquired until ``acquire``."""
        self.path = path
        self.held = False

    def acquire(self) -> bool:
        """Try to take the lock without blocking; return whether it was taken.

        Creation is atomic (``O_CREAT | O_EXCL``). If the file already exists
        but its pid is dead, the stale lock is removed and acquisition retried
        once; a live holder makes this return False immediately.
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(str(os.getpid()))
                self.held = True
                return True
            except FileExistsError:
                if not self._steal_if_stale():
                    return False
        return False

    def _steal_if_stale(self) -> bool:
        try:
            with open(self.path, encoding="utf-8") as f:
                pid = int(f.read().strip() or "0")
        except (OSError, ValueError):
            pid = 0
        if pid and _pid_alive(pid):
            return False
        try:
            os.remove(self.path)
        except OSError:
            pass
        return True

    def release(self) -> None:
        """Drop the lock if this instance holds it; safe to call repeatedly."""
        if self.held:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.held = False

    def __enter__(self) -> bool:
        """Attempt acquisition; the ``with ... as held`` value must be checked."""
        return self.acquire()

    def __exit__(self, *exc) -> None:
        """Release the lock (a no-op when acquisition failed)."""
        self.release()


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) on Windows TERMINATES the process — never use it.
        import ctypes

        SYNCHRONIZE = 0x00100000
        WAIT_TIMEOUT = 0x00000102
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def read_ts(path: str) -> float:
    """Read a Unix timestamp from a marker file; 0.0 if missing or malformed.

    The 0.0 fallback means "never happened", so time-based cooldowns treat a
    lost marker as long expired.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return float(f.read().strip() or "0")
    except (OSError, ValueError):
        return 0.0


def write_ts(path: str) -> None:
    """Write the current Unix time to a marker file, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(time.time()))
