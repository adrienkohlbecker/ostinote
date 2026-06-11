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
        self.path = path
        self.agent = agent
        self.session_id = session_id
        self.transcript_path = ""
        self.line = 0  # raw transcript line offset already saved
        self.last_attempt_ts = 0.0  # last time a save got past the cooldown
        self.last_save_ts = 0.0  # last time content was actually written

    @classmethod
    def load(cls, sessions_dir: str, agent: str, session_id: str) -> SessionState:
        path = os.path.join(sessions_dir, "%s--%s.json" % (_sanitize(agent), _sanitize(session_id)))
        state = cls(path, agent, session_id)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            state.transcript_path = data.get("transcript_path", "")
            state.line = int(data.get("line", 0))
            state.last_attempt_ts = float(data.get("last_attempt_ts", 0))
            state.last_save_ts = float(data.get("last_save_ts", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return state

    def save(self) -> None:
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
        self.path = path
        self.held = False

    def acquire(self) -> bool:
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
        if self.held:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.held = False

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc) -> None:
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
    try:
        with open(path, encoding="utf-8") as f:
            return float(f.read().strip() or "0")
    except (OSError, ValueError):
        return 0.0


def write_ts(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(time.time()))
