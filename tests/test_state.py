import os

from ostinote.state import PidLock, SessionState


def test_session_state_roundtrip(tmp_path):
    """Persist and reload one session's save position.

    Expected: line number, transcript path, session cwd, and save timestamp
    survive a disk round trip, and another agent using the same session id gets
    a separate state file.
    """
    sessions = str(tmp_path / "sessions")
    # load() realpaths the stored transcript, so the expectation must be a
    # canonical platform path — a bare "/t.jsonl" gains a drive on Windows.
    transcript = os.path.realpath(str(tmp_path / "t.jsonl"))
    s = SessionState.load(sessions, "codex", "id-1")
    assert s.line == 0
    s.line = 42
    s.transcript_path = transcript
    s.cwd = str(tmp_path / "proj")
    s.last_save_ts = 123.0
    s.save()
    s2 = SessionState.load(sessions, "codex", "id-1")
    assert (s2.line, s2.transcript_path, s2.cwd, s2.last_save_ts) == (42, transcript, str(tmp_path / "proj"), 123.0)
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


def test_session_state_sanitizes_filename_and_recovers_corrupt_file(tmp_path):
    """Keep session-state filenames bounded and recover from corrupt JSON."""
    sessions = str(tmp_path / "sessions")
    long_session = "id/with/slashes-" + "x" * 200
    state = SessionState.load(sessions, "co/dex", long_session)

    assert os.path.dirname(state.path) == sessions
    assert os.path.basename(state.path).startswith("co-dex--id-with-slashes-")
    assert "/" not in os.path.basename(state.path)

    state.line = 7
    state.save()
    with open(state.path, "w", encoding="utf-8") as f:
        f.write("{")

    reloaded = SessionState.load(sessions, "co/dex", long_session)
    assert reloaded.line == 0
    assert reloaded.transcript_path == ""


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
