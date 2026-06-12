import json
import subprocess
import types

import pytest

from ostinote import summarize

_OK_JSON = json.dumps({"result": "ok", "usage": {}})


def _fake_run(captured, returncode=0, stdout=_OK_JSON, exc=None):
    """Build a subprocess.run stand-in that records its invocation."""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        if exc is not None:
            raise exc
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="boom-details")

    return fake_run


def test_summarizer_never_persists_sessions(monkeypatch):
    """Keep nested Claude summarizer calls from creating Claude sessions.

    Expected: the default command includes `--no-session-persistence`, custom
    Claude commands get that flag appended automatically, and non-Claude
    summarizer commands are left alone.
    """
    assert "--no-session-persistence" in summarize.DEFAULT_COMMAND

    captured = {}
    monkeypatch.setattr(summarize.subprocess, "run", _fake_run(captured))
    # A custom claude command missing the flag gets it appended.
    cfg = {"summarizer": {"command": ["claude", "-p", "--model", "haiku"], "timeout": 5}}
    summarize.call_model("hi", cfg)
    assert "--no-session-persistence" in captured["cmd"]
    # Non-claude engines are left alone.
    cfg = {"summarizer": {"command": ["my-engine", "--fast"], "timeout": 5}}
    summarize.call_model("hi", cfg)
    assert "--no-session-persistence" not in captured["cmd"]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        pytest.param({"returncode": 1}, "exited 1", id="nonzero-exit"),
        pytest.param({"exc": subprocess.TimeoutExpired(cmd="x", timeout=5)}, "timed out", id="timeout"),
        pytest.param({"exc": OSError("no such file")}, "failed to start", id="os-error"),
    ],
)
def test_call_model_normalizes_failures_to_runtime_error(monkeypatch, kwargs, match):
    """Convert every summarizer failure mode into RuntimeError.

    Expected: non-zero exit, timeout, and a missing executable all surface as
    RuntimeError — the pipeline's whole failure handling is `except
    RuntimeError`, so any other escaping exception type would crash the
    detached background save unlogged.
    """
    captured = {}
    monkeypatch.setattr(summarize.subprocess, "run", _fake_run(captured, **kwargs))
    cfg = {"summarizer": {"command": ["my-engine"], "timeout": 5}}

    with pytest.raises(RuntimeError, match=match):
        summarize.call_model("hi", cfg)


def test_call_model_strips_claudecode_from_child_env(monkeypatch):
    """Drop CLAUDECODE so nested claude invocations are not refused.

    Expected: when a Claude Code hook spawned this process, the inherited
    CLAUDECODE variable is removed from the summarizer subprocess env (it
    blocks nested claude sessions) while the rest of the environment passes
    through.
    """
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("OSTINOTE_TEST_SENTINEL", "kept")
    captured = {}
    monkeypatch.setattr(summarize.subprocess, "run", _fake_run(captured))
    cfg = {"summarizer": {"command": ["my-engine"], "timeout": 5}}

    summarize.call_model("hi", cfg)

    child_env = captured["kwargs"]["env"]
    assert "CLAUDECODE" not in child_env
    assert child_env["OSTINOTE_TEST_SENTINEL"] == "kept"


def test_call_model_empty_command_falls_back_to_default(monkeypatch):
    """Use the built-in claude/haiku command when none is configured.

    Expected: an empty `summarizer.command` resolves to DEFAULT_COMMAND
    verbatim (executable resolution stubbed out), so a fresh install with no
    user config still summarizes.
    """
    captured = {}
    monkeypatch.setattr(summarize.subprocess, "run", _fake_run(captured))
    monkeypatch.setattr(summarize.shutil, "which", lambda _name: None)
    cfg = {"summarizer": {"command": [], "timeout": 5}}

    summarize.call_model("hi", cfg)

    assert captured["cmd"] == summarize.DEFAULT_COMMAND
    assert captured["kwargs"]["timeout"] == 5

