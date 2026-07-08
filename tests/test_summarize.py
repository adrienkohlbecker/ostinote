import json
import subprocess
import tempfile
import types

import pytest

from ostinote import summarize

_OK_JSON = json.dumps({"result": "ok", "usage": {}})


def _fake_run(captured, returncode=0, stdout=_OK_JSON, stderr="boom-details", exc=None):
    """Build a subprocess.run stand-in that records its invocation."""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        if exc is not None:
            raise exc
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return fake_run


def test_summarizer_never_persists_sessions(monkeypatch):
    """Keep nested summarizer calls from persisting transcripts as sessions.

    Expected: both built-in commands carry their no-persistence flag, custom
    claude/codex commands get the flag appended automatically, and other
    summarizer commands are left alone.
    """
    assert "--no-session-persistence" in summarize.DEFAULT_COMMAND
    assert "--ephemeral" in summarize.DEFAULT_COMMAND_CODEX

    captured = {}
    monkeypatch.setattr(summarize.subprocess, "run", _fake_run(captured))
    # A custom claude command missing the flag gets it appended.
    cfg = {"summarizer": {"command": ["claude", "-p", "--model", "haiku"], "timeout": 5}}
    summarize.call_model("hi", cfg)
    assert "--no-session-persistence" in captured["cmd"]
    # Same for a custom codex command.
    cfg = {"summarizer": {"command": ["codex", "exec", "--json", "-"], "timeout": 5}}
    summarize.call_model("hi", cfg)
    assert "--ephemeral" in captured["cmd"]
    # Other engines are left alone.
    cfg = {"summarizer": {"command": ["my-engine", "--fast"], "timeout": 5}}
    summarize.call_model("hi", cfg)
    assert "--no-session-persistence" not in captured["cmd"]
    assert "--ephemeral" not in captured["cmd"]


def test_resolve_command_explicit_command_wins():
    """Keep an explicit summarizer.command authoritative over engine selection."""
    cfg = {"summarizer": {"command": ["my-engine"], "engine": "codex", "timeout": 5}}
    assert summarize.resolve_command(cfg) == ["my-engine"]


@pytest.mark.parametrize(
    "engine,expected",
    [
        pytest.param("claude", summarize.DEFAULT_COMMAND, id="claude"),
        pytest.param("codex", summarize.DEFAULT_COMMAND_CODEX, id="codex"),
    ],
)
def test_resolve_command_engine_picks_builtin(monkeypatch, engine, expected):
    """Map summarizer.engine onto the matching built-in command.

    Expected: an explicit engine choice is honored without probing PATH, so a
    user with both CLIs installed can force codex (or claude) for summaries.
    """
    monkeypatch.setattr(summarize.shutil, "which", lambda _name: pytest.fail("engine choice must not probe PATH"))
    cfg = {"summarizer": {"command": [], "engine": engine, "timeout": 5}}
    assert summarize.resolve_command(cfg) == expected


@pytest.mark.parametrize(
    "installed,expected",
    [
        pytest.param({"claude", "codex"}, summarize.DEFAULT_COMMAND, id="both-prefers-claude"),
        pytest.param({"codex"}, summarize.DEFAULT_COMMAND_CODEX, id="codex-only"),
        pytest.param(set(), summarize.DEFAULT_COMMAND, id="neither-names-claude-in-error"),
    ],
)
def test_resolve_command_auto_detects_installed_cli(monkeypatch, installed, expected):
    """Fall back to codex automatically on codex-only machines.

    Expected: with no engine configured, claude is preferred when installed,
    codex is used when it is the only CLI present, and the claude command is
    kept when neither resolves so the failure message names the primary engine.
    """
    monkeypatch.setattr(summarize.shutil, "which", lambda name: "/bin/%s" % name if name in installed else None)
    cfg = {"summarizer": {"command": [], "engine": "", "timeout": 5}}
    assert summarize.resolve_command(cfg) == expected


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


def test_call_model_redacts_nonzero_stderr(monkeypatch):
    """Redact common credential shapes from summarizer failure text.

    Expected: stderr is still included enough to debug the failing command, but
    bearer tokens and key/value secrets are removed before RuntimeError reaches
    the pipeline logs.
    """
    captured = {}
    monkeypatch.setattr(
        summarize.subprocess,
        "run",
        _fake_run(
            captured,
            returncode=1,
            stderr="Authorization: Bearer secret-token api_key=abc123",
        ),
    )
    cfg = {"summarizer": {"command": ["my-engine"], "timeout": 5}}

    with pytest.raises(RuntimeError) as exc:
        summarize.call_model("hi", cfg)

    text = str(exc.value)
    assert "secret-token" not in text
    assert "abc123" not in text
    assert text.count("[REDACTED]") == 2


def test_call_model_keeps_prompt_on_stdin_and_uses_temp_cwd(monkeypatch):
    """Keep transcript text out of argv/env while running from a neutral cwd."""
    prompt = "session transcript with private content"
    captured = {}
    monkeypatch.setattr(summarize.subprocess, "run", _fake_run(captured))
    cfg = {"summarizer": {"command": ["my-engine", "--fast"], "timeout": 5}}

    summarize.call_model(prompt, cfg)

    assert captured["kwargs"]["input"] == prompt
    assert prompt not in captured["cmd"]
    assert prompt not in captured["kwargs"]["env"].values()
    assert captured["kwargs"]["cwd"] == tempfile.gettempdir()


def test_call_model_resolves_executable(monkeypatch):
    """Resolve configured executables before passing them to subprocess.run."""
    captured = {}
    monkeypatch.setattr(summarize.subprocess, "run", _fake_run(captured))
    monkeypatch.setattr(summarize.shutil, "which", lambda _name: "/resolved/my-engine")
    cfg = {"summarizer": {"command": ["my-engine", "--fast"], "timeout": 5}}

    summarize.call_model("hi", cfg)

    assert captured["cmd"] == ["/resolved/my-engine", "--fast"]


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


# --- Response parsing ------------------------------------------------------------


def test_parse_response_dict():
    """Parse the common JSON object shape returned by the summarizer command.

    Expected: result text, skip status, token counts, and reported cost are all
    copied into the `ModelResult` wrapper.
    """
    raw = json.dumps(
        {
            "result": "## 10:00 | main\ndid stuff",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "total_cost_usd": 0.001,
        }
    )
    r = summarize.parse_response(raw)
    assert r.text.startswith("## 10:00")
    assert not r.is_skip
    assert r.tokens.input == 100
    assert r.tokens.cost_usd == 0.001


def test_parse_response_list_and_skip():
    """Parse the newer list-of-events JSON shape and recognize SKIP.

    Expected: the last result event is used, and a `SKIP` response is marked as
    a skip so the pipeline advances state without writing memory.
    """
    raw = json.dumps(
        [
            {
                "type": "result",
                "result": "SKIP",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        ]
    )
    r = summarize.parse_response(raw)
    assert r.is_skip


def test_parse_response_plain_text():
    """Accept plain text summarizer output.

    Expected: non-JSON stdout is treated as the model text verbatim, which keeps
    alternate summarizer commands usable.
    """
    r = summarize.parse_response("just words")
    assert r.text == "just words"


def test_parse_response_codex_jsonl():
    """Parse the JSONL event stream printed by `codex exec --json`.

    Expected: the text comes from the agent_message item, token counts from
    the turn.completed usage (cache read from codex's `cached_input_tokens`
    key), and intermediate events are ignored. Shape captured from
    codex-cli 0.139.
    """
    raw = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "019f"}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "hmm"}}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_1", "type": "agent_message", "text": "## 10:00 | main\ndid stuff"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 33856, "cached_input_tokens": 2432, "output_tokens": 5},
                }
            ),
        ]
    )
    r = summarize.parse_response(raw)
    assert r.text == "## 10:00 | main\ndid stuff"
    assert not r.is_skip
    assert r.tokens.input == 33856
    assert r.tokens.cache == 2432
    assert r.tokens.output == 5
    assert r.tokens.cost_usd == 0.0


def test_parse_response_codex_jsonl_skip():
    """Recognize SKIP in a codex agent_message so the pipeline advances state."""
    raw = "\n".join(
        [
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "SKIP"}}),
        ]
    )
    assert summarize.parse_response(raw).is_skip


def test_parse_response_multiline_plain_text_is_not_jsonl():
    """Keep multi-line plain text verbatim even though JSONL parsing was attempted.

    Expected: a line that fails to parse as JSON disqualifies the whole output
    from the codex JSONL path, so a plain-text engine's multi-line entry
    survives untouched.
    """
    raw = '{"type": "turn.started"}\n## 10:00 | main\ndid stuff'
    r = summarize.parse_response(raw)
    assert r.text == raw
