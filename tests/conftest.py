import json
import subprocess
import types

import pytest

from ostinote import config as config_mod
from ostinote import doctor as doctor_mod
from ostinote import env as env_mod
from ostinote import install as install_mod
from ostinote import summarize as summarize_mod
from tests.helpers import claude_line, codex_assistant, codex_item, codex_user, installer_home


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Keep every test away from the developer's real home directory.

    Path resolution in the SUT goes through ``~`` (user config, hook error
    log, agent settings, Codex sandbox config), so one forgotten patch in a
    test would silently read — or write — real dotfiles. This redirects HOME
    and the import-time-resolved paths to a per-test temp home as a suite
    invariant; tests that need specific locations override on top.
    """
    home = tmp_path / "isolated-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows uses USERPROFILE, not HOME
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(home / ".ostinote" / "config.json"))
    errors_log = str(home / ".ostinote" / "hook-errors.log")
    monkeypatch.setattr(env_mod, "HOOK_ERRORS_PATH", errors_log)
    # doctor binds the constant by value at import, so patch its copy too.
    monkeypatch.setattr(doctor_mod, "HOOK_ERRORS_PATH", errors_log)


@pytest.fixture(autouse=True)
def _no_real_summarizer(monkeypatch):
    """Fail loudly if a test reaches the real summarizer subprocess.

    In-process tests stub ``summarize.call_model``; a forgotten stub would
    otherwise shell out to the developer's installed `claude` CLI with
    transcript content — network egress and token spend from a unit test.
    Only summarize's view of the subprocess module is replaced, so git calls
    and functional subprocess tests are unaffected; tests that exercise
    ``call_model`` itself patch ``summarize.subprocess.run`` on this stub.
    """

    def refuse(*_args, **_kwargs):
        raise AssertionError("unmocked summarizer call — stub summarize.call_model or summarize.subprocess.run")

    monkeypatch.setattr(
        summarize_mod,
        "subprocess",
        types.SimpleNamespace(run=refuse, TimeoutExpired=subprocess.TimeoutExpired),
    )


@pytest.fixture
def installer_env(tmp_path, monkeypatch):
    """Temp home for installer tests, with a stable hook command.

    Redirects HOME/USERPROFILE/USER_CONFIG_PATH via ``installer_home`` and
    pins ``install.self_command`` so generated hook JSON is deterministic.
    Returns the temp home path.
    """
    home = installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    return home


@pytest.fixture
def claude_transcript(tmp_path):
    lines = [
        claude_line("user", "Fix the login bug"),
        claude_line("user", "<system-reminder>injected</system-reminder>"),
        claude_line("summary", "ignored"),
        claude_line(
            "assistant",
            [
                {"type": "text", "text": "Looking at auth.py now."},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/x/auth.py"}},
                {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -x tests/"}},
            ],
        ),
        claude_line("user", "ship it", is_meta=True),
        claude_line("user", [{"type": "tool_result", "content": "big output"}]),
        claude_line("user", "looks good, thanks"),
    ]
    path = tmp_path / "session-1.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def codex_transcript(tmp_path):
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": "abc", "cwd": "/proj"}}),
        codex_user("# AGENTS.md instructions for /proj\nstuff"),
        codex_user("Investigate the reboot delays"),
        json.dumps({"type": "event_msg", "payload": {"type": "noise"}}),
        codex_item({"type": "reasoning", "summary": []}),
        codex_item(
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "journalctl -b -1"}),
            }
        ),
        codex_item({"type": "function_call_output", "output": "logs..."}),
        codex_assistant("zram delayed shutdown by 90s."),
    ]
    path = tmp_path / "rollout-1.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
