import json

import pytest

from ostinote import config as config_mod
from ostinote.agents import get_agent
from ostinote.agents.claude import ClaudeAgent
from ostinote.agents.codex import CodexAgent
from ostinote.pipeline import _last_entry, format_exchanges, parse_consolidation_response
from ostinote.state import PidLock, SessionState
from ostinote.summarize import parse_response

# --- Claude transcript parsing -------------------------------------------------


def _claude_line(msg_type, content, is_meta=False):
    return json.dumps(
        {
            "type": msg_type,
            "isMeta": is_meta,
            "message": {"content": content},
        }
    )


@pytest.fixture
def claude_transcript(tmp_path):
    lines = [
        _claude_line("user", "Fix the login bug"),
        _claude_line("user", "<system-reminder>injected</system-reminder>"),
        _claude_line("summary", "ignored"),
        _claude_line(
            "assistant",
            [
                {"type": "text", "text": "Looking at auth.py now."},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/x/auth.py"}},
                {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -x tests/"}},
            ],
        ),
        _claude_line("user", "ship it", is_meta=True),
        _claude_line("user", [{"type": "tool_result", "content": "big output"}]),
        _claude_line("user", "looks good, thanks"),
    ]
    path = tmp_path / "session-1.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_claude_parse(claude_transcript):
    messages, total = ClaudeAgent().parse(claude_transcript)
    assert total == 7
    roles = [m[0] for m in messages]
    assert roles == ["HUMAN", "AGENT", "HUMAN"]
    agent_text = messages[1][1]
    assert "[TOOL: Read auth.py]" in agent_text
    assert "[TOOL: Bash `pytest -x tests/`]" in agent_text


def test_claude_parse_incremental(claude_transcript):
    _, total = ClaudeAgent().parse(claude_transcript)
    messages, total2 = ClaudeAgent().parse(claude_transcript, skip_lines=total)
    assert messages == []
    assert total2 == total
    messages, _ = ClaudeAgent().parse(claude_transcript, skip_lines=total - 1)
    assert messages == [("HUMAN", "looks good, thanks")]


# --- Codex transcript parsing ---------------------------------------------------


def _codex_item(payload):
    return json.dumps({"type": "response_item", "payload": payload})


@pytest.fixture
def codex_transcript(tmp_path):
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": "abc", "cwd": "/proj"}}),
        _codex_item(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "# AGENTS.md instructions for /proj\nstuff"}
                ],
            }
        ),
        _codex_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Investigate the reboot delays"}],
            }
        ),
        json.dumps({"type": "event_msg", "payload": {"type": "noise"}}),
        _codex_item({"type": "reasoning", "summary": []}),
        _codex_item(
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "journalctl -b -1"}),
            }
        ),
        _codex_item({"type": "function_call_output", "output": "logs..."}),
        _codex_item(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "zram delayed shutdown by 90s."}],
            }
        ),
    ]
    path = tmp_path / "rollout-1.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_codex_parse(codex_transcript):
    messages, total = CodexAgent().parse(codex_transcript)
    assert total == 8
    assert messages == [
        ("HUMAN", "Investigate the reboot delays"),
        ("AGENT", "[TOOL: exec_command `journalctl -b -1`]"),
        ("AGENT", "zram delayed shutdown by 90s."),
    ]


def test_agent_registry():
    assert get_agent("claude").name == "claude"
    assert get_agent("codex").name == "codex"
    with pytest.raises(ValueError):
        get_agent("cursor")


# --- Extract formatting ----------------------------------------------------------


def test_format_exchanges():
    text = format_exchanges("sid", 12, [("HUMAN", "hi"), ("AGENT", "hello")])
    assert text.startswith("Session: sid\nLines: 12")
    assert "[HUMAN]\nhi" in text
    assert "[AGENT]\nhello" in text


def test_last_entry(tmp_path):
    now = tmp_path / "now.md"
    assert _last_entry(str(now)) == "(no previous entry)"
    now.write_text("\n## 10:00 | main\nfirst\n\n## 11:30 | main\nsecond thing\n")
    assert _last_entry(str(now)) == "## 11:30 | main\nsecond thing"


# --- Summarizer response parsing ---------------------------------------------------


def test_parse_response_dict():
    raw = json.dumps(
        {
            "result": "## 10:00 | main\ndid stuff",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "total_cost_usd": 0.001,
        }
    )
    r = parse_response(raw)
    assert r.text.startswith("## 10:00")
    assert not r.is_skip
    assert r.tokens.input == 100
    assert r.tokens.cost_usd == 0.001


def test_parse_response_list_and_skip():
    raw = json.dumps(
        [
            {
                "type": "result",
                "result": "SKIP",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        ]
    )
    r = parse_response(raw)
    assert r.is_skip


def test_parse_response_plain_text():
    r = parse_response("just words")
    assert r.text == "just words"


# --- Consolidation parsing -----------------------------------------------------------


def test_parse_consolidation_full():
    text = "===RECENT===\n# Recent\n\nA\n\n===ARCHIVE===\n# Archive\n\nB"
    recent, archive = parse_consolidation_response(text)
    assert recent == "# Recent\n\nA"
    assert archive == "# Archive\n\nB"


def test_parse_consolidation_fallbacks():
    recent, archive = parse_consolidation_response("bare content")
    assert recent == "# Recent\n\nbare content"
    assert archive == ""


# --- State and locks -------------------------------------------------------------------


def test_session_state_roundtrip(tmp_path):
    sessions = str(tmp_path / "sessions")
    s = SessionState.load(sessions, "codex", "id-1")
    assert s.line == 0
    s.line = 42
    s.transcript_path = "/t.jsonl"
    s.last_save_ts = 123.0
    s.save()
    s2 = SessionState.load(sessions, "codex", "id-1")
    assert (s2.line, s2.transcript_path, s2.last_save_ts) == (42, "/t.jsonl", 123.0)
    # Parallel sessions don't collide.
    other = SessionState.load(sessions, "claude", "id-1")
    assert other.line == 0


def test_pid_lock(tmp_path):
    path = str(tmp_path / "x.lock")
    a, b = PidLock(path), PidLock(path)
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_pid_lock_stale_takeover(tmp_path):
    path = str(tmp_path / "x.lock")
    with open(path, "w") as f:
        f.write("999999999")  # certainly dead
    assert PidLock(path).acquire()


# --- Config ---------------------------------------------------------------------------


def test_config_project_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(
        json.dumps({"cooldowns": {"save_seconds": 60}, "timezone": "UTC"})
    )
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps({"cooldowns": {"save_seconds": 30}})
    )
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["save_seconds"] == 30
    assert cfg["cooldowns"]["compress_seconds"] == 3600  # default survives
    assert cfg["timezone"] == "UTC"  # user layer survives


# --- Installer -------------------------------------------------------------------------


def test_install_uninstall_idempotent(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)

    install_mod.install("codex", "project", root)
    install_mod.install("codex", "project", root)  # idempotent
    hooks_file = tmp_path / ".codex" / "hooks.json"
    data = json.loads(hooks_file.read_text())
    for event in ("SessionStart", "PostToolUse"):
        ours = [h for g in data["hooks"][event] for h in g["hooks"]]
        assert len(ours) == 1
        assert "--agent codex" in ours[0]["command"]
    assert "UserPromptSubmit" not in data["hooks"]

    install_mod.install("codex", "project", root, remove=True)
    data = json.loads(hooks_file.read_text())
    assert data.get("hooks", {}) == {}


def test_install_preserves_foreign_hooks(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "./lint.sh"}],
                        }
                    ]
                }
            }
        )
    )
    install_mod.install("codex", "project", root)
    data = json.loads(hooks_file.read_text())
    commands = [h["command"] for g in data["hooks"]["PostToolUse"] for h in g["hooks"]]
    assert "./lint.sh" in commands
    assert any("--agent codex" in c for c in commands)


def test_parse_consolidation_strips_fences():
    text = "```\n===RECENT===\n# Recent\n\nA\n```\n===ARCHIVE===\n```\n# Archive\n\nB\n```"
    recent, archive = parse_consolidation_response(text)
    assert recent == "# Recent\n\nA"
    assert archive == "# Archive\n\nB"


def test_data_dir_slug_placeholder(tmp_path, monkeypatch):
    from ostinote.env import Env

    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user-config.json"))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    store = tmp_path / "store"
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps(
            {
                "data_dir": str(store / "{slug}"),
                "share_worktrees": False,
            }
        )
    )
    env = Env(str(proj))
    import re

    # claude-remember / Claude Code slug scheme: leading dash kept
    expected_slug = re.sub(r"[^a-zA-Z0-9]", "-", str(proj))
    assert expected_slug.startswith("-")
    assert env.data_dir == str(store / expected_slug)


def test_config_legacy_remember_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(
        json.dumps(
            {
                "cooldowns": {"ndc_seconds": 1800, "git_backup_seconds": 900},
                "features": {"ndc_compression": False},
            }
        )
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["compress_seconds"] == 1800
    assert "ndc_seconds" not in cfg["cooldowns"]
    assert cfg["features"]["hourly_compression"] is False
    assert "ndc_compression" not in cfg["features"]

    # a file that sets both names keeps the new name's value
    (proj / ".ostinote").mkdir()
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps({"cooldowns": {"ndc_seconds": 60, "compress_seconds": 7200}})
    )
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["compress_seconds"] == 7200


def test_install_cleans_legacy_user_prompt_hook(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/usr/bin/ostinote hook"
                                    " user-prompt --agent codex",
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )
    install_mod.install("codex", "project", root)
    data = json.loads(hooks_file.read_text())
    assert "UserPromptSubmit" not in data["hooks"]


def test_summarizer_never_persists_sessions(monkeypatch):
    from ostinote import summarize

    assert "--no-session-persistence" in summarize.DEFAULT_COMMAND

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd

        class R:
            returncode = 0
            stdout = json.dumps({"result": "ok", "usage": {}})
            stderr = ""

        return R()

    monkeypatch.setattr(summarize.subprocess, "run", fake_run)
    # A custom claude command missing the flag gets it appended.
    cfg = {"summarizer": {"command": ["claude", "-p", "--model", "haiku"], "timeout": 5}}
    summarize.call_model("hi", cfg)
    assert "--no-session-persistence" in captured["cmd"]
    # Non-claude engines are left alone.
    cfg = {"summarizer": {"command": ["my-engine", "--fast"], "timeout": 5}}
    summarize.call_model("hi", cfg)
    assert "--no-session-persistence" not in captured["cmd"]
