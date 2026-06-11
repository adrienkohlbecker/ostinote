import json
import os
import time
import tomllib

import pytest

from ostinote import config as config_mod
from ostinote.agents import get_agent
from ostinote.agents.claude import ClaudeAgent
from ostinote.agents.codex import CodexAgent
from ostinote.env import Env
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
                "content": [{"type": "input_text", "text": "# AGENTS.md instructions for /proj\nstuff"}],
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


def test_codex_parse_tool_edge_cases(tmp_path):
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                _codex_item({"type": "message", "role": "developer", "content": [{"type": "text", "text": "ignore"}]}),
                _codex_item(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "<hook>ignore injected</hook>"}],
                    }
                ),
                _codex_item(
                    {
                        "type": "function_call",
                        "name": "apply_patch",
                        "arguments": json.dumps({"path": ["a.py", "b.py"]}),
                    }
                ),
                _codex_item({"type": "function_call", "name": "mystery", "arguments": "{"}),
                _codex_item({"type": "custom_tool_call", "name": "shell", "input": "x" * 300}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    messages, total = CodexAgent().parse(str(path))

    assert total == 5
    assert messages[0] == ("AGENT", "[TOOL: apply_patch `a.py b.py`]")
    assert messages[1] == ("AGENT", "[TOOL: mystery]")
    assert messages[2][0] == "AGENT"
    assert messages[2][1].startswith("[TOOL: shell ")
    assert len(messages[2][1]) < 140


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
    recent, archive, core = parse_consolidation_response(text)
    assert recent == "# Recent\n\nA"
    assert archive == "# Archive\n\nB"
    assert core == ""


def test_parse_consolidation_core_section():
    text = (
        "===RECENT===\n# Recent\n\nA\n===ARCHIVE===\n# Archive\n\nB\n"
        "===CORE===\n- 2026-06-10: chose MIT"
    )
    recent, archive, core = parse_consolidation_response(text)
    assert recent == "# Recent\n\nA"
    assert archive == "# Archive\n\nB"
    assert core == "- 2026-06-10: chose MIT"


def test_parse_consolidation_fallbacks():
    recent, archive, core = parse_consolidation_response("bare content")
    assert recent == "# Recent\n\nbare content"
    assert archive == ""
    assert core == ""


def test_append_core(tmp_path):
    from ostinote.pipeline import _append_core

    path = str(tmp_path / "core-memories.md")
    _append_core(path, "- 2026-06-10: a")
    _append_core(path, "- 2026-06-11: b")
    with open(path) as f:
        assert f.read() == "# Core Memories\n\n- 2026-06-10: a\n- 2026-06-11: b\n"


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


def test_recovery_uses_saved_line_not_failed_attempt_time(tmp_path, monkeypatch):
    from ostinote import hooks

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n{}\n")
    idle_mtime = time.time() - hooks._RECOVERY_ACTIVE_WINDOW - 1
    os.utime(transcript, (idle_mtime, idle_mtime))

    sessions = tmp_path / "sessions"
    state = SessionState.load(str(sessions), "codex", "session")
    state.transcript_path = str(transcript)
    state.line = 1
    state.last_attempt_ts = time.time()
    state.save()

    queued = []
    env = type("EnvStub", (), {"sessions_dir": str(sessions), "cwd": str(tmp_path)})()
    monkeypatch.setattr(hooks, "spawn", lambda _env, args: queued.append(args))

    assert hooks._recover_missed(env) == 1
    assert queued
    assert "--force" in queued[0]


def test_recovery_skips_fully_saved_transcripts(tmp_path, monkeypatch):
    from ostinote import hooks

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n")
    idle_mtime = time.time() - hooks._RECOVERY_ACTIVE_WINDOW - 1
    os.utime(transcript, (idle_mtime, idle_mtime))

    sessions = tmp_path / "sessions"
    state = SessionState.load(str(sessions), "codex", "session")
    state.transcript_path = str(transcript)
    state.line = 2
    state.save()

    queued = []
    env = type("EnvStub", (), {"sessions_dir": str(sessions), "cwd": str(tmp_path)})()
    monkeypatch.setattr(hooks, "spawn", lambda _env, args: queued.append(args))

    assert hooks._recover_missed(env) == 0
    assert queued == []


# --- Config ---------------------------------------------------------------------------


def test_config_project_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(json.dumps({"cooldowns": {"save_seconds": 60}, "timezone": "UTC"}))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(json.dumps({"cooldowns": {"save_seconds": 30}}))
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["save_seconds"] == 30
    assert cfg["cooldowns"]["compress_seconds"] == 3600  # default survives
    assert cfg["timezone"] == "UTC"  # user layer survives


def _project_env(tmp_path, monkeypatch, extra_cfg=None):
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user-config.json"))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    cfg = {
        "data_dir": str(tmp_path / "data"),
        "share_worktrees": False,
        "cooldowns": {"save_seconds": 0, "compress_seconds": 0},
        "thresholds": {"min_human_messages": 1, "delta_lines_trigger": 1},
        "features": {"hourly_compression": False, "consolidation": True, "recovery": True},
    }
    if extra_cfg:
        for key, value in extra_cfg.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    (proj / ".ostinote" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return Env(str(proj))


def _model_result(text, input_tokens=10, output_tokens=3, cache_tokens=0, cost=0.0):
    from ostinote.summarize import ModelResult, TokenUsage

    return ModelResult(
        text=text,
        tokens=TokenUsage(input=input_tokens, output=output_tokens, cache=cache_tokens, cost_usd=cost),
        is_skip=text.strip().upper().startswith("SKIP"),
    )


# --- Pipeline -------------------------------------------------------------------------


def test_run_save_appends_summary_and_advances_state(tmp_path, monkeypatch):
    from ostinote import pipeline as pipeline_mod

    env = _project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _codex_item(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Add startup memory"}],
                    }
                ),
                _codex_item(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Implemented it."}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prompts_seen = []
    monkeypatch.setattr(
        pipeline_mod.summarize,
        "call_model",
        lambda prompt, _cfg: prompts_seen.append(prompt) or _model_result("## 10:00 | main\nSaved startup memory"),
    )

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert (tmp_path / "data" / "now.md").read_text(encoding="utf-8") == "\n## 10:00 | main\nSaved startup memory\n"
    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert state.line == 2
    assert state.transcript_path == str(transcript)
    assert "Add startup memory" in prompts_seen[0]


def test_run_save_skip_advances_state_without_writing(tmp_path, monkeypatch):
    from ostinote import pipeline as pipeline_mod

    env = _project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        _codex_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Nothing useful"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_mod.summarize, "call_model", lambda _prompt, _cfg: _model_result("SKIP"))

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert not (tmp_path / "data" / "now.md").exists()
    assert SessionState.load(env.sessions_dir, "codex", "s1").line == 1


def test_run_save_hourly_compression_moves_now_into_today(tmp_path, monkeypatch):
    from ostinote import pipeline as pipeline_mod

    env = _project_env(tmp_path, monkeypatch, {"features": {"hourly_compression": True}})
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        _codex_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Compress this"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    responses = iter(
        [
            _model_result("## 11:00 | main\nCaptured thing"),
            _model_result("## 2026-06-11\nCompressed thing"),
        ]
    )
    monkeypatch.setattr(pipeline_mod.summarize, "call_model", lambda _prompt, _cfg: next(responses))

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert (tmp_path / "data" / "now.md").read_text(encoding="utf-8") == ""
    assert "Compressed thing" in (tmp_path / "data" / ("today-%s.md" % env.today())).read_text(encoding="utf-8")


def test_run_consolidation_writes_sections_and_marks_staging_done(tmp_path, monkeypatch):
    from ostinote import pipeline as pipeline_mod

    env = _project_env(tmp_path, monkeypatch)
    env.ensure_dirs()
    staging = tmp_path / "data" / "today-2000-01-01.md"
    staging.write_text("## old\nA", encoding="utf-8")
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nold recent", encoding="utf-8")
    (tmp_path / "data" / "archive.md").write_text("# Archive\n\nold archive", encoding="utf-8")
    seen = {}

    def fake_call_model(prompt, cfg):
        seen["prompt"] = prompt
        seen["timeout"] = cfg["summarizer"]["timeout"]
        return _model_result(
            "===RECENT===\n# Recent\n\nnew recent\n===ARCHIVE===\n# Archive\n\nnew archive\n===CORE===\n- stable fact"
        )

    monkeypatch.setattr(pipeline_mod.summarize, "call_model", fake_call_model)

    assert pipeline_mod.run_consolidation(env) == 0

    assert "today-2000-01-01.md" in seen["prompt"]
    assert seen["timeout"] >= 180
    assert (tmp_path / "data" / "recent.md").read_text(encoding="utf-8") == "# Recent\n\nnew recent\n"
    assert (tmp_path / "data" / "archive.md").read_text(encoding="utf-8") == "# Archive\n\nnew archive\n"
    assert "- stable fact" in (tmp_path / "data" / "core-memories.md").read_text(encoding="utf-8")
    assert not staging.exists()
    assert (tmp_path / "data" / "today-2000-01-01.done.md").exists()


# --- Installer -------------------------------------------------------------------------


def _installer_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows uses USERPROFILE, not HOME
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(home / ".ostinote/config.json"))
    return home


def test_install_uninstall_idempotent(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    _installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)

    install_mod.install("codex", "project", root)
    install_mod.install("codex", "project", root)  # idempotent
    hooks_file = tmp_path / ".codex" / "hooks.json"
    data = json.loads(hooks_file.read_text())
    for event in ("SessionStart", "PostToolUse", "Stop"):
        ours = [h for g in data["hooks"][event] for h in g["hooks"]]
        assert len(ours) == 1
        assert "--agent codex" in ours[0]["command"]
    assert set(data["hooks"]) == {"SessionStart", "PostToolUse", "Stop"}

    install_mod.install("codex", "project", root, remove=True)
    data = json.loads(hooks_file.read_text())
    assert data.get("hooks", {}) == {}


def test_skill_installed_per_agent_and_scope(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    home = _installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])

    root = str(tmp_path / "proj")
    report = install_mod.install("codex", "project", root)
    skill = tmp_path / "proj" / ".agents" / "skills" / "ostinote" / "SKILL.md"
    assert "codex $ostinote skill installed" in "\n".join(report)
    assert skill.read_text().startswith("---")

    install_mod.install("codex", "project", root, remove=True)
    assert not skill.exists()

    install_mod.install("codex", "user", root)
    assert (home / ".agents" / "skills" / "ostinote" / "SKILL.md").exists()
    install_mod.install("claude", "user", root)
    assert (home / ".claude" / "skills" / "ostinote" / "SKILL.md").exists()


def test_install_session_end_events_per_agent(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    _installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)
    install_mod.install("claude", "project", root)
    claude = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert set(claude["hooks"]) == {"SessionStart", "PostToolUse", "SessionEnd"}

    # Codex has no SessionEnd; its turn-scoped Stop maps to the same handler.
    install_mod.install("codex", "project", root)
    codex = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    assert set(codex["hooks"]) == {"SessionStart", "PostToolUse", "Stop"}
    stop = [h["command"] for g in codex["hooks"]["Stop"] for h in g["hooks"]]
    assert "hook session-end --agent codex" in stop[0]


@pytest.mark.parametrize(
    "source,injected",
    [("startup", True), ("clear", True), ("", True), ("resume", False), ("compact", False)],
)
def test_session_start_source_filter(tmp_path, monkeypatch, capsys, source, injected):
    import io

    from ostinote import hooks as hooks_mod

    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps(
            {
                "data_dir": str(tmp_path / "data"),
                "share_worktrees": False,
                "features": {"recovery": False, "consolidation": False},
            }
        )
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nsomething happened\n")

    payload = {"cwd": str(proj)}
    if source:
        payload["source"] = source
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    hooks_mod.session_start("claude")
    out = capsys.readouterr().out
    if injected:
        assert "=== MEMORY ===" in out
        assert "something happened" in out
    else:
        assert out == ""


def test_post_tool_registers_session_and_queues_save(tmp_path, monkeypatch):
    import io

    from ostinote import hooks as hooks_mod

    env = _project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n{}\n", encoding="utf-8")
    queued = []
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"cwd": env.cwd, "transcript_path": str(transcript), "session_id": "s1"})),
    )

    hooks_mod.post_tool("codex")

    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert state.transcript_path == str(transcript)
    assert queued == [
        [
            "save",
            "--agent",
            "codex",
            "--session",
            "s1",
            "--transcript",
            str(transcript),
            "--cwd",
            env.cwd,
        ]
    ]


def test_session_end_queues_final_save_from_transcript_basename(tmp_path, monkeypatch):
    import io

    from ostinote import hooks as hooks_mod

    env = _project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session-abc.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    queued = []
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"cwd": env.cwd, "transcript_path": str(transcript)})))

    hooks_mod.session_end("claude")

    assert queued == [
        [
            "save",
            "--agent",
            "claude",
            "--session",
            "session-abc",
            "--transcript",
            str(transcript),
            "--cwd",
            env.cwd,
            "--final",
        ]
    ]


def test_session_start_queues_consolidation_without_injecting_on_resume(tmp_path, monkeypatch, capsys):
    import io

    from ostinote import hooks as hooks_mod

    env = _project_env(tmp_path, monkeypatch, {"features": {"recovery": False}})
    env.ensure_dirs()
    (tmp_path / "data" / "today-2000-01-01.md").write_text("old", encoding="utf-8")
    queued = []
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"cwd": env.cwd, "source": "resume"})))

    hooks_mod.session_start("codex")

    assert queued == [["consolidate", "--cwd", env.cwd]]
    assert capsys.readouterr().out == ""


# --- CLI ------------------------------------------------------------------------------


def test_cli_dispatches_save_and_consolidate(tmp_path, monkeypatch):
    from ostinote import cli as cli_mod

    calls = []
    monkeypatch.setattr(
        cli_mod.pipeline,
        "run_save",
        lambda env, agent, session, transcript, force, dry, final: (
            calls.append(("save", env.cwd, agent, session, transcript, force, dry, final)) or 7
        ),
    )
    with pytest.raises(SystemExit) as save_exit:
        cli_mod.main(
            [
                "save",
                "--agent",
                "codex",
                "--session",
                "s1",
                "--transcript",
                "t.jsonl",
                "--cwd",
                str(tmp_path),
                "--force",
                "--dry",
            ]
        )
    assert save_exit.value.code == 7
    assert calls == [("save", str(tmp_path), "codex", "s1", "t.jsonl", True, True, False)]

    monkeypatch.setattr(
        cli_mod.pipeline,
        "run_consolidation",
        lambda env: calls.append(("consolidate", env.cwd)) or 3,
    )
    with pytest.raises(SystemExit) as consolidate_exit:
        cli_mod.main(["consolidate", "--cwd", str(tmp_path)])
    assert consolidate_exit.value.code == 3
    assert calls[-1] == ("consolidate", str(tmp_path))


def test_cli_hook_failures_are_logged_and_swallowed(tmp_path, monkeypatch):
    from argparse import Namespace

    from ostinote import cli as cli_mod

    errors = tmp_path / "hook-errors.log"
    monkeypatch.setattr(cli_mod.env_mod, "HOOK_ERRORS_PATH", str(errors))
    monkeypatch.setattr(cli_mod.hooks_mod, "post_tool", lambda _agent: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(SystemExit) as exc:
        cli_mod._run_hook(Namespace(event="post-tool", agent="codex"))

    assert exc.value.code == 0
    assert "RuntimeError: boom" in errors.read_text(encoding="utf-8")


def test_install_preserves_foreign_hooks(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    _installer_home(tmp_path, monkeypatch)
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


def test_codex_install_adds_memory_dir_to_writable_roots(tmp_path, monkeypatch):
    import re

    from ostinote import install as install_mod

    home = _installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        'model = "gpt-5"\n\n[sandbox_workspace_write]\nwritable_roots = ["~/already"]\nnetwork_access = true\n',
        encoding="utf-8",
    )

    install_mod.install("codex", "project", root)
    install_mod.install("codex", "project", root)
    install_mod.install("codex", "project", root, remove=True)

    expected = "~/.ostinote/projects/%s" % re.sub(r"[^a-zA-Z0-9]", "-", root)
    text = config.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    sandbox = data["sandbox_workspace_write"]
    assert sandbox["writable_roots"].count(expected) == 1
    assert sandbox["network_access"] is True
    assert "~/already" in sandbox["writable_roots"]


def test_codex_install_refuses_invalid_toml(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    home = _installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("[sandbox_workspace_write\n", encoding="utf-8")

    report = install_mod.install("codex", "project", root)

    assert any(line.startswith("ERROR: could not update Codex writable roots") for line in report)
    assert config.read_text(encoding="utf-8") == "[sandbox_workspace_write\n"


def test_install_refuses_invalid_hook_json(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    _installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text("{", encoding="utf-8")

    report = install_mod.install("codex", "project", root)

    assert report[0].startswith("ERROR: invalid JSON")
    assert hooks_file.read_text(encoding="utf-8") == "{"
    assert not (tmp_path / ".agents" / "skills" / "ostinote" / "SKILL.md").exists()


def test_install_preserves_similar_and_nonconforming_hooks(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    _installer_home(tmp_path, monkeypatch)
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
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo ostinote --agent codex",
                                },
                                {
                                    "type": "command",
                                    "command": "/old/ostinote hook post-tool --agent codex",
                                },
                                {"type": "command", "command": "./écho.sh"},
                            ],
                        },
                        {"matcher": "Opaque", "hooks": "leave-me"},
                        "strange-group",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    install_mod.install("codex", "project", root)

    text = hooks_file.read_text(encoding="utf-8")
    data = json.loads(text)
    groups = data["hooks"]["PostToolUse"]
    commands = [
        hook["command"]
        for group in groups
        if isinstance(group, dict) and isinstance(group.get("hooks"), list)
        for hook in group["hooks"]
    ]
    assert "echo ostinote --agent codex" in commands
    assert "/old/ostinote hook post-tool --agent codex" not in commands
    assert "./écho.sh" in commands
    assert "\\u00e9" not in text
    assert {"matcher": "Opaque", "hooks": "leave-me"} in groups
    assert "strange-group" in groups


def test_uninstall_clean_project_does_not_create_hook_files(tmp_path, monkeypatch):
    from ostinote import install as install_mod

    _installer_home(tmp_path, monkeypatch)
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()

    assert install_mod.install("codex", "project", root, remove=True) == []
    assert not (tmp_path / "proj" / ".codex").exists()
    assert not (tmp_path / "proj" / ".agents").exists()


def test_parse_consolidation_strips_fences():
    text = "```\n===RECENT===\n# Recent\n\nA\n```\n===ARCHIVE===\n```\n# Archive\n\nB\n```"
    recent, archive, core = parse_consolidation_response(text)
    assert recent == "# Recent\n\nA"
    assert archive == "# Archive\n\nB"
    assert core == ""


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

    # claude-remember / Claude Code slug scheme: leading dash kept.
    # Windows paths start with drive letters instead.
    expected_slug = re.sub(r"[^a-zA-Z0-9]", "-", str(proj))
    if os.name != "nt":
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


def test_costs_day_totals(tmp_path):
    from ostinote import costs

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "memory-2026-06-09.log").write_text(
        "12:00:00 [save] tokens: 100+50cache→20out ($0.000123)\n"
        "12:30:00 [compress] tokens: 200+0cache→40out\n"
        "12:31:00 [hook] not a token line\n",
        encoding="utf-8",
    )
    (logs / "memory-2026-06-10.log").write_text("09:00:00 [hook] no calls today\n", encoding="utf-8")
    (logs / "background.log").write_text("[save] tokens: 9+9cache→9out ($9)\n", encoding="utf-8")

    days = costs.day_totals(str(logs))
    assert [d for d, _ in days] == ["2026-06-09"]  # only daily logs with calls
    totals = days[0][1]
    assert totals["calls"] == 2
    assert totals["input"] == 300
    assert totals["cache"] == 50
    assert totals["output"] == 60
    assert totals["cost"] == pytest.approx(0.000123)  # unreported cost not invented


def test_doctor_smoke(tmp_path, monkeypatch, capsys):
    from ostinote import doctor as doctor_mod
    from ostinote import install as install_mod
    from ostinote.env import Env

    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    monkeypatch.setattr(doctor_mod, "HOOK_ERRORS_PATH", str(tmp_path / "hook-errors.log"))
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])

    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps({"data_dir": str(tmp_path / "data"), "share_worktrees": False})
    )
    install_mod.install("claude", "project", str(proj))
    install_mod.install("codex", "project", str(proj))
    # Keep the user-scope lookup away from the real home directory.
    project_only = install_mod._hooks_file_for
    monkeypatch.setattr(
        doctor_mod,
        "_hooks_file_for",
        lambda agent, scope, root: (
            project_only(agent, scope, root) if scope == "project" else str(tmp_path / "no-user-hooks.json")
        ),
    )

    assert doctor_mod.run(Env(str(proj))) == 0
    out = capsys.readouterr().out
    assert "claude: hooks registered" in out
    assert "codex: hooks registered" in out
    assert "FAIL" not in out

    # A half-registered agent is a FAIL with a fix hint.
    claude_settings = proj / ".claude" / "settings.json"
    data = json.loads(claude_settings.read_text())
    del data["hooks"]["SessionEnd"]
    claude_settings.write_text(json.dumps(data))
    assert doctor_mod.run(Env(str(proj))) == 1
    assert "hooks missing SessionEnd" in capsys.readouterr().out


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
