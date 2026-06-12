import json
import os
import re
import time

import pytest

from ostinote.agents import get_agent
from ostinote.agents.claude import ClaudeAgent
from ostinote.agents.codex import CodexAgent
from tests.helpers import claude_line, codex_item


def test_claude_parse(claude_transcript):
    """Parse a Claude transcript into only the meaningful conversation.

    Expected: injected reminders, summaries, meta messages, and tool output are
    ignored; human/agent turns remain, and agent tool calls become readable
    `[TOOL: ...]` summaries.
    """
    messages, total = ClaudeAgent().parse(claude_transcript)
    assert total == 7
    roles = [m[0] for m in messages]
    assert roles == ["HUMAN", "AGENT", "HUMAN"]
    agent_text = messages[1][1]
    assert "[TOOL: Read auth.py]" in agent_text
    assert "[TOOL: Bash `pytest -x tests/`]" in agent_text


def test_parse_skips_malformed_lines(tmp_path, capsys):
    """Survive a transcript line truncated mid-write.

    Expected: the malformed JSONL line is skipped but still counted in the
    resume offset, surrounding messages parse normally, and the skip is
    reported on stderr so spawned saves leave a trace in background.log.
    """
    path = tmp_path / "session.jsonl"
    path.write_text(
        claude_line("user", "hello") + "\n" + '{"type": "user", "mess\n' + claude_line("user", "world") + "\n",
        encoding="utf-8",
    )

    messages, total = ClaudeAgent().parse(str(path))

    assert total == 3
    assert messages == [("HUMAN", "hello"), ("HUMAN", "world")]
    assert "skipped 1 malformed transcript line(s)" in capsys.readouterr().err


def test_parse_skips_non_dict_json_lines(tmp_path, capsys):
    """Treat valid-JSON lines that are not objects as malformed.

    Expected: a bare string or list line is counted as malformed instead of
    reaching the extractor (which indexes into a dict), and surrounding
    messages still parse.
    """
    path = tmp_path / "session.jsonl"
    path.write_text(
        claude_line("user", "hello") + "\n" + '"just a string"\n' + "[1, 2]\n",
        encoding="utf-8",
    )

    messages, total = ClaudeAgent().parse(str(path))

    assert total == 3
    assert messages == [("HUMAN", "hello")]
    assert "skipped 2 malformed transcript line(s)" in capsys.readouterr().err


def test_claude_parse_incremental(claude_transcript):
    """Resume Claude parsing from a saved line offset.

    Expected: parsing from the end returns no new messages, while backing up one
    line returns only the final human message. This protects incremental saves
    from re-summarizing old transcript content.
    """
    _, total = ClaudeAgent().parse(claude_transcript)
    messages, total2 = ClaudeAgent().parse(claude_transcript, skip_lines=total)
    assert messages == []
    assert total2 == total
    messages, _ = ClaudeAgent().parse(claude_transcript, skip_lines=total - 1)
    assert messages == [("HUMAN", "looks good, thanks")]


# --- Codex transcript parsing ---------------------------------------------------


def test_codex_parse(codex_transcript):
    """Parse a Codex rollout into user text, tool calls, and assistant text.

    Expected: injected AGENTS/context lines and non-conversation events are
    skipped; the real user request, shell command summary, and assistant answer
    are preserved in order.
    """
    messages, total = CodexAgent().parse(codex_transcript)
    assert total == 8
    assert messages == [
        ("HUMAN", "Investigate the reboot delays"),
        ("AGENT", "[TOOL: exec_command `journalctl -b -1`]"),
        ("AGENT", "zram delayed shutdown by 90s."),
    ]


def test_codex_parse_tool_edge_cases(tmp_path):
    """Handle Codex parser edge cases without leaking noisy context.

    Expected: developer and hook-injected messages are ignored, list-style tool
    details are joined into readable text, invalid JSON arguments fall back to a
    generic tool marker, and long custom tool input is truncated.
    """
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                codex_item({"type": "message", "role": "developer", "content": [{"type": "text", "text": "ignore"}]}),
                codex_item(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "<hook>ignore injected</hook>"}],
                    }
                ),
                codex_item(
                    {
                        "type": "function_call",
                        "name": "apply_patch",
                        "arguments": json.dumps({"path": ["a.py", "b.py"]}),
                    }
                ),
                codex_item({"type": "function_call", "name": "mystery", "arguments": "{"}),
                codex_item({"type": "custom_tool_call", "name": "shell", "input": "x" * 300}),
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
    """Look up supported agent adapters by name.

    Expected: `claude` and `codex` return working adapters, while an unknown
    agent name raises `ValueError` instead of silently choosing the wrong parser.
    """
    assert get_agent("claude").name == "claude"
    assert get_agent("codex").name == "codex"
    with pytest.raises(ValueError, match="unknown agent 'cursor'"):
        get_agent("cursor")


# --- Transcript discovery --------------------------------------------------------


def test_claude_find_latest_transcript(tmp_path, monkeypatch):
    """Pick the newest session file from the project's Claude slug directory.

    Expected: among several transcripts in `~/.claude/projects/<slug>/`, the
    one with the most recent mtime wins — this is what `ostinote save` runs on
    when invoked by hand — and a project with no session directory returns
    None instead of raising.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    cwd = str(tmp_path / "proj")
    # Claude Code's documented slug scheme (drive letter lowercased on Windows).
    if os.name == "nt" and re.match(r"^[A-Za-z]:", cwd):
        cwd_for_slug = cwd[0].lower() + cwd[1:]
    else:
        cwd_for_slug = cwd
    sdir = home / ".claude" / "projects" / re.sub(r"[^a-zA-Z0-9]", "-", cwd_for_slug)
    sdir.mkdir(parents=True)
    for name, age in (("older.jsonl", 300), ("newest.jsonl", 100)):
        path = sdir / name
        path.write_text("{}\n", encoding="utf-8")
        os.utime(path, (time.time() - age, time.time() - age))

    assert ClaudeAgent().find_latest_transcript(cwd) == str(sdir / "newest.jsonl")
    assert ClaudeAgent().find_latest_transcript(str(tmp_path / "no-sessions")) is None


def test_codex_find_latest_transcript(tmp_path, monkeypatch):
    """Match the newest rollout whose session_meta cwd is this project.

    Expected: a newer rollout belonging to a different project is skipped, an
    unreadable first line is tolerated, and the newest matching rollout wins —
    matching on the recorded cwd is what keeps another project's session from
    being summarized into this project's memory. No match returns None.
    """
    from ostinote.agents import codex as codex_mod

    root = tmp_path / "sessions"
    monkeypatch.setattr(codex_mod, "SESSIONS_ROOT", str(root))
    day_dir = root / time.strftime("%Y/%m/%d", time.localtime())
    day_dir.mkdir(parents=True)
    cwd = str(tmp_path / "proj")

    def rollout(name, first_line, age):
        path = day_dir / name
        path.write_text(first_line + "\n", encoding="utf-8")
        os.utime(path, (time.time() - age, time.time() - age))
        return str(path)

    def meta(meta_cwd):
        return json.dumps({"type": "session_meta", "payload": {"id": "x", "cwd": meta_cwd}})

    rollout("rollout-1.jsonl", meta(cwd), age=300)
    matching_new = rollout("rollout-2.jsonl", meta(cwd), age=200)
    rollout("rollout-3.jsonl", meta(str(tmp_path / "other-project")), age=100)
    rollout("rollout-4.jsonl", "{ not json", age=50)

    assert CodexAgent().find_latest_transcript(cwd) == matching_new
    assert CodexAgent().find_latest_transcript(str(tmp_path / "nowhere")) is None
