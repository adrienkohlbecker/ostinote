import json

import pytest

from ostinote.agents import get_agent
from ostinote.agents.claude import ClaudeAgent
from ostinote.agents.codex import CodexAgent
from tests.helpers import codex_item


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
    from tests.conftest import claude_line

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
    from tests.conftest import claude_line

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
    with pytest.raises(ValueError):
        get_agent("cursor")
