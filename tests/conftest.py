import json

import pytest

from tests.helpers import codex_item


def claude_line(msg_type, content, is_meta=False):
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
        codex_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "# AGENTS.md instructions for /proj\nstuff"}],
            }
        ),
        codex_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Investigate the reboot delays"}],
            }
        ),
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
        codex_item(
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
