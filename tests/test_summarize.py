import json


def test_summarizer_never_persists_sessions(monkeypatch):
    """Keep nested Claude summarizer calls from creating Claude sessions.

    Expected: the default command includes `--no-session-persistence`, custom
    Claude commands get that flag appended automatically, and non-Claude
    summarizer commands are left alone.
    """
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
