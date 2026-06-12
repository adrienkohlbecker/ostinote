from ostinote.state import SessionState
from tests.helpers import codex_item, model_result, project_env

# --- Pipeline -------------------------------------------------------------------------


def test_run_save_appends_summary_and_advances_state(tmp_path, monkeypatch):
    """Run the save pipeline with a fake Codex transcript and fake model.

    Expected: `run_save` writes the model summary to `now.md`, records the
    transcript path, advances the session line marker, and sends the user
    message into the prompt.
    """
    from ostinote import pipeline as pipeline_mod

    env = project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                codex_item(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Add startup memory"}],
                    }
                ),
                codex_item(
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
    # Replace the expensive external summarizer with a deterministic result,
    # while keeping the prompt available for assertions.
    monkeypatch.setattr(
        pipeline_mod.summarize,
        "call_model",
        lambda prompt, _cfg: prompts_seen.append(prompt) or model_result("## 10:00 | main\nSaved startup memory"),
    )

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert (tmp_path / "data" / "now.md").read_text(encoding="utf-8") == "\n## 10:00 | main\nSaved startup memory\n"
    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert state.line == 2
    assert state.transcript_path == str(transcript)
    assert "Add startup memory" in prompts_seen[0]


def test_run_save_skip_advances_state_without_writing(tmp_path, monkeypatch):
    """Handle a summarizer `SKIP` response as consumed-but-not-written work.

    Expected: no `now.md` file is created, but the session line marker advances
    so the same unimportant transcript line is not reconsidered forever.
    """
    from ostinote import pipeline as pipeline_mod

    env = project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        codex_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Nothing useful"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # A SKIP still consumes transcript lines; the fake model lets us assert that
    # state behavior without writing real memory.
    monkeypatch.setattr(pipeline_mod.summarize, "call_model", lambda _prompt, _cfg: model_result("SKIP"))

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert not (tmp_path / "data" / "now.md").exists()
    assert SessionState.load(env.sessions_dir, "codex", "s1").line == 1


def test_run_save_hourly_compression_moves_now_into_today(tmp_path, monkeypatch):
    """Exercise save followed by hourly compression under the save lock.

    Expected: the first fake model response is appended to `now.md`, the second
    compresses that buffer into today's daily file, and `now.md` is emptied.
    """
    from ostinote import pipeline as pipeline_mod

    env = project_env(tmp_path, monkeypatch, {"features": {"hourly_compression": True}})
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        codex_item(
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
            model_result("## 11:00 | main\nCaptured thing"),
            model_result("## 2026-06-11\nCompressed thing"),
        ]
    )
    # The save path calls the model once for the immediate summary and once for
    # compression, so the iterator order mirrors that control flow.
    monkeypatch.setattr(pipeline_mod.summarize, "call_model", lambda _prompt, _cfg: next(responses))

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert (tmp_path / "data" / "now.md").read_text(encoding="utf-8") == ""
    assert "Compressed thing" in (tmp_path / "data" / ("today-%s.md" % env.today())).read_text(encoding="utf-8")


def test_run_consolidation_writes_sections_and_marks_staging_done(tmp_path, monkeypatch):
    """Consolidate a past daily memory file with a fake model response.

    Expected: recent and archive files are replaced from the parsed sections,
    new core memory text is appended, the prompt includes the staging filename,
    and the processed daily file is renamed to `.done.md`.
    """
    from ostinote import pipeline as pipeline_mod

    env = project_env(tmp_path, monkeypatch)
    env.ensure_dirs()
    staging = tmp_path / "data" / "today-2000-01-01.md"
    staging.write_text("## old\nA", encoding="utf-8")
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nold recent", encoding="utf-8")
    (tmp_path / "data" / "archive.md").write_text("# Archive\n\nold archive", encoding="utf-8")
    seen = {}

    def fake_call_model(prompt, cfg):
        # Capture the prompt and timeout to verify the consolidation-specific
        # wrapper was used, not just the parser after the fact.
        seen["prompt"] = prompt
        seen["timeout"] = cfg["summarizer"]["timeout"]
        return model_result(
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


def test_run_save_rejects_malformed_header(tmp_path, monkeypatch):
    """Refuse to append a summary whose first line is not a journal header.

    Expected: `run_save` returns 1, nothing is written to `now.md`, and the
    session line marker is not persisted, so the same transcript lines are
    retried on the next save instead of being lost.
    """
    from ostinote import pipeline as pipeline_mod

    env = project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        codex_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Do a thing"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline_mod.summarize,
        "call_model",
        lambda _prompt, _cfg: model_result("I could not find a header for this session."),
    )

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 1

    assert not (tmp_path / "data" / "now.md").exists()
    assert SessionState.load(env.sessions_dir, "codex", "s1").line == 0
