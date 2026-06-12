import time

import pytest

from ostinote import pipeline as pipeline_mod
from ostinote.state import SessionState
from tests.helpers import codex_assistant, codex_item, codex_user, model_result, project_env

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


@pytest.mark.parametrize(
    "flags,min_human,expect_saved",
    [
        pytest.param({}, 1, False, id="cooldown-blocks-plain-save"),
        pytest.param({"final": True}, 1, True, id="final-bypasses-cooldown"),
        pytest.param({"final": True}, 2, False, id="final-keeps-min-human-gate"),
        pytest.param({"force": True}, 2, True, id="force-bypasses-both-gates"),
    ],
)
def test_run_save_gating_matrix(tmp_path, monkeypatch, flags, min_human, expect_saved):
    """Exercise the cooldown / min-human / --final / --force gate combinations.

    Expected: an active cooldown skips a plain save; `--final` bypasses the
    cooldown (staleness doesn't apply to a closing session) but still honors
    `min_human_messages` so trivial sessions don't burn a model call; `--force`
    bypasses both. These gates are the tool's cost control — a regression here
    means either a paid model call per tool use or silently dropped saves.
    """
    env = project_env(
        tmp_path,
        monkeypatch,
        {"cooldowns": {"save_seconds": 1000}, "thresholds": {"min_human_messages": min_human}},
    )
    transcript = tmp_path / "session.jsonl"
    # One human message: below min_human=2, enough for min_human=1.
    transcript.write_text(
        codex_user("Gate this save") + "\n" + codex_assistant("Done.") + "\n",
        encoding="utf-8",
    )
    env.ensure_dirs()
    state = SessionState.load(env.sessions_dir, "codex", "s1")
    state.last_attempt_ts = time.time()  # cooldown is running
    state.save()
    calls = []
    monkeypatch.setattr(
        pipeline_mod.summarize,
        "call_model",
        lambda prompt, _cfg: calls.append(prompt) or model_result("## 10:00 | main\nGated entry"),
    )

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript), **flags) == 0

    assert (len(calls) == 1) == expect_saved
    assert (tmp_path / "data" / "now.md").exists() == expect_saved
    assert SessionState.load(env.sessions_dir, "codex", "s1").line == (2 if expect_saved else 0)


def test_run_save_dry_prints_extract_without_state_changes(tmp_path, monkeypatch, capsys):
    """Preview the extract with `--dry` and leave every artifact untouched.

    Expected: `--dry` ignores the running cooldown, prints the formatted
    extract instead of calling the model, and persists nothing — neither
    `now.md` nor the session state (`last_attempt_ts` keeps its exact stored
    value), so a dry run never perturbs real save scheduling.
    """
    env = project_env(tmp_path, monkeypatch, {"cooldowns": {"save_seconds": 1000}})
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(codex_user("Preview me") + "\n", encoding="utf-8")
    env.ensure_dirs()
    state = SessionState.load(env.sessions_dir, "codex", "s1")
    state.last_attempt_ts = time.time() - 5  # cooldown active, value distinguishable
    state.save()
    stored_ts = SessionState.load(env.sessions_dir, "codex", "s1").last_attempt_ts
    monkeypatch.setattr(
        pipeline_mod.summarize,
        "call_model",
        lambda _prompt, _cfg: pytest.fail("dry run must not call the model"),
    )

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript), dry=True) == 0

    out = capsys.readouterr().out
    assert "=== DRY RUN (codex s1) ===" in out
    assert "Preview me" in out
    assert not (tmp_path / "data" / "now.md").exists()
    reloaded = SessionState.load(env.sessions_dir, "codex", "s1")
    assert reloaded.line == 0
    assert reloaded.last_attempt_ts == stored_ts


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


@pytest.mark.parametrize("failure", ["error", "empty"], ids=["model-error", "empty-result"])
def test_run_save_failed_compression_preserves_now(tmp_path, monkeypatch, failure):
    """Keep the session buffer intact when hourly compression fails.

    Expected: the save itself succeeds, and a failing compression call leaves
    `now.md` holding the just-saved entry with no `today-*.md` created — the
    buffer truncation must stay strictly after a successful model response, or
    every failed compression would erase uncompressed memory.
    """
    env = project_env(tmp_path, monkeypatch, {"features": {"hourly_compression": True}})
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(codex_user("Compress this") + "\n", encoding="utf-8")
    responses = iter([model_result("## 11:00 | main\nCaptured thing"), failure])

    def fake_call_model(_prompt, _cfg):
        response = next(responses)
        if response == "error":
            raise RuntimeError("summarizer exploded")
        if response == "empty":
            return model_result("")
        return response

    monkeypatch.setattr(pipeline_mod.summarize, "call_model", fake_call_model)

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert (tmp_path / "data" / "now.md").read_text(encoding="utf-8") == "\n## 11:00 | main\nCaptured thing\n"
    assert not (tmp_path / "data" / ("today-%s.md" % env.today())).exists()


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
            "===RECENT===\n# Recent\n\nnew recent\n===ARCHIVE===\n# Archive\n\nnew archive\n"
            "===CORE===\n- 2026-06-12: stable fact"
        )

    monkeypatch.setattr(pipeline_mod.summarize, "call_model", fake_call_model)

    assert pipeline_mod.run_consolidation(env) == 0

    assert "today-2000-01-01.md" in seen["prompt"]
    assert seen["timeout"] >= 180
    assert (tmp_path / "data" / "recent.md").read_text(encoding="utf-8") == "# Recent\n\nnew recent\n"
    assert (tmp_path / "data" / "archive.md").read_text(encoding="utf-8") == "# Archive\n\nnew archive\n"
    assert "- 2026-06-12: stable fact" in (tmp_path / "data" / "core-memories.md").read_text(encoding="utf-8")
    assert not staging.exists()
    assert (tmp_path / "data" / "today-2000-01-01.done.md").exists()
    # The pre-consolidation contents survive as one-deep .bak copies.
    assert (tmp_path / "data" / "recent.md.bak").read_text(encoding="utf-8") == "# Recent\n\nold recent"
    assert (tmp_path / "data" / "archive.md.bak").read_text(encoding="utf-8") == "# Archive\n\nold archive"


@pytest.mark.parametrize("failure", ["error", "empty"], ids=["model-error", "empty-response"])
def test_run_consolidation_failure_leaves_memory_untouched(tmp_path, monkeypatch, failure):
    """Abort consolidation without consuming staging or rewriting memory.

    Expected: when the model errors out or returns an empty response, the run
    exits 1, the staging file keeps its name (no `.done.md`), and recent.md /
    archive.md are byte-identical — renaming staging before validation would
    silently destroy a day of memory on every bad model reply.
    """
    env = project_env(tmp_path, monkeypatch)
    env.ensure_dirs()
    staging = tmp_path / "data" / "today-2000-01-01.md"
    staging.write_text("## old\nA", encoding="utf-8")
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nold recent", encoding="utf-8")
    (tmp_path / "data" / "archive.md").write_text("# Archive\n\nold archive", encoding="utf-8")

    def fake_call_model(_prompt, _cfg):
        if failure == "error":
            raise RuntimeError("summarizer exploded")
        return model_result("")

    monkeypatch.setattr(pipeline_mod.summarize, "call_model", fake_call_model)

    assert pipeline_mod.run_consolidation(env) == 1

    assert staging.read_text(encoding="utf-8") == "## old\nA"
    assert not (tmp_path / "data" / "today-2000-01-01.done.md").exists()
    assert (tmp_path / "data" / "recent.md").read_text(encoding="utf-8") == "# Recent\n\nold recent"
    assert (tmp_path / "data" / "archive.md").read_text(encoding="utf-8") == "# Archive\n\nold archive"
    assert not (tmp_path / "data" / "recent.md.bak").exists()


def test_run_consolidation_missing_archive_section_keeps_archive(tmp_path, monkeypatch):
    """Tolerate a reply that only rewrites recent memory.

    Expected: a response with a RECENT section but no ARCHIVE section replaces
    recent.md, leaves the existing archive.md unchanged rather than blanking
    it, and still consumes the staging file.
    """
    env = project_env(tmp_path, monkeypatch)
    env.ensure_dirs()
    staging = tmp_path / "data" / "today-2000-01-01.md"
    staging.write_text("## old\nA", encoding="utf-8")
    (tmp_path / "data" / "archive.md").write_text("# Archive\n\nold archive", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_mod.summarize,
        "call_model",
        lambda _prompt, _cfg: model_result("===RECENT===\n# Recent\n\nnew recent"),
    )

    assert pipeline_mod.run_consolidation(env) == 0

    assert (tmp_path / "data" / "recent.md").read_text(encoding="utf-8") == "# Recent\n\nnew recent\n"
    assert (tmp_path / "data" / "archive.md").read_text(encoding="utf-8") == "# Archive\n\nold archive"
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
