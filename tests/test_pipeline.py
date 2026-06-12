import os
import time

import pytest

from ostinote import pipeline as pipeline_mod
from ostinote.pipeline import _append_core, _last_entry, format_exchanges, parse_consolidation_response
from ostinote.state import SessionState
from tests.helpers import age_file, codex_assistant, codex_user, model_result


def _seed_consolidation(tmp_path, env):
    """Seed a past-day staging file plus existing recent/archive memory.

    The consolidation tests differ only in the model response and its
    aftermath, so the arrange step is shared. Returns the staging path.
    """
    env.ensure_dirs()
    data = tmp_path / "data"
    staging = data / "today-2000-01-01.md"
    staging.write_text("## old\nA", encoding="utf-8")
    (data / "recent.md").write_text("# Recent\n\nold recent", encoding="utf-8")
    (data / "archive.md").write_text("# Archive\n\nold archive", encoding="utf-8")
    return staging


# --- Pipeline -------------------------------------------------------------------------


def test_run_save_appends_summary_and_advances_state(tmp_path, monkeypatch, make_project_env):
    """Run the save pipeline with a fake Codex transcript and fake model.

    Expected: `run_save` writes the model summary to `now.md`, records the
    transcript path, advances the session line marker, and sends the user
    message into the prompt.
    """
    env = make_project_env()
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        codex_user("Add startup memory") + "\n" + codex_assistant("Implemented it.") + "\n",
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


def test_run_save_skip_advances_state_without_writing(tmp_path, monkeypatch, make_project_env):
    """Handle a summarizer `SKIP` response as consumed-but-not-written work.

    Expected: no `now.md` file is created, but the session line marker advances
    so the same unimportant transcript line is not reconsidered forever.
    """
    env = make_project_env()
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(codex_user("Nothing useful") + "\n", encoding="utf-8")
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
def test_run_save_gating_matrix(tmp_path, monkeypatch, make_project_env, flags, min_human, expect_saved):
    """Exercise the cooldown / min-human / --final / --force gate combinations.

    Expected: an active cooldown skips a plain save; `--final` bypasses the
    cooldown (staleness doesn't apply to a closing session) but still honors
    `min_human_messages` so trivial sessions don't burn a model call; `--force`
    bypasses both. These gates are the tool's cost control — a regression here
    means either a paid model call per tool use or silently dropped saves.
    """
    env = make_project_env(
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


def test_run_save_dry_prints_extract_without_state_changes(tmp_path, monkeypatch, make_project_env, capsys):
    """Preview the extract with `--dry` and leave every artifact untouched.

    Expected: `--dry` ignores the running cooldown, prints the formatted
    extract instead of calling the model, and persists nothing — neither
    `now.md` nor the session state (`last_attempt_ts` keeps its exact stored
    value), so a dry run never perturbs real save scheduling.
    """
    env = make_project_env({"cooldowns": {"save_seconds": 1000}})
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


def test_run_save_hourly_compression_moves_now_into_today(tmp_path, monkeypatch, make_project_env):
    """Exercise save followed by hourly compression under the save lock.

    Expected: the first fake model response is appended to `now.md`, the second
    compresses that buffer into today's daily file, and `now.md` is emptied.
    """
    env = make_project_env({"features": {"hourly_compression": True}})
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(codex_user("Compress this") + "\n", encoding="utf-8")
    responses = iter(
        [
            model_result("## 11:00 | main\nCaptured thing"),
            model_result("## 2026-06-11\nCompressed thing"),
        ]
    )
    # The save path calls the model once for the immediate summary and once for
    # compression, so the iterator order mirrors that control flow.
    monkeypatch.setattr(pipeline_mod.summarize, "call_model", lambda _prompt, _cfg: next(responses))
    today = env.today()  # pinned before the act so a midnight rollover can't flake the assert

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert (tmp_path / "data" / "now.md").read_text(encoding="utf-8") == ""
    assert "Compressed thing" in (tmp_path / "data" / ("today-%s.md" % today)).read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["error", "empty"], ids=["model-error", "empty-result"])
def test_run_save_failed_compression_preserves_now(tmp_path, monkeypatch, make_project_env, failure):
    """Keep the session buffer intact when hourly compression fails.

    Expected: the save itself succeeds, and a failing compression call leaves
    `now.md` holding the just-saved entry with no `today-*.md` created — the
    buffer truncation must stay strictly after a successful model response, or
    every failed compression would erase uncompressed memory.
    """
    env = make_project_env({"features": {"hourly_compression": True}})
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
    today = env.today()  # pinned before the act so a midnight rollover can't flake the assert

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert (tmp_path / "data" / "now.md").read_text(encoding="utf-8") == "\n## 11:00 | main\nCaptured thing\n"
    assert not (tmp_path / "data" / ("today-%s.md" % today)).exists()


def test_run_consolidation_writes_sections_and_marks_staging_done(tmp_path, monkeypatch, make_project_env):
    """Consolidate a past daily memory file with a fake model response.

    Expected: recent and archive files are replaced from the parsed sections,
    new core memory text is appended, the prompt includes the staging filename,
    and the processed daily file is renamed to `.done.md`.
    """
    env = make_project_env()
    staging = _seed_consolidation(tmp_path, env)
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
def test_run_consolidation_failure_leaves_memory_untouched(tmp_path, monkeypatch, make_project_env, failure):
    """Abort consolidation without consuming staging or rewriting memory.

    Expected: when the model errors out or returns an empty response, the run
    exits 1, the staging file keeps its name (no `.done.md`), and recent.md /
    archive.md are byte-identical — renaming staging before validation would
    silently destroy a day of memory on every bad model reply.
    """
    env = make_project_env()
    staging = _seed_consolidation(tmp_path, env)

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


def test_run_consolidation_missing_archive_section_keeps_archive(tmp_path, monkeypatch, make_project_env):
    """Tolerate a reply that only rewrites recent memory.

    Expected: a response with a RECENT section but no ARCHIVE section replaces
    recent.md, leaves the existing archive.md unchanged rather than blanking
    it, and still consumes the staging file.
    """
    env = make_project_env()
    staging = _seed_consolidation(tmp_path, env)
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


def test_run_save_skips_when_lock_held(tmp_path, monkeypatch, make_project_env):
    """Return cleanly when another process holds the save lock.

    Expected: with a live PID recorded in `save.lock`, `run_save` exits 0
    without calling the model or touching session state — the lock is the
    only guard against two detached saves (Claude and Codex at once)
    interleaving writes to now.md, and returning nonzero would mark the
    spawning hook as failed for a perfectly normal race.
    """
    env = make_project_env()
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(codex_user("Locked out") + "\n", encoding="utf-8")
    env.ensure_dirs()
    # This test process is the live holder, so the lock cannot be stolen.
    with open(os.path.join(env.state_dir, "save.lock"), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    monkeypatch.setattr(
        pipeline_mod.summarize,
        "call_model",
        lambda _prompt, _cfg: pytest.fail("a contended save must not call the model"),
    )

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 0

    assert not (tmp_path / "data" / "now.md").exists()
    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert (state.line, state.last_attempt_ts) == (0, 0.0)


def test_rotate_logs_drops_stale_and_caps_background(tmp_path, make_project_env):
    """Rotate logs without touching fresh files.

    Expected: a daily log older than 30 days is deleted, today's log
    survives, an oversized background.log is emptied, and a small one is
    left alone — consolidation runs this every time, so a boundary mistake
    would silently destroy the silent-by-design pipeline's only diagnostic
    surface.
    """
    env = make_project_env()
    env.ensure_dirs()
    logs = tmp_path / "data" / "logs"
    stale = logs / "memory-2000-01-01.log"
    stale.write_text("old", encoding="utf-8")
    age_file(stale, 31 * 86400)
    fresh = logs / ("memory-%s.log" % env.today())
    fresh.write_text("fresh\n", encoding="utf-8")
    background = logs / "background.log"
    background.write_text("x" * (512 * 1024 + 1), encoding="utf-8")

    env.rotate_logs()

    assert not stale.exists()
    assert fresh.read_text(encoding="utf-8").startswith("fresh")
    assert background.read_text(encoding="utf-8") == ""

    background.write_text("small", encoding="utf-8")
    env.rotate_logs()
    assert background.read_text(encoding="utf-8") == "small"


def test_run_save_rejects_malformed_header(tmp_path, monkeypatch, make_project_env):
    """Refuse to append a summary whose first line is not a journal header.

    Expected: `run_save` returns 1, nothing is written to `now.md`, and the
    session line marker is not persisted, so the same transcript lines are
    retried on the next save instead of being lost.
    """
    env = make_project_env()
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(codex_user("Do a thing") + "\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_mod.summarize,
        "call_model",
        lambda _prompt, _cfg: model_result("I could not find a header for this session."),
    )

    assert pipeline_mod.run_save(env, "codex", "s1", str(transcript)) == 1

    assert not (tmp_path / "data" / "now.md").exists()
    assert SessionState.load(env.sessions_dir, "codex", "s1").line == 0


# --- Extract formatting ----------------------------------------------------------


def test_format_exchanges():
    """Format parsed transcript messages into the model prompt excerpt.

    Expected: the output includes the session id, total transcript line count,
    and clearly labeled HUMAN/AGENT blocks so the summarizer sees structured
    context.
    """
    text = format_exchanges("sid", 12, [("HUMAN", "hi"), ("AGENT", "hello")])
    assert text.startswith("Session: sid\nLines: 12")
    assert "[HUMAN]\nhi" in text
    assert "[AGENT]\nhello" in text


def test_last_entry(tmp_path):
    """Find the last saved `now.md` entry for deduplication context.

    Expected: a missing file reports no previous entry, and a file with multiple
    `## time | branch` blocks returns only the final block.
    """
    now = tmp_path / "now.md"
    assert _last_entry(str(now)) == "(no previous entry)"
    now.write_text("\n## 10:00 | main\nfirst\n\n## 11:30 | main\nsecond thing\n")
    assert _last_entry(str(now)) == "## 11:30 | main\nsecond thing"


# --- Consolidation parsing -------------------------------------------------------


def test_parse_consolidation_full():
    """Split a complete consolidation response into recent and archive files.

    Expected: `===RECENT===` and `===ARCHIVE===` markers are stripped, each
    section keeps its Markdown heading, and the optional core section is empty.
    """
    text = "===RECENT===\n# Recent\n\nA\n\n===ARCHIVE===\n# Archive\n\nB"
    recent, archive, core = parse_consolidation_response(text)
    assert recent == "# Recent\n\nA"
    assert archive == "# Archive\n\nB"
    assert core == ""


def test_parse_consolidation_core_section():
    """Parse a consolidation response that promotes a new core memory.

    Expected: recent and archive are parsed as usual, and text after
    `===CORE===` is returned separately for appending to `core-memories.md`.
    """
    text = "===RECENT===\n# Recent\n\nA\n===ARCHIVE===\n# Archive\n\nB\n===CORE===\n- 2026-06-10: chose MIT"
    recent, archive, core = parse_consolidation_response(text)
    assert recent == "# Recent\n\nA"
    assert archive == "# Archive\n\nB"
    assert core == "- 2026-06-10: chose MIT"


def test_parse_consolidation_fallbacks():
    """Treat unmarked consolidation output as replacement recent memory.

    Expected: bare content becomes a `# Recent` document, while archive and core
    stay empty. This is the forgiving path for imperfect model formatting.
    """
    recent, archive, core = parse_consolidation_response("bare content")
    assert recent == "# Recent\n\nbare content"
    assert archive == ""
    assert core == ""


def test_parse_consolidation_strips_fences():
    """Ignore Markdown code fences copied into model consolidation output.

    Expected: fence-only lines are removed before marker parsing, so recent and
    archive content are extracted normally even if the model wrapped examples in
    triple backticks.
    """
    text = "```\n===RECENT===\n# Recent\n\nA\n```\n===ARCHIVE===\n```\n# Archive\n\nB\n```"
    recent, archive, core = parse_consolidation_response(text)
    assert recent == "# Recent\n\nA"
    assert archive == "# Archive\n\nB"
    assert core == ""


def test_append_core(tmp_path):
    """Append promoted core-memory lines to the persistent core file.

    Expected: the first append creates the `# Core Memories` heading, later
    appends preserve the existing content and add one line per promoted fact,
    and the appended lines are returned for logging.
    """
    path = str(tmp_path / "core-memories.md")
    assert _append_core(path, "- 2026-06-10: a") == ["- 2026-06-10: a"]
    assert _append_core(path, "- 2026-06-11: b") == ["- 2026-06-11: b"]
    with open(path) as f:
        assert f.read() == "# Core Memories\n\n- 2026-06-10: a\n- 2026-06-11: b\n"


def test_append_core_filters_malformed_lines(tmp_path):
    """Keep only `- YYYY-MM-DD: fact` lines when promoting core memories.

    Expected: model commentary, undated bullets, and whitespace-only sections
    never reach core-memories.md (which is injected verbatim into every future
    session), and a section with no valid line writes nothing at all.
    """
    path = str(tmp_path / "core-memories.md")
    kept = _append_core(path, "Here are the memories:\n- 2026-06-12: real fact\n- undated noise\n")
    assert kept == ["- 2026-06-12: real fact"]
    with open(path) as f:
        assert f.read() == "# Core Memories\n\n- 2026-06-12: real fact\n"

    assert _append_core(str(tmp_path / "other.md"), "   \n") == []
    assert not (tmp_path / "other.md").exists()
