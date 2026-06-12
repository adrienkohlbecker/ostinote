import json

from ostinote.pipeline import _last_entry, format_exchanges, parse_consolidation_response
from ostinote.summarize import parse_response

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


# --- Summarizer response parsing ---------------------------------------------------


def test_parse_response_dict():
    """Parse the common JSON object shape returned by the summarizer command.

    Expected: result text, skip status, token counts, and reported cost are all
    copied into the `ModelResult` wrapper.
    """
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
    """Parse the newer list-of-events JSON shape and recognize SKIP.

    Expected: the last result event is used, and a `SKIP` response is marked as
    a skip so the pipeline advances state without writing memory.
    """
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
    """Accept plain text summarizer output.

    Expected: non-JSON stdout is treated as the model text verbatim, which keeps
    alternate summarizer commands usable.
    """
    r = parse_response("just words")
    assert r.text == "just words"


# --- Consolidation parsing -----------------------------------------------------------


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


def test_append_core(tmp_path):
    """Append promoted core-memory lines to the persistent core file.

    Expected: the first append creates the `# Core Memories` heading, later
    appends preserve the existing content and add one line per promoted fact,
    and the appended lines are returned for logging.
    """
    from ostinote.pipeline import _append_core

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
    from ostinote.pipeline import _append_core

    path = str(tmp_path / "core-memories.md")
    kept = _append_core(path, "Here are the memories:\n- 2026-06-12: real fact\n- undated noise\n")
    assert kept == ["- 2026-06-12: real fact"]
    with open(path) as f:
        assert f.read() == "# Core Memories\n\n- 2026-06-12: real fact\n"

    assert _append_core(str(tmp_path / "other.md"), "   \n") == []
    assert not (tmp_path / "other.md").exists()


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


# --- Prompt building -----------------------------------------------------------


def test_build_save_prompt_single_pass_substitution():
    """Substitute placeholders without re-expanding placeholder-shaped values.

    Expected: a transcript extract containing the literal text `{{LAST_ENTRY}}`
    survives verbatim instead of being replaced with the last-entry value —
    untrusted content must not be able to pull other fields into itself.
    """
    from ostinote.prompts import build_save_prompt

    prompt = build_save_prompt(
        time="10:00",
        branch="main",
        last_entry="previous entry text",
        extract="user pasted {{LAST_ENTRY}} into the session",
    )
    assert "## 10:00 | main" in prompt
    assert "user pasted {{LAST_ENTRY}} into the session" in prompt
    assert prompt.count("previous entry text") == 1


def test_build_consolidation_prompt_tags_staging_files():
    """Wrap each staging file in a labeled tag, not bare `---` fences.

    Expected: every file appears as `<staging_file name="...">content</staging_file>`
    so content containing dashes cannot blur file boundaries, files are sorted
    by name, and an empty core file is rendered as `(empty)`.
    """
    from ostinote.prompts import build_consolidation_prompt

    prompt = build_consolidation_prompt(
        {"today-2026-06-11.md": "## 09:00 | main\n---\nA", "today-2026-06-10.md": "B"},
        recent="# Recent",
        archive="# Archive",
        core="",
    )
    assert '<staging_file name="today-2026-06-10.md">\nB\n</staging_file>' in prompt
    assert '<staging_file name="today-2026-06-11.md">\n## 09:00 | main\n---\nA\n</staging_file>' in prompt
    assert prompt.index("today-2026-06-10.md") < prompt.index("today-2026-06-11.md")
    assert "(empty)" in prompt
