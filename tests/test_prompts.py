from ostinote.prompts import build_consolidation_prompt, build_save_prompt


def test_build_save_prompt_single_pass_substitution():
    """Substitute placeholders without re-expanding placeholder-shaped values.

    Expected: a transcript extract containing the literal text `{{LAST_ENTRY}}`
    survives verbatim instead of being replaced with the last-entry value —
    untrusted content must not be able to pull other fields into itself.
    """
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
