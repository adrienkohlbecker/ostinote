"""Prompt template loading and {{PLACEHOLDER}} substitution."""

from __future__ import annotations

import os
import re

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _read(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), encoding="utf-8") as f:
        return f.read()


def _render(name: str, values: dict[str, str]) -> str:
    """Load a template and substitute its ``{{NAME}}`` placeholders in one pass.

    Single-pass substitution matters because the values are untrusted
    (transcripts, prior model output): placeholder-shaped text inside a value
    is emitted verbatim instead of being expanded with another field's value,
    which chained ``str.replace`` calls would do. Placeholders with no entry
    in ``values`` are left as-is.
    """
    return re.sub(
        r"\{\{([A-Z_]+)\}\}",
        lambda m: values.get(m.group(1), m.group(0)),
        _read(name),
    )


def build_save_prompt(time: str, branch: str, last_entry: str, extract: str) -> str:
    """Prompt asking the model to condense one session into a journal entry."""
    return _render(
        "save-session.prompt.txt",
        {"TIME": time, "BRANCH": branch, "LAST_ENTRY": last_entry, "EXTRACT": extract},
    )


def build_compress_prompt(now_content: str) -> str:
    """Prompt asking the model to compress the now.md day buffer."""
    return _render("compress-daily.prompt.txt", {"NOW_CONTENT": now_content})


def build_consolidation_prompt(staging: dict[str, str], recent: str, archive: str, core: str) -> str:
    """Prompt asking the model to fold staging logs into recent.md/archive.md.

    Each staging file is wrapped in a labeled ``<staging_file>`` tag rather
    than bare ``---`` fences so file content containing dashes (or text that
    mimics a fence) cannot blur the boundary between files.
    """
    staging_section = ""
    for filename in sorted(staging):
        staging_section += '\n<staging_file name="%s">\n%s\n</staging_file>\n' % (filename, staging[filename])
    return _render(
        "consolidate-staging.prompt.txt",
        {
            "STAGING_FILES": staging_section,
            "RECENT": recent,
            "ARCHIVE": archive,
            "CORE": core or "(empty)",
        },
    )
