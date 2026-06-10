"""Prompt template loading and {{PLACEHOLDER}} substitution."""

from __future__ import annotations

import os

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _read(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), encoding="utf-8") as f:
        return f.read()


def build_save_prompt(time: str, branch: str, last_entry: str, extract: str) -> str:
    return (
        _read("save-session.prompt.txt")
        .replace("{{TIME}}", time)
        .replace("{{BRANCH}}", branch)
        .replace("{{LAST_ENTRY}}", last_entry)
        .replace("{{EXTRACT}}", extract)
    )


def build_compress_prompt(now_content: str) -> str:
    return _read("compress-daily.prompt.txt").replace("{{NOW_CONTENT}}", now_content)


def build_consolidation_prompt(
    staging: dict[str, str], recent: str, archive: str, core: str
) -> str:
    staging_section = ""
    for filename in sorted(staging):
        staging_section += "\n--- %s ---\n%s\n" % (filename, staging[filename])
    return (
        _read("consolidate-staging.prompt.txt")
        .replace("{{STAGING_FILES}}", staging_section)
        .replace("{{RECENT}}", recent)
        .replace("{{ARCHIVE}}", archive)
        .replace("{{CORE}}", core or "(empty)")
    )
