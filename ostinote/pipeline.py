"""Core memory pipeline: save, hourly compression, consolidation.

Layered compression, each layer feeding the next:

    transcript ──save──▶ now.md ──compress──▶ today-YYYY-MM-DD.md ──consolidate──▶
        recent.md + archive.md

``run_save`` is invoked as a detached background process by the hooks; all
mutations of the shared memory files happen under ``state/save.lock`` so
parallel sessions (Claude and Codex at once) can never interleave writes.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta

from . import prompts, summarize, tzutil
from .agents import get_agent
from .agents.base import Message, TranscriptReadError
from .env import Env
from .state import PidLock, SessionState, read_day, read_ts, write_day, write_ts

_HEADER_RE = re.compile(r"^## ([0-9]{2}:[0-9]{2}|[0-9]{1,2}:[0-9]{2} (AM|PM)) \|")


def format_exchanges(session_id: str, total_lines: int, messages: list[Message]) -> str:
    """Render parsed transcript messages as the plain-text extract fed to the model.

    The header records the session id and total transcript line count (the
    position the save will advance to); each message follows as a ``[ROLE]``
    block. Also what ``save --dry`` prints for inspection.
    """
    lines = ["Session: %s" % session_id, "Lines: %d" % total_lines, "=" * 60]
    for role, text in messages:
        lines.append("\n[%s]" % role)
        lines.append(text)
        lines.append("-" * 40)
    return "\n".join(lines)


def _last_entry(now_file: str) -> str:
    """Last `## ` block of now.md, given to the model as dedup context."""
    try:
        with open(now_file, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return "(no previous entry)"
    idx = content.rfind("\n## ")
    if idx == -1:
        return content.strip() if content.startswith("## ") else "(no previous entry)"
    return content[idx + 1 :].strip()


def run_save(
    env: Env,
    agent_name: str,
    session_id: str | None = None,
    transcript_path: str | None = None,
    force: bool = False,
    dry: bool = False,
    final: bool = False,
) -> int:
    """Save one session's new exchanges into now.md. Returns exit code.

    ``final`` (session just ended) skips the cooldown — staleness doesn't
    apply — but keeps the min-human-messages gate, so closing a trivial
    session doesn't burn a model call. ``force`` skips both.
    """
    env.ensure_dirs()
    agent = get_agent(agent_name)

    if not transcript_path:
        transcript_path = agent.find_latest_transcript(env.cwd)
        if not transcript_path:
            env.log("save", "ERROR: no transcript found for %s in %s" % (agent_name, env.cwd))
            return 1
    if not session_id:
        session_id = os.path.basename(transcript_path).rsplit(".", 1)[0]

    lock = PidLock(os.path.join(env.state_dir, "save.lock"))
    if not lock.acquire():
        if env.cfg["debug"]:
            env.log("save", "lock held, skipping (%s/%s)" % (agent_name, session_id))
        return 0

    try:
        return _save_locked(env, agent, session_id, transcript_path, force, dry, final)
    finally:
        lock.release()


def _save_locked(env: Env, agent, session_id: str, transcript_path: str, force: bool, dry: bool, final: bool) -> int:
    state = SessionState.load(env.sessions_dir, agent.name, session_id)
    state.transcript_path = transcript_path
    state.cwd = env.cwd

    # --- Cooldown (per session) ---
    cooldown = env.cfg["cooldowns"]["save_seconds"]
    elapsed = time.time() - state.last_attempt_ts
    if not force and not final and not dry and elapsed < cooldown:
        if env.cfg["debug"]:
            env.log("cooldown", "%ds < %ds, skip (%s)" % (elapsed, cooldown, session_id))
        return 0

    # --- Extract ---
    env.log("extract", "%s session %s from line %d" % (agent.name, session_id, state.line))
    try:
        messages, total_lines, last_time = agent.read(transcript_path, skip_lines=state.line)
    except TranscriptReadError as exc:
        env.log("extract", "ERROR: %s" % exc)
        return 1
    if not dry:
        state.last_attempt_ts = time.time()
        state.save()

    human_count = sum(1 for role, _ in messages if role == "HUMAN")
    env.log("extract", "%d exchanges (%d human)" % (len(messages), human_count))
    if not messages:
        return 0

    min_human = env.cfg["thresholds"]["min_human_messages"]
    if human_count < min_human and not force and not dry:
        env.log("extract", "%d human msgs < %d, skip" % (human_count, min_human))
        return 0

    extract = format_exchanges(session_id, total_lines, messages)
    if dry:
        print("\n=== DRY RUN (%s %s) ===\n" % (agent.name, session_id))
        print(extract)
        return 0

    # --- Summarize ---
    # Date the entry by when the conversation happened, not by when this save
    # runs: recovery and end-of-session saves can run days later.
    today = env.today()
    activity = tzutil.local(last_time, env.tz) if last_time else tzutil.now(env.tz)
    entry_day = min(activity.strftime("%Y-%m-%d"), today)
    prompt = prompts.build_save_prompt(
        time=tzutil.clock(activity, env.cfg["time_format"]),
        branch=env.git_branch(),
        last_entry=_last_entry(env.now_file),
        extract=extract,
    )
    env.log("model", "summarizing (branch: %s)" % env.git_branch())
    try:
        result = summarize.call_model(prompt, env.cfg)
    except RuntimeError as e:
        env.log("model", "ERROR: %s" % e)
        return 1
    env.log_tokens(
        "save",
        result.tokens.input,
        result.tokens.output,
        result.tokens.cache,
        result.tokens.cost_usd,
    )

    text = _strip_fences(result.text)
    if not text:
        env.log("model", "ERROR: empty response")
        return 1

    # --- Advance position (also on SKIP — those lines are accounted for) ---
    state.line = total_lines
    if result.is_skip:
        state.save()
        env.log("model", "SKIP — position → %d" % total_lines)
        return 0

    # Reject malformed entries instead of appending them: now.md feeds two
    # more model stages, so a bad header corrupts the whole chain. The new
    # position is not persisted on this path, so the lines are retried on the
    # next save (the cooldown gates how often).
    entry_lines = text.splitlines()
    if not _HEADER_RE.match(entry_lines[0]):
        env.log("validate", "ERROR: malformed entry rejected: %s" % entry_lines[0][:80])
        return 1
    if len(entry_lines) != 2:
        env.log("validate", "WARNING: entry has %d lines, expected 2" % len(entry_lines))

    # --- File the entry under its day ---
    flush_stale_buffer(env, today)
    if entry_day < today:
        _append_block(env.today_file(entry_day), text)
        env.log("write", "filed under today-%s.md: %s (position → %d)" % (entry_day, entry_lines[0][:80], total_lines))
    else:
        with open(env.now_file, "a", encoding="utf-8") as f:
            f.write("\n" + text + "\n")
        write_day(_buffer_day_marker(env), today)
        env.log("write", "appended: %s (position → %d)" % (entry_lines[0][:80], total_lines))
    state.last_save_ts = time.time()
    state.save()

    # --- Hourly compression (still under the save lock) ---
    if env.cfg["features"]["hourly_compression"]:
        _maybe_compress(env, today)
    return 0


def _buffer_day_marker(env: Env) -> str:
    return os.path.join(env.state_dir, "now-day")


def buffer_day(env: Env) -> str | None:
    """Day (YYYY-MM-DD) the entries in now.md belong to; None when it is empty.

    Read from the ``state/now-day`` marker each append writes. A now.md
    written before the marker existed is dated by its mtime, the time of its
    last append.
    """
    try:
        with open(env.now_file, encoding="utf-8") as f:
            if not f.read().strip():
                return None
        mtime = os.path.getmtime(env.now_file)
    except OSError:
        return None
    return read_day(_buffer_day_marker(env)) or datetime.fromtimestamp(mtime, env.tz).strftime("%Y-%m-%d")


def flush_stale_buffer(env: Env, today: str) -> str | None:
    """Move a past day's now.md verbatim into that day's ``today-*.md``.

    now.md only ever holds one day: entries saved after that day's last
    hourly compression would otherwise wait for the project's next save and
    be compressed into whatever day that falls on. No model call, so it is
    cheap enough for session start. The caller holds the save lock. Returns
    the day moved, or None when now.md is empty or still today's.
    """
    day = buffer_day(env)
    if not day or day >= today:
        return None
    with open(env.now_file, encoding="utf-8") as f:
        content = f.read()
    _append_block(env.today_file(day), content.strip())
    # Truncate only after the content is safely in the day file.
    with open(env.now_file, "w", encoding="utf-8") as f:
        f.write("")
    env.log("compress", "now.md from %s → today-%s.md (verbatim)" % (day, day))
    return day


def _append_block(path: str, text: str) -> None:
    """Append ``text`` to a daily file, blank-line separated from what is there."""
    with open(path, "a", encoding="utf-8") as f:
        if os.path.exists(path) and os.path.getsize(path):
            f.write("\n")
        f.write(text + "\n")


def _maybe_compress(env: Env, today: str) -> None:
    """Compress now.md (today's entries) into today's daily file (upstream calls this NDC)."""
    marker = os.path.join(env.state_dir, "last-compress.ts")
    if time.time() - read_ts(marker) < env.cfg["cooldowns"]["compress_seconds"]:
        return
    try:
        with open(env.now_file, encoding="utf-8") as f:
            now_content = f.read()
    except OSError:
        return
    if not now_content.strip():
        return

    write_ts(marker)
    env.log("compress", "now.md → today-%s.md" % today)
    try:
        result = summarize.call_model(prompts.build_compress_prompt(now_content), env.cfg)
    except RuntimeError as e:
        env.log("compress", "ERROR: %s" % e)
        return
    text = _strip_fences(result.text)
    if not text:
        env.log("compress", "ERROR: produced empty result")
        return

    _append_block(env.today_file(today), text)
    # Truncate the buffer — its content now lives in today-*.md.
    with open(env.now_file, "w", encoding="utf-8") as f:
        f.write("")
    env.log_tokens(
        "compress",
        result.tokens.input,
        result.tokens.output,
        result.tokens.cache,
        result.tokens.cost_usd,
    )
    env.log("compress", "%db → %db" % (len(now_content), len(text)))


# --- Consolidation -----------------------------------------------------------


def staging_files(env: Env) -> list[str]:
    """Past-day ``today-*.md`` files awaiting consolidation."""
    out = []
    today_name = os.path.basename(env.today_file())
    try:
        names = sorted(os.listdir(env.data_dir))
    except OSError:
        return out
    for name in names:
        if name.startswith("today-") and name.endswith(".md") and not name.endswith(".done.md") and name != today_name:
            out.append(os.path.join(env.data_dir, name))
    return out


def run_consolidation(env: Env) -> int:
    """Merge past-day staging files into recent.md / archive.md."""
    env.ensure_dirs()
    env.rotate_logs()

    lock = PidLock(os.path.join(env.state_dir, "consolidate.lock"))
    if not lock.acquire():
        env.log("consolidation", "locked, skip")
        return 0
    try:
        staging = staging_files(env)
        if not staging:
            env.log("consolidation", "no staging files")
            return 0

        contents = {}
        unreadable = []
        for path in staging:
            try:
                with open(path, encoding="utf-8") as f:
                    contents[os.path.basename(path)] = f.read()
            except OSError as exc:
                unreadable.append((path, exc))
        if unreadable:
            for path, exc in unreadable:
                env.log("consolidation", "ERROR: cannot read staging file %s: %s" % (os.path.basename(path), exc))
            return 1

        recent = _read_or_empty(env.recent_file)
        archive = _read_or_empty(env.archive_file)
        core = _read_or_empty(env.core_memories_file)

        env.log("consolidation", "start: %d staging file(s)" % len(contents))
        cfg = dict(env.cfg)
        cfg["summarizer"] = dict(env.cfg["summarizer"])
        cfg["summarizer"]["timeout"] = max(180, cfg["summarizer"]["timeout"])
        today = env.today()
        try:
            result = summarize.call_model(
                prompts.build_consolidation_prompt(
                    contents,
                    recent,
                    archive,
                    core,
                    today=today,
                    archive_cutoff=_archive_cutoff(today),
                ),
                cfg,
            )
        except RuntimeError as e:
            env.log("consolidation", "ERROR: %s" % e)
            return 1

        new_recent, new_archive, new_core = parse_consolidation_response(result.text)
        if not new_recent:
            env.log("consolidation", "ERROR: empty recent section, aborting")
            return 1
        if not new_archive and archive.strip():
            env.log("consolidation", "WARNING: reply had no archive section; archive.md left unchanged")

        # The model rewrites the whole archive and has silently dropped weeks
        # while logging success. Every day it was given must survive somewhere,
        # and every archived week must outlive the rewrite; otherwise its
        # archive is discarded and the uncovered days are folded in verbatim.
        days = _staged_days(contents)
        for day, text in _dated_sections(recent):
            days.setdefault(day, []).append(text)
        archive_out = new_archive or archive
        gaps = _archive_gaps(days, archive, new_recent, archive_out)
        if gaps:
            env.log(
                "consolidation",
                "WARNING: reply dropped week(s) %s; keeping old archive and appending the missing days verbatim"
                % ", ".join(week.isoformat() for week in gaps),
            )
            recent_days = {day for day, _ in _dated_sections(new_recent)}
            archive_out = _fold_days(archive, {day: texts for day, texts in days.items() if day not in recent_days})
        archive_out = _sort_weeks(archive_out)

        # Consolidation rewrites long-term memory wholesale from model output,
        # so keep a one-deep .bak of each file it replaces; a bad run can be
        # undone by hand (staging inputs survive as .done.md as well).
        _backup(env.recent_file)
        _write(env.recent_file, new_recent + "\n")
        if archive_out.strip() and archive_out != archive:
            _backup(env.archive_file)
            _write(env.archive_file, archive_out.rstrip("\n") + "\n")
        if new_core:
            promoted = _append_core(env.core_memories_file, new_core)
            if promoted:
                env.log("consolidation", "core memories promoted: %s" % promoted[0])
            else:
                env.log("consolidation", "WARNING: core section had no valid '- YYYY-MM-DD:' lines, skipped")
        env.log_tokens(
            "consolidation",
            result.tokens.input,
            result.tokens.output,
            result.tokens.cache,
            result.tokens.cost_usd,
        )

        for path in staging:
            if os.path.exists(path):
                _mark_done(path)
        env.log("consolidation", "done: %d file(s) consolidated" % len(staging))
        return 0
    finally:
        lock.release()


def run_archive_repair(env: Env, dry: bool = False) -> int:
    """Fold consumed daily files that no memory file covers back into archive.md.

    Recovers weeks an earlier consolidation dropped: a ``today-*.done.md`` day
    is missing when recent.md has no entry for it and archive.md no week
    containing it (for a week made of ``- YYYY-MM-DD:`` lines, as this
    command writes them, only the listed days count). Missing days are appended verbatim (flattened) under their
    Monday week, without a model call; the previous archive is kept as .bak.
    ``dry`` only prints what would be added.
    """
    env.ensure_dirs()
    lock = PidLock(os.path.join(env.state_dir, "consolidate.lock"))
    if not lock.acquire():
        print("consolidation running, try again later")
        return 1
    try:
        days: dict[date, list[str]] = {}
        for name in sorted(os.listdir(env.data_dir)):
            day = _staging_day(name, ".done.md")
            if day:
                days[day] = [_read_or_empty(os.path.join(env.data_dir, name))]
        recent = _read_or_empty(env.recent_file)
        archive = _read_or_empty(env.archive_file)
        covered_weeks, listed_days = _archive_coverage(archive)
        covered_days = {day for day, _ in _dated_sections(recent)} | listed_days
        missing = {
            day: texts for day, texts in days.items() if day not in covered_days and _monday(day) not in covered_weeks
        }
        if not missing:
            print("nothing to repair")
            return 0
        print("missing day(s): %s" % ", ".join(day.isoformat() for day in sorted(missing)))
        if dry:
            return 0
        _backup(env.archive_file)
        _write(env.archive_file, _sort_weeks(_fold_days(archive, missing)).rstrip("\n") + "\n")
        env.log("repair", "archive.md: restored %d day(s) from .done.md files" % len(missing))
        print("archive.md updated (previous copy: archive.md.bak)")
        return 0
    finally:
        lock.release()


# --- Archive coverage ----------------------------------------------------------

_SECTION_DATE_RE = re.compile(r"## (?:Week of )?(\d{4}-\d{2}-\d{2})\b")


def _parse_day(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _staging_day(name: str, suffix: str = ".md") -> date | None:
    """Date encoded in a ``today-YYYY-MM-DD<suffix>`` filename, if any."""
    if not (name.startswith("today-") and name.endswith(suffix)):
        return None
    return _parse_day(name[len("today-") : -len(suffix)])


def _staged_days(contents: dict[str, str]) -> dict[date, list[str]]:
    days: dict[date, list[str]] = {}
    for name, text in contents.items():
        day = _staging_day(name)
        if day:
            days.setdefault(day, []).append(text)
    return days


def _split_sections(text: str) -> tuple[str, list[str]]:
    """Split markdown into the text before the first ``## `` header and each ``## `` section."""
    parts = re.split(r"(?m)^(?=## )", text)
    return parts[0], parts[1:]


def _section_day(section: str) -> date | None:
    m = _SECTION_DATE_RE.match(section)
    return _parse_day(m.group(1)) if m else None


def _dated_sections(text: str) -> list[tuple[date, str]]:
    """``(date, body)`` for each ``## YYYY-MM-DD`` / ``## Week of`` section."""
    out = []
    for section in _split_sections(text)[1]:
        day = _section_day(section)
        if day:
            out.append((day, section.split("\n", 1)[1] if "\n" in section else ""))
    return out


def _archive_weeks(archive: str) -> set[date]:
    # Normalized to Monday so a week header dated mid-week still counts.
    return {_monday(day) for day, _ in _dated_sections(archive)}


_DAY_LINE_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}): ", re.M)


def _archive_coverage(archive: str) -> tuple[set[date], set[date]]:
    """Return (summarized weeks, individually listed days) of an archive.

    A week whose body lists ``- YYYY-MM-DD:`` lines (verbatim days folded in
    by a repair or the consolidation guard) covers only those days; any other
    week is a summary and covers all seven.
    """
    weeks: set[date] = set()
    days: set[date] = set()
    for day, body in _dated_sections(archive):
        listed = {d for text in _DAY_LINE_RE.findall(body) if (d := _parse_day(text))}
        if listed:
            days |= listed
        else:
            weeks.add(_monday(day))
    return weeks, days


def _archive_gaps(days: dict[date, list[str]], old_archive: str, new_recent: str, new_archive: str) -> list[date]:
    """Weeks lost by a consolidation reply, oldest first.

    A day is kept when the new recent.md has its entry or the new archive.md
    a week containing it; an old archive week is kept when the new archive
    still has it.
    """
    recent_days = {day for day, _ in _dated_sections(new_recent)}
    weeks = _archive_weeks(new_archive)
    gaps = {_monday(day) for day in days if day not in recent_days and _monday(day) not in weeks}
    gaps |= _archive_weeks(old_archive) - weeks
    return sorted(gaps)


def drop_oldest_section(text: str) -> str | None:
    """Remove the oldest dated ``## `` section of a memory file.

    Goes by the header date, not file position, since models write archive
    weeks in either order. Once no dated section is left, any remaining
    content is dropped whole (returns ""); returns None when ``text`` has
    nothing left to drop.
    """
    preamble, sections = _split_sections(text)
    dated = [(day, i) for i, section in enumerate(sections) if (day := _section_day(section))]
    if dated:
        _, oldest = min(dated)
        return preamble + "".join(section for i, section in enumerate(sections) if i != oldest)
    return "" if text.strip() else None


def _flatten(text: str) -> str:
    """One line of a day's text: header lines dropped, the rest joined by spaces."""
    lines = [ln.strip() for ln in text.splitlines()]
    return " ".join(ln for ln in lines if ln and not ln.startswith("#") and not _FENCE_RE.match(ln))


def _fold_days(archive: str, days: dict[date, list[str]]) -> str:
    """Append each day as a ``- YYYY-MM-DD: text`` line under its Monday week."""
    preamble, sections = _split_sections(archive)
    # Every block must end in a newline, or the next "## " header would not
    # start a line and the week would merge into the text before it.
    preamble = (preamble.rstrip("\n") or "# Archive") + "\n\n"
    sections = [section.rstrip("\n") + "\n\n" for section in sections]
    by_week = {}
    for i, section in enumerate(sections):
        day = _section_day(section)
        if day:
            by_week.setdefault(_monday(day), i)
    for day in sorted(days):
        line = "- %s: %s" % (day.isoformat(), " ".join(_flatten(text) for text in days[day]).strip())
        week = _monday(day)
        if week in by_week:
            i = by_week[week]
            sections[i] = sections[i].rstrip("\n") + "\n" + line + "\n\n"
        else:
            by_week[week] = len(sections)
            sections.append("## Week of %s\n%s\n\n" % (week.isoformat(), line))
    return preamble + "".join(sections)


def _sort_weeks(archive: str) -> str:
    """Order archive sections chronologically; undated sections go last."""
    preamble, sections = _split_sections(archive)
    if not sections:
        return archive
    ordered = sorted(sections, key=lambda section: _section_day(section) or date.max)
    blocks = [preamble.strip()] if preamble.strip() else []
    blocks += [section.strip() for section in ordered]
    return "\n\n".join(blocks) + "\n"


def parse_consolidation_response(text: str):
    """Split the model response on ===RECENT=== / ===ARCHIVE=== / ===CORE=== markers.

    Returns ``(recent, archive, core)``. CORE is optional and holds only NEW
    items to append; a response with no markers at all is treated as RECENT.
    """
    text = _strip_fences(text)
    sections = {"RECENT": "", "ARCHIVE": "", "CORE": ""}
    parts = re.split(r"===(RECENT|ARCHIVE|CORE)===", text)
    if len(parts) == 1:
        sections["RECENT"] = text
    else:
        for marker, content in zip(parts[1::2], parts[2::2], strict=False):
            sections[marker] = content

    recent = _strip_fences(sections["RECENT"])
    archive = _strip_fences(sections["ARCHIVE"])
    core = _strip_fences(sections["CORE"])
    if recent and not recent.startswith("# Recent"):
        recent = "# Recent\n\n" + recent
    if archive and not archive.startswith("# Archive"):
        archive = "# Archive\n\n" + archive
    return recent, archive, core


def _archive_cutoff(today: str) -> str:
    """Return the newest date that should age from recent.md into archive.md."""
    return (date.fromisoformat(today) - timedelta(days=3)).isoformat()


_CORE_LINE_RE = re.compile(r"^- \d{4}-\d{2}-\d{2}: \S")


def _append_core(path: str, new_core: str) -> list[str]:
    """Append promoted core-memory lines, keeping only well-formed ones.

    core-memories.md is injected verbatim into every future session, so only
    lines matching the `- YYYY-MM-DD: fact` contract are appended; whitespace
    and free-form model commentary are dropped. Returns the appended lines so
    the caller can log what was actually promoted.
    """
    lines = [ln.strip() for ln in new_core.splitlines() if _CORE_LINE_RE.match(ln.strip())]
    if not lines:
        return []
    existing = _read_or_empty(path)
    with open(path, "a", encoding="utf-8") as f:
        if not existing.strip():
            f.write("# Core Memories\n\n")
        elif not existing.endswith("\n"):
            f.write("\n")
        f.write("\n".join(lines) + "\n")
    return lines


_FENCE_RE = re.compile(r"^```[\w+-]*$")


def _strip_fences(text: str) -> str:
    """Drop markdown code-fence lines (bare or language-tagged) the model wraps output in.

    Memory files are plain text that later model stages re-read; a fence
    adds nothing there, and a stray one swallows the rest of the file when
    rendered.
    """
    lines = [ln for ln in text.strip().splitlines() if not _FENCE_RE.match(ln.strip())]
    return "\n".join(lines).strip()


def _mark_done(path: str) -> None:
    """Rename a consumed ``today-*.md`` to ``.done.md``, merging into an existing one.

    A late save can reopen a day that was already consolidated; its earlier
    ``.done.md`` keeps the day's verbatim history, so it is appended to, not
    replaced.
    """
    done = path[:-3] + ".done.md"
    if not os.path.exists(done):
        os.replace(path, done)
        return
    _append_block(done, _read_or_empty(path).strip())
    os.remove(path)


def _read_or_empty(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def _backup(path: str) -> None:
    """Copy ``path`` to ``path.bak`` (one generation) before it is replaced."""
    content = _read_or_empty(path)
    if content:
        _write(path + ".bak", content)
