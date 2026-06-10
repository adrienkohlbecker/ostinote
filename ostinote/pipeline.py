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

from . import prompts, summarize
from .agents import get_agent
from .agents.base import Message
from .env import Env
from .state import PidLock, SessionState, read_ts, write_ts

_HEADER_RE = re.compile(r"^## ([0-9]{2}:[0-9]{2}|[0-9]{1,2}:[0-9]{2} (AM|PM)) \|")


def format_exchanges(session_id: str, total_lines: int, messages: list[Message]) -> str:
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
) -> int:
    """Save one session's new exchanges into now.md. Returns exit code."""
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
        return _save_locked(env, agent, session_id, transcript_path, force, dry)
    finally:
        lock.release()


def _save_locked(
    env: Env, agent, session_id: str, transcript_path: str, force: bool, dry: bool
) -> int:
    state = SessionState.load(env.sessions_dir, agent.name, session_id)
    state.transcript_path = transcript_path

    # --- Cooldown (per session) ---
    cooldown = env.cfg["cooldowns"]["save_seconds"]
    elapsed = time.time() - state.last_attempt_ts
    if not force and not dry and elapsed < cooldown:
        if env.cfg["debug"]:
            env.log("cooldown", "%ds < %ds, skip (%s)" % (elapsed, cooldown, session_id))
        return 0

    # --- Extract ---
    env.log("extract", "%s session %s from line %d" % (agent.name, session_id, state.line))
    messages, total_lines = agent.parse(transcript_path, skip_lines=state.line)
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
    prompt = prompts.build_save_prompt(
        time=env.time_now(),
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

    text = result.text.strip()
    if not text:
        env.log("model", "ERROR: empty response")
        return 1

    # --- Advance position (also on SKIP — those lines are accounted for) ---
    state.line = total_lines
    if result.is_skip:
        state.save()
        env.log("model", "SKIP — position → %d" % total_lines)
        return 0

    if not _HEADER_RE.match(text.splitlines()[0]):
        env.log("validate", "WARNING: unexpected format: %s" % text.splitlines()[0][:80])

    # --- Append to now.md ---
    with open(env.now_file, "a", encoding="utf-8") as f:
        f.write("\n" + text + "\n")
    state.last_save_ts = time.time()
    state.save()
    env.log("write", "appended: %s (position → %d)" % (text.splitlines()[0][:80], total_lines))

    # --- Hourly compression (still under the save lock) ---
    if env.cfg["features"]["hourly_compression"]:
        _maybe_compress(env)
    return 0


def _maybe_compress(env: Env) -> None:
    """Compress now.md into today's daily file (upstream calls this NDC)."""
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
    env.log("compress", "now.md → today-%s.md" % env.today())
    try:
        result = summarize.call_model(prompts.build_compress_prompt(now_content), env.cfg)
    except RuntimeError as e:
        env.log("compress", "ERROR: %s" % e)
        return
    text = result.text.strip()
    if not text:
        env.log("compress", "ERROR: produced empty result")
        return

    today_file = env.today_file()
    with open(today_file, "a", encoding="utf-8") as f:
        if os.path.getsize(today_file) if os.path.exists(today_file) else 0:
            f.write("\n")
        f.write(text + "\n")
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
        if (
            name.startswith("today-")
            and name.endswith(".md")
            and not name.endswith(".done.md")
            and name != today_name
        ):
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
        for path in staging:
            try:
                with open(path, encoding="utf-8") as f:
                    contents[os.path.basename(path)] = f.read()
            except OSError:
                pass

        recent = _read_or_empty(env.recent_file)
        archive = _read_or_empty(env.archive_file)

        env.log("consolidation", "start: %d staging file(s)" % len(contents))
        cfg = dict(env.cfg)
        cfg["summarizer"] = dict(env.cfg["summarizer"])
        cfg["summarizer"]["timeout"] = max(180, cfg["summarizer"]["timeout"])
        try:
            result = summarize.call_model(
                prompts.build_consolidation_prompt(contents, recent, archive), cfg
            )
        except RuntimeError as e:
            env.log("consolidation", "ERROR: %s" % e)
            return 1

        new_recent, new_archive = parse_consolidation_response(result.text)
        if not new_recent:
            env.log("consolidation", "ERROR: empty recent section, aborting")
            return 1

        _write(env.recent_file, new_recent + "\n")
        if new_archive:
            _write(env.archive_file, new_archive + "\n")
        env.log_tokens(
            "consolidation",
            result.tokens.input,
            result.tokens.output,
            result.tokens.cache,
            result.tokens.cost_usd,
        )

        for path in staging:
            if os.path.exists(path):
                os.replace(path, path[:-3] + ".done.md")
        env.log("consolidation", "done: %d file(s) consolidated" % len(staging))
        return 0
    finally:
        lock.release()


def parse_consolidation_response(text: str):
    """Split the model response on ===RECENT=== / ===ARCHIVE=== markers."""
    text = _strip_fences(text)
    recent, archive = "", ""
    if "===RECENT===" in text and "===ARCHIVE===" in text:
        head, tail = text.split("===ARCHIVE===", 1)
        recent = head.replace("===RECENT===", "").strip()
        archive = tail.strip()
    elif "===RECENT===" in text:
        recent = text.replace("===RECENT===", "").strip()
    else:
        recent = text.strip()

    recent, archive = _strip_fences(recent), _strip_fences(archive)
    if recent and not recent.startswith("# Recent"):
        recent = "# Recent\n\n" + recent
    if archive and not archive.startswith("# Archive"):
        archive = "# Archive\n\n" + archive
    return recent, archive


def _strip_fences(text: str) -> str:
    """Drop markdown code-fence lines the model sometimes copies from the
    prompt's output-format example."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip() != "```"]
    return "\n".join(lines).strip()


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
