"""``ostinote`` command-line interface.

Subcommands:

    hook session-start|post-tool|session-end --agent claude|codex
        Lifecycle hook entry points (read agent JSON on stdin, never fail).
    save [--agent A] [--session ID] [--transcript PATH] [--force|--final] [--dry]
        Extract + summarize one session into now.md.
    consolidate
        Merge past-day staging files into recent.md / archive.md.
    status [--costs]
        Show resolved paths, memory files, and tracked sessions.
    doctor [--live]
        Check every link of the (silent-by-design) pipeline, loudly.
    install|uninstall claude|codex|all [--user|--project]
        (Un)register hooks and the /ostinote command.

Every command handler returns an int exit code and ``main`` returns it, so
the console-script wrapper and ``python -m ostinote`` hand it to ``sys.exit``.
The one exception is ``hook``, which always exits 0: agents treat nonzero
hook exits as errors (Claude Code blocks on exit code 2), so hook failures
are logged to ``HOOK_ERRORS_PATH`` instead of surfacing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from . import costs as costs_mod
from . import doctor as doctor_mod
from . import env as env_mod
from . import hooks as hooks_mod
from . import install as install_mod
from . import pipeline
from .agents import agent_names
from .env import Env

# Hook tracebacks can quote transcript-derived strings in exception messages;
# keep only the tail (innermost frames plus the message) so the error log
# stays debuggable without accumulating session content.
_TRACEBACK_TAIL_LINES = 30


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the matching command handler.

    Returns the command's exit code rather than calling ``sys.exit`` itself,
    so tests can invoke it directly. ``hook`` invocations are routed around
    normal parsing: argparse exits 2 on bad arguments, which Claude Code
    treats as a blocking hook error, so for hooks even parsing must run
    inside the never-fail shield.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["hook"]:
        return _hook_main(argv)
    args = _build_parser().parse_args(argv)
    return args.func(args)


def _add_cwd(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cwd", default=None, help="project directory (default: the current directory)")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree; each subparser binds its handler via ``func``."""
    # suggest_on_error / color shipped in Python 3.14; older interpreters
    # (requires-python is 3.11) get the plain parser.
    extra: dict[str, bool] = {"suggest_on_error": True, "color": True} if sys.version_info >= (3, 14) else {}
    parser = argparse.ArgumentParser(
        prog="ostinote",
        description="Continuous memory for coding agents (Claude Code, Codex).",
        **extra,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_hook = sub.add_parser("hook", help="lifecycle hook entry points")
    p_hook.add_argument("event", choices=["session-start", "post-tool", "session-end"])
    p_hook.add_argument("--agent", required=True, choices=agent_names())
    p_hook.set_defaults(func=_cmd_hook)

    p_save = sub.add_parser("save", help="save a session into memory")
    p_save.add_argument("--agent", default="claude", choices=agent_names())
    p_save.add_argument("--session", default=None)
    p_save.add_argument("--transcript", default=None)
    _add_cwd(p_save)
    mode = p_save.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="bypass cooldown and min-message threshold")
    mode.add_argument(
        "--final",
        action="store_true",
        help="end-of-session save: bypass cooldown, keep min-message threshold",
    )
    p_save.add_argument("--dry", action="store_true", help="print the extract, skip the model call")
    p_save.set_defaults(func=_cmd_save)

    p_cons = sub.add_parser("consolidate", help="compress past days into recent/archive")
    _add_cwd(p_cons)
    p_cons.set_defaults(func=_cmd_consolidate)

    p_status = sub.add_parser("status", help="show memory state")
    _add_cwd(p_status)
    p_status.add_argument("--costs", action="store_true", help="show per-day token usage and cost instead")
    p_status.set_defaults(func=_cmd_status)

    p_doctor = sub.add_parser("doctor", help="check the whole pipeline, loudly")
    _add_cwd(p_doctor)
    p_doctor.add_argument("--live", action="store_true", help="also run one real (paid) summarizer call")
    p_doctor.set_defaults(func=_cmd_doctor)

    for name in ("install", "uninstall"):
        p = sub.add_parser(name, help="%s hooks for an agent" % name)
        p.add_argument("agent", choices=agent_names() + ["all"])
        scope = p.add_mutually_exclusive_group()
        scope.add_argument(
            "--user",
            dest="scope",
            action="store_const",
            const="user",
            help="register globally (default)",
        )
        scope.add_argument(
            "--project",
            dest="scope",
            action="store_const",
            const="project",
            help="register for this project only",
        )
        p.set_defaults(scope="user", func=_cmd_install)
        _add_cwd(p)

    return parser


def _env(args) -> Env:
    """Build the Env for the command's --cwd, defaulting to the process cwd."""
    return Env(args.cwd or os.getcwd())


def _hook_main(argv: list[str]) -> int:
    """Run a hook invocation; never propagate failures into the agent.

    Anything that goes wrong — handler exceptions, but also argparse errors
    such as a stale binary being fed an event name it doesn't know yet — is
    appended to the hook error log and the process still exits 0.
    """
    try:
        args = _build_parser().parse_args(argv)
        args.func(args)
    except SystemExit as e:
        # argparse raises SystemExit(0) for --help; only log real failures.
        if e.code not in (0, None):
            _log_hook_failure(argv)
    except Exception:
        _log_hook_failure(argv)
    return 0


def _cmd_hook(args) -> int:
    """Dispatch one lifecycle hook event; ``_hook_main`` shields failures."""
    handlers = {
        "session-start": hooks_mod.session_start,
        "post-tool": hooks_mod.post_tool,
        "session-end": hooks_mod.session_end,
    }
    handlers[args.event](args.agent)
    return 0


def _log_hook_failure(argv: list[str]) -> None:
    """Append the active exception to the hook error log, best effort.

    Hooks are silent by design, so this log is their only failure surface.
    The file is created owner-only (0o600) because tracebacks can mention
    config and transcript paths. If even the log write fails, emit one line
    on stderr — agents ignore hook stderr on exit 0 — rather than failing
    silently twice in a row.
    """
    tb = traceback.format_exc()
    lines = tb.splitlines()
    if len(lines) > _TRACEBACK_TAIL_LINES:
        dropped = len(lines) - _TRACEBACK_TAIL_LINES
        tb = "\n".join(["... (%d traceback lines truncated) ..." % dropped, *lines[-_TRACEBACK_TAIL_LINES:]])
    entry = "[%s %s]\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(argv), tb)
    try:
        os.makedirs(os.path.dirname(env_mod.HOOK_ERRORS_PATH), mode=0o700, exist_ok=True)
        fd = os.open(env_mod.HOOK_ERRORS_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError:
        print("ostinote: hook failed and the error log %s is unwritable" % env_mod.HOOK_ERRORS_PATH, file=sys.stderr)


def _cmd_save(args) -> int:
    # Canonicalize the transcript to match the hook ingestion path. The cwd is
    # deliberately left literal: project slugs derive from it, and resolving
    # symlinks would orphan existing memory dirs.
    transcript = os.path.realpath(args.transcript) if args.transcript else None
    return pipeline.run_save(_env(args), args.agent, args.session, transcript, args.force, args.dry, args.final)


def _cmd_consolidate(args) -> int:
    return pipeline.run_consolidation(_env(args))


def _cmd_status(args) -> int:
    env = _env(args)
    _costs(env) if args.costs else _status(env)
    return 0


def _cmd_doctor(args) -> int:
    return doctor_mod.run(_env(args), live=args.live)


def _cmd_install(args) -> int:
    root = os.path.abspath(args.cwd or os.getcwd())
    targets = agent_names() if args.agent == "all" else [args.agent]
    for agent in targets:
        for line in install_mod.install(agent, args.scope, root, remove=args.command == "uninstall"):
            print(line)
    return 0


def _status(env: Env) -> None:
    print("project root : %s" % env.project_root)
    print("data dir     : %s" % env.data_dir)
    print("timezone     : %s" % (env.cfg["timezone"] or "(system local)"))
    print()
    print("memory files:")
    for path in (
        env.identity_file,
        env.core_memories_file,
        env.today_file(),
        env.now_file,
        env.recent_file,
        env.archive_file,
    ):
        if os.path.exists(path):
            print("  %-22s %6d bytes" % (os.path.basename(path), os.path.getsize(path)))
    staging = pipeline.staging_files(env)
    if staging:
        print("  %d past day(s) awaiting consolidation" % len(staging))
    print()
    from .state import all_states

    states = all_states(env.sessions_dir)
    print("tracked sessions: %d" % len(states))
    for state in sorted(states, key=lambda s: s.last_attempt_ts, reverse=True)[:10]:
        age = time.time() - state.last_attempt_ts if state.last_attempt_ts else None
        print(
            "  %-7s %s  line %-6d %s"
            % (
                state.agent,
                state.session_id[:24],
                state.line,
                "saved %dm ago" % (age // 60) if age is not None else "never saved",
            )
        )


def _costs(env: Env) -> None:
    def usd(cost: float) -> str:
        return "$%.4f" % cost if cost else "-"

    days = costs_mod.day_totals(env.logs_dir)
    if not days:
        print("no model calls logged in %s" % env.logs_dir)
        return
    header = ("day", "calls", "input", "cache", "output", "cost")
    print("%-12s %6s %12s %12s %10s %10s" % header)
    total = {"calls": 0, "input": 0, "cache": 0, "output": 0, "cost": 0.0}
    for day, t in days:
        for key in total:
            total[key] += t[key]
        print("%-12s %6d %12d %12d %10d %10s" % (day, t["calls"], t["input"], t["cache"], t["output"], usd(t["cost"])))
    print(
        "%-12s %6d %12d %12d %10d %10s"
        % (
            "total",
            total["calls"],
            total["input"],
            total["cache"],
            total["output"],
            usd(total["cost"]),
        )
    )
    print("\n(cost is summed only from calls whose engine reported it)")


if __name__ == "__main__":
    sys.exit(main())
