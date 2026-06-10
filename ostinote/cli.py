"""``ostinote`` command-line interface.

Subcommands:

    hook session-start|post-tool --agent claude|codex
        Lifecycle hook entry points (read agent JSON on stdin, never fail).
    save [--agent A] [--session ID] [--transcript PATH] [--force] [--dry]
        Extract + summarize one session into now.md.
    consolidate
        Merge past-day staging files into recent.md / archive.md.
    status
        Show resolved paths, memory files, and tracked sessions.
    install|uninstall claude|codex|all [--user|--project]
        (Un)register hooks and the /ostinote command.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from . import hooks as hooks_mod
from . import install as install_mod
from . import pipeline
from .agents import agent_names
from .env import Env


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="ostinote",
        description="Continuous memory for coding agents (Claude Code, Codex).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_hook = sub.add_parser("hook", help="lifecycle hook entry points")
    p_hook.add_argument("event", choices=["session-start", "post-tool"])
    p_hook.add_argument("--agent", required=True, choices=agent_names())

    p_save = sub.add_parser("save", help="save a session into memory")
    p_save.add_argument("--agent", default="claude", choices=agent_names())
    p_save.add_argument("--session", default=None)
    p_save.add_argument("--transcript", default=None)
    p_save.add_argument("--cwd", default=None)
    p_save.add_argument(
        "--force", action="store_true", help="bypass cooldown and min-message threshold"
    )
    p_save.add_argument(
        "--dry", action="store_true", help="print the extract, skip the model call"
    )

    p_cons = sub.add_parser("consolidate", help="compress past days into recent/archive")
    p_cons.add_argument("--cwd", default=None)

    p_status = sub.add_parser("status", help="show memory state")
    p_status.add_argument("--cwd", default=None)

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
        p.set_defaults(scope="user")
        p.add_argument("--cwd", default=None)

    args = parser.parse_args(argv)

    if args.command == "hook":
        _run_hook(args)
    elif args.command == "save":
        env = Env(args.cwd or os.getcwd())
        sys.exit(
            pipeline.run_save(
                env, args.agent, args.session, args.transcript, args.force, args.dry
            )
        )
    elif args.command == "consolidate":
        env = Env(args.cwd or os.getcwd())
        sys.exit(pipeline.run_consolidation(env))
    elif args.command == "status":
        _status(Env(args.cwd or os.getcwd()))
    elif args.command in ("install", "uninstall"):
        root = os.path.abspath(args.cwd or os.getcwd())
        targets = agent_names() if args.agent == "all" else [args.agent]
        for agent in targets:
            for line in install_mod.install(
                agent, args.scope, root, remove=args.command == "uninstall"
            ):
                print(line)


def _run_hook(args) -> None:
    """Run a hook handler; never propagate failures into the agent."""
    handlers = {
        "session-start": hooks_mod.session_start,
        "post-tool": hooks_mod.post_tool,
    }
    try:
        handlers[args.event](args.agent)
    except Exception:
        # Env construction may be what crashed, so don't resolve the data
        # dir — and never create files inside the project.
        try:
            err_path = os.path.expanduser("~/.ostinote/hook-errors.log")
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(
                    "[%s %s --agent %s]\n%s\n"
                    % (
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        args.event,
                        args.agent,
                        traceback.format_exc(),
                    )
                )
        except OSError:
            pass
    sys.exit(0)


def _status(env: Env) -> None:
    print("project root : %s" % env.project_root)
    print("data dir     : %s" % env.data_dir)
    print("timezone     : %s" % (env.cfg["timezone"] or "(system local)"))
    print()
    print("memory files:")
    for path in (
        env.identity_file,
        env.core_memories_file,
        env.handoff_file,
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


if __name__ == "__main__":
    main()
