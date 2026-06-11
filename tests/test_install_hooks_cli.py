import json
import tomllib

import pytest

from ostinote import config as config_mod
from ostinote.state import SessionState
from tests.helpers import installer_home, project_env

# --- Installer -------------------------------------------------------------------------


def test_install_uninstall_idempotent(tmp_path, monkeypatch):
    """Install and uninstall Codex project hooks repeatedly.

    Expected: installing twice leaves exactly one managed hook per Codex event,
    and uninstall removes the managed hooks cleanly without leaving stale
    `hooks` entries behind.
    """
    from ostinote import install as install_mod

    installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)

    install_mod.install("codex", "project", root)
    # The second install is the regression check: managed hooks should be
    # replaced in-place instead of duplicated.
    install_mod.install("codex", "project", root)  # idempotent
    hooks_file = tmp_path / ".codex" / "hooks.json"
    data = json.loads(hooks_file.read_text())
    for event in ("SessionStart", "PostToolUse", "Stop"):
        ours = [h for g in data["hooks"][event] for h in g["hooks"]]
        assert len(ours) == 1
        assert "--agent codex" in ours[0]["command"]
    assert set(data["hooks"]) == {"SessionStart", "PostToolUse", "Stop"}

    install_mod.install("codex", "project", root, remove=True)
    data = json.loads(hooks_file.read_text())
    assert data.get("hooks", {}) == {}


def test_skill_installed_per_agent_and_scope(tmp_path, monkeypatch):
    """Install the `$ostinote` or `/ostinote` skill in the right place.

    Expected: project-scope Codex writes under the project `.agents` directory,
    uninstall removes it, and user-scope installs go to Codex and Claude's
    distinct user skill directories.
    """
    from ostinote import install as install_mod

    home = installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])

    root = str(tmp_path / "proj")
    report = install_mod.install("codex", "project", root)
    skill = tmp_path / "proj" / ".agents" / "skills" / "ostinote" / "SKILL.md"
    assert "codex $ostinote skill installed" in "\n".join(report)
    assert skill.read_text().startswith("---")

    install_mod.install("codex", "project", root, remove=True)
    assert not skill.exists()

    install_mod.install("codex", "user", root)
    assert (home / ".agents" / "skills" / "ostinote" / "SKILL.md").exists()
    install_mod.install("claude", "user", root)
    assert (home / ".claude" / "skills" / "ostinote" / "SKILL.md").exists()


def test_install_session_end_events_per_agent(tmp_path, monkeypatch):
    """Register the correct final-save hook event for each agent.

    Expected: Claude gets `SessionEnd`, while Codex gets `Stop` mapped to the
    same `session-end` handler because Codex has no true session-exit hook.
    """
    from ostinote import install as install_mod

    installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)
    install_mod.install("claude", "project", root)
    claude = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert set(claude["hooks"]) == {"SessionStart", "PostToolUse", "SessionEnd"}

    # Codex has no SessionEnd; its turn-scoped Stop maps to the same handler.
    install_mod.install("codex", "project", root)
    codex = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    assert set(codex["hooks"]) == {"SessionStart", "PostToolUse", "Stop"}
    stop = [h["command"] for g in codex["hooks"]["Stop"] for h in g["hooks"]]
    assert "hook session-end --agent codex" in stop[0]


@pytest.mark.parametrize(
    "source,injected",
    [("startup", True), ("clear", True), ("", True), ("resume", False), ("compact", False)],
)
def test_session_start_source_filter(tmp_path, monkeypatch, capsys, source, injected):
    """Inject memory only for fresh starts, not resumes or compactions.

    Expected: startup, clear, and missing-source hook payloads print memory
    context; resume and compact payloads print nothing to avoid duplicated
    context in an already-running conversation.
    """
    import io

    from ostinote import hooks as hooks_mod

    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user.json"))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps(
            {
                "data_dir": str(tmp_path / "data"),
                "share_worktrees": False,
                "features": {"recovery": False, "consolidation": False},
            }
        )
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "recent.md").write_text("# Recent\n\nsomething happened\n")

    payload = {"cwd": str(proj)}
    if source:
        payload["source"] = source
    # Hook handlers read their payload from stdin, matching how agents invoke
    # them in real sessions.
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    hooks_mod.session_start("claude")
    out = capsys.readouterr().out
    if injected:
        assert "=== MEMORY ===" in out
        assert "something happened" in out
    else:
        assert out == ""


def test_post_tool_registers_session_and_queues_save(tmp_path, monkeypatch):
    """Queue a background save after enough new transcript lines appear.

    Expected: `post_tool` records the transcript path for recovery and queues
    the exact `save --agent codex ...` command instead of running the heavy save
    inline inside the hook.
    """
    import io

    from ostinote import hooks as hooks_mod

    env = project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n{}\n{}\n", encoding="utf-8")
    queued = []
    # Hooks should enqueue background work, not run summarization synchronously.
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"cwd": env.cwd, "transcript_path": str(transcript), "session_id": "s1"})),
    )

    hooks_mod.post_tool("codex")

    state = SessionState.load(env.sessions_dir, "codex", "s1")
    assert state.transcript_path == str(transcript)
    assert queued == [
        [
            "save",
            "--agent",
            "codex",
            "--session",
            "s1",
            "--transcript",
            str(transcript),
            "--cwd",
            env.cwd,
        ]
    ]


def test_session_end_queues_final_save_from_transcript_basename(tmp_path, monkeypatch):
    """Queue a final save when the hook payload omits `session_id`.

    Expected: `session_end` derives the session id from the transcript filename
    and queues a `save ... --final` command so session close captures remaining
    transcript content.
    """
    import io

    from ostinote import hooks as hooks_mod

    env = project_env(tmp_path, monkeypatch)
    transcript = tmp_path / "session-abc.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    queued = []
    # No session id in stdin exercises the filename-derived fallback.
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"cwd": env.cwd, "transcript_path": str(transcript)})))

    hooks_mod.session_end("claude")

    assert queued == [
        [
            "save",
            "--agent",
            "claude",
            "--session",
            "session-abc",
            "--transcript",
            str(transcript),
            "--cwd",
            env.cwd,
            "--final",
        ]
    ]


def test_session_start_queues_consolidation_without_injecting_on_resume(tmp_path, monkeypatch, capsys):
    """Start consolidation on resume without re-injecting memory context.

    Expected: a past `today-*.md` file queues `ostinote consolidate`, but because
    the source is `resume`, stdout stays empty.
    """
    import io

    from ostinote import hooks as hooks_mod

    env = project_env(tmp_path, monkeypatch, {"features": {"recovery": False}})
    env.ensure_dirs()
    (tmp_path / "data" / "today-2000-01-01.md").write_text("old", encoding="utf-8")
    queued = []
    # Capture the consolidation request while still letting `session_start`
    # decide whether memory should be printed for this source.
    monkeypatch.setattr(hooks_mod, "spawn", lambda _env, args: queued.append(args))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"cwd": env.cwd, "source": "resume"})))

    hooks_mod.session_start("codex")

    assert queued == [["consolidate", "--cwd", env.cwd]]
    assert capsys.readouterr().out == ""


# --- CLI ------------------------------------------------------------------------------


def test_cli_dispatches_save_and_consolidate(tmp_path, monkeypatch):
    """Check argparse wiring for the `save` and `consolidate` commands.

    Expected: CLI arguments are passed to the correct pipeline functions, and
    `main()` exits with the pipeline return codes instead of swallowing them.
    """
    from ostinote import cli as cli_mod

    calls = []
    monkeypatch.setattr(
        cli_mod.pipeline,
        "run_save",
        lambda env, agent, session, transcript, force, dry, final: (
            calls.append(("save", env.cwd, agent, session, transcript, force, dry, final)) or 7
        ),
    )
    with pytest.raises(SystemExit) as save_exit:
        # `main()` exits instead of returning for these commands, so the test
        # asserts through pytest's SystemExit capture.
        cli_mod.main(
            [
                "save",
                "--agent",
                "codex",
                "--session",
                "s1",
                "--transcript",
                "t.jsonl",
                "--cwd",
                str(tmp_path),
                "--force",
                "--dry",
            ]
        )
    assert save_exit.value.code == 7
    assert calls == [("save", str(tmp_path), "codex", "s1", "t.jsonl", True, True, False)]

    monkeypatch.setattr(
        cli_mod.pipeline,
        "run_consolidation",
        lambda env: calls.append(("consolidate", env.cwd)) or 3,
    )
    with pytest.raises(SystemExit) as consolidate_exit:
        cli_mod.main(["consolidate", "--cwd", str(tmp_path)])
    assert consolidate_exit.value.code == 3
    assert calls[-1] == ("consolidate", str(tmp_path))


def test_cli_hook_failures_are_logged_and_swallowed(tmp_path, monkeypatch):
    """Make hook entrypoints fail closed from the agent's point of view.

    Expected: if a hook handler raises, `_run_hook` logs the traceback to the
    hook error file and still exits 0 so the agent session is not broken.
    """
    from argparse import Namespace

    from ostinote import cli as cli_mod

    errors = tmp_path / "hook-errors.log"
    monkeypatch.setattr(cli_mod.env_mod, "HOOK_ERRORS_PATH", str(errors))
    # A generator throw is a compact way to make the fake handler raise exactly
    # when `_run_hook` calls it.
    monkeypatch.setattr(cli_mod.hooks_mod, "post_tool", lambda _agent: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(SystemExit) as exc:
        cli_mod._run_hook(Namespace(event="post-tool", agent="codex"))

    assert exc.value.code == 0
    assert "RuntimeError: boom" in errors.read_text(encoding="utf-8")


def test_install_preserves_foreign_hooks(tmp_path, monkeypatch):
    """Keep user-defined hooks when adding Ostinote's managed hooks.

    Expected: an existing `./lint.sh` PostToolUse hook remains present, and the
    Codex Ostinote hook is added alongside it.
    """
    from ostinote import install as install_mod

    installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "./lint.sh"}],
                        }
                    ]
                }
            }
        )
    )
    install_mod.install("codex", "project", root)
    data = json.loads(hooks_file.read_text())
    commands = [h["command"] for g in data["hooks"]["PostToolUse"] for h in g["hooks"]]
    assert "./lint.sh" in commands
    assert any("--agent codex" in c for c in commands)


def test_codex_install_adds_memory_dir_to_writable_roots(tmp_path, monkeypatch):
    """Bootstrap Codex sandbox access to the project's external memory dir.

    Expected: project install adds the computed `~/.ostinote/projects/...` path
    exactly once, preserves existing writable roots and network settings, and
    does not remove the root during hook uninstall.
    """
    import re

    from ostinote import install as install_mod

    home = installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        'model = "gpt-5"\n\n[sandbox_workspace_write]\nwritable_roots = ["~/already"]\nnetwork_access = true\n',
        encoding="utf-8",
    )

    install_mod.install("codex", "project", root)
    install_mod.install("codex", "project", root)
    # Uninstall removes hooks/skills but intentionally leaves sandbox access in
    # config, because other sessions may still need the memory directory.
    install_mod.install("codex", "project", root, remove=True)

    expected = "~/.ostinote/projects/%s" % re.sub(r"[^a-zA-Z0-9]", "-", root)
    text = config.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    sandbox = data["sandbox_workspace_write"]
    assert sandbox["writable_roots"].count(expected) == 1
    assert sandbox["network_access"] is True
    assert "~/already" in sandbox["writable_roots"]


def test_codex_install_refuses_invalid_toml(tmp_path, monkeypatch):
    """Fail closed when Codex config TOML cannot be parsed.

    Expected: install reports an error about writable-root update failure and
    leaves the invalid `config.toml` bytes unchanged for the user to repair.
    """
    from ostinote import install as install_mod

    home = installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("[sandbox_workspace_write\n", encoding="utf-8")

    report = install_mod.install("codex", "project", root)

    assert any(line.startswith("ERROR: could not update Codex writable roots") for line in report)
    assert config.read_text(encoding="utf-8") == "[sandbox_workspace_write\n"


def test_install_refuses_invalid_hook_json(tmp_path, monkeypatch):
    """Fail closed when an existing hook config file is invalid JSON.

    Expected: install reports the JSON error, leaves the broken file unchanged,
    and does not install the Ostinote skill after hook registration failed.
    """
    from ostinote import install as install_mod

    installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text("{", encoding="utf-8")

    report = install_mod.install("codex", "project", root)

    assert report[0].startswith("ERROR: invalid JSON")
    assert hooks_file.read_text(encoding="utf-8") == "{"
    assert not (tmp_path / ".agents" / "skills" / "ostinote" / "SKILL.md").exists()


def test_install_preserves_similar_and_nonconforming_hooks(tmp_path, monkeypatch):
    """Only replace hooks that are truly managed Ostinote commands.

    Expected: a lookalike `echo ostinote --agent codex` command and odd-shaped
    hook groups survive, the old managed Ostinote command is replaced, and
    non-ASCII hook text remains readable JSON.
    """
    from ostinote import install as install_mod

    installer_home(tmp_path, monkeypatch)
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    root = str(tmp_path)
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo ostinote --agent codex",
                                },
                                {
                                    "type": "command",
                                    "command": "/old/ostinote hook post-tool --agent codex",
                                },
                                {"type": "command", "command": "./écho.sh"},
                            ],
                        },
                        {"matcher": "Opaque", "hooks": "leave-me"},
                        "strange-group",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    install_mod.install("codex", "project", root)

    text = hooks_file.read_text(encoding="utf-8")
    data = json.loads(text)
    # The command list is flattened only for well-formed hook groups; the
    # malformed groups below are asserted separately as preserved data.
    groups = data["hooks"]["PostToolUse"]
    commands = [
        hook["command"]
        for group in groups
        if isinstance(group, dict) and isinstance(group.get("hooks"), list)
        for hook in group["hooks"]
    ]
    assert "echo ostinote --agent codex" in commands
    assert "/old/ostinote hook post-tool --agent codex" not in commands
    assert "./écho.sh" in commands
    assert "\\u00e9" not in text
    assert {"matcher": "Opaque", "hooks": "leave-me"} in groups
    assert "strange-group" in groups


def test_uninstall_clean_project_does_not_create_hook_files(tmp_path, monkeypatch):
    """Uninstall should be a no-op for a project with no installed hooks.

    Expected: removing Codex project hooks from a clean checkout returns an
    empty report and does not create `.codex` or `.agents` directories.
    """
    from ostinote import install as install_mod

    installer_home(tmp_path, monkeypatch)
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()

    assert install_mod.install("codex", "project", root, remove=True) == []
    assert not (tmp_path / "proj" / ".codex").exists()
    assert not (tmp_path / "proj" / ".agents").exists()


def test_doctor_smoke(tmp_path, monkeypatch, capsys):
    """Smoke-test `doctor` against a project with both agents installed.

    Expected: a fully registered project passes with no FAIL lines; after
    deleting Claude's SessionEnd hook, doctor returns failure and prints a
    missing-hook diagnostic.
    """
    from ostinote import doctor as doctor_mod
    from ostinote import install as install_mod
    from ostinote.env import Env

    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    monkeypatch.setattr(doctor_mod, "HOOK_ERRORS_PATH", str(tmp_path / "hook-errors.log"))
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])

    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps({"data_dir": str(tmp_path / "data"), "share_worktrees": False})
    )
    install_mod.install("claude", "project", str(proj))
    install_mod.install("codex", "project", str(proj))
    # Keep the user-scope lookup away from the real home directory.
    project_only = install_mod._hooks_file_for
    monkeypatch.setattr(
        doctor_mod,
        "_hooks_file_for",
        lambda agent, scope, root: (
            project_only(agent, scope, root) if scope == "project" else str(tmp_path / "no-user-hooks.json")
        ),
    )

    assert doctor_mod.run(Env(str(proj))) == 0
    out = capsys.readouterr().out
    assert "claude: hooks registered" in out
    assert "codex: hooks registered" in out
    assert "FAIL" not in out

    # A half-registered agent is a FAIL with a fix hint.
    claude_settings = proj / ".claude" / "settings.json"
    data = json.loads(claude_settings.read_text())
    del data["hooks"]["SessionEnd"]
    claude_settings.write_text(json.dumps(data))
    assert doctor_mod.run(Env(str(proj))) == 1
    assert "hooks missing SessionEnd" in capsys.readouterr().out
