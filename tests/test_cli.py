import os

import pytest

from ostinote import cli as cli_mod

# --- CLI ------------------------------------------------------------------------------


def test_cli_dispatches_save_and_consolidate(tmp_path, monkeypatch):
    """Check argparse wiring for the `save` and `consolidate` commands.

    Expected: CLI arguments are passed to the correct pipeline functions, the
    transcript path is canonicalized like the hook ingestion path does, and
    `main()` returns the pipeline exit codes instead of swallowing them.
    """
    calls = []
    monkeypatch.setattr(
        cli_mod.pipeline,
        "run_save",
        lambda env, agent, session, transcript, force, dry, final: (
            calls.append(("save", env.cwd, agent, session, transcript, force, dry, final)) or 7
        ),
    )
    transcript = str(tmp_path / "t.jsonl")
    code = cli_mod.main(
        [
            "save",
            "--agent",
            "codex",
            "--session",
            "s1",
            "--transcript",
            transcript,
            "--cwd",
            str(tmp_path),
            "--force",
            "--dry",
        ]
    )
    assert code == 7
    assert calls == [("save", str(tmp_path), "codex", "s1", os.path.realpath(transcript), True, True, False)]

    monkeypatch.setattr(
        cli_mod.pipeline,
        "run_consolidation",
        lambda env: calls.append(("consolidate", env.cwd)) or 3,
    )
    assert cli_mod.main(["consolidate", "--cwd", str(tmp_path)]) == 3
    assert calls[-1] == ("consolidate", str(tmp_path))


def test_cli_save_force_and_final_are_mutually_exclusive(tmp_path):
    """Reject `save --force --final` instead of silently letting --force win.

    Expected: the flags express conflicting gates (--force also bypasses the
    min-message threshold), so argparse refuses the combination with its usual
    exit code 2.
    """
    with pytest.raises(SystemExit) as exc:
        cli_mod.main(["save", "--force", "--final", "--cwd", str(tmp_path)])

    assert exc.value.code == 2


def test_cli_hook_failures_are_logged_and_swallowed(tmp_path, monkeypatch):
    """Make hook entrypoints fail closed from the agent's point of view.

    Expected: if a hook handler raises, `main()` logs the traceback to the
    hook error file — created owner-only, since tracebacks can mention config
    and transcript paths — and still returns 0 so the agent session is not
    broken.
    """
    errors = tmp_path / "hook-errors.log"
    monkeypatch.setattr(cli_mod.env_mod, "HOOK_ERRORS_PATH", str(errors))
    # A generator throw is a compact way to make the fake handler raise exactly
    # when the hook dispatcher calls it.
    monkeypatch.setattr(cli_mod.hooks_mod, "post_tool", lambda _agent: (_ for _ in ()).throw(RuntimeError("boom")))

    assert cli_mod.main(["hook", "post-tool", "--agent", "codex"]) == 0

    assert "RuntimeError: boom" in errors.read_text(encoding="utf-8")
    if os.name == "posix":
        assert errors.stat().st_mode & 0o777 == 0o600


def test_cli_hook_bad_arguments_still_exit_zero(tmp_path, monkeypatch):
    """Shield hook invocations from argparse failures, not just handler ones.

    Expected: argparse exits 2 on unknown arguments, which Claude Code treats
    as a blocking hook error — e.g. a stale binary fed an event name a newer
    install wrote — so for `hook` commands main() logs the failure and returns
    0 instead.
    """
    errors = tmp_path / "hook-errors.log"
    monkeypatch.setattr(cli_mod.env_mod, "HOOK_ERRORS_PATH", str(errors))

    assert cli_mod.main(["hook", "no-such-event", "--agent", "codex"]) == 0

    assert "hook no-such-event --agent codex" in errors.read_text(encoding="utf-8")


def test_cli_install_propagates_failure_exit_code(tmp_path, capsys, installer_env):
    """Make `ostinote install` fail loudly when hook registration fails.

    Expected: a malformed existing hook config makes install() fail closed,
    and main() returns 1 so scripted installs notice instead of relying on
    ERROR text in stdout.
    """
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text("{", encoding="utf-8")

    assert cli_mod.main(["install", "codex", "--project", "--cwd", str(tmp_path)]) == 1
    assert "ERROR" in capsys.readouterr().out
