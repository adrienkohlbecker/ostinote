import json
import os
import shlex
import subprocess
import sys

import pytest

from tests.helpers import codex_assistant, codex_user, functional_cli_project, run_cli

# End-to-end subprocess tests of the real CLI; deselect with -m "not functional"
# for a fast inner loop while iterating on parsers or hooks.
pytestmark = pytest.mark.functional


def test_cli_save_functional_with_fake_summarizer(tmp_path):
    """Run `python -m ostinote save` end-to-end with a fake summarizer process.

    Expected: the real CLI writes `now.md`, persists session state, sends the
    transcript content to the fake model, and records token usage in daily logs.
    """
    proj, data, env, prompt_log = functional_cli_project(tmp_path)
    # The fake summarizer reads this env var and prints it as Claude-style JSON.
    env["OSTINOTE_FAKE_RESULT"] = "## 12:00 | main\nfunctional save"
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        codex_user("Remember functional path") + "\n" + codex_assistant("Done.") + "\n",
        encoding="utf-8",
    )

    result = run_cli(
        ["save", "--agent", "codex", "--session", "functional", "--transcript", str(transcript), "--cwd", str(proj)],
        proj,
        env,
    )

    assert result.returncode == 0, result.stderr
    assert (data / "now.md").read_text(encoding="utf-8") == "\n## 12:00 | main\nfunctional save\n"
    state = json.loads((data / "state" / "sessions" / "codex--functional.json").read_text(encoding="utf-8"))
    assert state["line"] == 2
    assert state["transcript_path"] == str(transcript)
    assert "Remember functional path" in prompt_log.read_text(encoding="utf-8")
    log_text = "\n".join(path.read_text(encoding="utf-8") for path in (data / "logs").glob("memory-*.log"))
    assert "tokens: 11+2cache" in log_text


def test_cli_consolidate_functional_with_fake_summarizer(tmp_path):
    """Run `python -m ostinote consolidate` through the real subprocess CLI.

    Expected: staging daily memory is included in the fake model prompt, recent
    and archive files are replaced, core memory is appended, and the staging
    file is renamed to `.done.md`.
    """
    proj, data, env, prompt_log = functional_cli_project(tmp_path)
    # This response exercises all three consolidation outputs in one subprocess
    # run: recent, archive, and newly promoted core memory.
    env["OSTINOTE_FAKE_RESULT"] = (
        "===RECENT===\n# Recent\n\nfunctional recent\n"
        "===ARCHIVE===\n# Archive\n\nfunctional archive\n"
        "===CORE===\n- 2026-06-12: functional core"
    )
    data.mkdir()
    staging = data / "today-2000-01-01.md"
    staging.write_text("old daily memory", encoding="utf-8")
    (data / "recent.md").write_text("# Recent\n\nold", encoding="utf-8")
    (data / "archive.md").write_text("# Archive\n\nold", encoding="utf-8")

    result = run_cli(["consolidate", "--cwd", str(proj)], proj, env)

    assert result.returncode == 0, result.stderr
    assert (data / "recent.md").read_text(encoding="utf-8") == "# Recent\n\nfunctional recent\n"
    assert (data / "archive.md").read_text(encoding="utf-8") == "# Archive\n\nfunctional archive\n"
    assert "- 2026-06-12: functional core" in (data / "core-memories.md").read_text(encoding="utf-8")
    assert not staging.exists()
    assert (data / "today-2000-01-01.done.md").exists()
    assert "old daily memory" in prompt_log.read_text(encoding="utf-8")


def test_cli_status_functional_reports_memory_files(tmp_path):
    """Run the real `status` command against a temp project.

    Expected: stdout includes the resolved project root, shows existing memory
    files such as `recent.md`, and reports zero tracked sessions.
    """
    proj, data, env, _prompt_log = functional_cli_project(tmp_path)
    data.mkdir()
    (data / "recent.md").write_text("# Recent\n\nfunctional status", encoding="utf-8")

    result = run_cli(["status", "--cwd", str(proj)], proj, env)

    assert result.returncode == 0, result.stderr
    assert "project root : %s" % proj in result.stdout
    assert "recent.md" in result.stdout
    assert "tracked sessions: 0" in result.stdout


def test_cli_status_costs_functional_reports_day_totals(tmp_path):
    """Run `status --costs` against seeded daily logs.

    Expected: a fresh project reports that no model calls are logged; with a
    seeded daily log, the day row aggregates calls/tokens and only the cost
    the engine actually reported, and the totals row matches.
    """
    proj, data, env, _prompt_log = functional_cli_project(tmp_path)

    empty = run_cli(["status", "--costs", "--cwd", str(proj)], proj, env)
    assert empty.returncode == 0, empty.stderr
    assert "no model calls logged" in empty.stdout

    logs = data / "logs"
    logs.mkdir(parents=True)
    (logs / "memory-2026-06-09.log").write_text(
        "12:00:00 [save] tokens: 100+50cache→20out ($0.000123)\n12:30:00 [compress] tokens: 200+0cache→40out\n",
        encoding="utf-8",
    )

    result = run_cli(["status", "--costs", "--cwd", str(proj)], proj, env)

    assert result.returncode == 0, result.stderr
    rows = {line.split()[0]: line.split() for line in result.stdout.splitlines() if line.strip()}
    assert rows["2026-06-09"] == ["2026-06-09", "2", "300", "50", "60", "$0.0001"]
    assert rows["total"] == ["total", "2", "300", "50", "60", "$0.0001"]


@pytest.mark.skipif(os.name == "nt", reason="drives the hook through a #!/bin/sh wrapper and POSIX shlex quoting")
def test_cli_install_codex_hook_functional_injects_memory(tmp_path):
    """Install Codex project hooks, then execute the generated hook command.

    Expected: `install codex --project` writes hook config and the skill file,
    and running the generated `SessionStart` command emits Codex JSON containing
    the temp project's recent memory.
    """
    proj, data, env, _prompt_log = functional_cli_project(tmp_path)
    data.mkdir()
    (data / "recent.md").write_text("# Recent\n\nfunctional injected memory", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "ostinote"
    # `install` records `self_command()` in hook JSON. The temp PATH makes that
    # command stable and executable without installing the package globally.
    wrapper.write_text('#!/bin/sh\nexec %s -m ostinote "$@"\n' % shlex.quote(sys.executable), encoding="utf-8")
    wrapper.chmod(0o755)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]

    install = run_cli(["install", "codex", "--project", "--cwd", str(proj)], proj, env)
    assert install.returncode == 0, install.stderr
    hooks = json.loads((proj / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    command = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]

    hook = subprocess.run(
        shlex.split(command),
        input=json.dumps({"cwd": str(proj), "source": "startup"}),
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert hook.returncode == 0, hook.stderr
    output = json.loads(hook.stdout)
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "functional injected memory" in context
    assert (proj / ".agents" / "skills" / "ostinote" / "SKILL.md").exists()
