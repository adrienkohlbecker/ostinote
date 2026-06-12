import json
import os
import re
import tomllib

import pytest

from ostinote import doctor as doctor_mod
from ostinote import install as install_mod
from ostinote.env import Env

# --- Installer -------------------------------------------------------------------------


def test_install_uninstall_idempotent(tmp_path, installer_env):
    """Install and uninstall Codex project hooks repeatedly.

    Expected: installing twice leaves exactly one managed hook per Codex event,
    and uninstall removes the managed hooks cleanly without leaving stale
    `hooks` entries behind.
    """
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


def test_skill_installed_per_agent_and_scope(tmp_path, installer_env):
    """Install the `$ostinote` or `/ostinote` skill in the right place.

    Expected: project-scope Codex writes under the project `.agents` directory,
    uninstall removes it, and user-scope installs go to Codex and Claude's
    distinct user skill directories.
    """
    home = installer_env

    root = str(tmp_path / "proj")
    code, report = install_mod.install("codex", "project", root)
    skill = tmp_path / "proj" / ".agents" / "skills" / "ostinote" / "SKILL.md"
    assert code == 0
    assert "codex $ostinote skill installed" in "\n".join(report)
    assert skill.read_text().startswith("---")

    install_mod.install("codex", "project", root, remove=True)
    assert not skill.exists()

    install_mod.install("codex", "user", root)
    assert (home / ".agents" / "skills" / "ostinote" / "SKILL.md").exists()
    install_mod.install("claude", "user", root)
    assert (home / ".claude" / "skills" / "ostinote" / "SKILL.md").exists()


def test_install_session_end_events_per_agent(tmp_path, installer_env):
    """Register the correct final-save hook event for each agent.

    Expected: Claude gets `SessionEnd`, while Codex gets `Stop` mapped to the
    same `session-end` handler because Codex has no true session-exit hook.
    """
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


def test_install_preserves_foreign_hooks(tmp_path, installer_env):
    """Keep user-defined hooks when adding Ostinote's managed hooks.

    Expected: an existing `./lint.sh` PostToolUse hook remains present, and the
    Codex Ostinote hook is added alongside it.
    """
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


def test_codex_install_adds_memory_dir_to_writable_roots(tmp_path, installer_env):
    """Bootstrap Codex sandbox access to the project's external memory dir.

    Expected: project install adds the computed `~/.ostinote/projects/...` path
    exactly once, preserves existing writable roots and network settings, and
    does not remove the root during hook uninstall.
    """
    home = installer_env
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


def test_codex_install_preserves_config_comments(tmp_path, installer_env):
    """Splice the writable root in without rewriting the user's Codex config.

    Expected: a hand-formatted config.toml keeps its comments and other keys
    after install, and the memory root is added to the existing array (proving
    the targeted text edit ran, not the formatting-losing reserialize).
    """
    home = installer_env
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        '# my codex config\nmodel = "gpt-5"  # keep this comment\n\n'
        '[sandbox_workspace_write]\nwritable_roots = ["~/already"]\nnetwork_access = true\n',
        encoding="utf-8",
    )

    install_mod.install("codex", "project", root)

    text = config.read_text(encoding="utf-8")
    assert "# my codex config" in text
    assert "# keep this comment" in text
    data = tomllib.loads(text)
    roots = data["sandbox_workspace_write"]["writable_roots"]
    assert "~/already" in roots and len(roots) == 2
    assert data["sandbox_workspace_write"]["network_access"] is True
    assert data["model"] == "gpt-5"


def test_codex_install_appends_section_when_absent(tmp_path, installer_env):
    """Append a sandbox table to a config that lacks one, keeping prior keys.

    Expected: a config.toml with no `[sandbox_workspace_write]` table gains one
    with the memory root while its existing content is preserved verbatim.
    """
    home = installer_env
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('# header comment\nmodel = "gpt-5"\n', encoding="utf-8")

    install_mod.install("codex", "project", root)

    text = config.read_text(encoding="utf-8")
    assert "# header comment" in text
    data = tomllib.loads(text)
    assert data["model"] == "gpt-5"
    assert len(data["sandbox_workspace_write"]["writable_roots"]) == 1


@pytest.mark.skipif(os.name == "nt", reason="double quotes are not legal in Windows file names")
def test_codex_install_escapes_hostile_project_path(tmp_path, installer_env):
    """Round-trip the writable root for a TOML-hostile project path.

    Expected: an in-repo data_dir under a project directory whose name contains
    TOML string metacharacters (quote, bracket) lands as exactly one correctly
    escaped writable root with other sandbox keys untouched — a regression to
    naive string interpolation would let a crafted directory name inject
    arbitrary keys (e.g. `network_access`) into the trust-sensitive config.
    """
    home = installer_env
    root_dir = tmp_path / 'we"ird ]dir'
    (root_dir / ".ostinote").mkdir(parents=True)
    (root_dir / ".ostinote" / "config.json").write_text(
        json.dumps({"data_dir": ".ostinote", "share_worktrees": False}), encoding="utf-8"
    )
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        '[sandbox_workspace_write]\nwritable_roots = ["~/already"]\nnetwork_access = true\n',
        encoding="utf-8",
    )

    code, _report = install_mod.install("codex", "project", str(root_dir))

    assert code == 0
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    sandbox = data["sandbox_workspace_write"]
    assert sandbox["writable_roots"] == [os.path.realpath(str(root_dir / ".ostinote")), "~/already"]
    assert sandbox["network_access"] is True


def test_codex_install_refuses_invalid_toml(tmp_path, installer_env):
    """Fail closed when Codex config TOML cannot be parsed.

    Expected: install reports an error about writable-root update failure and
    leaves the invalid `config.toml` bytes unchanged for the user to repair.
    """
    home = installer_env
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    config = home / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("[sandbox_workspace_write\n", encoding="utf-8")

    code, report = install_mod.install("codex", "project", root)

    assert code == 1
    assert any(line.startswith("ERROR: could not update Codex writable roots") for line in report)
    assert config.read_text(encoding="utf-8") == "[sandbox_workspace_write\n"


def test_install_refuses_invalid_hook_json(tmp_path, installer_env):
    """Fail closed when an existing hook config file is invalid JSON.

    Expected: install reports the JSON error, leaves the broken file unchanged,
    and does not install the Ostinote skill after hook registration failed.
    """
    root = str(tmp_path)
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text("{", encoding="utf-8")

    code, report = install_mod.install("codex", "project", root)

    assert code == 1
    assert report[0].startswith("ERROR: invalid JSON")
    assert hooks_file.read_text(encoding="utf-8") == "{"
    assert not (tmp_path / ".agents" / "skills" / "ostinote" / "SKILL.md").exists()


def test_install_preserves_similar_and_nonconforming_hooks(tmp_path, installer_env):
    """Only replace hooks that are truly managed Ostinote commands.

    Expected: a lookalike `echo ostinote --agent codex` command and odd-shaped
    hook groups survive, the old managed Ostinote command is replaced, and
    non-ASCII hook text remains readable JSON.
    """
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


def test_uninstall_clean_project_does_not_create_hook_files(tmp_path, installer_env):
    """Uninstall should be a no-op for a project with no installed hooks.

    Expected: removing Codex project hooks from a clean checkout returns an
    empty report and does not create `.codex` or `.agents` directories.
    """
    root = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()

    assert install_mod.install("codex", "project", root, remove=True) == (0, [])
    assert not (tmp_path / "proj" / ".codex").exists()
    assert not (tmp_path / "proj" / ".agents").exists()


def test_registered_events_tolerates_malformed_hooks(monkeypatch):
    """Scan hook settings for managed events without crashing on bad shapes.

    Expected: a hooks file whose structures are malformed (non-dict `hooks`,
    list-typed group, string group, non-list hook array, non-dict hook) is
    skipped rather than raising, while a well-formed managed command is still
    recognized — so `doctor` can diagnose a hand-broken config instead of
    dying on it.
    """
    monkeypatch.setattr(install_mod, "_is_ours", lambda command: command == "OURS")
    assert install_mod.registered_events({"hooks": "not-a-dict"}) == set()
    settings = {
        "hooks": {
            "Broken": "not-a-list",
            "Mixed": [
                "string-group",
                {"hooks": "not-a-list"},
                {"hooks": ["not-a-dict", {"command": 123}, {"command": "OURS"}]},
            ],
        }
    }
    assert install_mod.registered_events(settings) == {"Mixed"}


def test_doctor_smoke(tmp_path, monkeypatch, capsys, installer_env):
    """Smoke-test `doctor` against a project with both agents installed.

    Expected: a fully registered project passes with no FAIL lines; the codex
    install records the memory dir as a writable root in the temp home's
    config.toml (never the real one); after deleting Claude's SessionEnd hook,
    doctor returns failure and prints a missing-hook diagnostic.
    """
    home = installer_env
    (home / ".ostinote").mkdir(parents=True)
    (home / ".ostinote" / "config.json").write_text(json.dumps({"data_dir": str(tmp_path / "data")}))
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda _: "/usr/bin/claude")

    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    (proj / ".ostinote" / "config.json").write_text(json.dumps({"share_worktrees": False}))
    install_mod.install("claude", "project", str(proj))
    install_mod.install("codex", "project", str(proj))
    # The sandbox grant must land in the temp home's Codex config (the real
    # ~/.codex/config.toml used to be rewritten here) and name the memory dir.
    codex_cfg = tomllib.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8"))
    assert codex_cfg["sandbox_workspace_write"]["writable_roots"] == [
        os.path.realpath(str(tmp_path / "data")),
    ]

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
