import json
import os

from ostinote import config as config_mod
from ostinote.env import Env
from tests.helpers import expected_slug, project_with_config


def test_config_project_overrides(tmp_path, monkeypatch):
    """Merge default, user, and project config layers in the right order.

    Expected: project values override user values, user values still apply when
    project config omits them, and nested defaults survive partial overrides.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(json.dumps({"cooldowns": {"compress_seconds": 7200}, "timezone": "UTC"}))
    proj = project_with_config(tmp_path, {"cooldowns": {"compress_seconds": 9000}, "timezone": "Europe/Paris"})
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["compress_seconds"] == 9000
    assert cfg["cooldowns"]["save_seconds"] == 120  # default survives
    assert cfg["timezone"] == "Europe/Paris"  # project layer overrides user


def test_data_dir_slug_placeholder(tmp_path, monkeypatch):
    """Expand `{slug}` in `data_dir` using the project-root slug scheme.

    Expected: the resolved data directory uses the configured store directory
    plus a sanitized project path, matching the Claude/claude-remember style
    slug including the leading dash on Unix paths.
    """
    proj = project_with_config(tmp_path, {"share_worktrees": False})
    store = tmp_path / "store"
    # data_dir is honored only from the trusted user layer; a project-layer
    # value pointing outside the repo would be rejected as an untrusted redirect.
    user_cfg = tmp_path / "user-config.json"
    user_cfg.write_text(json.dumps({"data_dir": str(store / "{slug}")}))
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(user_cfg))
    env = Env(str(proj))

    # claude-remember / Claude Code slug scheme: leading dash kept.
    # Windows paths start with drive letters instead.
    slug = expected_slug(str(proj))
    if os.name != "nt":
        assert slug.startswith("-")
    assert env.data_dir == str(store / slug)


def test_config_legacy_remember_keys(tmp_path, monkeypatch):
    """Normalize old claude-remember config keys to Ostinote's current names.

    Expected: legacy `ndc_*` keys populate the new compression settings, legacy
    keys disappear from the loaded config, and an explicit new key wins if both
    old and new names are present.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(
        json.dumps(
            {
                "cooldowns": {"ndc_seconds": 1800, "git_backup_seconds": 900},
                "features": {"ndc_compression": False},
            }
        )
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["compress_seconds"] == 1800
    assert "ndc_seconds" not in cfg["cooldowns"]
    assert cfg["features"]["hourly_compression"] is False
    assert "ndc_compression" not in cfg["features"]

    # a file that sets both names keeps the new name's value
    (proj / ".ostinote").mkdir()
    (proj / ".ostinote" / "config.json").write_text(
        json.dumps({"cooldowns": {"ndc_seconds": 60, "compress_seconds": 7200}})
    )
    cfg = config_mod.load(str(proj))
    assert cfg["cooldowns"]["compress_seconds"] == 7200


def test_project_config_cannot_set_summarizer_command(tmp_path, monkeypatch):
    """Ignore an untrusted project config that tries to set the summarizer.

    Expected: a cloned repo's `.ostinote/config.json` cannot inject
    `summarizer.command` (which runs as a subprocess on every save); the loaded
    value falls back to the default while a user-layer command is honored.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(json.dumps({"summarizer": {"command": ["trusted-engine"]}}))
    proj = project_with_config(tmp_path, {"summarizer": {"command": ["rm", "-rf", "/"], "timeout": 1}})

    cfg, guarded = config_mod.load_trusted(str(proj))
    # The malicious project command is dropped; the trusted user command wins,
    # and a sibling key the project set (timeout) still merges normally.
    assert cfg["summarizer"]["command"] == ["trusted-engine"]
    assert cfg["summarizer"]["timeout"] == 1
    assert ("data_dir",) not in guarded


def test_project_config_cannot_set_summarizer_engine(tmp_path, monkeypatch):
    """Ignore an untrusted project config that tries to pick the engine.

    Expected: a cloned repo's `.ostinote/config.json` cannot flip
    `summarizer.engine` (which decides where transcript content is sent), while
    the trusted user layer may set it freely.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(json.dumps({"summarizer": {"engine": "codex"}}))
    proj = project_with_config(tmp_path, {"summarizer": {"engine": "claude"}})

    report = config_mod.load_report(str(proj))
    assert report.config["summarizer"]["engine"] == "codex"
    assert any(issue.key == "summarizer.engine" and issue.level == "warn" for issue in report.issues)


def test_config_rejects_unknown_summarizer_engine(tmp_path, monkeypatch):
    """Fall back to auto-detection when engine names an unknown value.

    Expected: a typo like "gemini" is dropped with a FAIL issue instead of
    reaching resolve_command, where it would silently behave like auto.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text(json.dumps({"summarizer": {"engine": "gemini"}}))
    proj = tmp_path / "proj"
    proj.mkdir()

    report = config_mod.load_report(str(proj))
    assert report.config["summarizer"]["engine"] == ""
    assert any(issue.key == "summarizer.engine" and issue.level == "FAIL" for issue in report.issues)


def test_config_invalid_shapes_and_project_cost_controls_fall_back(tmp_path, monkeypatch):
    """Ignore config values that would crash hooks or amplify model calls.

    Expected: wrong-typed values fall back to defaults, and an untrusted project
    cannot lower cooldowns/thresholds or raise summarizer timeout beyond the
    safe defaults.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    proj = project_with_config(
        tmp_path,
        {
            "data_dir": 123,
            "features": False,
            "cooldowns": {"save_seconds": 0, "compress_seconds": 0},
            "thresholds": {"min_human_messages": 0, "delta_lines_trigger": 0},
            "summarizer": {"timeout": 9999},
            "time_format": "metric",
        },
    )

    report = config_mod.load_report(str(proj))

    assert report.config["data_dir"] == config_mod.DEFAULTS["data_dir"]
    assert report.config["features"] == config_mod.DEFAULTS["features"]
    assert report.config["cooldowns"] == config_mod.DEFAULTS["cooldowns"]
    assert report.config["thresholds"] == config_mod.DEFAULTS["thresholds"]
    assert report.config["summarizer"]["timeout"] == config_mod.DEFAULTS["summarizer"]["timeout"]
    assert report.config["time_format"] == config_mod.DEFAULTS["time_format"]
    assert {issue.key for issue in report.issues} >= {
        "data_dir",
        "features",
        "cooldowns.save_seconds",
        "thresholds.min_human_messages",
        "summarizer.timeout",
        "time_format",
    }


def test_project_config_data_dir_is_flagged_guarded(tmp_path, monkeypatch):
    """Report when the project layer sets the guarded `data_dir` key.

    Expected: `load_trusted` keeps the value (the in-repo `.ostinote` workflow
    needs it) but flags it so `Env` can containment-check it; an unset key is
    not flagged.
    """
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    proj = project_with_config(tmp_path, {"data_dir": ".ostinote"})

    cfg, guarded = config_mod.load_trusted(str(proj))
    assert cfg["data_dir"] == ".ostinote"
    assert ("data_dir",) in guarded
