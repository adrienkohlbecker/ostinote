import json
import os
import subprocess
import sys

from ostinote import config as config_mod
from ostinote.env import Env


def codex_item(payload):
    return json.dumps({"type": "response_item", "payload": payload})


def project_env(tmp_path, monkeypatch, extra_cfg=None):
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(tmp_path / "no-user-config.json"))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    cfg = {
        "data_dir": str(tmp_path / "data"),
        "share_worktrees": False,
        "cooldowns": {"save_seconds": 0, "compress_seconds": 0},
        "thresholds": {"min_human_messages": 1, "delta_lines_trigger": 1},
        "features": {"hourly_compression": False, "consolidation": True, "recovery": True},
    }
    if extra_cfg:
        for key, value in extra_cfg.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    (proj / ".ostinote" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return Env(str(proj))


def model_result(text, input_tokens=10, output_tokens=3, cache_tokens=0, cost=0.0):
    from ostinote.summarize import ModelResult, TokenUsage

    return ModelResult(
        text=text,
        tokens=TokenUsage(input=input_tokens, output=output_tokens, cache=cache_tokens, cost_usd=cost),
        is_skip=text.strip().upper().startswith("SKIP"),
    )


def installer_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows uses USERPROFILE, not HOME
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(home / ".ostinote/config.json"))
    return home


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def functional_cli_project(tmp_path, extra_cfg=None):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    model = tmp_path / "fake_model.py"
    prompt_log = tmp_path / "prompts.log"
    home.mkdir()
    (proj / ".ostinote").mkdir(parents=True)
    model.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import sys",
                "prompt = sys.stdin.read()",
                "prompt_log = os.environ.get('OSTINOTE_FAKE_PROMPTS')",
                "if prompt_log:",
                "    with open(prompt_log, 'a', encoding='utf-8') as f:",
                "        f.write('===PROMPT===\\n' + prompt + '\\n')",
                "print(json.dumps({",
                "    'result': os.environ['OSTINOTE_FAKE_RESULT'],",
                "    'usage': {'input_tokens': 11, 'output_tokens': 4, 'cache_read_input_tokens': 2},",
                "    'total_cost_usd': 0.00042,",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    cfg = {
        "data_dir": str(tmp_path / "data"),
        "share_worktrees": False,
        "cooldowns": {"save_seconds": 0, "compress_seconds": 0},
        "thresholds": {"min_human_messages": 1, "delta_lines_trigger": 1},
        "features": {"hourly_compression": False, "consolidation": True, "recovery": False},
        "summarizer": {"command": [sys.executable, str(model)], "timeout": 5},
    }
    if extra_cfg:
        deep_update(cfg, extra_cfg)
    (proj / ".ostinote" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OSTINOTE_FAKE_PROMPTS"] = str(prompt_log)
    return proj, tmp_path / "data", env, prompt_log


def run_cli(args, cwd, env, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "ostinote", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        **kwargs,
    )
