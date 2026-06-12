import io
import json
import os
import re
import subprocess
import sys
import textwrap
import time

from ostinote import config as config_mod
from ostinote import summarize
from ostinote.env import Env


def claude_line(msg_type, content, is_meta=False):
    """One Claude transcript JSONL line with the given type and content."""
    return json.dumps(
        {
            "type": msg_type,
            "isMeta": is_meta,
            "message": {"content": content},
        }
    )


def codex_item(payload):
    """One Codex rollout JSONL line wrapping the given response_item payload."""
    return json.dumps({"type": "response_item", "payload": payload})


def codex_user(text):
    """One Codex rollout line carrying a user message with the given text."""
    return codex_item(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
    )


def codex_assistant(text):
    """One Codex rollout line carrying an assistant message with the given text."""
    return codex_item(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
    )


def hook_stdin(monkeypatch, payload):
    """Feed a hook payload on stdin, matching how agents invoke hook handlers.

    Accepts a dict (serialized to JSON) or a raw string so malformed-input
    tests can feed garbage through the same path.
    """
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def base_test_config(recovery=True):
    """The shared project config that neutralizes save gates for tests.

    Cooldowns and thresholds are zeroed so pipeline tests exercise behavior,
    not timing; gating tests override the specific gate they probe.
    """
    return {
        "share_worktrees": False,
        "cooldowns": {"save_seconds": 0, "compress_seconds": 0},
        "thresholds": {"min_human_messages": 1, "delta_lines_trigger": 1},
        "features": {"hourly_compression": False, "consolidation": True, "recovery": recovery},
    }


def project_env(tmp_path, monkeypatch, extra_cfg=None):
    # data_dir is a guarded key: a project-layer value outside the repo /
    # ~/.ostinote is rejected as an untrusted redirect, so tests supply it via
    # the trusted user layer (the monkeypatched USER_CONFIG_PATH).
    user_cfg = tmp_path / "user-config.json"
    user_cfg.write_text(json.dumps({"data_dir": str(tmp_path / "data")}), encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(user_cfg))
    proj = tmp_path / "proj"
    (proj / ".ostinote").mkdir(parents=True)
    cfg = base_test_config()
    if extra_cfg:
        deep_update(cfg, extra_cfg)
    (proj / ".ostinote" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return Env(str(proj))


def model_result(text):
    """Build a fake summarizer result for pipeline tests.

    Token counts are fixed — no caller asserts on them. Skip detection is
    derived through the SUT's own parser instead of re-implementing its
    heuristic, so the fixture cannot drift from how the pipeline really
    classifies a SKIP response.
    """
    return summarize.ModelResult(
        text=text,
        tokens=summarize.TokenUsage(input=10, output=3, cache=0, cost_usd=0.0),
        is_skip=summarize.parse_response(text).is_skip,
    )


def expected_slug(path):
    """Independent pin of the dashed-path slug scheme (claude-remember style).

    Deliberately re-implemented rather than imported from the SUT, so a slug
    regression cannot self-confirm. Claude Code's transcript directories
    additionally lowercase Windows drive letters before slugging; tests
    pinning that discovery scheme apply the lowering themselves.
    """
    return re.sub(r"[^a-zA-Z0-9]", "-", path)


def age_file(path, seconds):
    """Backdate a file's mtime by ``seconds``.

    Recovery and transcript-discovery tests use this to make a file look
    idle or older than a sibling; naming the operation keeps the window
    arithmetic at the call site, next to the gate it exercises.
    """
    mtime = time.time() - seconds
    os.utime(path, (mtime, mtime))


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


# Stands in for the real summarizer in functional subprocess tests: echoes
# OSTINOTE_FAKE_RESULT as Claude-style JSON and logs the prompt it received.
_FAKE_MODEL_SCRIPT = textwrap.dedent(
    """\
    import json
    import os
    import sys
    prompt = sys.stdin.read()
    prompt_log = os.environ.get('OSTINOTE_FAKE_PROMPTS')
    if prompt_log:
        with open(prompt_log, 'a', encoding='utf-8') as f:
            f.write('===PROMPT===\\n' + prompt + '\\n')
    print(json.dumps({
        'result': os.environ['OSTINOTE_FAKE_RESULT'],
        'usage': {'input_tokens': 11, 'output_tokens': 4, 'cache_read_input_tokens': 2},
        'total_cost_usd': 0.00042,
    }))
    """
)


def functional_cli_project(tmp_path, extra_cfg=None):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    model = tmp_path / "fake_model.py"
    prompt_log = tmp_path / "prompts.log"
    home.mkdir()
    (proj / ".ostinote").mkdir(parents=True)
    model.write_text(_FAKE_MODEL_SCRIPT, encoding="utf-8")
    cfg = base_test_config(recovery=False)
    if extra_cfg:
        deep_update(cfg, extra_cfg)
    (proj / ".ostinote" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    # data_dir and summarizer.command are guarded keys honored only from the
    # trusted user layer; the subprocess resolves "~" to this temp HOME.
    (home / ".ostinote").mkdir()
    (home / ".ostinote" / "config.json").write_text(
        json.dumps(
            {
                "data_dir": str(tmp_path / "data"),
                "summarizer": {"command": [sys.executable, str(model)], "timeout": 5},
            }
        ),
        encoding="utf-8",
    )

    # Build the subprocess env from scratch (allow-list). Copying os.environ
    # would leak the developer's shell secrets into the SUT, and a deny-list
    # rots: any variable the SUT later starts honoring would silently
    # re-point functional tests at real state.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": repo_root,
        "OSTINOTE_FAKE_PROMPTS": str(prompt_log),
    }
    # Platform essentials: Windows needs the system dirs to start processes,
    # and locale vars keep Python's text I/O matching the developer's setup.
    for key in ("SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "TEMP", "TMP", "LANG", "LC_ALL"):
        if key in os.environ:
            env[key] = os.environ[key]
    return proj, tmp_path / "data", env, prompt_log


def run_cli(args, cwd, env, **kwargs):
    # A regression that blocks on stdin must fail the test, not hang CI.
    kwargs.setdefault("timeout", 60)
    return subprocess.run(
        [sys.executable, "-m", "ostinote", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        **kwargs,
    )
