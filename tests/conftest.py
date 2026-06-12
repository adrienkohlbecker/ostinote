import os
import subprocess
import types

import pytest

from ostinote import config as config_mod
from ostinote import doctor as doctor_mod
from ostinote import env as env_mod
from ostinote import install as install_mod
from ostinote import summarize as summarize_mod
from ostinote.agents import codex as codex_agent_mod
from tests.helpers import project_env


@pytest.fixture(autouse=True)
def isolate_home(tmp_path, monkeypatch):
    """Keep every test away from the developer's real home directory.

    Path resolution in the SUT goes through ``~`` (user config, hook error
    log, agent settings, Codex sandbox config), so one forgotten patch in a
    test would silently read — or write — real dotfiles. This redirects HOME,
    the import-time-resolved paths, and the hook env-var fallbacks to a
    per-test temp home as a suite invariant; tests that need specific
    locations override on top. Returns the temp home path so tests can
    request the fixture by name and build paths under it.
    """
    home = tmp_path / "isolated-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows uses USERPROFILE, not HOME
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", str(home / ".ostinote" / "config.json"))
    errors_log = str(home / ".ostinote" / "hook-errors.log")
    monkeypatch.setattr(env_mod, "HOOK_ERRORS_PATH", errors_log)
    # doctor binds the constant by value at import, so patch its copy too.
    monkeypatch.setattr(doctor_mod, "HOOK_ERRORS_PATH", errors_log)
    # The Codex sessions root is resolved against HOME at import time; without
    # this patch, a test reaching find_latest_transcript would scan the real
    # ~/.codex/sessions and could pull a private transcript into a test.
    monkeypatch.setattr(codex_agent_mod, "SESSIONS_ROOT", str(home / ".codex" / "sessions"))
    # Hooks fall back to CLAUDE_PROJECT_DIR when stdin omits cwd. When this
    # suite runs inside a live agent session, the inherited value points at
    # the real project being developed — scrub it (and any OSTINOTE_* test
    # plumbing) so in-process tests can't resolve the developer's env.
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    for key in [k for k in os.environ if k.startswith("OSTINOTE_")]:
        monkeypatch.delenv(key)
    return home


@pytest.fixture(autouse=True)
def _no_real_summarizer(monkeypatch):
    """Fail loudly if a test reaches the real summarizer subprocess.

    In-process tests stub ``summarize.call_model``; a forgotten stub would
    otherwise shell out to the developer's installed `claude` CLI with
    transcript content — network egress and token spend from a unit test.
    Only summarize's view of the subprocess module is replaced, so git calls
    and functional subprocess tests are unaffected; tests that exercise
    ``call_model`` itself patch ``summarize.subprocess.run`` on this stub.
    """

    def refuse(*_args, **_kwargs):
        raise AssertionError("unmocked summarizer call — stub summarize.call_model or summarize.subprocess.run")

    monkeypatch.setattr(
        summarize_mod,
        "subprocess",
        types.SimpleNamespace(run=refuse, TimeoutExpired=subprocess.TimeoutExpired),
    )


@pytest.fixture
def installer_env(isolate_home, monkeypatch):
    """Temp home for installer tests, with a stable hook command.

    Builds on the autouse home isolation and pins ``install.self_command`` so
    generated hook JSON is deterministic. Returns the temp home path.
    """
    monkeypatch.setattr(install_mod, "self_command", lambda: ["/usr/bin/ostinote"])
    return isolate_home


@pytest.fixture
def make_project_env(tmp_path, monkeypatch):
    """Factory for a ready-to-use project Env (pytest factory-fixture idiom).

    Wraps ``helpers.project_env`` so tests stop threading tmp_path/monkeypatch
    by hand; pass ``extra_cfg`` to re-enable the specific gate a test probes.
    """

    def _make(extra_cfg=None):
        return project_env(tmp_path, monkeypatch, extra_cfg)

    return _make
