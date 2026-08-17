"""Runtime contract of the dojo server's ``dev`` entry point.

``render.yaml`` and ``apps/dojo/scripts/run-dojo-everything.js`` both run ``uv run dev``
and both set ``PORT``, and the uvicorn target has to stay an import string
for the app to be built once. Neither survives a well-meaning simplification
unless something asserts it, so this file asserts it.
"""

import os
import subprocess
import sys

import pytest

from agents import dojo as launcher


@pytest.fixture
def uvicorn_calls(monkeypatch):
    """Record the launcher's ``uvicorn.run`` calls instead of serving anything."""
    calls = []
    monkeypatch.setattr(
        launcher.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def _target(call):
    """The app the launcher handed uvicorn, positional or keyword."""
    args, kwargs = call
    return args[0] if args else kwargs["app"]


def test_uvicorn_gets_an_import_string_not_an_app_object(uvicorn_calls):
    launcher.main()

    target = _target(uvicorn_calls[0])
    assert isinstance(target, str), (
        "the target must stay an import string: an app object here is built in "
        f"this process and again in the reloader's worker, got {target!r}"
    )
    assert target == "agents.dojo:app"


def test_the_server_binds_every_interface_with_reload(uvicorn_calls):
    """render.yaml and the dojo runner both serve this from a container, so a
    loopback bind makes it unreachable, and reload is why the target has to stay an
    import string. Neither survives a well-meaning simplification unasserted."""
    launcher.main()

    kwargs = uvicorn_calls[0][1]
    assert kwargs.get("host") == "0.0.0.0", (
        f"the bind cannot be loopback, got {kwargs.get('host')!r}"
    )
    assert kwargs.get("reload") is True, (
        f"the dojo is served with reload, got {kwargs.get('reload')!r}"
    )


def test_importing_the_package_opts_out_before_crewai_loads():
    """The telemetry opt-out only works if it runs before crewai is imported, and
    importing the bridge imports crewai, so this package's ``__init__`` is the last
    point early enough."""
    probe = (
        "import os, sys, agents;"
        " print(os.environ.get('CREWAI_DISABLE_TELEMETRY'), 'crewai' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "true False", result.stderr


def test_port_falls_back_to_8000(uvicorn_calls, monkeypatch):
    monkeypatch.delenv("PORT", raising=False)

    launcher.main()

    assert uvicorn_calls[0][1]["port"] == 8000


def test_port_comes_from_the_environment(uvicorn_calls, monkeypatch):
    monkeypatch.setenv("PORT", "8003")

    launcher.main()

    assert uvicorn_calls[0][1]["port"] == 8003


def test_a_non_numeric_port_fails_loudly(uvicorn_calls, monkeypatch):
    monkeypatch.setenv("PORT", "eight-thousand")

    with pytest.raises(ValueError):
        launcher.main()

    assert uvicorn_calls == []


def test_main_returns_zero(uvicorn_calls):
    assert launcher.main() == 0


def _telemetry_after_import(env):
    """What ``CREWAI_DISABLE_TELEMETRY`` reads as in a fresh process that imported
    this package. A subprocess because this one imported it long ago."""
    result = subprocess.run(
        [sys.executable, "-c", "import os, agents; print(os.environ['CREWAI_DISABLE_TELEMETRY'])"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_telemetry_is_opted_out_before_the_worker_is_spawned():
    """Ctrl-C has to work.

    crewai's telemetry chains a SIGINT handler that force-flushes spans over the
    network, which wedges the reloader's worker mid-shutdown. uvicorn's worker
    inherits this environment, so the opt-out has to be set before it is spawned,
    which is why it lives at package import rather than in ``main()``.
    """
    env = {k: v for k, v in os.environ.items() if k != "CREWAI_DISABLE_TELEMETRY"}
    result = subprocess.run(
        [sys.executable, "-c", "import os, agents; print(os.environ['CREWAI_DISABLE_TELEMETRY'])"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "true"


def test_an_explicit_telemetry_choice_is_left_alone():
    """Opting back in is a developer's call to make, so don't overwrite it."""
    assert _telemetry_after_import({"CREWAI_DISABLE_TELEMETRY": "false"}) == "false"
