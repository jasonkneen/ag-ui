"""
Regression tests for Defect A follow-up: ensure LiteLLM streaming calls in
``crews.py`` are made with an explicit read timeout so a half-open TCP stream
cannot hang the request forever.

The earlier fix switched the streaming call from the sync ``litellm.completion``
to ``litellm.acompletion``, but LiteLLM still inherits whatever (possibly
unbounded) timeout the underlying HTTP client defaults to. These tests pin
down the timeout-forwarding behaviour.
"""

import os
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ag_ui_crewai.crews import _llm_timeout_seconds


def test_default_llm_timeout_is_set():
    """With no env var, the default is a finite, positive number of seconds."""
    if "AGUI_CREWAI_LLM_TIMEOUT_SECONDS" in os.environ:
        del os.environ["AGUI_CREWAI_LLM_TIMEOUT_SECONDS"]
    value = _llm_timeout_seconds()
    assert isinstance(value, float)
    assert value > 0.0
    # A sane ceiling — not minutes away, not hours.
    assert value < 3600.0


def test_llm_timeout_env_override(monkeypatch):
    monkeypatch.setenv("AGUI_CREWAI_LLM_TIMEOUT_SECONDS", "7.5")
    assert _llm_timeout_seconds() == pytest.approx(7.5)


def test_llm_timeout_disabled_for_non_positive(monkeypatch):
    monkeypatch.setenv("AGUI_CREWAI_LLM_TIMEOUT_SECONDS", "0")
    assert _llm_timeout_seconds() is None
    monkeypatch.setenv("AGUI_CREWAI_LLM_TIMEOUT_SECONDS", "-1")
    assert _llm_timeout_seconds() is None


def test_llm_timeout_bad_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AGUI_CREWAI_LLM_TIMEOUT_SECONDS", "not-a-number")
    value = _llm_timeout_seconds()
    assert isinstance(value, float)
    assert value > 0.0


@pytest.mark.asyncio
async def test_acompletion_called_with_timeout_kwarg():
    """``ChatWithCrewFlow.chat`` must forward the timeout to acompletion.

    We patch ``acompletion`` to capture the kwargs, and patch ``copilotkit_stream``
    to skip the real streaming path — we only care about the call shape.
    """
    from ag_ui_crewai import crews as crews_mod

    # A non-exception response is fine; chat() short-circuits on the mocked
    # copilotkit_stream and inspects its return value.
    sentinel = object()

    async def _fake_acompletion(**kwargs):
        _fake_acompletion.calls.append(kwargs)
        return sentinel

    _fake_acompletion.calls = []

    async def _fake_stream(resp):
        # Return a minimal object the chat() body can poke at; it accesses
        # response.choices[0]["message"].
        class _Msg(dict):
            def get(self, k, default=None):
                return dict.get(self, k, default)

        class _Resp:
            choices = [{"message": _Msg(role="assistant", content="done")}]

        return _Resp()

    # Build a tiny ChatWithCrewFlow without going through __init__ (which
    # requires a real Crew). We patch the parts chat() reads.
    flow = crews_mod.ChatWithCrewFlow.__new__(crews_mod.ChatWithCrewFlow)
    flow.crew = type("C", (), {"chat_llm": "gpt-4o"})()
    flow.crew_name = "dummy"
    flow.crew_tool_schema = {
        "type": "function",
        "function": {"name": "dummy_tool", "description": "", "parameters": {"type": "object"}},
    }
    flow.system_message = "sys"
    # chat() pulls from self.state — stash a minimal shape.
    flow._state = {  # pylint: disable=protected-access
        "messages": [],
        "inputs": {},
        "copilotkit": {"actions": []},
    }

    # Flow exposes state via descriptor; patch it to return our dict directly.
    with patch.object(type(flow), "state", new=property(lambda self: self._state)):
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                await flow.chat()

    assert _fake_acompletion.calls, "acompletion was never invoked"
    kwargs = _fake_acompletion.calls[0]
    assert "timeout" in kwargs, f"acompletion call missing timeout kwarg: {kwargs}"
    assert kwargs["timeout"] is None or kwargs["timeout"] > 0
