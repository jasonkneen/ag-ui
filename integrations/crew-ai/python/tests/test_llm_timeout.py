"""
Regression tests for Defect A follow-up: ensure LiteLLM streaming calls in
``crews.py`` are made with an explicit read timeout so a half-open TCP stream
cannot hang the request forever.

The earlier fix switched the streaming call from the sync ``litellm.completion``
to ``litellm.acompletion``, but LiteLLM still inherits whatever (possibly
unbounded) timeout the underlying HTTP client defaults to. These tests pin
down the timeout-forwarding behaviour for BOTH acompletion call sites in
``ChatWithCrewFlow.chat``.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from ag_ui_crewai.crews import _DEFAULT_LLM_TIMEOUT_SECONDS, _llm_timeout_seconds


@contextmanager
def _patch_instance_state(flow, state):
    """Install ``state`` on a single flow instance via a throwaway subclass.

    Flow.state is a class-level descriptor, so a naive attribute assignment
    goes through the descriptor's ``__set__``. Rather than mutate the shared
    class (which is unsafe under ``pytest-xdist`` and other parallel test
    runners — finding #9), we rebind the instance's ``__class__`` to a
    freshly-minted subclass that declares ``state`` as a plain property
    reading from ``flow._state``. The subclass is per-instance, so two
    parallel tests patching two different ``flow`` instances cannot race on
    the same descriptor.

    On exit the original class is restored. If the caller has already
    replaced ``__class__`` themselves we leave well enough alone.
    """

    flow._state = state  # pylint: disable=protected-access
    original_cls = type(flow)
    subclass = type(
        f"{original_cls.__name__}_StatePatched",
        (original_cls,),
        {"state": property(lambda self: self._state)},
    )
    flow.__class__ = subclass
    try:
        yield
    finally:
        if flow.__class__ is subclass:
            flow.__class__ = original_cls


def test_default_llm_timeout_is_set(monkeypatch):
    """With no env var, the default is the module constant — a finite,
    positive, non-None number of seconds. Using ``==`` locks the value in
    so a regression that disables the default (e.g. silently returning
    ``None``) is caught loudly.
    """
    monkeypatch.delenv("AGUI_CREWAI_LLM_TIMEOUT_SECONDS", raising=False)
    value = _llm_timeout_seconds()
    assert value == _DEFAULT_LLM_TIMEOUT_SECONDS
    # A sane ceiling — not minutes away, not hours.
    assert 0.0 < _DEFAULT_LLM_TIMEOUT_SECONDS < 3600.0


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


async def test_acompletion_called_with_timeout_kwarg():
    """``ChatWithCrewFlow.chat`` must forward the timeout to the first
    acompletion call site."""
    from ag_ui_crewai import crews as crews_mod

    sentinel = object()

    async def _fake_acompletion(**kwargs):
        _fake_acompletion.calls.append(kwargs)
        return sentinel

    _fake_acompletion.calls = []

    async def _fake_stream(resp):
        # Return a minimal object the chat() body can poke at; it accesses
        # response.choices[0]["message"]. A plain ``dict`` already supplies
        # the ``.get`` the code uses — no custom subclass needed.
        class _Resp:
            choices = [{"message": dict(role="assistant", content="done")}]

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
    state = {
        "messages": [],
        "inputs": {},
        "copilotkit": {"actions": []},
    }

    with _patch_instance_state(flow, state):
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                await flow.chat()

    assert _fake_acompletion.calls, "acompletion was never invoked"
    kwargs = _fake_acompletion.calls[0]
    assert "timeout" in kwargs, f"acompletion call missing timeout kwarg: {kwargs}"
    # With the default env, the timeout is the module default — lock the
    # value in so a regression that silently disables the default (None) is
    # caught loudly. This is finding #4: reject the "None or >0" tautology.
    assert kwargs["timeout"] == _DEFAULT_LLM_TIMEOUT_SECONDS


async def test_acompletion_crew_exit_path_also_forwards_timeout():
    """The second acompletion call site (after ``crew_exit`` tool call) must
    also forward the timeout kwarg.

    The flow: first acompletion returns a tool_call for CREW_EXIT_TOOL, which
    drives the code into the exit branch where a second acompletion is
    issued with ``tool_choice="none"``. Every acompletion invocation must
    carry the timeout.
    """
    from ag_ui_crewai import crews as crews_mod

    async def _fake_acompletion(**kwargs):
        _fake_acompletion.calls.append(kwargs)
        return {"marker": len(_fake_acompletion.calls)}

    _fake_acompletion.calls = []

    # The first call yields a CREW_EXIT_TOOL tool_call; the second yields a
    # plain assistant reply.
    def _stream_factory():
        call_index = {"n": 0}

        async def _fake_stream(resp):  # pylint: disable=unused-argument
            call_index["n"] += 1

            if call_index["n"] == 1:
                msg = dict(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "function": {
                                "name": crews_mod.CREW_EXIT_TOOL["function"]["name"],
                                "arguments": "{}",
                            },
                        }
                    ],
                )
            else:
                msg = dict(role="assistant", content="bye")

            class _Resp:
                choices = [{"message": msg}]

            return _Resp()

        return _fake_stream

    async def _fake_exit():
        return None

    flow = crews_mod.ChatWithCrewFlow.__new__(crews_mod.ChatWithCrewFlow)
    flow.crew = type("C", (), {"chat_llm": "gpt-4o"})()
    flow.crew_name = "dummy"
    flow.crew_tool_schema = {
        "type": "function",
        "function": {"name": "dummy_tool", "description": "", "parameters": {"type": "object"}},
    }
    flow.system_message = "sys"
    state = {
        "messages": [],
        "inputs": {},
        "copilotkit": {"actions": []},
    }

    with _patch_instance_state(flow, state):
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _stream_factory()):
                with patch.object(crews_mod, "copilotkit_exit", _fake_exit):
                    await flow.chat()

    assert len(_fake_acompletion.calls) == 2, (
        f"expected 2 acompletion calls (exit tool path), got {len(_fake_acompletion.calls)}"
    )
    for idx, kwargs in enumerate(_fake_acompletion.calls):
        assert "timeout" in kwargs, (
            f"acompletion call #{idx} missing timeout kwarg: {kwargs}"
        )
        # Default env → default timeout; locked in to prevent silent regression
        # to ``None`` (disabled).
        assert kwargs["timeout"] == _DEFAULT_LLM_TIMEOUT_SECONDS, (
            f"acompletion call #{idx} timeout should be the default "
            f"({_DEFAULT_LLM_TIMEOUT_SECONDS}); got {kwargs['timeout']!r}"
        )
