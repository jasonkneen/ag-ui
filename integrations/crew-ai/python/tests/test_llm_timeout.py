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

from unittest.mock import patch

import pytest

from ag_ui_crewai.crews import _llm_timeout_seconds


def test_default_llm_timeout_is_set(monkeypatch):
    """With no env var, the default is a finite, positive number of seconds."""
    monkeypatch.delenv("AGUI_CREWAI_LLM_TIMEOUT_SECONDS", raising=False)
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
    with patch.object(type(flow), "state", new=property(lambda self: flow._state)):
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                await flow.chat()

    assert _fake_acompletion.calls, "acompletion was never invoked"
    kwargs = _fake_acompletion.calls[0]
    assert "timeout" in kwargs, f"acompletion call missing timeout kwarg: {kwargs}"
    assert kwargs["timeout"] is None or kwargs["timeout"] > 0


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

            class _Msg(dict):
                def get(self, k, default=None):
                    return dict.get(self, k, default)

            if call_index["n"] == 1:
                msg = _Msg(
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
                msg = _Msg(role="assistant", content="bye")

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
    flow._state = {  # pylint: disable=protected-access
        "messages": [],
        "inputs": {},
        "copilotkit": {"actions": []},
    }

    with patch.object(type(flow), "state", new=property(lambda self: flow._state)):
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
        assert kwargs["timeout"] is None or kwargs["timeout"] > 0
