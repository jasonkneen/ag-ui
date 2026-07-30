"""Tests for native Strands interrupt <-> AG-UI interrupt round-trip.

Covers the four behaviors added to bridge ``tool_context.interrupt()`` to the
AG-UI interrupt lifecycle:

1. A paused run finishes with ``RunFinishedInterruptOutcome``.
2. ``RunAgentInput.resume`` is translated into the Strands resume prompt shape.
3. ``status == "cancelled"`` resumes with the documented denial sentinel.
4. Runs that never interrupt finish bare (no behavior change).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from strands.agent.state import AgentState
from strands.interrupt import Interrupt as StrandsInterrupt, _InterruptState

from ag_ui.core import EventType, ResumeEntry, RunAgentInput
from ag_ui_strands.agent import INTERRUPT_CANCELLED, StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_input(
    thread_id: str = "thread-1",
    run_id: str = "run-1",
    messages=None,
    resume=None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        state={},
        messages=messages or [],
        tools=[],
        context=[],
        forwarded_props={},
        resume=resume,
    )


async def _collect_events(agent: StrandsAgent, input_data: RunAgentInput) -> list:
    events = []
    async for event in agent.run(input_data):
        events.append(event)
    return events


def _make_base_agent() -> StrandsAgent:
    mock_core = MagicMock()
    mock_core.model = MagicMock()
    mock_core.system_prompt = "You are a test assistant."
    mock_core.tool_registry = MagicMock()
    mock_core.tool_registry.registry = {}
    mock_core.record_direct_tool_call = True
    # replay_history_into_strands defaults True; with no session manager this
    # takes the in-memory replay path (stream_async(None)). Disable it so the
    # legacy/resume paths are exercised straightforwardly in these unit tests.
    config = StrandsAgentConfig(replay_history_into_strands=False)
    return StrandsAgent(agent=mock_core, name="test_agent", config=config)


class _MockStrandsCore:
    """A minimal stand-in for ``StrandsAgentCore`` driving the stream loop.

    ``stream_async`` records the prompt it was called with and yields the
    provided terminal events. When ``interrupts`` are supplied it also flips its
    ``_interrupt_state`` to activated, mirroring a paused native run.
    """

    def __init__(self, terminal_events=None, interrupts=None):
        self.tool_registry = MagicMock()
        self.tool_registry.registry = {}
        self.state = AgentState()
        self.model = MagicMock()
        self.messages = []
        self.stream_prompts = []
        self._terminal_events = terminal_events or []
        self._interrupt_state = _InterruptState()
        if interrupts:
            for itr in interrupts:
                self._interrupt_state.interrupts[itr.id] = itr
            self._interrupt_state.activate()

    async def stream_async(self, prompt):
        self.stream_prompts.append(prompt)
        for event in self._terminal_events:
            yield event


def _agent_result_with_interrupt(interrupts):
    result = MagicMock()
    result.stop_reason = "interrupt"
    result.interrupts = interrupts
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInterruptOutcome:
    @pytest.mark.asyncio
    async def test_pause_emits_interrupt_outcome(self):
        """A native interrupt produces RUN_FINISHED with an interrupt outcome."""
        strands_interrupt = StrandsInterrupt(
            id="int-1", name="confirm", reason={"summary": "delete all"}
        )
        core = _MockStrandsCore(
            terminal_events=[{"result": _agent_result_with_interrupt([strands_interrupt])}],
            interrupts=[strands_interrupt],
        )
        agent = _make_base_agent()

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(agent, _make_run_input())

        finished = next(e for e in events if e.type == EventType.RUN_FINISHED)
        assert finished.outcome is not None
        assert finished.outcome.type == "interrupt"
        assert len(finished.outcome.interrupts) == 1

        agui_interrupt = finished.outcome.interrupts[0]
        assert agui_interrupt.id == "int-1"
        # Strands interrupt *name* maps to the categorical AG-UI reason.
        assert agui_interrupt.reason == "confirm"
        # The free-form Strands reason object is preserved under metadata.
        assert agui_interrupt.metadata == {"strands_reason": {"summary": "delete all"}}

    @pytest.mark.asyncio
    async def test_detects_interrupt_from_state_when_result_missing(self):
        """Falls back to _interrupt_state when no terminal result event arrives."""
        strands_interrupt = StrandsInterrupt(id="int-2", name="approve", reason=None)
        # No {"result": ...} event this time — only the live interrupt state.
        core = _MockStrandsCore(terminal_events=[], interrupts=[strands_interrupt])
        agent = _make_base_agent()

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(agent, _make_run_input())

        finished = next(e for e in events if e.type == EventType.RUN_FINISHED)
        assert finished.outcome is not None
        assert finished.outcome.type == "interrupt"
        assert finished.outcome.interrupts[0].id == "int-2"

    @pytest.mark.asyncio
    async def test_no_interrupt_finishes_bare(self):
        """A normal run finishes with no outcome (back-compat, no behavior change)."""
        result = MagicMock()
        result.stop_reason = "end_turn"
        result.interrupts = None
        core = _MockStrandsCore(terminal_events=[{"result": result}])
        agent = _make_base_agent()

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(agent, _make_run_input())

        finished = next(e for e in events if e.type == EventType.RUN_FINISHED)
        assert finished.outcome is None


class TestResumeConsumption:
    @pytest.mark.asyncio
    async def test_resolved_resume_builds_interrupt_response_prompt(self):
        """A resolved ResumeEntry is translated into the Strands resume prompt."""
        core = _MockStrandsCore(terminal_events=[])
        agent = _make_base_agent()
        resume = [ResumeEntry(interrupt_id="int-1", status="resolved", payload="yes")]

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            await _collect_events(agent, _make_run_input(resume=resume))

        assert core.stream_prompts == [
            [{"interruptResponse": {"interruptId": "int-1", "response": "yes"}}]
        ]

    @pytest.mark.asyncio
    async def test_cancelled_resume_uses_sentinel(self):
        """A cancelled ResumeEntry resumes with the denial sentinel as response."""
        core = _MockStrandsCore(terminal_events=[])
        agent = _make_base_agent()
        resume = [ResumeEntry(interrupt_id="int-1", status="cancelled", payload=None)]

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            await _collect_events(agent, _make_run_input(resume=resume))

        assert core.stream_prompts == [
            [{"interruptResponse": {"interruptId": "int-1", "response": INTERRUPT_CANCELLED}}]
        ]

    @pytest.mark.asyncio
    async def test_multiple_resume_entries(self):
        """Every ResumeEntry becomes one interruptResponse content block."""
        core = _MockStrandsCore(terminal_events=[])
        agent = _make_base_agent()
        resume = [
            ResumeEntry(interrupt_id="a", status="resolved", payload={"k": 1}),
            ResumeEntry(interrupt_id="b", status="cancelled", payload=None),
        ]

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            await _collect_events(agent, _make_run_input(resume=resume))

        assert core.stream_prompts == [
            [
                {"interruptResponse": {"interruptId": "a", "response": {"k": 1}}},
                {"interruptResponse": {"interruptId": "b", "response": INTERRUPT_CANCELLED}},
            ]
        ]
