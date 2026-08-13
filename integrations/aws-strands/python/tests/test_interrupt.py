"""Tests for native Strands interrupt <-> AG-UI interrupt round-trip.

Covers the four behaviors added to bridge ``tool_context.interrupt()`` to the
AG-UI interrupt lifecycle:

1. A paused run finishes with ``RunFinishedInterruptOutcome``.
2. ``RunAgentInput.resume`` is translated into the Strands resume prompt shape.
3. ``status == "cancelled"`` resumes with the documented denial sentinel.
4. Runs that never interrupt finish bare (no behavior change).
"""

from __future__ import annotations

import asyncio
import copy
import json
from unittest.mock import MagicMock, patch

import pytest
from ag_ui.core import (
    AssistantMessage,
    CustomEvent,
    EventType,
    FunctionCall,
    ResumeEntry,
    RunAgentInput,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ag_ui.encoder import EventEncoder
from strands import Agent as StrandsAgentCore
from strands import ToolContext, tool
from strands.agent.state import AgentState
from strands.hooks import HookProvider
from strands.hooks.events import (
    AgentInitializedEvent,
    AfterToolCallEvent,
    BeforeToolCallEvent,
)
from strands.interrupt import Interrupt as StrandsInterrupt
from strands.interrupt import InterruptException, _InterruptState
from strands.models.model import Model as StrandsModel
from strands.session import FileSessionManager, SessionManager
from strands.tools.executors import ConcurrentToolExecutor

import ag_ui_strands.agent as strands_agent_module
from ag_ui_strands.agent import (
    INTERRUPT_CANCELLED,
    StrandsAgent,
    _interrupt_metadata_to_json_safe,
)
from ag_ui_strands.config import StrandsAgentConfig, ToolBehavior
from ag_ui_strands.session_reconcile import (
    AG_UI_PENDING_PROXY_RESULTS_STATE_KEY,
    AG_UI_TOOL_CALL_MAP_STATE_KEY,
    AG_UI_WIRE_MAP_STATE_KEY,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_input(
    thread_id: str = "thread-1",
    run_id: str = "run-1",
    messages=None,
    resume=None,
    tools=None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        state={},
        messages=messages or [],
        tools=tools or [],
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


_UNSET = object()


class _OpaqueInterruptDetail:
    """Non-serializable detail with the default address-bearing repr."""


_CYCLIC_INTERRUPT_DETAIL = {}
_CYCLIC_INTERRUPT_DETAIL["self"] = _CYCLIC_INTERRUPT_DETAIL


class _MockStrandsCore:
    """A minimal stand-in for ``StrandsAgentCore`` driving the stream loop.

    ``stream_async`` records the prompt it was called with and yields the
    provided terminal events. When ``interrupts`` are supplied it also flips its
    ``_interrupt_state`` to activated, mirroring a paused native run.
    """

    def __init__(self, terminal_events=None, interrupts=None, session_manager=_UNSET):
        self.tool_registry = MagicMock()
        self.tool_registry.registry = {}
        self.state = AgentState()
        self.model = MagicMock()
        self.messages = []
        self.stream_prompts = []
        # Default to a mock session manager: the ``session_manager is None``
        # guard now rejects interrupts/resume without one, and most tests
        # here exercise the resume-translation logic, not that guard. Pass
        # ``session_manager=None`` explicitly to exercise the guard itself.
        self.session_manager = MagicMock() if session_manager is _UNSET else session_manager
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
    def test_interrupt_metadata_mapping_keys_are_injective(self):
        normalized = _interrupt_metadata_to_json_safe(
            {
                "bad\udcffkey": "surrogate-value",
                "bad\\udcffkey": "literal-value",
            }
        )

        assert normalized == {
            "__ag_ui_key_v1__:s:YmFk7bO_a2V5": "surrogate-value",
            "bad\\udcffkey": "literal-value",
        }

    def test_interrupt_metadata_mapping_key_namespace_cannot_be_spoofed(self):
        normalized = _interrupt_metadata_to_json_safe(
            {
                "bad\udcffkey": "surrogate-value",
                "__ag_ui_key_v1__:s:YmFk7bO_a2V5": "spoof-value",
                "__ag_ui_key_v1__:literal": "reserved-value",
            }
        )

        assert normalized == {
            "__ag_ui_key_v1__:s:YmFk7bO_a2V5": "surrogate-value",
            (
                "__ag_ui_key_v1__:v:"
                "X19hZ191aV9rZXlfdjFfXzpzOlltRms3Yk9fYTJWNQ"
            ): "spoof-value",
            (
                "__ag_ui_key_v1__:v:"
                "X19hZ191aV9rZXlfdjFfXzpsaXRlcmFs"
            ): "reserved-value",
        }

    def test_interrupt_metadata_preserves_normal_string_mapping_shape(self):
        reason = {
            "plain": "value",
            "café": {"東京": "値"},
        }

        assert _interrupt_metadata_to_json_safe(reason) == reason

    def test_interrupt_metadata_encodes_string_keys_in_mixed_mappings(self):
        normalized = _interrupt_metadata_to_json_safe(
            {
                "ordinary": "plain-value",
                "bad\udcffkey": "surrogate-key-value",
                7: "integer-key-value",
            }
        )

        assert normalized["__ag_ui_type__"] == "mapping"
        assert ["ordinary", "plain-value"] in normalized["items"]
        assert [
            "__ag_ui_key_v1__:s:YmFk7bO_a2V5",
            "surrogate-key-value",
        ] in normalized["items"]
        assert [7, "integer-key-value"] in normalized["items"]

    def test_interrupt_metadata_scalar_strings_keep_existing_normalization(self):
        normalized = _interrupt_metadata_to_json_safe(
            {
                "surrogate": "bad\udcffvalue",
                "reserved": "__ag_ui_key_v1__:literal",
            }
        )

        assert normalized == {
            "surrogate": "bad\\udcffvalue",
            "reserved": "__ag_ui_key_v1__:literal",
        }

    @pytest.mark.asyncio
    async def test_pause_emits_interrupt_outcome(self):
        """A native interrupt produces RUN_FINISHED with an interrupt outcome."""
        strands_interrupt = StrandsInterrupt(
            id="v1:tool_call:tu-1:00000000-0000-0000-0000-000000000000",
            name="confirm",
            reason={"summary": "delete all"},
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
        assert agui_interrupt.id == "v1:tool_call:tu-1:00000000-0000-0000-0000-000000000000"
        # Every Strands interrupt is tool-call-bound; id embeds the toolUseId.
        assert agui_interrupt.tool_call_id == "tu-1"
        assert agui_interrupt.reason == "tool_call"
        # The free-form Strands name/reason are preserved under metadata.
        assert agui_interrupt.metadata == {
            "strands_name": "confirm",
            "strands_reason": {"summary": "delete all"},
        }

    @pytest.mark.parametrize("session_backed", [False, True])
    @pytest.mark.parametrize(
        ("reason", "expected_reason", "expected_message"),
        [
            pytest.param(
                b"\xff\xfe",
                {"__bytes_encoded__": True, "data": "//4="},
                None,
                id="invalid-utf8-bytes",
            ),
            pytest.param(
                {"choices": {"beta", "alpha"}},
                {
                    "choices": {
                        "__ag_ui_type__": "set",
                        "items": ["alpha", "beta"],
                    }
                },
                None,
                id="set",
            ),
            pytest.param(
                _OpaqueInterruptDetail(),
                {
                    "__ag_ui_type__": "python_object",
                    "type": (
                        f"{_OpaqueInterruptDetail.__module__}."
                        f"{_OpaqueInterruptDetail.__qualname__}"
                    ),
                },
                None,
                id="custom-object",
            ),
            pytest.param(
                {
                    "details": [
                        b"\x00\xff",
                        {3, 1, 2},
                        _OpaqueInterruptDetail(),
                    ]
                },
                {
                    "details": [
                        {"__bytes_encoded__": True, "data": "AP8="},
                        {"__ag_ui_type__": "set", "items": [1, 2, 3]},
                        {
                            "__ag_ui_type__": "python_object",
                            "type": (
                                f"{_OpaqueInterruptDetail.__module__}."
                                f"{_OpaqueInterruptDetail.__qualname__}"
                            ),
                        },
                    ]
                },
                None,
                id="nested-non-json-values",
            ),
            pytest.param(
                {"summary": "delete all", "attempt": 2, "approved": False},
                {"summary": "delete all", "attempt": 2, "approved": False},
                None,
                id="ordinary-dict",
            ),
            pytest.param(
                "Approve deletion",
                "Approve deletion",
                "Approve deletion",
                id="ordinary-string",
            ),
            pytest.param(
                "invalid\udcffunicode",
                "invalid\\udcffunicode",
                "invalid\\udcffunicode",
                id="invalid-unicode-string",
            ),
            pytest.param(
                _CYCLIC_INTERRUPT_DETAIL,
                {"self": {"__ag_ui_type__": "circular_reference"}},
                None,
                id="circular-container",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_interrupt_metadata_is_wire_safe(
        self,
        tmp_path,
        session_backed,
        reason,
        expected_reason,
        expected_message,
    ):
        """Native terminal outcomes remain serializable through the SSE boundary."""
        strands_interrupt = StrandsInterrupt(
            id="v1:tool_call:tu-1:00000000-0000-0000-0000-000000000000",
            name="confirm",
            reason=reason,
        )
        session_manager = (
            FileSessionManager(session_id="thread-1", storage_dir=str(tmp_path))
            if session_backed
            else None
        )
        core = _MockStrandsCore(
            terminal_events=[
                {"result": _agent_result_with_interrupt([strands_interrupt])}
            ],
            interrupts=[strands_interrupt],
            session_manager=session_manager,
        )
        agent = _make_base_agent()

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(agent, _make_run_input())

        finished = next(event for event in events if event.type == EventType.RUN_FINISHED)
        model_payload = json.loads(
            finished.model_dump_json(by_alias=True, exclude_none=True)
        )
        encoded = EventEncoder().encode(finished)
        assert encoded.startswith("data: ") and encoded.endswith("\n\n")
        sse_payload = json.loads(encoded.removeprefix("data: ").removesuffix("\n\n"))
        assert sse_payload == model_payload

        [wire_interrupt] = sse_payload["outcome"]["interrupts"]
        assert wire_interrupt["reason"] == "tool_call"
        assert wire_interrupt.get("message") == expected_message
        assert wire_interrupt["metadata"] == {
            "strands_name": "confirm",
            "strands_reason": expected_reason,
        }
        assert "0x" not in json.dumps(wire_interrupt["metadata"])

    @pytest.mark.asyncio
    async def test_interrupt_metadata_normalization_is_depth_bounded(self):
        reason = {"leaf": "diagnostic"}
        for _ in range(30):
            reason = {"nested": reason}
        strands_interrupt = StrandsInterrupt(
            id="v1:tool_call:tu-1:00000000-0000-0000-0000-000000000000",
            name="confirm",
            reason=reason,
        )
        core = _MockStrandsCore(
            terminal_events=[
                {"result": _agent_result_with_interrupt([strands_interrupt])}
            ],
            interrupts=[strands_interrupt],
            session_manager=None,
        )
        agent = _make_base_agent()

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(agent, _make_run_input())

        finished = next(event for event in events if event.type == EventType.RUN_FINISHED)
        encoded = EventEncoder().encode(finished)
        assert '"__ag_ui_type__":"max_depth_exceeded"' in encoded
        assert "diagnostic" not in encoded

    @pytest.mark.asyncio
    async def test_terminal_result_captured_despite_halt_in_same_cycle(self):
        """The terminal ``AgentResult`` capture must run before the
        ``halt_event_stream`` break check — otherwise a native interrupt whose
        terminal event arrives on/after the same cycle that triggers a
        frontend-tool halt is silently dropped (the run finishes bare instead
        of surfacing the interrupt).
        """
        open_interrupt = StrandsInterrupt(
            id="v1:tool_call:tu-native:00000000-0000-0000-0000-000000000000",
            name="confirm",
        )
        events = [
            {
                "current_tool_use": {
                    "toolUseId": "tu-fe",
                    "name": "get_cell",
                    "input": '{"cell": "B4"}',
                }
            },
            {"event": {"contentBlockStop": {}}},
            # Empty content models the interrupted turn skipping
            # ToolResultMessageEvent; pending_halt still
            # latches halt_event_stream here regardless.
            {"message": {"role": "user", "content": []}},
            {"result": _agent_result_with_interrupt([open_interrupt])},
        ]
        core = _MockStrandsCore(terminal_events=events)
        agent = _make_base_agent()
        frontend_tool = Tool(name="get_cell", description="Read a cell", parameters={})

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events_out = await _collect_events(agent, _make_run_input(tools=[frontend_tool]))

        finished = next(e for e in events_out if e.type == EventType.RUN_FINISHED)
        assert (
            finished.outcome is not None
        ), "terminal interrupt result was dropped on the halt path (round1.md #7a)"
        assert finished.outcome.type == "interrupt"
        assert finished.outcome.interrupts[0].id == open_interrupt.id

    @pytest.mark.asyncio
    async def test_fallback_excludes_already_answered_interrupts(self):
        """When the terminal ``AgentResult`` is unavailable and ``_extract_interrupts``
        falls back to the live ``_interrupt_state``, an interrupt that was already
        answered by a prior partial resume (truthy ``.response``) must not be
        re-reported as still pending alongside the genuinely open one.
        """
        answered = StrandsInterrupt(
            id="v1:tool_call:tu-answered:00000000-0000-0000-0000-000000000000",
            name="answered",
            response={"response": "yes"},
        )
        open_interrupt = StrandsInterrupt(
            id="v1:tool_call:tu-open:00000000-0000-0000-0000-000000000000",
            name="open",
        )
        # No terminal ``{"result": ...}`` event — mirrors the halt-event-stream
        # path where the stream breaks before a terminal AgentResult is captured.
        core = _MockStrandsCore(
            terminal_events=[],
            interrupts=[answered, open_interrupt],
        )
        agent = _make_base_agent()

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(agent, _make_run_input())

        finished = next(e for e in events if e.type == EventType.RUN_FINISHED)
        assert finished.outcome is not None
        assert finished.outcome.type == "interrupt"
        reported_ids = {i.id for i in finished.outcome.interrupts}
        assert reported_ids == {open_interrupt.id}

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
        """A resolved ResumeEntry is translated into the Strands resume prompt.

        The raw payload is wrapped in ``{"response": ...}`` so Strands' truthiness
        gate always passes; the tool destructures via ``.get("response")``.
        """
        core = _MockStrandsCore(
            terminal_events=[],
            interrupts=[StrandsInterrupt(id="int-1", name="confirm")],
        )
        agent = _make_base_agent()
        resume = [ResumeEntry(interrupt_id="int-1", status="resolved", payload="yes")]

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            await _collect_events(agent, _make_run_input(resume=resume))

        assert core.stream_prompts == [
            [{"interruptResponse": {"interruptId": "int-1", "response": {"response": "yes"}}}]
        ]

    @pytest.mark.parametrize("falsy_payload", [None, False, "", 0, [], {}])
    @pytest.mark.asyncio
    async def test_resolved_resume_wraps_falsy_payload_in_truthy_envelope(self, falsy_payload):
        """Falsy resume payloads must be wrapped so Strands' ``if response:`` gate passes.

        Regression for ``round1.md`` #1: without the envelope, ``None``/``False``/
        ``""``/``0``/``[]``/``{}`` re-emit the same interrupt id on the resume
        run, re-running the tool body forever.
        """
        core = _MockStrandsCore(
            terminal_events=[],
            interrupts=[StrandsInterrupt(id="int-1", name="confirm")],
        )
        agent = _make_base_agent()
        resume = [ResumeEntry(interrupt_id="int-1", status="resolved", payload=falsy_payload)]

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            await _collect_events(agent, _make_run_input(resume=resume))

        [wrapped] = core.stream_prompts
        assert wrapped == [
            {"interruptResponse": {"interruptId": "int-1", "response": {"response": falsy_payload}}}
        ]
        # The envelope itself must be truthy — that is the whole point.
        assert bool(wrapped[0]["interruptResponse"]["response"])

    @pytest.mark.asyncio
    async def test_cancelled_resume_uses_sentinel(self):
        """A cancelled ResumeEntry resumes with the denial sentinel as response."""
        core = _MockStrandsCore(
            terminal_events=[],
            interrupts=[StrandsInterrupt(id="int-1", name="confirm")],
        )
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
        core = _MockStrandsCore(
            terminal_events=[],
            interrupts=[
                StrandsInterrupt(id="a", name="confirm-a"),
                StrandsInterrupt(id="b", name="confirm-b"),
            ],
        )
        agent = _make_base_agent()
        resume = [
            ResumeEntry(interrupt_id="a", status="resolved", payload={"k": 1}),
            ResumeEntry(interrupt_id="b", status="cancelled", payload=None),
        ]

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            await _collect_events(agent, _make_run_input(resume=resume))

        assert core.stream_prompts == [
            [
                {"interruptResponse": {"interruptId": "a", "response": {"response": {"k": 1}}}},
                {"interruptResponse": {"interruptId": "b", "response": INTERRUPT_CANCELLED}},
            ]
        ]

    @pytest.mark.asyncio
    async def test_partial_resume_subset_remains_allowed(self):
        """A resume batch need not answer every currently open interrupt."""
        core = _MockStrandsCore(
            terminal_events=[],
            interrupts=[
                StrandsInterrupt(id="a", name="confirm-a"),
                StrandsInterrupt(id="b", name="confirm-b"),
            ],
        )
        agent = _make_base_agent()

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(
                agent,
                _make_run_input(
                    resume=[
                        ResumeEntry(
                            interrupt_id="a", status="resolved", payload=True
                        )
                    ]
                ),
            )

        assert not any(event.type == EventType.RUN_ERROR for event in events)
        assert core.stream_prompts == [
            [
                {
                    "interruptResponse": {
                        "interruptId": "a",
                        "response": {"response": True},
                    }
                }
            ]
        ]


class TestLiveInterruptsWithoutSessionManager:
    """The cached per-thread core is a live interrupt checkpoint."""

    @pytest.mark.asyncio
    async def test_active_interrupt_without_session_manager_emits_outcome(self):
        """A paused native run does not require durable session storage."""
        strands_interrupt = StrandsInterrupt(
            id="v1:tool_call:tu-1:00000000-0000-0000-0000-000000000000",
            name="confirm",
            reason={"summary": "delete all"},
        )
        core = _MockStrandsCore(
            terminal_events=[{"result": _agent_result_with_interrupt([strands_interrupt])}],
            interrupts=[strands_interrupt],
            session_manager=None,
        )
        agent = _make_base_agent()

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(agent, _make_run_input())

        assert not any(e.type == EventType.RUN_ERROR for e in events)
        finished = next(e for e in events if e.type == EventType.RUN_FINISHED)
        assert finished.outcome is not None
        assert finished.outcome.type == "interrupt"
        assert finished.outcome.interrupts[0].id == strands_interrupt.id

    @pytest.mark.asyncio
    async def test_resume_entries_without_session_manager_are_streamed(self):
        """Resume entries are translated against the cached live core."""
        core = _MockStrandsCore(
            terminal_events=[],
            interrupts=[StrandsInterrupt(id="int-1", name="confirm")],
            session_manager=None,
        )
        agent = _make_base_agent()
        resume = [ResumeEntry(interrupt_id="int-1", status="resolved", payload="yes")]

        with patch("ag_ui_strands.agent.StrandsAgentCore", return_value=core):
            events = await _collect_events(agent, _make_run_input(resume=resume))

        assert not any(e.type == EventType.RUN_ERROR for e in events)
        assert any(e.type == EventType.RUN_FINISHED for e in events)
        assert core.stream_prompts == [
            [
                {
                    "interruptResponse": {
                        "interruptId": "int-1",
                        "response": {"response": "yes"},
                    }
                }
            ]
        ]


# ---------------------------------------------------------------------------
# Real-agent end-to-end regression
#
# The tests above replay canned events through ``_MockStrandsCore`` and never
# drive the real Strands event loop, tool executor, or interrupt machinery.
# This section runs a real ``strands.Agent`` with a scripted stub ``Model``
# and a real ``@tool(context=True)`` tool so the interrupt/resume round-trip
# is exercised for real.
# ---------------------------------------------------------------------------


@tool(context=True)
def confirm_action(key: str, tool_context: ToolContext) -> dict:
    # Resume envelope: {"cancelled": True} on cancel, {"response": <raw>} on
    # resolve. Destructure — do NOT truthiness-check the envelope, since it is
    # always truthy on resolve (that's the whole point of the wrap).
    envelope = tool_context.interrupt("confirm_action", reason={"key": key})
    if envelope.get("cancelled"):
        return {"status": "success", "content": [{"text": f"denied {key}"}]}
    if envelope.get("response"):
        return {"status": "success", "content": [{"text": f"confirmed {key}"}]}
    return {"status": "success", "content": [{"text": f"denied {key}"}]}


@tool
def native_placeholder() -> str:
    """Return text that happens to equal the frontend proxy's reserved result."""
    return "Forwarded to client"


@tool(context=True)
def interrupt_with_colliding_metadata(tool_context: ToolContext) -> str:
    tool_context.interrupt(
        "collision",
        reason={
            "bad\udcffkey": "surrogate-value",
            "bad\\udcffkey": "literal-value",
            "__ag_ui_key_v1__:literal": "reserved-value",
        },
    )
    return "resumed"


class _InterruptFlowModel(StrandsModel):
    """Turn 1 calls a sibling tool and an interrupting native tool together."""

    def __init__(
        self,
        sibling_tool_name: str = "approveTool",
        sibling_tool_names: tuple[str, ...] | None = None,
    ):
        self.turn = 0
        self.stream_calls_messages = []
        self.sibling_tool_names = sibling_tool_names or (sibling_tool_name,)

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(self, output_model, prompt=None, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover — make this an async generator

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.turn += 1
        self.stream_calls_messages.append(messages)
        if self.turn == 1:
            yield {"messageStart": {"role": "assistant"}}
            for index, sibling_tool_name in enumerate(self.sibling_tool_names):
                native_id = (
                    "native-approve"
                    if index == 0
                    else f"native-approve-{index + 1}"
                )
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": native_id,
                                "name": sibling_tool_name,
                            }
                        }
                    }
                }
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
                yield {"contentBlockStop": {}}
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "native-confirm", "name": "confirm_action"}}
                }
            }
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"key": "widget-1"}'}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


class _NativeInterruptFlowModel(StrandsModel):
    """Turn 1 interrupts in a native tool; turn 2 narrates completion."""

    def __init__(self):
        self.turn = 0
        self.stream_calls_messages = []

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(self, output_model, prompt=None, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover — make this an async generator

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.turn += 1
        self.stream_calls_messages.append(copy.deepcopy(messages))
        if self.turn == 1:
            yield {"messageStart": {"role": "assistant"}}
            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "native-confirm",
                            "name": "confirm_action",
                        }
                    }
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": '{"key": "widget-1"}'}}
                }
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


class _MetadataInterruptFlowModel(StrandsModel):
    """Call the real metadata interrupt tool once."""

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(
        self, output_model, prompt=None, system_prompt=None, **kwargs
    ):
        raise NotImplementedError
        yield  # pragma: no cover — make this an async generator

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "toolUseId": "native-metadata",
                        "name": "interrupt_with_colliding_metadata",
                    }
                }
            }
        }
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


class _NonRepositorySessionManager(SessionManager):
    """Valid Strands session manager without repository-specific attributes."""

    def initialize(self, agent, **kwargs):
        pass

    def append_message(self, message, agent, **kwargs):
        pass

    def sync_agent(self, agent, **kwargs):
        pass

    def redact_latest_message(self, redact_message, agent, **kwargs):
        pass


class _ProxyHookInterruptFlowModel(StrandsModel):
    """Turn 1 calls a frontend proxy; turn 2 records the resumed result."""

    def __init__(self, *, tool_turns: set[int] | None = None):
        self.turn = 0
        self.stream_calls_messages = []
        self.tool_turns = tool_turns or {1}

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(self, output_model, prompt=None, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover — make this an async generator

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.turn += 1
        self.stream_calls_messages.append(copy.deepcopy(messages))
        if self.turn in self.tool_turns:
            native_id = (
                "native-approve"
                if self.turn == 1
                else f"native-approve-{self.turn}"
            )
            yield {"messageStart": {"role": "assistant"}}
            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": native_id,
                            "name": "approveTool",
                        }
                    }
                }
            }
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


class _ConcurrentProxyHookFlowModel(StrandsModel):
    """Emit two identical frontend proxy calls in one concurrent tool batch."""

    native_ids = ("native-proxy-a", "native-proxy-b")

    def __init__(self):
        self.turn = 0
        self.stream_calls_messages = []

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(
        self, output_model, prompt=None, system_prompt=None, **kwargs
    ):
        raise NotImplementedError
        yield  # pragma: no cover — make this an async generator

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.turn += 1
        self.stream_calls_messages.append(copy.deepcopy(messages))
        if self.turn == 1:
            yield {"messageStart": {"role": "assistant"}}
            for native_id in self.native_ids:
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": native_id,
                                "name": "approveTool",
                            }
                        }
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": "{}"}}
                    }
                }
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


class _AnsweredProxyHookFlowModel(StrandsModel):
    """Emit distinct A/B frontend calls, then record the completion turn."""

    native_by_tool = {
        "approveA": "native-a",
        "approveB": "native-b",
    }

    def __init__(self, *, turn: int = 0):
        self.turn = turn
        self.stream_calls_messages = []

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(
        self, output_model, prompt=None, system_prompt=None, **kwargs
    ):
        raise NotImplementedError
        yield  # pragma: no cover — make this an async generator

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.turn += 1
        self.stream_calls_messages.append(copy.deepcopy(messages))
        if self.turn == 1:
            yield {"messageStart": {"role": "assistant"}}
            for tool_name, native_id in self.native_by_tool.items():
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": native_id,
                                "name": tool_name,
                            }
                        }
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": "{}"}}
                    }
                }
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


class _InterruptFrontendProxyHook(HookProvider):
    """Pause a frontend proxy immediately before Strands invokes it."""

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    @staticmethod
    def _before_tool_call(event):
        if event.tool_use["name"] == "approveTool":
            event.interrupt("approve_proxy", reason="Approve frontend action")


class _MutatingFrontendProxyHook(HookProvider):
    """Exercise writable proxy-hook surfaces against the real executor."""

    def __init__(self, mutation: str, *, should_interrupt: bool = True):
        self.mutation = mutation
        self.should_interrupt = should_interrupt
        self.enabled = True

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event):
        if not self.enabled or event.tool_use["name"] != "approveTool":
            return
        if self.mutation == "id-in-place":
            event.tool_use["toolUseId"] = "hook-mutated-id"
        elif self.mutation == "id-replacement":
            event.tool_use = {
                **event.tool_use,
                "toolUseId": "hook-replacement-id",
            }
        elif self.mutation == "input":
            event.tool_use["input"] = {"changed": True}
        elif self.mutation == "name":
            event.tool_use["name"] = "replacementTool"
        elif self.mutation == "selected-tool":
            event.selected_tool = event.agent.tool_registry.registry["replacementTool"]
        elif self.mutation == "tool-spec":
            event.selected_tool.tool_spec["description"] = "hook changed"
        elif self.mutation == "wire-map-delete":
            event.agent.state.set(AG_UI_WIRE_MAP_STATE_KEY, {})
        elif self.mutation == "tool-meta-delete":
            event.agent.state.set(AG_UI_TOOL_CALL_MAP_STATE_KEY, {})
        if not self.should_interrupt:
            return
        event.interrupt("approve_proxy", reason="Approve frontend action")
        if self.mutation == "input-after-resume":
            event.tool_use["input"] = {"changed_after_resume": True}
        elif self.mutation == "cancel-after-resume":
            event.cancel_tool = "hook cancelled"
        elif self.mutation == "provenance-delete":
            event.agent.state.delete("__ag_ui_proxy_hook_provenance__")
        elif self.mutation == "resume-value-in-place":
            working = event.invocation_state["__ag_ui_proxy_resume_results__"]
            result = working[event.tool_use["toolUseId"]]
            object.__setattr__(result, "content", "HOOK-TAMPERED")
        elif self.mutation == "resume-binding-in-place":
            working = event.invocation_state[
                "__ag_ui_proxy_resume_result_bindings__"
            ]
            working[event.tool_use["toolUseId"]] = "hook-tampered-wire"


class _AsyncPerCallAliasHook(HookProvider):
    """Give concurrent equal proxies distinct aliases after an async yield."""

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    @staticmethod
    async def _before_tool_call(event):
        if event.tool_use["name"] != "approveTool":
            return
        native_id = event.tool_use["toolUseId"]
        await asyncio.sleep(0)
        event.tool_use["toolUseId"] = f"hook-{native_id}"
        event.interrupt(f"approve_{native_id}", reason="Approve frontend action")


class _ConcurrentProxySubsetHook(HookProvider):
    """Interrupt equal proxy calls under aliases unrelated to native identity."""

    aliases = {
        "native-proxy-a": ("hook-alias-a", "proxy_a"),
        "native-proxy-b": ("hook-alias-b", "proxy_b"),
    }

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    async def _before_tool_call(self, event):
        if event.tool_use["name"] != "approveTool":
            return
        alias, interrupt_name = self.aliases[event.tool_use["toolUseId"]]
        await asyncio.sleep(0)
        event.tool_use["toolUseId"] = alias
        event.interrupt(interrupt_name, reason="Approve frontend action")


class _DelayedSecondProxyObserverHook(HookProvider):
    """Let A clean before B's resumed BeforeToolCall finalizer executes."""

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.a_cleaned = asyncio.Event()
        self.after_snapshots: list[tuple[str, set[str], set[str]]] = []

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)
        registry.add_callback(AfterToolCallEvent, self._after_tool_call)

    async def _before_tool_call(self, event):
        native_id = event.tool_use["toolUseId"]
        self.calls[native_id] = self.calls.get(native_id, 0) + 1
        if native_id == "native-proxy-b" and self.calls[native_id] > 1:
            await self.a_cleaned.wait()
        event.interrupt(
            f"proxy_{native_id}", reason=f"Approve {native_id}"
        )

    def _after_tool_call(self, event):
        native_id = event.tool_use["toolUseId"]
        results = event.invocation_state["__ag_ui_proxy_resume_results__"]
        bindings = event.invocation_state[
            "__ag_ui_proxy_resume_result_bindings__"
        ]
        self.after_snapshots.append(
            (native_id, set(results), set(bindings))
        )
        if native_id == "native-proxy-a":
            self.a_cleaned.set()


class _NamedProxyInterruptHook(HookProvider):
    """Pause each distinct A/B frontend proxy under a stable hook name."""

    interrupt_by_tool = {
        "approveA": "proxy_a",
        "approveB": "proxy_b",
    }

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event):
        interrupt_name = self.interrupt_by_tool.get(event.tool_use["name"])
        if interrupt_name is not None:
            event.interrupt(interrupt_name, reason="Approve frontend action")


class _FixedProxyInterruptHook(HookProvider):
    """Add one independently answerable hook interrupt to approveTool."""

    def __init__(self, interrupt_name: str):
        self.interrupt_name = interrupt_name

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event):
        if event.tool_use["name"] == "approveTool":
            event.interrupt(
                self.interrupt_name,
                reason=f"Approve {self.interrupt_name}",
            )


class _PendingStateTamperHook(HookProvider):
    """Delete the staged result only after stage-two returns on resume."""

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    @staticmethod
    def _before_tool_call(event):
        if event.tool_use["name"] != "approveTool":
            return
        event.interrupt("stage_two", reason="Approve stage_two")
        event.agent.state.delete(AG_UI_PENDING_PROXY_RESULTS_STATE_KEY)


class _SameAliasHook(HookProvider):
    """Force two different proxy calls onto the same caller-selected alias."""

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    @staticmethod
    def _before_tool_call(event):
        if event.tool_use["name"] != "approveTool":
            return
        event.tool_use["toolUseId"] = "shared-hook-alias"
        event.interrupt("approve_shared", reason="Approve frontend action")


class _PreemptiveThenPendingInterruptHook(HookProvider):
    """Pre-answer one proxy call, then pause the next proxy call normally."""

    def __init__(self):
        self.calls = 0

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event):
        if event.tool_use["name"] != "approveTool":
            return
        self.calls += 1
        if self.calls == 1:
            assert (
                event.interrupt(
                    "approve_proxy",
                    reason="Approve frontend action",
                    response=True,
                )
                is True
            )
            event.tool_use["input"] = {"preemptive": True}
            return
        event.interrupt("approve_proxy", reason="Approve frontend action")


class _CaughtInterruptThenMutateHook(HookProvider):
    """Catch a candidate interrupt and continue with a legal mutation."""

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    @staticmethod
    def _before_tool_call(event):
        if event.tool_use["name"] != "approveTool":
            return
        try:
            event.interrupt("caught_proxy", reason="caught")
        except InterruptException:
            pass
        event.tool_use["input"] = {"caught": True}


class _RetainingNonInterruptingProxyHook(HookProvider):
    """Retain the event so the temporary observer can be inspected."""

    def __init__(self):
        self.event = None

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event):
        if event.tool_use["name"] == "approveTool":
            self.event = event


class _InterruptThenReplaceWithoutRecheckingHook(HookProvider):
    """Pause initially, then try to bypass the checkpoint on resume."""

    def __init__(self):
        self.calls = 0

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event):
        if event.tool_use["name"] != "approveTool":
            return
        self.calls += 1
        if self.calls == 1:
            event.interrupt("approve_proxy", reason="Approve frontend action")
        else:
            event.selected_tool = event.agent.tool_registry.registry["replacementTool"]


class _RaiseOnSecondProxyHook(HookProvider):
    """Raise after the managed hook returns its resume response."""

    def __init__(self):
        self.calls = 0

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event):
        if event.tool_use["name"] != "approveTool":
            return
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("later static hook failed")


class _LateRegisteredInterruptHook(HookProvider):
    """Register a proxy callback during Strands agent initialization."""

    def register_hooks(self, registry):
        registry.add_callback(AgentInitializedEvent, self._initialized)

    @staticmethod
    def _initialized(event):
        def initialized_interrupt(tool_event):
            if tool_event.tool_use["name"] == "approveTool":
                tool_event.interrupt("initialized_proxy", reason="initialized")

        event.agent.hooks.add_callback(BeforeToolCallEvent, initialized_interrupt)


class _CrossSiblingMapTamperHook(HookProvider):
    """Corrupt the first finalized sibling while handling the second."""

    def __init__(self):
        self.calls = 0

    def register_hooks(self, registry):
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event):
        if event.tool_use["name"] != "approveTool":
            return
        self.calls += 1
        native_id = event.tool_use["toolUseId"]
        if self.calls == 2:
            wire_map = event.agent.state.get(AG_UI_WIRE_MAP_STATE_KEY) or {}
            first_wire = next(
                wire
                for wire, mapped_native in wire_map.items()
                if mapped_native != native_id
            )
            wire_map[first_wire] = "corrupted-native"
            event.agent.state.set(AG_UI_WIRE_MAP_STATE_KEY, wire_map)
        event.interrupt(f"approve_{native_id}", reason="approve")


_replacement_tool_calls: list[dict] = []


@tool
def replacementTool(**kwargs) -> str:
    """Record replacement execution; provenance failures must never call it."""
    _replacement_tool_calls.append(kwargs)
    return "SERVER-REPLACEMENT"


def _proxy_hook_agent(
    model,
    *,
    hook,
    config: StrandsAgentConfig,
    include_replacement: bool = False,
) -> StrandsAgent:
    core = StrandsAgentCore(
        model=model,
        tools=[replacementTool] if include_replacement else None,
        system_prompt="test",
        agent_id="proxy-hook-agent",
    )
    return StrandsAgent(
        core,
        name="proxy-hook-provenance",
        config=config,
        hooks=[hook],
    )


def _proxy_wire_and_interrupt(events):
    wire_id = next(
        event.tool_call_id
        for event in events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )
    finished = next(event for event in events if event.type == EventType.RUN_FINISHED)
    return wire_id, finished.outcome.interrupts[0]


@pytest.mark.asyncio
async def test_no_session_frontend_failure_replay_preserves_blank_diagnostic():
    """Default history replay must retain why a frontend action failed."""
    model = _ProxyHookInterruptFlowModel()
    core = StrandsAgentCore(model=model, system_prompt="test")
    agent = StrandsAgent(core, name="no-session-frontend-failure")
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )

    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id = next(
        event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                UserMessage(id="u1", role="user", content="approve"),
                AssistantMessage(
                    id="a1",
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=wire_id,
                            type="function",
                            function=FunctionCall(
                                name="approveTool", arguments="{}"
                            ),
                        )
                    ],
                ),
                ToolMessage(
                    id="t-failed",
                    role="tool",
                    tool_call_id=wire_id,
                    content="",
                    error="boom",
                ),
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(
        event.type == EventType.RUN_ERROR for event in resumed_events
    )
    frontend_result = next(
        block["toolResult"]
        for message in model.stream_calls_messages[-1]
        for block in message.get("content", [])
        if block.get("toolResult", {}).get("toolUseId") == wire_id
    )
    assert frontend_result == {
        "toolUseId": wire_id,
        "content": [{"text": "boom"}],
        "status": "error",
    }
    assert "Forwarded to client" not in json.dumps(
        model.stream_calls_messages[-1]
    )
    live_core = agent._agents_by_thread["thread-1"]
    assert AG_UI_PENDING_PROXY_RESULTS_STATE_KEY not in live_core.state.get()


def _make_e2e_agent(config: StrandsAgentConfig) -> tuple[StrandsAgent, _InterruptFlowModel]:
    model = _InterruptFlowModel()
    core = StrandsAgentCore(model=model, tools=[confirm_action], system_prompt="test")
    return StrandsAgent(core, name="e2e-interrupt", config=config), model


def _snapshot_resume_state(core: StrandsAgentCore) -> dict:
    """Capture every mutable surface an invalid resume must leave untouched."""
    interrupt_state = core._interrupt_state
    session_manager = getattr(core, "session_manager", None) or getattr(
        core, "_session_manager", None
    )
    repository = getattr(session_manager, "session_repository", None)
    repository_messages = None
    if repository is not None:
        repository_messages = copy.deepcopy(
            repository.list_messages(session_manager.session_id, core.agent_id)
        )

    return {
        "interrupt_responses": {
            interrupt_id: copy.deepcopy(interrupt.response)
            for interrupt_id, interrupt in interrupt_state.interrupts.items()
        },
        "interrupt_context": copy.deepcopy(interrupt_state.context),
        "interrupts": copy.deepcopy(interrupt_state.interrupts),
        "registry": {
            name: copy.deepcopy(tool.tool_spec)
            for name, tool in core.tool_registry.registry.items()
        },
        "messages": copy.deepcopy(core.messages),
        "repository_messages": repository_messages,
        "wire_map": copy.deepcopy(core.state.get(AG_UI_WIRE_MAP_STATE_KEY)),
        "tool_map": copy.deepcopy(core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)),
        "agent_state": copy.deepcopy(core.state.get()),
    }


async def _start_concurrent_proxy_hook_checkpoint(tmp_path):
    """Create a durable two-proxy hook checkpoint through real Strands."""
    model = _ConcurrentProxyHookFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=_ConcurrentProxySubsetHook(),
        config=config,
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="both")],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in initial_events)
    finished = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    )
    assert finished.outcome is not None
    assert len(finished.outcome.interrupts) == 2
    live_core = agent._agents_by_thread["thread-1"]
    assert isinstance(live_core.tool_executor, ConcurrentToolExecutor)
    provenance = live_core.state.get("__ag_ui_proxy_hook_provenance__")
    calls_by_native = {
        provenance["records"][interrupt.id]["original_native_tool_call_id"]: {
            "interrupt": interrupt,
            "wire_id": provenance["records"][interrupt.id][
                "wire_tool_call_id"
            ],
            "record": copy.deepcopy(provenance["records"][interrupt.id]),
        }
        for interrupt in finished.outcome.interrupts
    }
    assert set(calls_by_native) == set(model.native_ids)
    return agent, model, live_core, calls_by_native


def _answered_proxy_tools() -> dict[str, Tool]:
    return {
        name: Tool(name=name, description=f"approve {name}", parameters={})
        for name in ("approveA", "approveB")
    }


def _answered_proxy_hook_agent(
    model: _AnsweredProxyHookFlowModel,
    storage_dir,
) -> StrandsAgent:
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(storage_dir)
        ),
    )
    core = StrandsAgentCore(
        model=model,
        system_prompt="test",
        agent_id="proxy-hook-subset-agent",
    )
    return StrandsAgent(
        core,
        name="answered-proxy-hook",
        config=config,
        hooks=[_NamedProxyInterruptHook()],
    )


async def _start_answered_proxy_hook_checkpoint(tmp_path):
    """Run A/B through the first pause and consume only A on run two."""
    model = _AnsweredProxyHookFlowModel()
    agent = _answered_proxy_hook_agent(model, tmp_path)
    tools = _answered_proxy_tools()
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve both")],
            tools=list(tools.values()),
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in initial_events)
    initial_finished = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    )
    interrupts = {
        interrupt.metadata["strands_name"]: interrupt
        for interrupt in initial_finished.outcome.interrupts
    }
    wire_ids = {
        event.tool_call_name: event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name in tools
    }
    assert set(interrupts) == {"proxy_a", "proxy_b"}
    assert set(wire_ids) == set(tools)

    partial_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-a",
                    role="tool",
                    tool_call_id=wire_ids["approveA"],
                    content="CLIENT-A-RESULT",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["proxy_a"].id,
                    status="resolved",
                    payload=False,
                )
            ],
            tools=list(tools.values()),
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in partial_events)
    partial_finished = next(
        event for event in partial_events if event.type == EventType.RUN_FINISHED
    )
    assert [
        interrupt.metadata["strands_name"]
        for interrupt in partial_finished.outcome.interrupts
    ] == ["proxy_b"]

    live_core = agent._agents_by_thread["thread-1"]
    live_interrupts = live_core._interrupt_state.interrupts
    assert set(live_interrupts) == {
        interrupts["proxy_a"].id,
        interrupts["proxy_b"].id,
    }
    assert live_interrupts[interrupts["proxy_a"].id].response == {
        "response": False
    }
    assert not live_interrupts[interrupts["proxy_b"].id].response
    assert live_core._interrupt_state.activated

    stored = FileSessionManager(
        session_id="thread-1", storage_dir=str(tmp_path)
    ).read_agent("thread-1", "proxy-hook-subset-agent")
    assert stored is not None
    stored_interrupt_state = stored._internal_state["interrupt_state"]
    stored_interrupts = stored_interrupt_state["interrupts"]
    assert set(stored_interrupts) == set(live_interrupts)
    assert stored_interrupts[interrupts["proxy_a"].id]["response"] == {
        "response": False
    }
    assert not stored_interrupts[interrupts["proxy_b"].id]["response"]
    assert stored_interrupt_state["activated"] is True

    return agent, model, tools, interrupts, wire_ids


async def _finish_answered_proxy_b(
    agent: StrandsAgent,
    model: _AnsweredProxyHookFlowModel,
    tool_b: Tool,
    interrupt_b,
    wire_b: str,
) -> None:
    """Resume only pending B and assert an ordinary terminal completion."""
    finished_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            messages=[
                ToolMessage(
                    id="t-b",
                    role="tool",
                    tool_call_id=wire_b,
                    content="CLIENT-B-RESULT",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt_b.id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[tool_b],
        ),
    )

    errors = [
        (event.code, event.message)
        for event in finished_events
        if event.type == EventType.RUN_ERROR
    ]
    assert errors == []
    run_finished = [
        event for event in finished_events if event.type == EventType.RUN_FINISHED
    ]
    assert len(run_finished) == 1
    assert run_finished[0].outcome is None
    assert model.turn == 2
    model_messages = json.dumps(model.stream_calls_messages[-1])
    assert model_messages.count("CLIENT-B-RESULT") == 1
    assert "Forwarded to client" not in model_messages
    encoded_events = "".join(
        EventEncoder().encode(event) for event in finished_events
    )
    assert "Forwarded to client" not in encoded_events


def _assert_proxy_reconciliation_error(events: list) -> None:
    errors = [event for event in events if event.type == EventType.RUN_ERROR]
    assert len(errors) == 1
    assert errors[0].code == "INTERRUPT_RECONCILIATION_ERROR"
    assert not any(event.type == EventType.RUN_FINISHED for event in events)


def _assert_atomic_resume_error(events: list) -> None:
    errors = [event for event in events if event.type == EventType.RUN_ERROR]
    assert len(errors) == 1
    assert errors[0].code == "INTERRUPT_RESUME_ERROR"
    assert not any(event.type == EventType.RUN_FINISHED for event in events)


@pytest.mark.asyncio
async def test_real_interrupt_metadata_mapping_keys_survive_event_encoding():
    core = StrandsAgentCore(
        model=_MetadataInterruptFlowModel(),
        tools=[interrupt_with_colliding_metadata],
        system_prompt="test",
    )
    agent = StrandsAgent(core, name="metadata-collision")

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="interrupt")]
        ),
    )

    finished = next(event for event in events if event.type == EventType.RUN_FINISHED)
    encoded = EventEncoder().encode(finished)
    payload = json.loads(encoded.removeprefix("data: ").removesuffix("\n\n"))
    [wire_interrupt] = payload["outcome"]["interrupts"]
    assert wire_interrupt["metadata"]["strands_reason"] == {
        "__ag_ui_key_v1__:s:YmFk7bO_a2V5": "surrogate-value",
        "bad\\udcffkey": "literal-value",
        (
            "__ag_ui_key_v1__:v:"
            "X19hZ191aV9rZXlfdjFfXzpsaXRlcmFs"
        ): "reserved-value",
    }


@pytest.mark.asyncio
async def test_resume_preflight_rejects_valid_then_stale_without_mutation():
    """A later stale id cannot let Strands consume an earlier valid response."""
    model = _NativeInterruptFlowModel()
    core = StrandsAgentCore(model=model, tools=[confirm_action], system_prompt="test")
    agent = StrandsAgent(core, name="atomic-native-resume")
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="confirm widget-1")]
        ),
    )
    interrupt_id = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    ).outcome.interrupts[0].id
    live_core = agent._agents_by_thread["thread-1"]
    before = _snapshot_resume_state(live_core)

    with patch.object(
        live_core, "stream_async", wraps=live_core.stream_async
    ) as stream_spy:
        invalid_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupt_id,
                        status="resolved",
                        payload=True,
                    ),
                    ResumeEntry(
                        interrupt_id="stale-interrupt-id",
                        status="resolved",
                        payload=False,
                    ),
                ],
            ),
        )

    stream_spy.assert_not_called()
    _assert_atomic_resume_error(invalid_events)
    assert _snapshot_resume_state(live_core) == before


@pytest.mark.asyncio
async def test_resume_preflight_rejects_duplicate_ids_before_mixed_session_mutation(
    tmp_path,
):
    """Duplicate responses fail before proxy sync or repository reconciliation."""
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent, _ = _make_e2e_agent(config)
    approve_tool = Tool(name="approveTool", description="approve", parameters={})
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="handle widget-1")],
            tools=[approve_tool],
        ),
    )
    finished = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    )
    interrupt_id = finished.outcome.interrupts[0].id
    wire_id = next(
        event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )
    live_core = agent._agents_by_thread["thread-1"]
    before = _snapshot_resume_state(live_core)
    unexpected_tool = Tool(
        name="unexpectedTool", description="must not be registered", parameters={}
    )

    with patch.object(
        live_core, "stream_async", wraps=live_core.stream_async
    ) as stream_spy:
        invalid_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                messages=[
                    ToolMessage(
                        id="t-approve",
                        role="tool",
                        tool_call_id=wire_id,
                        content='{"approved": true}',
                    )
                ],
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupt_id,
                        status="resolved",
                        payload=True,
                    ),
                    ResumeEntry(
                        interrupt_id=interrupt_id,
                        status="cancelled",
                        payload=None,
                    ),
                ],
                tools=[approve_tool, unexpected_tool],
            ),
        )

    stream_spy.assert_not_called()
    _assert_atomic_resume_error(invalid_events)
    assert _snapshot_resume_state(live_core) == before


@pytest.mark.asyncio
async def test_resume_preflight_rejects_nonempty_resume_after_interrupt_state_is_lost():
    """Inactive Strands state cannot silently accept a stale resume batch."""
    model = _NativeInterruptFlowModel()
    core = StrandsAgentCore(model=model, tools=[confirm_action], system_prompt="test")
    agent = StrandsAgent(core, name="lost-native-resume")
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="confirm widget-1")]
        ),
    )
    interrupt_id = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    ).outcome.interrupts[0].id
    live_core = agent._agents_by_thread["thread-1"]
    live_core._interrupt_state.deactivate()
    before = _snapshot_resume_state(live_core)

    with patch.object(
        live_core, "stream_async", wraps=live_core.stream_async
    ) as stream_spy:
        invalid_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupt_id,
                        status="resolved",
                        payload=True,
                    )
                ],
            ),
        )

    stream_spy.assert_not_called()
    _assert_atomic_resume_error(invalid_events)
    assert _snapshot_resume_state(live_core) == before


@pytest.mark.asyncio
async def test_frontend_proxy_before_tool_hook_resume_retains_omitted_tool_once(
    tmp_path,
):
    """A hook checkpoint can finish after the client omits its proxy spec."""
    model = _ProxyHookInterruptFlowModel()
    core = StrandsAgentCore(model=model, system_prompt="test")
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = StrandsAgent(
        core,
        name="proxy-hook-interrupt",
        config=config,
        hooks=[_InterruptFrontendProxyHook()],
    )
    frontend_tool = Tool(name="approveTool", description="approve", parameters={})

    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )

    wire_id = next(
        event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )
    finished = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    )
    interrupt = finished.outcome.interrupts[0]
    live_core = agent._agents_by_thread["thread-1"]
    assert "approveTool" in live_core.tool_registry.registry
    parked_context = copy.deepcopy(live_core._interrupt_state.context)
    parked_interrupts = copy.deepcopy(live_core._interrupt_state.interrupts)

    with patch.object(
        live_core, "stream_async", wraps=live_core.stream_async
    ) as stream_spy:
        incomplete_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupt.id,
                        status="resolved",
                        payload=True,
                    )
                ],
                tools=[frontend_tool],
            ),
        )

    stream_spy.assert_not_called()
    assert [
        event.code
        for event in incomplete_events
        if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_RECONCILIATION_ERROR"]
    assert live_core._interrupt_state.context == parked_context
    assert live_core._interrupt_state.interrupts == parked_interrupts

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            messages=[
                ToolMessage(
                    id="t-approve",
                    role="tool",
                    tool_call_id=wire_id,
                    content='{"approved": true}',
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt.id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[],
        ),
    )

    resumed_model_messages = json.dumps(model.stream_calls_messages[-1])
    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    resumed_finished = [
        event for event in resumed_events if event.type == EventType.RUN_FINISHED
    ]
    assert len(resumed_finished) == 1
    assert resumed_finished[0].outcome is None
    assert not any(
        getattr(event, "tool_call_id", None) == "native-approve"
        for event in initial_events
    )
    assert (
        interrupt.tool_call_id,
        resumed_model_messages.count("approved"),
        "Forwarded to client" in resumed_model_messages,
    ) == (wire_id, 1, False)
    assert "Unknown tool" not in resumed_model_messages
    assert live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY) == {}
    assert "approveTool" in live_core.tool_registry.registry
    assert agent._proxy_tool_names_by_thread["thread-1"] == {"approveTool"}

    cleanup_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-4",
            messages=[UserMessage(id="u-cleanup", role="user", content="continue")],
            tools=[],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in cleanup_events)
    assert "approveTool" not in live_core.tool_registry.registry


@pytest.mark.asyncio
async def test_frontend_proxy_hook_resume_preserves_failed_result(tmp_path):
    """A failed client result reaches the model as an error, not proxy success."""
    model = _ProxyHookInterruptFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = StrandsAgent(
        StrandsAgentCore(model=model, system_prompt="test"),
        name="proxy-hook-failure",
        config=config,
        hooks=[_InterruptFrontendProxyHook()],
    )
    frontend_tool = Tool(name="approveTool", description="approve", parameters={})

    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id = next(
        event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )
    interrupt_id = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    ).outcome.interrupts[0].id

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-approve",
                    role="tool",
                    tool_call_id=wire_id,
                    content="",
                    error="boom",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt_id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    frontend_result = next(
        block["toolResult"]
        for message in model.stream_calls_messages[-1]
        for block in message.get("content", [])
        if block.get("toolResult", {}).get("toolUseId") == "native-approve"
    )
    assert frontend_result == {
        "toolUseId": "native-approve",
        "status": "error",
        "content": [{"text": "boom"}],
    }
    assert "Forwarded to client" not in json.dumps(model.stream_calls_messages[-1])


@pytest.mark.asyncio
async def test_recreated_proxy_hook_resume_without_tool_spec_fails_capability(
    tmp_path,
):
    """Persisted identity cannot safely recreate an omitted proxy schema."""
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    frontend_tool = Tool(name="approveTool", description="approve", parameters={})
    initial_model = _ProxyHookInterruptFlowModel()
    agent = StrandsAgent(
        StrandsAgentCore(model=initial_model, system_prompt="test"),
        name="proxy-hook-interrupt",
        config=config,
        hooks=[_InterruptFrontendProxyHook()],
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id = next(
        event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )
    interrupt_id = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    ).outcome.interrupts[0].id

    resumed_model = _ProxyHookInterruptFlowModel()
    resumed_model.turn = initial_model.turn
    recreated_agent = StrandsAgent(
        StrandsAgentCore(model=resumed_model, system_prompt="test"),
        name="proxy-hook-interrupt",
        config=config,
        hooks=[_InterruptFrontendProxyHook()],
    )
    resumed_events = await _collect_events(
        recreated_agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-approve",
                    role="tool",
                    tool_call_id=wire_id,
                    content='{"approved": true}',
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt_id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[],
        ),
    )

    errors = [event for event in resumed_events if event.type == EventType.RUN_ERROR]
    assert len(errors) == 1
    assert errors[0].code == "INTERRUPT_SESSION_CAPABILITY_ERROR"
    assert "marked proxy tool specification is unavailable" in errors[0].message
    assert not any(event.type == EventType.RUN_FINISHED for event in resumed_events)
    assert resumed_model.turn == initial_model.turn
    recreated_core = recreated_agent._agents_by_thread["thread-1"]
    assert "approveTool" not in recreated_core.tool_registry.registry
    assert recreated_core._interrupt_state.activated


@pytest.mark.asyncio
async def test_native_interrupt_resumes_without_session_manager_and_restores_behaviors():
    state_contexts = []
    custom_contexts = []

    def state_from_result(ctx):
        state_contexts.append(ctx)
        return {"confirmed_key": ctx.tool_input["key"]}

    async def custom_result_handler(ctx):
        custom_contexts.append(ctx)
        if False:
            yield  # pragma: no cover — async-generator contract

    model = _NativeInterruptFlowModel()
    core = StrandsAgentCore(model=model, tools=[confirm_action], system_prompt="test")
    config = StrandsAgentConfig(
        tool_behaviors={
            "confirm_action": ToolBehavior(
                state_from_result=state_from_result,
                custom_result_handler=custom_result_handler,
            )
        }
    )
    agent = StrandsAgent(core, name="live-native-interrupt", config=config)

    events1 = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="confirm widget-1")]
        ),
    )

    finished1 = next(e for e in events1 if e.type == EventType.RUN_FINISHED)
    assert finished1.outcome is not None
    assert finished1.outcome.type == "interrupt"
    interrupt = finished1.outcome.interrupts[0]
    assert interrupt.tool_call_id == "native-confirm"

    events2 = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt.id,
                    status="resolved",
                    payload=True,
                )
            ],
        ),
    )

    assert not any(e.type == EventType.RUN_ERROR for e in events2)
    finished2 = [e for e in events2 if e.type == EventType.RUN_FINISHED]
    assert len(finished2) == 1
    assert finished2[0].outcome is None
    tool_result_index = next(
        index
        for index, event in enumerate(events2)
        if event.type == EventType.TOOL_CALL_RESULT
        and event.tool_call_id == "native-confirm"
    )
    assert tool_result_index < events2.index(finished2[0])
    assert len(state_contexts) == 1
    assert len(custom_contexts) == 1
    for ctx in [state_contexts[0], custom_contexts[0]]:
        assert ctx.tool_name == "confirm_action"
        assert ctx.tool_use_id == "native-confirm"
        assert ctx.tool_input == {"key": "widget-1"}
        assert ctx.args_str == '{"key": "widget-1"}'


@pytest.mark.asyncio
async def test_live_native_resume_without_client_history_retains_cached_history():
    model = _NativeInterruptFlowModel()
    core = StrandsAgentCore(model=model, tools=[confirm_action], system_prompt="test")
    agent = StrandsAgent(core, name="live-native-history")

    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="confirm widget-1")]
        ),
    )
    interrupt = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    ).outcome.interrupts[0]
    live_core = agent._agents_by_thread["thread-1"]
    cached_history = copy.deepcopy(live_core.messages)

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt.id,
                    status="resolved",
                    payload=True,
                )
            ],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    resumed_history = model.stream_calls_messages[-1]
    assert resumed_history[: len(cached_history)] == cached_history
    assert resumed_history[-1]["content"][0]["toolResult"]["toolUseId"] == (
        "native-confirm"
    )


@pytest.mark.asyncio
async def test_live_native_resume_with_full_client_history_replaces_cached_history():
    model = _NativeInterruptFlowModel()
    core = StrandsAgentCore(model=model, tools=[confirm_action], system_prompt="test")
    agent = StrandsAgent(core, name="live-native-authoritative-history")

    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="cached prompt")]
        ),
    )
    interrupt = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    ).outcome.interrupts[0]
    authoritative_messages = [
        UserMessage(id="u-client", role="user", content="authoritative client prompt"),
        AssistantMessage(
            id="a-client",
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="native-confirm",
                    type="function",
                    function=FunctionCall(
                        name="confirm_action",
                        arguments='{"key": "widget-1"}',
                    ),
                )
            ],
        ),
    ]

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=authoritative_messages,
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt.id,
                    status="resolved",
                    payload=True,
                )
            ],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    resumed_history = model.stream_calls_messages[-1]
    assert resumed_history[0]["content"] == [{"text": "authoritative client prompt"}]
    assert "cached prompt" not in json.dumps(resumed_history)


@pytest.mark.asyncio
async def test_native_interrupt_resumes_with_non_repository_session_manager():
    """Native-only resume accepts the public SessionManager ABC contract."""
    session_manager = _NonRepositorySessionManager()
    model = _NativeInterruptFlowModel()
    core = StrandsAgentCore(model=model, tools=[confirm_action], system_prompt="test")
    agent = StrandsAgent(
        core,
        name="native-abc-session-manager",
        config=StrandsAgentConfig(
            session_manager_provider=lambda input_data: session_manager
        ),
    )

    with patch(
        "ag_ui_strands.agent.reconcile_frontend_tool_results",
        side_effect=AssertionError(
            "native-only resume must not reconcile a repository"
        ),
    ) as reconcile_spy:
        initial_events = await _collect_events(
            agent,
            _make_run_input(
                messages=[UserMessage(id="u1", role="user", content="confirm widget-1")]
            ),
        )
        initial_finished = next(
            event for event in initial_events if event.type == EventType.RUN_FINISHED
        )
        interrupt = initial_finished.outcome.interrupts[0]

        resumed_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupt.id,
                        status="resolved",
                        payload=True,
                    )
                ],
            ),
        )

    reconcile_spy.assert_not_called()
    assert not any(
        event.type == EventType.RUN_ERROR for event in initial_events + resumed_events
    )
    assert any(event.type == EventType.RUN_FINISHED for event in resumed_events)
    assert model.turn == 2


@pytest.mark.asyncio
async def test_mixed_interrupt_requires_repository_capability_before_outcome_or_resume():
    """Mixed proxy/native state fails closed for a non-repository manager."""
    session_manager = _NonRepositorySessionManager()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: session_manager
    )
    agent, model = _make_e2e_agent(config)
    approve_tool = Tool(name="approveTool", description="approve", parameters={})

    with patch(
        "ag_ui_strands.agent.reconcile_frontend_tool_results",
        side_effect=AssertionError("unsafe repository reconciliation was attempted"),
    ) as reconcile_spy:
        initial_events = await _collect_events(
            agent,
            _make_run_input(
                messages=[
                    UserMessage(
                        id="u1",
                        role="user",
                        content="please handle widget-1",
                    )
                ],
                tools=[approve_tool],
            ),
        )

        initial_errors = [
            event for event in initial_events if event.type == EventType.RUN_ERROR
        ]
        assert len(initial_errors) == 1
        assert initial_errors[0].code == "INTERRUPT_SESSION_CAPABILITY_ERROR"
        assert "repository reconciliation" in initial_errors[0].message
        assert not any(event.type == EventType.RUN_FINISHED for event in initial_events)

        core = agent._agents_by_thread["thread-1"]
        interrupt_state = core._interrupt_state
        parked_context = copy.deepcopy(interrupt_state.context)
        parked_interrupts = copy.deepcopy(interrupt_state.interrupts)
        parked_messages = copy.deepcopy(core.messages)
        wire_map = copy.deepcopy(core.state.get(AG_UI_WIRE_MAP_STATE_KEY))
        tool_metadata = copy.deepcopy(core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY))
        frontend_wire_id = next(
            event.tool_call_id
            for event in initial_events
            if event.type == EventType.TOOL_CALL_START
            and event.tool_call_name == "approveTool"
        )
        interrupt_id = next(iter(parked_interrupts))

        with patch.object(core, "stream_async", wraps=core.stream_async) as stream_spy:
            resumed_events = await _collect_events(
                agent,
                _make_run_input(
                    run_id="run-2",
                    messages=[
                        ToolMessage(
                            id="t-fe",
                            role="tool",
                            tool_call_id=frontend_wire_id,
                            content='{"approved": true}',
                        )
                    ],
                    resume=[
                        ResumeEntry(
                            interrupt_id=interrupt_id,
                            status="resolved",
                            payload=True,
                        )
                    ],
                    tools=[approve_tool],
                ),
            )

    reconcile_spy.assert_not_called()
    stream_spy.assert_not_called()
    resumed_errors = [
        event for event in resumed_events if event.type == EventType.RUN_ERROR
    ]
    assert len(resumed_errors) == 1
    assert resumed_errors[0].code == "INTERRUPT_SESSION_CAPABILITY_ERROR"
    assert not any(event.type == EventType.RUN_FINISHED for event in resumed_events)
    assert model.turn == 1
    assert interrupt_state.activated
    assert interrupt_state.context == parked_context
    assert interrupt_state.interrupts == parked_interrupts
    assert core.messages == parked_messages
    assert core.state.get(AG_UI_WIRE_MAP_STATE_KEY) == wire_map
    assert core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY) == tool_metadata


@pytest.mark.asyncio
async def test_native_placeholder_text_with_interrupt_does_not_require_session_manager():
    state_contexts = []
    custom_contexts = []

    def state_from_result(ctx):
        state_contexts.append(ctx)
        return {"native_placeholder_result": ctx.result_data}

    async def custom_result_handler(ctx):
        custom_contexts.append(ctx)
        yield CustomEvent(
            type=EventType.CUSTOM,
            name="native-placeholder-result",
            value={"content": ctx.result_data},
        )

    model = _InterruptFlowModel(sibling_tool_name="native_placeholder")
    core = StrandsAgentCore(
        model=model,
        tools=[native_placeholder, confirm_action],
        system_prompt="test",
    )
    agent = StrandsAgent(
        core,
        name="native-placeholder-interrupt",
        config=StrandsAgentConfig(
            tool_behaviors={
                "native_placeholder": ToolBehavior(
                    state_from_result=state_from_result,
                    custom_result_handler=custom_result_handler,
                )
            }
        ),
    )
    colliding_declaration = Tool(
        name="native_placeholder",
        description="frontend declaration colliding with a native tool",
        parameters={},
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[
                UserMessage(
                    id="u1",
                    role="user",
                    content="run both native tools",
                )
            ],
            tools=[colliding_declaration],
        ),
    )

    live_core = agent._agents_by_thread["thread-1"]
    assert live_core._interrupt_state.context["tool_results"] == [
        {
            "toolUseId": "native-approve",
            "status": "success",
            "content": [{"text": "Forwarded to client"}],
        }
    ]
    assert [
        event.code for event in events if event.type == EventType.RUN_ERROR
    ] == []
    finished = next(event for event in events if event.type == EventType.RUN_FINISHED)
    assert finished.outcome is not None
    assert finished.outcome.type == "interrupt"
    assert finished.outcome.interrupts[0].tool_call_id == "native-confirm"
    interrupt_id = finished.outcome.interrupts[0].id

    tool_metadata = live_core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)
    assert tool_metadata["native-approve"]["is_frontend"] is False
    assert tool_metadata["native-confirm"]["is_frontend"] is False

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt_id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[colliding_declaration],
        ),
    )

    native_results = [
        event
        for event in resumed_events
        if event.type == EventType.TOOL_CALL_RESULT
        and event.tool_call_id == "native-approve"
    ]
    assert len(native_results) == 1
    assert native_results[0].content == '"Forwarded to client"'
    custom_event_indices = [
        index
        for index, event in enumerate(resumed_events)
        if event.type == EventType.CUSTOM
        and event.name == "native-placeholder-result"
    ]
    assert len(custom_event_indices) == 1
    custom_event_index = custom_event_indices[0]
    assert [
        event.snapshot
        for event in resumed_events[:custom_event_index]
        if event.type == EventType.STATE_SNAPSHOT
        and event.snapshot.get("native_placeholder_result")
        == "Forwarded to client"
    ] == [{"native_placeholder_result": "Forwarded to client"}]
    assert [
        event.value
        for event in resumed_events
        if event.type == EventType.CUSTOM
        and event.name == "native-placeholder-result"
    ] == [{"content": "Forwarded to client"}]
    assert len(state_contexts) == 1
    assert len(custom_contexts) == 1
    for ctx in [state_contexts[0], custom_contexts[0]]:
        assert ctx.tool_name == "native_placeholder"
        assert ctx.tool_use_id == "native-approve"
        assert ctx.tool_input == {}
        assert ctx.args_str == "{}"
        assert ctx.result_data == "Forwarded to client"
    assert not any(
        event.type == EventType.RUN_ERROR for event in resumed_events
    )
    assert any(
        event.type == EventType.RUN_FINISHED for event in resumed_events
    )


@pytest.mark.asyncio
async def test_mixed_interrupt_without_session_manager_errors_before_outcome():
    agent, model = _make_e2e_agent(StrandsAgentConfig())
    approve_tool = Tool(name="approveTool", description="approve", parameters={})

    events1 = await _collect_events(
        agent,
        _make_run_input(
            messages=[
                UserMessage(
                    id="u1",
                    role="user",
                    content="please handle widget-1",
                )
            ],
            tools=[approve_tool],
        ),
    )

    errors1 = [event for event in events1 if event.type == EventType.RUN_ERROR]
    assert len(errors1) == 1
    assert errors1[0].code == "INTERRUPT_SESSION_REQUIRED"
    assert not any(event.type == EventType.RUN_FINISHED for event in events1)

    core = agent._agents_by_thread["thread-1"]
    interrupt_state = core._interrupt_state
    assert interrupt_state.activated
    parked_results = copy.deepcopy(interrupt_state.context["tool_results"])
    assert parked_results == [
        {
            "toolUseId": "native-approve",
            "status": "success",
            "content": [{"text": "Forwarded to client"}],
        }
    ]
    interrupts = copy.deepcopy(interrupt_state.interrupts)
    tool_metadata = copy.deepcopy(core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY))
    assert set(tool_metadata) == {"native-approve", "native-confirm"}
    assert tool_metadata["native-approve"]["is_frontend"] is True
    assert tool_metadata["native-confirm"]["is_frontend"] is False

    interrupt_id = next(iter(interrupts))
    events2 = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt_id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[approve_tool],
        ),
    )

    errors2 = [event for event in events2 if event.type == EventType.RUN_ERROR]
    assert len(errors2) == 1
    assert errors2[0].code == "INTERRUPT_SESSION_REQUIRED"
    assert not any(event.type == EventType.RUN_FINISHED for event in events2)
    assert model.turn == 1
    assert interrupt_state.activated
    assert interrupt_state.context["tool_results"] == parked_results
    assert interrupt_state.interrupts == interrupts
    assert core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY) == tool_metadata


async def _assert_active_reconciliation_failure_emits_run_error_before_stream_and_keeps_metadata(
    tmp_path, failure_target
):
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent, _ = _make_e2e_agent(config)
    approve_tool = Tool(name="approveTool", description="approve", parameters={})

    events1 = await _collect_events(
        agent,
        _make_run_input(
            messages=[
                UserMessage(
                    id="u1",
                    role="user",
                    content="please handle widget-1",
                )
            ],
            tools=[approve_tool],
        ),
    )
    finished1 = next(event for event in events1 if event.type == EventType.RUN_FINISHED)
    interrupt_id = finished1.outcome.interrupts[0].id
    fe_wire_id = next(
        event.tool_call_id
        for event in events1
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )

    core = agent._agents_by_thread["thread-1"]
    interrupt_state = core._interrupt_state
    parked_context = copy.deepcopy(interrupt_state.context)
    parked_interrupts = copy.deepcopy(interrupt_state.interrupts)
    wire_map = copy.deepcopy(core.state.get(AG_UI_WIRE_MAP_STATE_KEY))
    tool_metadata = copy.deepcopy(core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY))

    with (
        patch.object(core, "stream_async", wraps=core.stream_async) as stream_spy,
        patch(
            failure_target,
            side_effect=RuntimeError("boom"),
        ),
    ):
        resumed_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                messages=[
                    ToolMessage(
                        id="t-fe",
                        role="tool",
                        tool_call_id=fe_wire_id,
                        content='{"approved": true}',
                    )
                ],
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupt_id,
                        status="resolved",
                        payload=True,
                    )
                ],
                tools=[approve_tool],
            ),
        )

    stream_spy.assert_not_called()
    errors = [event for event in resumed_events if event.type == EventType.RUN_ERROR]
    assert len(errors) == 1
    assert errors[0].code == "INTERRUPT_RECONCILIATION_ERROR"
    assert not any(event.type == EventType.RUN_FINISHED for event in resumed_events)
    assert interrupt_state.activated
    assert interrupt_state.context == parked_context
    assert interrupt_state.interrupts == parked_interrupts
    assert core.state.get(AG_UI_WIRE_MAP_STATE_KEY) == wire_map
    assert core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY) == tool_metadata


@pytest.mark.asyncio
async def test_active_reconciliation_failure_emits_run_error_before_stream_and_keeps_metadata(
    tmp_path,
):
    await _assert_active_reconciliation_failure_emits_run_error_before_stream_and_keeps_metadata(
        tmp_path,
        "ag_ui_strands.session_reconcile._correct_all_tools",
    )


@pytest.mark.asyncio
async def test_active_repository_reconciliation_failure_emits_run_error_before_stream_and_keeps_metadata(
    tmp_path,
):
    await _assert_active_reconciliation_failure_emits_run_error_before_stream_and_keeps_metadata(
        tmp_path,
        "ag_ui_strands.session_reconcile._correct_message",
    )


@pytest.mark.parametrize(
    ("case", "frontend_names", "replay_history"),
    [
        pytest.param("missing", ("approveTool",), True, id="missing-proxy-result"),
        pytest.param("unmapped", ("approveTool",), True, id="unmapped-wire-id"),
        pytest.param(
            "partial",
            ("approveTool", "reviewTool"),
            True,
            id="partial-proxy-result-batch",
        ),
        pytest.param(
            "missing",
            ("approveTool",),
            False,
            id="history-replay-disabled",
        ),
    ],
)
@pytest.mark.asyncio
async def test_active_mixed_resume_rejects_unresolved_proxy_placeholders_before_stream(
    tmp_path, case, frontend_names, replay_history
):
    """A native response cannot consume an active sibling proxy placeholder."""
    config = StrandsAgentConfig(
        replay_history_into_strands=replay_history,
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    model = _InterruptFlowModel(sibling_tool_names=frontend_names)
    core_template = StrandsAgentCore(
        model=model,
        tools=[confirm_action],
        system_prompt="test",
    )
    agent = StrandsAgent(core_template, name="unresolved-mixed", config=config)
    frontend_tools = [
        Tool(name=name, description=f"run {name}", parameters={})
        for name in frontend_names
    ]

    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="handle widget-1")],
            tools=frontend_tools,
        ),
    )
    finished = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    )
    interrupt_id = finished.outcome.interrupts[0].id
    wire_ids = {
        event.tool_call_name: event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name in frontend_names
    }
    assert set(wire_ids) == set(frontend_names)

    core = agent._agents_by_thread["thread-1"]
    interrupt_state = core._interrupt_state
    parked_context = copy.deepcopy(interrupt_state.context)
    parked_interrupts = copy.deepcopy(interrupt_state.interrupts)
    parked_messages = copy.deepcopy(core.messages)
    session_manager = core._session_manager
    parked_repository_messages = copy.deepcopy(
        session_manager.session_repository.list_messages(
            session_manager.session_id, core.agent_id
        )
    )
    wire_map = copy.deepcopy(core.state.get(AG_UI_WIRE_MAP_STATE_KEY))
    tool_metadata = copy.deepcopy(core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY))

    if case == "missing":
        unsafe_messages = []
    elif case == "unmapped":
        unsafe_messages = [
            ToolMessage(
                id="t-wrong",
                role="tool",
                tool_call_id="wrong-wire-id",
                content='{"approved": true}',
            )
        ]
    elif case == "partial":
        unsafe_messages = [
            ToolMessage(
                id="t-partial",
                role="tool",
                tool_call_id=wire_ids[frontend_names[0]],
                content='{"approved": true}',
            )
        ]
    else:
        unsafe_messages = [
            ToolMessage(
                id="t-complete",
                role="tool",
                tool_call_id=wire_ids[frontend_names[0]],
                content='{"approved": true}',
            )
        ]

    with patch.object(core, "stream_async", wraps=core.stream_async) as stream_spy:
        unsafe_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                messages=unsafe_messages,
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupt_id,
                        status="resolved",
                        payload=True,
                    )
                ],
                tools=frontend_tools,
            ),
        )

    stream_spy.assert_not_called()
    errors = [event for event in unsafe_events if event.type == EventType.RUN_ERROR]
    assert len(errors) == 1
    assert errors[0].code == "INTERRUPT_RECONCILIATION_ERROR"
    assert not any(event.type == EventType.RUN_FINISHED for event in unsafe_events)
    assert interrupt_state.activated
    assert interrupt_state.context == parked_context
    assert interrupt_state.interrupts == parked_interrupts
    assert core.messages == parked_messages
    assert session_manager.session_repository.list_messages(
        session_manager.session_id, core.agent_id
    ) == parked_repository_messages
    assert core.state.get(AG_UI_WIRE_MAP_STATE_KEY) == wire_map
    assert core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY) == tool_metadata

    complete_messages = [
        ToolMessage(
            id=f"t-retry-{index}",
            role="tool",
            tool_call_id=wire_ids[name],
            content=f'{{"{name}": true}}',
        )
        for index, name in enumerate(frontend_names)
    ]
    retry_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            messages=complete_messages,
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt_id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=frontend_tools,
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in retry_events)
    assert any(event.type == EventType.RUN_FINISHED for event in retry_events)
    assert "Forwarded to client" not in json.dumps(model.stream_calls_messages[-1])


@pytest.mark.parametrize("recreate_agent", [False, True])
@pytest.mark.parametrize("fe_continues", [False, True])
@pytest.mark.asyncio
async def test_mixed_resume_batch_with_falsy_payload_and_tool_behaviors(
    tmp_path, recreate_agent, fe_continues
):
    """Regression for mixed FE tools & interrupts.

    Uses a real ``FileSessionManager`` — the no-session-manager path (in-memory
    ``replay_history_into_strands``) is out of scope for this regression.

    Parametrized on ``recreate_agent``: ``False`` exercises resume through the
    same in-memory ``StrandsAgent``/``StrandsAgentCore`` (the per-thread cache
    still holds the paused agent); ``True`` discards them after turn 1 and
    resumes through freshly constructed ones sharing the same
    ``FileSessionManager``-backed session — the cross-process resume scenario
    the README's "Persistence" caveat describes, where nothing survives in
    memory from turn 1.

    Parametrized on ``fe_continues`` (``continue_after_frontend_call`` for the
    frontend tool) because in THIS batch shape the flag is near-moot, and both
    settings must reach the same interrupt outcome:

    * The model commits both ``toolUse`` blocks in ONE assistant message, so
      the halt cannot pre-empt ``confirm_action`` — it is already dispatched
      concurrently by Strands' ``ConcurrentToolExecutor``.
    * ``confirm_action`` interrupts, so Strands returns early at
      ``event_loop.py:501`` WITHOUT appending the ``role=user`` tool-result
      message. That message is the only place ``pending_halt`` is promoted to
      ``halt_event_stream`` (``agent.py:1479-1480``), so the halt latches but
      never fires; the interrupt stops the loop instead. Measured consequence:
      the flag only changes where the frontend ``TOOL_CALL_END`` lands on the
      wire.

    ``False`` keeps coverage that a latched-but-unfired halt does not corrupt
    the interrupt path (moving the latch earlier would break this param and
    not the other); ``True`` models immediate hand-off.
    """
    tool_behaviors = {
        "confirm_action": ToolBehavior(
            state_from_result=lambda ctx: {"confirmed_key": ctx.result_data}
        )
    }
    if fe_continues:
        tool_behaviors["approveTool"] = ToolBehavior(continue_after_frontend_call=True)
    config = StrandsAgentConfig(
        tool_behaviors=tool_behaviors,
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent, model = _make_e2e_agent(config)

    approve_tool = Tool(name="approveTool", description="approve", parameters={})
    inp1 = _make_run_input(
        messages=[UserMessage(id="u1", role="user", content="please handle widget-1")],
        tools=[approve_tool],
    )
    events1 = await _collect_events(agent, inp1)

    finished1 = next(e for e in events1 if e.type == EventType.RUN_FINISHED)
    assert finished1.outcome is not None
    assert finished1.outcome.type == "interrupt"
    interrupt_id = finished1.outcome.interrupts[0].id
    fe_wire_id = next(
        e.tool_call_id
        for e in events1
        if e.type == EventType.TOOL_CALL_START and e.tool_call_name == "approveTool"
    )

    if recreate_agent:
        # Discard the wrapper and underlying core entirely — turn 2 must
        # restore interrupt state, the wire->native map, and history purely
        # from the FileSessionManager-backed session, not from memory. Carry
        # over the turn count so it reads the same as the non-recreated case.
        prior_turn = model.turn
        agent, model = _make_e2e_agent(config)
        model.turn = prior_turn

    inp2 = _make_run_input(
        run_id="run-2",
        messages=[
            ToolMessage(
                id="t-fe",
                role="tool",
                tool_call_id=fe_wire_id,
                content='{"approved": true}',
            )
        ],
        resume=[ResumeEntry(interrupt_id=interrupt_id, status="resolved", payload=False)],
        tools=[approve_tool],
    )
    events2 = await _collect_events(agent, inp2)

    assert not any(
        event.type == EventType.TOOL_CALL_RESULT
        and event.tool_call_id == "native-approve"
        for event in events2
    )

    # --- A falsy-but-explicit resume payload must resolve completely. ---
    assert not any(event.type == EventType.RUN_ERROR for event in events2)
    finished_events = [
        event for event in events2 if event.type == EventType.RUN_FINISHED
    ]
    assert len(finished_events) == 1
    assert finished_events[0].outcome is None

    # --- The frontend tool's REAL result must reach the model. ---
    assert model.turn >= 2, "resume never advanced the event loop past the interrupt"
    last_messages_text = json.dumps(model.stream_calls_messages[-1])
    assert "approved" in last_messages_text
    assert "Forwarded to client" not in last_messages_text

    # --- state_from_result must fire for a tool resolved on resume. ---
    assert any(
        e.type == EventType.STATE_SNAPSHOT and e.snapshot.get("confirmed_key") for e in events2
    ), "state_from_result did not fire for confirm_action on the resume run"


@pytest.mark.asyncio
async def test_mixed_active_resume_replays_failed_frontend_result_as_error(tmp_path):
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent, model = _make_e2e_agent(config)
    approve_tool = Tool(name="approveTool", description="approve", parameters={})
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="handle widget-1")],
            tools=[approve_tool],
        ),
    )
    finished = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    )
    interrupt_id = finished.outcome.interrupts[0].id
    wire_id = next(
        event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-failed",
                    role="tool",
                    tool_call_id=wire_id,
                    content="",
                    error="boom",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt_id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[approve_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    frontend_result = next(
        block["toolResult"]
        for message in model.stream_calls_messages[-1]
        for block in message.get("content", [])
        if block.get("toolResult", {}).get("toolUseId") == "native-approve"
    )
    assert frontend_result == {
        "toolUseId": "native-approve",
        "status": "error",
        "content": [{"text": "boom"}],
    }
    assert "boom" in json.dumps(model.stream_calls_messages[-1])
    assert "executed successfully" not in json.dumps(model.stream_calls_messages[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation,mutated_id",
    [
        ("id-in-place", "hook-mutated-id"),
        ("id-replacement", "hook-replacement-id"),
    ],
)
async def test_proxy_hook_id_mutation_round_trips_on_original_identity(
    tmp_path,
    mutation,
    mutated_id,
):
    """A caller-selected hook alias cannot replace native/wire identity."""
    model = _ProxyHookInterruptFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=_MutatingFrontendProxyHook(mutation),
        config=config,
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id, interrupt = _proxy_wire_and_interrupt(initial_events)

    assert interrupt.tool_call_id == wire_id
    assert not any(
        getattr(event, "tool_call_id", None) == mutated_id
        for event in initial_events
    )

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-approve",
                    role="tool",
                    tool_call_id=wire_id,
                    content="CLIENT-REAL",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt.id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    model_input = json.dumps(model.stream_calls_messages[-1])
    assert model_input.count("CLIENT-REAL") == 1
    assert "Forwarded to client" not in model_input
    assert mutated_id not in json.dumps(model.stream_calls_messages)
    live_core = agent._agents_by_thread["thread-1"]
    assistant_ids = [
        block["toolUse"]["toolUseId"]
        for message in live_core.messages
        for block in message.get("content", [])
        if "toolUse" in block and block["toolUse"]["name"] == "approveTool"
    ]
    result_ids = [
        block["toolResult"]["toolUseId"]
        for message in live_core.messages
        for block in message.get("content", [])
        if "toolResult" in block
        and block["toolResult"].get("content") == [{"text": "CLIENT-REAL"}]
    ]
    assert assistant_ids == result_ids == ["native-approve"]

@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["id-in-place", "id-replacement"])
async def test_proxy_hook_id_alias_survives_recreated_wrapper(
    tmp_path,
    mutation,
):
    """Durable alias provenance never relies on same-process inference."""
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_model = _ProxyHookInterruptFlowModel()
    initial_agent = _proxy_hook_agent(
        initial_model,
        hook=_MutatingFrontendProxyHook(mutation),
        config=config,
    )
    initial_events = await _collect_events(
        initial_agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id, interrupt = _proxy_wire_and_interrupt(initial_events)

    resumed_model = _ProxyHookInterruptFlowModel()
    resumed_model.turn = initial_model.turn
    recreated_agent = _proxy_hook_agent(
        resumed_model,
        hook=_MutatingFrontendProxyHook(mutation),
        config=config,
    )
    resumed_events = await _collect_events(
        recreated_agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-approve",
                    role="tool",
                    tool_call_id=wire_id,
                    content="DURABLE-REAL",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt.id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    assert "DURABLE-REAL" in json.dumps(
        resumed_model.stream_calls_messages[-1]
    )
    assert "Forwarded to client" not in json.dumps(
        resumed_model.stream_calls_messages[-1]
    )
    assert "hook-mutated-id" not in json.dumps(
        resumed_model.stream_calls_messages
    )
    assert "hook-replacement-id" not in json.dumps(
        resumed_model.stream_calls_messages
    )

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "input",
        "name",
        "selected-tool",
        "tool-spec",
        "wire-map-delete",
        "tool-meta-delete",
    ],
)
async def test_proxy_hook_semantic_mutation_fails_before_authorization(
    tmp_path,
    mutation,
):
    """Already-emitted proxy semantics cannot be rewritten by caller hooks."""
    _replacement_tool_calls.clear()
    model = _ProxyHookInterruptFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=_MutatingFrontendProxyHook(mutation),
        config=config,
        include_replacement=mutation == "selected-tool",
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert [
        event.code for event in events if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_PROXY_PROVENANCE_ERROR"]
    assert not any(event.type == EventType.TOOL_CALL_END for event in events)
    assert not any(event.type == EventType.TOOL_CALL_RESULT for event in events)
    assert not any(event.type == EventType.RUN_FINISHED for event in events)
    assert _replacement_tool_calls == []
    live_core = agent._agents_by_thread["thread-1"]
    assert not live_core._interrupt_state.activated
    assert live_core._interrupt_state.interrupts == {}
    assert not live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY)
    assert not live_core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)
    assert live_core.messages[-1]["role"] == "assistant"
    [redacted_block] = live_core.messages[-1]["content"]
    assert isinstance(redacted_block.get("text"), str)
    assert redacted_block["text"].strip()
    assert "toolUse" not in redacted_block
    assert "native-approve" not in json.dumps(live_core.messages[-1])
    assert "Forwarded to client" not in json.dumps(live_core.messages[-1])
    persisted = FileSessionManager(
        session_id="thread-1", storage_dir=str(tmp_path)
    ).session_repository.list_messages("thread-1", "proxy-hook-agent")
    assert persisted[-1].to_message() == live_core.messages[-1]
    assert persisted[-1].redact_message == live_core.messages[-1]

    safe_model = model
    safe_model.turn = 1
    safe_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[UserMessage(id="u2", role="user", content="continue")],
            tools=[],
        ),
    )
    assert not any(event.type == EventType.RUN_ERROR for event in safe_events)
    assert any(event.type == EventType.RUN_FINISHED for event in safe_events)

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "input-after-resume",
        "cancel-after-resume",
        "provenance-delete",
        "resume-value-in-place",
        "resume-binding-in-place",
    ],
)
async def test_proxy_hook_resume_semantic_mutation_restores_retry_checkpoint(
    tmp_path,
    mutation,
):
    """A mutation after interrupt returns cannot execute or consume the result."""
    model = _ProxyHookInterruptFlowModel()
    hook = _MutatingFrontendProxyHook(mutation)
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(model, hook=hook, config=config)
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id, interrupt = _proxy_wire_and_interrupt(initial_events)
    live_core = agent._agents_by_thread["thread-1"]
    before = _snapshot_resume_state(live_core)
    resume_input = _make_run_input(
        run_id="run-2",
        messages=[
            ToolMessage(
                id="t-approve",
                role="tool",
                tool_call_id=wire_id,
                content="CLIENT-REAL",
            )
        ],
        resume=[
            ResumeEntry(
                interrupt_id=interrupt.id,
                status="resolved",
                payload=True,
            )
        ],
        tools=[frontend_tool],
    )

    failed_events = await _collect_events(agent, resume_input)

    assert [
        event.code
        for event in failed_events
        if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_PROXY_PROVENANCE_ERROR"]
    assert not any(
        event.type == EventType.RUN_FINISHED for event in failed_events
    )
    assert model.turn == 1
    assert _snapshot_resume_state(live_core) == before

    hook.mutation = "none"
    retry_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            messages=resume_input.messages,
            resume=resume_input.resume,
            tools=[frontend_tool],
        ),
    )
    assert not any(event.type == EventType.RUN_ERROR for event in retry_events)
    assert "CLIENT-REAL" in json.dumps(model.stream_calls_messages[-1])

@pytest.mark.asyncio
async def test_proxy_hook_resume_remains_armed_without_second_interrupt_call(
    tmp_path,
):
    """Durable provenance prevents a stateful hook bypass on resume."""
    _replacement_tool_calls.clear()
    model = _ProxyHookInterruptFlowModel()
    hook = _InterruptThenReplaceWithoutRecheckingHook()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=hook,
        config=config,
        include_replacement=True,
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id, interrupt = _proxy_wire_and_interrupt(initial_events)
    live_core = agent._agents_by_thread["thread-1"]
    checkpoint = _snapshot_resume_state(live_core)

    resume_input = _make_run_input(
        run_id="run-2",
        messages=[
            ToolMessage(
                id="t-approve",
                role="tool",
                tool_call_id=wire_id,
                content="CLIENT-REAL",
            )
        ],
        resume=[
            ResumeEntry(
                interrupt_id=interrupt.id,
                status="resolved",
                payload=True,
            )
        ],
        tools=[frontend_tool],
    )
    failed_events = await _collect_events(agent, resume_input)

    assert [
        event.code
        for event in failed_events
        if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_PROXY_PROVENANCE_ERROR"]
    assert not any(
        event.type == EventType.RUN_FINISHED for event in failed_events
    )
    assert _replacement_tool_calls == []
    assert model.turn == 1
    assert _snapshot_resume_state(live_core) == checkpoint

@pytest.mark.asyncio
async def test_proxy_hook_resume_contains_later_static_hook_failure(
    tmp_path,
):
    """A callback failure preserves the live checkpoint for an exact retry."""
    model = _ProxyHookInterruptFlowModel()
    raising_hook = _RaiseOnSecondProxyHook()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    core = StrandsAgentCore(
        model=model,
        system_prompt="test",
        agent_id="proxy-hook-agent",
    )
    agent = StrandsAgent(
        core,
        name="proxy-hook-provenance",
        config=config,
        hooks=[_InterruptFrontendProxyHook(), raising_hook],
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id, interrupt = _proxy_wire_and_interrupt(initial_events)
    live_core = agent._agents_by_thread["thread-1"]
    checkpoint = _snapshot_resume_state(live_core)

    resume_input = _make_run_input(
        run_id="run-2",
        messages=[
            ToolMessage(
                id="t-approve",
                role="tool",
                tool_call_id=wire_id,
                content="CLIENT-REAL",
            )
        ],
        resume=[
            ResumeEntry(
                interrupt_id=interrupt.id,
                status="resolved",
                payload=True,
            )
        ],
        tools=[frontend_tool],
    )
    failed_events = await _collect_events(agent, resume_input)

    assert [
        event.code
        for event in failed_events
        if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_PROXY_PROVENANCE_ERROR"]
    assert not any(
        event.type == EventType.RUN_FINISHED for event in failed_events
    )
    assert "CLIENT-REAL" not in json.dumps(model.stream_calls_messages[-1])
    after_failure = _snapshot_resume_state(live_core)
    for key, value in checkpoint.items():
        if key != "repository_messages":
            assert after_failure[key] == value

    retry_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            messages=resume_input.messages,
            resume=resume_input.resume,
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in retry_events)
    assert any(event.type == EventType.RUN_FINISHED for event in retry_events)
    assert sum(
        json.dumps(messages).count("CLIENT-REAL")
        for messages in model.stream_calls_messages
    ) == 1

@pytest.mark.asyncio
async def test_proxy_hook_resume_rejects_changed_client_tool_spec_pre_stream(
    tmp_path,
):
    """A recreated client declaration cannot change a parked proxy contract."""
    model = _ProxyHookInterruptFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=_MutatingFrontendProxyHook("none"),
        config=config,
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    wire_id, interrupt = _proxy_wire_and_interrupt(initial_events)
    live_core = agent._agents_by_thread["thread-1"]
    checkpoint = _snapshot_resume_state(live_core)
    resume_messages = [
        ToolMessage(
            id="t-approve",
            role="tool",
            tool_call_id=wire_id,
            content="CLIENT-REAL",
        )
    ]
    resume = [
        ResumeEntry(
            interrupt_id=interrupt.id,
            status="resolved",
            payload=True,
        )
    ]

    failed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=resume_messages,
            resume=resume,
            tools=[
                Tool(
                    name="approveTool",
                    description="approve",
                    parameters={"changed": {"type": "string"}},
                )
            ],
        ),
    )

    assert [
        event.code
        for event in failed_events
        if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_PROXY_PROVENANCE_ERROR"]
    assert model.turn == 1
    assert _snapshot_resume_state(live_core) == checkpoint

    retry_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            messages=resume_messages,
            resume=resume,
            tools=[frontend_tool],
        ),
    )
    assert not any(event.type == EventType.RUN_ERROR for event in retry_events)
    assert "CLIENT-REAL" in json.dumps(model.stream_calls_messages[-1])

@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["none", "id-in-place"])
async def test_proxy_hook_without_session_restores_exact_live_checkpoint(
    mutation,
):
    """An unresumable proxy checkpoint is never authorized on the client."""
    model = _ProxyHookInterruptFlowModel()
    hook = _MutatingFrontendProxyHook(mutation)
    agent = _proxy_hook_agent(
        model,
        hook=hook,
        config=StrandsAgentConfig(),
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert [
        event.code for event in events if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_SESSION_REQUIRED"]
    assert not any(event.type == EventType.TOOL_CALL_END for event in events)
    assert not any(event.type == EventType.RUN_FINISHED for event in events)
    live_core = agent._agents_by_thread["thread-1"]
    assert live_core.messages == [
        {"role": "user", "content": [{"text": "approve"}]}
    ]
    assert "toolUse" not in json.dumps(live_core.messages)
    assert live_core._interrupt_state == _InterruptState()
    assert not live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY)
    assert not live_core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)
    assert "__ag_ui_proxy_hook_provenance__" not in live_core.state.get()

    hook.enabled = False
    retry_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[UserMessage(id="u2", role="user", content="continue")],
            tools=[],
        ),
    )
    assert not any(event.type == EventType.RUN_ERROR for event in retry_events)
    assert any(event.type == EventType.RUN_FINISHED for event in retry_events)

@pytest.mark.asyncio
async def test_non_interrupting_proxy_hook_preserves_legal_strands_mutation():
    """The integrity boundary applies to checkpoints, not ordinary hooks."""
    model = _ProxyHookInterruptFlowModel()
    agent = _proxy_hook_agent(
        model,
        hook=_MutatingFrontendProxyHook("input", should_interrupt=False),
        config=StrandsAgentConfig(),
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in events)
    live_core = agent._agents_by_thread["thread-1"]
    assert "__ag_ui_proxy_hook_provenance__" not in live_core.state.get()
    assert any(
        block["toolUse"]["input"] == {"changed": True}
        for message in live_core.messages
        for block in message.get("content", [])
        if "toolUse" in block
    )

@pytest.mark.asyncio
async def test_preemptive_proxy_hook_response_does_not_leave_provenance(
    tmp_path,
):
    """Only an unresolved hook interrupt establishes a durable checkpoint."""
    model = _ProxyHookInterruptFlowModel(tool_turns={1, 2})
    hook = _PreemptiveThenPendingInterruptHook()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(model, hook=hook, config=config)
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )

    preempted_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="first")],
            tools=[frontend_tool],
        ),
    )

    assert not any(
        event.type == EventType.RUN_ERROR for event in preempted_events
    )
    first_finished = next(
        event
        for event in preempted_events
        if event.type == EventType.RUN_FINISHED
    )
    assert first_finished.outcome is None
    live_core = agent._agents_by_thread["thread-1"]
    assert "__ag_ui_proxy_hook_provenance__" not in live_core.state.get()
    assert any(
        block["toolUse"]["input"] == {"preemptive": True}
        for message in live_core.messages
        for block in message.get("content", [])
        if "toolUse" in block
    )
    first_wire_id = next(
        event.tool_call_id
        for event in preempted_events
        if event.type == EventType.TOOL_CALL_START
    )

    pending_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-first",
                    role="tool",
                    tool_call_id=first_wire_id,
                    content="FIRST-REAL",
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in pending_events)
    wire_id, interrupt = _proxy_wire_and_interrupt(pending_events)
    assert interrupt.tool_call_id == wire_id
    assert model.turn == 2

@pytest.mark.asyncio
async def test_caught_proxy_hook_interrupt_preserves_legal_mutation():
    """A candidate caught inside its callback never arms the boundary."""
    model = _ProxyHookInterruptFlowModel()
    agent = _proxy_hook_agent(
        model,
        hook=_CaughtInterruptThenMutateHook(),
        config=StrandsAgentConfig(),
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in events)
    live_core = agent._agents_by_thread["thread-1"]
    assert "__ag_ui_proxy_hook_provenance__" not in live_core.state.get()
    assert any(
        block["toolUse"]["input"] == {"caught": True}
        for message in live_core.messages
        for block in message.get("content", [])
        if "toolUse" in block
    )


@pytest.mark.asyncio
async def test_proxy_hook_event_interrupt_method_is_restored_after_finalizer():
    """The call-exact observer never remains installed on a retained event."""
    model = _ProxyHookInterruptFlowModel()
    hook = _RetainingNonInterruptingProxyHook()
    agent = _proxy_hook_agent(
        model,
        hook=hook,
        config=StrandsAgentConfig(),
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in events)
    assert hook.event is not None
    assert "interrupt" not in vars(hook.event)
    assert hook.event.interrupt.__self__ is hook.event


@pytest.mark.asyncio
async def test_proxy_hook_and_native_sibling_interrupts_are_both_advertised(
    tmp_path,
):
    """Managed provenance never claims a concurrent native checkpoint."""
    model = _InterruptFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = StrandsAgent(
        StrandsAgentCore(
            model=model,
            tools=[confirm_action],
            system_prompt="test",
            agent_id="proxy-hook-agent",
        ),
        name="proxy-hook-provenance",
        config=config,
        hooks=[_MutatingFrontendProxyHook("none")],
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="run both")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in events)
    finished = next(
        event for event in events if event.type == EventType.RUN_FINISHED
    )
    assert finished.outcome is not None
    by_name = {
        interrupt.metadata["strands_name"]: interrupt
        for interrupt in finished.outcome.interrupts
    }
    assert set(by_name) == {"approve_proxy", "confirm_action"}
    wire_id = next(
        event.tool_call_id
        for event in events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )
    assert by_name["approve_proxy"].tool_call_id == wire_id
    assert by_name["confirm_action"].tool_call_id == "native-confirm"


@pytest.mark.asyncio
@pytest.mark.parametrize("recreate_wrapper", [False, True])
@pytest.mark.parametrize(
    ("client_content", "client_error", "expected_status", "expected_text"),
    [
        pytest.param(
            "CLIENT-PROXY-RESULT",
            None,
            "success",
            "CLIENT-PROXY-RESULT",
            id="success",
        ),
        pytest.param("", "boom", "error", "boom", id="blank-failure"),
    ],
)
async def test_native_only_resume_persists_unselected_proxy_result_for_later_resume(
    tmp_path,
    recreate_wrapper,
    client_content,
    client_error,
    expected_status,
    expected_text,
):
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )

    def make_agent(model):
        return StrandsAgent(
            StrandsAgentCore(
                model=model,
                tools=[confirm_action],
                system_prompt="test",
                agent_id="proxy-hook-agent",
            ),
            name="mixed-proxy-native-persistence",
            config=config,
            hooks=[_MutatingFrontendProxyHook("none")],
        )

    model = _InterruptFlowModel()
    agent = make_agent(model)
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="run both")],
            tools=[frontend_tool],
        ),
    )
    initial_finished = next(
        event for event in initial if event.type == EventType.RUN_FINISHED
    )
    interrupts = {
        interrupt.metadata["strands_name"]: interrupt
        for interrupt in initial_finished.outcome.interrupts
    }
    wire_id = next(
        event.tool_call_id
        for event in initial
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )

    native_resumed = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-proxy-result",
                    role="tool",
                    tool_call_id=wire_id,
                    content=client_content,
                    error=client_error,
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["confirm_action"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in native_resumed)
    native_finished = next(
        event
        for event in native_resumed
        if event.type == EventType.RUN_FINISHED
    )
    assert [
        interrupt.metadata["strands_name"]
        for interrupt in native_finished.outcome.interrupts
    ] == ["approve_proxy"]
    live_core = agent._agents_by_thread["thread-1"]
    assert live_core.state.get(AG_UI_PENDING_PROXY_RESULTS_STATE_KEY) == {
        "version": 1,
        "records": {
            "native-approve": {
                "wire_tool_call_id": wire_id,
                "content": client_content,
                "status": expected_status,
                "error": client_error,
            }
        },
    }
    assert model.turn == 1

    if recreate_wrapper:
        model = _InterruptFlowModel()
        model.turn = 1
        agent = make_agent(model)

    proxy_resumed = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["approve_proxy"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in proxy_resumed)
    proxy_finished = next(
        event
        for event in proxy_resumed
        if event.type == EventType.RUN_FINISHED
    )
    assert proxy_finished.outcome is None
    live_core = agent._agents_by_thread["thread-1"]
    observed = [
        block["toolResult"]
        for message in model.stream_calls_messages[-1]
        for block in message.get("content", [])
        if block.get("toolResult", {}).get("toolUseId") == "native-approve"
    ]
    assert observed == [
        {
            "toolUseId": "native-approve",
            "status": expected_status,
            "content": [{"text": expected_text}],
        }
    ]
    assert model.turn == 2
    assert AG_UI_PENDING_PROXY_RESULTS_STATE_KEY not in live_core.state.get()
    assert wire_id not in live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY)
    assert "__ag_ui_proxy_hook_provenance__" not in live_core.state.get()

@pytest.mark.asyncio
async def test_async_parallel_proxy_hook_aliases_remain_call_exact(tmp_path):
    """Concurrent finalizers accept sibling records and never cross results."""
    model = _InterruptFlowModel(
        sibling_tool_names=("approveTool", "approveTool")
    )
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=_AsyncPerCallAliasHook(),
        config=config,
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="both")],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in initial_events)
    finished = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    )
    assert finished.outcome is not None
    assert len(finished.outcome.interrupts) == 2
    live_core = agent._agents_by_thread["thread-1"]
    provenance = live_core.state.get("__ag_ui_proxy_hook_provenance__")
    records = provenance["records"]
    result_by_native = {
        "native-approve": "FIRST-REAL",
        "native-approve-2": "SECOND-REAL",
    }
    wire_by_native = {
        record["original_native_tool_call_id"]: record["wire_tool_call_id"]
        for record in records.values()
    }
    assert set(wire_by_native) == set(result_by_native)

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id=f"t-{native_id}",
                    role="tool",
                    tool_call_id=wire_by_native[native_id],
                    content=result,
                )
                for native_id, result in result_by_native.items()
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt.id,
                    status="resolved",
                    payload=True,
                )
                for interrupt in finished.outcome.interrupts
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    observed = {
        block["toolResult"]["toolUseId"]: block["toolResult"]["content"][0][
            "text"
        ]
        for message in model.stream_calls_messages[-1]
        for block in message.get("content", [])
        if block.get("toolResult", {}).get("toolUseId") in result_by_native
    }
    assert observed == result_by_native
    assert AG_UI_PENDING_PROXY_RESULTS_STATE_KEY not in live_core.state.get()


@pytest.mark.asyncio
async def test_delayed_sibling_finalizer_does_not_resurrect_consumed_proxy_result(
    tmp_path,
):
    model = _ConcurrentProxyHookFlowModel()
    hook = _DelayedSecondProxyObserverHook()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(model, hook=hook, config=config)
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="both")],
            tools=[frontend_tool],
        ),
    )
    finished = next(
        event for event in initial if event.type == EventType.RUN_FINISHED
    )
    live_core = agent._agents_by_thread["thread-1"]
    provenance = live_core.state.get("__ag_ui_proxy_hook_provenance__")
    interrupt_by_native = {
        provenance["records"][interrupt.id]["original_native_tool_call_id"]: (
            interrupt
        )
        for interrupt in finished.outcome.interrupts
    }
    wire_by_native = {
        record["original_native_tool_call_id"]: record["wire_tool_call_id"]
        for record in provenance["records"].values()
    }

    resumed = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id=f"t-{native_id}",
                    role="tool",
                    tool_call_id=wire_by_native[native_id],
                    content=f"CLIENT-{native_id}",
                )
                for native_id in model.native_ids
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt_by_native[native_id].id,
                    status="resolved",
                    payload=True,
                )
                for native_id in model.native_ids
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed)
    assert hook.after_snapshots == [
        (
            "native-proxy-a",
            {"native-proxy-b"},
            {"native-proxy-b"},
        ),
        ("native-proxy-b", set(), set()),
    ]
    boundary = agent._proxy_hook_boundaries_by_thread["thread-1"]
    assert boundary.working_resume_results == {}
    assert boundary.working_resume_bindings == {}
    assert AG_UI_PENDING_PROXY_RESULTS_STATE_KEY not in live_core.state.get()


@pytest.mark.asyncio
async def test_concurrent_proxy_hook_partial_resume_consumes_only_selected_call(
    tmp_path,
):
    """Selecting A requires/injects only A while B remains fully parked."""
    agent, model, live_core, calls = (
        await _start_concurrent_proxy_hook_checkpoint(tmp_path)
    )
    call_a = calls["native-proxy-a"]
    call_b = calls["native-proxy-b"]

    resumed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-a",
                    role="tool",
                    tool_call_id=call_a["wire_id"],
                    content="CLIENT-A",
                ),
                ToolMessage(
                    id="t-b",
                    role="tool",
                    tool_call_id=call_b["wire_id"],
                    content="CLIENT-B",
                ),
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=call_a["interrupt"].id,
                    status="resolved",
                    payload=True,
                )
            ],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in resumed_events)
    finished = next(
        event for event in resumed_events if event.type == EventType.RUN_FINISHED
    )
    assert finished.outcome is not None
    assert [
        interrupt.metadata["strands_name"]
        for interrupt in finished.outcome.interrupts
    ] == ["proxy_b"]
    assert finished.outcome.interrupts[0].tool_call_id == call_b["wire_id"]

    interrupt_state = live_core._interrupt_state
    result_a = [
        result
        for result in interrupt_state.context["tool_results"]
        if result["toolUseId"] == "native-proxy-a"
    ]
    assert result_a == [
        {
            "toolUseId": "native-proxy-a",
            "status": "success",
            "content": [{"text": "CLIENT-A"}],
        }
    ]
    assert not any(
        result["toolUseId"] == "native-proxy-b"
        for result in interrupt_state.context["tool_results"]
    )
    assert interrupt_state.interrupts[call_b["interrupt"].id].response is None
    assert model.turn == 1

    provenance = live_core.state.get("__ag_ui_proxy_hook_provenance__")
    assert provenance["records"][call_b["interrupt"].id] == call_b["record"]
    assert live_core.state.get(AG_UI_PENDING_PROXY_RESULTS_STATE_KEY) == {
        "version": 1,
        "records": {
            "native-proxy-b": {
                "wire_tool_call_id": call_b["wire_id"],
                "content": "CLIENT-B",
                "status": "success",
                "error": None,
            }
        },
    }
    assert live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY)[call_b["wire_id"]] == (
        "native-proxy-b"
    )
    assert live_core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)["native-proxy-b"][
        "wire_tool_call_id"
    ] == call_b["wire_id"]
    assert getattr(
        live_core.tool_registry.registry["approveTool"], "_ag_ui_proxy", False
    )

    encoded_events = "".join(
        EventEncoder().encode(event) for event in resumed_events
    )
    assert "Forwarded to client" not in encoded_events
    assert "native-proxy-a" not in encoded_events
    assert "native-proxy-b" not in encoded_events


@pytest.mark.asyncio
async def test_concurrent_proxy_hook_subset_missing_selected_result_is_atomic(
    tmp_path,
):
    """A selected proxy hook still requires its exact client result."""
    agent, model, live_core, calls = (
        await _start_concurrent_proxy_hook_checkpoint(tmp_path)
    )
    call_a = calls["native-proxy-a"]
    before = _snapshot_resume_state(live_core)

    with patch.object(
        live_core, "stream_async", wraps=live_core.stream_async
    ) as stream_spy:
        failed_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                resume=[
                    ResumeEntry(
                        interrupt_id=call_a["interrupt"].id,
                        status="resolved",
                        payload=True,
                    )
                ],
            ),
        )

    stream_spy.assert_not_called()
    _assert_proxy_reconciliation_error(failed_events)
    assert model.turn == 1
    assert _snapshot_resume_state(live_core) == before


@pytest.mark.asyncio
async def test_concurrent_proxy_hook_subset_sibling_result_cannot_authorize_selected(
    tmp_path,
):
    """B's equal-name result cannot substitute for selected A's result."""
    agent, model, live_core, calls = (
        await _start_concurrent_proxy_hook_checkpoint(tmp_path)
    )
    call_a = calls["native-proxy-a"]
    call_b = calls["native-proxy-b"]
    before = _snapshot_resume_state(live_core)

    with patch.object(
        live_core, "stream_async", wraps=live_core.stream_async
    ) as stream_spy:
        failed_events = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-2",
                messages=[
                    ToolMessage(
                        id="t-b",
                        role="tool",
                        tool_call_id=call_b["wire_id"],
                        content="CLIENT-B",
                    )
                ],
                resume=[
                    ResumeEntry(
                        interrupt_id=call_a["interrupt"].id,
                        status="resolved",
                        payload=True,
                    )
                ],
            ),
        )

    stream_spy.assert_not_called()
    _assert_proxy_reconciliation_error(failed_events)
    assert model.turn == 1
    assert _snapshot_resume_state(live_core) == before


@pytest.mark.asyncio
async def test_answered_proxy_hook_does_not_drive_later_same_wrapper_requirements(
    tmp_path,
):
    """Answered A stays stored but cannot retain or burden run-three B."""
    agent, model, tools, interrupts, wire_ids = (
        await _start_answered_proxy_hook_checkpoint(tmp_path)
    )

    await _finish_answered_proxy_b(
        agent,
        model,
        tools["approveB"],
        interrupts["proxy_b"],
        wire_ids["approveB"],
    )

    live_core = agent._agents_by_thread["thread-1"]
    assert "approveA" not in live_core.tool_registry.registry


@pytest.mark.asyncio
async def test_answered_proxy_hook_does_not_drive_later_recreated_requirements(
    tmp_path,
):
    """Durable answered A cannot become a capability requirement on restore."""
    _agent, _model, tools, interrupts, wire_ids = (
        await _start_answered_proxy_hook_checkpoint(tmp_path)
    )
    recreated_model = _AnsweredProxyHookFlowModel(turn=1)
    recreated_agent = _answered_proxy_hook_agent(recreated_model, tmp_path)

    await _finish_answered_proxy_b(
        recreated_agent,
        recreated_model,
        tools["approveB"],
        interrupts["proxy_b"],
        wire_ids["approveB"],
    )

    recreated_core = recreated_agent._agents_by_thread["thread-1"]
    assert "approveA" not in recreated_core.tool_registry.registry
    assert "approveB" in recreated_core.tool_registry.registry


async def _start_two_stage_proxy_result_checkpoint(tmp_path, *, hooks=None):
    model = _ProxyHookInterruptFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    core = StrandsAgentCore(
        model=model,
        system_prompt="test",
        agent_id="proxy-hook-two-stage-agent",
    )
    agent = StrandsAgent(
        core,
        name="two-stage-proxy-hook",
        config=config,
        hooks=hooks
        or [
            _FixedProxyInterruptHook("stage_one"),
            _FixedProxyInterruptHook("stage_two"),
        ],
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )
    finished = next(event for event in initial if event.type == EventType.RUN_FINISHED)
    interrupts = {
        interrupt.metadata["strands_name"]: interrupt
        for interrupt in finished.outcome.interrupts
    }
    wire_id = next(
        event.tool_call_id
        for event in initial
        if event.type == EventType.TOOL_CALL_START
    )
    partial = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-result",
                    role="tool",
                    tool_call_id=wire_id,
                    content="CLIENT-RESULT",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["stage_one"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )
    assert not any(event.type == EventType.RUN_ERROR for event in partial)
    return agent, model, frontend_tool, interrupts, wire_id


@pytest.mark.asyncio
async def test_pending_proxy_result_conflict_fails_closed_then_identical_retry_works(
    tmp_path,
):
    agent, model, frontend_tool, interrupts, wire_id = (
        await _start_two_stage_proxy_result_checkpoint(tmp_path)
    )
    live_core = agent._agents_by_thread["thread-1"]
    before = _snapshot_resume_state(live_core)

    conflict = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3-conflict",
            messages=[
                ToolMessage(
                    id="t-conflict",
                    role="tool",
                    tool_call_id=wire_id,
                    content="DIFFERENT",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["stage_two"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    _assert_proxy_reconciliation_error(conflict)
    assert _snapshot_resume_state(live_core) == before
    assert model.turn == 1

    retry = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3-retry",
            messages=[
                ToolMessage(
                    id="t-identical",
                    role="tool",
                    tool_call_id=wire_id,
                    content="CLIENT-RESULT",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["stage_two"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in retry)
    assert AG_UI_PENDING_PROXY_RESULTS_STATE_KEY not in live_core.state.get()


@pytest.mark.asyncio
async def test_pending_proxy_result_tamper_rolls_back_to_prior_accepted_result(
    tmp_path,
):
    agent, model, frontend_tool, interrupts, _wire_id = (
        await _start_two_stage_proxy_result_checkpoint(
            tmp_path,
            hooks=[
                _FixedProxyInterruptHook("stage_one"),
                _PendingStateTamperHook(),
            ],
        )
    )
    live_core = agent._agents_by_thread["thread-1"]
    before = _snapshot_resume_state(live_core)

    failed = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["stage_two"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert [
        event.code for event in failed if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_PROXY_PROVENANCE_ERROR"]
    assert _snapshot_resume_state(live_core) == before
    assert model.turn == 1


@pytest.mark.asyncio
async def test_malformed_pending_proxy_result_state_fails_closed_before_stream(
    tmp_path,
):
    agent, model, frontend_tool, interrupts, _wire_id = (
        await _start_two_stage_proxy_result_checkpoint(tmp_path)
    )
    live_core = agent._agents_by_thread["thread-1"]
    malformed = live_core.state.get(AG_UI_PENDING_PROXY_RESULTS_STATE_KEY)
    malformed["unexpected"] = True
    live_core.state.set(AG_UI_PENDING_PROXY_RESULTS_STATE_KEY, malformed)
    before = _snapshot_resume_state(live_core)

    failed = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["stage_two"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    _assert_proxy_reconciliation_error(failed)
    assert _snapshot_resume_state(live_core) == before
    assert model.turn == 1


@pytest.mark.asyncio
async def test_stored_pending_proxy_binding_conflict_rejects_correct_resend(
    tmp_path,
):
    agent, model, frontend_tool, interrupts, wire_id = (
        await _start_two_stage_proxy_result_checkpoint(tmp_path)
    )
    live_core = agent._agents_by_thread["thread-1"]
    original_pending = live_core.state.get(
        AG_UI_PENDING_PROXY_RESULTS_STATE_KEY
    )
    conflicting_pending = copy.deepcopy(original_pending)
    conflicting_pending["records"]["native-approve"][
        "wire_tool_call_id"
    ] = "well-typed-other-wire"
    live_core.state.set(
        AG_UI_PENDING_PROXY_RESULTS_STATE_KEY, conflicting_pending
    )
    before = _snapshot_resume_state(live_core)

    with patch.object(
        live_core, "stream_async", wraps=live_core.stream_async
    ) as stream_spy:
        failed = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-3-conflict",
                messages=[
                    ToolMessage(
                        id="t-correct-resend",
                        role="tool",
                        tool_call_id=wire_id,
                        content="CLIENT-RESULT",
                    )
                ],
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupts["stage_two"].id,
                        status="resolved",
                        payload=True,
                    )
                ],
                tools=[frontend_tool],
            ),
        )

    stream_spy.assert_not_called()
    _assert_proxy_reconciliation_error(failed)
    assert _snapshot_resume_state(live_core) == before
    assert model.turn == 1

    live_core.state.set(
        AG_UI_PENDING_PROXY_RESULTS_STATE_KEY, original_pending
    )
    retry = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3-retry",
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["stage_two"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )
    assert not any(event.type == EventType.RUN_ERROR for event in retry)
    assert model.turn == 2


@pytest.mark.asyncio
async def test_pending_proxy_result_write_failure_restores_retry_checkpoint(
    tmp_path,
):
    agent, model, frontend_tool, interrupts, _wire_id = (
        await _start_two_stage_proxy_result_checkpoint(tmp_path)
    )
    live_core = agent._agents_by_thread["thread-1"]
    before = _snapshot_resume_state(live_core)
    original_set = live_core.state.set
    failed_once = False

    def fail_first_pending_write(key, value):
        nonlocal failed_once
        if key == AG_UI_PENDING_PROXY_RESULTS_STATE_KEY and not failed_once:
            failed_once = True
            raise RuntimeError("storage rejected pending state")
        original_set(key, value)

    with patch.object(live_core.state, "set", side_effect=fail_first_pending_write):
        failed = await _collect_events(
            agent,
            _make_run_input(
                run_id="run-3",
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupts["stage_two"].id,
                        status="resolved",
                        payload=True,
                    )
                ],
                tools=[frontend_tool],
            ),
        )

    _assert_proxy_reconciliation_error(failed)
    assert failed_once
    assert _snapshot_resume_state(live_core) == before
    assert model.turn == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("recreate_wrapper", [False, True])
@pytest.mark.parametrize(
    ("client_content", "client_error", "expected_status", "expected_text"),
    [
        pytest.param(
            "CLIENT-RESULT", None, "success", "CLIENT-RESULT", id="success"
        ),
        pytest.param("", "boom", "error", "boom", id="blank-failure"),
    ],
)
async def test_proxy_result_survives_repeated_hook_interrupt_until_execution(
    tmp_path,
    recreate_wrapper,
    client_content,
    client_error,
    expected_status,
    expected_text,
):
    """An accepted raw result survives every hook stage until exact execution."""
    model = _ProxyHookInterruptFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    core = StrandsAgentCore(
        model=model,
        system_prompt="test",
        agent_id="proxy-hook-group-agent",
    )
    agent = StrandsAgent(
        core,
        name="grouped-proxy-hook",
        config=config,
        hooks=[
            _FixedProxyInterruptHook("stage_one"),
            _FixedProxyInterruptHook("stage_two"),
        ],
    )
    frontend_tool = Tool(
        name="approveTool", description="approve", parameters={}
    )
    initial_events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[frontend_tool],
        ),
    )

    initial_finished = next(
        event for event in initial_events if event.type == EventType.RUN_FINISHED
    )
    interrupts = {
        interrupt.metadata["strands_name"]: interrupt
        for interrupt in initial_finished.outcome.interrupts
    }
    assert set(interrupts) == {"stage_one", "stage_two"}
    wire_id = next(
        event.tool_call_id
        for event in initial_events
        if event.type == EventType.TOOL_CALL_START
        and event.tool_call_name == "approveTool"
    )
    live_core = agent._agents_by_thread["thread-1"]
    initial_provenance = live_core.state.get(
        "__ag_ui_proxy_hook_provenance__"
    )
    records = initial_provenance["records"]
    assert {
        record["original_native_tool_call_id"] for record in records.values()
    } == {"native-approve"}
    assert {record["wire_tool_call_id"] for record in records.values()} == {
        wire_id
    }

    partial_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="t-result",
                    role="tool",
                    tool_call_id=wire_id,
                    content=client_content,
                    error=client_error,
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["stage_one"].id,
                    status="resolved",
                    payload=False,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in partial_events)
    partial_finished = next(
        event for event in partial_events if event.type == EventType.RUN_FINISHED
    )
    assert [
        interrupt.metadata["strands_name"]
        for interrupt in partial_finished.outcome.interrupts
    ] == ["stage_two"]
    stored_interrupts = live_core._interrupt_state.interrupts
    assert stored_interrupts[interrupts["stage_one"].id].response == {
        "response": False
    }
    assert not stored_interrupts[interrupts["stage_two"].id].response
    pending_discovery = getattr(
        strands_agent_module, "_pending_proxy_hook_native_ids", None
    )
    assert pending_discovery is not None
    tool_call_meta = live_core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)
    assert pending_discovery(live_core, tool_call_meta) == {"native-approve"}
    assert getattr(
        live_core.tool_registry.registry["approveTool"], "_ag_ui_proxy", False
    )
    assert live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY)[wire_id] == (
        "native-approve"
    )
    assert tool_call_meta["native-approve"]["wire_tool_call_id"] == wire_id
    assert live_core.state.get("__ag_ui_proxy_hook_provenance__") == (
        initial_provenance
    )
    assert live_core.state.get(AG_UI_PENDING_PROXY_RESULTS_STATE_KEY) == {
        "version": 1,
        "records": {
            "native-approve": {
                "wire_tool_call_id": wire_id,
                "content": client_content,
                "status": expected_status,
                "error": client_error,
            }
        },
    }
    assert model.turn == 1
    encoded_events = "".join(
        EventEncoder().encode(event) for event in partial_events
    )
    assert "Forwarded to client" not in encoded_events
    assert AG_UI_PENDING_PROXY_RESULTS_STATE_KEY not in encoded_events

    if recreate_wrapper:
        model = _ProxyHookInterruptFlowModel()
        model.turn = 1
        core = StrandsAgentCore(
            model=model,
            system_prompt="test",
            agent_id="proxy-hook-group-agent",
        )
        agent = StrandsAgent(
            core,
            name="grouped-proxy-hook",
            config=config,
            hooks=[
                _FixedProxyInterruptHook("stage_one"),
                _FixedProxyInterruptHook("stage_two"),
            ],
        )

    completed_events = await _collect_events(
        agent,
        _make_run_input(
            run_id="run-3",
            resume=[
                ResumeEntry(
                    interrupt_id=interrupts["stage_two"].id,
                    status="resolved",
                    payload=True,
                )
            ],
            tools=[frontend_tool],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in completed_events)
    completed = next(
        event for event in completed_events if event.type == EventType.RUN_FINISHED
    )
    assert completed.outcome is None
    live_core = agent._agents_by_thread["thread-1"]
    observed = [
        block["toolResult"]
        for message in model.stream_calls_messages[-1]
        for block in message.get("content", [])
        if block.get("toolResult", {}).get("toolUseId") == "native-approve"
    ]
    assert observed == [
        {
            "toolUseId": "native-approve",
            "status": expected_status,
            "content": [{"text": expected_text}],
        }
    ]
    assert AG_UI_PENDING_PROXY_RESULTS_STATE_KEY not in live_core.state.get()
    assert wire_id not in live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY)
    assert "__ag_ui_proxy_hook_provenance__" not in live_core.state.get()


@pytest.mark.asyncio
async def test_parallel_proxy_hook_same_alias_fails_closed(tmp_path):
    """One caller alias cannot ambiguously claim two native proxy calls."""
    model = _InterruptFlowModel(
        sibling_tool_names=("approveTool", "approveTool")
    )
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=_SameAliasHook(),
        config=config,
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="both")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert [
        event.code for event in events if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_PROXY_PROVENANCE_ERROR"]
    assert model.turn == 1
    assert not any(event.type == EventType.RUN_FINISHED for event in events)
    live_core = agent._agents_by_thread["thread-1"]
    assert live_core._interrupt_state == _InterruptState()
    assert not live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY)
    assert not live_core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)
    assert "__ag_ui_proxy_hook_provenance__" not in live_core.state.get()

@pytest.mark.asyncio
async def test_parallel_proxy_sibling_binding_tamper_fails_closed(tmp_path):
    """A later sibling cannot make an earlier advertised call unresumable."""
    model = _InterruptFlowModel(
        sibling_tool_names=("approveTool", "approveTool")
    )
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=_CrossSiblingMapTamperHook(),
        config=config,
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="both")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert [
        event.code for event in events if event.type == EventType.RUN_ERROR
    ] == ["INTERRUPT_PROXY_PROVENANCE_ERROR"]
    assert not any(event.type == EventType.RUN_FINISHED for event in events)
    live_core = agent._agents_by_thread["thread-1"]
    assert live_core._interrupt_state == _InterruptState()
    assert not live_core.state.get(AG_UI_WIRE_MAP_STATE_KEY)
    assert not live_core.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)

@pytest.mark.asyncio
async def test_agent_initialized_proxy_interrupt_runs_before_finalizer(tmp_path):
    """Constructor-time callback registration remains inside the boundary."""
    model = _ProxyHookInterruptFlowModel()
    config = StrandsAgentConfig(
        session_manager_provider=lambda input_data: FileSessionManager(
            session_id=input_data.thread_id, storage_dir=str(tmp_path)
        ),
    )
    agent = _proxy_hook_agent(
        model,
        hook=_LateRegisteredInterruptHook(),
        config=config,
    )

    events = await _collect_events(
        agent,
        _make_run_input(
            messages=[UserMessage(id="u1", role="user", content="approve")],
            tools=[Tool(name="approveTool", description="approve", parameters={})],
        ),
    )

    assert not any(event.type == EventType.RUN_ERROR for event in events)
    wire_id, interrupt = _proxy_wire_and_interrupt(events)
    assert interrupt.tool_call_id == wire_id
    assert interrupt.metadata["strands_name"] == "initialized_proxy"
