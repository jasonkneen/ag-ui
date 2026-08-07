"""Tests for the RAW fallback and agent-as-tool inner-event forwarding.

Covers two halves of the same structural gap in the Strands adapter:

Issue #2291 — the main stream loop's ``if/elif`` chain has no terminal
``else``, so any event the adapter does not recognise is dropped silently.
Bedrock citation events are one such event: Strands surfaces them as
``{"callback": {"citation": ..., "delta": ...}}``, which matches no branch.

Issue #2304 — a Strands generator tool that wraps another ``Agent``
(agent-as-tool) surfaces the inner agent's whole event stream as
``tool_stream_event`` payloads. The only branch reading those handles
``state`` snapshots and A2UI progress, so the inner agent's tool calls never
reach the frontend.

Both suites drive a REAL ``strands.Agent`` (parent and inner) over a scripted
model provider that replays Bedrock-shaped stream chunks. Nothing here
hand-builds the adapter's input events — they come out of Strands' own event
loop, so the assertions hold against real framework behaviour.
"""

from __future__ import annotations

from typing import Any, AsyncIterable, Optional
from unittest.mock import MagicMock

import pytest
from ag_ui.core import EventType, RunAgentInput, UserMessage
from strands import Agent as StrandsAgentCore
from strands import tool
from strands.models.model import Model

from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig


# ---------------------------------------------------------------------------
# Scripted model provider
# ---------------------------------------------------------------------------


class ScriptedModel(Model):
    """Replays canned Bedrock-shaped stream turns, one turn per invocation."""

    def __init__(self, turns: list[list[dict]]) -> None:
        self._turns = list(turns)
        self.calls = 0

    def update_config(self, **model_config: Any) -> None:  # pragma: no cover
        pass

    def get_config(self) -> Any:  # pragma: no cover
        return {}

    def structured_output(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    async def stream(
        self,
        messages: Any,
        tool_specs: Optional[list] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterable[dict]:
        turn = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        for event in turn:
            yield event


def _text_turn(text: str, citation: dict | None = None) -> list[dict]:
    events: list[dict] = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": text}}},
    ]
    if citation is not None:
        events.append({"contentBlockDelta": {"delta": {"citation": citation}}})
    events += [
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    return events


def _tool_turn(tool_use_id: str, name: str, input_json: str) -> list[dict]:
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": input_json}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


CITATION = {
    "title": "quarterly-report.pdf",
    "sourceContent": [{"text": "revenue grew 12%"}],
    "location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 26}},
}


# ---------------------------------------------------------------------------
# Adapter wiring
# ---------------------------------------------------------------------------


def _template_agent() -> MagicMock:
    mock = MagicMock()
    mock.model = MagicMock()
    mock.system_prompt = "You are helpful"
    mock.tool_registry.registry = {}
    mock.record_direct_tool_call = True
    return mock


def _wrap(strands_agent: StrandsAgentCore, thread_id: str = "t1") -> StrandsAgent:
    agent = StrandsAgent(
        _template_agent(), name="test-agent", config=StrandsAgentConfig()
    )
    agent._agents_by_thread[thread_id] = strands_agent
    return agent


def _run_input(thread_id: str = "t1") -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id="r1",
        state={},
        messages=[UserMessage(id="u1", role="user", content="hello")],
        tools=[],
        context=[],
        forwarded_props={},
    )


async def _collect(agent: StrandsAgent) -> list:
    return [event async for event in agent.run(_run_input())]


# ---------------------------------------------------------------------------
# Nested-agent fixture (agent-as-tool)
# ---------------------------------------------------------------------------


def _nested_agent(
    parent_tool_use_id: str = "tooluse_parent_1",
    inner_tool_use_id: str = "tooluse_inner_1",
) -> StrandsAgentCore:
    """A real parent Agent whose only tool wraps a real inner Agent.

    The inner agent calls a real ``@tool`` of its own; Strands surfaces the
    whole inner stream to the parent as ``tool_stream_event`` payloads.
    """

    @tool
    def lookup_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f"sunny in {city}"

    inner = StrandsAgentCore(
        model=ScriptedModel(
            [
                _tool_turn(inner_tool_use_id, "lookup_weather", '{"city": "Paris"}'),
                _text_turn("Paris is sunny."),
            ]
        ),
        tools=[lookup_weather],
        callback_handler=None,
    )

    @tool
    async def research_agent(query: str):
        """Delegate research to a sub-agent."""
        async for event in inner.stream_async(query):
            yield event

    return StrandsAgentCore(
        model=ScriptedModel(
            [
                _tool_turn(parent_tool_use_id, "research_agent", '{"query": "weather"}'),
                _text_turn("Done."),
            ]
        ),
        tools=[research_agent],
        callback_handler=None,
    )


# ---------------------------------------------------------------------------
# Issue #2291 — RAW fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bedrock_citation_event_is_emitted_as_raw():
    """A citation delta must reach the wire instead of being dropped."""
    strands_agent = StrandsAgentCore(
        model=ScriptedModel([_text_turn("Revenue grew.", citation=CITATION)]),
        callback_handler=None,
    )

    events = await _collect(_wrap(strands_agent))

    raw_events = [e for e in events if e.type == EventType.RAW]
    citation_raws = [
        e
        for e in raw_events
        if isinstance(e.event, dict) and "citation" in (e.event.get("callback") or {})
    ]
    assert citation_raws, (
        "expected a RAW event carrying the Bedrock citation payload; "
        f"got RAW events: {[e.event for e in raw_events]}"
    )
    assert citation_raws[0].source == "strands"
    assert citation_raws[0].event["callback"]["citation"] == CITATION


@pytest.mark.asyncio
async def test_lifecycle_events_are_not_emitted_as_raw():
    """The deliberate plumbing skips must stay silent, not turn into RAW."""
    strands_agent = StrandsAgentCore(
        model=ScriptedModel([_text_turn("hi")]),
        callback_handler=None,
    )

    events = await _collect(_wrap(strands_agent))

    lifecycle_keys = {"init_event_loop", "start_event_loop", "start", "complete", "force_stop"}
    leaked = [
        e.event
        for e in events
        if e.type == EventType.RAW
        and isinstance(e.event, dict)
        and lifecycle_keys & set(e.event)
    ]
    assert leaked == [], f"lifecycle events leaked as RAW: {leaked}"


@pytest.mark.asyncio
async def test_raw_fallback_does_not_disturb_mapped_events():
    """Text still streams normally alongside the new RAW fallback."""
    strands_agent = StrandsAgentCore(
        model=ScriptedModel([_text_turn("Revenue grew.", citation=CITATION)]),
        callback_handler=None,
    )

    events = await _collect(_wrap(strands_agent))

    deltas = "".join(
        e.delta for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT
    )
    assert deltas == "Revenue grew."
    assert events[-1].type == EventType.RUN_FINISHED


# ---------------------------------------------------------------------------
# Issue #2304 — agent-as-tool inner event forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inner_agent_tool_call_lifecycle_is_forwarded():
    """The sub-agent's own tool call must produce start/args/end/result."""
    events = await _collect(_wrap(_nested_agent()))

    starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
    inner_starts = [e for e in starts if e.tool_call_name == "lookup_weather"]
    assert inner_starts, (
        "inner sub-agent tool call was never emitted; "
        f"only saw: {[e.tool_call_name for e in starts]}"
    )

    inner_id = inner_starts[0].tool_call_id

    args = [
        e
        for e in events
        if e.type == EventType.TOOL_CALL_ARGS and e.tool_call_id == inner_id
    ]
    assert "".join(e.delta for e in args) == '{"city": "Paris"}'

    ends = [
        e
        for e in events
        if e.type == EventType.TOOL_CALL_END and e.tool_call_id == inner_id
    ]
    assert len(ends) == 1

    results = [
        e
        for e in events
        if e.type == EventType.TOOL_CALL_RESULT and e.tool_call_id == inner_id
    ]
    assert len(results) == 1
    assert "sunny in Paris" in results[0].content


@pytest.mark.asyncio
async def test_inner_tool_call_lifecycle_is_ordered_within_the_parent_call():
    """Inner start/end must land between the parent tool's start and result."""
    events = await _collect(_wrap(_nested_agent()))

    def _index(predicate) -> int:
        return next(i for i, e in enumerate(events) if predicate(e))

    parent_start = _index(
        lambda e: e.type == EventType.TOOL_CALL_START
        and e.tool_call_name == "research_agent"
    )
    inner_start = _index(
        lambda e: e.type == EventType.TOOL_CALL_START
        and e.tool_call_name == "lookup_weather"
    )
    inner_end = _index(
        lambda e: e.type == EventType.TOOL_CALL_END
        and events[inner_start].tool_call_id == e.tool_call_id
    )
    parent_result = _index(
        lambda e: e.type == EventType.TOOL_CALL_RESULT
        and e.tool_call_id == "tooluse_parent_1"
    )

    assert parent_start < inner_start < inner_end < parent_result


@pytest.mark.asyncio
async def test_inner_tool_call_id_cannot_collide_with_a_parent_id():
    """An inner agent reusing the parent's toolUseId must not alias it."""
    events = await _collect(
        _wrap(
            _nested_agent(
                parent_tool_use_id="tooluse_1",
                inner_tool_use_id="tooluse_1",
            )
        )
    )

    starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
    by_name = {e.tool_call_name: e.tool_call_id for e in starts}
    assert by_name["research_agent"] == "tooluse_1"
    assert by_name["lookup_weather"] != by_name["research_agent"]
    assert "tooluse_1" in by_name["lookup_weather"]

    # And the inner result must not resolve the parent's tool card.
    inner_results = [
        e
        for e in events
        if e.type == EventType.TOOL_CALL_RESULT
        and e.tool_call_id == by_name["lookup_weather"]
    ]
    assert len(inner_results) == 1
