"""Tests for the AG-UI standard input.resume path in prepare_stream.

These tests verify that:
1. input.resume with a single resolved ResumeEntry produces Command(resume=payload).
2. input.resume with a single cancelled ResumeEntry produces Command(resume=sentinel).
3. input.resume takes precedence over forwardedProps.command.resume with a WARN.
4. Legacy forwardedProps.command.resume still works with a deprecation WARN.
5. Active interrupts without any resume emit outcome.interrupt in the short-circuit path.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ag_ui.core import EventType, ResumeEntry, UserMessage

from ag_ui_langgraph import agent as agent_module
from ag_ui_langgraph.interrupts import DEFAULT_RESUME_SENTINEL_CANCELLED
from tests._helpers import make_agent


@dataclass
class FakeInterrupt:
    value: Any
    id: str = None


@dataclass
class FakeTask:
    interrupts: List[FakeInterrupt] = field(default_factory=list)


def _make_state(messages, tasks=None):
    state = MagicMock()
    state.values = {"messages": messages}
    state.tasks = tasks or []
    state.next = []
    state.metadata = {"writes": {}}
    return state


def _make_input(
    messages,
    thread_id="t1",
    forwarded_props=None,
    resume=None,
):
    inp = MagicMock()
    inp.thread_id = thread_id
    inp.messages = messages
    inp.state = {}
    inp.tools = []
    inp.context = []
    inp.run_id = "run-1"
    inp.forwarded_props = forwarded_props or {}
    inp.resume = resume
    return inp


async def _empty_stream():
    if False:
        yield None


class TestInputResumeResolvedSingle(unittest.IsolatedAsyncioTestCase):
    async def test_input_resume_resolved_single(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

        checkpoint_messages = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=checkpoint_messages,
            tasks=[FakeTask(interrupts=[FakeInterrupt(value={"question": "Approve?"})])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            resume=[ResumeEntry(interrupt_id="i1", status="resolved", payload={"approved": True})],
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        agent.prepare_regenerate_stream.assert_not_awaited()
        self.assertIsNotNone(result.get("stream"))

        stream_input = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertIsInstance(stream_input, Command)
        self.assertEqual(stream_input.resume, {"approved": True})


class TestInputResumeCancelled(unittest.IsolatedAsyncioTestCase):
    async def test_input_resume_cancelled(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

        checkpoint_messages = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=checkpoint_messages,
            tasks=[FakeTask(interrupts=[FakeInterrupt(value={"question": "Approve?"})])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            resume=[ResumeEntry(interrupt_id="i1", status="cancelled")],
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        self.assertIsNotNone(result.get("stream"))
        stream_input = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertIsInstance(stream_input, Command)
        self.assertIsInstance(stream_input.resume, dict)
        self.assertTrue(stream_input.resume.get(DEFAULT_RESUME_SENTINEL_CANCELLED))
        self.assertEqual(stream_input.resume.get("interrupt_id"), "i1")


class TestInputResumeTakesPrecedenceOverLegacy(unittest.IsolatedAsyncioTestCase):
    async def test_input_resume_takes_precedence_over_legacy(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

        checkpoint_messages = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=checkpoint_messages,
            tasks=[FakeTask(interrupts=[FakeInterrupt(value={"question": "Approve?"})])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"command": {"resume": "legacy_value"}},
            resume=[ResumeEntry(interrupt_id="i1", status="resolved", payload={"new": True})],
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        with patch.object(agent_module, "logger") as mock_logger:
            result = await agent.prepare_stream(inp, state, config)

        self.assertIsNotNone(result.get("stream"))
        stream_input = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertIsInstance(stream_input, Command)
        self.assertEqual(stream_input.resume, {"new": True})

        warn_calls = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("both input.resume and forwardedProps.command.resume" in c for c in warn_calls),
            f"Expected precedence warning, got: {warn_calls}",
        )


class TestLegacyResumeStillWorks(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_resume_still_works(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

        checkpoint_messages = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=checkpoint_messages,
            tasks=[FakeTask(interrupts=[FakeInterrupt(value={"question": "Approve?"})])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"command": {"resume": "yes"}},
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        with patch.object(agent_module, "logger") as mock_logger:
            result = await agent.prepare_stream(inp, state, config)

        self.assertIsNotNone(result.get("stream"))
        stream_input = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertIsInstance(stream_input, Command)
        self.assertEqual(stream_input.resume, "yes")

        warn_calls = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("deprecated" in c for c in warn_calls),
            f"Expected deprecation warning, got: {warn_calls}",
        )


class TestActiveInterruptsNoResumeEmitsOutcome(unittest.IsolatedAsyncioTestCase):
    async def test_active_interrupts_no_resume_emits_outcome(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

        checkpoint_messages = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=checkpoint_messages,
            tasks=[FakeTask(interrupts=[
                FakeInterrupt(value={"reason": "confirm", "message": "ok?"}, id="int-1"),
            ])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(messages=frontend_messages, forwarded_props={})

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        self.assertIsNone(result.get("stream"))

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertIn(EventType.RUN_STARTED, types)
        self.assertIn(EventType.RUN_FINISHED, types)

        finished_events = [e for e in events if getattr(e, "type", None) == EventType.RUN_FINISHED]
        self.assertEqual(len(finished_events), 1)
        finished = finished_events[0]
        self.assertEqual(finished.outcome.type, "interrupt")
        self.assertEqual(len(finished.outcome.interrupts), 1)
        self.assertEqual(finished.outcome.interrupts[0].id, "int-1")
        self.assertEqual(finished.outcome.interrupts[0].reason, "confirm")
        self.assertEqual(finished.outcome.interrupts[0].message, "ok?")


class TestEmptyResumeArrayTreatedAsAbsent(unittest.IsolatedAsyncioTestCase):
    async def test_empty_resume_array_treated_as_absent(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

        checkpoint_messages = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=checkpoint_messages,
            tasks=[FakeTask(interrupts=[FakeInterrupt(value="confirm?")])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            resume=[],
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        self.assertIsNone(result.get("stream"))

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertIn(EventType.RUN_FINISHED, types)

        finished_events = [e for e in events if getattr(e, "type", None) == EventType.RUN_FINISHED]
        self.assertEqual(finished_events[0].outcome.type, "interrupt")
