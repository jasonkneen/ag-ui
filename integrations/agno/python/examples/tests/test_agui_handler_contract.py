from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import agno.os.interfaces.agui.router as agui_router
from ag_ui.core import (
    Context,
    EventType,
    ImageInputContent,
    InputContentDataSource,
    TextInputContent,
    Tool,
    ToolMessage,
    UserMessage,
)
from ag_ui.core.types import RunAgentInput
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.response import ToolExecution
from agno.os.interfaces.agui.handlers import process_event
from agno.os.interfaces.agui.input import (
    extract_context,
    extract_media,
    extract_tool_messages,
    extract_user_input,
    parse_client_tools,
    validate_state,
)
from agno.os.interfaces.agui.resume import (
    ensure_requirements_resolved,
    resolve_requirements_from_tool_messages,
    resume_paused_run,
)
from agno.os.interfaces.agui.router import run_entity
from agno.os.interfaces.agui.state import StreamState
from agno.os.interfaces.agui.stream import stream_agno_response_as_agui_events
from agno.run.agent import (
    ReasoningCompletedEvent,
    ReasoningContentDeltaEvent,
    RunCompletedEvent,
    RunContentEvent,
    RunOutput,
    RunPausedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from agno.run.base import RunContext, RunStatus
from agno.run.requirement import RunRequirement
from agno.session.agent import AgentSession


def _event_types(events: list) -> list[EventType]:
    return [event.type for event in events]


class _RecordingEntity:
    def __init__(self, run_state: dict[str, object]) -> None:
        self.run_state = run_state
        self.run_kwargs: dict[str, object] | None = None

    def arun(self, **kwargs):
        self.run_kwargs = kwargs

        async def response_stream():
            yield ReasoningContentDeltaEvent(reasoning_content="checking")
            yield ReasoningCompletedEvent()
            yield RunContentEvent(content="done")
            tool = ToolExecution(
                tool_call_id="server-tool-1",
                tool_name="lookup",
                tool_args={"city": "Amsterdam"},
                result="{'temperature': 18}",
            )
            yield ToolCallStartedEvent(tool=tool)
            self.run_state["count"] = 1
            yield ToolCallCompletedEvent(tool=tool)
            yield RunCompletedEvent(session_state=self.run_state)

        return response_stream()


class _FailingEntity:
    def arun(self, **kwargs):
        async def response_stream():
            raise RuntimeError("provider-specific internal failure detail")
            yield

        return response_stream()


class AguiInputContractTests(unittest.TestCase):
    def test_input_helpers_preserve_text_and_multimodal_content(self) -> None:
        messages = [
            UserMessage(id="old", content="ignore this"),
            UserMessage(
                id="current",
                content=[
                    TextInputContent(text="describe the image"),
                    ImageInputContent(
                        source=InputContentDataSource(
                            value="aW1hZ2UtYnl0ZXM=", mimeType="image/png"
                        )
                    ),
                ],
            ),
        ]

        images, audio, videos, files = extract_media(messages)

        self.assertEqual(extract_user_input(messages), "describe the image")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].content, b"image-bytes")
        self.assertEqual(images[0].mime_type, "image/png")
        self.assertEqual((audio, videos, files), ([], [], []))

    def test_input_helpers_build_run_context_values(self) -> None:
        state = {"count": 0}
        context = [
            Context(description="profile", value='{"name":"Ada"}'),
            Context(description="tone", value="concise"),
        ]
        tools = [
            Tool(
                name="change_background",
                description="Change the page background",
                parameters={
                    "type": "object",
                    "properties": {"color": {"type": "string"}},
                },
            )
        ]

        parsed_tools = parse_client_tools(tools)

        self.assertIs(validate_state(state, "thread-1"), state)
        self.assertEqual(
            extract_context(context),
            {"profile": {"name": "Ada"}, "tone": "concise"},
        )
        self.assertEqual(len(parsed_tools), 1)
        self.assertEqual(parsed_tools[0].name, "change_background")
        self.assertTrue(parsed_tools[0].external_execution)
        self.assertTrue(parsed_tools[0].external_execution_silent)


class AguiStreamingContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_stock_router_maps_the_complete_run_contract(self) -> None:
        state: dict[str, object] = {"count": 0}
        entity = _RecordingEntity(state)
        run_input = RunAgentInput(
            threadId="thread-1",
            runId="run-1",
            state=state,
            messages=[
                UserMessage(
                    id="user-1",
                    content=[
                        TextInputContent(text="describe the image"),
                        ImageInputContent(
                            source=InputContentDataSource(
                                value="aW1hZ2UtYnl0ZXM=", mimeType="image/png"
                            )
                        ),
                    ],
                )
            ],
            tools=[
                Tool(
                    name="change_background",
                    description="Change the page background",
                    parameters={"type": "object", "properties": {}},
                )
            ],
            context=[Context(description="profile", value='{"name":"Ada"}')],
            forwardedProps={},
        )

        events = [event async for event in run_entity(entity, run_input)]

        self.assertIsNotNone(entity.run_kwargs)
        run_kwargs = entity.run_kwargs or {}
        self.assertEqual(run_kwargs["input"], "describe the image")
        self.assertEqual(run_kwargs["images"][0].content, b"image-bytes")
        run_context = run_kwargs["run_context"]
        self.assertEqual(run_context.dependencies, {"profile": {"name": "Ada"}})
        self.assertIs(run_context.session_state, state)
        self.assertEqual(run_context.client_tools[0].name, "change_background")
        self.assertTrue(run_kwargs["add_dependencies_to_context"])

        self.assertEqual(
            _event_types(events),
            [
                EventType.RUN_STARTED,
                EventType.STATE_SNAPSHOT,
                EventType.REASONING_START,
                EventType.REASONING_MESSAGE_START,
                EventType.REASONING_MESSAGE_CONTENT,
                EventType.REASONING_MESSAGE_END,
                EventType.REASONING_END,
                EventType.TEXT_MESSAGE_START,
                EventType.TEXT_MESSAGE_CONTENT,
                EventType.TEXT_MESSAGE_END,
                EventType.TOOL_CALL_START,
                EventType.TOOL_CALL_ARGS,
                EventType.TOOL_CALL_END,
                EventType.TOOL_CALL_RESULT,
                EventType.STATE_DELTA,
                EventType.STATE_SNAPSHOT,
                EventType.RUN_FINISHED,
            ],
        )
        tool_result = next(
            event for event in events if event.type == EventType.TOOL_CALL_RESULT
        )
        self.assertEqual(json.loads(tool_result.content), {"temperature": 18})
        state_delta = next(
            event for event in events if event.type == EventType.STATE_DELTA
        )
        self.assertEqual(
            state_delta.delta, [{"op": "replace", "path": "/count", "value": 1}]
        )
        self.assertEqual(events[-2].snapshot, {"count": 1})

    async def test_stock_router_translates_stream_failures_to_run_error(self) -> None:
        run_input = RunAgentInput(
            threadId="thread-1",
            runId="run-1",
            state=None,
            messages=[UserMessage(id="user-1", content="hello")],
            tools=[],
            context=[],
            forwardedProps={},
        )

        with patch.object(agui_router, "log_error"):
            events = [event async for event in run_entity(_FailingEntity(), run_input)]

        self.assertEqual(
            _event_types(events), [EventType.RUN_STARTED, EventType.RUN_ERROR]
        )
        self.assertIsInstance(events[-1].message, str)
        self.assertTrue(events[-1].message)


class AguiHitlContractTests(unittest.IsolatedAsyncioTestCase):
    def test_paused_tools_share_one_assistant_parent_and_keep_pairing(self) -> None:
        tools = [
            ToolExecution(
                tool_call_id="tool-1",
                tool_name="first_tool",
                tool_args={"step": 1},
                external_execution_required=True,
            ),
            ToolExecution(
                tool_call_id="tool-2",
                tool_name="second_tool",
                tool_args={"step": 2},
                external_execution_required=True,
            ),
        ]

        events = list(
            stream_agno_response_as_agui_events(
                iter([RunPausedEvent(content="Choose tools", tools=tools)]),
                thread_id="thread-1",
                run_id="run-1",
            )
        )

        starts = [event for event in events if event.type == EventType.TOOL_CALL_START]
        args = [event for event in events if event.type == EventType.TOOL_CALL_ARGS]
        ends = [event for event in events if event.type == EventType.TOOL_CALL_END]
        parent_ids = {event.parent_message_id for event in starts}

        self.assertEqual([event.tool_call_id for event in starts], ["tool-1", "tool-2"])
        self.assertEqual([event.tool_call_id for event in args], ["tool-1", "tool-2"])
        self.assertEqual([event.tool_call_id for event in ends], ["tool-1", "tool-2"])
        self.assertEqual(len(parent_ids), 1)
        self.assertEqual(events[-1].type, EventType.RUN_FINISHED)

    def test_resume_pairs_external_results_and_confirmation_by_tool_call_id(
        self,
    ) -> None:
        external_tool = ToolExecution(
            tool_call_id="external-1",
            tool_name="change_background",
            tool_args={"color": "blue"},
            external_execution_required=True,
        )
        confirmation_tool = ToolExecution(
            tool_call_id="confirm-1",
            tool_name="publish",
            tool_args={},
            requires_confirmation=True,
        )
        requirements = [
            RunRequirement(external_tool),
            RunRequirement(confirmation_tool),
        ]
        tool_messages = [
            ToolMessage(
                id="result-1",
                toolCallId="external-1",
                content='{"status":"changed"}',
            ),
            ToolMessage(
                id="result-2",
                toolCallId="confirm-1",
                content='{"accepted":true}',
            ),
        ]

        resolved = resolve_requirements_from_tool_messages(requirements, tool_messages)
        ensure_requirements_resolved(resolved)

        self.assertEqual(external_tool.result, '{"status":"changed"}')
        self.assertTrue(confirmation_tool.confirmed)

    async def test_stock_resume_continues_the_matching_paused_run(self) -> None:
        tool = ToolExecution(
            tool_call_id="external-1",
            tool_name="change_background",
            tool_args={"color": "blue"},
            external_execution_required=True,
        )
        paused_run = RunOutput(
            run_id="paused-run",
            session_id="thread-1",
            status=RunStatus.paused,
            requirements=[RunRequirement(tool)],
        )
        entity = Agent(id="resume-agent", db=InMemoryDb())
        entity.aget_session = AsyncMock(
            return_value=AgentSession(session_id="thread-1", runs=[paused_run])
        )

        async def continued_stream():
            yield RunCompletedEvent(run_id="paused-run")

        entity.acontinue_run = MagicMock(return_value=continued_stream())
        run_context = RunContext(run_id="incoming-run", session_id="thread-1")

        stream = await resume_paused_run(
            entity=entity,
            session_id="thread-1",
            tool_messages=[
                ToolMessage(
                    id="result-1",
                    toolCallId="external-1",
                    content='{"status":"changed"}',
                )
            ],
            run_context=run_context,
            run_kwargs={},
        )
        chunks = [chunk async for chunk in stream]

        self.assertEqual([chunk.event for chunk in chunks], ["RunCompleted"])
        self.assertEqual(run_context.run_id, "paused-run")
        continued_requirements = entity.acontinue_run.call_args.kwargs["requirements"]
        self.assertEqual(
            continued_requirements[0].tool_execution.result,
            '{"status":"changed"}',
        )

    def test_requirement_guard_rejects_an_unpaired_tool_result(self) -> None:
        requirement = RunRequirement(
            ToolExecution(
                tool_call_id="missing-result",
                tool_name="change_background",
                tool_args={},
                external_execution_required=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "missing-result"):
            ensure_requirements_resolved([requirement])


class AguiHandlerModuleContractTests(unittest.TestCase):
    def test_current_handler_modules_expose_the_split_pipeline(self) -> None:
        state = StreamState(thread_id="thread-1", run_id="run-1")

        events = process_event(RunContentEvent(content="hello"), state)

        self.assertEqual(
            _event_types(events),
            [EventType.TEXT_MESSAGE_START, EventType.TEXT_MESSAGE_CONTENT],
        )
        self.assertEqual(extract_tool_messages([]), [])


if __name__ == "__main__":
    unittest.main()
