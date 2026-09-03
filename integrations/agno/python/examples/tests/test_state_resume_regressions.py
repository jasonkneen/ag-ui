from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from ag_ui.core import Tool, ToolMessage, UserMessage
from ag_ui.core.types import RunAgentInput
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.os.interfaces.agui import AGUI
from agno.os.interfaces.agui.input import validate_state
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.api.predictive_state_updates import agent as predictive_state_updates_agent
from server.api.shared_state import agent as shared_state_agent
from server.api.shared_state import app as shared_state_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[3]
DOJO_FILES = REPO_ROOT / "apps" / "dojo" / "src" / "files.json"
AGENTIC_GENERATIVE_UI_PAGE = (
    REPO_ROOT
    / "apps"
    / "dojo"
    / "src"
    / "app"
    / "[integrationId]"
    / "feature"
    / "(v2)"
    / "agentic_generative_ui"
    / "page.tsx"
)


def _generated_agentic_generative_ui_page() -> str:
    catalog = json.loads(DOJO_FILES.read_text())
    return next(
        entry["content"]
        for entry in catalog["agno::agentic_generative_ui"]
        if entry["name"] == "page.tsx"
    )


class _PredictiveHitlModel(Model):
    calls: int = 0
    tool_names_by_call: list[set[str]]
    tool_results_by_call: list[list[tuple[str | None, object]]]

    def __post_init__(self) -> None:
        self.tool_names_by_call = []
        self.tool_results_by_call = []

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield ModelResponse()

    async def ainvoke_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ModelResponse]:
        self.calls += 1
        tools = kwargs.get("tools") or []
        self.tool_names_by_call.append(
            {tool["function"]["name"] for tool in tools if "function" in tool}
        )
        messages = kwargs.get("messages") or []
        tool_results = [
            (message.tool_call_id, message.content)
            for message in messages
            if getattr(message, "role", None) == "tool"
        ]
        self.tool_results_by_call.append(tool_results)

        if self.calls == 1:
            yield ModelResponse(
                tool_calls=[
                    {
                        "id": "write-document-1",
                        "type": "function",
                        "function": {
                            "name": "write_document",
                            "arguments": '{"document":"# Revised document"}',
                        },
                    }
                ]
            )
        else:
            yield ModelResponse(content="The reviewed document was accepted.")

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


class _PlainReplyModel(Model):
    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield ModelResponse()

    async def ainvoke_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ModelResponse]:
        yield ModelResponse(content="The recipe is unchanged.")

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


def _sse_events(response_text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data:"))
        for line in response_text.splitlines()
        if line.startswith("data:")
    ]


def _run_input(
    *, run_id: str, messages: list[UserMessage | ToolMessage]
) -> RunAgentInput:
    return RunAgentInput(
        threadId="predictive-resume-thread",
        runId=run_id,
        state={"document": "# Revised document"},
        messages=messages,
        tools=[
            Tool(
                name="write_document",
                description="Present the proposed document for review",
                parameters={
                    "type": "object",
                    "properties": {"document": {"type": "string"}},
                    "required": ["document"],
                },
            )
        ],
        context=[],
        forwardedProps={},
    )


class StateAndResumeRegressionTests(unittest.TestCase):
    def test_predictive_hitl_pauses_and_resumes_through_the_stock_route(self) -> None:
        model = _PredictiveHitlModel(id="predictive-hitl-test")
        original_model = predictive_state_updates_agent.model
        predictive_state_updates_agent.model = model
        self.addCleanup(
            setattr,
            predictive_state_updates_agent,
            "model",
            original_model,
        )

        app = FastAPI()
        app.include_router(AGUI(agent=predictive_state_updates_agent).get_router())
        client = TestClient(app)

        paused_response = client.post(
            "/agui",
            json=_run_input(
                run_id="predictive-paused-run",
                messages=[UserMessage(id="user-1", content="Revise the document")],
            ).model_dump(by_alias=True),
        )
        resumed_response = client.post(
            "/agui",
            json=_run_input(
                run_id="incoming-resume-request",
                messages=[
                    ToolMessage(
                        id="tool-result-1",
                        toolCallId="write-document-1",
                        content='{"accepted":true}',
                    )
                ],
            ).model_dump(by_alias=True),
        )

        for label, response in {
            "paused": paused_response,
            "resumed": resumed_response,
        }.items():
            with self.subTest(response=label):
                self.assertEqual(response.status_code, 200)
                events = _sse_events(response.text)
                # The stock router folds model exceptions into a RAW RunError
                # payload and still ends with RUN_FINISHED over HTTP 200, so
                # a clean stream has to be asserted explicitly.
                self.assertNotIn("RUN_ERROR", [event["type"] for event in events])
                self.assertNotIn("RunError", response.text)
                self.assertEqual(events[-1]["type"], "RUN_FINISHED")

        paused_events = _sse_events(paused_response.text)
        tool_call_start = next(
            event for event in paused_events if event["type"] == "TOOL_CALL_START"
        )
        self.assertEqual(tool_call_start["toolCallName"], "write_document")
        self.assertEqual(tool_call_start["toolCallId"], "write-document-1")
        tool_call_end_index = paused_events.index(
            {"type": "TOOL_CALL_END", "toolCallId": "write-document-1"}
        )
        self.assertGreater(tool_call_end_index, paused_events.index(tool_call_start))
        self.assertEqual(
            [event["type"] for event in paused_events[tool_call_end_index + 1 :]],
            ["STATE_SNAPSHOT", "RUN_FINISHED"],
        )

        resumed_events = _sse_events(resumed_response.text)
        self.assertEqual(
            "".join(
                event["delta"]
                for event in resumed_events
                if event["type"] == "TEXT_MESSAGE_CONTENT"
            ),
            "The reviewed document was accepted.",
        )

        self.assertEqual(model.calls, 2)
        self.assertIn("write_document", model.tool_names_by_call[0])
        self.assertIn("update_session_state", model.tool_names_by_call[0])
        self.assertEqual(
            model.tool_results_by_call[1],
            [("write-document-1", '{"accepted":true}')],
        )

    def test_shared_state_snapshot_excludes_agno_session_bookkeeping(self) -> None:
        model = _PlainReplyModel(id="shared-state-snapshot-test")
        original_model = shared_state_agent.model
        shared_state_agent.model = model
        self.addCleanup(setattr, shared_state_agent, "model", original_model)

        recipe = {
            "title": "Pancakes",
            "skill_level": "Beginner",
            "cooking_time": "15 min",
            "special_preferences": ["Vegetarian"],
            "ingredients": [{"name": "Flour", "amount": "1 cup", "icon": "🌾"}],
            "instructions": ["Mix", "Fry"],
        }
        response = TestClient(shared_state_app).post(
            "/agui",
            json=RunAgentInput(
                threadId="shared-state-snapshot-thread",
                runId="shared-state-snapshot-run",
                state={"recipe": recipe},
                messages=[UserMessage(id="user-1", content="Show me the recipe")],
                tools=[],
                context=[],
                forwardedProps={},
            ).model_dump(by_alias=True),
        )

        self.assertEqual(response.status_code, 200)
        snapshots = [
            event["snapshot"]
            for event in _sse_events(response.text)
            if event["type"] == "STATE_SNAPSHOT"
        ]
        self.assertTrue(snapshots, "no STATE_SNAPSHOT event was streamed")
        final_snapshot = snapshots[-1]
        self.assertEqual(final_snapshot["recipe"], recipe)
        for key in ("current_session_id", "current_user_id", "current_run_id"):
            self.assertNotIn(key, final_snapshot)

    def test_agentic_steps_are_narrowed_at_the_render_boundary(self) -> None:
        malformed_state = validate_state(
            {"steps": "not-an-array"}, "agentic-state-thread"
        )
        mixed_state = validate_state(
            {
                "steps": [
                    None,
                    {"description": "Valid step", "status": "pending"},
                    {"description": 42, "status": "completed"},
                ]
            },
            "agentic-state-thread",
        )

        self.assertEqual(malformed_state, {"steps": "not-an-array"})
        self.assertEqual(len(mixed_state["steps"]), 3)

        source_page = AGENTIC_GENERATIVE_UI_PAGE.read_text()
        generated_page = _generated_agentic_generative_ui_page()
        self.assertEqual(generated_page, source_page)
        for label, page in {"source": source_page, "generated": generated_page}.items():
            with self.subTest(page=label):
                self.assertIn(
                    "function normalizeAgentSteps(state: unknown): AgentStep[]", page
                )
                self.assertIn("if (!Array.isArray(state.steps))", page)
                self.assertIn("return state.steps.filter(isAgentStep);", page)
                self.assertIn("const steps = normalizeAgentSteps(agent.state);", page)
                self.assertNotIn("agent.state as AgentState", page)


if __name__ == "__main__":
    unittest.main()
