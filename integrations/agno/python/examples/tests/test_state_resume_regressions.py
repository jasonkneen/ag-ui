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
            expected_result = ("write-document-1", '{"accepted":true}')
            if expected_result not in tool_results:
                raise AssertionError(
                    "resumed model call did not receive the correlated tool result"
                )
            yield ModelResponse(content="The reviewed document was accepted.")

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


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

        self.assertEqual(paused_response.status_code, 200)
        self.assertIn('"toolCallName":"write_document"', paused_response.text)
        self.assertEqual(resumed_response.status_code, 200)
        self.assertIn("The reviewed document was accepted.", resumed_response.text)
        self.assertEqual(model.calls, 2)
        self.assertIn("write_document", model.tool_names_by_call[0])
        self.assertIn("update_session_state", model.tool_names_by_call[0])
        self.assertIn(
            ("write-document-1", '{"accepted":true}'),
            model.tool_results_by_call[1],
        )

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
