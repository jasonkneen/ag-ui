"""HTTP contract for an explicitly waiting frontend tool."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest
from ag_ui.core import RunAgentInput, Tool, ToolMessage, UserMessage
from strands import Agent
from strands.models.model import Model

from ag_ui_strands import create_strands_app
from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig, ToolBehavior


class _WaitingToolModel(Model):
    """Emit one deterministic frontend tool call, then continue from its result."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_messages: list[list[dict[str, Any]]] = []

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(
        self, output_model, prompt, **kwargs
    ):  # pragma: no cover
        if False:
            yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.seen_messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 1:
            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "native-client-wait",
                            "name": "client_wait",
                        }
                    }
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": '{"value":"requested"}'}}
                }
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return
        yield {"contentBlockDelta": {"delta": {"text": "continued"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def _input(*, run_id: str, messages: list[Any]) -> RunAgentInput:
    return RunAgentInput(
        thread_id="client-contract-thread",
        run_id=run_id,
        state={},
        messages=messages,
        tools=[
            Tool(
                name="client_wait",
                description="Wait for the client",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            )
        ],
        context=[],
        forwarded_props={},
    )


def _decode_sse(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


async def _post(app: Any, input_data: RunAgentInput) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/agent",
            json=input_data.model_dump(mode="json", by_alias=True, exclude_none=True),
            headers={"accept": "text/event-stream"},
        )
    assert response.status_code == 200
    return _decode_sse(response.text)


@pytest.mark.asyncio
async def test_false_mode_preserves_tool_message_endpoint_contract() -> None:
    model = _WaitingToolModel()
    adapter = StrandsAgent(
        Agent(model=model, tools=[]),
        name="client-contract",
        config=StrandsAgentConfig(
            tool_behaviors={
                "client_wait": ToolBehavior(continue_after_frontend_call=False)
            }
        ),
    )
    app = create_strands_app(adapter, path="/agent", ping_path=None)

    first = await _post(
        app,
        _input(
            run_id="run-1",
            messages=[UserMessage(id="user-1", content="use the client tool")],
        ),
    )

    first_types = [event["type"] for event in first]
    assert first_types.count("TOOL_CALL_START") == 1
    assert first_types.count("TOOL_CALL_ARGS") == 1
    assert first_types.count("TOOL_CALL_END") == 1
    assert not any(event["type"] == "TOOL_CALL_RESULT" for event in first)
    first_finished = next(event for event in first if event["type"] == "RUN_FINISHED")
    assert first_finished["outcome"] == {"type": "success"}
    tool_call_id = next(
        event["toolCallId"] for event in first if event["type"] == "TOOL_CALL_START"
    )
    assert tool_call_id == "native-client-wait"

    second = await _post(
        app,
        _input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="client-result",
                    tool_call_id=tool_call_id,
                    content='{"accepted":true}',
                )
            ],
        ),
    )

    assert model.calls == 2
    assert any(event.get("delta") == "continued" for event in second)
    assert not any(event["type"] == "TOOL_CALL_START" for event in second)
    assert not any(event["type"] == "TOOL_CALL_RESULT" for event in second)
    second_finished = next(
        event for event in second if event["type"] == "RUN_FINISHED"
    )
    assert second_finished["outcome"] == {"type": "success"}
    assert "{\"accepted\":true}" in repr(model.seen_messages[-1])
