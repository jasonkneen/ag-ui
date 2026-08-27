"""HTTP contract for a frontend tool that waits on a native Strands interrupt."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest
from ag_ui.core import (
    ResumeEntry,
    RunAgentInput,
    Tool,
    ToolMessage,
    UserMessage,
)
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


def _input(
    *,
    run_id: str,
    messages: list[Any],
    thread_id: str = "client-contract-thread",
    resume: list[Any] | None = None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
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
        resume=resume,
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


def _adapter(behavior: ToolBehavior | None) -> tuple[StrandsAgent, _WaitingToolModel]:
    model = _WaitingToolModel()
    adapter = StrandsAgent(
        Agent(model=model, tools=[]),
        name="client-contract",
        config=StrandsAgentConfig(
            tool_behaviors={"client_wait": behavior} if behavior else {}
        ),
    )
    return adapter, model


def _finished(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(event for event in events if event["type"] == "RUN_FINISHED")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "behavior",
    [None, ToolBehavior(continue_after_frontend_call=False)],
    ids=["unconfigured", "explicit-false"],
)
async def test_waiting_tool_reports_a_pause_and_resumes_from_a_tool_message(
    behavior: ToolBehavior | None,
) -> None:
    """The default and an explicit ``False`` both wait, and both say so.

    A run parked on a frontend wait finishes with an interrupt outcome carrying
    the exact emitted ``toolCallId``, so the client can tell a pause from a
    completion. Answering with an ordinary ``ToolMessage`` still works and
    drives exactly one continuation.
    """
    adapter, model = _adapter(behavior)
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

    tool_call_id = next(
        event["toolCallId"] for event in first if event["type"] == "TOOL_CALL_START"
    )
    assert tool_call_id == "native-client-wait"

    outcome = _finished(first)["outcome"]
    assert outcome["type"] == "interrupt"
    [interrupt] = outcome["interrupts"]
    assert interrupt["reason"] == "frontend_tool_call"
    assert interrupt["toolCallId"] == tool_call_id
    assert interrupt["responseSchema"]["required"] == ["content"]

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
    assert _finished(second)["outcome"] == {"type": "success"}
    assert '{"accepted":true}' in repr(model.seen_messages[-1])


@pytest.mark.asyncio
async def test_waiting_tool_resumes_through_the_canonical_resume_channel() -> None:
    """``resume[]`` answers the published interrupt without a ``ToolMessage``."""
    adapter, model = _adapter(None)
    app = create_strands_app(adapter, path="/agent", ping_path=None)

    first = await _post(
        app,
        _input(
            run_id="run-1",
            thread_id="canonical-resume-thread",
            messages=[UserMessage(id="user-1", content="use the client tool")],
        ),
    )
    [interrupt] = _finished(first)["outcome"]["interrupts"]

    second = await _post(
        app,
        _input(
            run_id="run-2",
            thread_id="canonical-resume-thread",
            messages=[],
            resume=[
                ResumeEntry(
                    interrupt_id=interrupt["id"],
                    status="resolved",
                    payload={"content": '{"accepted":true}', "error": False},
                )
            ],
        ),
    )

    assert model.calls == 2
    assert _finished(second)["outcome"] == {"type": "success"}
    assert '{"accepted":true}' in repr(model.seen_messages[-1])


@pytest.mark.asyncio
async def test_cancelling_a_wait_reaches_the_model_as_a_failed_tool_call() -> None:
    """A cancelled wait still closes the tool call, as an error."""
    adapter, model = _adapter(None)
    app = create_strands_app(adapter, path="/agent", ping_path=None)

    first = await _post(
        app,
        _input(
            run_id="run-1",
            thread_id="cancel-thread",
            messages=[UserMessage(id="user-1", content="use the client tool")],
        ),
    )
    [interrupt] = _finished(first)["outcome"]["interrupts"]

    second = await _post(
        app,
        _input(
            run_id="run-2",
            thread_id="cancel-thread",
            messages=[],
            resume=[
                ResumeEntry(interrupt_id=interrupt["id"], status="cancelled")
            ],
        ),
    )

    assert model.calls == 2
    assert _finished(second)["outcome"] == {"type": "success"}
    transcript = repr(model.seen_messages[-1])
    assert "cancelled by the client" in transcript
    assert "'status': 'error'" in transcript


@pytest.mark.asyncio
async def test_continue_after_frontend_call_true_finishes_the_run() -> None:
    """The opt-out keeps the legacy placeholder-and-continue contract."""
    adapter, model = _adapter(ToolBehavior(continue_after_frontend_call=True))
    app = create_strands_app(adapter, path="/agent", ping_path=None)

    events = await _post(
        app,
        _input(
            run_id="run-1",
            thread_id="continue-thread",
            messages=[UserMessage(id="user-1", content="use the client tool")],
        ),
    )

    assert _finished(events)["outcome"] == {"type": "success"}
    # The model runs straight on from the placeholder rather than pausing.
    assert model.calls == 2
