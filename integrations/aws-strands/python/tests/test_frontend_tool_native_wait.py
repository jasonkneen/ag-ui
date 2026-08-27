"""Native-checkpoint authority tests for explicitly waiting frontend tools."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

import pytest
from ag_ui.core import (
    EventType,
    ResumeEntry,
    RunAgentInput,
    Tool,
    ToolMessage,
    UserMessage,
)
from strands import Agent, ToolContext, tool
from strands.models.model import Model
from strands.session.file_session_manager import FileSessionManager

from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig, ToolBehavior
from ag_ui_strands.frontend_tool_interrupt import index_frontend_tool_interrupts


class _ParallelWaitModel(Model):
    """Emit two frontend calls together, then continue once both resolve."""

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
            for index, name in enumerate(("first_client_tool", "second_client_tool")):
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": f"native-{index}",
                                "name": name,
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
        yield {"contentBlockDelta": {"delta": {"text": "continued once"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


class _MixedWaitModel(_ParallelWaitModel):
    """Emit one frontend wait and one ordinary native interrupt together."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.seen_messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 1:
            for tool_use_id, name in (
                ("native-client", "first_client_tool"),
                ("native-server", "server_approval"),
            ):
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": tool_use_id,
                                "name": name,
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
            return
        yield {"contentBlockDelta": {"delta": {"text": "mixed continued"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


@tool(name="server_approval", description="Approve server work", context=True)
def _server_approval(tool_context: ToolContext) -> str:
    response = tool_context.interrupt(
        "server_approval",
        reason={"question": "approve?"},
    )
    return f"server response: {response!r}"


def _tools() -> list[Tool]:
    return [
        Tool(name=name, description=name, parameters={"type": "object"})
        for name in ("first_client_tool", "second_client_tool")
    ]


def _input(
    thread_id: str,
    *,
    run_id: str,
    messages: Sequence[Any],
    resume: Sequence[ResumeEntry] | None = None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        state={},
        messages=list(messages),
        tools=_tools(),
        context=[],
        forwarded_props={},
        resume=list(resume) if resume is not None else None,
    )


def _adapter(
    model: Model,
    storage_dir: Path,
    thread_id: str,
    *,
    core_tools: Sequence[Any] = (),
) -> StrandsAgent:
    async def session_manager_provider(_input_data: RunAgentInput):
        return FileSessionManager(
            session_id=thread_id,
            storage_dir=str(storage_dir),
        )

    return StrandsAgent(
        Agent(
            model=model,
            tools=list(core_tools),
            agent_id="stable-native-wait-agent",
        ),
        name="native-wait-test",
        config=StrandsAgentConfig(
            session_manager_provider=session_manager_provider,
            tool_behaviors={
                tool.name: ToolBehavior(continue_after_frontend_call=False)
                for tool in _tools()
            },
        ),
    )


async def _collect(
    adapter: StrandsAgent,
    input_data: RunAgentInput,
) -> list[Any]:
    return [event async for event in adapter.run(input_data)]


def _assert_success(events: Sequence[Any]) -> None:
    assert not any(event.type == EventType.RUN_ERROR for event in events)
    [finished] = [
        event for event in events if event.type == EventType.RUN_FINISHED
    ]
    assert getattr(finished.outcome, "type", None) == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unconfigured", "continue"])
async def test_legacy_placeholder_modes_also_emit_native_tool_ids(mode: str) -> None:
    model = _ParallelWaitModel()
    behaviors = (
        {}
        if mode == "unconfigured"
        else {
            tool.name: ToolBehavior(continue_after_frontend_call=True)
            for tool in _tools()
        }
    )
    adapter = StrandsAgent(
        Agent(model=model, tools=[]),
        name=f"native-id-{mode}",
        config=StrandsAgentConfig(tool_behaviors=behaviors),
    )

    events = await _collect(
        adapter,
        _input(
            f"native-id-{mode}",
            run_id="run-1",
            messages=[UserMessage(id="user-1", content="call both tools")],
        ),
    )

    assert [
        event.tool_call_id
        for event in events
        if event.type == EventType.TOOL_CALL_START
    ] == ["native-0", "native-1"]


@pytest.mark.asyncio
async def test_partial_native_wait_survives_fresh_wrapper_and_continues_once(
    tmp_path: Path,
) -> None:
    thread_id = "partial-native-wait"
    model = _ParallelWaitModel()

    first_adapter = _adapter(model, tmp_path, thread_id)
    first = await _collect(
        first_adapter,
        _input(
            thread_id,
            run_id="run-1",
            messages=[UserMessage(id="user-1", content="call both tools")],
        ),
    )

    _assert_success(first)
    assert model.calls == 1
    assert [
        event.tool_call_id
        for event in first
        if event.type == EventType.TOOL_CALL_START
    ] == ["native-0", "native-1"]

    partial_adapter = _adapter(model, tmp_path, thread_id)
    partial = await _collect(
        partial_adapter,
        _input(
            thread_id,
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="second-result",
                    tool_call_id="native-1",
                    content="second-value",
                )
            ],
        ),
    )

    _assert_success(partial)
    assert model.calls == 1
    partial_core = partial_adapter._agents_by_thread[thread_id]
    partial_interrupts = index_frontend_tool_interrupts(partial_core)
    assert set(partial_interrupts) == {"native-0", "native-1"}
    assert partial_interrupts["native-0"].response is None
    assert partial_interrupts["native-1"].response is not None
    assert "ag_ui_frontend_tool_wait" not in partial_core.state.get()

    final_adapter = _adapter(model, tmp_path, thread_id)
    final = await _collect(
        final_adapter,
        _input(
            thread_id,
            run_id="run-3",
            messages=[
                ToolMessage(
                    id="first-result",
                    tool_call_id="native-0",
                    content="first-value",
                )
            ],
        ),
    )

    _assert_success(final)
    assert model.calls == 2
    assert sum(event.type == EventType.TEXT_MESSAGE_START for event in final) == 1
    assert not any(event.type == EventType.TOOL_CALL_RESULT for event in partial)
    assert not any(event.type == EventType.TOOL_CALL_RESULT for event in final)
    final_messages = repr(model.seen_messages[-1])
    assert "first-value" in final_messages
    assert "second-value" in final_messages


@pytest.mark.asyncio
async def test_mixed_checkpoint_accepts_server_response_before_frontend_result(
    tmp_path: Path,
) -> None:
    thread_id = "mixed-native-wait"
    model = _MixedWaitModel()

    first_adapter = _adapter(
        model,
        tmp_path,
        thread_id,
        core_tools=[_server_approval],
    )
    first = await _collect(
        first_adapter,
        _input(
            thread_id,
            run_id="run-1",
            messages=[UserMessage(id="user-1", content="call both")],
        ),
    )

    assert model.calls == 1
    [first_finished] = [
        event for event in first if event.type == EventType.RUN_FINISHED
    ]
    assert first_finished.outcome.type == "interrupt"
    [server_interrupt] = first_finished.outcome.interrupts
    assert server_interrupt.reason == "server_approval"

    server_adapter = _adapter(
        model,
        tmp_path,
        thread_id,
        core_tools=[_server_approval],
    )
    server_only = await _collect(
        server_adapter,
        _input(
            thread_id,
            run_id="run-2",
            messages=[],
            resume=[
                ResumeEntry(
                    interrupt_id=server_interrupt.id,
                    status="resolved",
                    payload={"approved": True},
                )
            ],
        ),
    )

    _assert_success(server_only)
    assert model.calls == 1

    frontend_adapter = _adapter(
        model,
        tmp_path,
        thread_id,
        core_tools=[_server_approval],
    )
    final = await _collect(
        frontend_adapter,
        _input(
            thread_id,
            run_id="run-3",
            messages=[
                ToolMessage(
                    id="frontend-result",
                    tool_call_id="native-client",
                    content="frontend-value",
                )
            ],
        ),
    )

    _assert_success(final)
    assert model.calls == 2
    final_messages = repr(model.seen_messages[-1])
    assert "frontend-value" in final_messages
    assert "approved" in final_messages


@pytest.mark.asyncio
async def test_mixed_checkpoint_accepts_both_client_channels_in_one_request(
    tmp_path: Path,
) -> None:
    thread_id = "mixed-native-wait-combined"
    model = _MixedWaitModel()
    first_adapter = _adapter(
        model,
        tmp_path,
        thread_id,
        core_tools=[_server_approval],
    )
    first = await _collect(
        first_adapter,
        _input(
            thread_id,
            run_id="run-1",
            messages=[UserMessage(id="user-1", content="call both")],
        ),
    )
    [finished] = [event for event in first if event.type == EventType.RUN_FINISHED]
    [server_interrupt] = finished.outcome.interrupts

    resume_adapter = _adapter(
        model,
        tmp_path,
        thread_id,
        core_tools=[_server_approval],
    )
    resumed = await _collect(
        resume_adapter,
        _input(
            thread_id,
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="frontend-result",
                    tool_call_id="native-client",
                    content="frontend-value",
                )
            ],
            resume=[
                ResumeEntry(
                    interrupt_id=server_interrupt.id,
                    status="resolved",
                    payload={"approved": True},
                )
            ],
        ),
    )

    _assert_success(resumed)
    assert model.calls == 2
    final_messages = repr(model.seen_messages[-1])
    assert "frontend-value" in final_messages
    assert "approved" in final_messages
