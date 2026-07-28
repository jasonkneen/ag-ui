"""The on_error hook makes swallowed best-effort failures observable."""

import asyncio
import threading
from typing import Any

from ag_ui.core import RunAgentInput

from ag_ui_claude_managed_agents import BackendTool, ManagedAgentsAgent

from .fake_client import FakeClient


def base_input(**overrides: Any) -> RunAgentInput:
    data: dict[str, Any] = {
        "thread_id": "thread_1",
        "run_id": "run_1",
        "state": {},
        "messages": [{"id": "u1", "role": "user", "content": "Hello"}],
        "tools": [],
        "context": [],
        "forwarded_props": {},
    }
    data.update(overrides)
    return RunAgentInput(**data)


IDLE_END_TURN = {
    "type": "session.status_idle",
    "id": "idle_1",
    "stop_reason": {"type": "end_turn"},
}


async def test_reports_an_interrupt_that_could_not_be_posted() -> None:
    """Drive the teardown via the turn timeout rather than task.cancel(): the
    timeout path is deterministic across Python versions, whereas the exact
    point at which an external cancellation unwinds is not."""
    reported: list[tuple[BaseException, dict[str, Any]]] = []
    gate = asyncio.Event()
    # Send 0 is the outbound user message; send 1 is the teardown interrupt.
    fake = FakeClient(
        streams=[[gate]],
        send_failures={1: RuntimeError("interrupt rejected")},
    )
    agent = ManagedAgentsAgent(
        managed_agent_id="agent_1",
        environment_id="env_1",
        client=fake,  # type: ignore[arg-type]
        turn_timeout_s=0.05,
        on_error=lambda error, context: reported.append((error, context)),
    )

    events = [event async for event in agent.run(base_input())]

    # The run still reports the timeout to the client...
    assert events[-1].code == "turn_timeout"
    # ...and the interrupt that could not be delivered is no longer silent.
    assert [c["operation"] for _, c in reported] == ["interrupt"]
    assert str(reported[0][0]) == "interrupt rejected"
    assert reported[0][1]["session_id"] == "sesn_1"
    gate.set()


async def test_reports_an_interrupted_result_the_session_never_received() -> None:
    """The interrupted-result post keeps a session from being left parked. When
    it fails, that failure is the operator's only signal."""
    reported: list[tuple[BaseException, dict[str, Any]]] = []
    release = asyncio.Event()

    async def slow_tool(_input: Any) -> str:
        await release.wait()
        return "never"

    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.custom_tool_use",
                    "id": "ctu_1",
                    "name": "slow_tool",
                    "input": {},
                },
                asyncio.Event(),  # hold the stream open
            ]
        ],
        # Send 0 is the user message; send 1 is the interrupted-result post.
        send_failures={1: RuntimeError("interrupted result rejected")},
    )
    agent = ManagedAgentsAgent(
        managed_agent_id="agent_1",
        environment_id="env_1",
        client=fake,  # type: ignore[arg-type]
        backend_tools=[
            BackendTool(
                name="slow_tool", description="", parameters={}, handler=slow_tool
            )
        ],
        turn_timeout_s=0.05,
        on_error=lambda error, context: reported.append((error, context)),
    )

    events = [event async for event in agent.run(base_input())]
    assert events[-1].code == "turn_timeout"
    for _ in range(20):
        await asyncio.sleep(0)

    failures = [
        str(error)
        for error, context in reported
        if context["operation"] == "post_interrupted_tool_result"
    ]
    assert "interrupted result rejected" in failures
    release.set()


async def test_reports_a_shielded_send_that_fails_after_the_run_unwinds() -> None:
    """Regression: when a second cancellation lands while the interrupted-result
    send is shielded, the shield re-raises and the send finishes in the
    background. Its own failure was consumed there, so a session left parked had
    nothing in the logs."""
    reported: list[tuple[BaseException, dict[str, Any]]] = []
    sending = asyncio.Event()
    release_send = asyncio.Event()
    release_tool = asyncio.Event()

    async def slow_tool(_input: Any) -> str:
        await release_tool.wait()
        return "never"

    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.custom_tool_use",
                    "id": "ctu_1",
                    "name": "slow_tool",
                    "input": {},
                },
                asyncio.Event(),
            ]
        ]
    )
    original_send = fake.beta.sessions.events.send

    async def stalling_send(session_id: str, *, events: list[Any]) -> Any:
        if any(e.get("type") == "user.custom_tool_result" for e in events):
            sending.set()
            await release_send.wait()
            raise RuntimeError("interrupted result rejected")
        return await original_send(session_id, events=events)

    fake.beta.sessions.events.send = stalling_send

    agent = ManagedAgentsAgent(
        managed_agent_id="agent_1",
        environment_id="env_1",
        client=fake,  # type: ignore[arg-type]
        backend_tools=[
            BackendTool(
                name="slow_tool", description="", parameters={}, handler=slow_tool
            )
        ],
        on_error=lambda error, context: reported.append((error, context)),
    )

    generator = agent.run(base_input())
    await generator.__anext__()
    for _ in range(20):
        await asyncio.sleep(0)

    worker = next(iter(agent._tasks))
    # First cancellation: the tool is abandoned and the interrupted-result post
    # starts, shielded.
    worker.cancel()
    await asyncio.wait_for(sending.wait(), 1.0)
    # Second cancellation while shielded: the shield gives up and the send is
    # left running in the background.
    worker.cancel()
    for _ in range(20):
        await asyncio.sleep(0)

    release_send.set()
    release_tool.set()
    await asyncio.gather(worker, return_exceptions=True)
    for _ in range(20):
        await asyncio.sleep(0)
    await generator.aclose()

    failures = [
        str(error)
        for error, context in reported
        if context["operation"] == "post_interrupted_tool_result"
    ]
    assert "interrupted result rejected" in failures, reported


async def test_reports_a_sync_backend_tool_that_fails_after_the_run_walked_away() -> None:
    """Regression: a plain handler runs in a worker thread, which cannot be
    cancelled. The run stops waiting on teardown and the thread runs on, so its
    eventual failure was discarded along with the abandoned wait — a backend tool
    that broke after a timeout left no trace at all."""
    reported: list[dict[str, Any]] = []
    release = threading.Event()

    def blocking_tool(_input: Any) -> str:
        release.wait(5)
        raise RuntimeError("tool blew up late")

    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.custom_tool_use",
                    "id": "ctu_1",
                    "name": "slow_tool",
                    "input": {},
                },
                asyncio.Event(),  # hold the stream open past the timeout
            ]
        ]
    )
    agent = ManagedAgentsAgent(
        managed_agent_id="agent_1",
        environment_id="env_1",
        client=fake,  # type: ignore[arg-type]
        backend_tools=[
            BackendTool(
                name="slow_tool",
                description="",
                parameters={},
                handler=blocking_tool,
            )
        ],
        turn_timeout_s=0.05,
        on_error=lambda _error, context: reported.append(context),
    )

    events = [event async for event in agent.run(base_input())]
    assert events[-1].code == "turn_timeout"

    # The handler only fails once the run is gone.
    release.set()
    for _ in range(200):
        if any(c["operation"] == "abandoned_backend_tool" for c in reported):
            break
        await asyncio.sleep(0.005)

    assert "abandoned_backend_tool" in [c["operation"] for c in reported]


async def test_an_in_run_sync_handler_failure_still_answers_the_tool_call() -> None:
    """The abandoned-handler path must not change what an ordinary failure does:
    the agent still gets the error as the tool result."""
    reported: list[dict[str, Any]] = []

    def failing_tool(_input: Any) -> str:
        raise RuntimeError("bad input")

    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.custom_tool_use",
                    "id": "ctu_1",
                    "name": "boom",
                    "input": {},
                },
                IDLE_END_TURN,
            ]
        ]
    )
    agent = ManagedAgentsAgent(
        managed_agent_id="agent_1",
        environment_id="env_1",
        client=fake,  # type: ignore[arg-type]
        backend_tools=[
            BackendTool(
                name="boom", description="", parameters={}, handler=failing_tool
            )
        ],
        on_error=lambda _error, context: reported.append(context),
    )

    events = [event async for event in agent.run(base_input())]

    assert events[-1].type.value == "RUN_FINISHED"
    posted = [event for send in fake.sent for event in send["events"]]
    assert {
        "type": "user.custom_tool_result",
        "custom_tool_use_id": "ctu_1",
        "content": [{"type": "text", "text": "bad input"}],
        "is_error": True,
    } in posted
    # An in-run failure is reported to the agent, not routed to the hook.
    assert "abandoned_backend_tool" not in [c["operation"] for c in reported]


async def test_a_broken_hook_does_not_break_the_run() -> None:
    def boom(error: BaseException, context: dict[str, Any]) -> None:
        raise RuntimeError("hook is broken")

    fake = FakeClient(streams=[[IDLE_END_TURN]])
    agent = ManagedAgentsAgent(
        managed_agent_id="agent_1",
        environment_id="env_1",
        client=fake,  # type: ignore[arg-type]
        on_error=boom,
    )

    events = [event async for event in agent.run(base_input())]
    assert events[-1].type.value == "RUN_FINISHED"
