"""The on_error hook makes swallowed best-effort failures observable."""

import asyncio
from typing import Any

from ag_ui.core import RunAgentInput

from ag_ui_claude_managed_agents import ManagedAgentsAgent

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
    reported: list[tuple[BaseException, dict[str, Any]]] = []
    gate = asyncio.Event()
    # The stream stays open, so cancelling the run triggers the teardown
    # interrupt — which is the send scripted to fail here.
    fake = FakeClient(
        streams=[[gate]],
        send_failures={1: RuntimeError("interrupt rejected")},
    )
    agent = ManagedAgentsAgent(
        managed_agent_id="agent_1",
        environment_id="env_1",
        client=fake,  # type: ignore[arg-type]
        on_error=lambda error, context: reported.append((error, context)),
    )

    async def drive() -> None:
        async for _ in agent.run(base_input()):
            pass

    task = asyncio.create_task(drive())
    await asyncio.sleep(0.05)
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await task

    assert [c["operation"] for _, c in reported] == ["interrupt"]
    assert str(reported[0][0]) == "interrupt rejected"
    assert reported[0][1]["session_id"] == "sesn_1"


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
