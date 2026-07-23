"""Ports of the TypeScript `agent.test.ts` assertions, plus lifecycle guards."""

import asyncio
from typing import Any

from ag_ui.core import RunAgentInput, RunErrorEvent

from ag_ui_claude_managed_agents import (
    BackendTool,
    InMemorySessionStore,
    ManagedAgentsAgent,
    SessionRecord,
)

from .fake_client import FakeClient

IDLE_END_TURN = {
    "type": "session.status_idle",
    "id": "idle_1",
    "stop_reason": {"type": "end_turn"},
}


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


async def collect(agent: ManagedAgentsAgent, run_input: RunAgentInput) -> list[Any]:
    return [event async for event in agent.run(run_input)]


def types(events: list[Any]) -> list[str]:
    return [event.type.value for event in events]


def new_agent(
    fake: FakeClient, store: InMemorySessionStore | None = None, **kwargs: Any
) -> ManagedAgentsAgent:
    return ManagedAgentsAgent(
        agent_id="agent_1",
        environment_id="env_1",
        client=fake,  # type: ignore[arg-type]
        session_store=store,
        **kwargs,
    )


async def test_creates_session_for_new_thread_and_streams_reply():
    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.message",
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "Hi!"}],
                },
                IDLE_END_TURN,
            ]
        ]
    )
    events = await collect(new_agent(fake), base_input())

    assert fake.create_calls == [
        {
            "agent": {"type": "agent", "id": "agent_1"},
            "environment_id": "env_1",
            "title": "AG-UI thread thread_1",
        }
    ]
    assert types(events) == [
        "RUN_STARTED",
        "STATE_SNAPSHOT",
        "CUSTOM",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert events[2].name == "managed_agents.session"
    assert events[2].value == {"sessionId": "sesn_1", "threadId": "thread_1"}
    assert fake.sent[0]["events"] == [
        {"type": "user.message", "content": [{"type": "text", "text": "Hello"}]}
    ]


async def test_pins_agent_version_and_custom_title_when_configured():
    fake = FakeClient(streams=[[IDLE_END_TURN]])
    agent = new_agent(
        fake, agent_version=3, session_title=lambda thread_id: f"Chat {thread_id}"
    )
    await collect(agent, base_input())
    assert fake.create_calls[0]["agent"] == {
        "type": "agent",
        "id": "agent_1",
        "version": 3,
    }
    assert fake.create_calls[0]["title"] == "Chat thread_1"


async def test_reuses_session_on_next_run_and_sends_only_new_message():
    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.message",
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "one"}],
                },
                IDLE_END_TURN,
            ],
            [
                {
                    "type": "agent.message",
                    "id": "msg_2",
                    "content": [{"type": "text", "text": "two"}],
                },
                IDLE_END_TURN,
            ],
        ]
    )
    store = InMemorySessionStore()
    await collect(new_agent(fake, store), base_input())
    await collect(
        new_agent(fake, store),
        base_input(
            run_id="run_2",
            messages=[
                {"id": "u1", "role": "user", "content": "Hello"},
                {"id": "a1", "role": "assistant", "content": "one"},
                {"id": "u2", "role": "user", "content": "Follow-up"},
            ],
        ),
    )

    assert len(fake.create_calls) == 1
    assert fake.sent[1]["events"] == [
        {"type": "user.message", "content": [{"type": "text", "text": "Follow-up"}]}
    ]


async def test_registers_frontend_tools_as_custom_tools_when_creating_session():
    fake = FakeClient(
        streams=[[IDLE_END_TURN]],
        agent_tools=[
            {"type": "agent_toolset_20260401", "configs": [], "default_config": {}}
        ],
    )
    await collect(
        new_agent(fake),
        base_input(
            tools=[
                {
                    "name": "show_chart",
                    "description": "Render a chart",
                    "parameters": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                    },
                }
            ]
        ),
    )

    assert fake.create_calls[0]["agent"] == {
        "type": "agent_with_overrides",
        "id": "agent_1",
        "tools": [
            {"type": "agent_toolset_20260401", "configs": [], "default_config": {}},
            {
                "type": "custom",
                "name": "show_chart",
                "description": "Render a chart",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": [],
                },
            },
        ],
    }


async def test_registers_backend_tools_and_normalizes_names_with_frontend_winning():
    fake = FakeClient(streams=[[IDLE_END_TURN]], agent_tools=[])
    backend = BackendTool(
        name="lookup docs",
        description="Backend lookup",
        parameters={},
        handler=lambda _i: "x",
    )
    agent = new_agent(fake, backend_tools=[backend])
    await collect(
        agent,
        base_input(
            tools=[
                {
                    "name": "lookup docs",
                    "description": "Frontend lookup",
                    "parameters": {"type": "object"},
                }
            ]
        ),
    )

    tools = fake.create_calls[0]["agent"]["tools"]
    assert tools == [
        {
            "type": "custom",
            "name": "lookup_docs",
            "description": "Frontend lookup",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]


async def test_round_trips_frontend_tool_park_then_resume_with_client_result():
    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.custom_tool_use",
                    "id": "ctu_1",
                    "name": "show_chart",
                    "input": {"title": "Sales"},
                },
                {
                    "type": "session.status_idle",
                    "id": "idle_1",
                    "stop_reason": {"type": "requires_action", "event_ids": ["ctu_1"]},
                },
            ],
            [
                {
                    "type": "agent.message",
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "Chart shown."}],
                },
                IDLE_END_TURN,
            ],
        ]
    )
    store = InMemorySessionStore()
    tools = [
        {
            "name": "show_chart",
            "description": "Render a chart",
            "parameters": {"type": "object"},
        }
    ]

    first = await collect(new_agent(fake, store), base_input(tools=tools))
    assert types(first) == [
        "RUN_STARTED",
        "STATE_SNAPSHOT",
        "CUSTOM",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "RUN_FINISHED",
    ]
    assert store.get("thread_1").pending_client_tool_use_ids == ["ctu_1"]

    second = await collect(
        new_agent(fake, store),
        base_input(
            run_id="run_2",
            tools=tools,
            messages=[
                {"id": "u1", "role": "user", "content": "Hello"},
                {
                    "id": "t1",
                    "role": "tool",
                    "toolCallId": "ctu_1",
                    "content": "rendered",
                },
            ],
        ),
    )

    assert fake.sent[1]["events"] == [
        {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": "ctu_1",
            "content": [{"type": "text", "text": "rendered"}],
            "is_error": False,
        }
    ]
    assert "TEXT_MESSAGE_CONTENT" in types(second)
    assert types(second)[-1] == "RUN_FINISHED"
    assert store.get("thread_1").pending_client_tool_use_ids == []


async def test_forwards_tool_message_error_flag():
    fake = FakeClient(streams=[[IDLE_END_TURN]])
    store = InMemorySessionStore()
    store.set(
        "thread_1",
        SessionRecord(
            session_id="sesn_1",
            tool_names=[],
            pending_client_tool_use_ids=["ctu_1"],
            last_user_message_id="u1",
        ),
    )
    await collect(
        new_agent(fake, store),
        base_input(
            messages=[
                {"id": "u1", "role": "user", "content": "Hello"},
                {
                    "id": "t1",
                    "role": "tool",
                    "toolCallId": "ctu_1",
                    "content": "boom",
                    "error": "failed",
                },
            ]
        ),
    )
    assert fake.sent[0]["events"] == [
        {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": "ctu_1",
            "content": [{"type": "text", "text": "boom"}],
            "is_error": True,
        }
    ]


async def test_stays_parked_when_some_tool_calls_remain_unanswered():
    fake = FakeClient(streams=[])
    store = InMemorySessionStore()
    store.set(
        "thread_1",
        SessionRecord(
            session_id="sesn_1",
            tool_names=[],
            pending_client_tool_use_ids=["ctu_1", "ctu_2"],
            last_user_message_id="u1",
        ),
    )
    events = await collect(
        new_agent(fake, store),
        base_input(
            messages=[
                {"id": "u1", "role": "user", "content": "Hello"},
                {"id": "t1", "role": "tool", "toolCallId": "ctu_1", "content": "done"},
            ]
        ),
    )

    # The answered call is posted, the run finishes without streaming, and
    # the unanswered call stays pending.
    assert fake.sent[0]["events"] == [
        {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": "ctu_1",
            "content": [{"type": "text", "text": "done"}],
            "is_error": False,
        }
    ]
    assert fake.stream_calls == []
    assert types(events)[-1] == "RUN_FINISHED"
    assert store.get("thread_1").pending_client_tool_use_ids == ["ctu_2"]


async def test_default_session_store_persists_across_runs():
    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.custom_tool_use",
                    "id": "ctu_1",
                    "name": "show_chart",
                    "input": {},
                },
                {
                    "type": "session.status_idle",
                    "id": "idle_1",
                    "stop_reason": {"type": "requires_action", "event_ids": ["ctu_1"]},
                },
            ],
            [
                {
                    "type": "agent.message",
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "Done."}],
                },
                IDLE_END_TURN,
            ],
        ]
    )
    # No session_store passed: the default in-memory store must survive between runs.
    agent = ManagedAgentsAgent(agent_id="agent_1", environment_id="env_1", client=fake)  # type: ignore[arg-type]
    tools = [
        {
            "name": "show_chart",
            "description": "Render a chart",
            "parameters": {"type": "object"},
        }
    ]

    await collect(agent, base_input(tools=tools))
    await collect(
        agent,
        base_input(
            run_id="run_2",
            tools=tools,
            messages=[
                {"id": "u1", "role": "user", "content": "Hello"},
                {
                    "id": "t1",
                    "role": "tool",
                    "toolCallId": "ctu_1",
                    "content": "rendered",
                },
            ],
        ),
    )

    assert len(fake.create_calls) == 1
    assert fake.sent[1]["events"] == [
        {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": "ctu_1",
            "content": [{"type": "text", "text": "rendered"}],
            "is_error": False,
        }
    ]


async def test_forwards_every_undelivered_user_message_in_order():
    fake = FakeClient(streams=[[IDLE_END_TURN]])
    store = InMemorySessionStore()
    store.set(
        "thread_1",
        SessionRecord(
            session_id="sesn_1",
            tool_names=[],
            pending_client_tool_use_ids=[],
            last_user_message_id="u1",
        ),
    )

    await collect(
        new_agent(fake, store),
        base_input(
            messages=[
                {"id": "u1", "role": "user", "content": "delivered"},
                {"id": "u2", "role": "user", "content": "second"},
                {"id": "u3", "role": "user", "content": "third"},
            ]
        ),
    )

    assert fake.sent[0]["events"] == [
        {"type": "user.message", "content": [{"type": "text", "text": "second"}]},
        {"type": "user.message", "content": [{"type": "text", "text": "third"}]},
    ]
    assert store.get("thread_1").last_user_message_id == "u3"


async def test_extracts_text_from_multimodal_user_content():
    fake = FakeClient(streams=[[IDLE_END_TURN]])
    await collect(
        new_agent(fake),
        base_input(
            messages=[
                {
                    "id": "u1",
                    "role": "user",
                    "content": [{"type": "text", "text": "Look here"}],
                }
            ]
        ),
    )
    assert fake.sent[0]["events"] == [
        {"type": "user.message", "content": [{"type": "text", "text": "Look here"}]}
    ]


async def test_abandons_parked_tool_calls_when_user_sends_new_message_instead():
    fake = FakeClient(
        streams=[
            [
                {
                    "type": "agent.message",
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "Moving on."}],
                },
                IDLE_END_TURN,
            ]
        ]
    )
    store = InMemorySessionStore()
    store.set(
        "thread_1",
        SessionRecord(
            session_id="sesn_1",
            tool_names=[],
            pending_client_tool_use_ids=["ctu_1"],
            last_user_message_id="u1",
        ),
    )

    events = await collect(
        new_agent(fake, store),
        base_input(
            messages=[
                {"id": "u1", "role": "user", "content": "old"},
                {"id": "u2", "role": "user", "content": "never mind"},
            ]
        ),
    )

    # Tool results go first (resuming the parked session), then the message,
    # as two separate sends: the API rejects a user.message in the same batch
    # as the results while the session is still parked.
    assert fake.sent[0]["events"] == [
        {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": "ctu_1",
            "content": [{"type": "text", "text": "The user did not provide a result for this tool call."}],
            "is_error": True,
        }
    ]
    assert fake.sent[1]["events"] == [
        {"type": "user.message", "content": [{"type": "text", "text": "never mind"}]}
    ]
    assert types(events)[-1] == "RUN_FINISHED"
    record = store.get("thread_1")
    assert record.pending_client_tool_use_ids == []
    assert record.last_user_message_id == "u2"


async def test_errors_when_run_has_nothing_new_to_send():
    fake = FakeClient(streams=[[IDLE_END_TURN]])
    store = InMemorySessionStore()
    store.set(
        "thread_1",
        SessionRecord(
            session_id="sesn_1",
            tool_names=[],
            pending_client_tool_use_ids=[],
            last_user_message_id="u1",
        ),
    )

    events = await collect(new_agent(fake, store), base_input())
    assert isinstance(events[-1], RunErrorEvent)
    assert fake.sent == []


async def test_errors_before_creating_session_when_input_has_nothing_sendable():
    fake = FakeClient(streams=[[IDLE_END_TURN]])
    events = await collect(
        new_agent(fake),
        base_input(messages=[{"id": "a1", "role": "assistant", "content": "hi"}]),
    )
    assert isinstance(events[-1], RunErrorEvent)
    assert fake.create_calls == []


async def test_updates_session_tools_when_frontend_adds_new_one():
    fake = FakeClient(streams=[[IDLE_END_TURN], [IDLE_END_TURN]])
    store = InMemorySessionStore()
    await collect(new_agent(fake, store), base_input())
    assert fake.update_calls == []

    await collect(
        new_agent(fake, store),
        base_input(
            run_id="run_2",
            messages=[
                {"id": "u1", "role": "user", "content": "Hello"},
                {"id": "u2", "role": "user", "content": "Show me a chart"},
            ],
            tools=[
                {
                    "name": "show_chart",
                    "description": "Render a chart",
                    "parameters": {"type": "object"},
                }
            ],
        ),
    )

    assert fake.update_calls == [
        (
            "sesn_1",
            {
                "agent": {
                    "tools": [
                        {
                            "type": "agent_toolset_20260401",
                            "configs": [],
                            "default_config": {},
                        },
                        {
                            "type": "custom",
                            "name": "show_chart",
                            "description": "Render a chart",
                            "input_schema": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        },
                    ]
                }
            },
        )
    ]


async def test_deletes_thread_record_when_session_ends():
    fake = FakeClient(
        streams=[
            [{"type": "session.status_terminated", "id": "term_1"}],
            [IDLE_END_TURN],
        ]
    )
    store = InMemorySessionStore()
    events = await collect(new_agent(fake, store), base_input())
    assert isinstance(events[-1], RunErrorEvent)
    assert events[-1].code == "session_ended"
    assert store.get("thread_1") is None

    # The next run creates a fresh session.
    await collect(new_agent(fake, store), base_input(run_id="run_2"))
    assert len(fake.create_calls) == 2


async def test_rejects_second_concurrent_run_on_same_thread():
    gate = asyncio.Event()
    fake = FakeClient(streams=[[gate, IDLE_END_TURN]])
    store = InMemorySessionStore()
    first_events: list[Any] = []

    async def first() -> None:
        async for event in new_agent(fake, store).run(base_input()):
            first_events.append(event)

    task = asyncio.create_task(first())
    # Let the first run get into the busy section.
    for _ in range(10):
        await asyncio.sleep(0)

    second = await collect(new_agent(fake, store), base_input(run_id="run_2"))
    assert isinstance(second[-1], RunErrorEvent)
    assert second[-1].message == "A run is already in progress on this thread."

    gate.set()
    await task
    assert types(first_events)[-1] == "RUN_FINISHED"


async def test_interrupts_session_when_client_disconnects():
    gate = asyncio.Event()
    fake = FakeClient(streams=[[gate]])
    agent = new_agent(fake)

    generator = agent.run(base_input())
    first = await generator.__anext__()
    assert first.type.value == "RUN_STARTED"
    # Let the turn open its stream and post the user message.
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(fake.sent) == 1

    await generator.aclose()
    for _ in range(10):
        await asyncio.sleep(0)

    assert {"type": "user.interrupt"} in [
        event for send in fake.sent for event in send["events"]
    ]


async def test_interrupts_session_when_turn_times_out():
    gate = asyncio.Event()
    fake = FakeClient(streams=[[gate]])
    agent = new_agent(fake, turn_timeout_s=0.05)

    events = await collect(agent, base_input())

    assert isinstance(events[-1], RunErrorEvent)
    assert events[-1].message == "The turn exceeded the 0s limit and was interrupted."
    assert {"type": "user.interrupt"} in [
        event for send in fake.sent for event in send["events"]
    ]
    gate.set()


async def test_stops_streaming_deltas_when_disabled():
    fake = FakeClient(streams=[[IDLE_END_TURN]])
    await collect(new_agent(fake, stream_deltas=False), base_input())
    assert fake.stream_calls == [("sesn_1", {})]


async def test_scope_partitions_sessions_across_identical_thread_ids():
    fake = FakeClient(
        streams=[
            [{"type": "agent.message", "id": "msg_1", "content": [{"type": "text", "text": "a"}]}, IDLE_END_TURN],
            [{"type": "agent.message", "id": "msg_2", "content": [{"type": "text", "text": "b"}]}, IDLE_END_TURN],
        ]
    )
    store = InMemorySessionStore()

    await collect(new_agent(fake, store, scope="alice"), base_input())
    await collect(new_agent(fake, store, scope="bob"), base_input())

    # Same thread id, different scopes: two sessions and two store entries.
    assert len(fake.create_calls) == 2
    assert store.get("alice:thread_1") is not None
    assert store.get("bob:thread_1") is not None
    assert store.get("thread_1") is None


async def test_tool_result_on_unknown_thread_does_not_create_session():
    fake = FakeClient(streams=[[IDLE_END_TURN]])
    events = await collect(
        new_agent(fake),
        base_input(messages=[{"id": "t1", "role": "tool", "tool_call_id": "ctu_ghost", "content": "late"}]),
    )
    assert fake.create_calls == []
    assert isinstance(events[-1], RunErrorEvent)
