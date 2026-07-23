"""Ports of the TypeScript `turn.test.ts` assertions."""

from typing import Any

from ag_ui.core import (
    ReasoningEndEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunErrorEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from ag_ui_claude_managed_agents import BackendTool, TurnOutcome, run_turn

from .fake_client import FakeClient

IDLE_END_TURN = {
    "type": "session.status_idle",
    "id": "idle_1",
    "stop_reason": {"type": "end_turn"},
}


async def collect(
    stream_events: list[Any],
    client_options: dict[str, Any] | None = None,
    **overrides: Any,
):
    fake = FakeClient(streams=[stream_events], **(client_options or {}))
    emitted: list[Any] = []
    options: dict[str, Any] = {
        "client": fake,
        "session_id": "sesn_1",
        "outbound": [
            {"type": "user.message", "content": [{"type": "text", "text": "hi"}]}
        ],
        "client_tools": {},
        "backend_tools": {},
        "tool_confirmation": None,
        "stream_deltas": True,
        "emit": emitted.append,
    }
    options.update(overrides)
    outcome = await run_turn(**options)
    return emitted, outcome, fake


def types(events: list[Any]) -> list[str]:
    return [event.type.value for event in events]


async def test_streams_text_preview_tops_up_from_buffered_message_and_finishes():
    emitted, outcome, fake = await collect(
        [
            {"type": "session.status_running", "id": "run_1"},
            {"type": "event_start", "event": {"type": "agent.message", "id": "msg_1"}},
            {
                "type": "event_delta",
                "event_id": "msg_1",
                "delta": {
                    "type": "content_delta",
                    "index": 0,
                    "content": {"type": "text", "text": "Hel"},
                },
            },
            {
                "type": "event_delta",
                "event_id": "msg_1",
                "delta": {
                    "type": "content_delta",
                    "index": 0,
                    "content": {"type": "text", "text": "lo"},
                },
            },
            {
                "type": "agent.message",
                "id": "msg_1",
                "content": [{"type": "text", "text": "Hello there"}],
            },
            IDLE_END_TURN,
        ]
    )
    assert outcome == TurnOutcome(status="finished")
    assert fake.sent[0]["events"] == [
        {"type": "user.message", "content": [{"type": "text", "text": "hi"}]}
    ]
    assert emitted == [
        TextMessageStartEvent(message_id="msg_1", role="assistant"),
        TextMessageContentEvent(message_id="msg_1", delta="Hel"),
        TextMessageContentEvent(message_id="msg_1", delta="lo"),
        TextMessageContentEvent(message_id="msg_1", delta=" there"),
        TextMessageEndEvent(message_id="msg_1"),
    ]


async def test_requests_previews_when_streaming_deltas():
    _, _, fake = await collect([IDLE_END_TURN])
    assert fake.stream_calls == [
        ("sesn_1", {"event_deltas": ["agent.message", "agent.thinking"]})
    ]

    _, _, fake = await collect([IDLE_END_TURN], stream_deltas=False)
    assert fake.stream_calls == [("sesn_1", {})]


async def test_emits_whole_message_when_there_was_no_preview():
    emitted, _, _ = await collect(
        [
            {
                "type": "agent.message",
                "id": "msg_1",
                "content": [{"type": "text", "text": "All at once"}],
            },
            IDLE_END_TURN,
        ]
    )
    assert emitted == [
        TextMessageStartEvent(message_id="msg_1", role="assistant"),
        TextMessageContentEvent(message_id="msg_1", delta="All at once"),
        TextMessageEndEvent(message_id="msg_1"),
    ]


async def test_re_emits_corrected_message_when_preview_diverges():
    emitted, _, _ = await collect(
        [
            {"type": "event_start", "event": {"type": "agent.message", "id": "msg_1"}},
            {
                "type": "event_delta",
                "event_id": "msg_1",
                "delta": {
                    "type": "content_delta",
                    "index": 0,
                    "content": {"type": "text", "text": "Draft"},
                },
            },
            {
                "type": "agent.message",
                "id": "msg_1",
                "content": [{"type": "text", "text": "Final"}],
            },
            IDLE_END_TURN,
        ]
    )
    assert emitted == [
        TextMessageStartEvent(message_id="msg_1", role="assistant"),
        TextMessageContentEvent(message_id="msg_1", delta="Draft"),
        TextMessageEndEvent(message_id="msg_1"),
        TextMessageStartEvent(message_id="corrected_msg_1", role="assistant"),
        TextMessageContentEvent(message_id="corrected_msg_1", delta="Final"),
        TextMessageEndEvent(message_id="corrected_msg_1"),
    ]


async def test_maps_thinking_stretch_to_reasoning_start_and_end():
    emitted, _, _ = await collect(
        [
            {
                "type": "event_start",
                "event": {"type": "agent.thinking", "id": "think_1"},
            },
            {"type": "agent.thinking", "id": "think_1"},
            IDLE_END_TURN,
        ]
    )
    assert emitted == [
        ReasoningStartEvent(message_id="think_1"),
        ReasoningMessageStartEvent(message_id="think_1", role="reasoning"),
        ReasoningMessageEndEvent(message_id="think_1"),
        ReasoningEndEvent(message_id="think_1"),
    ]


async def test_maps_unpreviewed_thinking_to_reasoning_pair():
    emitted, _, _ = await collect(
        [{"type": "agent.thinking", "id": "think_1"}, IDLE_END_TURN]
    )
    assert types(emitted) == ["REASONING_START", "REASONING_END"]


async def test_streams_builtin_tool_calls_and_their_results():
    emitted, _, _ = await collect(
        [
            {
                "type": "agent.tool_use",
                "id": "tu_1",
                "name": "bash",
                "input": {"command": "ls"},
            },
            {
                "type": "agent.tool_result",
                "id": "tr_1",
                "tool_use_id": "tu_1",
                "content": [{"type": "text", "text": "file.txt"}],
            },
            IDLE_END_TURN,
        ]
    )
    assert emitted == [
        ToolCallStartEvent(tool_call_id="tu_1", tool_call_name="bash"),
        ToolCallArgsEvent(tool_call_id="tu_1", delta='{"command":"ls"}'),
        ToolCallEndEvent(tool_call_id="tu_1"),
        ToolCallResultEvent(
            message_id="result_tu_1",
            tool_call_id="tu_1",
            content="file.txt",
            role="tool",
        ),
    ]


async def test_maps_mcp_tool_calls_with_server_qualified_name():
    emitted, _, _ = await collect(
        [
            {
                "type": "agent.mcp_tool_use",
                "id": "mcp_1",
                "name": "search",
                "mcp_server_name": "docs",
                "input": {"q": "x"},
            },
            {
                "type": "agent.mcp_tool_result",
                "id": "mr_1",
                "mcp_tool_use_id": "mcp_1",
                "content": [{"type": "text", "text": "found"}],
            },
            IDLE_END_TURN,
        ]
    )
    assert emitted[0] == ToolCallStartEvent(
        tool_call_id="mcp_1", tool_call_name="docs: search"
    )
    assert emitted[3] == ToolCallResultEvent(
        message_id="result_mcp_1", tool_call_id="mcp_1", content="found", role="tool"
    )


async def test_flattens_mixed_tool_result_content():
    emitted, _, _ = await collect(
        [
            {
                "type": "agent.tool_result",
                "id": "tr_1",
                "tool_use_id": "tu_1",
                "content": [
                    {
                        "type": "text",
                        "text": "Caf&eacute; &amp; &#39;more&#39; &lt;b&gt;",
                    },
                    {
                        "type": "search_result",
                        "title": "Docs &amp; guides",
                        "source": "https://example.com",
                        "content": [{"type": "text", "text": "body text"}],
                    },
                    {"type": "image", "source": {}},
                ],
            },
            IDLE_END_TURN,
        ]
    )
    assert emitted == [
        ToolCallResultEvent(
            message_id="result_tu_1",
            tool_call_id="tu_1",
            content="Caf&eacute; & 'more' <b>\n[search result] Docs & guides — https://example.com\nbody text\n[image]",
            role="tool",
        )
    ]


async def test_runs_backend_tool_and_posts_result_back_into_session():
    async def handler(_input: Any) -> str:
        return "noon"

    backend = BackendTool(
        name="get_time", description="", parameters={}, handler=handler
    )
    emitted, _, fake = await collect(
        [
            {
                "type": "agent.custom_tool_use",
                "id": "ctu_1",
                "name": "get_time",
                "input": {},
            },
            {
                "type": "session.status_idle",
                "id": "idle_1",
                "stop_reason": {"type": "requires_action", "event_ids": ["ctu_1"]},
            },
            IDLE_END_TURN,
        ],
        backend_tools={"get_time": backend},
    )
    assert (
        ToolCallResultEvent(
            message_id="result_ctu_1", tool_call_id="ctu_1", content="noon", role="tool"
        )
        in emitted
    )
    assert fake.sent[1]["events"] == [
        {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": "ctu_1",
            "content": [{"type": "text", "text": "noon"}],
            "is_error": False,
        }
    ]


async def test_reports_backend_tool_exception_as_error_result():
    def handler(_input: Any) -> str:
        raise ValueError("clock offline")

    backend = BackendTool(
        name="get_time", description="", parameters={}, handler=handler
    )
    _, _, fake = await collect(
        [
            {
                "type": "agent.custom_tool_use",
                "id": "ctu_1",
                "name": "get_time",
                "input": {},
            },
            IDLE_END_TURN,
        ],
        backend_tools={"get_time": backend},
    )
    assert fake.sent[1]["events"] == [
        {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": "ctu_1",
            "content": [{"type": "text", "text": "clock offline"}],
            "is_error": True,
        }
    ]


async def test_posts_error_result_for_tool_nothing_can_execute():
    _, _, fake = await collect(
        [
            {
                "type": "agent.custom_tool_use",
                "id": "ctu_1",
                "name": "mystery",
                "input": {},
            },
            IDLE_END_TURN,
        ]
    )
    result = fake.sent[1]["events"][0]
    assert result["type"] == "user.custom_tool_result"
    assert result["custom_tool_use_id"] == "ctu_1"
    assert result["is_error"] is True
    assert result["content"] == [
        {"type": "text", "text": 'No handler is registered for tool "mystery".'}
    ]


async def test_parks_turn_when_frontend_must_execute_tool():
    emitted, outcome, fake = await collect(
        [
            {
                "type": "agent.custom_tool_use",
                "id": "ctu_1",
                "name": "confirm_purchase",
                "input": {"amount": 5},
            },
            {
                "type": "session.status_idle",
                "id": "idle_1",
                "stop_reason": {"type": "requires_action", "event_ids": ["ctu_1"]},
            },
        ],
        client_tools={"confirm_purchase": "confirm_purchase"},
    )
    assert outcome == TurnOutcome(status="parked", client_tool_use_ids=["ctu_1"])
    assert len(fake.sent) == 1  # only the user message; no result posted
    assert types(emitted) == ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"]


async def test_reports_frontends_original_name_for_normalized_tool():
    emitted, outcome, _ = await collect(
        [
            {
                "type": "agent.custom_tool_use",
                "id": "ctu_1",
                "name": "search_web",
                "input": {},
            },
            {
                "type": "session.status_idle",
                "id": "idle_1",
                "stop_reason": {"type": "requires_action", "event_ids": ["ctu_1"]},
            },
        ],
        client_tools={"search_web": "search web"},
    )
    assert outcome == TurnOutcome(status="parked", client_tool_use_ids=["ctu_1"])
    assert emitted[0] == ToolCallStartEvent(
        tool_call_id="ctu_1", tool_call_name="search web"
    )


async def test_answers_confirmation_gated_tool_when_policy_is_configured():
    _, outcome, fake = await collect(
        [
            {
                "type": "agent.tool_use",
                "id": "tu_1",
                "name": "bash",
                "input": {},
                "evaluated_permission": "ask",
            },
            {
                "type": "session.status_idle",
                "id": "idle_1",
                "stop_reason": {"type": "requires_action", "event_ids": ["tu_1"]},
            },
            {
                "type": "agent.tool_result",
                "id": "tr_1",
                "tool_use_id": "tu_1",
                "content": [],
            },
            IDLE_END_TURN,
        ],
        tool_confirmation="allow",
    )
    assert outcome == TurnOutcome(status="finished")
    assert fake.sent[1]["events"] == [
        {"type": "user.tool_confirmation", "tool_use_id": "tu_1", "result": "allow"}
    ]


async def test_fails_run_on_confirmation_gated_tool_with_no_policy():
    emitted, outcome, fake = await collect(
        [
            {
                "type": "agent.tool_use",
                "id": "tu_1",
                "name": "bash",
                "input": {},
                "evaluated_permission": "ask",
            },
            {
                "type": "session.status_idle",
                "id": "idle_1",
                "stop_reason": {"type": "requires_action", "event_ids": ["tu_1"]},
            },
        ]
    )
    assert outcome == TurnOutcome(status="errored")
    assert isinstance(emitted[-1], RunErrorEvent)
    assert emitted[-1].code == "tool_confirmation_required"
    assert fake.sent[1]["events"] == [{"type": "user.interrupt"}]


async def test_interrupts_and_errors_on_an_unknown_blocking_action():
    emitted, outcome, fake = await collect(
        [
            {
                "type": "session.status_idle",
                "id": "idle_1",
                "stop_reason": {"type": "requires_action", "event_ids": ["unknown_1"]},
            }
        ]
    )
    assert outcome == TurnOutcome(status="errored")
    assert emitted[-1].code == "unsupported_action"
    assert fake.sent[1]["events"] == [{"type": "user.interrupt"}]


async def test_surfaces_terminal_session_error_with_its_type_as_code():
    emitted, outcome, _ = await collect(
        [
            {
                "type": "session.error",
                "id": "err_1",
                "error": {
                    "type": "billing_error",
                    "message": "Out of credits",
                    "retry_status": {"type": "terminal"},
                },
            }
        ]
    )
    assert outcome == TurnOutcome(status="errored")
    assert emitted == [RunErrorEvent(message="Out of credits", code="billing_error")]


async def test_ignores_retrying_session_error_and_completes():
    _, outcome, _ = await collect(
        [
            {
                "type": "session.error",
                "id": "err_1",
                "error": {
                    "type": "model_overloaded_error",
                    "message": "busy",
                    "retry_status": {"type": "retrying"},
                },
            },
            {
                "type": "agent.message",
                "id": "msg_1",
                "content": [{"type": "text", "text": "ok"}],
            },
            IDLE_END_TURN,
        ]
    )
    assert outcome == TurnOutcome(status="finished")


async def test_treats_retries_exhausted_as_error_not_clean_finish():
    emitted, outcome, _ = await collect(
        [
            {
                "type": "session.status_idle",
                "id": "idle_1",
                "stop_reason": {"type": "retries_exhausted"},
            }
        ]
    )
    assert outcome == TurnOutcome(status="errored")
    assert emitted[-1].code == "retries_exhausted"


async def test_reports_terminated_session_as_ended():
    _, outcome, _ = await collect(
        [{"type": "session.status_terminated", "id": "term_1"}]
    )
    assert outcome == TurnOutcome(status="errored", session_ended=True)


async def test_reports_deleted_session_as_ended():
    emitted, outcome, _ = await collect([{"type": "session.deleted", "id": "del_1"}])
    assert outcome == TurnOutcome(status="errored", session_ended=True)
    assert emitted[-1].code == "session_ended"


async def test_closes_dangling_preview_when_model_request_ends_without_message():
    emitted, _, _ = await collect(
        [
            {"type": "event_start", "event": {"type": "agent.message", "id": "msg_1"}},
            {
                "type": "event_delta",
                "event_id": "msg_1",
                "delta": {
                    "type": "content_delta",
                    "index": 0,
                    "content": {"type": "text", "text": "partia"},
                },
            },
            {
                "type": "span.model_request_end",
                "id": "span_1",
                "model_request_start_id": "s_1",
                "is_error": True,
                "model_usage": {},
            },
            IDLE_END_TURN,
        ]
    )
    assert types(emitted) == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
    ]


async def test_errors_when_stream_ends_before_turn_completes():
    emitted, outcome, _ = await collect(
        [{"type": "event_start", "event": {"type": "agent.message", "id": "msg_1"}}]
    )
    assert outcome == TurnOutcome(status="errored")
    # The open message is closed before the error.
    assert types(emitted) == ["TEXT_MESSAGE_START", "TEXT_MESSAGE_END", "RUN_ERROR"]
    assert emitted[-1].code == "stream_ended"
