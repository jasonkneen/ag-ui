"""Drive one turn of a managed session and translate its events into AG-UI events."""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ag_ui.core import (
    BaseEvent,
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

from ._util import get
from .text import describe_tool_result, text_of
from .types import BackendTool, TurnOutcome

TOOL_RESULT_MAX_CHARS = 4000
"""Tool results can be large; the UI only needs a readable prefix."""

Emit = Callable[[BaseEvent], None]


async def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def run_turn(
    *,
    client: Any,
    session_id: str,
    outbound: list[dict[str, Any]],
    client_tools: Mapping[str, str],
    backend_tools: Mapping[str, BackendTool],
    tool_confirmation: str | None,
    stream_deltas: bool,
    emit: Emit,
    on_sent: Callable[[], Awaitable[None] | None] | None = None,
) -> TurnOutcome:
    """Open the event stream, post the outbound events, and translate the
    session's events into AG-UI events until the session goes idle.

    `client_tools` maps normalized frontend tool names to their original AG-UI
    names; calls to these park the session. `backend_tools` maps normalized
    names to tools executed on this server.

    Invariant: no TEXT_MESSAGE or REASONING block is left open when this returns
    or raises. Every exit path closes them.
    """
    # Open the stream before sending so no early events are missed.
    # "agent.thinking" opts into the live thinking indicator (event_start);
    # thinking carries no text deltas today.
    stream_kwargs: dict[str, Any] = (
        {"event_deltas": ["agent.message", "agent.thinking"]} if stream_deltas else {}
    )
    stream = await client.beta.sessions.events.stream(session_id, **stream_kwargs)

    # A parked session accepts only tool results, so post those first (which
    # resumes it) and any user messages in a second call: the API validates a
    # whole batch against the session's current state. It also un-parks
    # asynchronously, so retry the follow-ups briefly on that specific error.
    follow_ups = [e for e in outbound if e.get("type") in ("user.message", "system.message")]
    results = [e for e in outbound if e.get("type") not in ("user.message", "system.message")]
    try:
        try:
            if results:
                await client.beta.sessions.events.send(session_id, events=results)
            if follow_ups:
                await _send_follow_ups(client, session_id, follow_ups)
        except BaseException:
            await _close_stream(stream)
            raise
        if on_sent is not None:
            sent = on_sent()
            if inspect.isawaitable(sent):
                await sent

        return await _consume(
            client=client,
            session_id=session_id,
            stream=stream,
            client_tools=client_tools,
            backend_tools=backend_tools,
            tool_confirmation=tool_confirmation,
            emit=emit,
        )
    finally:
        await _close_stream(stream)


async def _consume(
    *,
    client: Any,
    session_id: str,
    stream: Any,
    client_tools: Mapping[str, str],
    backend_tools: Mapping[str, BackendTool],
    tool_confirmation: str | None,
    emit: Emit,
) -> TurnOutcome:
    previews: dict[str, str] = {}
    closed_messages: set[str] = set()
    open_reasoning: set[str] = set()
    acked_tool_uses: set[str] = set()
    client_parks: set[str] = set()
    asked_confirmations: set[str] = set()

    def close_message(message_id: str) -> None:
        emit(TextMessageEndEvent(message_id=message_id))
        previews.pop(message_id, None)
        closed_messages.add(message_id)

    def close_reasoning(message_id: str) -> None:
        emit(ReasoningMessageEndEvent(message_id=message_id))
        emit(ReasoningEndEvent(message_id=message_id))
        open_reasoning.discard(message_id)

    def close_all() -> None:
        for message_id in list(previews):
            close_message(message_id)
        for reasoning_id in list(open_reasoning):
            close_reasoning(reasoning_id)

    def emit_tool_call(tool_call_id: str, name: str, tool_input: Any) -> None:
        emit(ToolCallStartEvent(tool_call_id=tool_call_id, tool_call_name=name))
        delta = json.dumps(
            tool_input if tool_input is not None else {}, separators=(",", ":")
        )
        emit(ToolCallArgsEvent(tool_call_id=tool_call_id, delta=delta))
        emit(ToolCallEndEvent(tool_call_id=tool_call_id))

    def emit_tool_result(tool_use_id: str, content: str) -> None:
        emit(
            ToolCallResultEvent(
                message_id=f"result_{tool_use_id}",
                tool_call_id=tool_use_id,
                content=content,
                role="tool",
            )
        )

    async def interrupt() -> None:
        try:
            await client.beta.sessions.events.send(
                session_id, events=[{"type": "user.interrupt"}]
            )
        except Exception:  # noqa: BLE001 - best-effort interrupt
            pass

    def fail(message: str, code: str | None = None) -> TurnOutcome:
        close_all()
        emit(RunErrorEvent(message=message, code=code))
        return TurnOutcome(status="errored")

    async def send_custom_tool_result(
        tool_use_id: str, text: str, is_error: bool
    ) -> None:
        await client.beta.sessions.events.send(
            session_id,
            events=[
                {
                    "type": "user.custom_tool_result",
                    "custom_tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": text}],
                    "is_error": is_error,
                }
            ],
        )
        acked_tool_uses.add(tool_use_id)

    async def run_backend_tool(
        tool_use_id: str, tool: BackendTool, tool_input: Any
    ) -> None:
        """Run a backend custom tool and post its result back into the session."""
        is_error = False
        try:
            result = tool.handler(tool_input)
            if inspect.isawaitable(result):
                result = await result
            text = str(result)
        except Exception as err:  # noqa: BLE001 - the tool's failure is reported to the agent
            is_error = True
            text = str(err) or err.__class__.__name__
        emit_tool_result(tool_use_id, text)
        await send_custom_tool_result(tool_use_id, text, is_error)

    async def consume() -> TurnOutcome:
        async for event in stream:
            event_type = get(event, "type")

            if event_type == "event_start":
                preview = get(event, "event")
                preview_type = get(preview, "type")
                preview_id = get(preview, "id")
                if preview_type == "agent.message":
                    emit(TextMessageStartEvent(message_id=preview_id, role="assistant"))
                    previews[preview_id] = ""
                elif preview_type == "agent.thinking":
                    open_reasoning.add(preview_id)
                    emit(ReasoningStartEvent(message_id=preview_id))
                    emit(
                        ReasoningMessageStartEvent(
                            message_id=preview_id, role="reasoning"
                        )
                    )

            elif event_type == "event_delta":
                event_id = get(event, "event_id")
                # Best-effort; the buffered agent.message is canonical.
                if event_id not in previews:
                    continue
                delta = get(event, "delta")
                content = get(delta, "content")
                if (
                    get(delta, "type") == "content_delta"
                    and get(content, "type") == "text"
                ):
                    text = get(content, "text") or ""
                    previews[event_id] += text
                    emit(TextMessageContentEvent(message_id=event_id, delta=text))

            elif event_type == "agent.thinking":
                # The thinking stretch finished. Its text is not exposed by the
                # API today, so this is a progress signal: close the reasoning
                # block we opened.
                event_id = get(event, "id")
                if event_id in open_reasoning:
                    close_reasoning(event_id)
                else:
                    emit(ReasoningStartEvent(message_id=event_id))
                    emit(ReasoningEndEvent(message_id=event_id))

            elif event_type == "agent.message":
                event_id = get(event, "id")
                if event_id in closed_messages:
                    continue
                final_text = text_of(get(event, "content"))
                if event_id not in previews:
                    emit(TextMessageStartEvent(message_id=event_id, role="assistant"))
                    previews[event_id] = ""
                    if final_text:
                        emit(
                            TextMessageContentEvent(
                                message_id=event_id, delta=final_text
                            )
                        )
                else:
                    previewed = previews[event_id]
                    if final_text.startswith(previewed):
                        if len(final_text) > len(previewed):
                            emit(
                                TextMessageContentEvent(
                                    message_id=event_id,
                                    delta=final_text[len(previewed) :],
                                )
                            )
                    else:
                        # Preview diverged from the final text: close it and re-emit the corrected whole.
                        close_message(event_id)
                        if final_text:
                            corrected_id = f"corrected_{event_id}"
                            emit(
                                TextMessageStartEvent(
                                    message_id=corrected_id, role="assistant"
                                )
                            )
                            emit(
                                TextMessageContentEvent(
                                    message_id=corrected_id, delta=final_text
                                )
                            )
                            emit(TextMessageEndEvent(message_id=corrected_id))
                        continue
                close_message(event_id)

            elif event_type == "agent.custom_tool_use":
                event_id = get(event, "id")
                name = get(event, "name")
                tool_input = get(event, "input")
                # Report the frontend's original tool name, which may differ
                # from the normalized name registered on the managed agent.
                emit_tool_call(event_id, client_tools.get(name, name), tool_input)
                if name in client_tools:
                    # The frontend executes this tool. Leave it unanswered; the
                    # session parks on it and the next run supplies the result.
                    client_parks.add(event_id)
                    continue
                backend = backend_tools.get(name)
                if backend is not None:
                    await run_backend_tool(event_id, backend, tool_input)
                    continue
                # Nothing can execute this tool. Answer with an error so the agent recovers.
                text = f'No handler is registered for tool "{name}".'
                emit_tool_result(event_id, text)
                await send_custom_tool_result(event_id, text, True)

            elif event_type == "agent.tool_use":
                event_id = get(event, "id")
                emit_tool_call(event_id, get(event, "name"), get(event, "input"))
                if get(event, "evaluated_permission") == "ask":
                    asked_confirmations.add(event_id)

            elif event_type == "agent.mcp_tool_use":
                event_id = get(event, "id")
                emit_tool_call(
                    event_id,
                    f"{get(event, 'mcp_server_name')}: {get(event, 'name')}",
                    get(event, "input"),
                )
                if get(event, "evaluated_permission") == "ask":
                    asked_confirmations.add(event_id)

            elif event_type == "agent.tool_result":
                emit_tool_result(
                    get(event, "tool_use_id"),
                    describe_tool_result(get(event, "content"))[:TOOL_RESULT_MAX_CHARS],
                )

            elif event_type == "agent.mcp_tool_result":
                emit_tool_result(
                    get(event, "mcp_tool_use_id"),
                    describe_tool_result(get(event, "content"))[:TOOL_RESULT_MAX_CHARS],
                )

            elif event_type == "span.model_request_end":
                # Closes any preview whose buffered agent.message never
                # arrived (e.g. an interrupted or errored model request).
                close_all()

            elif event_type == "session.error":
                error = get(event, "error")
                retry_status = get(error, "retry_status")
                if get(retry_status, "type") == "retrying":
                    continue  # transient; the session recovers on its own
                return fail(
                    get(error, "message") or "The session reported an error.",
                    get(error, "type"),
                )

            elif event_type == "session.status_idle":
                stop_reason = get(event, "stop_reason")
                reason_type = get(stop_reason, "type")
                if reason_type == "end_turn":
                    close_all()
                    return TurnOutcome(status="finished")
                if reason_type == "retries_exhausted":
                    return fail(
                        "The session gave up after exhausting its retries.",
                        "retries_exhausted",
                    )
                # requires_action: work out what the session is blocked on.
                event_ids: Sequence[str] = get(stop_reason, "event_ids") or []
                blocked_on = [
                    event_id
                    for event_id in event_ids
                    if event_id not in acked_tool_uses
                ]
                if not blocked_on:
                    continue  # everything is already answered; wait for it to resume

                confirmations = [
                    event_id
                    for event_id in blocked_on
                    if event_id in asked_confirmations
                ]
                if confirmations:
                    if not tool_confirmation:
                        await interrupt()
                        return fail(
                            "A tool requires confirmation but no confirmation policy is configured. "
                            'Set `tool_confirmation` to "allow" or "deny", or use a permission '
                            "policy that does not ask.",
                            "tool_confirmation_required",
                        )
                    await client.beta.sessions.events.send(
                        session_id,
                        events=[
                            {
                                "type": "user.tool_confirmation",
                                "tool_use_id": tool_use_id,
                                "result": tool_confirmation,
                            }
                            for tool_use_id in confirmations
                        ],
                    )
                    acked_tool_uses.update(confirmations)
                    if len(confirmations) == len(blocked_on):
                        continue

                client_tool_use_ids = [
                    event_id for event_id in blocked_on if event_id in client_parks
                ]
                unknown = [
                    event_id
                    for event_id in blocked_on
                    if event_id not in asked_confirmations
                    and event_id not in client_parks
                ]
                if unknown:
                    await interrupt()
                    return fail(
                        "The agent is waiting on an action this integration cannot answer.",
                        "unsupported_action",
                    )
                if client_tool_use_ids:
                    # Hand control back to the frontend to execute its tools.
                    close_all()
                    return TurnOutcome(
                        status="parked", client_tool_use_ids=client_tool_use_ids
                    )

            elif event_type in ("session.status_terminated", "session.deleted"):
                close_all()
                emit(
                    RunErrorEvent(
                        message="The managed session ended on the server. Send another message to start a fresh one.",
                        code="session_ended",
                    )
                )
                return TurnOutcome(status="errored", session_ended=True)

            # status_running, rescheduled, spans, thread events, echoed user events: ignored

        close_all()
        return fail(
            "The session event stream ended before the reply completed.", "stream_ended"
        )

    try:
        return await consume()
    finally:
        close_all()


def _is_sent_while_parked(exc: BaseException) -> bool:
    """The API rejects user messages while a session is parked on tool results."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status == 400 and "waiting on responses" in str(exc)


async def _send_follow_ups(client: Any, session_id: str, events: list[dict[str, Any]]) -> None:
    """Post follow-up messages, retrying while the session finishes un-parking."""
    delays = [0.15, 0.3, 0.6, 1.0, 1.5, 2.0]
    attempt = 0
    while True:
        try:
            await client.beta.sessions.events.send(session_id, events=events)
            return
        except Exception as exc:  # noqa: BLE001 - retry only the parked race
            if attempt >= len(delays) or not _is_sent_while_parked(exc):
                raise
            await asyncio.sleep(delays[attempt])
            attempt += 1
