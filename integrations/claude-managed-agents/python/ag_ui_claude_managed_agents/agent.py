"""ManagedAgentsAgent: an AG-UI agent backed by Claude Managed Agents."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
)
from anthropic import AsyncAnthropic

from ._util import maybe_await
from .sessions import InMemorySessionStore
from .tools import custom_tool_from, normalize_tool_name
from .turn import Emit, run_turn
from .types import BackendTool, SessionRecord, SessionStore, TurnOutcome

_DONE = object()


def _user_text(message: Any) -> str:
    """The text of a user message (string content or multimodal parts)."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content or []:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


@dataclass
class _Outbound:
    events: list[dict[str, Any]] = field(default_factory=list)
    still_parked: list[str] = field(default_factory=list)
    last_user_message_id: str | None = None


@dataclass
class _RunState:
    session_id: str | None = None


class ManagedAgentsAgent:
    """An AG-UI agent backed by Claude Managed Agents.

    Each AG-UI thread maps to one managed session; each run drives one turn
    of that session.
    """

    # Keys are scoped to the managed agent so distinct agents never collide.
    _busy_threads: ClassVar[set[str]] = set()

    def __init__(
        self,
        *,
        agent_id: str,
        environment_id: str,
        agent_version: int | None = None,
        client: AsyncAnthropic | None = None,
        session_store: SessionStore | None = None,
        backend_tools: list[BackendTool] | None = None,
        session_title: Callable[[str], str] | None = None,
        tool_confirmation: Literal["allow", "deny"] | None = None,
        turn_timeout_s: float = 300.0,
        stream_deltas: bool = True,
        scope: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.agent_version = agent_version
        self.client = client if client is not None else AsyncAnthropic()
        self.store: SessionStore = (
            session_store if session_store is not None else InMemorySessionStore()
        )
        self.backend_tools: dict[str, BackendTool] = {
            normalize_tool_name(tool.name): tool for tool in backend_tools or []
        }
        self.session_title = session_title
        self.tool_confirmation = tool_confirmation
        self.turn_timeout_s = turn_timeout_s
        self.stream_deltas = stream_deltas
        # Partitions thread<->session state (e.g. by authenticated user). Thread
        # ids are client-supplied, so without a scope any caller that knows a
        # thread id resumes that thread's session.
        self.scope = scope
        # Strong references to in-flight worker tasks so they are not collected.
        self._tasks: set[asyncio.Task[None]] = set()

    async def run(self, input: RunAgentInput) -> AsyncIterator[BaseEvent]:  # noqa: A002 - matches AG-UI adapters
        """Run one turn for `input`, yielding AG-UI events."""
        queue: asyncio.Queue[Any] = asyncio.Queue()
        state = _RunState()
        worker = asyncio.create_task(self._drive(input, queue.put_nowait, state))
        self._tasks.add(worker)
        worker.add_done_callback(self._tasks.discard)
        worker.add_done_callback(lambda _task: queue.put_nowait(_DONE))
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                yield item
        finally:
            # The client went away (or iteration stopped): cancel the turn,
            # which interrupts the session.
            if not worker.done():
                worker.cancel()

    async def _drive(self, input: RunAgentInput, emit: Emit, state: _RunState) -> None:  # noqa: A002
        try:
            await asyncio.wait_for(
                self._run_turn_for_input(input, emit, state),
                timeout=self.turn_timeout_s,
            )
        except asyncio.TimeoutError:
            await self._interrupt(state.session_id)
            seconds = round(self.turn_timeout_s)
            emit(
                RunErrorEvent(
                    message=f"The turn exceeded the {seconds}s limit and was interrupted."
                )
            )
        except asyncio.CancelledError:
            # Client disconnected mid-turn: stop the session; there is nobody left to tell.
            await self._interrupt(state.session_id)
            raise
        except Exception as err:  # noqa: BLE001 - surfaced to the client as RUN_ERROR
            emit(RunErrorEvent(message=str(err) or "The run failed."))

    async def _interrupt(self, session_id: str | None) -> None:
        if not session_id:
            return
        try:
            await self.client.beta.sessions.events.send(
                session_id, events=[{"type": "user.interrupt"}]
            )
        except Exception:  # noqa: BLE001 - best-effort interrupt
            pass

    async def _run_turn_for_input(
        self, input: RunAgentInput, emit: Emit, state: _RunState
    ) -> None:  # noqa: A002
        thread_id, run_id = input.thread_id, input.run_id
        busy_key = self._thread_key(thread_id)
        emit(RunStartedEvent(thread_id=thread_id, run_id=run_id))
        if input.state is not None:
            emit(StateSnapshotEvent(snapshot=input.state))

        if busy_key in ManagedAgentsAgent._busy_threads:
            emit(RunErrorEvent(message="A run is already in progress on this thread."))
            return
        # Check for something sendable before touching the API, so a malformed
        # run does not create an orphan session.
        if not self._has_sendable_content(input.messages):
            emit(
                RunErrorEvent(
                    message="There is nothing to send: this run has no user message or tool result."
                )
            )
            return

        ManagedAgentsAgent._busy_threads.add(busy_key)
        try:
            record = await self._get_or_create_session(thread_id, input, emit)
            if record is None:
                emit(
                    RunErrorEvent(
                        message="There is nothing to send: a tool result arrived for a thread with no session."
                    )
                )
                return
            state.session_id = record.session_id
            await self._sync_client_tools(record, input.tools or [])

            outbound = self._outbound_events(record, input.messages)
            if not outbound.events:
                emit(
                    RunErrorEvent(
                        message="There is nothing new to send: no user message or tool result in this run."
                    )
                )
                return

            # Some parked tool calls are still unanswered: post what we have and
            # stay parked instead of waiting on a session that will not resume.
            if outbound.still_parked:
                await self.client.beta.sessions.events.send(
                    record.session_id, events=outbound.events
                )
                record.pending_client_tool_use_ids = outbound.still_parked
                record.last_user_message_id = (
                    outbound.last_user_message_id or record.last_user_message_id
                )
                await maybe_await(self.store.set(self._store_key(thread_id), record))
                emit(RunFinishedEvent(thread_id=thread_id, run_id=run_id))
                return

            # Persist delivery as soon as the events land, so a timeout or
            # disconnect later in the turn does not re-post them next run.
            async def on_sent() -> None:
                record.pending_client_tool_use_ids = []
                if outbound.last_user_message_id:
                    record.last_user_message_id = outbound.last_user_message_id
                await maybe_await(self.store.set(self._store_key(thread_id), record))

            outcome = await run_turn(
                client=self.client,
                session_id=record.session_id,
                outbound=outbound.events,
                on_sent=on_sent,
                client_tools={
                    normalize_tool_name(tool.name): tool.name
                    for tool in input.tools or []
                },
                backend_tools=self.backend_tools,
                tool_confirmation=self.tool_confirmation,
                stream_deltas=self.stream_deltas,
                emit=emit,
            )

            await self._record_outcome(thread_id, record, outcome)
            if outcome.status != "errored":
                emit(RunFinishedEvent(thread_id=thread_id, run_id=run_id))
        finally:
            ManagedAgentsAgent._busy_threads.discard(busy_key)

    def _has_sendable_content(self, messages: Sequence[Any]) -> bool:
        """Whether the run carries a user message with text or a tool result."""
        for message in messages:
            role = getattr(message, "role", None)
            if role == "tool":
                return True
            if role == "user" and _user_text(message).strip():
                return True
        return False

    def _thread_key(self, thread_id: str) -> str:
        """Shared-state key scoped to this managed agent (and scope, if set)."""
        return f"{self.agent_id}:{self._store_key(thread_id)}"

    def _store_key(self, thread_id: str) -> str:
        """Store key: the thread id, partitioned by scope when configured."""
        return f"{self.scope}:{thread_id}" if self.scope else thread_id

    async def _record_outcome(
        self, thread_id: str, record: SessionRecord, outcome: TurnOutcome
    ) -> None:
        if outcome.status == "errored" and outcome.session_ended:
            await maybe_await(self.store.delete(self._store_key(thread_id)))
            return
        if outcome.status != "parked":
            return
        record.pending_client_tool_use_ids = list(outcome.client_tool_use_ids)
        await maybe_await(self.store.set(self._store_key(thread_id), record))

    def _outbound_events(
        self, record: SessionRecord, messages: Sequence[Any]
    ) -> _Outbound:
        """Work out what to post into the session for this run: results for any
        tool calls the frontend was asked to run, plus every user message not
        yet delivered (in order).
        """
        events: list[dict[str, Any]] = []
        pending = list(record.pending_client_tool_use_ids)

        for message in messages:
            if getattr(message, "role", None) != "tool":
                continue
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id not in pending:
                continue
            events.append(
                {
                    "type": "user.custom_tool_result",
                    "custom_tool_use_id": tool_call_id,
                    "content": [{"type": "text", "text": message.content or ""}],
                    "is_error": bool(getattr(message, "error", None)),
                }
            )
            pending.remove(tool_call_id)

        # User messages after the last delivered one; on first contact, just the newest.
        last_user_message_id: str | None = None
        user_messages = [
            message for message in messages if getattr(message, "role", None) == "user"
        ]
        delivered_index = next(
            (
                i
                for i, message in enumerate(user_messages)
                if message.id == record.last_user_message_id
            ),
            -1,
        )
        undelivered = (
            user_messages[delivered_index + 1 :]
            if delivered_index >= 0
            else user_messages[-1:]
        )
        for message in undelivered:
            text = _user_text(message).strip()
            if not text:
                continue
            events.append(
                {"type": "user.message", "content": [{"type": "text", "text": text}]}
            )
            last_user_message_id = message.id

        # The user moved on without answering the tools the frontend was asked
        # to run: fail those calls so the agent can respond to the new message.
        if last_user_message_id is not None and pending:
            abandoned = [
                {
                    "type": "user.custom_tool_result",
                    "custom_tool_use_id": tool_use_id,
                    "content": [
                        {
                            "type": "text",
                            "text": "The user did not provide a result for this tool call.",
                        }
                    ],
                    "is_error": True,
                }
                for tool_use_id in pending
            ]
            events[:0] = abandoned
            pending.clear()

        return _Outbound(
            events=events,
            still_parked=pending,
            last_user_message_id=last_user_message_id,
        )

    async def _get_or_create_session(
        self, thread_id: str, input: RunAgentInput, emit: Emit
    ) -> SessionRecord | None:  # noqa: A002
        existing = await maybe_await(self.store.get(self._store_key(thread_id)))
        if existing is not None:
            return existing

        # A tool result only answers a pending call on an existing session;
        # never create a session to receive one.
        if not any(
            getattr(m, "role", None) == "user" and _user_text(m).strip()
            for m in input.messages
        ):
            return None

        # The busy-thread gate serializes runs per thread, so creation cannot race.
        record = await self._create_session(thread_id, input.tools or [])
        await maybe_await(self.store.set(self._store_key(thread_id), record))
        emit(
            CustomEvent(
                name="managed_agents.session",
                value={"sessionId": record.session_id, "threadId": thread_id},
            )
        )
        return record

    async def _create_session(
        self, thread_id: str, client_tools: Sequence[Any]
    ) -> SessionRecord:
        custom_tools = self._custom_tools(client_tools)
        title = (
            self.session_title(thread_id)
            if self.session_title
            else f"AG-UI thread {thread_id}"
        )

        agent: dict[str, Any]
        if not custom_tools:
            agent = {"type": "agent", "id": self.agent_id}
            if self.agent_version is not None:
                agent["version"] = self.agent_version
        else:
            agent = {"type": "agent_with_overrides", "id": self.agent_id}
            if self.agent_version is not None:
                agent["version"] = self.agent_version
            # Overrides replace the tool list, so keep the agent's own tools.
            agent["tools"] = [*(await self._base_tools()), *custom_tools]

        session = await self.client.beta.sessions.create(
            agent=agent, environment_id=self.environment_id, title=title
        )
        return SessionRecord(
            session_id=session.id, tool_names=[tool["name"] for tool in custom_tools]
        )

    def _custom_tools(self, client_tools: Sequence[Any]) -> list[dict[str, Any]]:
        """Frontend tools plus configured backend tools, as custom tool definitions.

        Keyed by normalized name; on a collision the frontend tool wins, matching
        dispatch order in the turn loop.
        """
        by_name: dict[str, dict[str, Any]] = {}
        for tool in self.backend_tools.values():
            by_name[normalize_tool_name(tool.name)] = custom_tool_from(tool)
        for tool in client_tools:
            custom = custom_tool_from(tool)
            by_name[custom["name"]] = custom
        return list(by_name.values())

    async def _sync_client_tools(
        self, record: SessionRecord, client_tools: Sequence[Any]
    ) -> None:
        """Register any client tools the session's agent does not yet have.

        The tool list is a full replacement, so we merge with what the agent has.
        """
        desired = self._custom_tools(client_tools)
        known = set(record.tool_names)
        if all(tool["name"] in known for tool in desired):
            return
        await self.client.beta.sessions.update(
            record.session_id,
            agent={"tools": [*(await self._base_tools()), *desired]},
        )
        record.tool_names = [tool["name"] for tool in desired]

    async def _base_tools(self) -> list[Any]:
        """The tools defined on the managed agent itself, fetched fresh so console edits apply."""
        if self.agent_version is not None:
            agent = await self.client.beta.agents.retrieve(
                self.agent_id, version=self.agent_version
            )
        else:
            agent = await self.client.beta.agents.retrieve(self.agent_id)
        # The read shape is structurally compatible with the params shape.
        return list(getattr(agent, "tools", None) or [])
