"""Public data types for the Managed Agents adapter."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class BackendTool:
    """A tool the agent may call, executed on this server rather than in the
    browser.

    Registered on the managed agent as a `custom` tool. When the agent calls
    it we run `handler`, stream the call and result to the UI, and post the
    result back into the session.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema for the tool input."""
    handler: Callable[[Any], Awaitable[str] | str]


@dataclass
class SessionRecord:
    """Persistent mapping between an AG-UI thread and a managed session."""

    session_id: str
    tool_names: list[str] = field(default_factory=list)
    """Custom tool names currently registered on the session's agent."""
    pending_client_tool_use_ids: list[str] = field(default_factory=list)
    """Custom tool calls handed to the frontend that the session is parked on.

    The next run must answer them with `role: "tool"` messages.
    """
    last_user_message_id: str | None = None
    """ID of the last user message forwarded into the session."""


@runtime_checkable
class SessionStore(Protocol):
    """Where thread-to-session mappings live.

    The default is in-memory (lost on restart, in which case a fresh session
    is created). Provide your own to survive restarts or run several replicas.
    Methods may be plain functions or coroutines.
    """

    def get(
        self, thread_id: str
    ) -> SessionRecord | None | Awaitable[SessionRecord | None]: ...

    def set(self, thread_id: str, record: SessionRecord) -> None | Awaitable[None]: ...

    def delete(self, thread_id: str) -> None | Awaitable[None]: ...


TurnStatus = Literal["finished", "parked", "errored"]


@dataclass
class TurnOutcome:
    """How a turn ended.

    `parked`: the session is parked on custom tool calls the frontend must
    answer. `errored`: a RUN_ERROR was already emitted.
    """

    status: TurnStatus
    client_tool_use_ids: list[str] = field(default_factory=list)
    session_ended: bool = False
