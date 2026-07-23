"""A scripted stand-in for the Anthropic client's managed-agents surface.

Nothing here touches the network. Stream events are plain dicts shaped like
the SDK's session events; the adapter reads them through attribute-or-key
access, so dicts and models behave the same.
"""

import asyncio
from types import SimpleNamespace
from typing import Any


class FakeStream:
    """An async-iterable of scripted events. An `asyncio.Event` entry blocks
    the stream until it is set (used to keep a run in flight)."""

    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            if self.closed:
                return
            if isinstance(event, asyncio.Event):
                await event.wait()
                continue
            yield event

    async def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(
        self,
        *,
        streams: list[list[Any]] | None = None,
        agent_tools: list[Any] | None = None,
        session_id: str = "sesn_1",
    ) -> None:
        self._streams = list(streams or [])
        self.agent_tools = (
            agent_tools
            if agent_tools is not None
            else [
                {"type": "agent_toolset_20260401", "configs": [], "default_config": {}}
            ]
        )
        self.session_id = session_id
        self.sent: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.update_calls: list[tuple[str, dict[str, Any]]] = []
        self.retrieve_calls: list[tuple[str, dict[str, Any]]] = []
        self.stream_calls: list[tuple[str, dict[str, Any]]] = []
        self.streams_opened: list[FakeStream] = []

        events = SimpleNamespace(stream=self._stream, send=self._send)
        self.beta = SimpleNamespace(
            agents=SimpleNamespace(retrieve=self._retrieve),
            sessions=SimpleNamespace(
                create=self._create, update=self._update, events=events
            ),
        )

    async def _stream(self, session_id: str, **kwargs: Any) -> FakeStream:
        self.stream_calls.append((session_id, kwargs))
        events = self._streams.pop(0) if self._streams else []
        stream = FakeStream(events)
        self.streams_opened.append(stream)
        return stream

    async def _send(self, session_id: str, *, events: list[Any]) -> SimpleNamespace:
        self.sent.append({"session_id": session_id, "events": list(events)})
        return SimpleNamespace(data=[])

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.create_calls.append(kwargs)
        return SimpleNamespace(id=self.session_id)

    async def _update(self, session_id: str, **kwargs: Any) -> SimpleNamespace:
        self.update_calls.append((session_id, kwargs))
        return SimpleNamespace(id=session_id)

    async def _retrieve(self, agent_id: str, **kwargs: Any) -> SimpleNamespace:
        self.retrieve_calls.append((agent_id, kwargs))
        return SimpleNamespace(tools=self.agent_tools)
