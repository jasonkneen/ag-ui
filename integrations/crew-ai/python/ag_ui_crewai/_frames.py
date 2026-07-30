"""Translate crewai stream events into AG-UI wire events.

crewai's public streaming contract (``StreamFrame`` / ``AsyncStreamSession``,
landed 1.6.0) emits one ordered frame per event a flow raises. The frame gives
us ORDERING and the identity of the emitting flow; the RAW event object (parked
by our scoped stream sink, keyed by ``event.event_id == StreamFrame.id``) gives
us the EXACT payload. This module is the SINGLE translation seam that maps the
raw events the bridge cares about onto AG-UI events.

Why raw events, not ``frame.data``:
``StreamFrame.data`` is ``event.to_json(exclude=...)`` -> crewai's
``to_serializable(max_depth=5)``, which (a) ``repr()``-quotes anything at depth
>= 5 (plain strings become ``"'text'"``) and (b) recursively drops any dict key
named ``type`` / ``timestamp`` / ``event_id`` / ... . That silently corrupts the
progressive ``STATE_SNAPSHOT`` payloads ``copilotkit_emit_state`` exists to
stream (verified against the crewai 1.15.7 wheel). Emitting straight from the
raw event object bypasses all of it.

Two raw event sources feed us (verified against the 1.15.7 wheel):

* Flow-lifecycle events (``FlowStartedEvent`` / ``MethodExecutionStartedEvent``
  / ``MethodExecutionFinishedEvent`` / ``FlowFinishedEvent``), whose ``.type``
  is the crewai lifecycle string (``"flow_started"`` etc.).
* The bridge's own ``Bridged*`` events (emitted by ``sdk.copilotkit_stream`` &
  friends via ``crewai_event_bus.emit``), whose ``.type`` is the AG-UI
  ``EventType`` string and whose typed attributes (``snapshot`` / ``delta`` /
  ``message_id`` / ...) carry the verbatim, un-serialized payload.

Source filtering: a ``crew.kickoff`` inside a flow method drives a NESTED flow
whose own lifecycle/method events leak onto the same scoped sink. The driver
drops those nested flow/method events by source identity (only events whose
``source is flow_copy`` are parked) before they reach this translator. Crew and
Agent lifecycle events are the exception: they arrive with a non-outer source
but are surfaced deliberately (the sink parks them because they are scoped to
this run) so this translator can attribute the Crew / Agent hierarchy. The
translator therefore sees outer-source flow/method events plus run-scoped
crew/agent events, but never nested-flow frames.

UPSTREAM FRAME RETENTION: crewai's ``AsyncStreamSession``
(``crewai.types.streaming.StreamSessionBase``) appends EVERY iterated frame to
its ``self._frames`` list and keeps them for the session's life (to back its
``.frames`` / ``.result`` replay accessors) — we cannot drop consumed frames
without forking upstream, and doing so would break those accessors. The growth
is bounded per-request (one session per run, torn down when the request ends),
NOT a cross-request leak. The bridge's OWN raw-event lookup buffer does not
share this behavior: ``endpoint._run_flow_frame_stream`` ``pop``s each parked
raw event by ``frame.id`` the moment its frame is consumed, so our buffer stays
proportional to in-flight (not total) frames. If a future crewai release makes
frame retention opt-out, revisit the session consumption in ``endpoint.py``.

SWAPPABLE EMISSION SHAPE: CrewAI is currently
the only integration emitting TEXT_MESSAGE_CHUNK / TOOL_CALL_CHUNK (chunks)
rather than the START/CONTENT/END triples the six other integrations emit. That
final choice belongs to a Parity-lane ticket, not this migration. The translator
therefore routes LLM text / LLM-tool-call emission through a single
``emission_shape`` strategy that DEFAULTS to ``"chunks"`` so this migration is
byte-for-byte behavior-preserving on that channel. The ``"triples"`` strategy is
a deliberate NotImplementedError placeholder — the Parity ticket owns wiring it
up and flipping the default.

MCP EVENTS are the ONE exception to "chunks-only": crewai's discrete
MCP tool executions (name + full args + result arrive together, not streamed)
map to canonical ``TOOL_CALL_START/ARGS/END/RESULT`` triples via the shared
``mcp.translate_mcp_event`` seam, and MCP lifecycle events map to ``CUSTOM``.
This is independent of the ``emission_shape`` strategy above (which governs only
the streaming LLM text / tool-call channel); see the wire-shape note in
``mcp.py`` for why discrete MCP calls use triples.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
)
from ag_ui.core.events import (
    RawEvent,
    TextMessageChunkEvent,
    ToolCallChunkEvent,
    StepFinishedEvent,
    MessagesSnapshotEvent,
    StateSnapshotEvent,
    CustomEvent,
)

from .sdk import litellm_messages_to_ag_ui_messages
from .mcp import is_mcp_event, translate_mcp_event
from .attribution import (
    BoundaryTracker,
    FLOW_METHOD,
    CREW,
    AGENT,
    step_started_event,
    step_finished_event,
)

_LOGGER = logging.getLogger(__name__)

# crewai lifecycle-event ``type`` strings. Pinned as constants so a rename
# upstream surfaces here rather than silently producing no wire events.
_FLOW_STARTED = "flow_started"
_FLOW_FINISHED = "flow_finished"
_METHOD_STARTED = "method_execution_started"
_METHOD_FINISHED = "method_execution_finished"
_METHOD_FAILED = "method_execution_failed"

# Crew- and Agent-level ``type`` strings. The ordered StreamFrame path sees
# these (crew/agent frames arrive on the outer flow's ``astream`` with a source
# that is NOT the outer flow; see ``endpoint._sink``), so it can reconstruct the
# Flow-method -> Crew -> Agent hierarchy. Pinned as constants for rename safety.
_CREW_STARTED = "crew_kickoff_started"
_CREW_COMPLETED = "crew_kickoff_completed"
_CREW_FAILED = "crew_kickoff_failed"
_AGENT_STARTED = "agent_execution_started"
_AGENT_COMPLETED = "agent_execution_completed"
_AGENT_ERROR = "agent_execution_error"

# Single source of truth for identifying crew/agent lifecycle events. The sink
# in ``endpoint.py`` parks these regardless of source; the translator matches
# them by the individual constants above.
CREW_AGENT_LIFECYCLE_TYPES = frozenset({
    _CREW_STARTED,
    _CREW_COMPLETED,
    _CREW_FAILED,
    _AGENT_STARTED,
    _AGENT_COMPLETED,
    _AGENT_ERROR,
})

_SUPPORTED_EMISSION_SHAPES = frozenset({"chunks", "triples"})


def _coerce_name(value: Any, fallback: str) -> str:
    """Coerce a crewai identity (method / crew name) to a non-empty ``str``.

    crewai populates ``method_name`` / ``crew_name`` as strings today, but the
    attribution path joins them into ``path`` / ``qualified_name`` and uses them
    as the boundary pairing key, so a stray ``None`` (an event constructed via
    ``model_construct`` that skipped the field) or a non-str must never reach it
    and raise. Empty / whitespace-only values fall back too.
    """
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _agent_role(event: Any) -> str:
    """Derive a stable, always-``str`` role for an agent lifecycle event.

    Prefers the agent's ``role``, falls back to its ``id`` (a UUID object on
    crewai's ``BaseAgent``, hence the ``str(...)`` coercion), then to the
    literal ``"agent"``. The SAME derivation is used on start and on
    completion/error so the boundary pairs by a stable name.
    """
    agent = getattr(event, "agent", None)
    role = getattr(agent, "role", None) if agent is not None else None
    if role is None or not str(role).strip():
        role = getattr(agent, "id", None) if agent is not None else None
    return _coerce_name(role, "agent")


class StreamFrameTranslator:
    """Stateless-ish mapper from a RAW crewai/bridge event to AG-UI events.

    One instance per run (it carries the run correlation ids and a
    ``state_provider`` closure over the live flow copy). ``translate`` returns
    zero or more AG-UI events for a single raw event; unrecognised event types
    return ``[]`` (behavior-preserving: the legacy listener produced nothing for
    crewai's native llm / tools / messages channels or internal events).

    The driver forwards outer-flow lifecycle/method events (``source is
    flow_copy``) plus the run-scoped Crew/Agent lifecycle events (which carry a
    non-outer source); nested-flow frames are filtered out before they reach
    the translator, so every event it sees belongs to the outer run.
    """

    def __init__(
        self,
        *,
        thread_id: str,
        run_id: str,
        state_provider: Callable[[], Any],
        emission_shape: str = "chunks",
    ) -> None:
        if emission_shape not in _SUPPORTED_EMISSION_SHAPES:
            raise ValueError(
                f"Unknown emission_shape {emission_shape!r}; "
                f"expected one of {sorted(_SUPPORTED_EMISSION_SHAPES)}"
            )
        self._thread_id = thread_id
        self._run_id = run_id
        self._state_provider = state_provider
        self.emission_shape = emission_shape
        # One ordered boundary stack per run, driven in emit order by
        # ``translate``: ``enter`` on a start frame, ``exit`` on the matching
        # finish, ``drain_all`` at run end so every STEP_STARTED is closed.
        self._tracker = BoundaryTracker()
        # Run-lifecycle idempotency. A single AG-UI HTTP run must
        # emit EXACTLY ONE ``RUN_STARTED`` (first) and ONE ``RUN_FINISHED``
        # (last). Nested-flow lifecycle events are already filtered out by the
        # driver's source gate, so no depth counter is needed; these flags are
        # a cheap belt-and-braces guard against a double emit.
        self._run_started_emitted = False
        self._run_finished_emitted = False

    # -- public API --------------------------------------------------------

    @property
    def run_started(self) -> bool:
        """Whether RUN_STARTED has been emitted for this run."""
        return self._run_started_emitted

    @property
    def run_finished(self) -> bool:
        """Whether RUN_FINISHED has been emitted (the run terminated)."""
        return self._run_finished_emitted

    def translate(self, event: Any) -> list[Any]:
        """Map one raw crewai/bridge event to the AG-UI events it produces."""
        event_type = getattr(event, "type", None)

        if event_type == _FLOW_STARTED:
            # Emit RUN_STARTED for the outermost flow only; the driver drops
            # nested-flow events, so a second one would be a defect.
            if self._run_started_emitted:
                return []
            self._run_started_emitted = True
            return [
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=self._thread_id,
                    run_id=self._run_id,
                )
            ]
        if event_type == _FLOW_FINISHED:
            if self._run_finished_emitted or not self._run_started_emitted:
                # Never emit a second RUN_FINISHED, and never synthesize one for
                # a run that never opened.
                return []
            self._run_finished_emitted = True
            # Close every STEP_STARTED still open before RUN_FINISHED so a
            # boundary whose finish frame never arrived does not dangle.
            # ``drain_all`` returns them deepest-first for balanced closes.
            events: list[Any] = [
                step_finished_event(b) for b in self._tracker.drain_all()
            ]
            events.append(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=self._thread_id,
                    run_id=self._run_id,
                )
            )
            return events
        if event_type == _METHOD_STARTED:
            return self._method_started_events(event)
        if event_type == _METHOD_FINISHED:
            return self._method_finished_events(event)
        if event_type == _METHOD_FAILED:
            # Close the method boundary so a failed flow method (the flow may
            # continue) does not leave a dangling STEP_STARTED. No snapshots.
            method_name = _coerce_name(getattr(event, "method_name", None), "method")
            return self._close_boundaries(
                self._tracker.exit(FLOW_METHOD, method_name), _METHOD_FAILED
            )
        if event_type == _CREW_STARTED:
            return self._crew_started_events(event)
        if event_type in (_CREW_COMPLETED, _CREW_FAILED):
            return self._crew_finished_events(event)
        if event_type == _AGENT_STARTED:
            return self._agent_started_events(event)
        if event_type in (_AGENT_COMPLETED, _AGENT_ERROR):
            return self._agent_finished_events(event)

        # crewai's first-class MCP events (crewai >= 1.4). Emitted with
        # the agent/crew as source (not the flow), so the driver's sink parks
        # them by TYPE rather than source identity; here they map to TOOL_CALL_*
        # (tool executions) and CUSTOM (lifecycle) via the shared translator.
        if is_mcp_event(event):
            return translate_mcp_event(event)

        # Bridge-emitted events carry the AG-UI EventType string as ``.type``
        # and the verbatim payload on typed attributes (no ``to_serializable``).
        if event_type == EventType.TEXT_MESSAGE_CHUNK:
            return self._text_events(event)
        if event_type == EventType.TOOL_CALL_CHUNK:
            return self._tool_events(event)
        if event_type == EventType.CUSTOM:
            return [
                CustomEvent(
                    type=EventType.CUSTOM,
                    name=getattr(event, "name", None),
                    value=getattr(event, "value", None),
                )
            ]
        if event_type == EventType.STATE_SNAPSHOT:
            return [
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=getattr(event, "snapshot", None),
                )
            ]

        # crewai native llm / tools / messages / lifecycle events and internal
        # events (e.g. ``cc_env``) are intentionally dropped — the legacy
        # listener never surfaced them, so ignoring them keeps the wire output
        # identical. Surfacing them is a Parity-lane decision.
        return []

    def finalize(self) -> list[Any]:
        """Belt-and-braces terminal.

        Called once when the frame stream exhausts cleanly. If the run opened
        (RUN_STARTED) but no outer ``flow_finished`` closed it — e.g. the outer
        method caught a nested-flow error and the stream just ended — emit the
        missing RUN_FINISHED so the client NEVER sees a run that never ends. The
        errored path terminates via RUN_ERROR instead and must not call this.
        """
        if self._run_started_emitted and not self._run_finished_emitted:
            self._run_finished_emitted = True
            # Same drain-before-terminal discipline as the ``flow_finished``
            # branch: close every open boundary (deepest-first) so the client
            # never sees a dangling STEP_STARTED.
            events: list[Any] = [
                step_finished_event(b) for b in self._tracker.drain_all()
            ]
            events.append(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=self._thread_id,
                    run_id=self._run_id,
                )
            )
            return events
        return []

    # -- lifecycle --------------------------------------------------------

    def _method_started_events(self, event: Any) -> list[Any]:
        """Open a Flow-method boundary and emit its attributed STEP_STARTED.

        The ``step_name`` on the wire is unchanged (the method name); the
        ``raw_event.attribution`` payload is the additive part so nested Crew /
        Agent steps chain under it.
        """
        method_name = _coerce_name(getattr(event, "method_name", None), "method")
        boundary = self._tracker.enter(
            FLOW_METHOD,
            method_name,
            fingerprint=getattr(event, "source_fingerprint", None),
            flow_name=getattr(event, "flow_name", None),
        )
        return [step_started_event(boundary, source_event_type=_METHOD_STARTED)]

    def _crew_started_events(self, event: Any) -> list[Any]:
        """Open a Crew boundary (child of the current method) -> STEP_STARTED."""
        crew_name = _coerce_name(getattr(event, "crew_name", None), "crew")
        boundary = self._tracker.enter(
            CREW,
            crew_name,
            fingerprint=getattr(event, "source_fingerprint", None),
        )
        return [step_started_event(boundary, source_event_type=_CREW_STARTED)]

    def _crew_finished_events(self, event: Any) -> list[Any]:
        """Close the nearest matching Crew boundary -> balanced STEP_FINISHED(s).

        ``exit`` returns the matched boundary plus any dangling inners
        (deepest-first); only the matched (last) boundary is tagged with the
        source event type. An empty return (no open crew of that name) emits
        nothing rather than an unbalanced close.
        """
        crew_name = _coerce_name(getattr(event, "crew_name", None), "crew")
        source_type = getattr(event, "type", None)
        return self._close_boundaries(
            self._tracker.exit(CREW, crew_name), source_type
        )

    def _agent_started_events(self, event: Any) -> list[Any]:
        """Open an Agent boundary (child of the current crew) -> STEP_STARTED."""
        role = _agent_role(event)
        boundary = self._tracker.enter(
            AGENT,
            role,
            fingerprint=getattr(event, "source_fingerprint", None),
        )
        return [step_started_event(boundary, source_event_type=_AGENT_STARTED)]

    def _agent_finished_events(self, event: Any) -> list[Any]:
        """Close the nearest matching Agent boundary -> balanced STEP_FINISHED(s).

        Uses the SAME ``_agent_role`` derivation as the start so the boundary
        pairs by a stable name. Empty return => emit nothing.
        """
        role = _agent_role(event)
        source_type = getattr(event, "type", None)
        return self._close_boundaries(
            self._tracker.exit(AGENT, role), source_type
        )

    @staticmethod
    def _close_boundaries(closed: list[Any], source_event_type: Any) -> list[Any]:
        """Turn an ``exit``/``drain`` result into balanced STEP_FINISHED events.

        ``closed`` is deepest-first (inner boundaries first). Only the matched
        boundary (the LAST element, i.e. the one whose finish frame we actually
        received) is tagged with ``source_event_type`` for provenance; the
        dangling inners closed alongside it get no source tag (their own finish
        frame never arrived). An empty list yields no events.
        """
        events: list[Any] = []
        last_index = len(closed) - 1
        for index, boundary in enumerate(closed):
            tag = source_event_type if index == last_index else None
            events.append(step_finished_event(boundary, source_event_type=tag))
        return events

    def _method_finished_events(self, event: Any) -> list[Any]:
        """MESSAGES_SNAPSHOT + STATE_SNAPSHOT + balanced STEP_FINISHED(s).

        Reads the LIVE flow state via ``state_provider`` — exactly what the
        legacy listener did with ``source.state`` — rather than the serialized
        ``event`` payload, so message / state shapes round-trip through the same
        ``litellm_messages_to_ag_ui_messages`` path as before. The two snapshots
        are emitted VERBATIM (unchanged behaviour).

        The final close is balanced: ``exit(FLOW_METHOD, method_name)`` returns
        the matched method boundary plus any Crew / Agent boundaries left
        dangling by a lost completion frame (deepest-first). If no boundary
        matches (the tracker never saw this method's start), fall back to a flat
        ``StepFinishedEvent(step_name=method_name)`` rather than dropping the
        close entirely.
        """
        state = self._state_provider()
        raw_messages = (
            getattr(state, "messages", None)
            or (state.get("messages") if isinstance(state, dict) else None)
            or []
        )
        messages = litellm_messages_to_ag_ui_messages(raw_messages)
        snapshot = (
            state
            if isinstance(state, dict)
            else state.model_dump()
            if hasattr(state, "model_dump")
            else {}
        )
        method_name = _coerce_name(getattr(event, "method_name", None), "method")
        events: list[Any] = [
            MessagesSnapshotEvent(
                type=EventType.MESSAGES_SNAPSHOT,
                messages=messages,
            ),
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=snapshot,
            ),
        ]
        closed = self._tracker.exit(FLOW_METHOD, method_name)
        if closed:
            events.extend(self._close_boundaries(closed, _METHOD_FINISHED))
        else:
            # No open boundary for this method: fall back to a flat close,
            # using the same coerced ``method_name`` the START used so a None
            # cannot leak onto the wire.
            events.append(
                StepFinishedEvent(
                    type=EventType.STEP_FINISHED,
                    step_name=method_name,
                )
            )
        return events

    # -- emission-shape strategy (default "chunks") -----------------------

    def _text_events(self, event: Any) -> list[Any]:
        if self.emission_shape == "chunks":
            return [
                TextMessageChunkEvent(
                    type=EventType.TEXT_MESSAGE_CHUNK,
                    message_id=getattr(event, "message_id", None),
                    role=getattr(event, "role", None),
                    delta=getattr(event, "delta", None),
                )
            ]
        # TODO(Parity lane): emit TEXT_MESSAGE_START / _CONTENT / _END
        # triples here to match the other six integrations. Do NOT flip the
        # default in this migration (it must stay behavior-preserving on the
        # wire).
        raise NotImplementedError(
            "emission_shape='triples' is a Parity-lane placeholder; "
            "the StreamFrame migration ships the behavior-preserving "
            "'chunks' shape only."
        )

    def _tool_events(self, event: Any) -> list[Any]:
        if self.emission_shape == "chunks":
            return [
                ToolCallChunkEvent(
                    type=EventType.TOOL_CALL_CHUNK,
                    tool_call_id=getattr(event, "tool_call_id", None),
                    tool_call_name=getattr(event, "tool_call_name", None),
                    delta=getattr(event, "delta", None),
                )
            ]
        # TODO(Parity lane): TOOL_CALL_START / _ARGS / _END triples — see
        # the note in ``_text_events``.
        raise NotImplementedError(
            "emission_shape='triples' is a Parity-lane placeholder; "
            "the StreamFrame migration ships the behavior-preserving "
            "'chunks' shape only."
        )


# Event types the translator recognises. Used to decide whether an event is
# "unmapped" and therefore eligible for RAW passthrough: an event we DO map but
# deliberately suppress must not leak out as a RAW event as well.
_RECOGNIZED_EVENT_TYPES = frozenset(
    {
        _FLOW_STARTED,
        _FLOW_FINISHED,
        _METHOD_STARTED,
        _METHOD_FINISHED,
        EventType.TEXT_MESSAGE_CHUNK,
        EventType.TOOL_CALL_CHUNK,
        EventType.CUSTOM,
        EventType.STATE_SNAPSHOT,
    }
)

# Source tag on emitted RAW events, so a consumer can tell CrewAI passthrough apart
# from other producers' raw channels.
_RAW_SOURCE = "crewai"

# One WARNING per process for the first RAW-passthrough loss, then DEBUG. An
# operator who turned RAW on is debugging something and would never see a
# DEBUG-only drop, but a per-event WARNING under sustained pressure is its own
# problem.
_RAW_LOSS_WARNED = False


def log_raw_loss(message: str, *args: Any) -> None:
    """Log a RAW-passthrough loss: WARNING the first time, DEBUG thereafter."""
    global _RAW_LOSS_WARNED  # pylint: disable=global-statement
    if _RAW_LOSS_WARNED:
        _LOGGER.debug(message, *args)
        return
    _RAW_LOSS_WARNED = True
    _LOGGER.warning(message, *args)


def is_recognized_event(event: Any) -> bool:
    """True when the translator maps this event, so RAW must not duplicate it."""
    return getattr(event, "type", None) in _RECOGNIZED_EVENT_TYPES


def raw_event_for(event: Any) -> RawEvent | None:
    """Wrap an unmapped crewai event as an AG-UI ``RAW`` event.

    Payload preference order:

    1. ``model_dump(mode="json")`` - every crewai event is a Pydantic model, and
       ``mode="json"`` resolves enums / datetimes for us.
    2. the event's own ``__dict__``, filtered to JSON-safe scalars and containers.

    A RAW passthrough must never be able to break the run, so a payload we cannot
    build at all yields ``None`` (the caller drops the mirror) rather than raising
    into the driver's event loop.
    """
    event_type = getattr(event, "type", None)
    if event_type is None:
        return None

    payload: Any = None
    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump(mode="json")
        except Exception as dump_exc:  # noqa: BLE001 - falls back below
            # Surface WHY the richer payload was unavailable; silence made a degraded
            # RAW payload indistinguishable from a complete one.
            _LOGGER.debug(
                "ag-ui-crewai RAW passthrough could not model_dump a %r event (%s); "
                "falling back to instance attributes",
                event_type,
                type(dump_exc).__name__,
            )
            payload = None

    if payload is None:
        source_dict = getattr(event, "__dict__", None)
        if isinstance(source_dict, dict):
            payload = {
                key: value
                for key, value in source_dict.items()
                if not key.startswith("_")
                and isinstance(value, (str, int, float, bool, type(None), list, dict))
            }

    if payload is None:
        return None
    if not isinstance(payload, dict):
        payload = {"value": payload}
    payload.setdefault("type", str(event_type))
    return RawEvent(type=EventType.RAW, event=payload, source=_RAW_SOURCE)
