"""Translate crewai stream events into AG-UI wire events (CPK-7719).

crewai's public streaming contract (``StreamFrame`` / ``AsyncStreamSession``,
landed 1.6.0) emits one ordered frame per event a flow raises. The frame gives
us ORDERING and the identity of the emitting flow; the RAW event object (parked
by our scoped stream sink, keyed by ``event.event_id == StreamFrame.id``) gives
us the EXACT payload. This module is the SINGLE translation seam that maps the
raw events the bridge cares about onto AG-UI events.

Why raw events, not ``frame.data`` (CPK-7719 review blocker 1):
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

OUTER-flow filtering (CPK-7719 review blockers 2 + 3): a ``crew.kickoff`` inside
a flow method drives a NESTED flow whose own lifecycle/method events leak onto
the same scoped sink. The driver filters those out by source identity BEFORE
they reach this translator (only events whose ``source is flow_copy`` are parked
in the lookup buffer), so the translator never sees nested frames. This mirrors
the legacy listener's ``source is flow_copy`` gate and needs no depth counter.

SWAPPABLE EMISSION SHAPE (cross-lane constraint, CPK-7719): CrewAI is currently
the only integration emitting TEXT_MESSAGE_CHUNK / TOOL_CALL_CHUNK (chunks)
rather than the START/CONTENT/END triples the six other integrations emit. That
final choice belongs to a Parity-lane ticket, not this migration. The translator
therefore routes LLM text / LLM-tool-call emission through a single
``emission_shape`` strategy that DEFAULTS to ``"chunks"`` so this migration is
byte-for-byte behavior-preserving on that channel. The ``"triples"`` strategy is
a deliberate NotImplementedError placeholder — the Parity ticket owns wiring it
up and flipping the default.

MCP EVENTS (PNI-130) are the ONE exception to "chunks-only": crewai's discrete
MCP tool executions (name + full args + result arrive together, not streamed)
map to canonical ``TOOL_CALL_START/ARGS/END/RESULT`` triples via the shared
``mcp.translate_mcp_event`` seam, and MCP lifecycle events map to ``CUSTOM``.
This is independent of the ``emission_shape`` strategy above (which governs only
the streaming LLM text / tool-call channel); see the wire-shape note in
``mcp.py`` for why discrete MCP calls use triples regardless of PNI-136.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
)
from ag_ui.core.events import (
    TextMessageChunkEvent,
    ToolCallChunkEvent,
    StepStartedEvent,
    StepFinishedEvent,
    MessagesSnapshotEvent,
    StateSnapshotEvent,
    CustomEvent,
)

from .sdk import litellm_messages_to_ag_ui_messages
from .mcp import is_mcp_event, translate_mcp_event

# crewai lifecycle-event ``type`` strings. Pinned as constants so a rename
# upstream surfaces here rather than silently producing no wire events.
_FLOW_STARTED = "flow_started"
_FLOW_FINISHED = "flow_finished"
_METHOD_STARTED = "method_execution_started"
_METHOD_FINISHED = "method_execution_finished"

_SUPPORTED_EMISSION_SHAPES = frozenset({"chunks", "triples"})


class StreamFrameTranslator:
    """Stateless-ish mapper from a RAW crewai/bridge event to AG-UI events.

    One instance per run (it carries the run correlation ids and a
    ``state_provider`` closure over the live flow copy). ``translate`` returns
    zero or more AG-UI events for a single raw event; unrecognised event types
    return ``[]`` (behavior-preserving: the legacy listener produced nothing for
    crewai's native llm / tools / messages channels or internal events).

    The driver is responsible for OUTER-flow filtering (only forwarding events
    whose ``source is flow_copy``); the translator assumes every event it sees
    belongs to the outer run.
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
        # Run-lifecycle idempotency (CPK-7719). A single AG-UI HTTP run must
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
            return [
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=self._thread_id,
                    run_id=self._run_id,
                )
            ]
        if event_type == _METHOD_STARTED:
            return [
                StepStartedEvent(
                    type=EventType.STEP_STARTED,
                    step_name=getattr(event, "method_name", None),
                )
            ]
        if event_type == _METHOD_FINISHED:
            return self._method_finished_events(event)

        # PNI-130: crewai's first-class MCP events (crewai >= 1.4). Emitted with
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
        """Belt-and-braces terminal (CPK-7719 review blocker 3).

        Called once when the frame stream exhausts cleanly. If the run opened
        (RUN_STARTED) but no outer ``flow_finished`` closed it — e.g. the outer
        method caught a nested-flow error and the stream just ended — emit the
        missing RUN_FINISHED so the client NEVER sees a run that never ends. The
        errored path terminates via RUN_ERROR instead and must not call this.
        """
        if self._run_started_emitted and not self._run_finished_emitted:
            self._run_finished_emitted = True
            return [
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=self._thread_id,
                    run_id=self._run_id,
                )
            ]
        return []

    # -- lifecycle --------------------------------------------------------

    def _method_finished_events(self, event: Any) -> list[Any]:
        """MESSAGES_SNAPSHOT + STATE_SNAPSHOT + STEP_FINISHED, in that order.

        Reads the LIVE flow state via ``state_provider`` — exactly what the
        legacy listener did with ``source.state`` — rather than the serialized
        ``event`` payload, so message / state shapes round-trip through the same
        ``litellm_messages_to_ag_ui_messages`` path as before.
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
        method_name = getattr(event, "method_name", None)
        return [
            MessagesSnapshotEvent(
                type=EventType.MESSAGES_SNAPSHOT,
                messages=messages,
            ),
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=snapshot,
            ),
            StepFinishedEvent(
                type=EventType.STEP_FINISHED,
                step_name=method_name,
            ),
        ]

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
        # TODO(CPK Parity lane): emit TEXT_MESSAGE_START / _CONTENT / _END
        # triples here to match the other six integrations. That cross-lane
        # decision is owned by the Parity ticket; do NOT flip the default in
        # this migration (it must stay behavior-preserving on the wire).
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
        # TODO(CPK Parity lane): TOOL_CALL_START / _ARGS / _END triples — see
        # the note in ``_text_events``.
        raise NotImplementedError(
            "emission_shape='triples' is a Parity-lane placeholder; "
            "the StreamFrame migration ships the behavior-preserving "
            "'chunks' shape only."
        )
