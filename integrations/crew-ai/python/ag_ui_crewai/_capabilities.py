"""Runtime capability detection for the CrewAI AG-UI bridge.

crewai's public surface shifted across the 0.x -> 1.x boundary. Rather than
gate code paths on ``crewai.__version__`` (brittle — a version string is not a
feature probe, and users run forks / pre-releases), we RESOLVE each crewai
symbol we depend on here, trying the 1.x location first and falling back to the
0.x location. The probe runs exactly ONCE at import time and its results are
cached on the module-level ``CAPABILITIES`` object.

``crewai.__version__`` is used ONLY for human-facing warning text and the docs
capability table — never as a code-path gate.

Posture: "we support that feature; for this specific one you need crewai >= X."

This module is a LEAF: it imports only ``crewai`` / ``litellm`` and the stdlib,
so ``events`` / ``sdk`` / ``endpoint`` / ``crews`` can all import from it at
module-load time without a circular dependency (mirrors ``_env``).
"""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger(__name__)


def _crewai_version() -> str:
    try:
        import crewai

        return getattr(crewai, "__version__", "unknown")
    except Exception:  # pragma: no cover - crewai is a hard dependency
        return "unknown"


def _first_module(candidates: list[str]) -> tuple[Any, str | None]:
    """Import the first importable module from ``candidates``.

    Returns ``(module, dotted_name)`` for the first hit, or ``(None, None)`` if
    none import. Used to resolve a symbol home that moved between crewai
    releases without version-gating.
    """
    for name in candidates:
        try:
            return importlib.import_module(name), name
        except (ImportError, ModuleNotFoundError):
            # Only "module genuinely not here" is a soft miss (fall through to
            # the next candidate). A DIFFERENT error raised while importing an
            # existing module — e.g. a real bug inside ``crewai.events`` (a
            # SyntaxError, an AttributeError from a broken re-export, a failing
            # top-level side effect) — must NOT be swallowed: doing so
            # misreports a genuinely broken install as "install crewai>=1.0".
            # ``ModuleNotFoundError`` is an ``ImportError`` subclass; both are
            # listed for clarity.
            continue
    return None, None


def _resolve_attrs(module: Any, names: list[str]) -> dict[str, Any]:
    return {n: getattr(module, n, None) for n in names}


# --------------------------------------------------------------------------
# Event system resolution
# --------------------------------------------------------------------------
# crewai 1.0.0 DELETED ``crewai.utilities.events`` and re-homed the event bus,
# the flow/method lifecycle events, and the listener base at ``crewai.events``.
# ``BaseEvent`` is no longer re-exported at the package root — it lives at
# ``<events>.base_events.BaseEvent`` under either parent. We try the 1.x home
# first, then the 0.x home, so the bridge keeps working across the declared
# ``crewai>=1.0`` floor AND a 0.x install (belt-and-suspenders).
_EVENTS_MODULE, _EVENTS_MODULE_NAME = _first_module(
    ["crewai.events", "crewai.utilities.events"]
)

_LIFECYCLE_EVENT_NAMES = [
    "crewai_event_bus",
    "FlowStartedEvent",
    "FlowFinishedEvent",
    "MethodExecutionStartedEvent",
    "MethodExecutionFinishedEvent",
    "BaseEventListener",
]

if _EVENTS_MODULE is not None:
    _events_attrs = _resolve_attrs(_EVENTS_MODULE, _LIFECYCLE_EVENT_NAMES)
else:  # pragma: no cover - crewai without an events module is unsupported
    _events_attrs = dict.fromkeys(_LIFECYCLE_EVENT_NAMES, None)

crewai_event_bus = _events_attrs["crewai_event_bus"]
FlowStartedEvent = _events_attrs["FlowStartedEvent"]
FlowFinishedEvent = _events_attrs["FlowFinishedEvent"]
MethodExecutionStartedEvent = _events_attrs["MethodExecutionStartedEvent"]
MethodExecutionFinishedEvent = _events_attrs["MethodExecutionFinishedEvent"]
BaseEventListener = _events_attrs["BaseEventListener"]

# ``BaseEvent`` moved with the events package but is NOT re-exported at the
# root; it stays at ``<events pkg>.base_events.BaseEvent``. The
# ``base_event_listener`` submodule likewise stays under the resolved parent.
_BASE_EVENTS_MODULE, _ = _first_module(
    ["crewai.events.base_events", "crewai.utilities.events.base_events"]
)
BaseEvent = getattr(_BASE_EVENTS_MODULE, "BaseEvent", None) if _BASE_EVENTS_MODULE else None

# The event bus split its single ``_handlers`` mapping into ``_sync_handlers``
# / ``_async_handlers`` at 1.0.0 and now dispatches sync handlers on a
# ThreadPoolExecutor instead of inline on the caller's thread. Detect it so the
# endpoint enqueues thread-safely and the test harness snapshots the right
# attribute(s).
_event_bus_offthread = bool(crewai_event_bus is not None and hasattr(crewai_event_bus, "_sync_handlers"))
_event_bus_has_flush = bool(crewai_event_bus is not None and callable(getattr(crewai_event_bus, "flush", None)))


# --------------------------------------------------------------------------
# StreamFrame streaming contract resolution
# --------------------------------------------------------------------------
# crewai landed a public, ordered streaming envelope — ``StreamFrame`` and the
# ``AsyncStreamSession`` returned by ``Flow.astream()`` — in 1.6.0 (hardened in
# 1.15.2). It supersedes the event-bus-listener bridge: a scoped stream sink
# converts every emitted event into an ordered frame, and ``aclose()`` gives us
# real cancellation. We RESOLVE the symbol (never version-gate) and, at the call
# site, ALSO probe ``hasattr(flow, "astream")`` per-flow so test doubles that
# implement only ``kickoff_async`` transparently fall back to the legacy path.
#
# On crewai 1.0-1.5 (StreamFrame absent) the bridge falls back to the legacy
# bus-listener path with a one-time warning naming 1.6.
_STREAMING_TYPES_MODULE, _STREAMING_TYPES_MODULE_NAME = _first_module(
    ["crewai.types.streaming"]
)
StreamFrame = (
    getattr(_STREAMING_TYPES_MODULE, "StreamFrame", None)
    if _STREAMING_TYPES_MODULE is not None
    else None
)

# The scoped stream-sink API (``crewai.events.stream_context``) landed together
# with ``StreamFrame`` in 1.6. The bridge registers its OWN sink so the frame
# translator receives the RAW AG-UI / lifecycle event object (source + exact
# payload) rather than the ``to_serializable``-mangled ``frame.data`` snapshot.
# ``publish_stream_event`` invokes every sink
# synchronously on ``emit``, so a sink parked by ``event_id`` is guaranteed
# populated before the corresponding frame is dequeued.
_STREAM_CONTEXT_MODULE, _STREAM_CONTEXT_MODULE_NAME = _first_module(
    ["crewai.events.stream_context"]
)
add_stream_sink = (
    getattr(_STREAM_CONTEXT_MODULE, "add_stream_sink", None)
    if _STREAM_CONTEXT_MODULE is not None
    else None
)
reset_stream_sinks = (
    getattr(_STREAM_CONTEXT_MODULE, "reset_stream_sinks", None)
    if _STREAM_CONTEXT_MODULE is not None
    else None
)

# The StreamFrame path needs BOTH the frame type and the sink API. They ship
# together (1.6), but require both so a partial install falls back cleanly.
_stream_frame_available = (
    StreamFrame is not None
    and callable(add_stream_sink)
    and callable(reset_stream_sinks)
)


def flow_supports_stream_frames(flow: Any) -> bool:
    """Return True when ``flow`` can be driven via the StreamFrame contract.

    Two conditions, both required:

    * The installed crewai exposes ``StreamFrame`` (resolved once at import) —
      i.e. crewai >= 1.6. On 1.0-1.5 this is ``None`` and we fall back.
    * This SPECIFIC flow object exposes ``astream`` — real crewai ``Flow``
      instances do, but the test doubles in ``tests/test_task_cancellation.py``
      implement only ``kickoff_async`` and MUST keep taking the legacy path so
      their cancellation / timeout coverage is unaffected.
    """
    return _stream_frame_available and hasattr(flow, "astream")


# --------------------------------------------------------------------------
# crew-chat helper resolution
# --------------------------------------------------------------------------
# The five crew-chat helpers moved from ``crewai.cli.crew_chat`` to
# ``crewai.utilities.crew_chat`` at crewai 1.15.0 (``crewai.cli`` is now a
# deprecation shim that no longer re-exports them). Try the new home first,
# fall back to the old one, so the crew-serving path works across the whole
# ``crewai>=1.0`` floor.
_CREW_CHAT_HELPER_NAMES = [
    "initialize_chat_llm",
    "generate_crew_chat_inputs",
    "generate_crew_tool_schema",
    "build_system_message",
    "create_tool_function",
]
_CREW_CHAT_MODULE, _CREW_CHAT_MODULE_NAME = _first_module(
    ["crewai.utilities.crew_chat", "crewai.cli.crew_chat"]
)
if _CREW_CHAT_MODULE is not None:
    _crew_chat_attrs = _resolve_attrs(_CREW_CHAT_MODULE, _CREW_CHAT_HELPER_NAMES)
else:
    _crew_chat_attrs = dict.fromkeys(_CREW_CHAT_HELPER_NAMES, None)

initialize_chat_llm = _crew_chat_attrs["initialize_chat_llm"]
generate_crew_chat_inputs = _crew_chat_attrs["generate_crew_chat_inputs"]
generate_crew_tool_schema = _crew_chat_attrs["generate_crew_tool_schema"]
build_system_message = _crew_chat_attrs["build_system_message"]
create_tool_function = _crew_chat_attrs["create_tool_function"]
_crew_chat_available = all(v is not None for v in _crew_chat_attrs.values())


# --------------------------------------------------------------------------
# litellm availability
# --------------------------------------------------------------------------
# crewai moved litellm to the optional ``crewai[litellm]`` extra at 1.0.0. We
# declare ``litellm`` as a DIRECT dependency of ag-ui-crewai (we import
# ``acompletion`` and ``litellm.types`` ourselves) so it resolves regardless of
# crewai extras. Probe it anyway for the capability table / a clear warning.
try:
    import litellm  # noqa: F401

    _litellm_available = True
except Exception:  # pragma: no cover - litellm is a declared direct dep
    _litellm_available = False


# --------------------------------------------------------------------------
# Checkpointing resolution
# --------------------------------------------------------------------------
# crewai's checkpointing pieces landed in different releases, so each is
# resolved/probed independently (never gated on ``__version__``) and its
# enabling version named in the warning text. ``from_checkpoint`` (1.13)
# predates ``CheckpointConfig`` (1.14), so the two are probed separately: on
# 1.13.x the kwarg exists but no config can be built, and the bridge stays on
# the no-checkpoint path.
CHECKPOINT_ENABLING_VERSIONS: dict[str, str] = {
    "from_checkpoint": "1.13.0",
    "checkpoint_config": "1.14.0",
    "fork": "1.14.2",
    "checkpoint_events": "1.14.3",
    "restore_from_state_id": "1.14.5",
}

_CREWAI_MODULE, _ = _first_module(["crewai"])
_Flow = getattr(_CREWAI_MODULE, "Flow", None) if _CREWAI_MODULE else None
_Crew = getattr(_CREWAI_MODULE, "Crew", None) if _CREWAI_MODULE else None


def _kwarg_in_signature(func: Any, name: str) -> bool:
    """True when ``func`` declares a parameter ``name`` (or accepts ``**kwargs``).

    Used to probe whether a crewai release grew a given keyword argument
    without gating on the version string. A ``**kwargs`` catch-all counts as
    "accepts it": passing an unknown kwarg through ``**kwargs`` is safe.
    """
    if func is None:
        return False
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - C builtins etc.
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# ``CheckpointConfig`` is re-exported at the crewai root (1.14+); its canonical
# home is ``crewai.state.checkpoint_config``. Try the root first, then the
# module, so a partial / future re-org still resolves.
CheckpointConfig = getattr(_CREWAI_MODULE, "CheckpointConfig", None) if _CREWAI_MODULE else None
_CKPT_STATE_MODULE, _CKPT_STATE_MODULE_NAME = _first_module(["crewai.state"])
if CheckpointConfig is None and _CKPT_STATE_MODULE is not None:
    CheckpointConfig = getattr(_CKPT_STATE_MODULE, "CheckpointConfig", None)

# ``JsonProvider`` / ``SqliteProvider`` live on ``crewai.state`` (NOT the crewai
# root, verified on the 1.15.7 wheel).
JsonProvider = getattr(_CKPT_STATE_MODULE, "JsonProvider", None) if _CKPT_STATE_MODULE else None
SqliteProvider = getattr(_CKPT_STATE_MODULE, "SqliteProvider", None) if _CKPT_STATE_MODULE else None

# The Checkpoint*Event lifecycle types live at
# ``crewai.events.types.checkpoint_events`` (not re-exported at the
# ``crewai.events`` root on 1.15.x). Resolved for callers that surface them;
# the persistence wiring here does not depend on them.
_CKPT_EVENTS_MODULE, _CKPT_EVENTS_MODULE_NAME = _first_module(
    ["crewai.events.types.checkpoint_events"]
)
_checkpoint_events_available = _CKPT_EVENTS_MODULE is not None and (
    getattr(_CKPT_EVENTS_MODULE, "CheckpointCompletedEvent", None) is not None
)

# ``from_checkpoint`` / ``restore_from_state_id`` are probed on ``Flow`` (the
# bridge only checkpoints flows; the crew endpoint wraps its crew in a
# ``ChatWithCrewFlow``). These are the CLASS-level probes for the capability
# table / warnings; the per-flow guard below re-probes the SPECIFIC instance so
# test doubles that implement only ``kickoff_async(self, inputs=None)`` stay on
# the no-checkpoint path.
_flow_from_checkpoint_supported = _kwarg_in_signature(
    getattr(_Flow, "kickoff_async", None), "from_checkpoint"
)
_flow_restore_from_state_id_supported = _kwarg_in_signature(
    getattr(_Flow, "kickoff_async", None), "restore_from_state_id"
)
_checkpoint_fork_supported = callable(getattr(_Flow, "fork", None)) or callable(
    getattr(_Crew, "fork", None)
)
# Checkpointing needs a config type AND at least one provider to build one. The
# ``from_checkpoint`` kwarg alone (crewai 1.13) is inert without them.
_checkpoint_config_available = CheckpointConfig is not None and (
    JsonProvider is not None or SqliteProvider is not None
)
# The full persistence path is usable when we can both build a config and pass
# it: i.e. the config type, a provider, and the kwarg are all present.
_checkpointing_available = _checkpoint_config_available and _flow_from_checkpoint_supported


def flow_supports_checkpointing(flow: Any) -> bool:
    """Return True when THIS flow can be checkpointed via ``from_checkpoint``.

    Two conditions, both required (mirrors ``flow_supports_stream_frames``):

    * the installed crewai can build a ``CheckpointConfig`` and exposes the
      ``from_checkpoint`` kwarg (crewai >= 1.14 for both), and
    * this SPECIFIC flow object exposes a driving method (``astream`` or
      ``kickoff_async``) whose signature actually accepts ``from_checkpoint``.

    The per-flow re-probe is what keeps the cancellation test doubles in
    ``tests/test_task_cancellation.py`` (which implement only
    ``kickoff_async(self, inputs=None)``) on the no-checkpoint path, so their
    27 cancellation / timeout tests are unaffected.
    """
    if not _checkpointing_available:
        return False
    for method_name in ("astream", "kickoff_async"):
        if _kwarg_in_signature(getattr(flow, method_name, None), "from_checkpoint"):
            return True
    return False


def supported_checkpoint_kwargs(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter ``kwargs`` to those the bound ``method`` actually declares.

    The last line of defence at the call site: even after
    ``flow_supports_checkpointing`` gates the build, the frame path calls
    ``astream`` and the legacy path calls ``kickoff_async`` (different methods).
    Filtering per-method means a flow that grew one kwarg but not the other (or
    a test double that grew neither) degrades to a no-op instead of raising
    ``TypeError: unexpected keyword argument``.
    """
    if not kwargs:
        return {}
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):  # pragma: no cover - C builtins etc.
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


# --------------------------------------------------------------------------
# Async human-feedback (HITL) resolution
# --------------------------------------------------------------------------
# crewai's async HITL landed at 1.8: a flow method wrapped with
# ``@human_feedback`` whose provider RAISES ``HumanFeedbackPending`` pauses the
# run; the framework persists the pending state and ``Flow.from_pending(flow_id)``
# + ``flow.resume_async(feedback)`` resume it. The pause / feedback lifecycle
# events live on ``crewai.events.types.flow_events`` and are NOT re-exported at
# the ``crewai.events`` root (verified on the 1.15.7 wheel), so resolve them
# there first, with the root as a fallback for a future re-export.
_FLOW_EVENTS_MODULE, _FLOW_EVENTS_MODULE_NAME = _first_module(
    ["crewai.events.types.flow_events", "crewai.events"]
)
_HITL_EVENT_NAMES = [
    "HumanFeedbackRequestedEvent",
    "HumanFeedbackReceivedEvent",
    "FlowPausedEvent",
    "MethodExecutionPausedEvent",
]
if _FLOW_EVENTS_MODULE is not None:
    _hitl_event_attrs = _resolve_attrs(_FLOW_EVENTS_MODULE, _HITL_EVENT_NAMES)
else:  # pragma: no cover - crewai without a flow-events module is pre-HITL
    _hitl_event_attrs = dict.fromkeys(_HITL_EVENT_NAMES, None)
# Fall back to the crewai.events root for any name the primary module missed.
_EVENTS_ROOT_MODULE, _ = _first_module(["crewai.events"])
if _EVENTS_ROOT_MODULE is not None:
    for _name, _value in list(_hitl_event_attrs.items()):
        if _value is None:
            _hitl_event_attrs[_name] = getattr(_EVENTS_ROOT_MODULE, _name, None)

HumanFeedbackRequestedEvent = _hitl_event_attrs["HumanFeedbackRequestedEvent"]
HumanFeedbackReceivedEvent = _hitl_event_attrs["HumanFeedbackReceivedEvent"]
FlowPausedEvent = _hitl_event_attrs["FlowPausedEvent"]
MethodExecutionPausedEvent = _hitl_event_attrs["MethodExecutionPausedEvent"]

# The pause signal + provider protocol live on ``crewai.flow``.
_FLOW_PKG_MODULE, _ = _first_module(["crewai.flow"])
HumanFeedbackPending = (
    getattr(_FLOW_PKG_MODULE, "HumanFeedbackPending", None) if _FLOW_PKG_MODULE else None
)
HumanFeedbackProvider = (
    getattr(_FLOW_PKG_MODULE, "HumanFeedbackProvider", None) if _FLOW_PKG_MODULE else None
)

# Resume API, probed on the resolved Flow class (``from_pending`` is a
# classmethod, ``resume_async`` an instance coroutine).
_flow_from_pending_supported = callable(getattr(_Flow, "from_pending", None))
_flow_resume_async_supported = callable(getattr(_Flow, "resume_async", None))


def _model_has_field(model: Any, field_name: str) -> bool:
    """True when a Pydantic ``model`` declares ``field_name``."""
    fields = getattr(model, "model_fields", None)
    return bool(fields) and field_name in fields


# ``HumanFeedbackRequestedEvent.request_id`` (crewai 1.12.2+) is the stable,
# non-synthesizable id the bridge maps onto ``AGUIInterrupt.id``. Probe for the
# field rather than the version; below it there is no stable id and HITL is not
# advertised.
_human_feedback_request_id_supported = (
    HumanFeedbackRequestedEvent is not None
    and _model_has_field(HumanFeedbackRequestedEvent, "request_id")
)

# Two levels, so a pause that surfaces as an interrupt is never stranded by a
# too-strict resume gate:
#
# * ``_human_feedback_resume_available`` gates the pause / resume LIFECYCLE. It
#   needs the pause signal, the resume classmethod + coroutine, and the
#   StreamFrame transport (async HITL >=1.8 always ships alongside StreamFrame
#   >=1.6, and the bridge only drives the lifecycle on the frame path). It does
#   NOT require a stable request id: the interrupt id falls back to the flow id
#   (== thread_id), which resume keys by, so 1.8-1.12.1 pauses still resume.
# * ``_human_feedback_available`` is the ADVERTISED capability (stable interrupt
#   ids). It adds the request-event class and its ``request_id`` field (1.12.2+).
#   Below that, the lifecycle still works with flow-id ids and a warning.
_human_feedback_resume_available = (
    HumanFeedbackPending is not None
    and FlowPausedEvent is not None
    and _flow_from_pending_supported
    and _flow_resume_async_supported
    and _stream_frame_available
)
_human_feedback_available = (
    _human_feedback_resume_available
    and HumanFeedbackRequestedEvent is not None
    and _human_feedback_request_id_supported
)

# Named enabling versions for warning text only (never a code-path gate).
HITL_ENABLING_VERSIONS: dict[str, str] = {
    "human_feedback": "1.8.0",
    "request_id": "1.12.2",
    "stream_frame": "1.6.0",
}


def flow_supports_human_feedback(flow: Any) -> bool:
    """Return True when THIS flow can pause / resume via async human feedback.

    Mirrors ``flow_supports_stream_frames`` / ``flow_supports_checkpointing``:
    the installed crewai must expose the async-HITL API (probed above) AND this
    specific flow must expose the resume coroutine and the frame transport, so
    the ``kickoff_async``-only test doubles in ``tests/test_task_cancellation.py``
    stay off the HITL path.
    """
    if not _human_feedback_resume_available:
        return False
    return callable(getattr(flow, "resume_async", None)) and hasattr(flow, "astream")


@dataclass(frozen=True)
class _Capabilities:
    """Cached, immutable snapshot of the detected crewai capabilities.

    Instantiated exactly once (``CAPABILITIES`` below). ``crewai_version`` is
    for warning text / the docs table only — code paths key off the boolean
    probes and the resolved symbols above, never the version string.
    """

    crewai_version: str
    events_module: str | None
    has_event_bus: bool
    event_bus_offthread: bool
    event_bus_has_flush: bool
    crew_chat_module: str | None
    crew_chat_available: bool
    litellm_available: bool
    stream_frame_available: bool = False
    # Checkpointing: informational; the wiring keys off the resolved
    # symbols / ``flow_supports_checkpointing`` per-flow probe, not these fields.
    checkpoint_config_available: bool = False
    checkpointing_available: bool = False
    flow_from_checkpoint_supported: bool = False
    flow_restore_from_state_id_supported: bool = False
    checkpoint_fork_supported: bool = False
    checkpoint_events_available: bool = False
    checkpoint_state_module: str | None = None
    # Async human-feedback (HITL): informational; the wiring keys off the
    # resolved symbols / ``flow_supports_human_feedback`` per-flow probe.
    flow_events_module: str | None = None
    human_feedback_available: bool = False
    human_feedback_resume_available: bool = False
    human_feedback_request_id_supported: bool = False
    missing: tuple[str, ...] = field(default_factory=tuple)

    def warn_on_gaps(self) -> None:
        """Emit one WARNING per missing capability, naming the fix.

        Kept idempotent-friendly (call once at import). Each message names the
        crewai version / extra that unlocks the missing capability so operators
        get an actionable signal instead of a deep ImportError later.
        """
        if not self.has_event_bus:
            _LOGGER.warning(
                "ag-ui-crewai could not resolve the crewai event bus "
                "(tried crewai.events, crewai.utilities.events) on crewai %s. "
                "The FastAPI bridge needs it; install crewai>=1.0.",
                self.crewai_version,
            )
        if not self.crew_chat_available:
            _LOGGER.warning(
                "ag-ui-crewai could not resolve the crew-chat helpers (tried "
                "crewai.utilities.crew_chat, crewai.cli.crew_chat) on crewai "
                "%s. The crew-serving endpoint "
                "(add_crewai_crew_fastapi_endpoint) requires them; the flow "
                "endpoint is unaffected.",
                self.crewai_version,
            )
        if not self.litellm_available:
            _LOGGER.warning(
                "ag-ui-crewai could not import litellm on crewai %s. Streaming "
                "completions require it; install litellm (a direct dependency) "
                "or crewai[litellm].",
                self.crewai_version,
            )
        if not self.stream_frame_available:
            # NOT a hard gap — the legacy bus-listener path still works. Emit
            # an INFO-level note (not a WARNING) so operators on 1.0-1.5 know
            # the richer StreamFrame transport unlocks at crewai>=1.6.
            _LOGGER.info(
                "ag-ui-crewai: crewai %s does not expose the StreamFrame "
                "streaming contract (crewai.types.streaming.StreamFrame); the "
                "FastAPI bridge will use the legacy event-bus-listener path. "
                "Upgrade to crewai>=1.6 for the ordered StreamFrame transport.",
                self.crewai_version,
            )
        if not self.human_feedback_resume_available:
            # NOT a hard gap; chat / tool-based HITL is unaffected. Emit an
            # INFO note so operators know async interrupt (pause / resume) needs
            # the async-HITL API + the StreamFrame transport.
            _LOGGER.info(
                "ag-ui-crewai: crewai %s does not expose the async human-feedback "
                "interrupt API the bridge needs (async @human_feedback pause, "
                "Flow.from_pending/resume_async, StreamFrame); interrupt/resume "
                "is disabled. Upgrade to crewai>=%s for AG-UI interrupts.",
                self.crewai_version,
                HITL_ENABLING_VERSIONS["human_feedback"],
            )
        elif not self.human_feedback_request_id_supported:
            # Lifecycle works, but without a stable per-request id the interrupt
            # id falls back to the flow id (== thread_id). Fine for one pending
            # per thread; upgrade for a stable id across multiple pauses.
            _LOGGER.info(
                "ag-ui-crewai: crewai %s supports async human-feedback but not "
                "HumanFeedbackRequestedEvent.request_id; interrupt ids fall back "
                "to the flow id. Upgrade to crewai>=%s for stable request ids.",
                self.crewai_version,
                HITL_ENABLING_VERSIONS["request_id"],
            )


def _detect() -> _Capabilities:
    missing: list[str] = []
    if crewai_event_bus is None:
        missing.append("event_bus")
    if not _crew_chat_available:
        missing.append("crew_chat")
    if not _litellm_available:
        missing.append("litellm")
    caps = _Capabilities(
        crewai_version=_crewai_version(),
        events_module=_EVENTS_MODULE_NAME,
        has_event_bus=crewai_event_bus is not None,
        event_bus_offthread=_event_bus_offthread,
        event_bus_has_flush=_event_bus_has_flush,
        crew_chat_module=_CREW_CHAT_MODULE_NAME,
        crew_chat_available=_crew_chat_available,
        litellm_available=_litellm_available,
        stream_frame_available=_stream_frame_available,
        checkpoint_config_available=_checkpoint_config_available,
        checkpointing_available=_checkpointing_available,
        flow_from_checkpoint_supported=_flow_from_checkpoint_supported,
        flow_restore_from_state_id_supported=_flow_restore_from_state_id_supported,
        checkpoint_fork_supported=_checkpoint_fork_supported,
        checkpoint_events_available=_checkpoint_events_available,
        checkpoint_state_module=_CKPT_STATE_MODULE_NAME,
        flow_events_module=_FLOW_EVENTS_MODULE_NAME,
        human_feedback_available=_human_feedback_available,
        human_feedback_resume_available=_human_feedback_resume_available,
        human_feedback_request_id_supported=_human_feedback_request_id_supported,
        missing=tuple(missing),
    )
    caps.warn_on_gaps()
    return caps


# Run the probe ONCE at import time and cache the result.
CAPABILITIES = _detect()
