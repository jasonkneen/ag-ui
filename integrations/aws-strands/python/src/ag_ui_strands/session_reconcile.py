"""Reconcile frontend (proxy) tool results into a Strands ``SessionManager``.

Frontend tools are executed on the client, so server-side the proxy returns a
placeholder ``toolResult`` (``"Forwarded to client"``). The real result only
arrives on the next run inside ``RunAgentInput.messages``, keyed by the client's
wire ``tool_call_id`` — which differs from the native ``toolUseId`` Strands
persisted. The adapter records that wire->native mapping durably on the agent's
session state (see ``AG_UI_WIRE_MAP_STATE_KEY``), so this module can find the
persisted placeholder by native id and overwrite it with the real result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Protocol

from .client_proxy_tool import PROXY_RESULT_PLACEHOLDER

# Key under which the adapter stores the ``{wire_tool_call_id: native_toolUseId}``
# map on the Strands agent's session state. Namespaced to avoid clashing with
# user-managed state keys.
AG_UI_WIRE_MAP_STATE_KEY = "__ag_ui_wire_to_native__"

# Key under which the adapter stores every ``toolUseId`` tool call metadata
# (name, args, input, strands_tool_id, is_frontend) on the Strands agent's
# session state.
# On a native-interrupt RESUME run Strands does not re-invoke the model for the
# interrupted tool, so no ``current_tool_use`` events fire and the in-run
# ``tool_calls_seen`` dict is empty when the ``toolResult`` arrives. Reading
# from this durable map at that point restores ``tool_name`` (and thus every
# ``tool_behaviors`` gate + the frontend-placeholder skip) for the resumed
# tool. Namespaced to avoid clashing with user-managed state keys.
AG_UI_TOOL_CALL_MAP_STATE_KEY = "__ag_ui_tool_call_map__"

# Strict adapter-owned persistence for accepted frontend results whose exact
# proxy invocation is still parked behind a ``BeforeToolCallEvent`` interrupt.
AG_UI_PENDING_PROXY_RESULTS_STATE_KEY = "__ag_ui_pending_proxy_results__"


@dataclass(frozen=True)
class _FrontendToolResult:
    """Client result data that must stay coupled during reconciliation."""

    content: str
    status: Literal["success", "error"]
    error: str | None = None

    @property
    def is_void(self) -> bool:
        """Return whether this is a successful result with no meaningful content."""
        return self.status == "success" and not self.content.strip()

    @property
    def provider_safe_content(self) -> str:
        """Return model-visible text, retaining a failed result's diagnostic."""
        if self.content.strip() or self.status != "error":
            return self.content
        diagnostic = (self.error or "").strip()
        if not diagnostic:
            return self.content
        try:
            diagnostic.encode("utf-8")
        except UnicodeEncodeError:
            return diagnostic.encode(
                "utf-8", errors="backslashreplace"
            ).decode("utf-8")
        return diagnostic


@dataclass(frozen=True)
class PendingProxyResult:
    """Raw client result durably bound to one exact emitted wire call."""

    wire_tool_call_id: str
    content: str
    status: Literal["success", "error"]
    error: str | None = None

    @property
    def frontend_result(self) -> _FrontendToolResult:
        return _FrontendToolResult(
            content=self.content,
            status=self.status,
            error=self.error,
        )


def _is_utf8_string(value: Any, *, nonempty: bool = False) -> bool:
    if type(value) is not str or (nonempty and not value):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def decode_pending_proxy_results(raw: Any) -> dict[str, PendingProxyResult]:
    """Decode the strict version-1 pending-result envelope, failing closed."""
    if (
        type(raw) is not dict
        or set(raw) != {"version", "records"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or type(raw["records"]) is not dict
    ):
        raise ValueError("malformed pending proxy result state")

    decoded: dict[str, PendingProxyResult] = {}
    wire_ids: set[str] = set()
    for native_id, record in raw["records"].items():
        if (
            not _is_utf8_string(native_id, nonempty=True)
            or type(record) is not dict
            or set(record)
            != {"wire_tool_call_id", "content", "status", "error"}
            or not _is_utf8_string(record["wire_tool_call_id"], nonempty=True)
            or not _is_utf8_string(record["content"])
            or type(record["status"]) is not str
            or record["status"] not in ("success", "error")
            or (
                record["error"] is not None
                and not _is_utf8_string(record["error"])
            )
            or record["wire_tool_call_id"] in wire_ids
        ):
            raise ValueError("malformed pending proxy result state")
        wire_ids.add(record["wire_tool_call_id"])
        decoded[native_id] = PendingProxyResult(**record)
    return decoded


def encode_pending_proxy_results(
    records: Mapping[str, PendingProxyResult],
) -> dict[str, Any]:
    """Encode pending records through the strict decoder's validation."""
    raw = {
        "version": 1,
        "records": {
            native_id: {
                "wire_tool_call_id": record.wire_tool_call_id,
                "content": record.content,
                "status": record.status,
                "error": record.error,
            }
            for native_id, record in records.items()
        },
    }
    decode_pending_proxy_results(raw)
    return raw


def merge_pending_proxy_results(
    accepted: Mapping[str, PendingProxyResult],
    incoming: Mapping[str, PendingProxyResult],
) -> dict[str, PendingProxyResult]:
    """Return an idempotent merge, rejecting every native or wire conflict."""
    merged = dict(accepted)
    native_by_wire = {
        record.wire_tool_call_id: native_id
        for native_id, record in accepted.items()
    }
    for native_id, record in incoming.items():
        existing = merged.get(native_id)
        if existing is not None and existing != record:
            raise ValueError("pending proxy result conflict")
        wire_native = native_by_wire.get(record.wire_tool_call_id)
        if wire_native is not None and wire_native != native_id:
            raise ValueError("pending proxy result conflict")
        merged[native_id] = record
        native_by_wire[record.wire_tool_call_id] = native_id
    return merged


class _SessionRepository(Protocol):
    def list_messages(self, session_id: str, agent_id: str) -> Iterable[Any]: ...

    def update_message(
        self, session_id: str, agent_id: str, session_message: Any
    ) -> None: ...


class _RepositoryReconciliationSessionManager(Protocol):
    """Structural capability required to rewrite persisted tool results."""

    @property
    def session_id(self) -> str: ...

    @property
    def session_repository(self) -> _SessionRepository: ...


def _supports_repository_reconciliation(session_manager: Any) -> bool:
    """Return whether a session manager exposes the repository rewrite API."""
    if session_manager is None:
        return False
    try:
        session_id = session_manager.session_id
        repository = session_manager.session_repository
    except (AttributeError, TypeError):
        # Capability absence is expected for the public SessionManager ABC;
        # mixed-state callers convert False into a structured RUN_ERROR.
        return False
    return (
        isinstance(session_id, str)
        and callable(getattr(repository, "list_messages", None))
        and callable(getattr(repository, "update_message", None))
    )


class ActiveInterruptReconciliationError(RuntimeError):
    """An activated interrupt's parked tool results could not be reconciled."""

    def __init__(self, affected_native_ids: Iterable[str]) -> None:
        self.affected_native_ids: frozenset[str] = frozenset(affected_native_ids)
        super().__init__("Active interrupt tool result reconciliation failed")


def resolve_native_ids(
    wire_to_native: Mapping[str, str],
    frontend_results: Mapping[str, _FrontendToolResult],
) -> dict[str, _FrontendToolResult]:
    """Map client frontend results to Strands-native ``toolUseId``s.

    Frontend tools are emitted under a fresh wire ``tool_call_id`` that differs
    from the native ``toolUseId`` Strands persists. ``wire_to_native`` is the
    durable map recorded at emission on the agent's session state and restored
    on a continuation run (works cross-process and for delta-only payloads,
    since it is keyed on the wire id the client always sends). Results whose
    wire id is not in the map are dropped — the caller then degrades to the
    legacy path rather than guessing.

    Args:
        wire_to_native: Map of wire ``tool_call_id`` -> native ``toolUseId``.
        frontend_results: Map of wire ``tool_call_id`` -> typed client result.

    Returns:
        Map of native ``toolUseId`` -> typed client result (unresolvable dropped).
    """
    resolved: dict[str, _FrontendToolResult] = {}
    for wire_id, result in frontend_results.items():
        native = wire_to_native.get(wire_id)
        if native is not None:
            resolved[native] = result
    return resolved


def reconcile_frontend_tool_results(
    session_manager: _RepositoryReconciliationSessionManager,
    agent: Any,
    pending_results: Mapping[str, _FrontendToolResult],
) -> set[str]:
    """Overwrite persisted placeholder ``toolResult`` blocks with real results.

    ``pending_results`` MUST be keyed by the Strands-native ``toolUseId`` (use
    :func:`resolve_native_ids` to translate client wire ids first).

    Args:
        session_manager: A Strands ``RepositorySessionManager`` (exposes
            ``session_id`` and ``session_repository``).
        agent: The Strands agent (exposes ``agent_id``).
        pending_results: Map of native ``toolUseId`` -> typed client result.

    Returns:
        The set of ``toolUseId``s whose placeholder was corrected (in the store
        and/or the agent's in-memory history).
    """
    session_id = session_manager.session_id
    agent_id = agent.agent_id
    repository = session_manager.session_repository

    corrected: set[str] = set()
    for session_message in repository.list_messages(session_id, agent_id):
        changed = _correct_message(session_message.message, pending_results)
        if changed:
            repository.update_message(session_id, agent_id, session_message)
            corrected |= changed

    # Correct the agent's live in-memory history too, so a same-process
    # continuation run (and ``stream_async(None)``) sees the real result
    # rather than the placeholder.
    for message in getattr(agent, "messages", None) or []:
        corrected |= _correct_message(message, pending_results)

    # Correct the agent's live interrupt state too. Once activation is known,
    # a failure to access or correct its parked tool results must stop the
    # resume before Strands clears the context. Repository and live-message
    # failures remain outside this typed boundary and retain their caller's
    # generic fallback behavior.
    interrupt_state = getattr(agent, "_interrupt_state", None)
    if interrupt_state is not None and getattr(interrupt_state, "activated", False):
        try:
            tool_results = interrupt_state.context.get("tool_results")
            if tool_results:
                corrected |= _correct_all_tools(tool_results, pending_results)
        except Exception as e:
            raise ActiveInterruptReconciliationError(pending_results) from e

    return corrected


def has_placeholder_results(messages: Iterable[Any], only_ids: Any = None) -> bool:
    """Return True if a matching ``toolResult`` is still the proxy stub.

    Used to gate the continuation stream: it is only safe to replay the native
    history to the model (``stream_async(None)``) when no relevant ``"Forwarded
    to client"`` placeholder remains to be fed to it.

    Args:
        messages: The native Strands history to scan.
        only_ids: If given, restrict the scan to ``toolResult`` blocks whose
            ``toolUseId`` is in this set — so stale placeholders from prior
            turns (e.g. intentionally-uncorrected void calls) don't count.
    """
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            tool_result = block.get("toolResult")
            if not isinstance(tool_result, dict):
                continue
            if only_ids is not None and tool_result.get("toolUseId") not in only_ids:
                continue
            if _is_placeholder(tool_result.get("content")):
                return True
    return False


def has_active_proxy_placeholder(
    agent: Any, tool_call_meta: Mapping[str, Any]
) -> bool:
    """Return whether an active interrupt parks an exact, proven proxy result."""
    return bool(active_proxy_placeholder_ids(agent, tool_call_meta))


def active_proxy_placeholder_ids(
    agent: Any, tool_call_meta: Mapping[str, Any]
) -> set[str]:
    """Return native ids for exact, provenance-backed active proxy stubs."""
    interrupt_state = getattr(agent, "_interrupt_state", None)
    if interrupt_state is None or not getattr(interrupt_state, "activated", False):
        return set()

    if not isinstance(tool_call_meta, Mapping):
        return set()
    context = getattr(interrupt_state, "context", None)
    if not isinstance(context, Mapping):
        return set()
    tool_results = context.get("tool_results")
    if not isinstance(tool_results, list):
        return set()

    return {
        tool_result["toolUseId"]
        for tool_result in tool_results
        if isinstance(tool_result, dict)
        and set(tool_result) == {"toolUseId", "status", "content"}
        and isinstance(tool_result["toolUseId"], str)
        and bool(tool_result["toolUseId"])
        and tool_result["status"] == "success"
        and tool_result["content"] == [{"text": PROXY_RESULT_PLACEHOLDER}]
        and isinstance(tool_call_meta.get(tool_result["toolUseId"]), Mapping)
        and tool_call_meta[tool_result["toolUseId"]].get("is_frontend") is True
    }


def _correct_single_tool(
    tool_result, pending_results: Mapping[str, _FrontendToolResult]
) -> str | None:
    """Rewrite matching placeholder ToolResult dict. Return the corrected tool_use_id or None if no
    correction was done."""
    if not isinstance(tool_result, dict):
        return None

    tool_use_id = tool_result.get("toolUseId")
    if tool_use_id in pending_results and _is_placeholder(tool_result.get("content")):
        result = pending_results[tool_use_id]
        tool_result.update(
            status=result.status,
            content=[{"text": result.provider_safe_content}],
        )
        return tool_use_id


def _correct_all_tools(
    tool_results, pending_results: Mapping[str, _FrontendToolResult]
) -> set[str]:
    """Rewrite matching placeholder ToolResult dicts in *tool_results* in place."""
    changed: set[str] = set()
    for tool_result in tool_results:
        tool_use_id = _correct_single_tool(tool_result, pending_results)
        if tool_use_id:
            changed.add(tool_use_id)
    return changed


def _correct_message(
    message: Any, pending_results: Mapping[str, _FrontendToolResult]
) -> set[str]:
    """Rewrite matching placeholder ``toolResult`` blocks in *message* in place.

    Returns the set of ``toolUseId``s whose block was corrected.
    """
    if not isinstance(message, dict):
        return set()
    changed: set[str] = set()
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        tool_result = block.get("toolResult")
        tool_use_id = _correct_single_tool(tool_result, pending_results)
        if tool_use_id:
            changed.add(tool_use_id)
    return changed


def _is_placeholder(content: Any) -> bool:
    """Return True if *content* is the proxy's ``"Forwarded to client"`` stub."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("text") == PROXY_RESULT_PLACEHOLDER
        for block in content
    )
