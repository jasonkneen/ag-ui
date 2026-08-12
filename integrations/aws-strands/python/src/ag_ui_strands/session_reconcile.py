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

import logging
from typing import Any, Iterable, Mapping

from .client_proxy_tool import PROXY_RESULT_PLACEHOLDER

logger = logging.getLogger(__name__)

# Key under which the adapter stores the ``{wire_tool_call_id: native_toolUseId}``
# map on the Strands agent's session state. Namespaced to avoid clashing with
# user-managed state keys.
AG_UI_WIRE_MAP_STATE_KEY = "__ag_ui_wire_to_native__"

# Key under which the adapter stores every ``toolUseId`` tool call metadata
# (name, args, input, strands_tool_id) on the Strands agent's session state.
# On a native-interrupt RESUME run Strands does not re-invoke the model for the
# interrupted tool, so no ``current_tool_use`` events fire and the in-run
# ``tool_calls_seen`` dict is empty when the ``toolResult`` arrives. Reading
# from this durable map at that point restores ``tool_name`` (and thus every
# ``tool_behaviors`` gate + the frontend-placeholder skip) for the resumed
# tool. Namespaced to avoid clashing with user-managed state keys.
AG_UI_TOOL_CALL_MAP_STATE_KEY = "__ag_ui_tool_call_map__"


def resolve_native_ids(
    wire_to_native: Mapping[str, str],
    frontend_results: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
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
        frontend_results: Items with ``wire_id`` and ``text``.

    Returns:
        Map of native ``toolUseId`` -> real result text (unresolvable dropped).
    """
    resolved: dict[str, str] = {}
    for result in frontend_results:
        native = wire_to_native.get(result.get("wire_id"))
        if native is not None:
            resolved[native] = result.get("text", "")
    return resolved


def reconcile_frontend_tool_results(
    session_manager: Any,
    agent: Any,
    pending_results: Mapping[str, str],
) -> set[str]:
    """Overwrite persisted placeholder ``toolResult`` blocks with real results.

    ``pending_results`` MUST be keyed by the Strands-native ``toolUseId`` (use
    :func:`resolve_native_ids` to translate client wire ids first).

    Args:
        session_manager: A Strands ``RepositorySessionManager`` (exposes
            ``session_id`` and ``session_repository``).
        agent: The Strands agent (exposes ``agent_id``).
        pending_results: Map of native ``toolUseId`` -> real result text.

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

    # Correct the agent's live interrupt state too. Isolated in its own
    # try/except: a failure here must not discard the ``corrected`` ids
    # already fixed above (agent.messages / session), and — unlike those,
    # which stay retryable via ``wire_to_native`` on a later turn — this
    # correction has no retry path. Strands clears
    # ``_interrupt_state.context`` once the resume succeeds (see
    # ``event_loop.py``), so a placeholder missed here (e.g. due to a
    # transient error) is unrecoverable after this turn.
    try:
        interrupt_state = getattr(agent, "_interrupt_state", None)
        if interrupt_state is not None and getattr(interrupt_state, "activated", False):
            tool_results = interrupt_state.context.get("tool_results")
            if tool_results:
                corrected |= _correct_all_tools(tool_results, pending_results)
    except Exception as e:  # noqa: BLE001 — degrade, don't lose already-corrected ids
        logger.warning(
            "Interrupt-context tool result reconciliation failed; the "
            f"stashed placeholder may reach the model unresolved: {e}",
            exc_info=True,
        )

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


def _correct_single_tool(tool_result, pending_results: Mapping[str, str]) -> str | None:
    """Rewrite matching placeholder ToolResult dict. Return the corrected tool_use_id or None if no
    correction was done."""
    if not isinstance(tool_result, dict):
        return None

    tool_use_id = tool_result.get("toolUseId")
    if tool_use_id in pending_results and _is_placeholder(tool_result.get("content")):
        tool_result["content"] = [{"text": pending_results[tool_use_id]}]
        return tool_use_id


def _correct_all_tools(tool_results, pending_results: Mapping[str, str]) -> set[str]:
    """Rewrite matching placeholder ToolResult dicts in *tool_results* in place."""
    changed: set[str] = set()
    for tool_result in tool_results:
        tool_use_id = _correct_single_tool(tool_result, pending_results)
        if tool_use_id:
            changed.add(tool_use_id)
    return changed


def _correct_message(message: Any, pending_results: Mapping[str, str]) -> set[str]:
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
