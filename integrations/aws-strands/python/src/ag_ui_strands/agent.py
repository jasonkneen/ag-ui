"""AWS Strands Agent adapter for AG-UI.

Translates Strands streaming events into the AG-UI event protocol.
"""

import asyncio
import base64
import copy
import inspect
import json
import logging
import math
import uuid
from typing import Any, AsyncIterator, Dict, List

from strands import Agent as StrandsAgentCore
from strands.hooks import HookProvider
from strands.hooks.events import (
    AfterInvocationEvent,
    BeforeToolCallEvent,
)
from strands.interrupt import InterruptException
from strands.session import SessionManager
from strands.tools import normalize_tool_spec
from strands.types.interrupt import InterruptResponseContent

# Params handled explicitly by StrandsAgent — excluded from auto-forwarding.
# "messages" is excluded: per-thread agents start with no history;
# AG-UI injects messages at runtime via RunAgentInput.
# "hooks" is excluded: Agent stores hooks as a HookRegistry after init, not
# the original list the constructor expects — forwarding it causes a TypeError.
# "session_manager" is excluded: it is supplied per-thread via
# StrandsAgentConfig.session_manager_provider (see run()). Forwarding a
# template-level session_manager would make every thread share one session_id.
_AGUI_EXPLICIT_PARAMS = {
    "self",
    "model",
    "system_prompt",
    "tools",
    "messages",
    "hooks",
    "session_manager",
}


def _extract_agent_kwargs(agent: StrandsAgentCore) -> dict:
    """Build kwargs for StrandsAgentCore by introspecting its constructor signature.

    Tries ``self.<name>`` first, falls back to ``self._<name>`` — Strands stores
    some init params with an underscore prefix (e.g. ``retry_strategy`` lives at
    ``self._retry_strategy``). This keeps the adapter forward-compatible with
    any future param that follows either naming convention.
    """
    kwargs = {}
    for name in inspect.signature(StrandsAgentCore.__init__).parameters:
        if name in _AGUI_EXPLICIT_PARAMS:
            continue
        if hasattr(agent, name):
            value = getattr(agent, name)
        elif hasattr(agent, f"_{name}"):
            value = getattr(agent, f"_{name}")
        else:
            continue
        if value is None:
            continue
        # state is an AgentState container; extract the underlying plain dict
        if name == "state" and hasattr(value, "get"):
            value = value.get()
        kwargs[name] = value
    return kwargs


# Upper bound on the per-agent wire->native map held in session state. Bounds
# growth from frontend calls that never receive a client result (abandoned HITL)
# and so are never consumed/pruned. Generous — a thread rarely has this many
# outstanding frontend calls at once.
_WIRE_MAP_MAX = 512

# Upper bound on the per-agent tool-call metadata map held in session state.
# It bounds abandoned entries (tool calls whose result never returns)
# so state cannot grow without bound.
_TOOL_CALL_MAP_MAX = 512

# Sentinel handed back to a paused ``tool_context.interrupt()`` when the client
# cancels (``ResumeEntry.status == "cancelled"``) rather than resolving. The
# tool receives this in place of a real answer and can treat it as a denial.
INTERRUPT_CANCELLED = {"cancelled": True}


def _wrap_resume_response(status: str, payload: Any) -> dict:
    """Package a ``ResumeEntry`` for Strands' ``interruptResponse`` shape.

    Strands' resume gate is truthiness-based (i.e. ``if interrupt_.response:``),
    so a raw falsy payload (``None``, ``False``, ``""``, ``0``, ``[]``, ``{}``)
    re-raises the same interrupt and re-runs the tool body — an infinite approve loop.
    Always hand Strands a truthy envelope; up to the tool implementation to properly
    destructures it (e.g. via ``.get("cancelled")`` / ``.get("response")``).
    """
    if status == "cancelled":
        return dict(INTERRUPT_CANCELLED)
    return {"response": payload}


def _get_strands_session_manager(agent: Any) -> Any:
    """Return the agent's Strands ``SessionManager``, or ``None``.

    Strands stores it publicly as ``session_manager``; some versions keep a
    private ``_session_manager`` alias.
    """
    return getattr(agent, "session_manager", None) or getattr(
        agent, "_session_manager", None
    )


def _interrupt_tool_call_id(strands_interrupt: Any) -> str | None:
    """Extract the native ``toolUseId`` embedded in a Strands interrupt id."""
    s_id = getattr(strands_interrupt, "id", "")
    s_id_parts = s_id.split(":") if isinstance(s_id, str) else []
    if len(s_id_parts) >= 4:
        # toolUseId is freeform and can itself contain ":" — slice the parts
        # list to drop only the "v1"/"<kind>" prefix and the trailing uuid.
        return ":".join(s_id_parts[2:-1])
    return None


_INTERRUPT_METADATA_MAX_DEPTH = 20
_INTERRUPT_METADATA_KEY_PREFIX = "__ag_ui_key_v1__:"


def _wire_safe_text(value: str) -> str:
    """Return UTF-8-safe text, escaping lone surrogate code points."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return value.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return value


def _wire_safe_mapping_key(value: str) -> str:
    """Encode mapping keys injectively while retaining ordinary UTF-8 keys."""
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        encoded = value.encode("utf-8", errors="surrogatepass")
        tag = "s:"
    else:
        if not value.startswith(_INTERRUPT_METADATA_KEY_PREFIX):
            return value
        tag = "v:"

    payload = base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")
    return f"{_INTERRUPT_METADATA_KEY_PREFIX}{tag}{payload}"


def _stable_json_sort_key(value: Any) -> str:
    """Return a canonical key for values already normalized for JSON."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _interrupt_metadata_to_json_safe(
    value: Any,
    *,
    _ancestors: frozenset[int] = frozenset(),
    _depth: int = 0,
) -> Any:
    """Normalize free-form Strands interrupt details for the AG-UI wire.

    JSON-native values are retained. Bytes use Strands' session serialization
    marker, unordered containers are canonically sorted, and opaque objects
    retain only their qualified type (never their address-bearing ``repr``).
    Container recursion is cycle-safe and depth-bounded.
    """
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "infinity"
        else:
            label = "-infinity"
        return {"__ag_ui_type__": "non_finite_float", "value": label}
    if type(value) is str:
        return _wire_safe_text(value)
    if type(value) is bytes:
        return {
            "__bytes_encoded__": True,
            "data": base64.b64encode(value).decode("ascii"),
        }

    if _depth >= _INTERRUPT_METADATA_MAX_DEPTH:
        return {"__ag_ui_type__": "max_depth_exceeded"}

    if type(value) in (dict, list, tuple, set, frozenset):
        identity = id(value)
        if identity in _ancestors:
            return {"__ag_ui_type__": "circular_reference"}
        descendants = _ancestors | {identity}

        if type(value) is dict:
            if all(type(key) is str for key in value):
                return {
                    _wire_safe_mapping_key(key): _interrupt_metadata_to_json_safe(
                        item,
                        _ancestors=descendants,
                        _depth=_depth + 1,
                    )
                    for key, item in value.items()
                }
            entries = [
                [
                    (
                        _wire_safe_mapping_key(key)
                        if type(key) is str
                        else _interrupt_metadata_to_json_safe(
                            key,
                            _ancestors=descendants,
                            _depth=_depth + 1,
                        )
                    ),
                    _interrupt_metadata_to_json_safe(
                        item,
                        _ancestors=descendants,
                        _depth=_depth + 1,
                    ),
                ]
                for key, item in value.items()
            ]
            entries.sort(key=_stable_json_sort_key)
            return {"__ag_ui_type__": "mapping", "items": entries}

        normalized_items = [
            _interrupt_metadata_to_json_safe(
                item,
                _ancestors=descendants,
                _depth=_depth + 1,
            )
            for item in value
        ]
        if type(value) in (set, frozenset):
            normalized_items.sort(key=_stable_json_sort_key)
            return {
                "__ag_ui_type__": "set" if type(value) is set else "frozenset",
                "items": normalized_items,
            }
        return normalized_items

    value_type = type(value)
    return {
        "__ag_ui_type__": "python_object",
        "type": _wire_safe_text(f"{value_type.__module__}.{value_type.__qualname__}"),
    }


AG_UI_PROXY_HOOK_PROVENANCE_STATE_KEY = "__ag_ui_proxy_hook_provenance__"
_PROXY_HOOK_FAILURE_MESSAGE = (
    "Frontend tool call was not executed because a BeforeToolCall hook "
    "changed its semantics."
)


def _json_native_copy(value: Any) -> Any:
    """Copy internal identity without applying wire metadata encoding."""
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True))


def _agent_state_key_present(agent: Any, key: str) -> bool:
    """Check one adapter key without copying the complete AgentState.

    Strands 1.18 AgentState has no public membership API, while ``get`` cannot
    distinguish a missing key from a present ``None`` value. Its backing
    mapping is therefore the only efficient presence check and is used only
    for adapter-owned keys; values still come from the public keyed ``get``.
    """
    state_store = getattr(agent.state, "_state", None)
    if isinstance(state_store, dict):
        return key in state_store
    return agent.state.get(key) is not None


def _proxy_hook_provenance_value(
    agent: Any,
) -> tuple[
    bool,
    frozenset[str] | None,
    dict[str, dict[str, Any]] | None,
]:
    """Decode durable proxy-hook provenance, distinguishing absence/malformed."""
    present = _agent_state_key_present(agent, AG_UI_PROXY_HOOK_PROVENANCE_STATE_KEY)
    if not present:
        return False, frozenset(), {}
    raw = agent.state.get(AG_UI_PROXY_HOOK_PROVENANCE_STATE_KEY)
    if (
        not isinstance(raw, dict)
        or raw.get("version") != 1
        or set(raw) != {"version", "managed_interrupt_ids", "records"}
        or not isinstance(raw.get("managed_interrupt_ids"), list)
        or not all(
            isinstance(interrupt_id, str)
            and interrupt_id.startswith("v1:before_tool_call:")
            for interrupt_id in raw["managed_interrupt_ids"]
        )
        or len(set(raw["managed_interrupt_ids"])) != len(raw["managed_interrupt_ids"])
        or not isinstance(raw.get("records"), dict)
    ):
        return True, None, None
    managed_ids = frozenset(raw["managed_interrupt_ids"])
    records = raw["records"]
    if set(records) != managed_ids or not all(
        isinstance(interrupt_id, str)
        and isinstance(record, dict)
        and isinstance(record.get("original_native_tool_call_id"), str)
        and isinstance(record.get("wire_tool_call_id"), str)
        and isinstance(record.get("name"), str)
        and isinstance(record.get("input"), dict)
        and isinstance(record.get("tool_spec"), dict)
        for interrupt_id, record in records.items()
    ):
        return True, None, None
    return True, managed_ids, records


def _proxy_hook_provenance_payload(
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the strict on-session proxy provenance envelope."""
    return {
        "version": 1,
        "managed_interrupt_ids": sorted(records),
        "records": records,
    }


def _proxy_hook_error(changed_fields: set[str] | None = None) -> "RunErrorEvent":
    from ag_ui.core import EventType, RunErrorEvent

    changed = sorted(changed_fields or ())
    suffix = f" ({', '.join(changed)})" if changed else ""
    return RunErrorEvent(
        type=EventType.RUN_ERROR,
        message=(
            "A BeforeToolCall hook changed frontend proxy semantics after "
            f"the call was emitted{suffix}"
        ),
        code="INTERRUPT_PROXY_PROVENANCE_ERROR",
    )


class _ProxyHookBoundary:
    """Per-core authoritative state inaccessible to caller invocation state."""

    def __init__(self) -> None:
        self.captures: dict[int, dict[str, Any]] = {}
        self.claims: dict[str, str] = {}
        self.candidate_records: dict[str, dict[str, Any]] = {}
        self.candidate_failures: dict[str, set[str]] = {}
        self.candidate_captures: dict[str, dict[str, Any]] = {}
        self.candidate_collisions: set[str] = set()
        self.expected_records: dict[str, dict[str, Any]] = {}
        self.failure_fields: set[str] = set()
        self.authoritative_resume_results: dict[str, Any] = {}
        self.authoritative_records: dict[str, dict[str, Any]] = {}
        self.working_resume_results: dict[str, Any] | None = None
        self.run_checkpoint: dict[str, Any] | None = None
        self.failed = False

    def prepare_run(
        self,
        agent: Any,
        resume_results: dict[str, Any] | None,
    ) -> None:
        self.captures.clear()
        self.claims.clear()
        self.candidate_records.clear()
        self.candidate_failures.clear()
        self.candidate_captures.clear()
        self.candidate_collisions.clear()
        self.expected_records.clear()
        self.failure_fields.clear()
        self.failed = False
        self.authoritative_resume_results = copy.deepcopy(resume_results or {})
        self.working_resume_results = resume_results if resume_results else None
        interrupt_state = agent._interrupt_state
        _present, managed_ids, records = _proxy_hook_provenance_value(agent)
        self.authoritative_records = copy.deepcopy(records or {})
        exact_resume = bool(
            interrupt_state.activated
            and managed_ids
            and managed_ids.intersection(interrupt_state.interrupts)
        )
        session_manager = _get_strands_session_manager(agent)
        no_session_proxy_turn = bool(
            session_manager is None and registered_proxy_tool_names(agent.tool_registry)
        )
        message_snapshot: list | None = None
        message_snapshot_is_deep = False
        conversation_manager_state: dict[str, Any] | None = None
        if exact_resume:
            message_snapshot = copy.deepcopy(agent.messages)
            message_snapshot_is_deep = True
            conversation_manager_state = copy.deepcopy(
                agent.conversation_manager.get_state()
            )
        elif no_session_proxy_turn:
            # Initial no-session failures are restored to the exact live
            # checkpoint without copying arbitrary existing message payloads.
            message_snapshot = list(agent.messages)
            conversation_manager_state = _json_native_copy(
                agent.conversation_manager.get_state()
            )
        self.run_checkpoint = {
            "interrupts": dict(interrupt_state.interrupts),
            "responses": {
                interrupt_id: interrupt.response
                for interrupt_id, interrupt in interrupt_state.interrupts.items()
            },
            "context": (
                copy.deepcopy(interrupt_state.context)
                if exact_resume
                else interrupt_state.context
            ),
            "activated": interrupt_state.activated,
            "exact_resume": exact_resume,
            "messages_object": agent.messages,
            "messages": message_snapshot,
            "messages_deep": message_snapshot_is_deep,
            "conversation_manager_state": conversation_manager_state,
            "state": {
                key: (
                    _agent_state_key_present(agent, key),
                    agent.state.get(key),
                )
                for key in (
                    AG_UI_WIRE_MAP_STATE_KEY,
                    AG_UI_TOOL_CALL_MAP_STATE_KEY,
                    AG_UI_PROXY_HOOK_PROVENANCE_STATE_KEY,
                )
            },
        }

    def capture(self, event: Any) -> None:
        if getattr(event.selected_tool, "_ag_ui_proxy", False) is not True:
            return
        original_tool_use = copy.deepcopy(event.tool_use)
        native_id = original_tool_use.get("toolUseId")
        tool_meta = event.agent.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY) or {}
        metadata = tool_meta.get(native_id) if isinstance(tool_meta, dict) else None
        wire_id = (
            metadata.get("wire_tool_call_id") if isinstance(metadata, dict) else None
        )
        wire_map = event.agent.state.get(AG_UI_WIRE_MAP_STATE_KEY) or {}
        if not isinstance(wire_id, str) and isinstance(wire_map, dict):
            wire_id = next(
                (
                    wire
                    for wire, mapped_native in wire_map.items()
                    if mapped_native == native_id
                ),
                None,
            )
        if not isinstance(wire_id, str):
            wire_id = next(
                (
                    data.get("wire_tool_call_id")
                    for data in self.captures.values()
                    if data["original_tool_use"]["toolUseId"] == native_id
                    and isinstance(data.get("wire_tool_call_id"), str)
                ),
                None,
            )
        capture = {
            "original_tool_use": original_tool_use,
            "original_tool_use_object": event.tool_use,
            "selected_tool": event.selected_tool,
            "tool_spec": _json_native_copy(event.selected_tool.tool_spec),
            "cancel_tool": event.cancel_tool,
            "wire_tool_call_id": wire_id,
            "interrupt_ids": [],
            "wire_map_entry": (
                (
                    wire_id in wire_map
                    if isinstance(wire_map, dict) and isinstance(wire_id, str)
                    else False
                ),
                (
                    copy.deepcopy(wire_map.get(wire_id))
                    if isinstance(wire_map, dict) and isinstance(wire_id, str)
                    else None
                ),
            ),
            "tool_meta_entry": (
                native_id in tool_meta if isinstance(tool_meta, dict) else False,
                (
                    copy.deepcopy(tool_meta.get(native_id))
                    if isinstance(tool_meta, dict)
                    else None
                ),
            ),
            "original_interrupt": event.interrupt,
            "event": event,
        }
        self.captures[id(event)] = capture

        original_interrupt = event.interrupt

        def observed_interrupt(
            name: str, reason: Any = None, response: Any = None
        ) -> Any:
            try:
                return original_interrupt(name, reason=reason, response=response)
            except InterruptException as exception:
                self._record_claim(capture, exception.interrupt.id)
                raise

        # BeforeToolCallEvent intentionally allows only its documented fields
        # through __setattr__. Strands exposes no public interrupt-observation
        # hook, so object.__setattr__ installs the smallest call-exact seam on
        # this event only; the finalizer removes it before the event escapes.
        object.__setattr__(event, "interrupt", observed_interrupt)

    @staticmethod
    def _restore_interrupt_method(capture: dict[str, Any]) -> None:
        event = capture["event"]
        if "interrupt" in vars(event):
            object.__delattr__(event, "interrupt")

    def _record_claim(self, capture: dict[str, Any], interrupt_id: str) -> None:
        native_id = capture["original_tool_use"]["toolUseId"]
        prior_native = self.claims.setdefault(interrupt_id, native_id)
        if prior_native != native_id:
            self.candidate_collisions.add(interrupt_id)
        if interrupt_id not in capture["interrupt_ids"]:
            capture["interrupt_ids"].append(interrupt_id)

    def promote_interrupts(self, agent: Any, interrupt_ids: set[str]) -> None:
        """Persist only candidates Strands actually advertised as pending."""
        if not interrupt_ids:
            return
        present, managed_ids, records = _proxy_hook_provenance_value(agent)
        if present and (managed_ids is None or records is None):
            self.failure_fields.add("provenance_state")
            self.failed = True
            return
        promoted = copy.deepcopy(records or {})
        restored_captures: set[int] = set()
        for interrupt_id in interrupt_ids:
            if interrupt_id in self.candidate_collisions:
                self.failure_fields.add("interrupt_id_collision")
                self.failed = True
            record = self.candidate_records.get(interrupt_id)
            if record is None:
                self.failure_fields.add("provenance_state")
                self.failed = True
                continue
            capture = self.candidate_captures[interrupt_id]
            if id(capture) not in restored_captures:
                # HookRegistry has now proven the candidate escaped as a real
                # pause. Restore the native tool-use object before Strands
                # resumes and parks it in interrupt context. Non-pausing calls
                # never pass this seam, so their ordinary hook mutations stay
                # untouched.
                original_object = capture["original_tool_use_object"]
                original_object.clear()
                original_object.update(copy.deepcopy(capture["original_tool_use"]))
                live_spec = getattr(capture["selected_tool"], "tool_spec", None)
                if isinstance(live_spec, dict):
                    live_spec.clear()
                    live_spec.update(_json_native_copy(capture["tool_spec"]))
                restored_captures.add(id(capture))
            candidate_failures = self.candidate_failures.get(interrupt_id, set())
            if candidate_failures:
                self.failure_fields.update(candidate_failures)
                self.failed = True
            promoted[interrupt_id] = copy.deepcopy(record)
            self.expected_records[interrupt_id] = copy.deepcopy(record)
        if promoted:
            agent.state.set(
                AG_UI_PROXY_HOOK_PROVENANCE_STATE_KEY,
                _proxy_hook_provenance_payload(promoted),
            )

    def finalize(self, event: Any) -> None:
        capture = self.captures.pop(id(event), None)
        if capture is None:
            return
        self._restore_interrupt_method(capture)
        original = capture["original_tool_use"]
        current = event.tool_use
        changed: set[str] = set()
        if not isinstance(current, dict) or current.get("name") != original["name"]:
            changed.add("name")
        if not isinstance(current, dict) or current.get("input") != original.get(
            "input", {}
        ):
            changed.add("input")
        if event.selected_tool is not capture["selected_tool"]:
            changed.add("selected_tool")
        if getattr(event.selected_tool, "tool_spec", None) != capture["tool_spec"]:
            changed.add("tool_spec")
        if event.cancel_tool != capture["cancel_tool"]:
            changed.add("cancel_tool")

        native_id = original["toolUseId"]
        prior_records = [
            record
            for record in self.authoritative_records.values()
            if record["original_native_tool_call_id"] == native_id
        ]
        is_managed_resume = bool(
            self.run_checkpoint
            and self.run_checkpoint["exact_resume"]
            and prior_records
        )
        has_expected_resume = native_id in self.authoritative_resume_results
        expected_resume = self.authoritative_resume_results.get(native_id)
        working_results = event.invocation_state.get(PROXY_RESUME_RESULTS_KEY)
        if has_expected_resume and (
            not isinstance(working_results, dict)
            or working_results.get(native_id) != expected_resume
        ):
            changed.add("proxy_resume_result")

        if not capture["interrupt_ids"] and not is_managed_resume:
            # No proxy checkpoint was established. Caller mutation remains
            # ordinary legal Strands behavior and must not be constrained.
            return

        if is_managed_resume:
            for prior_record in prior_records:
                if prior_record["wire_tool_call_id"] != capture["wire_tool_call_id"]:
                    changed.add("wire_tool_call_id")
                if prior_record["name"] != original["name"]:
                    changed.add("name")
                if prior_record["input"] != original.get("input", {}):
                    changed.add("input")
                if prior_record["tool_spec"] != capture["tool_spec"]:
                    changed.add("tool_spec")

        current_wire_map = event.agent.state.get(AG_UI_WIRE_MAP_STATE_KEY) or {}
        wire_present, wire_value = capture["wire_map_entry"]
        if (
            not isinstance(current_wire_map, dict)
            or (
                capture["wire_tool_call_id"] in current_wire_map
                if isinstance(capture["wire_tool_call_id"], str)
                else False
            )
            != wire_present
            or (
                isinstance(capture["wire_tool_call_id"], str)
                and current_wire_map.get(capture["wire_tool_call_id"]) != wire_value
            )
        ):
            changed.add("wire_map")
        current_tool_meta = event.agent.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY) or {}
        meta_present, meta_value = capture["tool_meta_entry"]
        if (
            not isinstance(current_tool_meta, dict)
            or (native_id in current_tool_meta) != meta_present
            or current_tool_meta.get(native_id) != meta_value
        ):
            changed.add("tool_call_metadata")

        present_now, managed_now, records_now = _proxy_hook_provenance_value(
            event.agent
        )
        allowed_records = {
            **copy.deepcopy(self.authoritative_records),
            **copy.deepcopy(self.expected_records),
        }
        if (bool(allowed_records) and not present_now) or (
            present_now
            and (
                managed_now is None
                or records_now is None
                or records_now != allowed_records
            )
        ):
            changed.add("provenance_state")

        if not isinstance(capture["wire_tool_call_id"], str):
            changed.add("wire_tool_call_id")
        for interrupt_id in capture["interrupt_ids"]:
            record = {
                "original_native_tool_call_id": native_id,
                "wire_tool_call_id": capture["wire_tool_call_id"],
                "name": original["name"],
                "input": _json_native_copy(original.get("input", {})),
                "tool_spec": _json_native_copy(capture["tool_spec"]),
            }
            self.candidate_records[interrupt_id] = record
            self.candidate_failures[interrupt_id] = set(changed)
            self.candidate_captures[interrupt_id] = capture
        if not is_managed_resume:
            # A call to event.interrupt() is only a candidate until Strands
            # emits ToolInterruptEvent. Preemptively answered or caller-caught
            # interrupts are ordinary non-pausing hooks and remain unrestricted.
            return

        original_object = capture["original_tool_use_object"]
        original_object.clear()
        original_object.update(copy.deepcopy(original))
        event.tool_use = original_object
        event.selected_tool = capture["selected_tool"]
        event.cancel_tool = capture["cancel_tool"]
        live_spec = getattr(event.selected_tool, "tool_spec", None)
        if isinstance(live_spec, dict):
            live_spec.clear()
            live_spec.update(_json_native_copy(capture["tool_spec"]))
        if self.working_resume_results is not None:
            # Re-pin the caller-visible working map from the private copy. The
            # authoritative dict itself never enters invocation_state, so a
            # hook cannot retain and mutate our source of truth.
            self.working_resume_results.clear()
            self.working_resume_results.update(
                copy.deepcopy(self.authoritative_resume_results)
            )
            event.invocation_state[PROXY_RESUME_RESULTS_KEY] = (
                self.working_resume_results
            )
        if changed:
            self.failure_fields.update(changed)
            self.failed = True
            # HookRegistry's supported control path is an interrupt. A fresh
            # name guarantees this checkpoint cannot already have a response;
            # rollback removes it before any retry is exposed to the caller.
            capture["original_interrupt"](
                f"__ag_ui_integrity_failure_{uuid.uuid4()}__",
                reason=_PROXY_HOOK_FAILURE_MESSAGE,
            )

    def after_invocation(self, event: Any) -> None:
        for capture in self.captures.values():
            self._restore_interrupt_method(capture)
        if (
            self.run_checkpoint
            and self.run_checkpoint["exact_resume"]
            and self.captures
        ):
            managed_native_ids = {
                record["original_native_tool_call_id"]
                for record in self.authoritative_records.values()
            }
            if any(
                capture["original_tool_use"].get("toolUseId") in managed_native_ids
                for capture in self.captures.values()
            ):
                self.failure_fields.add("hook_completion")
                self.failed = True
        if not self.expected_records and not self.failed:
            return
        _present, managed_ids, records = _proxy_hook_provenance_value(event.agent)
        if (
            records is None
            or managed_ids is None
            or any(
                interrupt_id not in managed_ids or records.get(interrupt_id) != expected
                for interrupt_id, expected in self.expected_records.items()
            )
        ):
            self.failure_fields.add("provenance_state")
            self.failed = True
        if not self.failed:
            return
        self._restore_failed_attempt(event.agent)

    def _restore_failed_attempt(self, agent: Any) -> None:
        checkpoint = self.run_checkpoint
        if checkpoint is None:
            return
        interrupt_state = agent._interrupt_state
        interrupt_state.interrupts = dict(checkpoint["interrupts"])
        for interrupt_id, response in checkpoint["responses"].items():
            interrupt_state.interrupts[interrupt_id].response = response
        interrupt_state.context = (
            copy.deepcopy(checkpoint["context"])
            if checkpoint["exact_resume"]
            else checkpoint["context"]
        )
        interrupt_state.activated = checkpoint["activated"]
        for key, (present, value) in checkpoint["state"].items():
            if not present:
                agent.state.delete(key)
            else:
                agent.state.set(key, value)

        session_manager = _get_strands_session_manager(agent)
        if checkpoint["messages"] is not None and (
            checkpoint["exact_resume"] or session_manager is None
        ):
            original_messages = checkpoint["messages_object"]
            restored_messages = (
                copy.deepcopy(checkpoint["messages"])
                if checkpoint["messages_deep"]
                else list(checkpoint["messages"])
            )
            original_messages[:] = restored_messages
            agent.messages = original_messages
            conversation_manager_state = checkpoint["conversation_manager_state"]
            if conversation_manager_state is not None:
                agent.conversation_manager.restore_from_session(
                    copy.deepcopy(conversation_manager_state)
                )
        elif not checkpoint["activated"] and session_manager is not None:
            # Initial session-backed failures cannot delete the durable batch
            # through the public API. Find the exact assistant toolUse batch,
            # replace it live, and publicly redact the persisted latest copy.
            managed_native_ids = set(self.claims.values()) | {
                record["original_native_tool_call_id"]
                for record in self.expected_records.values()
            }
            target_index = next(
                (
                    index
                    for index in range(len(agent.messages) - 1, -1, -1)
                    if any(
                        block.get("toolUse", {}).get("toolUseId") in managed_native_ids
                        for block in agent.messages[index].get("content", [])
                    )
                ),
                None,
            )
            replacement = {
                "role": "assistant",
                "content": [{"text": _PROXY_HOOK_FAILURE_MESSAGE}],
            }
            if target_index is not None:
                agent.messages[target_index] = replacement
            if managed_native_ids:
                session_manager.redact_latest_message(replacement, agent)


class _ProxyHookCaptureProvider(HookProvider):
    def __init__(self, boundary: _ProxyHookBoundary) -> None:
        self.boundary = boundary

    def register_hooks(self, registry: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.boundary.capture)
        registry.add_callback(AfterInvocationEvent, self.boundary.after_invocation)


class _ProxyHookFinalizerProvider(HookProvider):
    def __init__(self, boundary: _ProxyHookBoundary) -> None:
        self.boundary = boundary

    def register_hooks(self, registry: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.boundary.finalize)


def _strands_interrupt_to_agui(
    strands_interrupt: Any,
    native_to_wire: dict[str, str] | None = None,
    interrupt_to_wire: dict[str, str] | None = None,
) -> "Interrupt":
    """Map a native Strands ``Interrupt`` onto an AG-UI ``Interrupt``.

    Every Strands interrupt originates from ``tool_context.interrupt()`` or a
    ``BeforeToolCallEvent`` hook, so its id always embeds the triggering
    ``toolUseId`` (``v1:<kind>:<toolUseId>:<uuid>``) and is inherently
    tool-call-bound. This maps onto AG-UI's reserved ``reason="tool_call"``
    core value, with ``tool_call_id`` extracted from the id.

    Strands' free-form ``name`` and ``reason`` are preserved under ``metadata``
    (``strands_name`` / ``strands_reason``), normalized into deterministic,
    JSON-safe values. ``message`` additionally carries a UTF-8-safe ``reason``
    when it is a plain string, since AG-UI clients render ``message`` directly.
    """
    s_id = getattr(strands_interrupt, "id", "")
    name = getattr(strands_interrupt, "name", None) or "interrupt"
    raw_reason = getattr(strands_interrupt, "reason", None)

    native_tool_call_id = _interrupt_tool_call_id(strands_interrupt)
    tool_call_id = (interrupt_to_wire or {}).get(s_id)
    if tool_call_id is None:
        tool_call_id = (native_to_wire or {}).get(
            native_tool_call_id, native_tool_call_id
        )

    metadata = {"strands_name": _interrupt_metadata_to_json_safe(name)}
    if raw_reason is not None:
        metadata["strands_reason"] = _interrupt_metadata_to_json_safe(raw_reason)

    return Interrupt(
        id=s_id,
        tool_call_id=tool_call_id,
        reason="tool_call",
        message=_wire_safe_text(raw_reason) if type(raw_reason) is str else None,
        metadata=metadata,
    )


def _unanswered_interrupt_ids(interrupt_state: Any) -> frozenset[str]:
    """Return ids whose Strands interrupt response is still falsy."""
    return frozenset(
        interrupt_id
        for interrupt_id, interrupt in getattr(
            interrupt_state, "interrupts", {}
        ).items()
        if not getattr(interrupt, "response", None)
    )


def _pending_proxy_hook_native_ids(
    agent: Any, tool_call_meta: dict[str, dict[str, Any]]
) -> set[str]:
    """Return native ids for unanswered before-tool-call proxy interrupts."""
    interrupt_state = getattr(agent, "_interrupt_state", None)
    if interrupt_state is None or not getattr(interrupt_state, "activated", False):
        return set()

    present, managed_ids, records = _proxy_hook_provenance_value(agent)
    if present and (managed_ids is None or records is None):
        return set()
    native_ids: set[str] = set()
    unanswered_ids = _unanswered_interrupt_ids(interrupt_state)
    for stored_interrupt_id, interrupt in getattr(
        interrupt_state, "interrupts", {}
    ).items():
        if stored_interrupt_id not in unanswered_ids:
            continue
        interrupt_id = getattr(interrupt, "id", "")
        if not isinstance(interrupt_id, str) or not interrupt_id.startswith(
            "v1:before_tool_call:"
        ):
            continue
        if present and interrupt_id not in (managed_ids or frozenset()):
            # A native BeforeToolCall interrupt may be a sibling of a managed
            # proxy interrupt. Presence of our manifest does not claim it.
            continue
        record = (records or {}).get(interrupt_id)
        native_id = (
            record.get("original_native_tool_call_id")
            if isinstance(record, dict)
            else _interrupt_tool_call_id(interrupt)
        )
        metadata = tool_call_meta.get(native_id) if native_id else None
        if isinstance(metadata, dict) and metadata.get("is_frontend") is True:
            native_ids.add(native_id)
    return native_ids


def _validate_active_proxy_hook_provenance(
    agent: Any, tool_call_meta: dict[str, dict[str, Any]]
) -> "RunErrorEvent | None":
    interrupt_state = getattr(agent, "_interrupt_state", None)
    if (
        interrupt_state is None
        or getattr(interrupt_state, "activated", False) is not True
    ):
        return None
    present, managed_ids, records = _proxy_hook_provenance_value(agent)
    if present and (managed_ids is None or records is None):
        return _proxy_hook_error()
    if not present:
        # Do not claim genuine native/legacy checkpoints. A paused frontend
        # proxy, however, is adapter-owned and cannot be resumed safely after
        # its durable provenance marker has disappeared.
        unanswered_ids = _unanswered_interrupt_ids(interrupt_state)
        for stored_interrupt_id, interrupt in interrupt_state.interrupts.items():
            if stored_interrupt_id not in unanswered_ids:
                continue
            interrupt_id = getattr(interrupt, "id", "")
            if not isinstance(interrupt_id, str) or not interrupt_id.startswith(
                "v1:before_tool_call:"
            ):
                continue
            native_id = _interrupt_tool_call_id(interrupt)
            metadata = tool_call_meta.get(native_id) if native_id else None
            if isinstance(metadata, dict) and metadata.get("is_frontend") is True:
                return _proxy_hook_error()
        return None
    active_interrupt_ids = set(interrupt_state.interrupts)
    if not (managed_ids or frozenset()).issubset(active_interrupt_ids):
        return _proxy_hook_error()
    unanswered_ids = _unanswered_interrupt_ids(interrupt_state)
    pending_records = {
        interrupt_id: record
        for interrupt_id, record in (records or {}).items()
        if interrupt_id in unanswered_ids
    }
    return _validate_proxy_hook_record_bindings(
        agent, tool_call_meta, pending_records
    )


def _validate_proxy_hook_record_bindings(
    agent: Any,
    tool_call_meta: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> "RunErrorEvent | None":
    """Validate every managed record against adapter-owned routing state."""
    wire_map = agent.state.get(AG_UI_WIRE_MAP_STATE_KEY) or {}
    if not isinstance(tool_call_meta, dict) or not isinstance(wire_map, dict):
        return _proxy_hook_error()
    for record in records.values():
        native_id = record["original_native_tool_call_id"]
        wire_id = record["wire_tool_call_id"]
        metadata = tool_call_meta.get(native_id)
        if (
            not isinstance(metadata, dict)
            or metadata.get("is_frontend") is not True
            or metadata.get("name") != record["name"]
            or metadata.get("input") != record["input"]
            or wire_map.get(wire_id) != native_id
        ):
            return _proxy_hook_error()
    return None


def _proxy_tool_spec_from_agui(tool: Any) -> dict[str, Any]:
    """Build the normalized Strands ToolSpec for a client declaration."""
    return _json_native_copy(
        normalize_tool_spec(copy.deepcopy(create_proxy_tool(tool).tool_spec))
    )


def _extract_interrupts(agent: Any, terminal_result: Any) -> list:
    """Return the native Strands interrupts for a paused run, or ``[]``.

    Prefers the terminal ``AgentResult`` (``stop_reason == "interrupt"`` with a
    populated ``interrupts``); falls back to the live agent's
    ``_interrupt_state`` so a pause is still detected if the result event was
    consumed by the stream's early-break path.
    """
    if terminal_result is not None:
        if getattr(terminal_result, "stop_reason", None) == "interrupt":
            interrupts = getattr(terminal_result, "interrupts", None) or []
            if interrupts:
                return [
                    interrupt
                    for interrupt in interrupts
                    if not getattr(interrupt, "response", None)
                ]
    interrupt_state = getattr(agent, "_interrupt_state", None)
    if interrupt_state is not None and getattr(interrupt_state, "activated", False):
        # Mirrors Strands' own gate (strands/types/interrupt.py: ``if interrupt_.response:``)
        # — an interrupt with a truthy response was already answered by a prior partial
        # resume and must not be re-reported as still pending.
        return [
            interrupt
            for interrupt in getattr(interrupt_state, "interrupts", {}).values()
            if not getattr(interrupt, "response", None)
        ]
    return []


def _interrupt_session_required_error() -> "RunErrorEvent":
    return RunErrorEvent(
        type=EventType.RUN_ERROR,
        message=(
            "A SessionManager is required to resume a native interrupt "
            "that was created alongside a frontend tool call"
        ),
        code="INTERRUPT_SESSION_REQUIRED",
    )


def _interrupt_session_capability_error() -> "RunErrorEvent":
    return RunErrorEvent(
        type=EventType.RUN_ERROR,
        message=(
            "Mixed frontend/native interrupt state requires repository reconciliation; "
            "the configured SessionManager must expose session_id and a "
            "session_repository with list_messages() and update_message()"
        ),
        code="INTERRUPT_SESSION_CAPABILITY_ERROR",
    )


def _proxy_resume_tool_capability_error(native_ids: set[str]) -> "RunErrorEvent":
    return RunErrorEvent(
        type=EventType.RUN_ERROR,
        message=(
            "Cannot resume frontend proxy hook checkpoint because its marked "
            "proxy tool specification is unavailable for native tool ids: "
            + ", ".join(sorted(native_ids))
        ),
        code="INTERRUPT_SESSION_CAPABILITY_ERROR",
    )


def _preflight_resume_entries(
    agent: Any, resume_entries: list[Any]
) -> "RunErrorEvent | None":
    """Validate a complete resume batch without mutating native state.

    Strands applies responses one at a time, so a later stale id can raise only
    after an earlier valid interrupt was already answered. It also ignores a
    resume prompt when its interrupt state is inactive. Validate the whole AG-UI
    batch before any adapter or Strands mutation while preserving partial resume
    semantics: a unique subset of the currently tracked interrupts is valid.
    """
    interrupt_state = getattr(agent, "_interrupt_state", None)
    if interrupt_state is None or not getattr(interrupt_state, "activated", False):
        return RunErrorEvent(
            type=EventType.RUN_ERROR,
            message="Cannot resume interrupts without an active interrupt state",
            code="INTERRUPT_RESUME_ERROR",
        )

    current_interrupts = getattr(interrupt_state, "interrupts", {})
    seen_ids: set[str] = set()
    for entry in resume_entries:
        interrupt_id = getattr(entry, "interrupt_id", None)
        if not isinstance(interrupt_id, str) or not interrupt_id:
            return RunErrorEvent(
                type=EventType.RUN_ERROR,
                message="Resume entries must contain a non-empty interrupt id",
                code="INTERRUPT_RESUME_ERROR",
            )
        if interrupt_id in seen_ids:
            return RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=f"Resume contains duplicate interrupt id: {interrupt_id}",
                code="INTERRUPT_RESUME_ERROR",
            )
        seen_ids.add(interrupt_id)
        if interrupt_id not in current_interrupts:
            return RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=f"Resume references unknown interrupt id: {interrupt_id}",
                code="INTERRUPT_RESUME_ERROR",
            )

    return None


logger = logging.getLogger(__name__)
from ag_ui.core import (
    AssistantMessage,
    CustomEvent,
    EventType,
    FunctionCall,
    Interrupt,
    MessagesSnapshotEvent,
    ReasoningEncryptedValueEvent,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunFinishedInterruptOutcome,
    RunStartedEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCall,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
    UserMessage,
)

from ag_ui_a2ui_toolkit import split_a2ui_schema_context

from .a2ui_tool import (
    A2UI_STREAM_KEY,
    is_auto_injected_a2ui_tool,
    plan_a2ui_injection,
)
from .client_proxy_tool import (
    PROXY_RESUME_RESULTS_KEY,
    create_proxy_tool,
    registered_proxy_tool_names,
    sync_proxy_tools,
)
from .session_reconcile import (
    AG_UI_TOOL_CALL_MAP_STATE_KEY,
    AG_UI_WIRE_MAP_STATE_KEY,
    ActiveInterruptReconciliationError,
    _FrontendToolResult,
    _supports_repository_reconciliation,
    active_proxy_placeholder_ids,
    has_active_proxy_placeholder,
    has_placeholder_results,
    reconcile_frontend_tool_results,
    resolve_native_ids,
)
from .config import (
    StrandsAgentConfig,
    ToolCallContext,
    ToolResultContext,
    maybe_await,
    normalize_predict_state,
)
from .utils import convert_agui_content_to_strands, flatten_content_to_text


def _coerce_text(content: Any) -> str:
    """Best-effort string view of an AG-UI message content field."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _normalize_frontend_tool_result(
    content: str, error: str | None
) -> _FrontendToolResult:
    """Couple frontend result text with its provider-visible status."""
    return _FrontendToolResult(
        content=content,
        status="error" if error is not None else "success",
        error=error,
    )


def _coerce_id(value: Any) -> str:
    """Return ``value`` if it is a non-empty string, else a fresh UUID."""
    return value if isinstance(value, str) and value else str(uuid.uuid4())


def _build_snapshot_messages(input_messages: List[Any]) -> List[Any]:
    """Convert ``RunAgentInput.messages`` to AG-UI message objects.

    Used to seed the running ``MessagesSnapshotEvent`` payload so each
    snapshot carries the full thread history (prior turns + whatever
    this turn produces).
    """
    out: List[Any] = []
    for msg in input_messages or []:
        role = getattr(msg, "role", None)
        if role not in ("user", "assistant", "tool"):
            continue
        msg_id = _coerce_id(getattr(msg, "id", None))
        if role == "user":
            raw = msg.content
            # Preserve list content (multimodal) as-is; only stringify unexpected types.
            content = raw if isinstance(raw, (str, list)) else _coerce_text(raw)
            out.append(UserMessage(id=msg_id, role="user", content=content))
        elif role == "assistant":
            tool_calls_list = None
            raw_tool_calls = getattr(msg, "tool_calls", None)
            if raw_tool_calls:
                tool_calls_list = []
                for tc in raw_tool_calls:
                    fn = getattr(tc, "function", None)
                    if isinstance(fn, dict):
                        fn_name = fn.get("name") or "unknown"
                        fn_args = fn.get("arguments") or "{}"
                    else:
                        fn_name = getattr(fn, "name", None) or "unknown"
                        fn_args = getattr(fn, "arguments", None) or "{}"
                    tc_id = _coerce_id(getattr(tc, "id", None))
                    tool_calls_list.append(
                        ToolCall(
                            id=tc_id,
                            type="function",
                            function=FunctionCall(
                                name=str(fn_name),
                                arguments=str(fn_args),
                            ),
                        )
                    )
            out.append(
                AssistantMessage(
                    id=msg_id,
                    role="assistant",
                    content=_coerce_text(msg.content),
                    tool_calls=tool_calls_list,
                )
            )
        elif role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "")
            if not isinstance(tool_call_id, str):
                tool_call_id = ""
            out.append(
                ToolMessage(
                    id=msg_id,
                    role="tool",
                    content=_coerce_text(msg.content),
                    tool_call_id=tool_call_id,
                    # This is an AG-UI -> AG-UI rebuild of the client's own message, so
                    # preserve its error/encrypted_value on the snapshot echo instead of
                    # silently dropping the client's own fields.
                    error=getattr(msg, "error", None),
                    encrypted_value=getattr(msg, "encrypted_value", None),
                )
            )
    return out


def _build_strands_history(input_messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert ``RunAgentInput.messages`` to Strands native ``Messages``.

    Strands has only ``user`` and ``assistant`` roles; tool calls and
    tool results live as ``toolUse`` / ``toolResult`` ContentBlocks.
    Reconciling the cached agent's ``self.messages`` with this list
    before invoking ``stream_async(None)`` ensures the LLM sees the
    real conversation state — including frontend tool results — rather
    than a fresh prompt that re-fires the same tool every turn.
    """
    out: List[Dict[str, Any]] = []
    pending_tool_results: List[Dict[str, Any]] = []

    def flush_tool_results() -> None:
        if not pending_tool_results:
            return
        out.append({"role": "user", "content": list(pending_tool_results)})
        pending_tool_results.clear()

    for msg in input_messages or []:
        role = getattr(msg, "role", None)
        if role == "tool":
            result = _normalize_frontend_tool_result(
                _coerce_text(msg.content), getattr(msg, "error", None)
            )
            pending_tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": getattr(msg, "tool_call_id", "") or "",
                        "content": [{"text": result.provider_safe_content}],
                        # Carry the AG-UI failure signal onto Bedrock's toolResult status,
                        # so a client-reported tool failure is not asserted to the model as
                        # a success.
                        "status": result.status,
                    }
                }
            )
            continue

        flush_tool_results()

        if role == "user":
            content = msg.content
            if isinstance(content, list):
                has_media = any(
                    getattr(item, "type", None) in ("image", "audio", "video", "document")
                    for item in content
                )
                if has_media:
                    blocks = convert_agui_content_to_strands(content)
                    if isinstance(blocks, list) and blocks:
                        out.append({"role": "user", "content": blocks})
                        continue
                text = flatten_content_to_text(content) or ""
                out.append({"role": "user", "content": [{"text": text}]})
            else:
                out.append({"role": "user", "content": [{"text": _coerce_text(content)}]})
        elif role == "assistant":
            blocks: List[Dict[str, Any]] = []
            text = _coerce_text(msg.content)
            if text:
                blocks.append({"text": text})
            raw_tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None)
                if isinstance(fn, dict):
                    name = fn.get("name") or "unknown"
                    args = fn.get("arguments") or "{}"
                else:
                    name = getattr(fn, "name", None) or "unknown"
                    args = getattr(fn, "arguments", None) or "{}"
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": tc.id,
                            "name": name,
                            "input": parsed if isinstance(parsed, dict) else {},
                        }
                    }
                )
            if not blocks:
                blocks = [{"text": ""}]
            out.append({"role": "assistant", "content": blocks})

    flush_tool_results()
    # Normalize so Bedrock's toolUse/toolResult pairing holds even when results
    # arrive out of order, are wedged apart by other messages, or span multiple
    # consecutive tool-call turns (parallel tool calls).
    return _normalize_tool_turns(out)


def _is_tooluse_only_assistant(m):
    return (
        m.get("role") == "assistant"
        and m.get("content")
        and all("toolUse" in b for b in m["content"])
    )


def _is_toolresult_only_user(m):
    return (
        m.get("role") == "user"
        and m.get("content")
        and all("toolResult" in b for b in m["content"])
    )


def _normalize_tool_turns(msgs):
    """Merge same-turn toolUse into one assistant msg and their toolResults
    into the immediately following user msg, dropping any messages wedged
    between a toolUse turn and its toolResults so Bedrock accepts the history.

    Messages that legitimately *follow* a completed toolUse/toolResult pair are
    preserved in place; only messages wedged *between* the toolUse turn and its
    results are dropped.
    """
    out = []
    i = 0
    n = len(msgs)
    while i < n:
        m = msgs[i]
        if not _is_tooluse_only_assistant(m):
            out.append(m)
            i += 1
            continue

        # Collect consecutive toolUse-only assistant messages into one.
        merged_tooluse = list(m["content"])
        j = i + 1
        while j < n and _is_tooluse_only_assistant(msgs[j]):
            merged_tooluse.extend(msgs[j]["content"])
            j += 1
        # Preserve first-seen order and de-duplicate ids: a repeated toolUseId
        # must not later emit a duplicate toolResult (Bedrock rejects that).
        tooluse_ids = []
        seen_ids = set()
        for b in merged_tooluse:
            rid = b["toolUse"]["toolUseId"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                tooluse_ids.append(rid)

        # Scan forward for the matching toolResults. Anything that is not a
        # matching result and appears *before* results are complete is "wedged"
        # and dropped; once every result is collected, the remaining messages
        # are left untouched to be processed in place by the outer loop.
        results_by_id = {}
        k = j
        while k < n and len(results_by_id) < len(tooluse_ids):
            mk = msgs[k]
            if _is_toolresult_only_user(mk):
                for b in mk["content"]:
                    rid = b["toolResult"].get("toolUseId")
                    if rid in seen_ids and rid not in results_by_id:
                        results_by_id[rid] = b
                    # non-matching / duplicate result blocks wedged in are dropped
            # non-toolResult messages wedged before completion are dropped
            k += 1

        # Emit merged assistant(toolUse) + merged user(toolResult) adjacently.
        out.append({"role": "assistant", "content": merged_tooluse})
        ordered = [results_by_id[tid] for tid in tooluse_ids if tid in results_by_id]
        if ordered:
            out.append({"role": "user", "content": ordered})

        # Continue with whatever legitimately follows, in place (no reordering).
        i = k
    return out


class StrandsAgent:
    """AWS Strands Agent wrapper for AG-UI integration."""

    def __init__(
        self,
        agent: StrandsAgentCore,
        name: str,
        description: str = "",
        config: "StrandsAgentConfig | None" = None,
        hooks: "list | None" = None,
    ):
        # Store template agent configuration for creating fresh instances
        self._model = agent.model
        self._system_prompt = agent.system_prompt
        self._tools = (
            list(agent.tool_registry.registry.values())
            if hasattr(agent, "tool_registry")
            else []
        )
        self._agent_kwargs = _extract_agent_kwargs(agent)

        # Hook providers forwarded to each per-thread StrandsAgentCore.
        #
        # Why a dedicated kwarg instead of reading them off the template?
        # Strands initializes ``Agent.hooks`` as a ``HookRegistry`` containing
        # only the registered callbacks — the original list of HookProvider
        # objects is not retained, and the registry also contains callbacks
        # bound to internal Strands objects (conversation manager, retry
        # strategy) that belong to the template and must not be cross-wired
        # into per-thread agents. We therefore take providers directly from
        # the caller and forward them to every per-thread instance so any
        # observability / loop-cap / policy-enforcement hook actually fires.
        self._hooks = list(hooks) if hooks else []

        self.name = name
        self.description = description
        self.config = config or StrandsAgentConfig()

        # Detect the common footgun: session_manager set on the template Agent
        # (stored as `_session_manager` by Strands) with no per-thread provider.
        # Forwarding it would make every AG-UI thread share one session_id.
        template_session_manager = getattr(agent, "_session_manager", None)
        if (
            template_session_manager is not None
            and self.config.session_manager_provider is None
        ):
            logger.warning(
                "session_manager was set on the template Agent but will be ignored: "
                "forwarding it would cause every AG-UI thread to share the same "
                "session_id. Construct per-thread session managers via "
                "StrandsAgentConfig.session_manager_provider instead."
            )

        # Dictionary to store agent instances per thread
        self._agents_by_thread: Dict[str, StrandsAgentCore] = {}
        self._proxy_hook_boundaries_by_thread: Dict[
            str, _ProxyHookBoundary
        ] = {}
        # Track proxy tool names registered per thread
        self._proxy_tool_names_by_thread: Dict[str, set] = {}
        # Guards first-time thread initialization. The session_manager_provider
        # call introduces an async yield point between the "is this thread
        # new?" check and the dict assignment, so concurrent requests for the
        # same new thread_id could otherwise both create an agent and one
        # would clobber the other.
        self._thread_init_lock = asyncio.Lock()

    def _will_emit_tool_snapshot(self, behavior: Any, emit_snapshots: bool) -> bool:
        # ``emit_snapshots`` is the per-run gate (config flag AND not a
        # delta-only payload); callers pass it so snapshot emission stays
        # suppressed on delta payloads that would otherwise wipe prior turns.
        return emit_snapshots and not (
            behavior and behavior.skip_messages_snapshot
        )

    async def run(self, input_data: RunAgentInput) -> AsyncIterator[Any]:
        """Run the Strands agent and yield AG-UI events."""

        # Get or create agent instance for this thread. When a
        # session_manager_provider is configured, the SessionManager handles
        # conversation persistence; otherwise state is held in-memory per thread.
        thread_id = input_data.thread_id or "default"
        if thread_id not in self._agents_by_thread:
            async with self._thread_init_lock:
                # Double-check inside the lock: another coroutine may have
                # completed initialization while we were waiting.
                if thread_id not in self._agents_by_thread:
                    session_manager = None
                    if self.config.session_manager_provider:
                        try:
                            session_manager = await maybe_await(
                                self.config.session_manager_provider(input_data)
                            )
                        except Exception as e:
                            # ERROR (not WARNING): the run is being aborted.
                            # exc_info=True preserves the full traceback so
                            # programming errors (TypeError, NameError, ...)
                            # in the provider surface clearly rather than
                            # looking like an infrastructure problem.
                            logger.error(
                                f"session_manager_provider failed: {e}",
                                exc_info=True,
                            )
                            yield RunStartedEvent(
                                type=EventType.RUN_STARTED,
                                thread_id=input_data.thread_id,
                                run_id=input_data.run_id,
                            )
                            yield RunErrorEvent(
                                type=EventType.RUN_ERROR,
                                message=f"Failed to initialize session manager: {e}",
                                code="SESSION_MANAGER_ERROR",
                            )
                            return
                        # Validate the provider return type at the boundary —
                        # otherwise a forgotten call or wrong type surfaces
                        # deep inside Strands with a confusing traceback.
                        if session_manager is not None and not isinstance(
                            session_manager, SessionManager
                        ):
                            actual = type(session_manager).__name__
                            logger.error(
                                "session_manager_provider returned %s; "
                                "expected a SessionManager instance.",
                                actual,
                            )
                            yield RunStartedEvent(
                                type=EventType.RUN_STARTED,
                                thread_id=input_data.thread_id,
                                run_id=input_data.run_id,
                            )
                            yield RunErrorEvent(
                                type=EventType.RUN_ERROR,
                                message=(
                                    f"session_manager_provider returned {actual}; "
                                    "expected a SessionManager instance"
                                ),
                                code="SESSION_MANAGER_INVALID_TYPE",
                            )
                            return
                    if session_manager is None and self.config.session_manager_provider:
                        logger.warning(
                            f"session_manager_provider returned None for thread_id={thread_id}; "
                            "agent will run without session persistence"
                        )
                    # Only forward ``hooks`` when the caller actually
                    # supplied providers. Passing ``hooks=None`` or
                    # ``hooks=[]`` risks being interpreted differently by
                    # future StrandsAgentCore versions (e.g. as "disable
                    # default hooks"), so we omit the kwarg entirely when
                    # there's nothing to forward.
                    core_kwargs = dict(self._agent_kwargs)
                    if self._hooks:
                        proxy_hook_boundary = _ProxyHookBoundary()
                        core_kwargs["hooks"] = [
                            _ProxyHookCaptureProvider(proxy_hook_boundary),
                            *self._hooks,
                        ]
                    core_agent = StrandsAgentCore(
                        model=self._model,
                        system_prompt=self._system_prompt,
                        tools=self._tools,
                        session_manager=session_manager,
                        **core_kwargs,
                    )
                    if self._hooks:
                        # Constructor-time AgentInitialized callbacks may add
                        # more BeforeToolCall callbacks. Append the finalizer
                        # only after construction so it remains last.
                        core_agent.hooks.add_hook(
                            _ProxyHookFinalizerProvider(proxy_hook_boundary)
                        )
                        self._proxy_hook_boundaries_by_thread[
                            thread_id
                        ] = proxy_hook_boundary
                    self._agents_by_thread[thread_id] = core_agent
        strands_agent = self._agents_by_thread[thread_id]

        resume_entries = getattr(input_data, "resume", None)
        has_resume_entries = bool(
            isinstance(resume_entries, list) and resume_entries
        )
        if has_resume_entries:
            resume_error = _preflight_resume_entries(strands_agent, resume_entries)
            if resume_error is not None:
                yield RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
                yield resume_error
                return

        # Read checkpoint provenance before proxy synchronization. A
        # ``BeforeToolCallEvent`` pause parks the exact native proxy invocation
        # for re-execution, so an omitted ``RunAgentInput.tools`` list must not
        # delete that marked proxy before Strands consumes its resume override.
        persisted_tool_call_meta: Dict[str, Dict[str, Any]] = {}
        _agent_state = getattr(strands_agent, "state", None)
        if _agent_state is not None:
            persisted_tool_call_meta = (
                _agent_state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY) or {}
            )
        provenance_error = _validate_active_proxy_hook_provenance(
            strands_agent, persisted_tool_call_meta
        )
        if provenance_error is not None:
            yield RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
            )
            yield provenance_error
            return
        pending_proxy_hook_native_ids = _pending_proxy_hook_native_ids(
            strands_agent, persisted_tool_call_meta
        )
        selected_proxy_hook_native_ids: set[str] = set()
        retained_checkpoint_proxy_names: set[str] = set()
        if has_resume_entries and pending_proxy_hook_native_ids:
            _present, _managed_ids, active_records = (
                _proxy_hook_provenance_value(strands_agent)
            )
            selected_resume_interrupt_ids = {
                entry.interrupt_id for entry in resume_entries
            }
            selected_resume_interrupt_ids.intersection_update(
                _unanswered_interrupt_ids(strands_agent._interrupt_state)
            )
            selected_proxy_hook_native_ids = {
                record["original_native_tool_call_id"]
                for interrupt_id, record in (active_records or {}).items()
                if interrupt_id in selected_resume_interrupt_ids
            }
            marked_proxy_names = registered_proxy_tool_names(
                strands_agent.tool_registry
            )
            incoming_proxy_names = {
                getattr(tool, "name", None)
                or (tool.get("name", "") if isinstance(tool, dict) else "")
                for tool in (input_data.tools or [])
            }
            incoming_proxy_tools = {
                getattr(tool, "name", None)
                or (tool.get("name", "") if isinstance(tool, dict) else ""):
                tool
                for tool in (input_data.tools or [])
            }
            unavailable_native_ids: set[str] = set()
            for native_id in pending_proxy_hook_native_ids:
                metadata = persisted_tool_call_meta.get(native_id)
                tool_name = (
                    metadata.get("name") if isinstance(metadata, dict) else None
                )
                prior_specs = {
                    json.dumps(record["tool_spec"], sort_keys=True)
                    for record in (active_records or {}).values()
                    if record["original_native_tool_call_id"] == native_id
                }
                incoming_tool = incoming_proxy_tools.get(tool_name)
                if incoming_tool is not None and (
                    len(prior_specs) != 1
                    or json.dumps(
                        _proxy_tool_spec_from_agui(incoming_tool),
                        sort_keys=True,
                    )
                    not in prior_specs
                ):
                    yield RunStartedEvent(
                        type=EventType.RUN_STARTED,
                        thread_id=input_data.thread_id,
                        run_id=input_data.run_id,
                    )
                    yield _proxy_hook_error({"tool_spec"})
                    return
                if tool_name in marked_proxy_names:
                    retained_checkpoint_proxy_names.add(tool_name)
                    continue
                if (
                    tool_name in incoming_proxy_names
                    and tool_name not in strands_agent.tool_registry.registry
                ):
                    # The current client declaration is a safe reconstruction
                    # source and ordinary synchronization will register it.
                    continue
                unavailable_native_ids.add(native_id)
            if unavailable_native_ids:
                yield RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
                yield _proxy_resume_tool_capability_error(
                    unavailable_native_ids
                )
                return

        # Forward ``RunAgentInput.context`` to the per-thread Strands agent's
        # state so user tools can read it (e.g. catalog/component schemas
        # injected by the CopilotKit FE for A2UI rendering). Mirrors the
        # langgraph integration where tools read ``runtime.state["copilotkit"]
        # ["context"]``. Stored as a plain list of ``{description, value}``
        # dicts to satisfy ``JSONSerializableDict`` validation.
        agui_context = []
        for ctx in (input_data.context or []):
            if isinstance(ctx, dict):
                agui_context.append(
                    {
                        "description": ctx.get("description", ""),
                        "value": ctx.get("value", ""),
                    }
                )
            else:
                agui_context.append(
                    {
                        "description": getattr(ctx, "description", "") or "",
                        "value": getattr(ctx, "value", "") or "",
                    }
                )
        try:
            strands_agent.state.set("agui_context", agui_context)
        except Exception as e:
            logger.warning(f"Failed to set agui_context on strands_agent.state: {e}")

        # Sync proxy tools from client-defined tools. Only exact marked proxies
        # required by the active hook checkpoint survive an omitted tools list;
        # the returned bookkeeping makes the next ordinary sync stale-delete
        # them once the checkpoint is consumed.
        tracked_proxy_names = self._proxy_tool_names_by_thread.get(
            thread_id, set()
        )
        if input_data.tools or tracked_proxy_names or retained_checkpoint_proxy_names:
            proxy_names = sync_proxy_tools(
                strands_agent.tool_registry,
                input_data.tools or [],
                tracked_proxy_names,
                retain_names=retained_checkpoint_proxy_names,
            )
            self._proxy_tool_names_by_thread[thread_id] = proxy_names

        # A2UI auto-injection. When the runtime forwards
        # ``injectA2UITool`` (or the host opts in via ``config.a2ui``), register
        # a ``generate_a2ui`` recovery tool bound to this agent's model and drop
        # the injected ``render_a2ui`` proxy so the model calls generate_a2ui
        # directly. Best-effort: a failure here logs and runs without A2UI
        # rather than crashing the turn.
        try:
            registry = strands_agent.tool_registry
            # Remove our OWN prior-turn auto-injected tool first, so (a) the
            # refreshed tool carries THIS turn's messages/state, and (b) the
            # USER-PREVAILS check only ever sees a dev-wired
            # generate_a2ui — not our own from a previous turn on this cached
            # agent. Without this, turn 2+ leaks the re-synced render_a2ui back
            # to the model.
            for name in [
                n for n, t in list(registry.registry.items())
                if is_auto_injected_a2ui_tool(t)
            ]:
                registry.registry.pop(name, None)
                getattr(registry, "dynamic_tools", {}).pop(name, None)
            # Lift the A2UI component schema + remaining context under
            # state["ag-ui"] so the generate_a2ui sub-agent prompt carries the
            # "## Available Components" block + context — same routing the
            # LangGraph adapter does in its state merge. Uses the shared toolkit
            # split so both adapters agree on the schema-context description.
            a2ui_schema_value, a2ui_regular_ctx = split_a2ui_schema_context(
                input_data.context
            )
            a2ui_state = (
                dict(input_data.state)
                if isinstance(input_data.state, dict)
                else {}
            )
            a2ui_ag_ui: dict = {"context": a2ui_regular_ctx}
            if a2ui_schema_value is not None:
                a2ui_ag_ui["a2ui_schema"] = a2ui_schema_value
            a2ui_state["ag-ui"] = a2ui_ag_ui

            a2ui_plan = plan_a2ui_injection(
                model=getattr(strands_agent, "model", None),
                input=input_data,
                existing_tool_names=list(registry.registry.keys()),
                config=self.config.a2ui,
                log=logger,
                strands_agent=strands_agent,
                agui_state=a2ui_state,
            )
            if a2ui_plan:
                # Register FIRST: if this raises, the except below degrades to
                # "render proxy leaks through" (middleware still paints,
                # unvalidated) instead of a turn with no A2UI path at all.
                registry.register_tool(a2ui_plan["tool"])
                for name in a2ui_plan["drop_tool_names"]:
                    registry.registry.pop(name, None)
                    getattr(registry, "dynamic_tools", {}).pop(name, None)
                    # Keep the proxy bookkeeping honest — the dropped render
                    # tool is no longer registered.
                    self._proxy_tool_names_by_thread.get(thread_id, set()).discard(name)
        except Exception as e:  # noqa: BLE001 — never crash the turn here
            # ERROR, not warning: the runtime explicitly requested injection
            # (injectA2UITool) and this turn runs without it.
            logger.error(
                "A2UI auto-injection failed; running without A2UI for this turn: %s",
                e,
                exc_info=True,
            )

        # Snapshot provenance only after proxy sync and A2UI registry edits.
        # The stream uses this immutable view rather than client declarations:
        # a declaration can collide with a native tool that sync correctly
        # preserves, and callbacks must not observe mid-stream registry changes.
        registered_frontend_tool_names = frozenset(
            registered_proxy_tool_names(strands_agent.tool_registry)
        )

        # Start run
        yield RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id,
        )

        try:
            # Detect delta-only payloads (where the client sent fewer
            # messages than the session has — e.g. only the trailing
            # tool result, or only the new user message in a continued
            # chat). CopilotKit V2's MESSAGES_SNAPSHOT handler treats
            # the snapshot as authoritative: any existing client message
            # whose id is not in the snapshot gets dropped. Emitting a
            # partial snapshot on a delta payload would wipe prior turns
            # from the UI. The frontend already has the full history with
            # the original ids, so we suppress snapshot emission for this
            # run and let TEXT_MESSAGE_*/TOOL_CALL_* streaming events
            # reconcile naturally.
            session_msgs = getattr(strands_agent, "messages", None) or []
            is_delta_payload = (
                bool(session_msgs)
                and len(session_msgs) > len(input_data.messages or [])
            )
            emit_snapshots = (
                self.config.emit_messages_snapshot and not is_delta_payload
            )

            # Seed the running ``MessagesSnapshotEvent`` payload from the
            # full conversation history sent by the client. Each emitted
            # snapshot then carries prior turns + whatever this turn adds.
            snapshot_messages: List[Any] = (
                _build_snapshot_messages(input_data.messages)
                if emit_snapshots
                else []
            )

            # Emit state snapshot if provided
            if hasattr(input_data, "state") and input_data.state is not None:
                # Filter out messages from state to avoid "Unknown message role" errors
                # The frontend manages messages separately and doesn't recognize "tool" role
                state_snapshot = {
                    k: v for k, v in input_data.state.items() if k != "messages"
                }
                yield StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT, snapshot=state_snapshot
                )

            # Splice point 1 of 4: emit the initial messages snapshot right
            # after ``RunStartedEvent`` / ``StateSnapshotEvent`` so the
            # frontend can render the seeded thread before any new content
            # streams in.
            if emit_snapshots and snapshot_messages:
                yield MessagesSnapshotEvent(
                    type=EventType.MESSAGES_SNAPSHOT,
                    messages=list(snapshot_messages),
                )

            # Extract frontend tool names from input_data.tools
            frontend_tool_names = set()
            if input_data.tools:
                for tool_def in input_data.tools:
                    tool_name = (
                        tool_def.get("name")
                        if isinstance(tool_def, dict)
                        else getattr(tool_def, "name", None)
                    )
                    if tool_name:
                        frontend_tool_names.add(tool_name)

            # Collect tool_call_ids that already have results in the message history
            # so we suppress duplicate TOOL_CALL_START events only for those specific calls
            pending_tool_result_ids: set[str] = set()
            if input_data.messages:
                for msg in reversed(input_data.messages):
                    if msg.role == "tool":
                        tool_call_id = getattr(msg, "tool_call_id", None)
                        if tool_call_id:
                            pending_tool_result_ids.add(tool_call_id)
                    else:
                        break
                if pending_tool_result_ids:
                    logger.debug(
                        f"Has pending tool results detected: tool_call_ids={pending_tool_result_ids}, thread_id={input_data.thread_id}"
                    )

            # Convert AG-UI messages to Strands format
            # Strands expects content as List[ContentBlock] for most messages
            # OpenAI requires tool messages to follow assistant messages with tool_calls
            strands_messages = []
            last_msg_had_tool_calls = False
            expected_tool_call_ids = set()  # Track which tool_call_ids are valid

            logger.debug(
                f"Converting {len(input_data.messages)} messages to Strands format, thread_id={input_data.thread_id}"
            )

            for i, msg in enumerate(input_data.messages):
                logger.debug(
                    f"Message {i}: role={msg.role}, has_tool_calls={hasattr(msg, 'tool_calls') and bool(msg.tool_calls)}, tool_call_id={getattr(msg, 'tool_call_id', None)}"
                )
                strands_msg: Dict[str, Any] = {"role": msg.role}

                # Handle assistant messages with tool_calls
                if (
                    msg.role == "assistant"
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
                    # Convert tool calls to format expected by Strands/OpenAI
                    strands_msg["content"] = []
                    if msg.content:
                        if isinstance(msg.content, str):
                            strands_msg["content"].append({"text": msg.content})
                        elif isinstance(msg.content, list):
                            strands_msg["content"] = msg.content

                    strands_msg["tool_calls"] = []
                    expected_tool_call_ids.clear()  # Reset for this assistant message
                    for tc in msg.tool_calls:
                        expected_tool_call_ids.add(tc.id)  # Track this tool call ID
                        strands_msg["tool_calls"].append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.get("name")
                                    if isinstance(tc.function, dict)
                                    else tc.function.name,
                                    "arguments": tc.function.get("arguments")
                                    if isinstance(tc.function, dict)
                                    else tc.function.arguments,
                                },
                            }
                        )
                    last_msg_had_tool_calls = True
                    strands_messages.append(strands_msg)

                # Handle tool messages (must follow assistant message with tool_calls)
                elif msg.role == "tool":
                    # Skip tool messages that don't have a preceding assistant message
                    # with tool_calls — UNLESS this is a pending frontend tool result
                    # (delta-only payloads only contain the tool result, so the
                    # assistant message is absent but the result is still valid).
                    is_pending_frontend_result = (
                        msg.tool_call_id in pending_tool_result_ids
                    )
                    if (
                        not last_msg_had_tool_calls
                        or msg.tool_call_id not in expected_tool_call_ids
                    ) and not is_pending_frontend_result:
                        logger.debug(
                            f"Skipping orphaned tool message: tool_call_id={msg.tool_call_id}, last_msg_had_tool_calls={last_msg_had_tool_calls}, valid_ids={expected_tool_call_ids}, thread_id={input_data.thread_id}"
                        )
                        continue

                    # Include the tool message for OpenAI format compliance
                    strands_msg["tool_call_id"] = msg.tool_call_id
                    if isinstance(msg.content, str):
                        strands_msg["content"] = [{"text": msg.content}]
                    else:
                        strands_msg["content"] = msg.content

                    expected_tool_call_ids.discard(msg.tool_call_id)
                    if not expected_tool_call_ids:
                        last_msg_had_tool_calls = False
                    strands_messages.append(strands_msg)

                # Handle regular messages (user, assistant without tool_calls)
                else:
                    if isinstance(msg.content, str):
                        strands_msg["content"] = [{"text": msg.content}]
                    elif isinstance(msg.content, list):
                        strands_msg["content"] = msg.content
                    else:
                        strands_msg["content"] = [{"text": ""}]
                    last_msg_had_tool_calls = False
                    strands_messages.append(strands_msg)

            # Build a lookup of tool_call_id -> tool_name from the input messages
            # directly (the assistant message in Run 2 already carries the name).
            _tool_call_id_to_name: dict = {}
            for _msg in (input_data.messages or []):
                if _msg.role == "assistant" and hasattr(_msg, "tool_calls") and _msg.tool_calls:
                    for tc in _msg.tool_calls:
                        tc_name = tc.function.get("name") if isinstance(tc.function, dict) else tc.function.name
                        if tc.id and tc_name:
                            _tool_call_id_to_name[tc.id] = tc_name

            # On delta-only continuation payloads, the assistant message that
            # carries the tool_call is absent from input_data.messages, so the
            # lookup above misses. The session manager still holds the full
            # native history — scan its ``toolUse`` blocks so we resolve the
            # tool that actually executed rather than guessing.
            for _smsg in session_msgs:
                if not isinstance(_smsg, dict) or _smsg.get("role") != "assistant":
                    continue
                for _block in (_smsg.get("content") or []):
                    tool_use = _block.get("toolUse") if isinstance(_block, dict) else None
                    if tool_use:
                        tu_id = tool_use.get("toolUseId")
                        tu_name = tool_use.get("name")
                        if tu_id and tu_name and tu_id not in _tool_call_id_to_name:
                            _tool_call_id_to_name[tu_id] = tu_name

            # Get the latest user message for state context builder.
            # For continuation runs (has_pending_tool_result), derive a meaningful
            # message from the frontend tool that was just executed so the agent
            # understands the context and can generate a proper conclusion.
            user_message = ""
            if pending_tool_result_ids and input_data.messages:
                # Collect ALL trailing tool results (not just the first). A parallel
                # frontend-tool turn sends N results in one continuation run; the model
                # must see every answer.
                _result_parts: list[str] = []
                for msg in reversed(input_data.messages):
                    if msg.role == "tool" and hasattr(msg, "tool_call_id"):
                        tool_name = _tool_call_id_to_name.get(msg.tool_call_id)
                        if tool_name and tool_name in frontend_tool_names:
                            # Forward the ACTUAL result so the model can act on the
                            # human's decision (e.g. an approval resolving to
                            # {"approved": false}). Hardcoding a success string here
                            # silently breaks HITL — the model would be told the tool
                            # "executed successfully with no return value" regardless
                            # of what the human returned. Only use that synthetic
                            # acknowledgement when the result is genuinely empty.
                            result_text = (
                                msg.content
                                if isinstance(msg.content, str)
                                else flatten_content_to_text(msg.content)
                            )
                            result = _normalize_frontend_tool_result(
                                result_text or "", getattr(msg, "error", None)
                            )
                            provider_text = result.provider_safe_content
                            if result.status == "error":
                                _result_parts.append(
                                    f"{tool_name} failed"
                                    + (
                                        f": {provider_text}"
                                        if provider_text.strip()
                                        else "."
                                    )
                                )
                            elif provider_text.strip():
                                _result_parts.append(
                                    f"{tool_name} returned: {provider_text}"
                                )
                            else:
                                _result_parts.append(
                                    f"{tool_name} executed successfully with no return value."
                                )
                        else:
                            # Could not resolve this tool's name from input messages
                            # or session history (e.g. a delta-only payload with no
                            # assistant tool_calls). Skip it rather than guessing:
                            # picking an arbitrary frontend tool would feed false
                            # context to the LLM when several frontend tools exist.
                            # Strands still has the real result in session history to
                            # conclude the round-trip from.
                            logger.warning(
                                f"Could not resolve tool name for tool_call_id={msg.tool_call_id} "
                                f"from input messages or session history (delta-only payload). "
                                f"Skipping this tool result in the continuation message."
                            )
                    else:
                        break
                user_message = "\n".join(reversed(_result_parts))
            elif input_data.messages:
                for msg in reversed(input_data.messages):
                    if (msg.role == "user" or msg.role == "tool") and msg.content:
                        if isinstance(msg.content, list):
                            has_media = any(
                                getattr(item, "type", None) in ("image", "audio", "video", "document")
                                for item in msg.content
                            )
                            if has_media:
                                user_message = convert_agui_content_to_strands(msg.content)
                                if not user_message:
                                    # All content blocks failed conversion — fall back to text
                                    user_message = flatten_content_to_text(msg.content) or ""
                                    logger.warning("All media content blocks failed conversion, falling back to text")
                            else:
                                user_message = flatten_content_to_text(msg.content)
                        else:
                            user_message = msg.content
                        break

            # Optionally allow configuration to adjust the outgoing user message
            if self.config.state_context_builder:
                try:
                    text_for_builder = flatten_content_to_text(user_message) if isinstance(user_message, list) else user_message
                    builder_result = self.config.state_context_builder(
                        input_data, text_for_builder
                    )
                    if not isinstance(user_message, list):
                        user_message = builder_result
                    else:
                        logger.debug("state_context_builder result not applied to multimodal message — multimodal content preserved")
                    # If state_context_builder modifies the message, update the last user message
                    if not isinstance(user_message, list) and strands_messages and strands_messages[-1]["role"] == "user":
                        strands_messages[-1]["content"] = [{"text": user_message}]
                except Exception as e:
                    # If the builder fails, keep the original message
                    logger.warning(f"State context builder failed: {e}", exc_info=True)

            # Generate unique message ID
            message_id = str(uuid.uuid4())
            message_started = False
            accumulated_text = ""
            # Tracks the latest assistant text id that was actually emitted on
            # the wire. Tool calls use it only when no snapshot will expose the
            # tool-call AssistantMessage id.
            last_emitted_text_message_id: str | None = None
            tool_calls_seen = {}
            current_state = dict(input_data.state or {})  # Track state for final snapshot
            stop_text_streaming = False
            halt_event_stream = False
            pending_halt = False
            # Frontend-tool ToolCallEnd ids are buffered here so the client's
            # "execute this frontend tool" signal is delayed until AFTER this
            # turn's backend tool results have been emitted. This prevents the
            # client dispatching its follow-up run before the backend results
            # reach it, narrowing the ConcurrencyException race window.
            deferred_frontend_tool_ends = []
            # Native ``toolUseId``s whose ``toolResult`` was processed this
            # run. Drained after each result batch to prune the persisted
            # tool-call meta map.
            processed_result_native_ids: set[str] = set()
            # Terminal ``AgentResult`` from Strands (carried on the final
            # ``{"result": ...}`` stream event). Used after the loop to detect a
            # native interrupt pause (``stop_reason == "interrupt"``).
            terminal_result = None
            proxy_boundary_error: RunErrorEvent | None = None

            # Reasoning/thinking state tracking
            reasoning_started = False
            reasoning_message_id = None

            logger.debug(
                f"Starting agent run: thread_id={input_data.thread_id}, run_id={input_data.run_id}, pending_tool_result_ids={pending_tool_result_ids}, message_count={len(input_data.messages)}, strands_message_count={len(strands_messages)}"
            )

            # Collect the real results the client produced for proxied
            # frontend tools. These arrive in ``RunAgentInput.messages`` on the
            # continuation run and are used to reconcile the session-persisted
            # "Forwarded to client" placeholder. A tool result is a frontend
            # result when its tool name is client-declared, or (for delta-only
            # payloads that omit the assistant message) when its wire id was
            # recorded in the wire->native map when the call was emitted.
            session_manager = _get_strands_session_manager(strands_agent)
            # The durable per-``toolUseId`` call metadata map recorded at
            # emission (see the ``current_tool_use`` handler). On a RESUME
            # run this is the ONLY source of ``{name, args, input,
            # strands_tool_id}`` for the interrupted tool, since Strands does
            # not re-emit ``current_tool_use`` events for it. It was loaded
            # before proxy sync so the same provenance can also protect the
            # exact checkpoint-required marked proxy from stale deletion.
            active_proxy_native_ids = active_proxy_placeholder_ids(
                strands_agent, persisted_tool_call_meta
            )
            if active_proxy_native_ids:
                if session_manager is None:
                    yield _interrupt_session_required_error()
                    return
                if not _supports_repository_reconciliation(session_manager):
                    yield _interrupt_session_capability_error()
                    return
            if session_manager is None and pending_proxy_hook_native_ids:
                yield _interrupt_session_required_error()
                return

            # The durable wire->native map recorded at emission, read back from
            # session state (restored from the store on a fresh process).
            wire_to_native: Dict[str, str] = {}
            if session_manager is not None:
                wire_to_native = (
                    strands_agent.state.get(AG_UI_WIRE_MAP_STATE_KEY) or {}
                )
            original_wire_native_ids = frozenset(wire_to_native.values())
            # Scope to the TRAILING tool results (this continuation's just-
            # returned results). ``pending_tool_result_ids`` holds those ids;
            # without this, a multi-turn continuation re-sends already-reconciled
            # historical results, which can never be re-corrected and would force
            # the legacy fallback every turn.
            frontend_results: Dict[str, _FrontendToolResult] = {}
            for msg in (input_data.messages or []):
                if getattr(msg, "role", None) != "tool":
                    continue
                wire_id = getattr(msg, "tool_call_id", None)
                if not wire_id or wire_id not in pending_tool_result_ids:
                    continue
                name = _tool_call_id_to_name.get(wire_id)
                if name not in frontend_tool_names and wire_id not in wire_to_native:
                    continue
                content = msg.content
                text = (
                    content
                    if isinstance(content, str)
                    else flatten_content_to_text(content)
                )
                frontend_results[wire_id] = _normalize_frontend_tool_result(
                    text or "", getattr(msg, "error", None)
                )

            # Translate the client's wire tool_call_id back to the native
            # toolUseId Strands persisted (they differ for frontend tools — see
            # the fresh-uuid assignment in the streaming loop). Only reconcile
            # when there is at least one NON-EMPTY frontend result: a void tool
            # returns nothing, and the synthetic "executed successfully with no
            # return value" continuation message conveys that better than an
            # empty toolResult. When reconciling, void placeholders in the same
            # turn are still cleared (to "") so the literal "Forwarded to client"
            # is never fed to the model.
            resolved_native_results: Dict[str, _FrontendToolResult] = {}
            corrected_native_ids: set[str] = set()
            has_nonvoid_frontend_result = any(
                not result.is_void for result in frontend_results.values()
            )
            if (
                session_manager is not None
                and (
                    self.config.replay_history_into_strands
                    or (has_resume_entries and active_proxy_native_ids)
                )
            ) or selected_proxy_hook_native_ids:
                resolved_native_results = resolve_native_ids(
                    wire_to_native, frontend_results
                )

            # A before-tool hook pauses before the proxy can create a
            # placeholder, so ordinary repository reconciliation has nothing
            # to overwrite. Require the client's visible wire result before
            # resuming; the proxy consumes this map when Strands invokes it.
            proxy_resume_results: Dict[str, _FrontendToolResult] = {}
            proxy_resume_native_ids: set[str] = set()
            if selected_proxy_hook_native_ids:
                missing_hook_results = (
                    selected_proxy_hook_native_ids
                    - resolved_native_results.keys()
                )
                if missing_hook_results:
                    error = ActiveInterruptReconciliationError(missing_hook_results)
                    logger.error(
                        "Active proxy hook interrupt is missing mapped frontend "
                        f"results for native ids {sorted(missing_hook_results)}"
                    )
                    yield RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=str(error),
                        code="INTERRUPT_RECONCILIATION_ERROR",
                    )
                    return
                proxy_resume_results = {
                    native_id: resolved_native_results[native_id]
                    for native_id in selected_proxy_hook_native_ids
                }
                proxy_resume_native_ids = set(proxy_resume_results)

            # A native resume consumes and clears the active interrupt context.
            # Refuse to enter Strands unless every exact, provenance-backed
            # proxy placeholder parked there has a client result mapped back to
            # its native id. This validation must precede reconciliation so an
            # incomplete batch cannot partially mutate the retry checkpoint.
            if has_resume_entries and active_proxy_native_ids:
                missing_active_results = (
                    active_proxy_native_ids - resolved_native_results.keys()
                )
                if missing_active_results:
                    error = ActiveInterruptReconciliationError(
                        missing_active_results
                    )
                    logger.error(
                        "Active interrupt is missing mapped frontend results for "
                        f"native ids {sorted(missing_active_results)}"
                    )
                    yield RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=str(error),
                        code="INTERRUPT_RECONCILIATION_ERROR",
                    )
                    return

            # Reconcile Strands' internal conversation history with
            # ``RunAgentInput.messages``. Without this, frontend tool results
            # sent by the client never reach the LLM — Strands sees an open
            # ``toolUse`` from the prior turn and the LLM re-fires the same tool
            # every run, producing the "chart loops forever" symptom.
            #
            # No session manager: rebuild history in-memory and stream it.
            # With a session manager (which owns persistence): overwrite the
            # persisted placeholder toolResult(s) with the real client result
            # via the session repository, then continue from the corrected
            # native history — keeping a single source of truth rather than a
            # placeholder plus a synthetic "tool returned: X" message.
            replay_history = (
                self.config.replay_history_into_strands and session_manager is None
            )
            # An exact active proxy placeholder may be stashed in
            # ``_interrupt_state.context["tool_results"]``; reconcile even when
            # this turn's frontend result is void, since that stash has no other
            # correction path and is destroyed once the interrupt resumes. A
            # native-only active interrupt needs no repository access.
            has_active_interrupt = bool(
                getattr(getattr(strands_agent, "_interrupt_state", None), "activated", False)
            )
            reconcile_session_results = _supports_repository_reconciliation(
                session_manager
            ) and (
                (
                    self.config.replay_history_into_strands
                    and (has_nonvoid_frontend_result or bool(active_proxy_native_ids))
                )
                or (has_resume_entries and bool(active_proxy_native_ids))
            )

            # Default prompt: the legacy path, passing only the latest user
            # message and trusting Strands (via session_manager) to track
            # history. Each branch below may narrow this further; a resume run
            # can carry BOTH a fresh frontend tool result and an interrupt
            # response in the same batch, so the resume-entries translation
            # below runs unconditionally after the other branches and layers
            # on top, rather than short-circuiting them.
            resume_prompt: str | List[Dict[str, Any]] | list[InterruptResponseContent] | None = user_message
            if replay_history:
                native_history = _build_strands_history(input_data.messages)
                # Apply ``state_context_builder`` to the last user-text
                # message in the reconciled history rather than to the
                # synthetic ``user_message`` string. This matches what the
                # builder is actually trying to enrich (the prompt the LLM
                # will see).
                if self.config.state_context_builder and native_history:
                    for native_msg in reversed(native_history):
                        if (
                            native_msg.get("role") == "user"
                            and native_msg.get("content")
                            and isinstance(native_msg["content"], list)
                            and "text" in native_msg["content"][0]
                        ):
                            try:
                                augmented = self.config.state_context_builder(
                                    input_data, native_msg["content"][0]["text"]
                                )
                                if isinstance(augmented, str):
                                    native_msg["content"][0]["text"] = augmented
                            except Exception as e:
                                logger.warning(
                                    f"state_context_builder failed: {e}", exc_info=True
                                )
                            break
                # A live native interrupt already has the authoritative
                # pre-interrupt conversation in this cached core. Resume-only
                # clients commonly omit that history; replacing it with the
                # shorter delta would leave Strands to append the resumed
                # toolResult without its user/toolUse prefix. A full client
                # history is still authoritative and keeps the replacement
                # behavior used by ordinary in-memory replay.
                preserve_live_interrupt_history = (
                    has_resume_entries and has_active_interrupt and is_delta_payload
                )
                if not preserve_live_interrupt_history:
                    strands_agent.messages = native_history
                # ``None`` tells Strands to use existing ``self.messages`` as-is.
                # The LLM sees real tool results (including ones produced by the
                # frontend) and emits a proper follow-up turn instead of
                # re-calling the tool.
                resume_prompt = None
            elif reconcile_session_results:
                try:
                    corrected_native_ids = reconcile_frontend_tool_results(
                        session_manager, strands_agent, resolved_native_results
                    )
                except ActiveInterruptReconciliationError as e:
                    logger.error(
                        "Active interrupt tool result reconciliation failed for "
                        f"native ids {sorted(e.affected_native_ids)}",
                        exc_info=True,
                    )
                    yield RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=str(e),
                        code="INTERRUPT_RECONCILIATION_ERROR",
                    )
                    return
                except Exception as e:  # noqa: BLE001 — degrade, don't crash the turn
                    if has_active_interrupt:
                        logger.error(
                            "Active interrupt tool result reconciliation failed",
                            exc_info=True,
                        )
                        yield RunErrorEvent(
                            type=EventType.RUN_ERROR,
                            message=str(e),
                            code="INTERRUPT_RECONCILIATION_ERROR",
                        )
                        return
                    logger.warning(
                        "Frontend tool result reconciliation failed; falling back to "
                        f"the legacy continuation path: {e}",
                        exc_info=True,
                )
                if active_proxy_native_ids - corrected_native_ids:
                    missing_corrections = (
                        active_proxy_native_ids - corrected_native_ids
                    )
                    error = ActiveInterruptReconciliationError(missing_corrections)
                    logger.error(
                        "Active interrupt frontend results were not corrected for "
                        f"native ids {sorted(missing_corrections)}"
                    )
                    yield RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=str(error),
                        code="INTERRUPT_RECONCILIATION_ERROR",
                    )
                    return
                # Continue from the corrected native history only when every
                # NON-EMPTY frontend result this turn resolved to a native id
                # (i.e. was present in the wire->native map) AND none of those
                # placeholders remain uncleared. The scan is scoped to this
                # turn's results so a stale placeholder from a prior (e.g. void)
                # turn doesn't force the legacy path. Any shortfall means
                # forwarding the real result as a synthetic user message is
                # safer than replaying a stub.
                non_void_results = [
                    result
                    for result in frontend_results.values()
                    if not result.is_void
                ]
                resolved_non_void = {
                    native
                    for native, result in resolved_native_results.items()
                    if not result.is_void
                }
                all_non_void_resolved = len(resolved_non_void) == len(non_void_results)
                # Scan all of this turn's resolved native ids (void included, so a
                # resolved-but-uncleared void placeholder also blocks) — but not
                # unrelated historical placeholders.
                reconciled = all_non_void_resolved and not has_placeholder_results(
                    getattr(strands_agent, "messages", None) or [],
                    only_ids=set(resolved_native_results),
                )
                resume_prompt = None if reconciled else user_message

            # A client answering to an interrupt sends its responses
            # in ``RunAgentInput.resume`` (as per the AG-UI interrupt round-trip),
            # not as a new user message. Translate those into the Strands resume
            # prompt shape ``[{"interruptResponse": {"interruptId", "response"}}]``
            # and drive the stream with it — this runs after (and takes
            # precedence over) every branch above, since a resume batch may
            # still carry a fresh frontend tool result that needed reconciling.
            if has_resume_entries:
                resume_prompt = [
                    {
                        "interruptResponse": {
                            "interruptId": entry.interrupt_id,
                            "response": _wrap_resume_response(entry.status, entry.payload),
                        }
                    }
                    for entry in resume_entries
                ]
            # Drop only the entries whose placeholder was actually corrected
            # this turn — they won't recur. Entries that were NOT corrected
            # (unresolved, or a reconcile that raised) are kept so a later turn
            # can retry; pruning them would strand the persisted placeholder
            # forever. (Genuinely-abandoned entries are bounded by the size cap
            # applied at emission.)
            if wire_to_native and corrected_native_ids:
                remaining = {
                    wire: native
                    for wire, native in wire_to_native.items()
                    if native not in corrected_native_ids
                }
                if len(remaining) != len(wire_to_native):
                    strands_agent.state.set(AG_UI_WIRE_MAP_STATE_KEY, remaining)

            proxy_hook_boundary = self._proxy_hook_boundaries_by_thread.get(
                thread_id
            )
            if proxy_hook_boundary is not None:
                proxy_hook_boundary.prepare_run(
                    strands_agent, proxy_resume_results
                )

            if proxy_resume_results:
                agent_stream = strands_agent.stream_async(
                    resume_prompt,
                    invocation_state={
                        PROXY_RESUME_RESULTS_KEY: proxy_resume_results,
                    },
                )
            else:
                agent_stream = strands_agent.stream_async(resume_prompt)
            try:
                async for event in agent_stream:
                    # Capture the terminal ``AgentResult`` (always emitted last
                    # by ``stream_async``) so a native interrupt pause can be
                    # detected after the loop. Recorded first so it is never
                    # dropped, even on the halt-event-stream break below.
                    if "result" in event and event["result"] is not None:
                        terminal_result = event["result"]

                    if event.get("tool_interrupt_event"):
                        proxy_hook_boundary = (
                            self._proxy_hook_boundaries_by_thread.get(thread_id)
                        )
                        payload = event["tool_interrupt_event"]
                        raw_interrupts = payload.get("interrupts") or []
                        boundary_ids = {
                            getattr(interrupt, "id", None)
                            for interrupt in raw_interrupts
                            if isinstance(getattr(interrupt, "id", None), str)
                        }
                        owned_ids = (
                            boundary_ids.intersection(
                                proxy_hook_boundary.claims
                            )
                            if proxy_hook_boundary is not None
                            else set()
                        )
                        if proxy_hook_boundary is not None:
                            proxy_hook_boundary.promote_interrupts(
                                strands_agent, owned_ids
                            )
                        present, managed_ids, provenance = (
                            _proxy_hook_provenance_value(strands_agent)
                        )
                        malformed = bool(
                            owned_ids
                            and present
                            and (
                                managed_ids is None or provenance is None
                            )
                        )
                        missing = bool(
                            owned_ids
                            and any(
                                interrupt_id
                                not in (managed_ids or frozenset())
                                or interrupt_id not in (provenance or {})
                                for interrupt_id in owned_ids
                            )
                        )
                        binding_error = (
                            _validate_proxy_hook_record_bindings(
                                strands_agent,
                                strands_agent.state.get(
                                    AG_UI_TOOL_CALL_MAP_STATE_KEY
                                )
                                or {},
                                {
                                    interrupt_id: record
                                    for interrupt_id, record in (
                                        provenance or {}
                                    ).items()
                                    if interrupt_id
                                    in _unanswered_interrupt_ids(
                                        strands_agent._interrupt_state
                                    )
                                },
                            )
                            if (
                                owned_ids
                                and provenance is not None
                                and session_manager is not None
                            )
                            else None
                        )
                        if (
                            malformed
                            or missing
                            or binding_error is not None
                            or (
                                proxy_hook_boundary is not None
                                and proxy_hook_boundary.failed
                            )
                        ):
                            changed = (
                                proxy_hook_boundary.failure_fields
                                if proxy_hook_boundary is not None
                                else set()
                            )
                            proxy_boundary_error = _proxy_hook_error(changed)
                            deferred_frontend_tool_ends = []
                            if (
                                proxy_hook_boundary is not None
                                and (
                                    malformed
                                    or missing
                                    or binding_error is not None
                                )
                            ):
                                proxy_hook_boundary.failure_fields.add(
                                    "provenance_state"
                                )
                                proxy_hook_boundary.failed = True
                        elif (
                            session_manager is None
                            and bool(
                                owned_ids.intersection(
                                    managed_ids or frozenset()
                                )
                            )
                        ):
                            proxy_boundary_error = (
                                _interrupt_session_required_error()
                            )
                            deferred_frontend_tool_ends = []
                            if proxy_hook_boundary is not None:
                                proxy_hook_boundary.failed = True
                        continue

                    # Frontend-tool halt: STOP the loop rather than muting the
                    # wire and draining it. The proxy tool returns a SUCCESSFUL
                    # "Forwarded to client" placeholder, so Strands has every
                    # reason to run another model cycle — and another. Draining
                    # those cycles costs: frontend tool calls the client never
                    # sees (so it can never answer them), real backend tool
                    # side effects, phantom assistant turns persisted to the
                    # session store, and RUN_FINISHED stuck behind work the
                    # client is not watching. Single-agent Strands has no cycle
                    # cap, so that tail is unbounded — a model that keeps
                    # retrying the read never yields a terminal event at all.
                    #
                    # Safe here because the halt latches only AFTER Strands
                    # appended the assistant toolUse + placeholder toolResult
                    # and MessageAddedEvent synced agent state (see
                    # SessionManager.register_hooks), so the next run's
                    # reconcile still finds a placeholder to overwrite and the
                    # wire->native map to key it by.
                    if halt_event_stream:
                        break

                    logger.debug(f"Received event: {event}")

                    # Skip lifecycle events
                    if event.get("init_event_loop") or event.get("start_event_loop"):
                        continue
                    if event.get("complete") or event.get("force_stop"):
                        logger.debug(
                            f"Breaking event stream: received complete or force_stop event (thread_id={input_data.thread_id}, complete={event.get('complete')}, force_stop={event.get('force_stop')})"
                        )
                        # Generator will end naturally, no need to break
                        break

                    # Handle text streaming
                    if "data" in event and event["data"]:
                        if stop_text_streaming:
                            continue

                        if not message_started:
                            yield TextMessageStartEvent(
                                type=EventType.TEXT_MESSAGE_START,
                                message_id=message_id,
                                role="assistant",
                            )
                            message_started = True
                            last_emitted_text_message_id = message_id

                        text_chunk = str(event["data"])
                        accumulated_text += text_chunk
                        yield TextMessageContentEvent(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            message_id=message_id,
                            delta=text_chunk,
                        )

                    # Handle reasoning/thinking text streaming
                    elif "reasoningText" in event and event.get("reasoning"):
                        reasoning_text = event["reasoningText"]

                        if not reasoning_started:
                            reasoning_message_id = str(uuid.uuid4())

                            # Emit reasoning events
                            yield ReasoningStartEvent(
                                type=EventType.REASONING_START,
                                message_id=reasoning_message_id
                            )
                            yield ReasoningMessageStartEvent(
                                type=EventType.REASONING_MESSAGE_START,
                                message_id=reasoning_message_id,
                                role="reasoning"
                            )
                            reasoning_started = True

                        # Stream reasoning content
                        if reasoning_text:
                            yield ReasoningMessageContentEvent(
                                type=EventType.REASONING_MESSAGE_CONTENT,
                                message_id=reasoning_message_id,
                                delta=reasoning_text
                            )

                    # Handle encrypted/redacted reasoning content
                    elif "reasoningRedactedContent" in event and event.get("reasoning"):
                        redacted_content = event["reasoningRedactedContent"]

                        if redacted_content is None:
                            logger.debug(f"Ignoring reasoning event with None redacted content (thread_id={input_data.thread_id})")
                            continue

                        if not reasoning_started:
                            reasoning_message_id = str(uuid.uuid4())
                            yield ReasoningStartEvent(
                                type=EventType.REASONING_START,
                                message_id=reasoning_message_id
                            )
                            yield ReasoningMessageStartEvent(
                                type=EventType.REASONING_MESSAGE_START,
                                message_id=reasoning_message_id,
                                role="reasoning"
                            )
                            reasoning_started = True

                        # Encode bytes to base64 string for transport
                        if isinstance(redacted_content, bytes):
                            encrypted_value = base64.b64encode(redacted_content).decode()
                        elif isinstance(redacted_content, str):
                            encrypted_value = redacted_content
                        else:
                            logger.warning(f"Unexpected type for reasoningRedactedContent: {type(redacted_content)}, converting to str")
                            encrypted_value = str(redacted_content)

                        yield ReasoningEncryptedValueEvent(
                            type=EventType.REASONING_ENCRYPTED_VALUE,
                            subtype="message",
                            entity_id=reasoning_message_id,
                            encrypted_value=encrypted_value
                        )

                    # Handle reasoning signature (verification token) - typically not exposed to UI
                    elif "reasoning_signature" in event and event.get("reasoning"):
                        sig = event.get("reasoning_signature", "")
                        logger.debug(f"Received reasoning signature: {str(sig)[:20]}...")

                    # Handle multi-agent node start (maps to STEP_STARTED)
                    elif isinstance(event, dict) and event.get("type") == "multiagent_node_start":
                        node_id = event.get("node_id", "unknown")
                        node_type = event.get("node_type", "agent")
                        yield StepStartedEvent(
                            type=EventType.STEP_STARTED,
                            step_name=f"{node_type}:{node_id}"
                        )

                    # Handle multi-agent node stop (maps to STEP_FINISHED)
                    elif isinstance(event, dict) and event.get("type") == "multiagent_node_stop":
                        node_id = event.get("node_id", "unknown")
                        node_type = event.get("node_type", "agent")
                        yield StepFinishedEvent(
                            type=EventType.STEP_FINISHED,
                            step_name=f"{node_type}:{node_id}"
                        )

                    # Handle multi-agent handoff (emit as CUSTOM event)
                    elif isinstance(event, dict) and event.get("type") == "multiagent_handoff":
                        yield CustomEvent(
                            type=EventType.CUSTOM,
                            name="MultiAgentHandoff",
                            value={
                                "from_nodes": event.get("from_node_ids", []),
                                "to_nodes": event.get("to_node_ids", []),
                                "message": event.get("message")
                            }
                        )

                    # Handle tool streaming events for real-time state updates
                    # Strands tools can yield intermediate results as tool_stream_event
                    elif "tool_stream_event" in event:
                        tool_stream = event["tool_stream_event"]
                        stream_data = tool_stream.get("data", {})

                        # Emit state snapshot if tool yielded state
                        if isinstance(stream_data, dict) and "state" in stream_data:
                            yield StateSnapshotEvent(
                                type=EventType.STATE_SNAPSHOT,
                                snapshot=stream_data["state"],
                            )
                        # A2UI sub-agent streaming: re-emit the
                        # generate_a2ui tool's inner render_a2ui progress as
                        # synthetic TOOL_CALL events. The a2ui middleware's
                        # streaming path keys its "building" skeleton +
                        # progressive paint off these — without them the
                        # surface only paints in bulk from the final result.
                        elif (
                            isinstance(stream_data, dict)
                            and isinstance(stream_data.get(A2UI_STREAM_KEY), dict)
                        ):
                            a2ui_ev = stream_data[A2UI_STREAM_KEY]
                            kind = a2ui_ev.get("kind")
                            a2ui_call_id = a2ui_ev.get("tool_call_id", "")
                            if kind == "start":
                                yield ToolCallStartEvent(
                                    type=EventType.TOOL_CALL_START,
                                    tool_call_id=a2ui_call_id,
                                    tool_call_name=a2ui_ev.get(
                                        "tool_call_name", "render_a2ui"
                                    ),
                                )
                            elif kind == "args" and a2ui_ev.get("delta"):
                                yield ToolCallArgsEvent(
                                    type=EventType.TOOL_CALL_ARGS,
                                    tool_call_id=a2ui_call_id,
                                    delta=a2ui_ev["delta"],
                                )
                            elif kind == "end":
                                yield ToolCallEndEvent(
                                    type=EventType.TOOL_CALL_END,
                                    tool_call_id=a2ui_call_id,
                                )

                    # Handle tool results from Strands for backend tool rendering
                    elif "message" in event and event["message"].get("role") == "user":
                        # A deferred frontend-tool halt takes effect here — but
                        # do NOT skip the message. In a parallel batch mixing a
                        # frontend tool with backend tools, THIS message carries
                        # the backend tools' real results, and dropping it loses
                        # them permanently: the client's tool card never
                        # resolves, the result never reaches MESSAGES_SNAPSHOT
                        # (the only path into client-side history — the
                        # TOOL_CALL_RESULT below is deliberately role-less and
                        # is not history), and state_from_result /
                        # custom_result_handler never fire. Consumers that
                        # persist from the event stream then hold a transcript
                        # whose toolUse has no toolResult, which the next run
                        # replays straight to the model provider.
                        #
                        # Fall through instead: the per-item loop already skips
                        # frontend placeholders (the client produces the real
                        # result), so only genuine backend results go out. Stop
                        # after the batch, before the next model cycle.
                        if pending_halt:
                            halt_event_stream = True
                        message_content = event["message"].get("content", [])
                        if not message_content or not isinstance(message_content, list):
                            continue

                        for item in message_content:
                            if not isinstance(item, dict) or "toolResult" not in item:
                                continue

                            tool_result = item["toolResult"]
                            result_tool_id = tool_result.get("toolUseId")
                            result_content = tool_result.get("content", [])

                            result_data = None
                            if result_content and isinstance(result_content, list):
                                for content_item in result_content:
                                    if (
                                        isinstance(content_item, dict)
                                        and "text" in content_item
                                    ):
                                        text_content = content_item["text"]
                                        try:
                                            result_data = json.loads(text_content)
                                        except json.JSONDecodeError:
                                            try:
                                                json_text = text_content.replace(
                                                    "'", '"'
                                                )
                                                result_data = json.loads(json_text)
                                            except Exception:
                                                result_data = text_content

                            if not result_tool_id or result_data is None:
                                continue

                            # Direct lookup works for backend tools (keyed by Strands ID).
                            # Frontend tools are keyed by a generated UUID, so we fall back
                            # to scanning by strands_tool_id when the direct lookup misses.
                            call_info = tool_calls_seen.get(result_tool_id, {})
                            if not call_info:
                                for _tid, _data in tool_calls_seen.items():
                                    if _data.get("strands_tool_id") == result_tool_id:
                                        call_info = _data
                                        break
                            # RESUME-run fallback: the interrupted tool never
                            # re-emits ``current_tool_use`` on resume, so
                            # ``tool_calls_seen`` is empty for it. The
                            # persisted meta map was populated when the call
                            # was originally streamed (possibly in a prior
                            # process). Direct native-id first, then scan by
                            # ``strands_tool_id`` to match the frontend-tool
                            # case.
                            if not call_info:
                                call_info = persisted_tool_call_meta.get(
                                    result_tool_id, {}
                                )
                            if not call_info:
                                for _pdata in persisted_tool_call_meta.values():
                                    if (
                                        isinstance(_pdata, dict)
                                        and _pdata.get("strands_tool_id")
                                        == result_tool_id
                                    ):
                                        call_info = _pdata
                                        break
                            # Record consumption once the lookup is complete
                            # (even if it missed): the result was processed
                            # this turn, so any persisted entry keyed on this
                            # native id is safe to prune. Recording BEFORE the
                            # frontend-skip / behavior branches ensures a
                            # ``stop_streaming_after_result`` early break still
                            # flags this id for prune.
                            processed_result_native_ids.add(result_tool_id)
                            tool_name = call_info.get("name")
                            tool_args = call_info.get("args")
                            tool_input = call_info.get("input")
                            behavior = (
                                self.config.tool_behaviors.get(tool_name)
                                if tool_name
                                else None
                            )

                            logger.debug(
                                f"Processing tool result: tool_name={tool_name}, result_tool_id={result_tool_id}, pending_tool_result_ids={pending_tool_result_ids}, thread_id={input_data.thread_id}"
                            )

                            # Skip server-side proxy placeholders: explicit
                            # per-call provenance is authoritative (including
                            # False for a native tool whose name collides with a
                            # client declaration). Older metadata lacks that
                            # flag, so fall back only to the original durable
                            # wire map or the immutable actual-registry snapshot.
                            is_frontend_provenance = call_info.get("is_frontend")
                            if isinstance(is_frontend_provenance, bool):
                                is_frontend_result = is_frontend_provenance
                            else:
                                is_frontend_result = (
                                    result_tool_id in original_wire_native_ids
                                    or (
                                        bool(tool_name)
                                        and tool_name
                                        in registered_frontend_tool_names
                                    )
                                )
                            if is_frontend_result:
                                continue

                            # Emit ToolCallResultEvent WITHOUT role field to complete the tool in UI
                            # but prevent it from being added to conversation history.
                            # A fresh message ID is used so CopilotKit creates a proper standalone
                            # ToolMessage and closes the spinner correctly.
                            tool_result_message_id = str(uuid.uuid4())
                            tool_result_content = json.dumps(result_data)
                            yield ToolCallResultEvent(
                                type=EventType.TOOL_CALL_RESULT,
                                tool_call_id=result_tool_id,
                                message_id=tool_result_message_id,
                                content=tool_result_content,
                                # role is intentionally omitted - without role="tool",
                                # the frontend won't add this to conversation history
                            )

                            # Splice point 3 of 4: append the ToolMessage
                            # carrying the backend tool result to the
                            # running snapshot so the frontend can pair
                            # call + result in the message tree.
                            if (
                                emit_snapshots
                                and not (
                                    behavior
                                    and behavior.skip_messages_snapshot
                                )
                            ):
                                snapshot_messages.append(
                                    ToolMessage(
                                        id=tool_result_message_id,
                                        role="tool",
                                        content=tool_result_content,
                                        tool_call_id=result_tool_id,
                                    )
                                )
                                yield MessagesSnapshotEvent(
                                    type=EventType.MESSAGES_SNAPSHOT,
                                    messages=list(snapshot_messages),
                                )

                            result_context = ToolResultContext(
                                input_data=input_data,
                                tool_name=tool_name or "",
                                tool_use_id=result_tool_id,
                                tool_input=tool_input,
                                args_str=tool_args or "{}",
                                result_data=result_data,
                                message_id=message_id,
                            )

                            if behavior and behavior.state_from_result:
                                try:
                                    snapshot = await maybe_await(
                                        behavior.state_from_result(result_context)
                                    )
                                    if snapshot:
                                        current_state.update(snapshot)
                                        yield StateSnapshotEvent(
                                            type=EventType.STATE_SNAPSHOT,
                                            snapshot=snapshot,
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"state_from_result failed for {tool_name}: {e}",
                                        exc_info=True,
                                    )

                            if behavior and behavior.custom_result_handler:
                                try:
                                    async for (
                                        custom_event
                                    ) in behavior.custom_result_handler(result_context):
                                        if custom_event is not None:
                                            yield custom_event
                                except Exception as e:
                                    logger.warning(
                                        f"custom_result_handler failed for {tool_name}: {e}",
                                        exc_info=True,
                                    )

                            if behavior and behavior.stop_streaming_after_result:
                                stop_text_streaming = True
                                if message_started:
                                    yield TextMessageEndEvent(
                                        type=EventType.TEXT_MESSAGE_END,
                                        message_id=message_id,
                                    )
                                    message_started = False
                                    # Splice point 4 of 4 (early-exit
                                    # variant): commit any accumulated
                                    # assistant text into the snapshot.
                                    if (
                                        emit_snapshots
                                        and accumulated_text
                                    ):
                                        snapshot_messages.append(
                                            AssistantMessage(
                                                id=message_id,
                                                role="assistant",
                                                content=accumulated_text,
                                            )
                                        )
                                        accumulated_text = ""
                                        yield MessagesSnapshotEvent(
                                            type=EventType.MESSAGES_SNAPSHOT,
                                            messages=list(snapshot_messages),
                                        )
                                halt_event_stream = True
                                logger.debug(
                                    f"Breaking event stream: stop_streaming_after_result behavior triggered (thread_id={input_data.thread_id}, tool_name={tool_name})"
                                )
                                # Break inner loop — no further results should be emitted
                                break

                        # Prune the persisted tool-call meta map for entries
                        # whose native id (or ``strands_tool_id`` for frontend
                        # tools stored under a wire key) was just consumed.
                        # The emission-time size cap (``_TOOL_CALL_MAP_MAX``) is
                        # only a backstop for abandoned entries.
                        if (
                            persisted_tool_call_meta
                            and processed_result_native_ids
                        ):
                            _remaining = {
                                _k: _v
                                for _k, _v in persisted_tool_call_meta.items()
                                if _k not in processed_result_native_ids
                                and (
                                    not isinstance(_v, dict)
                                    or _v.get("strands_tool_id")
                                    not in processed_result_native_ids
                                )
                            }
                            if len(_remaining) != len(persisted_tool_call_meta):
                                strands_agent.state.set(
                                    AG_UI_TOOL_CALL_MAP_STATE_KEY, _remaining
                                )
                                persisted_tool_call_meta = _remaining
                        processed_result_native_ids.clear()

                        # Defer hand-off: now that this turn's backend
                        # TOOL_CALL_RESULT(s) have been emitted above, flush the
                        # buffered frontend-tool ToolCallEnd(s). Flushing here —
                        # after the per-item loop and before the halt break below —
                        # guarantees the wire order backend TOOL_CALL_RESULT ->
                        # frontend TOOL_CALL_END, so the client only starts the
                        # frontend tool once backend work has reached it.
                        if deferred_frontend_tool_ends:
                            for _fe_tool_use_id in deferred_frontend_tool_ends:
                                yield ToolCallEndEvent(
                                    type=EventType.TOOL_CALL_END,
                                    tool_call_id=_fe_tool_use_id,
                                )
                            deferred_frontend_tool_ends = []

                        # The batch is fully emitted; stop before Strands runs
                        # another model cycle. Breaking HERE rather than relying
                        # on the check at the top of the loop means termination
                        # does not depend on Strands happening to yield one more
                        # event after this message.
                        if halt_event_stream:
                            break

                    # Handle tool calls
                    elif "current_tool_use" in event and event["current_tool_use"]:
                        tool_use = event["current_tool_use"]
                        tool_name = tool_use.get("name")
                        strands_tool_id = tool_use.get("toolUseId")
                        _raw_in = tool_use.get("input", "")

                        # Generate unique ID for frontend tools (to avoid ID conflicts across requests)
                        # Use Strands' ID for backend tools (so result lookup works)
                        is_frontend_tool = tool_name in registered_frontend_tool_names

                        # Check if we've already seen this tool (by Strands' internal ID)
                        existing_entry = None
                        for tid, data in tool_calls_seen.items():
                            if data.get("strands_tool_id") == strands_tool_id:
                                existing_entry = tid
                                break

                        if existing_entry:
                            # Reuse the existing ID
                            tool_use_id = existing_entry
                        elif is_frontend_tool:
                            # Generate new UUID for frontend tools
                            tool_use_id = str(uuid.uuid4())
                            # Record wire id -> Strands native id on the agent's
                            # SESSION STATE so a later continuation run — even on
                            # a different process — can reconcile the persisted
                            # placeholder (keyed by the native id) with the real
                            # client result (which arrives keyed by the wire id).
                            # Strands persists agent state durably at end of run.
                            # Only maintained when a session manager is actually
                            # active for this agent (matching the continuation
                            # read/prune gate); otherwise it would never be read.
                            if strands_tool_id and _get_strands_session_manager(
                                strands_agent
                            ):
                                _wire_map = dict(
                                    strands_agent.state.get(AG_UI_WIRE_MAP_STATE_KEY)
                                    or {}
                                )
                                _wire_map[tool_use_id] = strands_tool_id
                                # Bound growth: entries for frontend calls that
                                # never get a client result (abandoned/dismissed
                                # HITL) are never consumed/pruned. Keep only the
                                # most-recent ``_WIRE_MAP_MAX`` (insertion order).
                                if len(_wire_map) > _WIRE_MAP_MAX:
                                    for _stale in list(_wire_map)[
                                        : len(_wire_map) - _WIRE_MAP_MAX
                                    ]:
                                        _wire_map.pop(_stale, None)
                                strands_agent.state.set(
                                    AG_UI_WIRE_MAP_STATE_KEY, _wire_map
                                )
                        else:
                            # Use Strands' ID for backend tools
                            tool_use_id = strands_tool_id or str(uuid.uuid4())

                        logger.debug(
                            f"Tool call event received: tool_name={tool_name}, tool_use_id={tool_use_id}, strands_id={strands_tool_id}, is_frontend={is_frontend_tool}, already_seen={tool_use_id in tool_calls_seen}, thread_id={input_data.thread_id}"
                        )

                        # Update tool input as it streams in
                        tool_input_raw = tool_use.get("input", "")

                        # Raw string form is what FE incrementally parses for
                        # predict_state. Use it as-is for delta computation so
                        # the wire stream matches what the LLM actually emitted.
                        raw_str = (
                            tool_input_raw
                            if isinstance(tool_input_raw, str)
                            else json.dumps(tool_input_raw, default=str)
                        )

                        # Try to parse as JSON if it looks complete
                        tool_input = {}
                        if isinstance(tool_input_raw, str) and tool_input_raw:
                            try:
                                tool_input = json.loads(tool_input_raw)
                            except json.JSONDecodeError:
                                # Input is still streaming, keep as string
                                tool_input = tool_input_raw
                        elif isinstance(tool_input_raw, dict):
                            tool_input = tool_input_raw

                        args_str = (
                            json.dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input)
                        )

                        # Track or update tool call as input streams in
                        is_new_tool_call = (
                            tool_name and tool_use_id not in tool_calls_seen
                        )
                        if is_new_tool_call:
                            is_pending_now = tool_use_id in pending_tool_result_ids
                            behavior_now = self.config.tool_behaviors.get(tool_name)
                            # Use the streaming path (emit ToolCallStart +
                            # PredictState now, ToolCallArgs on each growth,
                            # ToolCallEnd at contentBlockStop) unless the tool
                            # is a continuation (already-resolved) or supplies
                            # a custom args_streamer that wants to drive args
                            # emission itself at contentBlockStop.
                            use_streaming = not is_pending_now and not (
                                behavior_now and behavior_now.args_streamer
                            )
                            tool_calls_seen[tool_use_id] = {
                                "name": tool_name,
                                "args": args_str,
                                "input": tool_input,
                                "raw": raw_str,
                                "emitted": False,  # legacy flag (still used by contentBlockStop scan)
                                "start_emitted": False,
                                "end_emitted": False,
                                "last_emitted_raw_len": 0,
                                "is_pending": is_pending_now,
                                "is_frontend": is_frontend_tool,
                                "use_streaming": use_streaming,
                                "strands_tool_id": strands_tool_id,
                            }

                            # Mirror the minimum-sufficient subset into agent
                            # state so a RESUME run — which does not
                            # re-emit ``current_tool_use`` for the interrupted
                            # tool — can still resolve ``tool_name``/behavior/
                            # context at the ``toolResult`` site. The cached
                            # per-thread agent provides the live checkpoint;
                            # a SessionManager additionally makes it durable.
                            if getattr(strands_agent, "state", None) is not None:
                                _tc_meta = dict(
                                    strands_agent.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)
                                    or {}
                                )
                                # Key by the NATIVE ``toolUseId`` — that is what
                                # arrives on ``toolResult``. For backend tools
                                # this equals ``tool_use_id``; for frontend
                                # tools ``tool_use_id`` is a fresh wire UUID
                                # while ``strands_tool_id`` is native.
                                _tc_key = strands_tool_id or tool_use_id
                                _tc_meta[_tc_key] = {
                                    "name": tool_name,
                                    "args": args_str,
                                    "input": tool_input,
                                    "wire_tool_call_id": tool_use_id,
                                    "strands_tool_id": strands_tool_id,
                                    "is_frontend": is_frontend_tool,
                                }
                                if len(_tc_meta) > _TOOL_CALL_MAP_MAX:
                                    for _stale in list(_tc_meta)[
                                        : len(_tc_meta) - _TOOL_CALL_MAP_MAX
                                    ]:
                                        _tc_meta.pop(_stale, None)
                                strands_agent.state.set(
                                    AG_UI_TOOL_CALL_MAP_STATE_KEY, _tc_meta
                                )
                                # Keep the in-run view aligned so downstream
                                # result lookups see the same entry a fresh
                                # process would restore from the store.
                                persisted_tool_call_meta = _tc_meta

                            if use_streaming:
                                # Close any open assistant text turn so the
                                # snapshot order matches the wire-event order
                                # and so message_id can rotate cleanly.
                                if message_started:
                                    yield TextMessageEndEvent(
                                        type=EventType.TEXT_MESSAGE_END,
                                        message_id=message_id,
                                    )
                                    if (
                                        emit_snapshots
                                        and accumulated_text
                                    ):
                                        snapshot_messages.append(
                                            AssistantMessage(
                                                id=message_id,
                                                role="assistant",
                                                content=accumulated_text,
                                            )
                                        )
                                        accumulated_text = ""
                                        yield MessagesSnapshotEvent(
                                            type=EventType.MESSAGES_SNAPSHOT,
                                            messages=list(snapshot_messages),
                                        )
                                    message_started = False
                                    message_id = str(uuid.uuid4())

                                # PredictState mapping must reach the FE BEFORE
                                # any args delta so the FE knows which tool
                                # argument feeds which state key while parsing
                                # incremental JSON.
                                if behavior_now:
                                    predict_state_payload = [
                                        mapping.to_payload()
                                        for mapping in normalize_predict_state(
                                            behavior_now.predict_state
                                        )
                                    ]
                                    if predict_state_payload:
                                        yield CustomEvent(
                                            type=EventType.CUSTOM,
                                            name="PredictState",
                                            value=predict_state_payload,
                                        )

                                # Must mirror the later tool snapshot emission condition.
                                tool_parent_message_id = (
                                    message_id
                                    if self._will_emit_tool_snapshot(behavior_now, emit_snapshots)
                                    else last_emitted_text_message_id
                                )
                                yield ToolCallStartEvent(
                                    type=EventType.TOOL_CALL_START,
                                    tool_call_id=tool_use_id,
                                    tool_call_name=tool_name,
                                    parent_message_id=tool_parent_message_id,
                                )
                                tool_calls_seen[tool_use_id]["start_emitted"] = True
                        elif tool_name and tool_use_id in tool_calls_seen:
                            # Update the input and args as they stream in
                            tool_calls_seen[tool_use_id]["input"] = tool_input
                            tool_calls_seen[tool_use_id]["args"] = args_str
                            tool_calls_seen[tool_use_id]["raw"] = raw_str

                            # Keep the persisted meta in sync with the final
                            # streamed args. Without this refresh, resume runs
                            # would see the first partial-JSON delta rather
                            # than the complete args the model emitted.
                            if getattr(strands_agent, "state", None) is not None:
                                _tc_meta = dict(
                                    strands_agent.state.get(AG_UI_TOOL_CALL_MAP_STATE_KEY)
                                    or {}
                                )
                                _tc_key = strands_tool_id or tool_use_id
                                _existing = _tc_meta.get(_tc_key)
                                if _existing is not None:
                                    _existing["input"] = tool_input
                                    _existing["args"] = args_str
                                    strands_agent.state.set(
                                        AG_UI_TOOL_CALL_MAP_STATE_KEY, _tc_meta
                                    )
                                    persisted_tool_call_meta = _tc_meta

                        # Stream incremental ToolCallArgs deltas as the LLM
                        # produces more characters of the JSON args. The FE
                        # uses these to drive predictive state updates per the
                        # PredictState mapping that was just emitted.
                        entry = tool_calls_seen.get(tool_use_id)
                        if (
                            entry
                            and entry.get("start_emitted")
                            and entry.get("use_streaming")
                        ):
                            new_len = len(raw_str)
                            last_len = entry.get("last_emitted_raw_len", 0)
                            if new_len > last_len:
                                yield ToolCallArgsEvent(
                                    type=EventType.TOOL_CALL_ARGS,
                                    tool_call_id=tool_use_id,
                                    delta=raw_str[last_len:new_len],
                                )
                                entry["last_emitted_raw_len"] = new_len

                    # Handle content block stop - this signals tool input is complete
                    elif "event" in event and isinstance(event.get("event"), dict):
                        inner_event = event["event"]
                        if "contentBlockStop" in inner_event:
                            # Close reasoning events if active
                            if reasoning_started:
                                yield ReasoningMessageEndEvent(
                                    type=EventType.REASONING_MESSAGE_END,
                                    message_id=reasoning_message_id
                                )
                                yield ReasoningEndEvent(
                                    type=EventType.REASONING_END,
                                    message_id=reasoning_message_id
                                )
                                reasoning_started = False
                                reasoning_message_id = None

                            # Find the most recent tool call that hasn't been emitted yet
                            tool_name = None
                            tool_input = None
                            args_str = None
                            tool_use_id = None

                            for tid, tool_data in tool_calls_seen.items():
                                if not tool_data.get("emitted", True):
                                    tool_name = tool_data["name"]
                                    tool_input = tool_data["input"]
                                    args_str = tool_data["args"]
                                    tool_use_id = tid
                                    break  # Process one tool at a time

                            # Only process if we found a tool to emit
                            if tool_name and tool_use_id:
                                entry = tool_calls_seen[tool_use_id]
                                # Mark as emitted (legacy compat)
                                entry["emitted"] = True
                                entry["end_emitted"] = True

                                is_frontend_tool = entry.get("is_frontend", tool_name in frontend_tool_names)
                                behavior = self.config.tool_behaviors.get(tool_name)
                                is_pending = entry.get("is_pending", tool_use_id in pending_tool_result_ids)
                                use_streaming = entry.get("use_streaming", False)

                                logger.debug(
                                    f"contentBlockStop close: tool_name={tool_name}, tool_use_id={tool_use_id}, is_frontend_tool={is_frontend_tool}, is_pending={is_pending}, use_streaming={use_streaming}, thread_id={input_data.thread_id}"
                                )
                                call_context = ToolCallContext(
                                    input_data=input_data,
                                    tool_name=tool_name,
                                    tool_use_id=tool_use_id,
                                    tool_input=tool_input,
                                    args_str=args_str,
                                )

                                if use_streaming:
                                    # Streaming path: ToolCallStart, PredictState
                                    # and the args deltas have already been
                                    # emitted from the current_tool_use handler.
                                    # Flush any final delta the LLM tacked on
                                    # between the last current_tool_use update
                                    # and contentBlockStop, then close the call.
                                    raw_str = entry.get("raw", "") or ""
                                    last_len = entry.get("last_emitted_raw_len", 0)
                                    if len(raw_str) > last_len:
                                        yield ToolCallArgsEvent(
                                            type=EventType.TOOL_CALL_ARGS,
                                            tool_call_id=tool_use_id,
                                            delta=raw_str[last_len:],
                                        )
                                        entry["last_emitted_raw_len"] = len(raw_str)

                                    # Emit ``state_from_args`` BEFORE
                                    # ``ToolCallEnd``. CopilotKit v2 releases
                                    # the predict_state buffer at ToolCallEnd;
                                    # if the authoritative StateSnapshot lands
                                    # after that, the FE momentarily reverts
                                    # to the last server-confirmed state and
                                    # re-applies, producing a "re-stream"
                                    # animation. Delivering the snapshot first
                                    # means the FE has the real state in hand
                                    # at the moment prediction is released.
                                    if behavior and behavior.state_from_args:
                                        try:
                                            snapshot = await maybe_await(
                                                behavior.state_from_args(call_context)
                                            )
                                            if snapshot:
                                                current_state.update(snapshot)
                                                yield StateSnapshotEvent(
                                                    type=EventType.STATE_SNAPSHOT,
                                                    snapshot=snapshot,
                                                )
                                        except Exception as e:
                                            logger.warning(
                                                f"state_from_args failed for {tool_name}: {e}",
                                                exc_info=True,
                                            )

                                    # Defer hand-off: for frontend tools, buffer the
                                    # ToolCallEnd instead of emitting it now. It is
                                    # flushed after this turn's backend results (see
                                    # the pending_halt handler). Backend tools and
                                    # continue_after_frontend_call tools emit now.
                                    if is_frontend_tool and not (
                                        behavior and behavior.continue_after_frontend_call
                                    ):
                                        deferred_frontend_tool_ends.append(tool_use_id)
                                    else:
                                        yield ToolCallEndEvent(
                                            type=EventType.TOOL_CALL_END,
                                            tool_call_id=tool_use_id,
                                        )

                                    if self._will_emit_tool_snapshot(behavior, emit_snapshots):
                                        snapshot_messages.append(
                                            AssistantMessage(
                                                id=message_id,
                                                role="assistant",
                                                content="",
                                                tool_calls=[
                                                    ToolCall(
                                                        id=tool_use_id,
                                                        type="function",
                                                        function=FunctionCall(
                                                            name=tool_name or "unknown",
                                                            arguments=args_str or "{}",
                                                        ),
                                                    )
                                                ],
                                            )
                                        )
                                        yield MessagesSnapshotEvent(
                                            type=EventType.MESSAGES_SNAPSHOT,
                                            messages=list(snapshot_messages),
                                        )
                                        # Rotate so the next assistant message
                                        # in the snapshot (text or another
                                        # tool call) carries a distinct id —
                                        # CopilotKit v2 dedupes by id.
                                        message_id = str(uuid.uuid4())

                                    if is_frontend_tool and not (
                                        behavior
                                        and behavior.continue_after_frontend_call
                                    ):
                                        logger.debug(
                                            f"Deferring halt after frontend tool call: tool_name={tool_name}, tool_call_id={tool_use_id}, thread_id={input_data.thread_id}"
                                        )
                                        pending_halt = True
                                elif is_pending:
                                    # Continuation turn — tool already resolved
                                    # in conversation history. Don't re-emit any
                                    # wire events but still let state callbacks
                                    # fire so derived state stays consistent.
                                    if behavior and behavior.state_from_args:
                                        try:
                                            snapshot = await maybe_await(
                                                behavior.state_from_args(call_context)
                                            )
                                            if snapshot:
                                                current_state.update(snapshot)
                                                yield StateSnapshotEvent(
                                                    type=EventType.STATE_SNAPSHOT,
                                                    snapshot=snapshot,
                                                )
                                        except Exception as e:
                                            logger.warning(
                                                f"state_from_args failed for {tool_name}: {e}",
                                                exc_info=True,
                                            )
                                else:
                                    # Legacy path: behavior.args_streamer is
                                    # configured. Emit the full burst at
                                    # contentBlockStop using the custom
                                    # streamer so existing args_streamer
                                    # consumers keep working.
                                    if behavior and behavior.state_from_args:
                                        try:
                                            snapshot = await maybe_await(
                                                behavior.state_from_args(call_context)
                                            )
                                            if snapshot:
                                                current_state.update(snapshot)
                                                yield StateSnapshotEvent(
                                                    type=EventType.STATE_SNAPSHOT,
                                                    snapshot=snapshot,
                                                )
                                        except Exception as e:
                                            logger.warning(
                                                f"state_from_args failed for {tool_name}: {e}",
                                                exc_info=True,
                                            )

                                    if behavior:
                                        predict_state_payload = [
                                            mapping.to_payload()
                                            for mapping in normalize_predict_state(
                                                behavior.predict_state
                                            )
                                        ]
                                        if predict_state_payload:
                                            yield CustomEvent(
                                                type=EventType.CUSTOM,
                                                name="PredictState",
                                                value=predict_state_payload,
                                            )

                                    if message_started:
                                        yield TextMessageEndEvent(
                                            type=EventType.TEXT_MESSAGE_END, message_id=message_id
                                        )
                                        if (
                                            emit_snapshots
                                            and accumulated_text
                                        ):
                                            snapshot_messages.append(
                                                AssistantMessage(
                                                    id=message_id,
                                                    role="assistant",
                                                    content=accumulated_text,
                                                )
                                            )
                                            accumulated_text = ""
                                            yield MessagesSnapshotEvent(
                                                type=EventType.MESSAGES_SNAPSHOT,
                                                messages=list(snapshot_messages),
                                            )
                                        message_started = False
                                        message_id = str(uuid.uuid4())

                                    # Must mirror the later tool snapshot emission condition.
                                    tool_parent_message_id = (
                                        message_id
                                        if self._will_emit_tool_snapshot(behavior, emit_snapshots)
                                        else last_emitted_text_message_id
                                    )
                                    yield ToolCallStartEvent(
                                        type=EventType.TOOL_CALL_START,
                                        tool_call_id=tool_use_id,
                                        tool_call_name=tool_name,
                                        parent_message_id=tool_parent_message_id,
                                    )

                                    try:
                                        async for chunk in behavior.args_streamer(
                                            call_context
                                        ):
                                            if chunk is None:
                                                continue
                                            yield ToolCallArgsEvent(
                                                type=EventType.TOOL_CALL_ARGS,
                                                tool_call_id=tool_use_id,
                                                delta=str(chunk),
                                            )
                                    except Exception as e:
                                        logger.warning(
                                            f"args_streamer failed for {tool_name}, falling back to full args: {e}"
                                        )
                                        yield ToolCallArgsEvent(
                                            type=EventType.TOOL_CALL_ARGS,
                                            tool_call_id=tool_use_id,
                                            delta=args_str,
                                        )

                                    yield ToolCallEndEvent(
                                        type=EventType.TOOL_CALL_END,
                                        tool_call_id=tool_use_id,
                                    )

                                    if self._will_emit_tool_snapshot(behavior, emit_snapshots):
                                        snapshot_messages.append(
                                            AssistantMessage(
                                                id=message_id,
                                                role="assistant",
                                                content="",
                                                tool_calls=[
                                                    ToolCall(
                                                        id=tool_use_id,
                                                        type="function",
                                                        function=FunctionCall(
                                                            name=tool_name or "unknown",
                                                            arguments=args_str or "{}",
                                                        ),
                                                    )
                                                ],
                                            )
                                        )
                                        yield MessagesSnapshotEvent(
                                            type=EventType.MESSAGES_SNAPSHOT,
                                            messages=list(snapshot_messages),
                                        )
                                        message_id = str(uuid.uuid4())

                                    if is_frontend_tool and not (
                                        behavior
                                        and behavior.continue_after_frontend_call
                                    ):
                                        logger.debug(
                                            f"Deferring halt after frontend tool call: tool_name={tool_name}, tool_call_id={tool_use_id}, thread_id={input_data.thread_id}"
                                        )
                                        pending_halt = True

                # Defer hand-off (safety flush): if the stream ended without a
                # backend tool-result message (e.g. a turn with ONLY frontend tool
                # calls), the per-batch flush above never ran and the buffered
                # frontend ToolCallEnd(s) would be lost — leaving TOOL_CALL_START
                # events with no matching END. Flush any remainder here.
                if deferred_frontend_tool_ends and proxy_boundary_error is None:
                    for _fe_tool_use_id in deferred_frontend_tool_ends:
                        yield ToolCallEndEvent(
                            type=EventType.TOOL_CALL_END,
                            tool_call_id=_fe_tool_use_id,
                        )
                    deferred_frontend_tool_ends = []
            finally:
                # Properly close the async generator to avoid context detachment errors
                # The generator should complete naturally when we consume all events,
                # but we still try to close it explicitly to be safe
                try:
                    # A frontend-tool halt breaks out of the loop with the
                    # generator SUSPENDED at a yield, where ``ag_running`` is
                    # False. The exhausted-generator check below would read
                    # that as "already closed" and defer teardown to GC,
                    # leaving the halted Strands cycle (and its model stream)
                    # open. Close it explicitly instead.
                    if halt_event_stream:
                        await agent_stream.aclose()
                    # Check if generator is already closed/exhausted
                    elif not agent_stream.ag_running:
                        # Generator is already closed, nothing to do
                        pass
                    else:
                        # Try to close gracefully, but suppress context-related errors
                        await agent_stream.aclose()
                except (
                    GeneratorExit,
                    ValueError,
                    RuntimeError,
                    StopAsyncIteration,
                ) as e:
                    # Suppress context detachment errors - they occur when the generator
                    # is closed in a different context, but don't affect functionality
                    # These errors are logged by Strands internally, we just prevent them from propagating
                    pass
                except AttributeError:
                    # Generator doesn't have ag_running attribute (older Python versions)
                    # Just try to close it
                    try:
                        await agent_stream.aclose()
                    except (
                        GeneratorExit,
                        ValueError,
                        RuntimeError,
                        StopAsyncIteration,
                    ):
                        pass
                except Exception as e:
                    # Log other errors but don't fail
                    logger.warning(f"Error closing agent stream: {e}")

            proxy_hook_boundary = self._proxy_hook_boundaries_by_thread.get(
                thread_id
            )
            if (
                proxy_boundary_error is None
                and proxy_hook_boundary is not None
                and proxy_hook_boundary.failed
            ):
                proxy_boundary_error = _proxy_hook_error(
                    proxy_hook_boundary.failure_fields
                )
            if proxy_boundary_error is not None:
                yield proxy_boundary_error
                return

            # ``create_proxy_tool`` pops only after the resumed hook allows the
            # exact native proxy invocation to proceed. Remove the consumed
            # wire mapping then; an unconsumed override remains retryable.
            consumed_proxy_resume_ids = (
                proxy_resume_native_ids - proxy_resume_results.keys()
            )
            if consumed_proxy_resume_ids and getattr(
                strands_agent, "state", None
            ) is not None:
                current_wire_map = dict(
                    strands_agent.state.get(AG_UI_WIRE_MAP_STATE_KEY) or {}
                )
                remaining_wire_map = {
                    wire: native
                    for wire, native in current_wire_map.items()
                    if native not in consumed_proxy_resume_ids
                }
                strands_agent.state.set(
                    AG_UI_WIRE_MAP_STATE_KEY, remaining_wire_map
                )
                _present, managed_ids, current_provenance = (
                    _proxy_hook_provenance_value(strands_agent)
                )
                if (
                    managed_ids is not None
                    and current_provenance is not None
                ):
                    remaining_provenance = {
                        interrupt_id: record
                        for interrupt_id, record in current_provenance.items()
                        if record.get("original_native_tool_call_id")
                        not in consumed_proxy_resume_ids
                    }
                    if remaining_provenance:
                        strands_agent.state.set(
                            AG_UI_PROXY_HOOK_PROVENANCE_STATE_KEY,
                            _proxy_hook_provenance_payload(
                                remaining_provenance
                            ),
                        )
                    else:
                        strands_agent.state.delete(
                            AG_UI_PROXY_HOOK_PROVENANCE_STATE_KEY
                        )

            # Close reasoning if still open
            if reasoning_started:
                yield ReasoningMessageEndEvent(
                    type=EventType.REASONING_MESSAGE_END,
                    message_id=reasoning_message_id
                )
                yield ReasoningEndEvent(
                    type=EventType.REASONING_END,
                    message_id=reasoning_message_id
                )

            # End message if started
            if message_started:
                yield TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END, message_id=message_id
                )
                # Splice point 4 of 4 (terminal): commit the final
                # assistant text turn into the snapshot so the frontend
                # has the closing message in canonical history.
                if emit_snapshots and accumulated_text:
                    snapshot_messages.append(
                        AssistantMessage(
                            id=message_id,
                            role="assistant",
                            content=accumulated_text,
                        )
                    )
                    accumulated_text = ""
                    yield MessagesSnapshotEvent(
                        type=EventType.MESSAGES_SNAPSHOT,
                        messages=list(snapshot_messages),
                    )

            pending_hook_native_ids = _pending_proxy_hook_native_ids(
                strands_agent, persisted_tool_call_meta
            )
            # A mixed frontend-proxy/native-interrupt batch parks the proxy's
            # placeholder inside the live interrupt context. A proxy hook
            # pause likewise needs durable wire/native identity before it can
            # advertise a resumable interrupt.
            if has_active_proxy_placeholder(strands_agent, persisted_tool_call_meta):
                if session_manager is None:
                    yield _interrupt_session_required_error()
                    return
                if not _supports_repository_reconciliation(session_manager):
                    yield _interrupt_session_capability_error()
                    return
            if session_manager is None and pending_hook_native_ids:
                yield _interrupt_session_required_error()
                return

            # Final state snapshot before finishing
            yield StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=current_state,
            )

            # If the run paused on a native Strands interrupt, surface it as an
            # AG-UI interrupt outcome so the client can collect a response and
            # resume via ``RunAgentInput.resume`` next turn. Otherwise finish
            # bare, exactly as before (no behavior change for normal runs).
            native_interrupts = _extract_interrupts(strands_agent, terminal_result)
            if native_interrupts:
                latest_wire_map = (
                    strands_agent.state.get(AG_UI_WIRE_MAP_STATE_KEY) or {}
                    if getattr(strands_agent, "state", None) is not None
                    else {}
                )
                native_to_wire = {
                    native: wire
                    for wire, native in latest_wire_map.items()
                    if native in pending_hook_native_ids
                }
                _present, _managed_ids, hook_provenance = (
                    _proxy_hook_provenance_value(
                    strands_agent
                    )
                )
                interrupt_to_wire = {
                    interrupt_id: record["wire_tool_call_id"]
                    for interrupt_id, record in (hook_provenance or {}).items()
                    if interrupt_id
                    in _unanswered_interrupt_ids(strands_agent._interrupt_state)
                }
                yield RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    outcome=RunFinishedInterruptOutcome(
                        type="interrupt",
                        interrupts=[
                            _strands_interrupt_to_agui(
                                i, native_to_wire, interrupt_to_wire
                            )
                            for i in native_interrupts
                        ],
                    ),
                )
            else:
                # Always finish the run - frontend handles keeping action executing
                yield RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )

        except Exception as e:
            import traceback

            traceback.print_exc()
            yield RunErrorEvent(
                type=EventType.RUN_ERROR, message=str(e), code="STRANDS_ERROR"
            )
