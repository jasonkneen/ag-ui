"""Strict helpers for frontend tools parked in Strands native interrupts."""

from __future__ import annotations

from typing import Any, Mapping

FRONTEND_TOOL_INTERRUPT_NAME = "ag_ui_frontend_tool_wait"
FRONTEND_TOOL_RESPONSE_KEY = "__ag_ui_frontend_tool_response__"

#: ``Interrupt.reason`` published for a frontend tool parked in a native wait.
FRONTEND_TOOL_INTERRUPT_REASON = "frontend_tool_call"


def frontend_tool_reason(tool_use_id: str, tool_name: str) -> dict[str, str]:
    """Tag a native interrupt with its canonical Strands tool-use identity."""
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise ValueError("frontend tool-use ID must be non-blank")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("frontend tool name must be non-blank")
    return {
        "name": FRONTEND_TOOL_INTERRUPT_NAME,
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
    }


def parse_frontend_tool_reason(reason: Any) -> tuple[str, str]:
    """Return ``(tool_use_id, tool_name)`` from one exact frontend-wait tag."""
    if not isinstance(reason, Mapping) or set(reason) != {
        "name",
        "tool_use_id",
        "tool_name",
    }:
        raise ValueError("malformed frontend tool interrupt reason")
    if reason["name"] != FRONTEND_TOOL_INTERRUPT_NAME:
        raise ValueError("not a frontend tool interrupt")
    parsed = frontend_tool_reason(reason["tool_use_id"], reason["tool_name"])
    return parsed["tool_use_id"], parsed["tool_name"]


def frontend_tool_response_schema() -> dict[str, Any]:
    """The canonical ``resume[]`` payload contract for a frontend wait.

    Published on the AG-UI ``Interrupt`` so a client can answer the wait through
    the standard resume channel. The legacy ``ToolMessage`` channel is a
    boundary translation onto exactly this shape.
    """
    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "error": {"type": "boolean"},
        },
        "required": ["content"],
    }


def is_frontend_tool_interrupt(interrupt: Any) -> bool:
    """Classify only interrupts carrying this adapter's reserved name."""
    return getattr(interrupt, "name", None) == FRONTEND_TOOL_INTERRUPT_NAME


def index_frontend_tool_interrupts(agent: Any) -> dict[str, Any]:
    """Index tagged native interrupts by canonical tool-use ID, without mutation."""
    state = getattr(agent, "_interrupt_state", None)
    if state is None or getattr(state, "activated", False) is not True:
        return {}
    interrupts = getattr(state, "interrupts", None)
    if not isinstance(interrupts, Mapping):
        raise ValueError("malformed Strands interrupt checkpoint")

    indexed: dict[str, Any] = {}
    for interrupt_id, interrupt in interrupts.items():
        if not isinstance(interrupt_id, str) or not interrupt_id.strip():
            raise ValueError("malformed Strands interrupt ID")
        if getattr(interrupt, "id", None) != interrupt_id:
            raise ValueError("Strands interrupt key does not match its ID")
        if not is_frontend_tool_interrupt(interrupt):
            continue
        tool_use_id, _tool_name = parse_frontend_tool_reason(
            getattr(interrupt, "reason", None)
        )
        if tool_use_id in indexed:
            raise ValueError(f"duplicate frontend tool-use ID: {tool_use_id}")
        indexed[tool_use_id] = interrupt
    return indexed


def wrap_frontend_tool_response(
    content: str, *, is_error: bool
) -> dict[str, dict[str, Any]]:
    """Wrap every result in a truthy envelope for Strands 1.15 compatibility."""
    if not isinstance(content, str) or not isinstance(is_error, bool):
        raise TypeError(
            "frontend tool response must contain string content and boolean error status"
        )
    return {
        FRONTEND_TOOL_RESPONSE_KEY: {
            "content": content,
            "is_error": is_error,
        }
    }


def unwrap_frontend_tool_response(value: Any) -> tuple[str, bool]:
    """Unwrap only the exact response envelope emitted by this adapter."""
    if not isinstance(value, Mapping) or set(value) != {FRONTEND_TOOL_RESPONSE_KEY}:
        raise ValueError("malformed frontend tool response envelope")
    response = value[FRONTEND_TOOL_RESPONSE_KEY]
    if not isinstance(response, Mapping) or set(response) != {"content", "is_error"}:
        raise ValueError("malformed frontend tool response envelope")
    content = response["content"]
    is_error = response["is_error"]
    if not isinstance(content, str) or not isinstance(is_error, bool):
        raise ValueError("malformed frontend tool response envelope")
    return content, is_error
