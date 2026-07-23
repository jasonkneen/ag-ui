"""Turn AG-UI and backend tool definitions into managed-agent custom tools."""

import re
from typing import Any

_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")
_INVALID_CHAR = re.compile(r"[^A-Za-z0-9_-]")


def normalize_tool_name(name: str) -> str:
    """Managed Agents tool names allow only [A-Za-z0-9_-], max 128 chars."""
    if _NAME_PATTERN.fullmatch(name):
        return name
    return _INVALID_CHAR.sub("_", name)[:128] or "tool"


def _input_schema(parameters: Any) -> dict[str, Any]:
    if isinstance(parameters, dict):
        properties = parameters.get("properties")
        required = parameters.get("required")
        return {
            "type": "object",
            "properties": properties if properties is not None else {},
            "required": required if required is not None else [],
        }
    return {"type": "object", "properties": {}}


def custom_tool_from(tool: Any) -> dict[str, Any]:
    """An AG-UI (frontend) or backend tool definition -> managed-agent custom tool.

    Accepts anything with `name`, `description`, and `parameters` attributes.
    """
    name = tool.name
    description = getattr(tool, "description", None)
    return {
        "type": "custom",
        "name": normalize_tool_name(name),
        "description": (description or f"Tool {name}")[:1024],
        "input_schema": _input_schema(getattr(tool, "parameters", None)),
    }
