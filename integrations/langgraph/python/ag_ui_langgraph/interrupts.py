from typing import Any, List, Mapping
from uuid import uuid4

from ag_ui.core import Interrupt as AGUIInterrupt
from langgraph.types import Interrupt as LangGraphInterrupt

from .utils import make_json_safe


def lg_interrupt_to_agui(lg: LangGraphInterrupt) -> AGUIInterrupt:
    raw = lg.value
    is_dict = isinstance(raw, Mapping)

    interrupt_id = getattr(lg, "id", None) or f"lg-{uuid4()}"
    reason = (raw.get("reason") if is_dict else None) or "langgraph:interrupt"

    message = (
        raw if isinstance(raw, str)
        else raw.get("message") if is_dict else None
    )
    tool_call_id = (
        raw.get("toolCallId") if is_dict else None
    ) or (
        raw.get("tool_call_id") if is_dict else None
    )
    response_schema = (
        raw.get("responseSchema") if is_dict else None
    ) or (
        raw.get("response_schema") if is_dict else None
    )
    expires_at = (
        raw.get("expiresAt") if is_dict else None
    ) or (
        raw.get("expires_at") if is_dict else None
    )

    metadata: dict[str, Any] = {
        "langgraph": {
            "raw": make_json_safe(raw),
            "ns": getattr(lg, "ns", None),
            "resumable": getattr(lg, "resumable", None),
            "when": getattr(lg, "when", None),
        }
    }

    return AGUIInterrupt(
        id=interrupt_id,
        reason=reason,
        message=message,
        tool_call_id=tool_call_id,
        response_schema=response_schema,
        expires_at=expires_at,
        metadata=metadata,
    )


def lg_interrupts_to_agui(items) -> List[AGUIInterrupt]:
    return [lg_interrupt_to_agui(i) for i in items]


DEFAULT_RESUME_SENTINEL_CANCELLED = "__agui_cancelled__"
DEFAULT_RESUME_SENTINEL_MAP = "__agui_resume_map__"
