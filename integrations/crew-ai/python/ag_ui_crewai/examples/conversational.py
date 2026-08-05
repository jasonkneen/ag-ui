"""Conversational variants of the regular CrewAI dojo Flows."""

from __future__ import annotations

from typing import Any

from crewai.experimental.conversational import (
    ConversationConfig,
    message_to_llm_dict,
)
from crewai.flow.flow import listen
from pydantic import BaseModel

from .a2ui_dynamic_schema import A2UIDynamicSchemaFlow
from .a2ui_fixed_schema import A2UIFixedSchemaFlow
from .a2ui_recovery import A2UIRecoveryFlow
from .agentic_chat import AgenticChatFlow
from .agentic_chat_multimodal import AgenticChatMultimodalFlow
from .agentic_chat_reasoning import AgenticChatReasoningFlow
from .agentic_generative_ui import AgenticGenerativeUIFlow
from .backend_tool_rendering import BackendToolRenderingFlow
from .human_in_the_loop import HumanInTheLoopFlow
from .interrupt_flow import InterruptFlow
from .predictive_state_updates import PredictiveStateUpdatesFlow
from .shared_state import SharedStateFlow
from .subgraphs import SubgraphsFlow
from .tool_based_generative_ui import ToolBasedGenerativeUIFlow


class _AGUIConversationalBehavior:
    """Route each public turn through the regular Flow's existing starts."""

    def receive_user_message(self, *args: Any, **kwargs: Any) -> Any:
        result = super().receive_user_message(*args, **kwargs)
        messages = getattr(self.state, "messages", None)
        if messages and isinstance(messages[-1], BaseModel):
            messages[-1] = message_to_llm_dict(messages[-1])
        return result

    def route_turn(self, _context: Any) -> str:
        return "ag_ui_complete"

    @listen("ag_ui_complete")
    def finish_ag_ui_turn(self) -> None:
        return None


def _conversational_type(base: type[Any]) -> type[Any]:
    return type(
        f"Conversational{base.__name__}",
        (_AGUIConversationalBehavior, base),
        {
            "__module__": __name__,
            "conversational": True,
            "conversational_config": ConversationConfig(
                defer_trace_finalization=False
            ),
        },
    )


CONVERSATIONAL_FLOW_TYPES = {
    "agentic_chat": _conversational_type(AgenticChatFlow),
    "agentic_chat_reasoning": _conversational_type(AgenticChatReasoningFlow),
    "agentic_chat_multimodal": _conversational_type(AgenticChatMultimodalFlow),
    "backend_tool_rendering": _conversational_type(BackendToolRenderingFlow),
    "interrupt": _conversational_type(InterruptFlow),
    "human_in_the_loop": _conversational_type(HumanInTheLoopFlow),
    "agentic_generative_ui": _conversational_type(AgenticGenerativeUIFlow),
    "predictive_state_updates": _conversational_type(PredictiveStateUpdatesFlow),
    "shared_state": _conversational_type(SharedStateFlow),
    "tool_based_generative_ui": _conversational_type(ToolBasedGenerativeUIFlow),
    "subgraphs": _conversational_type(SubgraphsFlow),
    "a2ui_dynamic_schema": _conversational_type(A2UIDynamicSchemaFlow),
    "a2ui_recovery": _conversational_type(A2UIRecoveryFlow),
    "a2ui_fixed_schema": _conversational_type(A2UIFixedSchemaFlow),
}
