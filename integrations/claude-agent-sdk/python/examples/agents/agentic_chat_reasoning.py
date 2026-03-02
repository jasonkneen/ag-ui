"""
Agentic chat with reasoning agent configuration.

This module provides a factory function for creating an agentic chat adapter
with extended thinking/reasoning enabled. Uses Claude Haiku 4.5 with adaptive
thinking for cost-effective reasoning demonstrations.
"""

from ag_ui_claude_sdk import ClaudeAgentAdapter
from .constants import DEFAULT_DISALLOWED_TOOLS


def create_agentic_chat_reasoning_adapter() -> ClaudeAgentAdapter:
    """Create adapter for agentic chat with reasoning enabled."""
    return ClaudeAgentAdapter(
        name="agentic_chat_reasoning",
        description="Chat assistant with extended thinking/reasoning",
        options={
            "model": "sonnet",
            "system_prompt": "You are a helpful assistant. Think step by step when answering questions.",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "disallowed_tools": list(DEFAULT_DISALLOWED_TOOLS),
        }
    )
