"""
An agentic chat flow that surfaces the model's reasoning.

The reasoning cell lets the user pick a provider from the frontend; the choice
arrives on ``state.model``. When the selected model emits reasoning tokens
(Anthropic extended thinking, DeepSeek, native Gemini), the bridge streams them
as REASONING_* events. OpenAI over chat-completions returns no reasoning
content, so that option answers without a thinking trace.
"""

from typing import Any, Dict, List

from crewai.flow.flow import Flow, start
from litellm import acompletion
from ..sdk import copilotkit_stream, CopilotKitState


class AgentState(CopilotKitState):
    """Chat state plus the frontend-selected reasoning model."""

    model: str = "OpenAI"


def _completion_kwargs(selected_model: str) -> Dict[str, Any]:
    """Map the frontend model choice to a LiteLLM model + reasoning config."""
    if selected_model == "Anthropic":
        return {
            "model": "anthropic/claude-sonnet-4-5",
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    if selected_model == "Gemini":
        return {
            "model": "gemini/gemini-2.5-pro",
            "reasoning_effort": "low",
        }
    # OpenAI over chat-completions rejects reasoning_effort and returns no
    # reasoning content, so the OpenAI option answers without a thinking trace.
    # A visible trace needs a reasoning-capable provider (Anthropic / Gemini).
    return {
        "model": "openai/gpt-5.4",
    }


class AgenticChatReasoningFlow(Flow[AgentState]):

    @start()
    async def chat(self):
        system_prompt = "You are a helpful assistant."

        tools: List[Any] = [*self.state.copilotkit.actions]

        response = await copilotkit_stream(
            await acompletion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.state.messages,
                ],
                tools=tools or None,
                parallel_tool_calls=False if tools else None,
                stream=True,
                **_completion_kwargs(self.state.model),
            )
        )

        self.state.messages.append(response.choices[0].message)
