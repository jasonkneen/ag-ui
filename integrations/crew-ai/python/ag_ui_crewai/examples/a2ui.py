"""A2UI generative-surface flow.

Demonstrates the four A2UI pillars on CrewAI: auto-injection with opt-out,
progressive streaming, error recovery, and subagent-based generation. The A2UI
component catalog + the ``injectA2UITool`` flag arrive via the frontend a2ui
middleware and are surfaced under ``state["ag-ui"]`` by the endpoint; this node
wires them with ``plan_a2ui_injection`` and runs ``generate_a2ui`` when the
model calls it.
"""

import json

from crewai.flow.flow import Flow, start
from litellm import acompletion

from ..sdk import copilotkit_stream
from ..a2ui_tool import (
    apply_a2ui_plan_to_tools,
    plan_a2ui_injection,
)

MODEL = "openai/gpt-4o"

SYSTEM_PROMPT = (
    "You are a helpful assistant that can render rich, interactive UI surfaces. "
    "When the user asks for anything visual - a card, form, list, dashboard, or "
    "comparison - call the generate_a2ui tool to design and render it. Use "
    "intent='create' for a new surface and intent='update' (with "
    "target_surface_id) to modify one you rendered earlier in the conversation."
)


class A2UIFlow(Flow):
    """A flow that renders A2UI surfaces via the auto-injected generate_a2ui tool."""

    @start()
    async def chat(self):
        state = self.state
        actions = (state.get("copilotkit") or {}).get("actions") or []
        existing_names = [
            a["function"]["name"]
            for a in actions
            if isinstance(a, dict)
            and isinstance(a.get("function"), dict)
            and a["function"].get("name")
        ]

        # Auto-inject generate_a2ui (opt-out honored inside plan_a2ui_injection):
        # off unless the runtime forwarded injectA2UITool. When active, swap the
        # injected render proxy for generate_a2ui in the tools the model sees.
        plan = plan_a2ui_injection(
            model=MODEL,
            state=state,
            existing_tool_names=existing_names,
        )
        tools = apply_a2ui_plan_to_tools(actions, plan)

        # Omit the tool kwargs entirely when there are none: A2UI opt-out with no
        # other frontend actions leaves ``tools`` empty, and OpenAI-compatible
        # providers reject an empty ``tools`` array alongside
        # ``parallel_tool_calls``.
        tool_kwargs = (
            {"tools": tools, "parallel_tool_calls": False} if tools else {}
        )

        response = await copilotkit_stream(
            await acompletion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *state["messages"],
                ],
                stream=True,
                **tool_kwargs,
            )
        )
        message = response.choices[0].message
        state["messages"].append(message)

        if not (plan and message.tool_calls):
            return

        for tool_call in message.tool_calls:
            if tool_call.function.name != plan["tool_name"]:
                continue
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            # Runs the render sub-agent through the recovery loop, streaming its
            # render_a2ui progress to the wire; returns the operations envelope.
            envelope = await plan["tool"].run(args)
            state["messages"].append(
                {
                    "role": "tool",
                    "content": envelope,
                    "tool_call_id": tool_call.id,
                }
            )
