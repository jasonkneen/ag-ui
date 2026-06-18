"""Integration tests for the LangGraph A2UI tool factory (``get_a2ui_tools``).

These run in the ``langgraph-python`` unit job, which builds the LOCAL adapter
and (via the adapter's ``[tool.uv.sources]`` path) the LOCAL toolkit — so they
exercise the real in-repo code. The dojo e2e suite can't cover this: it installs
the PUBLISHED ``ag-ui-langgraph`` (the langgraph-cloud build rejects local path
deps that escape the examples root), so the new single-arg ``A2UIToolParams`` /
``guidelines`` surface has no e2e coverage until it ships. This file is that
coverage.

A lightweight fake chat model STREAMS a fixed ``render_a2ui`` tool call as
several ``AIMessageChunk``s (mirroring how a real provider streams tool-call arg
fragments). The tests assert both the emitted operations envelope and that the
generation/design/composition guidance reaches the subagent — and, critically,
that the inner render call is surfaced as PROGRESSIVE TOOL_CALL_ARGS deltas (the
parity fix), not one bulk paint at the end.
"""

from __future__ import annotations

import asyncio
import json
import unittest

from langchain_core.messages import AIMessageChunk
from langchain_core.messages.tool import tool_call_chunk

from ag_ui_langgraph import get_a2ui_tools
from ag_ui_langgraph.a2ui_tool import _stream_render_subagent
from ag_ui_a2ui_toolkit import (
    A2UI_OPERATIONS_KEY,
    DEFAULT_DESIGN_GUIDELINES,
    DEFAULT_GENERATION_GUIDELINES,
)


# A structurally-valid render_a2ui result (root present, child resolves, no
# cycle) so the toolkit's recovery/validation commits on the first attempt.
VALID_ARGS = {
    "surfaceId": "s1",
    "components": [
        {"id": "root", "component": "Column", "children": ["t"]},
        {"id": "t", "component": "Text", "text": "hi"},
    ],
    "data": {},
}


def _arg_chunks(args: dict, parts: int = 3) -> list[str]:
    """Split the JSON of ``args`` into ``parts`` non-empty fragments, the way a
    provider streams tool-call arg deltas."""
    text = json.dumps(args)
    size = max(1, len(text) // parts)
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    return chunks or [text]


class _StreamingBoundModel:
    """What ``model.bind_tools(...)`` returns — records the system prompt it is
    streamed with and replays a fixed ``render_a2ui`` tool call as several
    ``AIMessageChunk``s (one per arg fragment), like a real streaming provider."""

    def __init__(self, parent: "FakeModel"):
        self._parent = parent

    async def astream(self, messages):
        # The adapter streams with [SystemMessage(prompt), *history]; capture the
        # system prompt so tests can assert what guidance the subagent saw.
        self._parent.captured_prompts.append(messages[0].content)
        fragments = _arg_chunks(self._parent.args)
        call_id = "call-1"
        for index, fragment in enumerate(fragments):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    tool_call_chunk(
                        # Name + id only on the first fragment, mirroring how
                        # providers stamp them once at the start of the call.
                        name="render_a2ui" if index == 0 else None,
                        args=fragment,
                        id=call_id if index == 0 else None,
                        index=0,
                    )
                ],
            )


class FakeModel:
    """Minimal chat-model stand-in: only ``bind_tools`` + ``astream`` are used."""

    def __init__(self, args):
        self.args = args
        self.captured_prompts: list[str] = []

    def bind_tools(self, tools, tool_choice=None):
        return _StreamingBoundModel(self)


class FakeRuntime:
    """Stand-in for LangGraph's ``ToolRuntime`` — the tool reads ``state`` and
    ``config`` (the latter forwarded to ``adispatch_custom_event``)."""

    def __init__(self, state, config=None):
        self.state = state
        self.config = config


def _invoke_tool(tool, runtime, **kwargs) -> str:
    """Drive the tool's async coroutine directly with a stub runtime, bypassing
    the graph's runtime injection. Runs to completion on a fresh event loop."""
    return asyncio.run(tool.coroutine(runtime, **kwargs))


class TestGetA2UITools(unittest.TestCase):
    def _make(self, guidelines=None, tool_name=None):
        model = FakeModel(VALID_ARGS)
        params = {"model": model, "default_catalog_id": "cat://custom"}
        if guidelines is not None:
            params["guidelines"] = guidelines
        if tool_name is not None:
            params["tool_name"] = tool_name
        return get_a2ui_tools(params), model

    def test_single_arg_params_produces_operations_envelope(self):
        # Guards the exact regression that broke CI: the factory must accept a
        # single A2UIToolParams dict (model inside) and drive a render.
        tool, _model = self._make()
        envelope = _invoke_tool(
            tool, FakeRuntime({"messages": []}), intent="create"
        )
        parsed = json.loads(envelope)
        ops = parsed[A2UI_OPERATIONS_KEY]
        self.assertTrue(any("createSurface" in o for o in ops))
        self.assertTrue(any("updateComponents" in o for o in ops))
        # Catalog ownership stays with the host (from params), never the model.
        create = next(o for o in ops if "createSurface" in o)
        self.assertEqual(create["createSurface"]["catalogId"], "cat://custom")

    def test_default_guidelines_reach_the_subagent_prompt(self):
        # No guidelines passed → the built-in generation + design defaults must
        # be injected into the subagent system prompt (OSS-248 re-enable).
        tool, model = self._make()
        _invoke_tool(tool, FakeRuntime({"messages": []}), intent="create")
        prompt = model.captured_prompts[0]
        self.assertIn(DEFAULT_GENERATION_GUIDELINES, prompt)
        self.assertIn("## Design Guidelines", prompt)
        self.assertIn(DEFAULT_DESIGN_GUIDELINES, prompt)

    def test_composition_guide_and_overrides_flow_through(self):
        tool, model = self._make(
            guidelines={
                "generation_guidelines": "CUSTOM_GEN",
                "composition_guide": "COMPMARK",
            }
        )
        _invoke_tool(tool, FakeRuntime({"messages": []}), intent="create")
        prompt = model.captured_prompts[0]
        # Per-field override replaces generation; design keeps its default.
        self.assertIn("CUSTOM_GEN", prompt)
        self.assertNotIn(DEFAULT_GENERATION_GUIDELINES, prompt)
        self.assertIn(DEFAULT_DESIGN_GUIDELINES, prompt)
        self.assertIn("COMPMARK", prompt)

    def test_tool_name_resolves(self):
        default_tool, _ = self._make()
        self.assertEqual(default_tool.name, "generate_a2ui")
        custom_tool, _ = self._make(tool_name="render_ui")
        self.assertEqual(custom_tool.name, "render_ui")


class TestStreamRenderSubagent(unittest.TestCase):
    """The parity fix: the inner render_a2ui call must be surfaced as PROGRESSIVE
    start -> many args deltas -> end, mirroring the Strands adapter — not one
    final bulk push."""

    def _run_stream(self, model_args, num_parts=4):
        model = FakeModel(model_args)
        # _stream_render_subagent expects an already-bound model (bind_tools is
        # done by the factory); the fake's bound model ignores the tool def.
        bound = model.bind_tools([])
        pushed: list[dict] = []

        async def _push(step: dict) -> None:
            pushed.append(step)

        captured = asyncio.run(
            _stream_render_subagent(bound, "PROMPT", [], _push)
        )
        return captured, pushed

    def test_progressive_deltas_are_pushed(self):
        captured, pushed = self._run_stream(VALID_ARGS)

        kinds = [p["kind"] for p in pushed]
        # Exactly one start, one end, and MULTIPLE args deltas in between —
        # this is the whole point: incremental emission, not one bulk paint.
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "end")
        self.assertEqual(kinds.count("start"), 1)
        self.assertEqual(kinds.count("end"), 1)
        args_deltas = [p for p in pushed if p["kind"] == "args"]
        self.assertGreater(
            len(args_deltas),
            1,
            "expected multiple incremental args deltas (progressive paint), "
            f"got {len(args_deltas)}",
        )

        # The start carries the render tool name + a stable id; every delta and
        # the end reuse that same id.
        start = pushed[0]
        self.assertEqual(start["tool_call_name"], "render_a2ui")
        call_id = start["tool_call_id"]
        self.assertTrue(all(p["tool_call_id"] == call_id for p in pushed))

        # Concatenating the streamed deltas reconstructs the full render args
        # JSON — the deltas ARE the surface, not a placeholder.
        joined = "".join(p["delta"] for p in args_deltas)
        self.assertEqual(json.loads(joined), VALID_ARGS)

        # And the captured args (fed to the recovery loop / envelope) parse back
        # to the same surface.
        self.assertEqual(captured["surfaceId"], "s1")

    def test_captured_args_returned_for_envelope(self):
        captured, _ = self._run_stream(VALID_ARGS)
        self.assertEqual(captured, VALID_ARGS)


if __name__ == "__main__":
    unittest.main()
