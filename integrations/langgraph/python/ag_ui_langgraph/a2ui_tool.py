"""
A2UI subagent tool factory for LangGraph agents.

Thin adapter over ``ag-ui-a2ui-toolkit`` — the heavy lifting (op builders,
prompt assembly, history walkers, output envelope) lives in the toolkit so
each new framework adapter (ADK, Mastra, Strands, …) only owns the
framework-specific glue: tool decorator, runtime state access, model
binding + invoke.

Streaming: the subagent's ``render_a2ui`` call must STREAM to the AG-UI wire —
the a2ui middleware's "building" skeleton and progressive paint key off the
inner tool-call's arg deltas, not the final result. A prior assumption that a
nested ``model.stream()`` would auto-surface via the graph's
``OnChatModelStream`` is FALSE — those deltas do not propagate, so this adapter
emits them EXPLICITLY. It mirrors the Strands adapter's per-delta ``push(...)``:
where Strands re-yields ``ToolStreamEvent`` payloads that its agent.ts turns
into inner TOOL_CALL_START/ARGS/END, this adapter dispatches granular
``a2ui_render_{start,args,end}`` custom events (via LangGraph's
``adispatch_custom_event``) that ``agent.py``'s OnCustomEvent handler turns into
the same inner TOOL_CALL_START/ARGS/END on the wire. That is the channel the
adapter ALREADY uses for manually-emitted tool calls — no new transport.

Example usage in a chat node::

    from ag_ui_langgraph import get_a2ui_tools

    a2ui = get_a2ui_tools({"model": ChatOpenAI(model="gpt-4o")})

    model_with_tools = chat_model.bind_tools(
        [*state["tools"], a2ui],
        parallel_tool_calls=False,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Optional

from langchain.tools import tool, ToolRuntime
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import SystemMessage

from ag_ui_a2ui_toolkit import (
    A2UI_OPERATIONS_KEY,
    A2UIGuidelines,
    A2UIToolParams,
    BASIC_CATALOG_ID,
    RENDER_A2UI_TOOL_DEF,
    build_a2ui_envelope,
    prepare_a2ui_request,
    resolve_a2ui_tool_params,
    wrap_error_envelope,
    run_a2ui_generation_with_recovery,
)

from .types import CustomEventNames

logger = logging.getLogger("ag_ui_langgraph")

#: Name of the render tool the A2UI middleware injects (and the subagent binds).
RENDER_A2UI_TOOL_NAME: str = RENDER_A2UI_TOOL_DEF["function"]["name"]


# Re-export the toolkit constants/types for callers that previously imported
# them from this package — keeps the public surface stable and lets consumers
# type the shared params object + its guidelines without depending on the
# toolkit package directly.
__all__ = [
    "get_a2ui_tools",
    "A2UI_OPERATIONS_KEY",
    "A2UIToolParams",
    "A2UIGuidelines",
    "BASIC_CATALOG_ID",
]


async def _stream_render_subagent(
    model_with_tool: Any,
    prompt: str,
    messages: list,
    push: Callable[[dict], Any],
) -> Optional[dict]:
    """Run the structured-output subagent once: stream the model, push per-event
    render progress (start / args deltas / end) via ``push``, and return the
    captured ``render_a2ui`` args — or ``None`` if the model produced no call.

    Mirrors the Strands adapter's ``_stream_render_subagent``: ``push`` is the
    LangGraph analogue of Strands' per-delta callback. ``args`` on each streamed
    ``ToolCallChunk`` is the INCREMENTAL JSON fragment, re-emitted as one
    ``"args"`` delta; the fragments accumulate (via chunk addition) into the
    final ``render_a2ui`` args returned to the recovery loop.
    """
    live_call_id: Optional[str] = None
    accumulated = None
    # Per-invocation fallback id: providers that never stamp a tool-call id must
    # not reuse one literal id across recovery attempts (two full lifecycles
    # under one toolCallId would mis-merge in id-keyed consumers).
    fallback_call_id = f"a2ui-render-{uuid.uuid4().hex[:8]}"

    def _chunk_field(chunk: Any, key: str) -> Any:
        if isinstance(chunk, dict):
            return chunk.get(key)
        return getattr(chunk, key, None)

    try:
        async for chunk in model_with_tool.astream(
            [SystemMessage(content=prompt), *messages]
        ):
            # Accumulate the streamed AIMessageChunks so the final parsed
            # tool_calls (the captured args) reconstruct even when each frame
            # only carries an incremental arg fragment.
            accumulated = chunk if accumulated is None else accumulated + chunk

            tool_call_chunks = _chunk_field(chunk, "tool_call_chunks") or []
            for tcc in tool_call_chunks:
                name = _chunk_field(tcc, "name")
                # Only the render call drives the synthetic stream; ignore any
                # foreign tool fragments (the subagent is tool_choice-pinned to
                # render_a2ui, but stay defensive).
                if name is not None and name != RENDER_A2UI_TOOL_NAME:
                    continue
                raw_id = _chunk_field(tcc, "id")
                call_id = raw_id or live_call_id or fallback_call_id
                if live_call_id == fallback_call_id and raw_id:
                    # Provider delivered the real id only after id-less frames:
                    # same logical call — keep the latched fallback id so the
                    # synthetic stream stays continuous (no spurious end/start).
                    call_id = live_call_id
                if call_id != live_call_id:
                    # New render call (normally the only one). Close any previous
                    # call first so streamed arg deltas never mis-attribute
                    # across ids (mirrors the Strands per-call reset).
                    if live_call_id is not None:
                        await push({"kind": "end", "tool_call_id": live_call_id})
                    live_call_id = call_id
                    await push(
                        {
                            "kind": "start",
                            "tool_call_id": call_id,
                            "tool_call_name": RENDER_A2UI_TOOL_NAME,
                        }
                    )
                args = _chunk_field(tcc, "args")
                if isinstance(args, str) and args:
                    await push(
                        {"kind": "args", "tool_call_id": live_call_id, "delta": args}
                    )
    except BaseException:
        # The provider stream died mid-call (model 429, network drop, ...):
        # close the live synthetic call before unwinding — an unclosed inner
        # TOOL_CALL_START is a wire-protocol violation, and the next recovery
        # attempt would open a fresh call on top of it.
        if live_call_id is not None:
            try:
                await push({"kind": "end", "tool_call_id": live_call_id})
            except BaseException:
                # A push failure during unwind must not REPLACE the original
                # exception (e.g. a CancelledError) mid-teardown.
                pass
        raise

    captured: Optional[dict] = None
    if accumulated is not None:
        tool_calls = _chunk_field(accumulated, "tool_calls") or []
        for call in tool_calls:
            call_name = call.get("name") if isinstance(call, dict) else None
            if call_name in (None, RENDER_A2UI_TOOL_NAME):
                raw_args = call.get("args") if isinstance(call, dict) else None
                captured = raw_args if isinstance(raw_args, dict) else {}
                break

    if live_call_id is not None:
        # Some providers deliver parsed tool_calls without streaming arg
        # fragments (no "args" deltas pushed). Emit the captured args as a
        # single delta so the middleware still sees components before the
        # result (no bulk paint).
        if captured is not None and not _any_args_streamed(accumulated):
            await push(
                {
                    "kind": "args",
                    "tool_call_id": live_call_id,
                    "delta": json.dumps(captured),
                }
            )
        await push({"kind": "end", "tool_call_id": live_call_id})
    elif captured is not None:
        # The provider returned the render_a2ui call without emitting ANY
        # tool_call_chunks: synthesize the full triplet so the middleware still
        # sees components before the result (no bulk paint).
        live_call_id = fallback_call_id
        await push(
            {
                "kind": "start",
                "tool_call_id": live_call_id,
                "tool_call_name": RENDER_A2UI_TOOL_NAME,
            }
        )
        await push(
            {
                "kind": "args",
                "tool_call_id": live_call_id,
                "delta": json.dumps(captured),
            }
        )
        await push({"kind": "end", "tool_call_id": live_call_id})

    return captured


def _any_args_streamed(accumulated: Any) -> bool:
    """True if the accumulated chunk carries any non-empty streamed arg
    fragment — i.e. the synthetic "args" deltas already covered the surface and
    a captured-args fallback delta would duplicate them."""
    if accumulated is None:
        return False
    chunks = (
        accumulated.get("tool_call_chunks")
        if isinstance(accumulated, dict)
        else getattr(accumulated, "tool_call_chunks", None)
    ) or []
    for tcc in chunks:
        args = tcc.get("args") if isinstance(tcc, dict) else getattr(tcc, "args", None)
        if isinstance(args, str) and args:
            return True
    return False


def get_a2ui_tools(params: A2UIToolParams):
    """Build a LangGraph tool that delegates A2UI surface generation to a subagent.

    The returned tool is decorated with ``@langchain.tools.tool`` and is
    ready to bind into a chat model alongside any other tools.

    Args:
        params: Shared ``A2UIToolParams`` (``model`` + behavior knobs). The
            toolkit owns the shape and fills defaults via
            ``resolve_a2ui_tool_params``. Every framework adapter takes this
            exact params type — only the body below is LangGraph-specific, so a
            new knob added to ``A2UIToolParams`` reaches this adapter with no
            signature change.

    Returns:
        A LangGraph tool callable suitable for ``bind_tools(...)``.
    """
    # Shared: normalize knobs + fill canonical defaults so this adapter never
    # re-implements default logic. A new params field + its default lives
    # entirely in the toolkit.
    cfg = resolve_a2ui_tool_params(params)
    model = cfg["model"]
    guidelines = cfg["guidelines"]
    default_surface_id = cfg["default_surface_id"]
    default_catalog_id = cfg["default_catalog_id"]
    catalog = cfg["catalog"]
    recovery = cfg["recovery"]
    on_a2ui_attempt = cfg["on_a2ui_attempt"]

    @tool(cfg["tool_name"], description=cfg["tool_description"])
    async def generate_a2ui(
        runtime: ToolRuntime[Any],
        intent: str = "create",
        target_surface_id: Optional[str] = None,
        changes: Optional[str] = None,
    ) -> str:
        """Generate or edit an A2UI surface.

        Args:
            intent: Either ``"create"`` to render a new surface, or ``"update"``
                to modify a surface previously rendered in this conversation.
            target_surface_id: Required when ``intent="update"``. The surface
                id of the prior render to modify.
            changes: Optional natural-language description of the changes to
                apply when ``intent="update"``.
        """
        # Defensive: a custom state schema may not preseed ``messages``, and
        # ``state["messages"]`` would then raise KeyError mid-tool — mirror the
        # TS adapter's `state.messages ?? []` graceful-degrade.
        messages = runtime.state.get("messages", [])[:-1]

        # Shared: decide create/update, find prior surface, build the prompt.
        prep = prepare_a2ui_request(
            intent=intent,
            target_surface_id=target_surface_id,
            changes=changes,
            messages=messages,
            state=runtime.state,
            guidelines=guidelines,
        )
        if prep.get("error"):
            return wrap_error_envelope(prep["error"])

        # Glue: bind the structured-output tool.
        model_with_tool = model.bind_tools(
            [RENDER_A2UI_TOOL_DEF], tool_choice="render_a2ui"
        )

        # The LangGraph analogue of the Strands adapter's `push`: surface each
        # render-stream step as a granular custom event on the run's config so
        # it routes through astream_events -> OnCustomEvent -> the inner
        # TOOL_CALL_START/ARGS/END the a2ui middleware paints from. `config` is
        # threaded explicitly (mirrors the example nodes' adispatch_custom_event
        # calls) so the events land on THIS run's stream — and so the dispatch
        # works when marshaled back onto the outer loop from the worker thread.
        config = getattr(runtime, "config", None)

        async def _dispatch(step: dict) -> None:
            kind = step["kind"]
            try:
                if kind == "start":
                    await adispatch_custom_event(
                        CustomEventNames.A2UIRenderStart.value,
                        {"id": step["tool_call_id"], "name": step["tool_call_name"]},
                        config=config,
                    )
                elif kind == "args":
                    await adispatch_custom_event(
                        CustomEventNames.A2UIRenderArgs.value,
                        {"id": step["tool_call_id"], "delta": step["delta"]},
                        config=config,
                    )
                elif kind == "end":
                    await adispatch_custom_event(
                        CustomEventNames.A2UIRenderEnd.value,
                        {"id": step["tool_call_id"]},
                        config=config,
                    )
            except RuntimeError as err:
                # ``adispatch_custom_event`` raises when there is no parent run
                # id to associate the event with — i.e. the tool was invoked
                # outside a graph run (no astream_events consumer to paint to).
                # The surface still generates from the captured args; there is
                # simply no live stream to surface the deltas onto, so degrade
                # to a no-op rather than crashing the generation.
                if "parent run id" not in str(err):
                    raise
                logger.debug(
                    "A2UI render stream step %r not surfaced (no parent run "
                    "id; tool invoked outside a graph run): %s",
                    kind,
                    err,
                )

        # The subagent streams on a worker-thread event loop (the sync recovery
        # loop runs there via ``asyncio.run``), but the run's callback manager —
        # the astream_events queue that turns these into wire events — lives on
        # the OUTER loop. Marshal each dispatch back onto the outer loop (the
        # LangGraph analogue of the Strands adapter's ``call_soon_threadsafe``
        # push) and await it so back-pressure and ordering hold. When no outer
        # loop is running (direct unit-test invocation of the inner stream), the
        # subagent awaits ``_dispatch`` directly on its own loop.
        outer_loop = asyncio.get_running_loop()

        async def _push(step: dict) -> None:
            fut = asyncio.run_coroutine_threadsafe(_dispatch(step), outer_loop)
            # Bridge the concurrent.futures.Future back to this worker loop
            # without blocking it (which would deadlock single-threaded test
            # loops); poll cooperatively.
            await asyncio.wrap_future(fut)

        async def _invoke_subagent(prompt, _attempt):
            return await _stream_render_subagent(
                model_with_tool, prompt, messages, _push
            )

        def _build_envelope(args):
            return build_a2ui_envelope(
                args=args,
                is_update=prep["is_update"],
                target_surface_id=target_surface_id,
                prior=prep["prior"],
                default_surface_id=default_surface_id,
                default_catalog_id=default_catalog_id,
            )

        # Shared: validate->retry loop (mirrors the TS adapter). On each retry the
        # prompt is re-augmented with the prior attempt's structured errors; only a
        # validated surface is committed (the middleware gate suppresses any
        # unvalidated attempt, so a rejected one never paints). Returns a structured
        # hard-failure envelope once the attempt cap is hit.
        #
        # The recovery loop is synchronous and calls ``invoke_subagent`` (here the
        # async streaming subagent) per attempt. Run it in a worker thread so its
        # blocking ``asyncio.run`` doesn't collide with THIS running event loop;
        # the pushed custom events are marshaled back onto the outer loop so they
        # land on the run's stream (see ``_push``).
        result = await asyncio.to_thread(
            run_a2ui_generation_with_recovery,
            base_prompt=prep["prompt"],
            catalog=catalog,
            config=recovery,
            invoke_subagent=lambda prompt, attempt: asyncio.run(
                _invoke_subagent(prompt, attempt)
            ),
            build_envelope=_build_envelope,
            on_attempt=on_a2ui_attempt,
        )
        return result["envelope"]

    return generate_a2ui
