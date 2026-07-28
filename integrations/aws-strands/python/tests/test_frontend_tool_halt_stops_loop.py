"""The frontend-tool halt must stop the Strands loop, not just mute the wire.

A proxy (frontend) tool returns a SUCCESSFUL ``"Forwarded to client"``
placeholder server-side, so Strands has every reason to run another model cycle
on that placeholder — and another. Gating only event *emission* leaves those
cycles running: frontend tool calls the client never sees (and therefore can
never answer), real backend side effects, phantom assistant turns written to
the session store, and RUN_FINISHED queued behind work nobody is watching.
Single-agent Strands has no cycle cap, so a model that keeps retrying the read
never produces a terminal event at all.

These tests drive the REAL ``StrandsAgent`` adapter over a REAL
``strands.Agent`` event loop with a scripted stub model (no network, no
credentials, deterministic cycle count), so the model-invocation count is
direct evidence about the loop rather than about a mocked stream.
"""

from __future__ import annotations

import asyncio

import pytest
from ag_ui.core import EventType, RunAgentInput, Tool, UserMessage
from strands import Agent
from strands.models.model import Model
from strands.session.file_session_manager import FileSessionManager

from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.client_proxy_tool import PROXY_RESULT_PLACEHOLDER
from ag_ui_strands.config import StrandsAgentConfig, ToolBehavior
from ag_ui_strands.session_reconcile import AG_UI_WIRE_MAP_STATE_KEY

# Ceiling on model invocations. The halt should stop the loop after ONE, so any
# regression trips this and fails loudly instead of hanging the suite.
_RUNAWAY_GUARD = 12


class _ScriptedModel(Model):
    """Stub model that keeps calling the frontend tool until told to stop.

    * turn 1 — ONE assistant message carrying ``first_turn_tools`` as parallel
      ``toolUse`` blocks (the reported repro: a read and a write together).
    * turns 2..``follow_ups``+1 — another ``get_cell`` call, modelling a model
      that received only ``"Forwarded to client"`` and retries the read.
    * afterwards — plain text, ``end_turn``.

    ``follow_ups=None`` never stops calling the tool, which is how a run with a
    drained (rather than halted) loop is shown to have no terminal event.
    """

    def __init__(
        self,
        *,
        follow_ups: int | None = 2,
        raise_on_call: int | None = None,
        first_turn_tools: tuple[str, ...] = ("get_cell", "update_cell"),
    ):
        self.calls = 0
        self.follow_ups = follow_ups
        self.raise_on_call = raise_on_call
        self.first_turn_tools = first_turn_tools

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(self, output_model, prompt, **kwargs):  # pragma: no cover
        if False:
            yield {}

    @staticmethod
    def _tool_use(tool_use_id: str, name: str):
        args = '{"cell":"B4"}' if name == "get_cell" else '{"cell":"B5","value":"?"}'
        return [
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": args}}}},
            {"contentBlockStop": {}},
        ]

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        turn = self.calls
        if turn > _RUNAWAY_GUARD:
            raise RuntimeError(
                f"model invoked {turn} times: the frontend-tool halt did not stop the loop"
            )
        if self.raise_on_call is not None and turn == self.raise_on_call:
            raise RuntimeError("simulated model failure on a post-halt cycle")

        yield {"messageStart": {"role": "assistant"}}
        if turn == 1:
            for index, name in enumerate(self.first_turn_tools):
                for event in self._tool_use(f"native-{name}-1", name):
                    yield event
            yield {"messageStop": {"stopReason": "tool_use"}}
        elif self.follow_ups is None or turn <= 1 + self.follow_ups:
            for event in self._tool_use(f"native-get_cell-{turn}", "get_cell"):
                yield event
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": "Done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def _frontend_tools() -> list[Tool]:
    return [
        Tool(
            name="get_cell",
            description="Read a cell",
            parameters={
                "type": "object",
                "properties": {"cell": {"type": "string"}},
                "required": ["cell"],
            },
        ),
        Tool(
            name="update_cell",
            description="Write a cell",
            parameters={
                "type": "object",
                "properties": {"cell": {"type": "string"}, "value": {"type": "string"}},
                "required": ["cell", "value"],
            },
        ),
    ]


def _run_input(thread_id: str) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id="r-1",
        parent_run_id=None,
        state={},
        messages=[UserMessage(id="u1", role="user", content="Could you duplicate my B4 cell?")],
        tools=_frontend_tools(),
        context=[],
        forwarded_props={},
    )


async def _collect(adapter: StrandsAgent, thread_id: str) -> list:
    """Run to completion, failing fast rather than hanging on a regression."""

    async def drive():
        return [event async for event in adapter.run(_run_input(thread_id))]

    return await asyncio.wait_for(drive(), timeout=30)


def _build(model: Model, config: StrandsAgentConfig | None = None) -> StrandsAgent:
    return StrandsAgent(Agent(model=model, tools=[]), name="halt-test-agent", config=config)


@pytest.mark.asyncio
async def test_halt_stops_the_strands_loop_after_one_model_cycle():
    """No model cycle may run after the halt latches."""
    model = _ScriptedModel(follow_ups=2)
    events = await _collect(_build(model), "t-stops")

    assert model.calls == 1, (
        f"model invoked {model.calls}x; cycles after the halt run invisibly to the client"
    )
    assert any(e.type == EventType.RUN_FINISHED for e in events)


@pytest.mark.asyncio
async def test_halt_emits_run_finished_when_the_model_would_never_stop():
    """A model that keeps retrying the read must not strand the run.

    Draining instead of halting produces NO terminal event here at all, because
    single-agent Strands has no cycle cap.
    """
    model = _ScriptedModel(follow_ups=None)
    events = await _collect(_build(model), "t-unbounded")

    assert model.calls == 1
    assert any(e.type == EventType.RUN_FINISHED for e in events)
    assert not any(e.type == EventType.RUN_ERROR for e in events)


@pytest.mark.asyncio
async def test_post_halt_model_failure_cannot_replace_run_finished():
    """A fault on a cycle the client never sees must not become its RUN_ERROR.

    Draining reaches the failing cycle and surfaces EventLoopException as
    RUN_ERROR, leaving the client holding tool calls it can never answer.
    """
    model = _ScriptedModel(follow_ups=2, raise_on_call=2)
    events = await _collect(_build(model), "t-error")

    assert model.calls == 1
    assert any(e.type == EventType.RUN_FINISHED for e in events)
    assert not any(e.type == EventType.RUN_ERROR for e in events)


@pytest.mark.asyncio
async def test_both_parallel_frontend_calls_reach_the_wire():
    """The halt must not truncate a parallel batch.

    It latches at the tool-RESULT boundary, so every ``toolUse`` block in the
    assistant message is emitted first. Covered against a mocked stream in
    ``test_parallel_tool_call_handling.py``; this pins it against the real loop.
    """
    model = _ScriptedModel(follow_ups=2)
    events = await _collect(_build(model), "t-parallel")

    started = [
        e.tool_call_name for e in events if e.type == EventType.TOOL_CALL_START
    ]
    assert started == ["get_cell", "update_cell"]


@pytest.mark.asyncio
async def test_continue_after_frontend_call_still_runs_further_cycles():
    """Tools that opt out of halting keep the loop running."""
    model = _ScriptedModel(follow_ups=1, first_turn_tools=("get_cell",))
    config = StrandsAgentConfig(
        tool_behaviors={"get_cell": ToolBehavior(continue_after_frontend_call=True)}
    )
    events = await _collect(_build(model, config), "t-continue")

    assert model.calls > 1, "continue_after_frontend_call must not halt the loop"
    assert any(e.type == EventType.RUN_FINISHED for e in events)


def _persisted_messages(sm: FileSessionManager, session_id: str) -> list[dict]:
    return [m.message for m in sm.list_messages(session_id, "default")]


@pytest.mark.asyncio
async def test_halted_turn_persists_tool_use_placeholder_and_wire_map(tmp_path):
    """Stopping the loop must not cost the next run's reconcile inputs.

    The reconcile overwrites a persisted placeholder ``toolResult``, keyed via
    the wire->native map on agent state. Both are written before the halt
    latches (``MessageAddedEvent`` drives ``append_message`` AND ``sync_agent``
    — see ``SessionManager.register_hooks``), so halting keeps them. If Strands
    ever stops syncing state on message-added, this test is the tripwire.

    Asserting EXACTLY one persisted ``toolUse`` also pins the other half: a
    drained loop persists the invisible retry calls too, leaving the store with
    several unresolved placeholders the client was never asked about.
    """
    session_id = "s-halt"
    sm = FileSessionManager(session_id=session_id, storage_dir=str(tmp_path))
    model = _ScriptedModel(follow_ups=2, first_turn_tools=("get_cell",))
    adapter = _build(model, StrandsAgentConfig(session_manager_provider=lambda _tid: sm))

    await _collect(adapter, "t-persist")

    messages = _persisted_messages(sm, session_id)
    tool_uses = [
        block["toolUse"]
        for message in messages
        for block in message.get("content") or []
        if isinstance(block, dict) and "toolUse" in block
    ]
    placeholders = [
        block["toolResult"]
        for message in messages
        for block in message.get("content") or []
        if isinstance(block, dict)
        and "toolResult" in block
        and any(
            PROXY_RESULT_PLACEHOLDER in (part.get("text") or "")
            for part in block["toolResult"].get("content") or []
            if isinstance(part, dict)
        )
    ]

    assert [t["name"] for t in tool_uses] == ["get_cell"]
    assert len(placeholders) == 1
    assert placeholders[0]["toolUseId"] == tool_uses[0]["toolUseId"]

    persisted_agent = sm.read_agent(session_id, "default")
    wire_map = persisted_agent.state.get(AG_UI_WIRE_MAP_STATE_KEY) or {}
    assert tool_uses[0]["toolUseId"] in wire_map.values(), (
        "reconcile cannot locate the placeholder without the wire->native map"
    )


@pytest.mark.asyncio
async def test_halted_turn_persists_no_assistant_turn_the_client_never_saw(tmp_path):
    """Draining writes phantom history: the model's post-halt answer lands in
    the session store even though it never reached the wire, so the next turn
    starts from a transcript claiming the question was already answered."""
    session_id = "s-phantom"
    sm = FileSessionManager(session_id=session_id, storage_dir=str(tmp_path))
    model = _ScriptedModel(follow_ups=0, first_turn_tools=("get_cell",))
    adapter = _build(model, StrandsAgentConfig(session_manager_provider=lambda _tid: sm))

    events = await _collect(adapter, "t-phantom")

    emitted_text = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
    persisted_text = [
        block["text"]
        for message in _persisted_messages(sm, session_id)
        if message.get("role") == "assistant"
        for block in message.get("content") or []
        if isinstance(block, dict) and "text" in block
    ]

    assert emitted_text == []
    assert persisted_text == [], f"assistant text persisted but never emitted: {persisted_text}"
