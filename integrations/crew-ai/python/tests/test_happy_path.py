"""Happy-path event-payload coverage for the CrewAI integration (CPK-7716).

The pre-existing suite (``test_task_cancellation.py`` /
``test_llm_timeout.py``) is exhaustive on the failure surface —
cancellation, disconnect, timeout ceilings, error wire-format — but
asserts almost nothing about the *nominal* path: what a well-behaved run
actually puts on the wire. These tests pin the CURRENT correct behaviour
of the happy path so the CrewAI Lane 1 rewrite (CPK-7718 and later) has a
regression net.

Boundaries mocked here mirror the existing suite: we patch ``acompletion``
/ ``copilotkit_stream`` inside ``crews`` (no live LLM), drive the crewai
event bus directly for the endpoint listener translation, and exercise the
pure helpers with plain fixtures. Nothing here reaches the network.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ag_ui.core import EventType, SystemMessage, Tool, UserMessage
from ag_ui_crewai import endpoint as ep
from ag_ui_crewai.context import flow_context
from ag_ui_crewai.events import (
    BridgedTextMessageChunkEvent,
    BridgedToolCallChunkEvent,
)
from ag_ui_crewai.sdk import (
    _copilotkit_stream_custom_stream_wrapper,
    copilotkit_emit_state,
    copilotkit_predict_state,
    copilotkit_stream,
    litellm_messages_to_ag_ui_messages,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@contextmanager
def _patch_instance_state(flow, state):
    """Install ``state`` on a single flow instance via a throwaway subclass.

    Copied from ``test_llm_timeout.py``: ``Flow.state`` is a class-level
    descriptor, so we rebind ``__class__`` to a per-instance subclass that
    exposes ``state`` as a plain property. Per-instance, so parallel tests
    cannot race on the shared descriptor.
    """
    flow._state = state  # pylint: disable=protected-access
    original_cls = type(flow)
    subclass = type(
        f"{original_cls.__name__}_StatePatched",
        (original_cls,),
        {"state": property(lambda self: self._state)},
    )
    flow.__class__ = subclass
    try:
        yield
    finally:
        if flow.__class__ is subclass:
            flow.__class__ = original_cls


def _stream_chunk(chunk_id, *, content=None, tool_calls=None, finish_reason=None):
    """Build a LiteLLM-shaped streaming chunk.

    ``copilotkit_stream`` reassembly reads ``chunk["id"]``,
    ``chunk["choices"][0]["delta"]["content"|"tool_calls"]``, the finish
    reason and the ``created``/``model``/``system_fingerprint`` scalars.
    Tool-call entries are accessed attribute-style (``.id``) with a dict
    ``.function`` — hence ``SimpleNamespace`` for those.
    """
    return {
        "id": chunk_id,
        "created": 1700000000,
        "model": "gpt-4o",
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "delta": {"content": content, "tool_calls": tool_calls},
                "finish_reason": finish_reason,
            }
        ],
    }


def _tool_call_delta(*, call_id, name, arguments):
    return SimpleNamespace(id=call_id, function={"name": name, "arguments": arguments})


class _FakeFlow:
    """Minimal flow stand-in the endpoint listener can route events to.

    Accepts arbitrary attribute writes (``create_queue`` stamps
    ``_agui_queue_key``) and carries a ``.state`` the
    MethodExecutionFinished handler reads.
    """

    def __init__(self, state=None):
        self.state = state if state is not None else {}


def _drain(queue):
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


# --------------------------------------------------------------------------
# litellm_messages_to_ag_ui_messages (sdk)
# --------------------------------------------------------------------------

def test_litellm_conversion_whitelists_and_strips_none():
    """Only whitelisted keys survive; ``None`` values and unknown keys are
    dropped; ``id``/``role``/``content`` are preserved verbatim."""
    out = litellm_messages_to_ag_ui_messages(
        [{"role": "assistant", "content": "hi", "id": "a1",
          "name": None, "unknown_field": "dropme"}]
    )
    assert len(out) == 1
    dumped = out[0].model_dump()
    assert dumped["id"] == "a1"
    assert dumped["role"] == "assistant"
    assert dumped["content"] == "hi"
    assert "unknown_field" not in dumped


def test_litellm_conversion_generates_id_when_missing():
    """A message without an ``id`` gets a generated UUID string."""
    out = litellm_messages_to_ag_ui_messages([{"role": "user", "content": "yo"}])
    assert isinstance(out[0].id, str)
    assert len(out[0].id) == 36  # canonical uuid4 string length


def test_litellm_conversion_injects_tool_call_type():
    """Tool calls missing an explicit ``type`` are stamped ``function``."""
    out = litellm_messages_to_ag_ui_messages(
        [{
            "role": "assistant",
            "id": "a2",
            "content": None,
            "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}],
        }]
    )
    tool_calls = out[0].model_dump()["tool_calls"]
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "f"


def test_litellm_conversion_accepts_litellm_message_object():
    """A non-Mapping LiteLLM ``Message`` goes through the ``model_dump``
    branch (rather than being treated as a dict)."""
    from litellm.types.utils import Message as LiteLLMMessage

    out = litellm_messages_to_ag_ui_messages(
        [LiteLLMMessage(role="assistant", content="from-object")]
    )
    assert out[0].role == "assistant"
    assert out[0].content == "from-object"
    assert isinstance(out[0].id, str)


# --------------------------------------------------------------------------
# crewai_prepare_inputs (endpoint)
# --------------------------------------------------------------------------

def test_prepare_inputs_strips_leading_system_message():
    """A leading system message is dropped; the remaining messages survive."""
    out = ep.crewai_prepare_inputs(
        state={},
        messages=[
            SystemMessage(id="s", role="system", content="sys"),
            UserMessage(id="u", role="user", content="hello"),
        ],
        tools=[],
    )
    assert len(out["messages"]) == 1
    assert out["messages"][0]["role"] == "user"
    assert out["messages"][0]["content"] == "hello"


def test_prepare_inputs_keeps_non_leading_system_message():
    """Only a *leading* system message is stripped; a user-first list is
    left intact."""
    out = ep.crewai_prepare_inputs(
        state={},
        messages=[UserMessage(id="u", role="user", content="hello")],
        tools=[],
    )
    assert len(out["messages"]) == 1
    assert out["messages"][0]["role"] == "user"


def test_prepare_inputs_reshapes_tools_to_copilotkit_actions():
    """Each ``Tool`` is reshaped into a ``{type:function, function:{...}}``
    action carrying name/description/parameters."""
    out = ep.crewai_prepare_inputs(
        state={},
        messages=[],
        tools=[Tool(
            name="searchTool",
            description="search the web",
            parameters={"type": "object", "properties": {}},
        )],
    )
    actions = out["copilotkit"]["actions"]
    assert actions == [{
        "type": "function",
        "function": {
            "name": "searchTool",
            "description": "search the web",
            "parameters": {"type": "object", "properties": {}},
        },
    }]


def test_prepare_inputs_merges_incoming_state():
    """Existing state keys are preserved alongside the injected
    messages/copilotkit keys."""
    out = ep.crewai_prepare_inputs(
        state={"existing": 1, "keep": "me"},
        messages=[],
        tools=[],
    )
    assert out["existing"] == 1
    assert out["keep"] == "me"
    assert out["messages"] == []
    assert out["copilotkit"] == {"actions": []}


# --------------------------------------------------------------------------
# copilotkit_stream reassembly (sdk)
# --------------------------------------------------------------------------

async def test_copilotkit_stream_reassembles_text_and_tool_calls():
    """A streamed text delta + a single tool call spread across two chunks
    reassemble into one ModelResponse with concatenated content and one
    fully-accumulated tool call. Scalars (id/created/model/fingerprint)
    are carried from the chunks."""
    flow_context.set(None)

    async def _gen():
        yield _stream_chunk("msg-1", content="Hello ")
        yield _stream_chunk("msg-1", content="world")
        # First tool-call chunk carries id + name + first args fragment.
        yield _stream_chunk("msg-1", tool_calls=[
            _tool_call_delta(call_id="call-1", name="searchTool", arguments='{"q":')
        ])
        # Continuation chunk: id is None, only more argument text.
        yield _stream_chunk("msg-1", tool_calls=[
            _tool_call_delta(call_id=None, name=None, arguments='1}')
        ])
        yield _stream_chunk("msg-1", finish_reason="stop")

    resp = await _copilotkit_stream_custom_stream_wrapper(_gen())

    message = resp.choices[0].message
    assert message.content == "Hello world"
    assert resp.id == "msg-1"
    assert resp.model == "gpt-4o"
    assert resp.system_fingerprint == "fp_test"
    assert resp.created == 1700000000
    assert resp.choices[0].finish_reason == "stop"

    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    tc = message.tool_calls[0]
    assert tc.id == "call-1"
    assert tc.function.name == "searchTool"
    assert tc.function.arguments == '{"q":1}'
    assert tc.type == "function"


async def test_copilotkit_stream_emits_chunk_events_per_delta():
    """Reassembly emits a bridged TEXT_MESSAGE_CHUNK per text delta and a
    bridged TOOL_CALL_CHUNK per argument delta, with the deltas passed
    through verbatim."""
    from crewai.utilities.events import crewai_event_bus

    flow_context.set(None)
    text_chunks = []
    tool_chunks = []

    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(BridgedTextMessageChunkEvent)
        def _on_text(source, event):  # pylint: disable=unused-argument
            text_chunks.append((event.message_id, event.role, event.delta))

        @crewai_event_bus.on(BridgedToolCallChunkEvent)
        def _on_tool(source, event):  # pylint: disable=unused-argument
            tool_chunks.append((event.tool_call_id, event.tool_call_name, event.delta))

        async def _gen():
            yield _stream_chunk("msg-2", content="A")
            yield _stream_chunk("msg-2", content="B")
            yield _stream_chunk("msg-2", tool_calls=[
                _tool_call_delta(call_id="c-1", name="tool", arguments="{}")
            ])
            yield _stream_chunk("msg-2", finish_reason="stop")

        await _copilotkit_stream_custom_stream_wrapper(_gen())

    assert text_chunks == [("msg-2", "assistant", "A"), ("msg-2", "assistant", "B")]
    assert tool_chunks == [("c-1", "tool", "{}")]


async def test_copilotkit_stream_passthrough_model_response():
    """A ready ``ModelResponse`` is returned unchanged (non-streaming)."""
    from litellm.types.utils import ModelResponse

    mr = ModelResponse()
    assert (await copilotkit_stream(mr)) is mr


async def test_copilotkit_stream_rejects_unknown_type():
    """An unrecognised response type raises ``ValueError`` rather than
    silently no-op'ing."""
    with pytest.raises(ValueError):
        await copilotkit_stream(object())


# --------------------------------------------------------------------------
# copilotkit_predict_state / copilotkit_emit_state (sdk -> endpoint listener)
# --------------------------------------------------------------------------

async def test_copilotkit_predict_state_emits_custom_event():
    """``copilotkit_predict_state`` emits a CUSTOM ``PredictState`` event
    whose value reshapes the config into state_key/tool/tool_argument."""
    ep.FastAPICrewFlowEventListener()  # registers bus handlers
    flow = _FakeFlow()
    queue = await ep.create_queue(flow)
    flow_context.set(flow)
    try:
        result = await copilotkit_predict_state(
            {"steps": {"tool_name": "SearchTool", "tool_argument": "steps"}}
        )
        assert result is True
        items = _drain(queue)
    finally:
        await ep.delete_queue(flow)

    assert len(items) == 1
    event = items[0]
    assert event.type == EventType.CUSTOM
    assert event.name == "PredictState"
    assert event.value == [
        {"state_key": "steps", "tool": "SearchTool", "tool_argument": "steps"}
    ]


async def test_copilotkit_emit_state_emits_state_snapshot():
    """``copilotkit_emit_state`` emits a STATE_SNAPSHOT carrying the passed
    state verbatim."""
    ep.FastAPICrewFlowEventListener()
    flow = _FakeFlow()
    queue = await ep.create_queue(flow)
    flow_context.set(flow)
    try:
        result = await copilotkit_emit_state({"progress": 5})
        assert result is True
        items = _drain(queue)
    finally:
        await ep.delete_queue(flow)

    assert len(items) == 1
    event = items[0]
    assert event.type == EventType.STATE_SNAPSHOT
    assert event.snapshot == {"progress": 5}


# --------------------------------------------------------------------------
# Endpoint listener translation (Bridged* -> wire events)
# --------------------------------------------------------------------------

async def test_listener_translates_text_and_tool_chunks():
    """The endpoint listener maps bridged text/tool chunk events onto wire
    TEXT_MESSAGE_CHUNK / TOOL_CALL_CHUNK events with payloads preserved."""
    from crewai.utilities.events import crewai_event_bus

    ep.FastAPICrewFlowEventListener()
    flow = _FakeFlow()
    queue = await ep.create_queue(flow)
    try:
        crewai_event_bus.emit(flow, BridgedTextMessageChunkEvent(
            type=EventType.TEXT_MESSAGE_CHUNK,
            message_id="m1", role="assistant", delta="hi",
        ))
        crewai_event_bus.emit(flow, BridgedToolCallChunkEvent(
            type=EventType.TOOL_CALL_CHUNK,
            tool_call_id="tc1", tool_call_name="searchTool", delta='{"q":1}',
        ))
        items = _drain(queue)
    finally:
        await ep.delete_queue(flow)

    assert [i.type for i in items] == [
        EventType.TEXT_MESSAGE_CHUNK, EventType.TOOL_CALL_CHUNK,
    ]
    text_event, tool_event = items
    assert text_event.message_id == "m1"
    assert text_event.role == "assistant"
    assert text_event.delta == "hi"
    assert tool_event.tool_call_id == "tc1"
    assert tool_event.tool_call_name == "searchTool"
    assert tool_event.delta == '{"q":1}'


async def test_listener_emits_messages_and_state_snapshot_on_method_finish():
    """When a flow method finishes, the listener emits MESSAGES_SNAPSHOT
    (converted from state messages) + STATE_SNAPSHOT (the raw dict state) +
    STEP_FINISHED, in that order."""
    from crewai.utilities.events import crewai_event_bus, MethodExecutionFinishedEvent

    state = {
        "messages": [{"role": "assistant", "content": "done", "id": "m9"}],
        "outputs": "result-text",
    }
    ep.FastAPICrewFlowEventListener()
    flow = _FakeFlow(state=state)
    queue = await ep.create_queue(flow)
    try:
        crewai_event_bus.emit(flow, MethodExecutionFinishedEvent(
            type="method_execution_finished",
            method_name="chat",
            flow_name="ChatWithCrewFlow",
            result=None,
            state=state,
        ))
        items = _drain(queue)
    finally:
        await ep.delete_queue(flow)

    assert [i.type for i in items] == [
        EventType.MESSAGES_SNAPSHOT,
        EventType.STATE_SNAPSHOT,
        EventType.STEP_FINISHED,
    ]
    messages_event, state_event, step_event = items

    assert len(messages_event.messages) == 1
    assert messages_event.messages[0].role == "assistant"
    assert messages_event.messages[0].content == "done"
    assert messages_event.messages[0].id == "m9"

    assert state_event.snapshot == state
    assert step_event.step_name == "chat"


# --------------------------------------------------------------------------
# ChatWithCrewFlow crew-invocation branch (crews)
# --------------------------------------------------------------------------

async def test_chat_runs_crew_and_records_string_output():
    """When the first tool call names the crew, ``chat`` invokes the crew
    tool function, records its string result under ``state['outputs']`` and
    appends a matching ``tool`` message."""
    from ag_ui_crewai import crews as crews_mod

    async def _fake_acompletion(**_kwargs):
        return object()

    async def _fake_stream(_resp):
        class _Resp:
            choices = [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call-crew",
                        "function": {"name": "dummy", "arguments": '{"topic": "ai"}'},
                    }],
                }
            }]
        return _Resp()

    captured = {}

    def _fake_tool_factory(crew, messages):  # pylint: disable=unused-argument
        def _fn(**kwargs):
            captured["args"] = kwargs
            return "CREW OUTPUT"
        return _fn

    flow = crews_mod.ChatWithCrewFlow.__new__(crews_mod.ChatWithCrewFlow)
    flow.crew = type("C", (), {"chat_llm": "gpt-4o"})()
    flow.crew_name = "dummy"
    flow.crew_tool_schema = {
        "type": "function",
        "function": {"name": "dummy", "description": "", "parameters": {"type": "object"}},
    }
    flow.system_message = "sys"
    state = {"messages": [], "inputs": {"topic": "ai"}, "copilotkit": {"actions": []}}

    with _patch_instance_state(flow, state):
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                with patch.object(
                    crews_mod, "crew_chat_create_tool_function", _fake_tool_factory
                ):
                    await flow.chat()

    assert captured["args"] == {"topic": "ai"}
    assert state["outputs"] == "CREW OUTPUT"
    # Two messages: the assistant tool-call message, then the tool result.
    assert len(state["messages"]) == 2
    tool_message = state["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "CREW OUTPUT"
    assert tool_message["tool_call_id"] == "call-crew"


async def test_chat_crew_output_from_raw_attribute():
    """A crew result object exposing ``.raw`` (and no ``.json_dict``) has
    its ``raw`` recorded as the output."""
    from ag_ui_crewai import crews as crews_mod

    async def _fake_acompletion(**_kwargs):
        return object()

    async def _fake_stream(_resp):
        class _Resp:
            choices = [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call-crew",
                        "function": {"name": "dummy", "arguments": "{}"},
                    }],
                }
            }]
        return _Resp()

    class _CrewResult:
        raw = "raw-output"

    def _fake_tool_factory(crew, messages):  # pylint: disable=unused-argument
        return lambda **_kwargs: _CrewResult()

    flow = crews_mod.ChatWithCrewFlow.__new__(crews_mod.ChatWithCrewFlow)
    flow.crew = type("C", (), {"chat_llm": "gpt-4o"})()
    flow.crew_name = "dummy"
    flow.crew_tool_schema = {
        "type": "function",
        "function": {"name": "dummy", "description": "", "parameters": {"type": "object"}},
    }
    flow.system_message = "sys"
    state = {"messages": [], "inputs": {}, "copilotkit": {"actions": []}}

    with _patch_instance_state(flow, state):
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                with patch.object(
                    crews_mod, "crew_chat_create_tool_function", _fake_tool_factory
                ):
                    await flow.chat()

    assert state["outputs"] == "raw-output"
