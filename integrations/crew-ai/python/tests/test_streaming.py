"""Streaming and event emission: ``copilotkit_stream`` reassembly, the
per-delta chunk events, ``copilotkit_predict_state``/``copilotkit_emit_state``,
and the endpoint listener's translation of bridged events to wire events.
No network."""

import asyncio
import json as _json
from types import SimpleNamespace

import pytest

from crewai.flow.flow import Flow, start

from ag_ui.core import EventType
from ag_ui_crewai import endpoint as ep
from ag_ui_crewai import _frames as frames_mod
from ag_ui_crewai._capabilities import CAPABILITIES, flow_supports_stream_frames
from ag_ui_crewai.context import flow_context
from ag_ui_crewai.events import (
    BridgedCustomEvent,
    BridgedTextMessageChunkEvent,
    BridgedToolCallChunkEvent,
)
from litellm import CustomStreamWrapper

from ag_ui_crewai.sdk import (
    CopilotKitState,
    copilotkit_emit_state,
    copilotkit_predict_state,
    copilotkit_stream,
)


async def _settle_bus(emit_result=None):
    """Let off-thread crewai 1.x event-bus handlers land on the queue.

    crewai 1.x dispatches our sync listener callbacks on a
    ThreadPoolExecutor worker thread, and ``_enqueue`` hops the result back
    onto the loop via ``call_soon_threadsafe``. A test that emits then drains
    synchronously must wait for the handler to finish AND give the loop one
    tick so the scheduled ``put_nowait`` runs.

    When the test holds the ``emit`` result (a ``concurrent.futures.Future``,
    or ``None`` when there are no handlers) we await it directly. When the emit
    happens INSIDE an SDK call (predict_state / emit_state / copilotkit_stream)
    we can't reach the future, so we ``flush`` the bus — which blocks until
    in-flight handlers complete — off the loop, then tick.
    """
    if emit_result is not None:
        try:
            await asyncio.wrap_future(emit_result)
        except Exception:  # noqa: BLE001 - handler errors surface elsewhere
            pass
    else:
        from ag_ui_crewai._capabilities import crewai_event_bus
        flush = getattr(crewai_event_bus, "flush", None)
        if callable(flush):
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: flush(5.0)
                )
            except Exception:  # noqa: BLE001 - flush is best-effort
                pass
    # One extra tick for the call_soon_threadsafe-scheduled put_nowait.
    await asyncio.sleep(0)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _stream_chunk(chunk_id, *, content=None, tool_calls=None, finish_reason=None):
    """A LiteLLM-shaped streaming chunk (tool-call entries are attribute-style)."""
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


class _FakeStreamWrapper(CustomStreamWrapper):
    """A real ``CustomStreamWrapper`` subclass (so ``copilotkit_stream``'s
    ``isinstance`` dispatch picks the streaming branch) that iterates a
    supplied async generator. Base ``__init__`` bypassed on purpose."""

    def __init__(self, gen):  # pylint: disable=super-init-not-called
        self._gen = gen

    def __aiter__(self):
        return self._gen


class _FakeFlow:
    """Minimal flow stand-in the endpoint listener can route events to."""

    def __init__(self, state=None):
        self.state = state if state is not None else {}


def _drain(queue):
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


# --------------------------------------------------------------------------
# copilotkit_stream reassembly (through the public dispatch)
# --------------------------------------------------------------------------

async def test_copilotkit_stream_reassembles_text_and_tool_calls():
    """Text deltas + a tool call spread across chunks reassemble into one
    ModelResponse with concatenated content and one accumulated tool call."""
    flow_context.set(None)

    async def _gen():
        yield _stream_chunk("msg-1", content="Hello ")
        yield _stream_chunk("msg-1", content="world")
        yield _stream_chunk("msg-1", tool_calls=[
            _tool_call_delta(call_id="call-1", name="searchTool", arguments='{"q":')
        ])
        yield _stream_chunk("msg-1", tool_calls=[
            _tool_call_delta(call_id=None, name=None, arguments='1}')
        ])
        yield _stream_chunk("msg-1", finish_reason="stop")

    resp = await copilotkit_stream(_FakeStreamWrapper(_gen()))

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
    TOOL_CALL_CHUNK per argument delta, deltas passed through verbatim."""
    from ag_ui_crewai._capabilities import crewai_event_bus

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

        await copilotkit_stream(_FakeStreamWrapper(_gen()))
        # crewai 1.x runs these handlers off-thread; settle before asserting.
        await _settle_bus()

    assert text_chunks == [("msg-2", "assistant", "A"), ("msg-2", "assistant", "B")]
    assert tool_chunks == [("c-1", "tool", "{}")]


async def test_copilotkit_stream_passthrough_model_response():
    """A ready ``ModelResponse`` is returned unchanged (non-streaming)."""
    from litellm.types.utils import ModelResponse

    mr = ModelResponse()
    assert (await copilotkit_stream(mr)) is mr


async def test_copilotkit_stream_rejects_unknown_type():
    """An unrecognised response type raises ``ValueError``."""
    with pytest.raises(ValueError):
        await copilotkit_stream(object())


# --------------------------------------------------------------------------
# copilotkit_predict_state / copilotkit_emit_state
# --------------------------------------------------------------------------

async def test_copilotkit_predict_state_emits_custom_event():
    """``copilotkit_predict_state`` emits a CUSTOM ``PredictState`` event."""
    ep.FastAPICrewFlowEventListener()  # registers bus handlers
    flow = _FakeFlow()
    queue = await ep.create_queue(flow)
    flow_context.set(flow)
    try:
        result = await copilotkit_predict_state(
            {"steps": {"tool_name": "SearchTool", "tool_argument": "steps"}}
        )
        assert result is True
        await _settle_bus()
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


async def test_copilotkit_predict_state_tool_argument_is_optional():
    """``tool_argument`` is documented optional: omitting it must not KeyError,
    and the wire value carries ``tool_argument=None`` (whole-object streaming)."""
    ep.FastAPICrewFlowEventListener()  # registers bus handlers
    flow = _FakeFlow()
    queue = await ep.create_queue(flow)
    flow_context.set(flow)
    try:
        result = await copilotkit_predict_state({"steps": {"tool_name": "SearchTool"}})
        assert result is True
        await _settle_bus()
        items = _drain(queue)
    finally:
        await ep.delete_queue(flow)

    assert len(items) == 1
    event = items[0]
    assert event.type == EventType.CUSTOM
    assert event.name == "PredictState"
    assert event.value == [
        {"state_key": "steps", "tool": "SearchTool", "tool_argument": None}
    ]


async def test_copilotkit_emit_state_emits_state_snapshot():
    """``copilotkit_emit_state`` emits a STATE_SNAPSHOT carrying the state."""
    ep.FastAPICrewFlowEventListener()
    flow = _FakeFlow()
    queue = await ep.create_queue(flow)
    flow_context.set(flow)
    try:
        result = await copilotkit_emit_state({"progress": 5})
        assert result is True
        await _settle_bus()
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
    """The listener maps bridged text/tool chunks onto wire
    TEXT_MESSAGE_CHUNK / TOOL_CALL_CHUNK events with payloads preserved."""
    from ag_ui_crewai._capabilities import crewai_event_bus

    ep.FastAPICrewFlowEventListener()
    flow = _FakeFlow()
    queue = await ep.create_queue(flow)
    try:
        await _settle_bus(crewai_event_bus.emit(flow, BridgedTextMessageChunkEvent(
            type=EventType.TEXT_MESSAGE_CHUNK,
            message_id="m1", role="assistant", delta="hi",
        )))
        await _settle_bus(crewai_event_bus.emit(flow, BridgedToolCallChunkEvent(
            type=EventType.TOOL_CALL_CHUNK,
            tool_call_id="tc1", tool_call_name="searchTool", delta='{"q":1}',
        )))
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
    """On flow-method finish the listener emits MESSAGES_SNAPSHOT +
    STATE_SNAPSHOT + STEP_FINISHED, in that order."""
    from ag_ui_crewai._capabilities import (
        crewai_event_bus,
        MethodExecutionFinishedEvent,
    )

    state = {
        "messages": [{"role": "assistant", "content": "done", "id": "m9"}],
        "outputs": "result-text",
    }
    ep.FastAPICrewFlowEventListener()
    flow = _FakeFlow(state=state)
    queue = await ep.create_queue(flow)
    try:
        await _settle_bus(crewai_event_bus.emit(flow, MethodExecutionFinishedEvent(
            type="method_execution_finished",
            method_name="chat",
            flow_name="ChatWithCrewFlow",
            result=None,
            state=state,
        )))
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
# StreamFrame path: flow.astream() -> frame translator -> wire
# --------------------------------------------------------------------------

# Tests that drive a REAL crewai ``Flow.astream`` require the StreamFrame
# contract (crewai >= 1.6). On the 1.0-1.5 fallback the bridge uses the legacy
# bus-listener path (covered by the tests above), so these are skipped there.
requires_stream_frames = pytest.mark.skipif(
    not CAPABILITIES.stream_frame_available,
    reason="crewai>=1.6 StreamFrame contract required; 1.0-1.5 uses the "
    "legacy bus-listener fallback path",
)


def _decode_sse(encoded_items):
    """Decode a list of EventEncoder SSE strings into JSON payload dicts."""
    payloads = []
    for chunk in encoded_items:
        for line in chunk.splitlines():
            if line.startswith("data:"):
                payloads.append(_json.loads(line[len("data:"):].strip()))
    return payloads


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def _ev(type, event_id=None, **attrs):  # noqa: A002 - mirror event.type
    """A RAW crewai/bridge event stand-in the translator reads by attribute.

    The translator now consumes raw event objects, so a
    lifecycle event is any object exposing ``.type`` (+ ``.method_name`` etc.)
    and a bridge event exposes its typed payload attributes directly — no
    ``to_serializable`` ``frame.data`` in the loop."""
    return SimpleNamespace(
        type=type,
        event_id=event_id or f"ev-{id(attrs)}",
        **attrs,
    )


class _FakeStreamSession:
    """Minimal AsyncStreamSession stand-in. Publishes each supplied RAW event to
    the scoped sink (as crewai's ``event_bus._prepare_event`` does) and then
    yields a StreamFrame-shaped stand-in whose ``id`` matches, so the driver's
    source-gated raw-event lookup finds it. Records whether ``aclose`` was
    called. Lets us unit-test the driver's teardown / ceiling handling without a
    live crewai run."""

    def __init__(self, events, *, source, hang=False):
        self._events = events
        self._source = source
        self._hang = hang
        self.aclosed = False
        # Instrumentation: how many frames the driver actually
        # consumed, and whether the iterator was drained to natural exhaustion
        # (vs stopped early via break + aclose).
        self.frames_yielded = 0
        self.exhausted = False

    async def _agen(self):
        from crewai.events.stream_context import publish_stream_event

        for ev in self._events:
            # The sink (registered by the driver in this same context) parks the
            # raw event; the frame supplies ordering + the id to look it up.
            publish_stream_event(self._source, ev)
            self.frames_yielded += 1
            yield _Frame(ev.type, id=ev.event_id)
        if self._hang:
            # Never terminate on its own — the ceiling / aclose must stop us.
            await asyncio.Event().wait()
        self.exhausted = True

    def __aiter__(self):
        return self._agen()

    async def aclose(self):
        self.aclosed = True


class _Frame:
    """StreamFrame-shaped stand-in (the driver reads only ``type`` / ``id``)."""

    def __init__(self, type, id):  # noqa: A002 - mirror StreamFrame.type / .id
        self.type = type
        self.id = id


# -- capability probe -------------------------------------------------------

def test_stream_frame_probe_is_per_flow_and_version_consistent():
    """The per-flow probe agrees with the resolved capability: a real Flow is
    routed to the StreamFrame path iff crewai exposes StreamFrame; a
    kickoff-only stub (the cancellation-test shape) ALWAYS takes the legacy
    path so its coverage is unaffected on either crewai line."""
    class _Real(Flow):
        @start()
        async def go(self):
            return None

    assert flow_supports_stream_frames(_Real()) is CAPABILITIES.stream_frame_available

    class _KickoffOnly:
        async def kickoff_async(self, inputs=None):
            return None

    assert flow_supports_stream_frames(_KickoffOnly()) is False


# -- translator wire shape (default = chunks) -------------------------------

def test_translator_produces_current_chunk_wire_shape():
    """The default translator maps bridge/lifecycle events onto exactly the
    events the legacy listener produced — chunks, not triples."""
    state = {"messages": [{"role": "assistant", "content": "hi", "id": "m1"}]}
    tr = frames_mod.StreamFrameTranslator(
        thread_id="t-1", run_id="r-1", state_provider=lambda: state,
    )

    assert [e.type for e in tr.translate(_ev("flow_started"))] == [
        EventType.RUN_STARTED
    ]
    assert tr.run_started is True
    start_ev = tr.translate(_ev("method_execution_started", method_name="chat"))
    assert start_ev[0].type == EventType.STEP_STARTED
    assert start_ev[0].step_name == "chat"

    text = tr.translate(_ev(
        "TEXT_MESSAGE_CHUNK", message_id="m1", role="assistant", delta="hi",
    ))
    assert len(text) == 1
    assert text[0].type == EventType.TEXT_MESSAGE_CHUNK
    assert (text[0].message_id, text[0].role, text[0].delta) == ("m1", "assistant", "hi")

    tool = tr.translate(_ev(
        "TOOL_CALL_CHUNK", tool_call_id="tc1", tool_call_name="searchTool",
        delta='{"q":1}',
    ))
    assert tool[0].type == EventType.TOOL_CALL_CHUNK
    assert (tool[0].tool_call_id, tool[0].tool_call_name, tool[0].delta) == (
        "tc1", "searchTool", '{"q":1}'
    )

    custom = tr.translate(_ev("CUSTOM", name="PredictState", value=[1]))
    assert custom[0].type == EventType.CUSTOM
    assert (custom[0].name, custom[0].value) == ("PredictState", [1])

    snap = tr.translate(_ev("STATE_SNAPSHOT", snapshot={"p": 5}))
    assert snap[0].type == EventType.STATE_SNAPSHOT
    assert snap[0].snapshot == {"p": 5}

    finished = tr.translate(_ev("method_execution_finished", method_name="chat"))
    assert [e.type for e in finished] == [
        EventType.MESSAGES_SNAPSHOT, EventType.STATE_SNAPSHOT, EventType.STEP_FINISHED,
    ]
    assert finished[0].messages[0].id == "m1"
    assert finished[2].step_name == "chat"

    fin = tr.translate(_ev("flow_finished"))
    assert fin[0].type == EventType.RUN_FINISHED
    assert tr.run_finished is True
    # Idempotent: a second flow_finished never re-emits RUN_FINISHED.
    assert tr.translate(_ev("flow_finished")) == []

    # Native crewai channels / unknown events are dropped (behavior-preserving).
    assert tr.translate(_ev("llm_stream_chunk", chunk="x")) == []
    assert tr.translate(_ev("cc_env")) == []


def test_translator_emission_shape_is_swappable_and_defaults_to_chunks():
    """The emission shape is a single seam defaulting to chunks; the parity
    'triples' shape is a documented NotImplementedError placeholder."""
    tr = frames_mod.StreamFrameTranslator(
        thread_id="t", run_id="r", state_provider=dict,
    )
    assert tr.emission_shape == "chunks"

    with pytest.raises(ValueError):
        frames_mod.StreamFrameTranslator(
            thread_id="t", run_id="r", state_provider=dict, emission_shape="bogus",
        )

    triples = frames_mod.StreamFrameTranslator(
        thread_id="t", run_id="r", state_provider=dict, emission_shape="triples",
    )
    with pytest.raises(NotImplementedError):
        triples.translate(_ev("TEXT_MESSAGE_CHUNK", message_id="m", delta="x"))


# -- end-to-end through a REAL crewai Flow via astream ----------------------

class _BridgeEmittingFlow(Flow):
    """A real crewai Flow whose method emits the bridge's own events (exactly
    as ``sdk.copilotkit_stream`` does) so they round-trip through astream."""

    @start()
    async def chat(self):
        f = flow_context.get(None)
        from ag_ui_crewai._capabilities import crewai_event_bus
        crewai_event_bus.emit(f, BridgedTextMessageChunkEvent(
            type=EventType.TEXT_MESSAGE_CHUNK, message_id="m1", role="assistant", delta="Hello ",
        ))
        crewai_event_bus.emit(f, BridgedTextMessageChunkEvent(
            type=EventType.TEXT_MESSAGE_CHUNK, message_id="m1", role="assistant", delta="world",
        ))
        crewai_event_bus.emit(f, BridgedCustomEvent(
            type=EventType.CUSTOM, name="Exit", value="",
        ))
        return "done"


@requires_stream_frames
async def test_frame_path_end_to_end_matches_legacy_wire_shape():
    """Driving a real Flow through the StreamFrame path yields RUN_STARTED,
    STEP_STARTED, per-delta TEXT_MESSAGE_CHUNK, a CUSTOM, then
    MESSAGES/STATE snapshot + STEP_FINISHED + RUN_FINISHED — the same wire
    shape the legacy listener produced."""
    from ag_ui.encoder import EventEncoder

    flow = _BridgeEmittingFlow()
    input_data = _make_run_input()
    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=flow,
        encoder=EventEncoder(),
        input_data=input_data,
        inputs={"id": "t-1"},
        timeout=30.0,
    ))
    payloads = _decode_sse(encoded)
    types = [p["type"] for p in payloads]

    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"
    assert types.count("TEXT_MESSAGE_CHUNK") == 2
    assert "STEP_STARTED" in types
    assert "STEP_FINISHED" in types
    assert "CUSTOM" in types
    assert "MESSAGES_SNAPSHOT" in types
    # No START/CONTENT/END triples — behavior-preserving chunk shape.
    assert not any(t.endswith("_START") or t.endswith("_END") for t in types
                   if t not in ("RUN_STARTED",))
    # Correlation ids are stamped, not the listener's "?" placeholders.
    run_started = next(p for p in payloads if p["type"] == "RUN_STARTED")
    assert run_started["threadId"] == "t-1"
    assert run_started["runId"] == "r-1"
    text_deltas = [p["delta"] for p in payloads if p["type"] == "TEXT_MESSAGE_CHUNK"]
    assert text_deltas == ["Hello ", "world"]


def _make_run_input(thread_id="t-1", run_id="r-1"):
    from ag_ui.core import RunAgentInput
    return RunAgentInput(
        thread_id=thread_id, run_id=run_id, state={}, messages=[], tools=[],
        context=[], forwarded_props={},
    )


# -- ONE RUN_STARTED / ONE RUN_FINISHED per HTTP run --------------

class _InnerKickoffFlow(Flow):
    """Stands in for the ``crew.kickoff`` a ``ChatWithCrewFlow.chat`` runs
    mid-method. A real crew kickoff drives crewai's experimental agent
    executor THROUGH the flow runtime, so it emits its own ``flow_started`` /
    ``flow_finished`` frames — reproduced here with a real nested Flow so the
    test needs no LLM/network."""

    @start()
    async def go(self):
        return "inner-done"


class _TwoCompletionCrewFlow(Flow):
    """A real Flow that performs TWO internal operations in ONE run — exactly
    the crew-tool path shape (``crew.kickoff`` off the event loop, then a
    follow-up completion). The nested kickoff runs via
    ``asyncio.to_thread`` (as the bridge offloads ``crew.kickoff``), which
    copies the scoped stream-sink contextvar, so the inner flow's
    ``flow_started`` / ``flow_finished`` frames land on THIS run's sink."""

    @start()
    async def chat(self):
        # Completion #1 surrogate: the nested (crew) kickoff. Off the loop, as
        # ``crews.py`` runs ``crew_function`` via ``asyncio.to_thread``.
        await asyncio.to_thread(lambda: _InnerKickoffFlow().kickoff())
        # Completion #2: the follow-up completion that
        # makes the assistant speak about the crew result. ``copilotkit_stream``
        # emits this as a bridged TEXT_MESSAGE_CHUNK on the same sink.
        f = flow_context.get(None)
        from ag_ui_crewai._capabilities import crewai_event_bus
        crewai_event_bus.emit(f, BridgedTextMessageChunkEvent(
            type=EventType.TEXT_MESSAGE_CHUNK,
            message_id="m-followup", role="assistant", delta="Crew is done.",
        ))
        return "done"


@requires_stream_frames
async def test_frame_path_two_completions_emit_single_run_lifecycle():
    """A run whose flow method performs two internal completions —
    a nested (crew) kickoff plus a follow-up — must emit EXACTLY ONE
    RUN_STARTED (first) and ONE RUN_FINISHED (last), with the follow-up text
    streaming in between.

    Pre-fix, the nested kickoff's ``flow_started`` produced a SECOND
    RUN_STARTED (the client rejects it: "Cannot send 'RUN_STARTED' while a run
    is still active"), and its ``flow_finished`` tripped ``is_run_end`` so the
    driver broke BEFORE the follow-up text streamed."""
    from ag_ui.encoder import EventEncoder

    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=_TwoCompletionCrewFlow(),
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={"id": "t-1"},
        timeout=30.0,
    ))
    payloads = _decode_sse(encoded)
    types = [p["type"] for p in payloads]

    # Exactly one RUN_STARTED and one RUN_FINISHED, bracketing the run.
    assert types.count("RUN_STARTED") == 1, types
    assert types.count("RUN_FINISHED") == 1, types
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"

    # The follow-up text reaches the client, inside the run.
    assert "TEXT_MESSAGE_CHUNK" in types, types
    follow = next(p for p in payloads if p["type"] == "TEXT_MESSAGE_CHUNK")
    assert follow["delta"] == "Crew is done."
    assert types.index("TEXT_MESSAGE_CHUNK") < types.index("RUN_FINISHED")


# -- Review invariants: raw-payload fidelity, nested non-leak, terminal


class _ProgressiveStateFlow(Flow):
    """Emits an intermediate ``copilotkit_emit_state`` (as agentic_generative_ui
    does mid-method) carrying string, deeply-nested, and reserved-name values."""

    @start()
    async def chat(self):
        await copilotkit_emit_state({
            "steps": [{"description": "Digging hole", "status": "completed"}],
            # user-state keys that collide with crewai's _FRAME_DATA_EXCLUDE set
            "type": "user-type",
            "timestamp": "user-ts",
            # a value at depth >= 5, where to_serializable() falls back to repr()
            "deep": {"a": {"b": {"c": {"d": {"e": "deep-string"}}}}},
        })
        return "done"


@requires_stream_frames
async def test_frame_path_progressive_state_snapshot_is_verbatim():
    """The intermediate STATE_SNAPSHOT must equal the LIVE
    state ``copilotkit_emit_state`` was given — no ``repr()`` quoting of strings
    at depth >= 5, no dropping of user keys named ``type`` / ``timestamp``.

    Pre-fix the translator built the snapshot from ``frame.data`` (crewai's
    ``to_serializable(max_depth=5)`` output), so ``description`` arrived as
    ``\"'Digging hole'\"`` and the depth-5 string was stringified — exactly the
    corruption those progressive demos exist to surface. Verified against the
    crewai 1.15.7 wheel."""
    from ag_ui.encoder import EventEncoder

    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=_ProgressiveStateFlow(),
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={"id": "t-1"},
        timeout=30.0,
    ))
    payloads = _decode_sse(encoded)
    # The intermediate emit (the one carrying "steps"), NOT the method-finished
    # snapshot built from the flow's own state.
    snap = next(
        p["snapshot"] for p in payloads
        if p["type"] == "STATE_SNAPSHOT" and "steps" in (p.get("snapshot") or {})
    )
    # No repr quoting of shallow strings.
    assert snap["steps"][0]["description"] == "Digging hole", snap
    assert snap["steps"][0]["status"] == "completed", snap
    # Depth-5 string survives intact (pre-fix: a repr'd dict string).
    assert snap["deep"]["a"]["b"]["c"]["d"]["e"] == "deep-string", snap
    # User keys colliding with _FRAME_DATA_EXCLUDE are preserved verbatim.
    assert snap["type"] == "user-type", snap
    assert snap["timestamp"] == "user-ts", snap


class _NestedNoLeakFlow(Flow):
    """One outer method that performs a nested (crew-shaped) kickoff off the
    loop. The nested flow emits its OWN method_execution_* / flow_* frames onto
    this run's sink (contextvars copied by ``to_thread``)."""

    @start()
    async def chat(self):
        await asyncio.to_thread(lambda: _InnerKickoffFlow().kickoff())
        return "done"


@requires_stream_frames
async def test_frame_path_nested_flow_frames_do_not_leak():
    """A nested kickoff must NOT inject a second
    STEP_STARTED / MESSAGES_SNAPSHOT / STATE_SNAPSHOT / STEP_FINISHED built from
    the OUTER flow's state. The outer run has exactly ONE method, so each of
    those appears exactly once — matching the legacy (``source is flow_copy``)
    wire shape.

    Pre-fix the nested ``method_execution_*`` frames passed the depth gate
    (which only guarded flow_started/finished), so a mid-run authoritative
    MESSAGES_SNAPSHOT from stale outer state could wipe streamed text."""
    from ag_ui.encoder import EventEncoder

    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=_NestedNoLeakFlow(),
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={"id": "t-1"},
        timeout=30.0,
    ))
    types = [p["type"] for p in _decode_sse(encoded)]

    assert types.count("RUN_STARTED") == 1, types
    assert types.count("RUN_FINISHED") == 1, types
    # Exactly one outer method => one of each step/snapshot event; no nested leak.
    assert types.count("STEP_STARTED") == 1, types
    assert types.count("STEP_FINISHED") == 1, types
    assert types.count("MESSAGES_SNAPSHOT") == 1, types
    assert types.count("STATE_SNAPSHOT") == 1, types
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"


class _NestedRaisingInnerFlow(Flow):
    @start()
    async def boom(self):
        raise RuntimeError("nested boom")


class _OuterCatchesNestedErrorFlow(Flow):
    """The outer method runs a nested kickoff that RAISES, catches it, and
    continues to completion — the exact shape where crewai emits NO nested
    ``flow_finished`` (it fires only on the nested success path)."""

    @start()
    async def chat(self):
        try:
            await asyncio.to_thread(lambda: _NestedRaisingInnerFlow().kickoff())
        except Exception:  # noqa: BLE001 - outer intentionally swallows + continues
            pass
        return "outer-survived"


@requires_stream_frames
async def test_frame_path_nested_error_still_terminates_run():
    """A nested flow that raises (so its ``flow_finished``
    is never emitted) while the outer method catches and continues must STILL
    terminate the run — exactly one RUN_STARTED and a final RUN_FINISHED (or
    RUN_ERROR), never a run that ends with neither.

    Pre-fix the depth counter stuck at a non-zero value (the nested
    ``flow_started`` bumped it, the missing nested ``flow_finished`` never
    unwound it), so the outer ``flow_finished`` saw depth > 0 and emitted no
    RUN_FINISHED — the client saw a run that never ended. Verified against the
    crewai 1.15.7 wheel."""
    from ag_ui.encoder import EventEncoder

    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=_OuterCatchesNestedErrorFlow(),
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={"id": "t-1"},
        timeout=30.0,
    ))
    types = [p["type"] for p in _decode_sse(encoded)]

    assert types.count("RUN_STARTED") == 1, types
    assert types[0] == "RUN_STARTED"
    # The run ALWAYS terminates — never ends with neither terminator.
    assert types[-1] in ("RUN_FINISHED", "RUN_ERROR"), types
    assert "RUN_FINISHED" in types or "RUN_ERROR" in types, types


# -- per-request flow COPY seeds state before @start runs ------

class _StateReadingFlow(Flow[CopilotKitState]):
    """A real crewai Flow shaped like the served example flows
    (``Flow[CopilotKitState]`` with attribute state access,
    ``self.state.messages`` / ``self.state.copilotkit.actions`` — exactly like
    ``examples/agentic_chat.py``). The class-level ``_seen`` sink records what
    the running @start observed."""

    _seen: dict = {}

    @start()
    async def chat(self):
        _StateReadingFlow._seen = {
            "self_id": id(self),
            "messages": [m for m in self.state.messages],
            "actions": [a for a in self.state.copilotkit.actions],
        }


@requires_stream_frames
async def test_copied_example_flow_astream_seeds_state_before_start_runs():
    """Flow-demo path: a per-request COPY of an example-shaped
    ``Flow[CopilotKitState]``, driven through the REAL
    ``crewai_prepare_inputs`` -> ``flow.astream(inputs=...)`` seam
    ``add_crewai_flow_fastapi_endpoint`` uses on crewai 1.6+, must seed
    ``messages`` / ``copilotkit`` into the COPY's state BEFORE ``@start`` runs.

    Same root cause as the crew path: pre-fix, ``_copy_flow``'s pin-and-share
    fallback shared the original's ``_methods`` (bound to the ORIGINAL), so
    ``astream`` seeded the COPY's state while ``chat`` executed against the
    un-seeded ORIGINAL -> ``AttributeError`` / empty reads. With the
    ``_copy_flow`` rebind the running method sees the seeded copy."""
    from ag_ui.core import Tool, UserMessage

    _StateReadingFlow._seen = {}
    flow = _StateReadingFlow()
    flow_copy = ep._copy_flow(flow)

    inputs = ep.crewai_prepare_inputs(
        state={},
        messages=[UserMessage(id="u1", role="user", content="hi flow")],
        tools=[Tool(name="do_thing", description="", parameters={"type": "object"})],
    )
    inputs["id"] = "thread-flow"

    session = flow_copy.astream(inputs=inputs)
    async for _frame in session:
        pass

    seen = _StateReadingFlow._seen
    assert [m["content"] for m in seen["messages"]] == ["hi flow"]
    assert [a["function"]["name"] for a in seen["actions"]] == ["do_thing"]
    # Executed against the COPY, and per-request isolation is preserved.
    assert seen["self_id"] == id(flow_copy)
    assert flow_copy._methods["chat"].__self__ is flow_copy
    assert flow._methods["chat"].__self__ is flow


# -- RUN_ERROR taxonomy + env knobs on the StreamFrame path -----------------

class _RaisingFlow(Flow):
    class _BoomError(Exception):
        pass

    @start()
    async def go(self):
        raise _RaisingFlow._BoomError("kaboom")


@requires_stream_frames
async def test_frame_path_flow_error_taxonomy_preserved():
    """A flow exception surfaces as AGUI_CREWAI_FLOW_ERROR_<Class> with a
    sanitized class name and camelCase correlation extras."""
    from ag_ui.encoder import EventEncoder

    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=_RaisingFlow(),
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={},
        timeout=30.0,
    ))
    payloads = _decode_sse(encoded)
    err = next(p for p in payloads if p["type"] == "RUN_ERROR")
    assert err["code"] == "AGUI_CREWAI_FLOW_ERROR_BOOMERROR"
    assert err["threadId"] == "t-1"
    assert err["runId"] == "r-1"
    # Coarse client message; no internal repr leak.
    assert "kaboom" not in err["message"]


class _HangingFlow(Flow):
    started = False

    @start()
    async def go(self):
        type(self).started = True
        await asyncio.sleep(60)
        return None


@requires_stream_frames
async def test_frame_path_ceiling_emits_flow_timeout_and_tears_down():
    """The wall-clock ceiling fires on the StreamFrame path, emits
    AGUI_CREWAI_FLOW_TIMEOUT, and aclose() tears the hung run down promptly."""
    from ag_ui.encoder import EventEncoder

    flow = _HangingFlow()
    start_t = asyncio.get_event_loop().time()
    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=flow,
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={},
        timeout=0.3,
    ))
    elapsed = asyncio.get_event_loop().time() - start_t
    payloads = _decode_sse(encoded)
    err = next(p for p in payloads if p["type"] == "RUN_ERROR")
    assert err["code"] == "AGUI_CREWAI_FLOW_TIMEOUT"
    # Ceiling ~0.3s; teardown must not hang for the flow's 60s sleep.
    assert elapsed < 10.0


async def test_frame_path_aclose_called_on_early_generator_close():
    """Closing the driver generator early (client disconnect) invokes
    aclose() on the session so the background kickoff task is torn down."""
    from ag_ui.encoder import EventEncoder

    class _AstreamFlow:
        state = {}

        def astream(self, inputs=None):
            return session

    flow_copy = _AstreamFlow()
    session = _FakeStreamSession(
        [
            _ev("flow_started", event_id="fs"),
            _ev("TEXT_MESSAGE_CHUNK", event_id="tx",
                message_id="m", role="assistant", delta="x"),
        ],
        source=flow_copy,
        hang=True,
    )

    gen = ep._run_flow_frame_stream(
        flow_copy=flow_copy,
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={},
        timeout=30.0,
    )
    # Consume the first couple of events, then close early.
    first = await gen.__anext__()
    assert "RUN_STARTED" in first
    await gen.aclose()
    assert session.aclosed is True


# -- raising astream is mapped to RUN_ERROR + no contextvar leak --

async def test_frame_path_raising_astream_emits_run_error_and_resets_context():
    """If ``astream`` (or ``__aiter__``) raises, the driver must
    (a) map it through the RUN_ERROR taxonomy — not let it escape the generator
    with no terminal event — and (b) never leak the ``flow_context`` token into
    the caller's context. Pre-fix, ``astream()``/``__aiter__()`` sat before the
    ``try``, so a raise skipped both the except-handlers and the finally reset."""
    from ag_ui.encoder import EventEncoder

    flow_context.set(None)

    class _AstreamBoom(Exception):
        pass

    class _RaisingAstreamFlow:
        state = {}

        def astream(self, inputs=None):
            raise _AstreamBoom("astream failed before any frame")

    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=_RaisingAstreamFlow(),
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={},
        timeout=30.0,
    ))
    payloads = _decode_sse(encoded)
    # (a) A single, taxonomy-coded RUN_ERROR — not a silent escape.
    assert [p["type"] for p in payloads] == ["RUN_ERROR"]
    assert payloads[0]["code"] == "AGUI_CREWAI_FLOW_ERROR_ASTREAMBOOM"
    assert payloads[0]["threadId"] == "t-1"
    assert payloads[0]["runId"] == "r-1"
    # (b) The contextvar set at driver entry was reset in the finally.
    assert flow_context.get(None) is None


# -- drain the terminal tail; don't cancel kickoff mid-finalize ---

async def test_frame_path_drains_tail_after_run_finished():
    """After RUN_FINISHED the driver drains the frame stream to
    natural exhaustion (so crewai's kickoff task finishes finalization) instead
    of breaking immediately and letting aclose() cancel it. A frame arriving
    AFTER flow_finished is consumed (drained) but produces no wire event."""
    from ag_ui.encoder import EventEncoder

    class _AstreamFlow:
        state = {}

        def astream(self, inputs=None):
            return session

    flow_copy = _AstreamFlow()
    session = _FakeStreamSession(
        [
            _ev("flow_started", event_id="fs"),
            _ev("flow_finished", event_id="ff"),
            # A trailing frame after flow_finished — the tail crewai keeps
            # producing while the kickoff task finalizes.
            _ev(EventType.CUSTOM, event_id="tail", name="late", value="x"),
        ],
        source=flow_copy,
    )

    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=flow_copy,
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={},
        timeout=30.0,
    ))
    types = [p["type"] for p in _decode_sse(encoded)]
    # RUN_FINISHED is terminal; the trailing CUSTOM is drained, never emitted.
    assert types == ["RUN_STARTED", "RUN_FINISHED"]
    # All three frames were consumed and the iterator hit StopAsyncIteration —
    # i.e. the driver drained rather than stopping at flow_finished.
    assert session.frames_yielded == 3
    assert session.exhausted is True


@requires_stream_frames
async def test_frame_path_does_not_cancel_kickoff_after_finish():
    """Real Flow: on the happy path the kickoff task must finish
    finalization — result recorded, not cancelled. Pre-fix the driver broke on
    RUN_FINISHED and the finally's aclose() cancelled the still-finalizing task
    on EVERY run (session ended is_cancelled=True with no result); verified
    against the crewai 1.15.7 wheel. Draining the tail to exhaustion fixes it."""
    from ag_ui.encoder import EventEncoder

    class _ResultFlow(Flow):
        @start()
        async def go(self):
            return "RESULT"

    flow = _ResultFlow()
    captured = {}
    real_astream = flow.astream

    def _capture(*args, **kwargs):
        stream_session = real_astream(*args, **kwargs)
        captured["session"] = stream_session
        return stream_session

    flow.astream = _capture

    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=flow,
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={},
        timeout=30.0,
    ))
    assert [p["type"] for p in _decode_sse(encoded)][-1] == "RUN_FINISHED"
    session = captured["session"]
    # The kickoff task completed normally rather than being cancelled by aclose.
    assert session.is_cancelled is False
    assert session.result == "RESULT"


# -- MCP events surface through the SHIPPED frame-path sink ----------

class _MCPEmittingFlow(Flow):
    """Emits crewai MCP events (connection lifecycle + a tool execution) with a
    NON-flow source, exactly as crewai core does. The frame-path ``_sink`` must
    therefore park them by TYPE (``is_mcp_event``), not by ``source is flow``."""

    @start()
    def go(self):
        from ag_ui_crewai._capabilities import crewai_event_bus
        from crewai.events import (
            MCPConnectionStartedEvent,
            MCPToolExecutionCompletedEvent,
        )

        agent = SimpleNamespace()  # non-flow source, like a crew/agent
        crewai_event_bus.emit(
            agent,
            MCPConnectionStartedEvent(server_name="files", transport_type="stdio"),
        )
        crewai_event_bus.emit(
            agent,
            MCPToolExecutionCompletedEvent(
                server_name="files",
                tool_name="read_file",
                tool_args={"path": "/x"},
                result="hello",
            ),
        )
        return "done"


@requires_stream_frames
async def test_frame_path_surfaces_mcp_tool_calls():
    """Agent-sourced MCP events surface through the real
    ``_run_flow_frame_stream`` sink as TOOL_CALL_* (tool executions) and CUSTOM
    (connection lifecycle), inside a single RUN_STARTED/RUN_FINISHED envelope."""
    from ag_ui.encoder import EventEncoder

    pytest.importorskip("crewai.mcp")

    flow = _MCPEmittingFlow()
    encoded = await _collect(ep._run_flow_frame_stream(
        flow_copy=flow,
        encoder=EventEncoder(),
        input_data=_make_run_input(),
        inputs={"id": "t-1"},
        timeout=30.0,
    ))
    payloads = _decode_sse(encoded)
    types = [p["type"] for p in payloads]

    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"
    assert types.count("RUN_STARTED") == 1
    assert types.count("RUN_FINISHED") == 1
    for expected in (
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
    ):
        assert expected in types, (expected, types)
    customs = [p for p in payloads if p["type"] == "CUSTOM"]
    assert any(c.get("name") == "mcp_connection_started" for c in customs)
