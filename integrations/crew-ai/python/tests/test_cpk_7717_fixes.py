"""Regression coverage for the five ChatWithCrewFlow chat-path defects
fixed in CPK-7717 (ag-ui-crewai 0.2.1).

Boundary mocking mirrors ``test_happy_path.py``: ``acompletion`` /
``copilotkit_stream`` / the crew tool factory are patched inside the
``crews`` module (no live LLM), and the crewai event bus / endpoint
listener are driven directly. Nothing here reaches the network.

Defects covered:
  1. Chat LLM drops api_key/base_url (creds forwarded to acompletion).
  2. No text follow-up after a backend tool result (follow-up completion).
  3. Per-tool state mutations never surfaced (state snapshot emitted).
  4. _CREW_INPUTS_CACHE keyed on optional field (distinct unnamed crews).
  5. add_crewai_crew_fastapi_endpoint mis-typed + unexported.
"""

from contextlib import contextmanager
from unittest.mock import patch

from fastapi import FastAPI

from ag_ui.core import EventType
from ag_ui_crewai import crews as crews_mod
from ag_ui_crewai import endpoint as ep
from ag_ui_crewai.context import flow_context


# --------------------------------------------------------------------------
# Helpers (kept local so this file does not depend on cross-test imports)
# --------------------------------------------------------------------------

@contextmanager
def _patch_instance_state(flow, state):
    """Install ``state`` on a single flow instance via a throwaway subclass.

    ``Flow.state`` is a class-level descriptor, so we rebind ``__class__``
    to a per-instance subclass exposing ``state`` as a plain property.
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


def _drain(queue):
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _fake_llm(model="ollama/llama3", api_key="sk-local", base_url="http://localhost:11434"):
    """A stand-in for the crewai ``LLM`` object ``initialize_chat_llm``
    returns — carries the resolved model/credentials/endpoint."""
    return type(
        "FakeLLM",
        (),
        {"model": model, "api_key": api_key, "base_url": base_url},
    )()


def _new_crew_flow(*, chat_llm=None, crew_model="crew-model-string"):
    """Build a ``ChatWithCrewFlow`` via ``__new__`` with the minimal
    attributes the ``chat`` method reads, bypassing the LLM-calling
    constructor."""
    flow = crews_mod.ChatWithCrewFlow.__new__(crews_mod.ChatWithCrewFlow)
    flow.crew = type("C", (), {"chat_llm": crew_model})()
    if chat_llm is not None:
        flow.chat_llm = chat_llm
    flow.crew_name = "dummy"
    flow.crew_tool_schema = {
        "type": "function",
        "function": {"name": "dummy", "description": "", "parameters": {"type": "object"}},
    }
    flow.system_message = "sys"
    return flow


# --------------------------------------------------------------------------
# Defect 1: creds forwarded to acompletion
# --------------------------------------------------------------------------

def test_completion_llm_kwargs_forwards_model_key_and_base_url():
    """``_completion_llm_kwargs`` forwards the resolved LLM's model,
    api_key and base_url — NOT the raw crew model string."""
    flow = _new_crew_flow(chat_llm=_fake_llm())
    assert flow._completion_llm_kwargs() == {
        "model": "ollama/llama3",
        "api_key": "sk-local",
        "base_url": "http://localhost:11434",
    }


def test_completion_llm_kwargs_falls_back_when_llm_unresolved():
    """With no resolved ``chat_llm``, the helper falls back to the crew's
    model string and omits api_key/base_url (never overrides litellm with
    ``None``)."""
    flow = _new_crew_flow(chat_llm=None, crew_model="gpt-4o")
    assert flow._completion_llm_kwargs() == {"model": "gpt-4o"}


def test_completion_llm_kwargs_omits_absent_credentials():
    """A resolved LLM missing api_key/base_url forwards only the model."""
    flow = _new_crew_flow(chat_llm=_fake_llm(api_key=None, base_url=None))
    assert flow._completion_llm_kwargs() == {"model": "ollama/llama3"}


async def test_chat_forwards_credentials_to_acompletion():
    """Both completion call sites receive the resolved credentials +
    endpoint (defect 1) — the crew-run turn AND the defect-2 follow-up."""
    calls = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return object()

    stream_n = {"n": 0}

    async def _fake_stream(_resp):
        stream_n["n"] += 1
        if stream_n["n"] == 1:
            class _R:
                choices = [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call-crew",
                            "function": {"name": "dummy", "arguments": "{}"},
                        }],
                    }
                }]
            return _R()

        class _F:
            choices = [{"message": {"role": "assistant", "content": "done"}}]
        return _F()

    async def _noop_emit_state(_state):
        return True

    flow = _new_crew_flow(chat_llm=_fake_llm())
    state = {"messages": [], "inputs": {}, "copilotkit": {"actions": []}}

    with _patch_instance_state(flow, state):
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                with patch.object(crews_mod, "copilotkit_emit_state", _noop_emit_state):
                    with patch.object(
                        crews_mod, "crew_chat_create_tool_function",
                        lambda crew, messages: (lambda **_k: "OUT"),
                    ):
                        await flow.chat()

    assert len(calls) == 2
    for call in calls:
        assert call["model"] == "ollama/llama3"
        assert call["api_key"] == "sk-local"
        assert call["base_url"] == "http://localhost:11434"
    # The follow-up (defect 2) forces text, not another tool call.
    assert calls[1]["tool_choice"] == "none"


# --------------------------------------------------------------------------
# Defect 2: text follow-up after a backend tool result
# --------------------------------------------------------------------------

async def test_crew_tool_result_triggers_followup_completion():
    """After the crew tool result lands, a second completion runs and the
    assistant produces a text message (defect 2)."""
    stream_n = {"n": 0}

    async def _fake_acompletion(**_kwargs):
        return object()

    async def _fake_stream(_resp):
        stream_n["n"] += 1
        if stream_n["n"] == 1:
            class _R:
                choices = [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call-crew",
                            "function": {"name": "dummy", "arguments": "{}"},
                        }],
                    }
                }]
            return _R()

        class _F:
            choices = [{"message": {"role": "assistant", "content": "The crew is done."}}]
        return _F()

    async def _noop_emit_state(_state):
        return True

    flow = _new_crew_flow(chat_llm=_fake_llm())
    state = {"messages": [], "inputs": {}, "copilotkit": {"actions": []}}

    with _patch_instance_state(flow, state):
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                with patch.object(crews_mod, "copilotkit_emit_state", _noop_emit_state):
                    with patch.object(
                        crews_mod, "crew_chat_create_tool_function",
                        lambda crew, messages: (lambda **_k: "OUT"),
                    ):
                        await flow.chat()

    assert stream_n["n"] == 2
    # assistant tool-call, tool result, assistant follow-up text.
    assert [m["role"] for m in state["messages"]] == ["assistant", "tool", "assistant"]
    assert state["messages"][-1]["content"] == "The crew is done."


# --------------------------------------------------------------------------
# Defect 3: crew-run state mutation surfaced as a StateSnapshotEvent
# --------------------------------------------------------------------------

async def test_crew_run_emits_state_snapshot():
    """Running the crew emits a STATE_SNAPSHOT reflecting the applied
    output, routed to the bridge via the endpoint listener (defect 3)."""
    async def _fake_acompletion(**_kwargs):
        return object()

    stream_n = {"n": 0}

    async def _fake_stream(_resp):
        stream_n["n"] += 1
        if stream_n["n"] == 1:
            class _R:
                choices = [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call-crew",
                            "function": {"name": "dummy", "arguments": "{}"},
                        }],
                    }
                }]
            return _R()

        class _F:
            choices = [{"message": {"role": "assistant", "content": "done"}}]
        return _F()

    ep.FastAPICrewFlowEventListener()  # registers bus handlers
    flow = _new_crew_flow(chat_llm=_fake_llm())
    queue = await ep.create_queue(flow)
    state = {"messages": [], "inputs": {}, "copilotkit": {"actions": []}}

    token = flow_context.set(flow)
    try:
        with _patch_instance_state(flow, state):
            with patch.object(crews_mod, "acompletion", _fake_acompletion):
                with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                    with patch.object(
                        crews_mod, "crew_chat_create_tool_function",
                        lambda crew, messages: (lambda **_k: "OUT"),
                    ):
                        await flow.chat()
        items = _drain(queue)
    finally:
        flow_context.reset(token)
        await ep.delete_queue(flow)

    snapshots = [i for i in items if i.type == EventType.STATE_SNAPSHOT]
    assert len(snapshots) >= 1
    assert snapshots[-1].snapshot["outputs"] == "OUT"


# --------------------------------------------------------------------------
# Defect 4: cache keyed on stable identity, not optional crew.name
# --------------------------------------------------------------------------

def _make_crewbase(name):
    """A minimal ``@CrewBase``-shaped instance: a ``name`` and a ``crew()``
    factory returning an object with a non-None ``chat_llm``."""
    inner = type("InnerCrew", (), {"chat_llm": "gpt-4o"})()

    class _CrewBase:
        def __init__(self):
            self.name = name

        def crew(self):
            return inner

    return _CrewBase()


def test_unnamed_crews_get_distinct_cached_inputs():
    """Two distinct unnamed crews (``name is None``) must NOT collide on the
    cache (defect 4): each gets its own generated chat-input schema, and a
    repeat of the same crew reuses its cached entry."""
    crews_mod._CREW_INPUTS_CACHE.clear()

    gen_calls = []

    def _fake_gen(crew, name, llm):  # pylint: disable=unused-argument
        obj = object()
        gen_calls.append(obj)
        return obj

    with patch.object(crews_mod, "crew_chat_initialize_chat_llm", lambda c: _fake_llm()):
        with patch.object(crews_mod, "crew_chat_generate_crew_chat_inputs", _fake_gen):
            with patch.object(crews_mod, "crew_chat_generate_crew_tool_schema", lambda i: {}):
                with patch.object(crews_mod, "crew_chat_build_system_message", lambda i: ""):
                    c1 = _make_crewbase(name=None)
                    c2 = _make_crewbase(name=None)
                    f1 = crews_mod.ChatWithCrewFlow(crew=c1)
                    f2 = crews_mod.ChatWithCrewFlow(crew=c2)
                    # Reconstructing the SAME crew must hit the cache.
                    f1_again = crews_mod.ChatWithCrewFlow(crew=c1)

    # Two unnamed crews produce two DISTINCT schemas (pre-fix they collided
    # on the ``None`` key and shared one).
    assert f1.crew_chat_inputs is not f2.crew_chat_inputs
    # Same crew reused its cached schema — no second generation call.
    assert f1_again.crew_chat_inputs is f1.crew_chat_inputs
    assert len(gen_calls) == 2

    crews_mod._CREW_INPUTS_CACHE.clear()


# --------------------------------------------------------------------------
# Defect 5: endpoint type + exports
# --------------------------------------------------------------------------

def test_crew_path_symbols_exported_from_package_top_level():
    """The four previously-hidden Crew-path symbols are importable from the
    package top level and declared in ``__all__`` (defect 5)."""
    import ag_ui_crewai as pkg

    for name in (
        "add_crewai_crew_fastapi_endpoint",
        "copilotkit_exit",
        "crewai_prepare_inputs",
        "ChatWithCrewFlow",
    ):
        assert hasattr(pkg, name), f"{name} not importable from ag_ui_crewai"
        assert name in pkg.__all__, f"{name} missing from ag_ui_crewai.__all__"


def test_add_crew_endpoint_accepts_crewbase_instance():
    """``add_crewai_crew_fastapi_endpoint`` accepts a ``@CrewBase``-shaped
    instance and registers a POST route without constructing the flow
    (construction is deferred to first request, so no LLM call) — defect 5
    type fix."""
    crew = _make_crewbase(name="researcher")
    # The structural protocol recognises the CrewBase shape.
    assert isinstance(crew, ep.CrewBaseInstance)
    # A bare object lacking ``crew()`` does not.
    assert not isinstance(object(), ep.CrewBaseInstance)

    app = FastAPI()
    add = ep.add_crewai_crew_fastapi_endpoint
    add(app, crew, path="/crew")

    post_paths = {
        route.path
        for route in app.routes
        if "POST" in getattr(route, "methods", set())
    }
    assert "/crew" in post_paths
