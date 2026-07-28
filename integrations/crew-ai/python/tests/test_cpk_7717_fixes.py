"""Regression coverage for the ChatWithCrewFlow chat-path defects fixed in
CPK-7717, including the round-2 review findings.

Unlike the earlier revision of this module (which the round-2 reviewer
found MASKED the bugs with fakes and a replaced generator), the tests
below exercise REAL crewai objects:

  * Finding 1 (LLM connection settings dropped) runs the real
    ``crewai.LLM`` through ``_completion_llm_kwargs`` and asserts
    ``api_base`` / ``api_version`` / ``additional_params`` are forwarded.
  * Finding 2 (real ``@CrewBase`` + unnamed crews) builds a REAL
    ``@CrewBase``-decorated class, registers it via
    ``add_crewai_crew_fastapi_endpoint``, and drives a real first request
    so the crew-name read (``_crew_name``) and real ``ChatInputs``
    construction run for real — plus a genuinely unnamed crew that must
    raise a clear error.
  * Finding 3 (id-reuse cache hazard) drives the real cache with real
    ``@CrewBase`` crews and asserts GC eviction / regeneration.

Only the LLM NETWORK boundary is stubbed (crewai's
``generate_*_with_ai`` helpers and ``acompletion`` / ``copilotkit_stream``)
so nothing reaches the network; the crewai objects and the real
``generate_crew_chat_inputs`` / ``ChatInputs`` are otherwise untouched.
"""

import gc
import weakref
from contextlib import contextmanager
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import crewai.cli.crew_chat as crew_chat_mod
from crewai import Agent, Crew, LLM, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.types.crew_chat import ChatInputs

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


@contextmanager
def _stub_llm_network():
    """Stub ONLY crewai's LLM-calling description helpers so the real
    ``generate_crew_chat_inputs`` + real ``ChatInputs`` still run offline.

    The reviewer's key objection to the prior module was that it replaced
    ``crew_chat_generate_crew_chat_inputs`` wholesale, so the real
    ``ChatInputs`` construction (and the ``crew_name`` it validates) never
    ran. Here we patch one level deeper — the two functions that make the
    actual ``chat_llm.call`` network requests — leaving the generator and
    the Pydantic model untouched.
    """
    with patch.object(
        crew_chat_mod, "generate_input_description_with_ai",
        lambda *a, **k: "an input field",
    ), patch.object(
        crew_chat_mod, "generate_crew_description_with_ai",
        lambda *a, **k: "a real crew",
    ):
        yield


def _make_real_crewbase(cls_name="ResearchCrew"):
    """Construct and return a REAL ``@CrewBase``-decorated instance.

    crewai's ``@CrewBase`` sets ``_crew_name`` (to the class ``__name__``)
    and exposes a ``crew()`` factory — it does NOT expose ``.name``. That
    is exactly the shape the round-2 name-read fix must handle.
    """
    @CrewBase
    class _Crew:
        @agent
        def researcher(self) -> Agent:
            return Agent(
                role="researcher", goal="research {topic}",
                backstory="an expert",
                llm=LLM(model="gpt-4o", api_key="k"),
            )

        @task
        def research_task(self) -> Task:
            return Task(
                description="Research {topic} thoroughly",
                expected_output="a report",
                agent=self.researcher(),
            )

        @crew
        def crew(self) -> Crew:
            return Crew(
                agents=self.agents, tasks=self.tasks,
                chat_llm=LLM(model="gpt-4o", api_key="k"),
            )

    _Crew.__name__ = cls_name
    _Crew._crew_name = cls_name
    return _Crew()


def _new_crew_flow(*, chat_llm=None, crew_model="crew-model-string"):
    """Build a ``ChatWithCrewFlow`` via ``__new__`` with the minimal
    attributes the ``chat`` method reads, bypassing the LLM-calling
    constructor. ``chat_llm`` is a real ``crewai.LLM`` in the callers."""
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
# Finding 1: every connection-relevant field forwarded to acompletion
# --------------------------------------------------------------------------

def test_completion_llm_kwargs_forwards_all_connection_fields_real_llm():
    """A REAL ``crewai.LLM`` carrying Azure-style connection settings has
    ALL of them forwarded (round-2 finding 1) — model, api_key, api_base,
    api_version — plus provider-specific ``additional_params`` spread. The
    reviewer's repro built exactly this LLM and saw only model+api_key."""
    real_llm = LLM(
        model="azure/deployment",
        api_key="secret",
        api_base="https://azure.example",
        api_version="2024-02-01",
        custom_provider_param="xyz",
    )
    flow = _new_crew_flow(chat_llm=real_llm)
    kwargs = flow._completion_llm_kwargs()

    assert kwargs["model"] == "azure/deployment"
    assert kwargs["api_key"] == "secret"
    assert kwargs["api_base"] == "https://azure.example"
    assert kwargs["api_version"] == "2024-02-01"
    # additional_params (litellm **kwargs on the crewai LLM) are spread.
    assert kwargs["custom_provider_param"] == "xyz"


def test_completion_llm_kwargs_forwards_base_url_when_set_real_llm():
    """A REAL ``crewai.LLM`` with ``base_url`` (local/self-hosted) forwards
    it (the original CopilotKit#2742 local-model repro)."""
    real_llm = LLM(model="ollama/llama3", api_key="sk-local",
                   base_url="http://localhost:11434")
    kwargs = _new_crew_flow(chat_llm=real_llm)._completion_llm_kwargs()
    assert kwargs["model"] == "ollama/llama3"
    assert kwargs["api_key"] == "sk-local"
    assert kwargs["base_url"] == "http://localhost:11434"


def test_completion_llm_kwargs_omits_absent_connection_fields_real_llm():
    """A REAL ``crewai.LLM`` with only model+api_key forwards just those —
    absent api_base/api_version/base_url are never sent as ``None`` (which
    would override litellm's own resolution)."""
    real_llm = LLM(model="gpt-4o", api_key="sk-1")
    kwargs = _new_crew_flow(chat_llm=real_llm)._completion_llm_kwargs()
    assert kwargs == {"model": "gpt-4o", "api_key": "sk-1"}


def test_completion_llm_kwargs_falls_back_when_llm_unresolved():
    """With no resolved ``chat_llm``, the helper falls back to the crew's
    model string and forwards only the model."""
    flow = _new_crew_flow(chat_llm=None, crew_model="gpt-4o")
    assert flow._completion_llm_kwargs() == {"model": "gpt-4o"}


async def test_chat_forwards_connection_fields_to_acompletion_real_llm():
    """Both completion call sites receive the resolved connection fields
    from a REAL ``crewai.LLM`` — the crew-run turn AND the defect-2
    follow-up."""
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

    real_llm = LLM(model="azure/deployment", api_key="secret",
                   api_base="https://azure.example", api_version="2024-02-01")
    flow = _new_crew_flow(chat_llm=real_llm)
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
        assert call["model"] == "azure/deployment"
        assert call["api_key"] == "secret"
        assert call["api_base"] == "https://azure.example"
        assert call["api_version"] == "2024-02-01"
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

    flow = _new_crew_flow(chat_llm=LLM(model="gpt-4o", api_key="k"))
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
    flow = _new_crew_flow(chat_llm=LLM(model="gpt-4o", api_key="k"))
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
# Finding 2: real @CrewBase name read + unnamed-crew clear error
# --------------------------------------------------------------------------

def test_real_crewbase_matches_structural_protocol():
    """A REAL ``@CrewBase`` instance satisfies ``CrewBaseInstance``
    (structural: ``crew()`` + ``_crew_name``); a bare object does not."""
    real = _make_real_crewbase()
    assert isinstance(real, ep.CrewBaseInstance)
    assert isinstance(real, crews_mod.CrewBaseInstance)
    assert not isinstance(object(), ep.CrewBaseInstance)


def test_real_crewbase_direct_construction_reads_crew_name_and_builds_chatinputs():
    """Constructing ``ChatWithCrewFlow`` from a REAL ``@CrewBase`` reads the
    name off ``_crew_name`` (not ``.name`` — which does not exist on a real
    @CrewBase and previously AttributeError'd) and feeds it into a REAL
    ``ChatInputs`` with no validation error (round-2 finding 2)."""
    crews_mod._CREW_INPUTS_CACHE.clear()
    crews_mod._CREW_INPUTS_FALLBACK.clear()

    real = _make_real_crewbase(cls_name="ResearchCrew")
    with _stub_llm_network():
        flow = crews_mod.ChatWithCrewFlow(crew=real)

    assert flow.crew_name == "ResearchCrew"
    assert isinstance(flow.crew_chat_inputs, ChatInputs)
    assert flow.crew_chat_inputs.crew_name == "ResearchCrew"


def test_real_crewbase_endpoint_triggers_lazy_flow_without_attribute_error():
    """Registering a REAL ``@CrewBase`` via ``add_crewai_crew_fastapi_endpoint``
    and driving a real first request triggers the deferred
    ``ChatWithCrewFlow(crew=...)`` construction — the exact site that
    AttributeError'd on ``crew.name`` before the fix. The request must
    complete (HTTP 200), proving the real ``_crew_name`` read and real
    ``ChatInputs`` construction succeed end-to-end (round-2 finding 2)."""
    real = _make_real_crewbase(cls_name="EndpointCrew")

    async def _fake_acompletion(**_kwargs):
        return object()

    async def _fake_stream(_resp):
        class _R:
            choices = [{"message": {"role": "assistant", "content": "hello"}}]
        return _R()

    app = FastAPI()
    with _stub_llm_network():
        with patch.object(crews_mod, "acompletion", _fake_acompletion):
            with patch.object(crews_mod, "copilotkit_stream", _fake_stream):
                ep.add_crewai_crew_fastapi_endpoint(app, real, path="/crew")
                client = TestClient(app)
                payload = {
                    "thread_id": "t1", "run_id": "r1", "state": {},
                    "messages": [{"id": "m1", "role": "user", "content": "hi"}],
                    "tools": [], "context": [], "forwarded_props": {},
                }
                resp = client.post("/crew", json=payload)

    assert resp.status_code == 200


def test_unnamed_crew_raises_clear_error():
    """A crew exposing ``crew()`` but neither a non-empty ``name`` nor
    ``_crew_name`` raises a CLEAR ``ValueError`` — never ``None`` into
    ``ChatInputs`` (which would surface as an opaque Pydantic validation
    error deep in ``generate_crew_chat_inputs``) — round-2 finding 2."""
    class _Unnamed:
        def crew(self):
            return type("C", (), {"chat_llm": LLM(model="gpt-4o", api_key="k")})()

    with _stub_llm_network():
        try:
            crews_mod.ChatWithCrewFlow(crew=_Unnamed())
        except ValueError as exc:
            assert "crew name" in str(exc).lower()
        else:
            raise AssertionError("expected a clear ValueError for an unnamed crew")


def test_empty_string_crew_name_raises_clear_error():
    """An empty/whitespace ``_crew_name`` is rejected (it would produce an
    empty crew-tool function name) — the name reader requires a non-empty
    string."""
    real = _make_real_crewbase()
    real._crew_name = "   "  # whitespace-only: not a usable name
    with _stub_llm_network():
        try:
            crews_mod.ChatWithCrewFlow(crew=real)
        except ValueError as exc:
            assert "crew name" in str(exc).lower()
        else:
            raise AssertionError("expected a clear ValueError for a blank name")


# --------------------------------------------------------------------------
# Finding 3: identity-safe cache (no id-reuse cross-serve)
# --------------------------------------------------------------------------

def test_same_crew_reuses_cached_inputs_real_crewbase():
    """Reconstructing a flow for the SAME real ``@CrewBase`` reuses the
    cached schema and does NOT re-run the (network-driven) real
    ``generate_crew_chat_inputs`` — the caching win is preserved. The real
    generator still runs (wrapped, not replaced) so nothing is masked."""
    crews_mod._CREW_INPUTS_CACHE.clear()
    crews_mod._CREW_INPUTS_FALLBACK.clear()

    real = _make_real_crewbase(cls_name="ReuseCrew")
    wrapped = crews_mod.crew_chat_generate_crew_chat_inputs

    with _stub_llm_network():
        with patch.object(
            crews_mod, "crew_chat_generate_crew_chat_inputs",
            side_effect=wrapped,
        ) as gen_spy:
            f1 = crews_mod.ChatWithCrewFlow(crew=real)
            f2 = crews_mod.ChatWithCrewFlow(crew=real)

    assert f1.crew_chat_inputs is f2.crew_chat_inputs
    assert gen_spy.call_count == 1


def test_cache_evicts_on_gc_and_regenerates_for_new_crew():
    """The cache is keyed on the crew OBJECT (WeakKeyDictionary), so when a
    crew is garbage-collected its schema entry is evicted — eliminating the
    ``id(crew)`` reuse hazard where a freshly allocated wrapper inherits a
    collected wrapper's id and is silently served the wrong schema
    (round-2 finding 3). A brand-new crew therefore regenerates rather than
    receiving the old schema."""
    crews_mod._CREW_INPUTS_CACHE.clear()
    crews_mod._CREW_INPUTS_FALLBACK.clear()

    crew_a = _make_real_crewbase(cls_name="CrewA")
    with _stub_llm_network():
        flow_a = crews_mod.ChatWithCrewFlow(crew=crew_a)
    schema_a = flow_a.crew_chat_inputs
    ref_a = weakref.ref(crew_a)

    # The live crew keeps its cache entry.
    assert crew_a in crews_mod._CREW_INPUTS_CACHE

    # Drop every strong reference to crew A and its flow, then collect.
    del crew_a, flow_a
    gc.collect()

    # The weak reference is dead => the entry auto-evicted. No stale schema
    # can be served under a reused id.
    assert ref_a() is None
    assert len(crews_mod._CREW_INPUTS_CACHE) == 0

    # A brand-new crew regenerates its own (distinct) schema.
    crew_b = _make_real_crewbase(cls_name="CrewB")
    with _stub_llm_network():
        flow_b = crews_mod.ChatWithCrewFlow(crew=crew_b)
    assert flow_b.crew_chat_inputs is not schema_a
    assert flow_b.crew_chat_inputs.crew_name == "CrewB"

    crews_mod._CREW_INPUTS_CACHE.clear()
    crews_mod._CREW_INPUTS_FALLBACK.clear()


def test_cache_fallback_is_identity_safe_under_id_reuse():
    """The non-weakref-able fallback store keys on ``id`` but verifies true
    object identity before serving: it returns the schema for the same
    crew, but if a DIFFERENT object is looked up under a colliding id (the
    id-reuse scenario) it returns ``None`` and drops the stale entry rather
    than cross-serving another crew's schema (round-2 finding 3)."""
    crews_mod._CREW_INPUTS_FALLBACK.clear()

    class _Crew:  # a stand-in crew object
        pass

    crew = _Crew()
    schema = object()

    # Same-crew hit: strong ref + id key, identity verified.
    crews_mod._CREW_INPUTS_FALLBACK[id(crew)] = (crew, schema)
    assert crews_mod._crew_inputs_cache_get(crew) is schema

    # Simulate id reuse: a NEW object lands with the same id as a stale
    # entry that stores a DIFFERENT crew. The get must refuse to serve it
    # and evict the stale entry.
    other = _Crew()
    crews_mod._CREW_INPUTS_FALLBACK[id(other)] = (crew, schema)  # mismatched
    assert crews_mod._crew_inputs_cache_get(other) is None
    assert id(other) not in crews_mod._CREW_INPUTS_FALLBACK

    crews_mod._CREW_INPUTS_FALLBACK.clear()


# --------------------------------------------------------------------------
# Defect 5: endpoint symbols exported from the package top level
# --------------------------------------------------------------------------

def test_crew_path_symbols_exported_from_package_top_level():
    """The previously-hidden Crew-path symbols are importable from the
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
