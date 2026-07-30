"""Per-thread isolation of CrewAI crew memory.

Everything here runs OFFLINE against a real ``Crew(memory=True)``: a fake
embedding function stands in for OpenAI, and every write supplies an explicit
scope / categories / importance, which is the condition under which crewai's
``EncodingFlow`` skips its LLM analysis step. The only thing mocked is the
embedder, so the store, the scope arithmetic and the recall path are all real.

The seam under test is the one the FastAPI endpoints use: copy the flow for this
request, then scope the copy's crew memory to the request's ``threadId``.
"""

import copy
import hashlib
import logging

import pytest
from crewai import Agent, Crew, Flow, Task
from crewai.flow import start
from chromadb.api.types import EmbeddingFunction as _ChromaEmbeddingFunction
from crewai.rag.embeddings.providers.custom.embedding_callable import (
    CustomEmbeddingFunction as _CrewEmbeddingFunction,
)

from ag_ui_crewai import _memory as memory_module
from ag_ui_crewai._memory import apply_thread_memory_scope, thread_scope_path
from ag_ui_crewai.endpoint import _copy_flow


class _DeterministicEmbedder(_CrewEmbeddingFunction, _ChromaEmbeddingFunction):
    """Offline embedder: the first 16 bytes of a SHA-256 digest, scaled to [0, 1].

    crewai validates the ``custom`` embedder spec against chromadb's
    ``EmbeddingFunction`` and then against its own ``CustomEmbeddingFunction``,
    so the class has to satisfy both.
    """

    def __init__(self):
        # chromadb warns when an embedding function omits ``__init__``.
        pass

    def __call__(self, input):  # noqa: A002 - the upstream parameter is named ``input``
        return [
            [byte / 255.0 for byte in hashlib.sha256(text.encode()).digest()[:16]]
            for text in input
        ]


_EMBEDDER_SPEC = {
    "provider": "custom",
    "config": {"embedding_callable": _DeterministicEmbedder},
}


def _remember(view, content):
    """Write through a ``Memory`` or ``MemoryScope`` without invoking an LLM."""
    return view.remember(
        content, scope="/facts", categories=["fact"], importance=0.9
    )


def _recall(view, query):
    return [match.record.content for match in view.recall(query, depth="shallow")]


@pytest.fixture
def memory_crew(tmp_path, monkeypatch):
    """A real ``Crew(memory=True)`` whose store lives under ``tmp_path``.

    ``CREWAI_STORAGE_DIR`` is read when the storage backend is CONSTRUCTED, which
    happens inside ``Crew(...)`` via the ``create_crew_memory`` model validator,
    so setting it here (before the crew is built) is early enough. It must be
    absolute; see the note in ``conftest.py``.
    """
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crewai-store"))
    agent = Agent(role="helper", goal="help", backstory="b", llm="gpt-4o-mini")
    task = Task(description="d", expected_output="e", agent=agent)
    return Crew(
        name="support-crew",
        agents=[agent],
        tasks=[task],
        memory=True,
        embedder=_EMBEDDER_SPEC,
    )


class _CrewHoldingFlow(Flow):
    """Minimal stand-in for ``ChatWithCrewFlow``: a flow with a ``crew`` attribute.

    ``ChatWithCrewFlow.__init__`` issues a live LLM call to generate its chat
    inputs, so it cannot be constructed offline. The only property this seam cares
    about is the one reproduced here: a crew reachable as a flow attribute, shared
    by every request because the flow is cached per endpoint.
    """

    def __init__(self, crew):
        super().__init__()
        self.crew = crew

    @start()
    def go(self):  # pragma: no cover - never kicked off in these tests
        return "ok"


@pytest.fixture
def crew_flow(memory_crew):
    return _CrewHoldingFlow(memory_crew)


def _serve(flow, thread_id):
    """Do to ``flow`` exactly what an endpoint does at the start of a request."""
    flow_copy = _copy_flow(flow)
    apply_thread_memory_scope(flow_copy, thread_id)
    return flow_copy


@pytest.fixture(autouse=True)
def _reset_degrade_warning():
    """Un-latch the one-shot degradation warning between tests."""
    memory_module._DEGRADE_WARNED = False
    yield
    memory_module._DEGRADE_WARNED = False


# ---------------------------------------------------------------------------
# The reported bug
# ---------------------------------------------------------------------------


def test_two_threads_cannot_read_each_others_memory(crew_flow):
    """The reported leak: chat B must not see what chat A remembered."""
    run_a = _serve(crew_flow, "thread-A")
    _remember(run_a.crew._memory, "THREAD-A-SECRET: the user's name is Ada")

    run_b = _serve(crew_flow, "thread-B")

    assert _recall(run_b.crew._memory, "what is the user's name") == []
    # ...and A did not lose its own memory in the process.
    assert any(
        "THREAD-A-SECRET" in content
        for content in _recall(run_a.crew._memory, "what is the user's name")
    )


def test_writes_from_both_threads_stay_separated(crew_flow):
    """Symmetry: neither direction leaks once both threads have written."""
    run_a = _serve(crew_flow, "thread-A")
    run_b = _serve(crew_flow, "thread-B")

    _remember(run_a.crew._memory, "A-ONLY: the favourite colour is teal")
    _remember(run_b.crew._memory, "B-ONLY: the favourite colour is amber")

    assert _recall(run_a.crew._memory, "favourite colour") == [
        "A-ONLY: the favourite colour is teal"
    ]
    assert _recall(run_b.crew._memory, "favourite colour") == [
        "B-ONLY: the favourite colour is amber"
    ]


def test_one_thread_keeps_its_memory_across_sequential_runs(crew_flow):
    """Isolation must not degenerate into amnesia: a thread still has a history."""
    first_run = _serve(crew_flow, "thread-A")
    _remember(first_run.crew._memory, "REMEMBERED: the deploy target is staging")

    second_run = _serve(crew_flow, "thread-A")
    third_run = _serve(crew_flow, "thread-A")

    assert _recall(second_run.crew._memory, "deploy target") == [
        "REMEMBERED: the deploy target is staging"
    ]
    assert _recall(third_run.crew._memory, "deploy target") == [
        "REMEMBERED: the deploy target is staging"
    ]


# ---------------------------------------------------------------------------
# Race safety: the template crew is shared across concurrent requests
# ---------------------------------------------------------------------------


def test_shared_template_crew_is_never_mutated(crew_flow):
    """Scoping must not re-point the crew every other in-flight request holds."""
    template_crew = crew_flow.crew
    template_memory = template_crew._memory

    run = _serve(crew_flow, "thread-A")

    assert crew_flow.crew is template_crew
    assert template_crew._memory is template_memory
    assert type(template_memory).__name__ == "Memory"
    # The request got its own crew view carrying a scoped memory.
    assert run.crew is not template_crew
    assert type(run.crew._memory).__name__ == "MemoryScope"


def test_concurrent_requests_get_independent_scopes(crew_flow):
    """Two overlapping requests must not clobber one another's scope."""
    run_a = _serve(crew_flow, "thread-A")
    run_b = _serve(crew_flow, "thread-B")

    assert run_a.crew is not run_b.crew
    assert run_a.crew._memory.root_path != run_b.crew._memory.root_path
    # Both views still share the one physical store; only the namespace differs.
    assert run_a.crew._memory._memory is run_b.crew._memory._memory


def test_scoped_view_keeps_the_rest_of_the_crew_shared(crew_flow):
    """Only ``_memory`` is re-pointed; nothing else about the crew changes."""
    run = _serve(crew_flow, "thread-A")

    assert run.crew.name == crew_flow.crew.name
    assert run.crew.agents == crew_flow.crew.agents
    assert run.crew.tasks == crew_flow.crew.tasks


# ---------------------------------------------------------------------------
# Scope-path derivation
# ---------------------------------------------------------------------------


def test_scope_path_is_derived_from_the_thread_id():
    path = thread_scope_path("Thread A")
    assert path.startswith("/thread/thread-a-")
    assert thread_scope_path("Thread A") == path


def test_scope_paths_differ_for_ids_that_sanitise_alike():
    """Sanitisation is lossy; the digest suffix is what keeps threads apart."""
    assert thread_scope_path("a/b") != thread_scope_path("a-b")
    assert thread_scope_path("x" * 200) != thread_scope_path("x" * 201)


def test_scope_path_survives_an_id_with_no_usable_characters():
    assert thread_scope_path("///").startswith("/thread/unknown-")


def test_two_threads_whose_ids_sanitise_alike_are_isolated(crew_flow):
    """The end-to-end consequence of the digest suffix."""
    run_slash = _serve(crew_flow, "tenant/7")
    run_dash = _serve(crew_flow, "tenant-7")

    _remember(run_slash.crew._memory, "SLASH-TENANT: quota is 40")

    assert _recall(run_dash.crew._memory, "quota") == []


# ---------------------------------------------------------------------------
# Opt-out
# ---------------------------------------------------------------------------


def test_opt_out_restores_the_shared_namespace(crew_flow, monkeypatch):
    """``AGUI_CREWAI_THREAD_SCOPED_MEMORY=false`` reinstates one namespace per crew."""
    monkeypatch.setenv("AGUI_CREWAI_THREAD_SCOPED_MEMORY", "false")

    run_a = _serve(crew_flow, "thread-A")
    _remember(run_a.crew._memory, "SHARED: the office is in Lisbon")
    run_b = _serve(crew_flow, "thread-B")

    assert run_a.crew is crew_flow.crew
    assert _recall(run_b.crew._memory, "where is the office") == [
        "SHARED: the office is in Lisbon"
    ]


def test_an_unrecognised_opt_out_value_keeps_isolation_on(crew_flow, monkeypatch):
    """A typo must not silently reopen the leak."""
    monkeypatch.setenv("AGUI_CREWAI_THREAD_SCOPED_MEMORY", "flase")

    run_a = _serve(crew_flow, "thread-A")
    _remember(run_a.crew._memory, "TYPO-GUARD: the office is in Lisbon")
    run_b = _serve(crew_flow, "thread-B")

    assert _recall(run_b.crew._memory, "where is the office") == []


# ---------------------------------------------------------------------------
# Degradation and no-ops
# ---------------------------------------------------------------------------


def test_a_crew_without_memory_is_left_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crewai-store"))
    agent = Agent(role="helper", goal="help", backstory="b", llm="gpt-4o-mini")
    task = Task(description="d", expected_output="e", agent=agent)
    crew = Crew(name="memoryless", agents=[agent], tasks=[task])
    flow = _CrewHoldingFlow(crew)

    run = _serve(flow, "thread-A")

    assert run.crew._memory is None


def test_a_missing_thread_id_leaves_the_crew_untouched(crew_flow):
    run = _serve(crew_flow, "")

    assert type(run.crew._memory).__name__ == "Memory"


def test_a_memory_without_the_view_api_degrades_with_one_warning(crew_flow, caplog):
    """No crash, no isolation, and exactly one warning per process."""

    class _LegacyMemory:
        """A memory object predating (or replacing) ``Memory.scope``."""

    unscopable_crew = copy.copy(crew_flow.crew)
    unscopable_crew.__pydantic_private__["_memory"] = _LegacyMemory()
    flow = _CrewHoldingFlow(unscopable_crew)

    with caplog.at_level(logging.WARNING, logger="ag_ui_crewai._memory"):
        first = _serve(flow, "thread-A")
        second = _serve(flow, "thread-B")

    assert isinstance(first.crew._memory, _LegacyMemory)
    assert isinstance(second.crew._memory, _LegacyMemory)
    warnings = [
        record for record in caplog.records
        if "PER-THREAD MEMORY ISOLATION IS NOT ACTIVE" in record.message
    ]
    assert len(warnings) == 1


def test_a_failing_scope_factory_degrades_instead_of_failing_the_run(
    crew_flow, caplog
):
    """A crewai-side error while building the view must not kill the chat."""

    class _AngryMemory:
        def scope(self, path):
            raise RuntimeError("boom")

    angry_crew = copy.copy(crew_flow.crew)
    angry_crew.__pydantic_private__["_memory"] = _AngryMemory()
    flow = _CrewHoldingFlow(angry_crew)

    with caplog.at_level(logging.WARNING, logger="ag_ui_crewai._memory"):
        run = _serve(flow, "thread-A")

    assert isinstance(run.crew._memory, _AngryMemory)
    assert any("could not scope crew memory" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Endpoint wiring
# ---------------------------------------------------------------------------


class _InertFlow:
    """A flow-shaped object that neither runs nor supports StreamFrames."""

    async def kickoff_async(self, inputs=None):  # pragma: no cover - never awaited
        return "ok"


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_both_endpoint_factories_scope_memory_before_running(
    factory, monkeypatch
):
    """Both endpoints scope the request's flow copy, and do it BEFORE a driver runs.

    Scoping happens in the endpoint body, ahead of ``_run_flow_stream``'s choice
    between the legacy ``kickoff_async`` driver and the ``astream`` StreamFrame
    driver, so it cannot be dead code on one of them.
    """
    from types import SimpleNamespace

    from ag_ui.core import RunAgentInput
    from fastapi import FastAPI

    from ag_ui_crewai import endpoint as ep

    calls = []
    monkeypatch.setattr(
        ep,
        "apply_thread_memory_scope",
        lambda flow_copy, thread_id: calls.append((flow_copy, thread_id)),
    )

    app = FastAPI()
    if factory == "flow":
        ep.add_crewai_flow_fastapi_endpoint(app, _InertFlow(), path="/run")
    else:
        monkeypatch.setattr(
            ep, "ChatWithCrewFlow", lambda *_a, **_kw: _InertFlow()
        )
        ep.add_crewai_crew_fastapi_endpoint(app, object(), path="/run")

    route = next(r for r in app.router.routes if getattr(r, "path", None) == "/run")
    request = SimpleNamespace(headers=SimpleNamespace(get=lambda *_a, **_kw: None))
    await route.endpoint(
        RunAgentInput(
            thread_id="thread-A",
            run_id="run-1",
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        ),
        request,
    )

    assert len(calls) == 1
    scoped_flow, thread_id = calls[0]
    assert thread_id == "thread-A"
    assert isinstance(scoped_flow, _InertFlow)
