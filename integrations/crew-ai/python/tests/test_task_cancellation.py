"""
Tests for Defect B: orphaned kickoff task and missing timeout in the CrewAI
FastAPI endpoint.

The event-generator in ``add_crewai_flow_fastapi_endpoint`` used to spawn the
flow via ``asyncio.create_task(flow.kickoff_async(...))`` *without* holding a
reference and *without* cancelling the task in the generator's ``finally``
block. If the HTTP client disconnected mid-stream, the kickoff coroutine would
continue running forever (leaking memory, file descriptors, and LLM
subscriptions). These tests pin down the desired behaviour across BOTH
endpoint factories (flow and crew):

* The generator keeps a handle to the kickoff task.
* When the generator is closed (as happens when Starlette detects a client
  disconnect), the kickoff task is cancelled and awaited.
* A hard wall-clock ceiling (``AGUI_CREWAI_FLOW_TIMEOUT_SECONDS``) is applied so
  a runaway flow cannot hang the process indefinitely.
* On timeout the stream surfaces a well-formed ``RunErrorEvent`` whose
  ``message`` carries thread/run correlation and the ceiling diagnostic.
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from ag_ui.core import RunAgentInput


# -- helpers ----------------------------------------------------------------

class _HangingFlow:
    """A minimal stand-in for a crewai.Flow that hangs inside kickoff_async.

    We only implement the surface the endpoint touches: ``kickoff_async`` and
    ``__deepcopy__`` (so copy.deepcopy is a no-op and we can observe the same
    instance the endpoint uses). A module-level event lets tests observe when
    the coroutine is cancelled.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None):  # noqa: D401
        self.started.set()
        try:
            # Hang forever until cancelled.
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _FakeCrew:
    """Placeholder Crew for the crew endpoint. Not used because
    ``ChatWithCrewFlow`` construction is monkeypatched in the crew tests."""


def _register_with_factory(factory_name: str, app: FastAPI, flow: _HangingFlow, path: str,
                           endpoint_module, monkeypatch) -> None:
    """Register the hanging flow against the given factory name.

    For ``add_crewai_flow_fastapi_endpoint`` the registration is direct.
    For ``add_crewai_crew_fastapi_endpoint`` we monkeypatch the internal
    ``ChatWithCrewFlow`` symbol so ``_get_flow()`` yields our hanging stub
    instead of constructing a real flow.
    """
    if factory_name == "flow":
        endpoint_module.add_crewai_flow_fastapi_endpoint(app, flow, path=path)
    elif factory_name == "crew":
        # Swap in a factory that returns the hanging stub, bypassing both
        # ChatWithCrewFlow.__init__ and the real Flow machinery.
        monkeypatch.setattr(
            endpoint_module, "ChatWithCrewFlow",
            lambda *_args, **_kwargs: flow,
        )
        endpoint_module.add_crewai_crew_fastapi_endpoint(app, _FakeCrew(), path=path)
    else:
        raise ValueError(factory_name)


def _make_input() -> RunAgentInput:
    return RunAgentInput(
        thread_id="t-1",
        run_id="r-1",
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={},
    )


def _make_request() -> SimpleNamespace:
    """Build a Request-ish stand-in; only ``headers.get`` is read."""
    return SimpleNamespace(
        headers=SimpleNamespace(get=lambda _k, default=None: "text/event-stream"),
    )


# -- tests ------------------------------------------------------------------


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_kickoff_task_is_cancelled_on_client_disconnect(factory, monkeypatch):
    """When the generator is closed, the kickoff task must be cancelled.

    Exercised against BOTH endpoint factories (flow + crew) — they share the
    same cancellation contract.
    """
    from ag_ui_crewai import endpoint as ep

    flow = _HangingFlow()
    app = FastAPI()
    _register_with_factory(factory, app, flow, "/run", ep, monkeypatch)

    # Reach into the route to invoke the generator directly. TestClient's
    # synchronous disconnect semantics don't exercise the cancel path we
    # care about, so we drive the async generator by hand.
    route = next(r for r in app.router.routes if getattr(r, "path", None) == "/run")
    endpoint_fn = route.endpoint

    fake_request = _make_request()

    response = await endpoint_fn(_make_input(), fake_request)
    body_iter = response.body_iterator

    # Pump once to ensure the task is scheduled and running.
    pump = asyncio.create_task(anext(body_iter))
    await asyncio.wait_for(flow.started.wait(), timeout=2.0)

    # Simulate client disconnect: cancel the pending __anext__ and aclose.
    # Starlette's real disconnect path calls aclose() on the body iterator
    # after cancelling outstanding reads; mirror that here.
    pump.cancel()
    try:
        await pump
    except (asyncio.CancelledError, StopAsyncIteration):
        pass
    await body_iter.aclose()

    # The kickoff coroutine must have been cancelled.
    await asyncio.wait_for(flow.cancelled.wait(), timeout=2.0)


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_flow_timeout_env_var_bounds_execution(monkeypatch, factory):
    """Setting ``AGUI_CREWAI_FLOW_TIMEOUT_SECONDS`` caps flow execution."""
    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "0.2")

    # The module reads the env var per-request, so no reload is needed;
    # reloading would re-register listeners on the crewai global event bus.
    from ag_ui_crewai import endpoint as ep

    flow = _HangingFlow()
    app = FastAPI()
    _register_with_factory(factory, app, flow, "/run", ep, monkeypatch)

    route = next(r for r in app.router.routes if getattr(r, "path", None) == "/run")
    endpoint_fn = route.endpoint
    fake_request = _make_request()

    response = await endpoint_fn(_make_input(), fake_request)
    body_iter = response.body_iterator

    # Drain until the generator completes. It must terminate in well under the
    # outer test timeout: the flow would otherwise hang forever.
    drained: list[bytes] = []

    async def _drain():
        async for chunk in body_iter:
            drained.append(chunk)

    await asyncio.wait_for(_drain(), timeout=5.0)

    # The kickoff coroutine must have been cancelled by the timeout path.
    assert flow.cancelled.is_set(), "kickoff task was not cancelled by the timeout"

    # Parse the SSE frames and assert we got a RUN_ERROR event whose
    # ``message`` contains the ceiling diagnostic.
    parts = [p.decode("utf-8", errors="replace") if isinstance(p, (bytes, bytearray)) else p
             for p in drained]
    joined = "".join(parts)

    # SSE frames look like: ``data: {...}\n\n`` — one per event. Extract the
    # JSON payloads.
    payloads = []
    for m in re.finditer(r"^data:\s*(\{.*\})\s*$", joined, flags=re.MULTILINE):
        try:
            payloads.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue

    run_errors = [p for p in payloads if p.get("type") == "RUN_ERROR"]
    assert run_errors, (
        "expected a RUN_ERROR event in the stream; "
        f"got payloads={payloads!r} raw={joined[:400]!r}"
    )
    error_msg = run_errors[0].get("message", "")
    assert "exceeded" in error_msg or "ceiling" in error_msg, (
        f"RunErrorEvent message should mention the ceiling; got: {error_msg!r}"
    )
    # Correlation IDs must be in the message for operator log-tracing.
    assert "t-1" in error_msg and "r-1" in error_msg, (
        f"RunErrorEvent message should carry thread/run correlation; got: {error_msg!r}"
    )
