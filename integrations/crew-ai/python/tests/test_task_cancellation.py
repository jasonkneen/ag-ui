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
  ``message`` carries thread/run correlation and the ceiling diagnostic, AND
  whose event-level ``thread_id`` / ``run_id`` extras expose the same IDs
  without requiring consumers to string-parse.
* If ``kickoff_async`` itself raises (auth failure, crewai internal bug), the
  stream surfaces a ``RUN_ERROR`` event carrying the real cause promptly,
  rather than blocking on the queue until the flow-timeout ceiling fires.
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

    Single-use contract: ``__deepcopy__`` returns ``self`` so the endpoint's
    ``copy.deepcopy(flow)`` is a no-op. This is safe for these tests because
    each test instantiates its own ``_HangingFlow`` with fresh ``Event``
    objects; do NOT share a single instance across multiple endpoint
    invocations within a test.
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


class _ExplodingFlow:
    """A flow whose ``kickoff_async`` raises immediately.

    Used to pin down finding #3: kickoff-task exceptions must be raced
    against the queue and surfaced as ``RUN_ERROR`` promptly, not held
    hostage by the flow-timeout ceiling.
    """

    def __init__(self, message: str) -> None:
        self._message = message

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None):  # noqa: D401
        raise RuntimeError(self._message)


class _FakeCrew:
    """Placeholder Crew for the crew endpoint. Internals are unused because
    ``ChatWithCrewFlow`` is monkeypatched in the crew tests."""


def _register_with_factory(factory_name: str, app: FastAPI, flow, path: str,
                           endpoint_module, monkeypatch) -> None:
    """Register the given flow against the given factory name.

    For ``add_crewai_flow_fastapi_endpoint`` the registration is direct.
    For ``add_crewai_crew_fastapi_endpoint`` we monkeypatch the internal
    ``ChatWithCrewFlow`` symbol so ``_get_flow()`` yields our stub instead
    of constructing a real flow.
    """
    if factory_name == "flow":
        endpoint_module.add_crewai_flow_fastapi_endpoint(app, flow, path=path)
    elif factory_name == "crew":
        # Swap in a factory that returns the stub, bypassing both
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
    """Build a Request-ish stand-in; only ``headers.get("accept")`` is read.

    The stub narrowly returns ``text/event-stream`` only for the ``accept``
    header; any other key lookup falls through to the supplied default, so
    future code paths that probe other headers do not silently collide with
    an SSE content-type.
    """

    def _get(key, default=None):
        if isinstance(key, str) and key.lower() == "accept":
            return "text/event-stream"
        return default

    return SimpleNamespace(headers=SimpleNamespace(get=_get))


def _parse_sse_payloads(raw: str) -> list[dict]:
    """Parse an SSE byte-stream (joined to str) into decoded JSON payloads.

    Splits on the SSE frame separator ``\\n\\n`` and extracts ``data:``
    lines. This is robust against multi-line JSON encoders that a plain
    regex over the whole string would miss.
    """

    payloads: list[dict] = []
    for frame in raw.split("\n\n"):
        data_lines = [
            line[len("data:"):].lstrip()
            for line in frame.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        # SSE allows a data payload to span multiple lines; rejoin with "\n".
        payload_text = "\n".join(data_lines).strip()
        if not payload_text:
            continue
        try:
            payloads.append(json.loads(payload_text))
        except json.JSONDecodeError:
            continue
    return payloads


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

    # Parse the SSE frames and assert we got a RUN_ERROR event.
    parts = [p.decode("utf-8", errors="replace") if isinstance(p, (bytes, bytearray)) else p
             for p in drained]
    joined = "".join(parts)
    payloads = _parse_sse_payloads(joined)

    run_errors = [p for p in payloads if p.get("type") == "RUN_ERROR"]
    assert run_errors, (
        "expected a RUN_ERROR event in the stream; "
        f"got payloads={payloads!r} raw={joined[:400]!r}"
    )
    err = run_errors[0]
    error_msg = err.get("message", "")

    # Tight assertions: both diagnostic words AND the specific ceiling value
    # must appear, so that a regression (e.g. empty message, missing ceiling
    # value) is caught.
    assert "exceeded" in error_msg, (
        f"RunErrorEvent message should mention 'exceeded'; got: {error_msg!r}"
    )
    assert "ceiling" in error_msg, (
        f"RunErrorEvent message should mention 'ceiling'; got: {error_msg!r}"
    )
    assert "0.2" in error_msg, (
        f"RunErrorEvent message should include the configured ceiling value (0.2); "
        f"got: {error_msg!r}"
    )

    # Correlation IDs must be in the message (human-readable log grep) AND
    # exposed as event-level extras (machine-parseable for downstream
    # consumers without string-parsing).
    assert "t-1" in error_msg and "r-1" in error_msg, (
        f"RunErrorEvent message should carry thread/run correlation; got: {error_msg!r}"
    )
    assert err.get("thread_id") == "t-1", (
        f"RunErrorEvent should expose thread_id as an event extra; got: {err!r}"
    )
    assert err.get("run_id") == "r-1", (
        f"RunErrorEvent should expose run_id as an event extra; got: {err!r}"
    )

    # Timeout path should also flag a distinguishing code so downstream log
    # consumers can filter.
    assert err.get("code") == "AGUI_CREWAI_FLOW_TIMEOUT", (
        f"RunErrorEvent timeout path should set code=AGUI_CREWAI_FLOW_TIMEOUT; "
        f"got: {err.get('code')!r}"
    )


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_kickoff_exception_is_surfaced_promptly(factory, monkeypatch):
    """If ``kickoff_async`` itself raises, the stream must surface the real
    cause as a ``RUN_ERROR`` event without waiting for the flow-timeout
    ceiling to fire. Red-green pin for finding #3.
    """
    # Set a generous flow ceiling so that if the fix regresses (loop blocks on
    # queue.get instead of racing the kickoff task), the test would have to
    # wait the full ceiling before giving up. Our outer wait_for is shorter
    # than the ceiling, so a regression manifests as a timeout here.
    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "30")

    from ag_ui_crewai import endpoint as ep

    marker = "kickoff blew up: unique-marker-7F3A"
    flow = _ExplodingFlow(marker)
    app = FastAPI()
    _register_with_factory(factory, app, flow, "/run", ep, monkeypatch)

    route = next(r for r in app.router.routes if getattr(r, "path", None) == "/run")
    endpoint_fn = route.endpoint
    fake_request = _make_request()

    response = await endpoint_fn(_make_input(), fake_request)
    body_iter = response.body_iterator

    drained: list[bytes] = []

    async def _drain():
        async for chunk in body_iter:
            drained.append(chunk)

    # 5s is dramatically shorter than the 30s env-var ceiling; if the
    # kickoff race is missing or broken, this wait_for raises.
    await asyncio.wait_for(_drain(), timeout=5.0)

    parts = [p.decode("utf-8", errors="replace") if isinstance(p, (bytes, bytearray)) else p
             for p in drained]
    joined = "".join(parts)
    payloads = _parse_sse_payloads(joined)

    run_errors = [p for p in payloads if p.get("type") == "RUN_ERROR"]
    assert run_errors, (
        "expected a RUN_ERROR event in the stream; "
        f"got payloads={payloads!r} raw={joined[:400]!r}"
    )
    err = run_errors[0]
    code = err.get("code", "")

    # Code must be the generic FLOW_ERROR family (carrying the exception
    # class name as a suffix), NOT the TIMEOUT code.
    assert code.startswith("AGUI_CREWAI_FLOW_ERROR"), (
        f"kickoff exceptions should use the FLOW_ERROR code family, not "
        f"FLOW_TIMEOUT; got code={code!r}"
    )
    assert "RuntimeError" in code, (
        f"FLOW_ERROR code should encode the exception class name; got code={code!r}"
    )

    # Coarse message: must carry correlation + class name; must NOT leak the
    # raw exception repr (which contained our unique marker). This is the
    # finding #10 contract.
    message = err.get("message", "")
    assert "t-1" in message and "r-1" in message, (
        f"RunErrorEvent message should carry thread/run correlation; got: {message!r}"
    )
    assert "RuntimeError" in message, (
        f"RunErrorEvent message should include the exception class name; got: {message!r}"
    )
    assert marker not in message, (
        f"RunErrorEvent message should NOT leak the internal exception repr; "
        f"got: {message!r}"
    )

    # Correlation extras should still be present.
    assert err.get("thread_id") == "t-1", err
    assert err.get("run_id") == "r-1", err
