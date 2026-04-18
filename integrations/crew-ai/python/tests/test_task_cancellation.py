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
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from ag_ui.core import EventType, RunAgentInput, RunStartedEvent


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
    """Inert Crew stand-in for the crew-endpoint factory.

    The crew-endpoint tests monkeypatch ``ChatWithCrewFlow`` itself, so the
    endpoint never routes calls back to this object — its only purpose is
    to be accepted by ``add_crewai_crew_fastapi_endpoint`` as the ``crew``
    argument. To catch accidental surface-additions (a future refactor
    that starts calling ``crew.<method>`` at request time would silently
    "succeed" under a plain stub), we raise on any attribute access.
    Finding #12 — spy pattern.
    """

    def __getattr__(self, name):  # noqa: D401
        raise AssertionError(
            f"_FakeCrew accessed unexpected attribute {name!r}; the "
            "crew endpoint should not touch the crew object directly "
            "when ChatWithCrewFlow is monkeypatched. If the endpoint "
            "now does, add the attribute to an allow-list here and in "
            "the factory contract."
        )


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

    We intentionally do NOT silently swallow JSON decode errors: a malformed
    SSE frame is a real bug and the test should fail loudly rather than
    hide it. Callers get a ``pytest.fail`` that pins the offending frame.
    """

    payloads: list[dict] = []
    for frame in raw.split("\n\n"):
        frame_lines = frame.splitlines()
        data_lines = [
            line[len("data:"):].lstrip()
            for line in frame_lines
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        # Defensive invariant (finding #13): each SSE frame produced by
        # EventEncoder carries exactly one ``data:`` line per payload.
        # A regression that pretty-prints JSON (inserting embedded blank
        # lines) would split the frame across the ``\n\n`` separator and
        # silently corrupt this parse; pin the invariant here.
        assert len(data_lines) == 1, (
            f"unexpected multi-line data frame (likely indented JSON "
            f"breaking \\n\\n frame separator): {frame!r}"
        )
        # SSE allows a data payload to span multiple lines; rejoin with "\n".
        payload_text = "\n".join(data_lines).strip()
        if not payload_text:
            continue
        try:
            payloads.append(json.loads(payload_text))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            pytest.fail(
                f"Malformed SSE data frame could not be parsed as JSON: "
                f"{payload_text!r} ({exc})"
            )
    return payloads


class _CompletingFlow:
    """A flow whose ``kickoff_async`` returns cleanly WITHOUT the listener
    having enqueued a ``None`` sentinel. Used to pin the CPU-spin regression
    (finding #1): if the main loop keeps re-creating get_task and waiting on
    ``{get_task, kickoff_task}`` after ``kickoff_task.done()``, it spins.
    """

    def __init__(self) -> None:
        self.done = asyncio.Event()

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None):  # noqa: D401
        # Return immediately and cleanly; we do NOT enqueue a sentinel,
        # because this test's purpose is to pin the behaviour when the
        # listener is not wired (e.g. if a listener misfires, or is
        # bypassed in a future refactor).
        self.done.set()
        return None


# -- tests ------------------------------------------------------------------


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_kickoff_task_is_cancelled_on_client_disconnect(monkeypatch, factory):
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
    await asyncio.wait_for(flow.started.wait(), timeout=10.0)

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
    await asyncio.wait_for(flow.cancelled.wait(), timeout=10.0)


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

    await asyncio.wait_for(_drain(), timeout=15.0)

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
    # exposed as event-level extras. We emit extras camelCased (``threadId``
    # / ``runId``) so the wire format lines up with peer events
    # (RunStartedEvent / RunFinishedEvent) whose declared fields are
    # camelCased by the alias generator. Finding #3: snake_case extras on
    # RunErrorEvent were a protocol inconsistency.
    assert "t-1" in error_msg and "r-1" in error_msg, (
        f"RunErrorEvent message should carry thread/run correlation; got: {error_msg!r}"
    )
    assert err.get("threadId") == "t-1", (
        f"RunErrorEvent should expose threadId as a camelCase event extra; got: {err!r}"
    )
    assert err.get("runId") == "r-1", (
        f"RunErrorEvent should expose runId as a camelCase event extra; got: {err!r}"
    )
    assert "thread_id" not in err, (
        f"RunErrorEvent should NOT expose snake_case thread_id (wire format "
        f"must match peer events); got: {err!r}"
    )
    assert "run_id" not in err, (
        f"RunErrorEvent should NOT expose snake_case run_id (wire format "
        f"must match peer events); got: {err!r}"
    )

    # Timeout path should also flag a distinguishing code so downstream log
    # consumers can filter.
    assert err.get("code") == "AGUI_CREWAI_FLOW_TIMEOUT", (
        f"RunErrorEvent timeout path should set code=AGUI_CREWAI_FLOW_TIMEOUT; "
        f"got: {err.get('code')!r}"
    )

    # Finding #8: RUN_ERROR must be terminal on the timeout path. A regression
    # that emits both RUN_FINISHED and RUN_ERROR would still parse — this
    # assertion pins the contract that only RUN_ERROR appears.
    all_types = [p.get("type") for p in payloads]
    assert "RUN_FINISHED" not in all_types, (
        "RUN_FINISHED must NOT appear in the stream on the timeout path; "
        f"got payload types={all_types!r}"
    )
    assert payloads[-1].get("type") == "RUN_ERROR", (
        f"RUN_ERROR must be the terminal event on the timeout path; "
        f"got trailing type={payloads[-1].get('type')!r} all={all_types!r}"
    )
    # Finding #4: the previous assertion scanned for ``event:`` header
    # lines, but EventEncoder's SSE format emits only ``data:`` frames —
    # the scan is vacuously empty, so the assertion was silently passing
    # regardless of whether RUN_FINISHED was present at the wire layer.
    # The payload-type assertion above already pins the contract; the
    # redundant frame-name scan has been removed.


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_kickoff_exception_is_surfaced_promptly(monkeypatch, factory):
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

    # 15s is dramatically shorter than the 30s env-var ceiling; if the
    # kickoff race is missing or broken, this wait_for raises (finding
    # #20: earlier comment said 5s but was already bumped to 15s).
    await asyncio.wait_for(_drain(), timeout=15.0)

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

    # Coarse message: must carry correlation; must NOT leak the raw
    # exception repr (which contained our unique marker) NOR duplicate the
    # class name (which already lives in the ``code`` field). Finding #5.
    message = err.get("message", "")
    assert "t-1" in message and "r-1" in message, (
        f"RunErrorEvent message should carry thread/run correlation; got: {message!r}"
    )
    assert marker not in message, (
        f"RunErrorEvent message should NOT leak the internal exception repr; "
        f"got: {message!r}"
    )
    # The message previously duplicated the run_id ("run=X ... see server
    # logs for run=X") and the class name (already in ``code``). Tighten:
    # run_id should appear exactly once in the message body.
    assert message.count("r-1") == 1, (
        f"RunErrorEvent message should not duplicate run_id; got: {message!r}"
    )
    assert "RuntimeError" not in message, (
        f"RunErrorEvent message should NOT duplicate the class name (already "
        f"in code={code!r}); got: {message!r}"
    )

    # Correlation extras should still be present, camelCased (finding #3).
    assert err.get("threadId") == "t-1", err
    assert err.get("runId") == "r-1", err
    assert "thread_id" not in err, err
    assert "run_id" not in err, err


# -- R3 additions ----------------------------------------------------------


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_happy_path_no_spin_when_kickoff_completes_without_sentinel(
    monkeypatch, factory
):
    """Finding #1: if ``kickoff_async`` returns cleanly but no ``None``
    sentinel is enqueued (listener disabled, misfire, or future refactor),
    the generator must NOT spin on ``asyncio.wait({get_task, kickoff_task})``
    — which always returns immediately because ``kickoff_task`` is already
    done.

    The test wires a flow that completes with no sentinel. A correct
    implementation breaks out of the main loop once it observes
    ``kickoff_task.done()`` with ``exception() is None``. A broken
    implementation spins forever, which we detect via an outer wait_for
    ceiling much shorter than the flow-timeout ceiling.
    """
    # 60s flow ceiling — 20x longer than the spin-detection window. A
    # regression spins and we catch it via the outer wait_for.
    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "60")

    from ag_ui_crewai import endpoint as ep

    flow = _CompletingFlow()
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

    # Generator must complete well under the flow ceiling. A CPU spin
    # regression either (a) never completes and we time out, or (b) emits
    # a FLOW_TIMEOUT event (if the deadline fires first); both fail this
    # assertion. 3.0s is comfortable headroom over the ~0 expected runtime.
    await asyncio.wait_for(_drain(), timeout=3.0)

    assert flow.done.is_set(), "kickoff must have completed"

    # Positive pin (finding #11): the happy-path-no-sentinel flow
    # should produce an empty payload list (the flow emits nothing, no
    # listeners fire, no ``None`` sentinel is delivered). Asserting
    # ``payloads == []`` pins that we neither drop nor fabricate events
    # on this path. Also double-check no error-type events leaked even
    # if a future change starts yielding benign events here.
    parts = [p.decode("utf-8", errors="replace") if isinstance(p, (bytes, bytearray)) else p
             for p in drained]
    joined = "".join(parts)
    payloads = _parse_sse_payloads(joined)
    types = [p.get("type") for p in payloads]
    error_types = {t for t in types if isinstance(t, str) and "ERROR" in t}
    assert not error_types, (
        f"happy-path completion without sentinel should NOT emit any "
        f"error-typed events; got types={types!r}"
    )
    assert payloads == [], (
        f"happy-path completion without sentinel should yield an empty "
        f"event stream; got payloads={payloads!r}"
    )


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_run_error_wire_format_camelcase_extras(monkeypatch, factory):
    """Finding #3: verify by-alias round-trip of the RunErrorEvent wire
    payload. ``threadId`` / ``runId`` must be present, snake_case variants
    must not. A targeted assertion — independent of the broader timeout test
    — so a regression to snake_case is caught even if other assertions drift.
    """
    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "0.2")
    from ag_ui_crewai import endpoint as ep

    flow = _HangingFlow()
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

    await asyncio.wait_for(_drain(), timeout=15.0)
    joined = "".join(
        p.decode("utf-8", errors="replace") if isinstance(p, (bytes, bytearray)) else p
        for p in drained
    )
    payloads = _parse_sse_payloads(joined)
    run_errors = [p for p in payloads if p.get("type") == "RUN_ERROR"]
    assert run_errors, f"no RUN_ERROR; payloads={payloads!r}"
    err = run_errors[0]

    # camelCase assertions (match peer RunStartedEvent/RunFinishedEvent).
    assert err.get("threadId") == "t-1", err
    assert err.get("runId") == "r-1", err
    # snake_case must be absent — wire format alignment.
    assert "thread_id" not in err, err
    assert "run_id" not in err, err


class _RaceFlow:
    """Flow that completes quickly after enqueueing a single event.

    Used to pin the cancel-race invariant (R4 HIGH #1): the main loop
    must not silently drop an item delivered to ``get_task`` between
    ``asyncio.wait`` returning and the ``finally`` cancelling it.

    CR6-7 LOW #3 / CR6-6 LOW #2: the registry is supplied by the test
    rather than held in a module-level global. A module-level global
    ``dict`` is xdist-unsafe (process-parallel workers share import-time
    state via ``pickle``-ish boundaries, but mutating module state
    across parametrized cases serialised on one worker is still a
    cross-test coupling surface). Scoping the registry to the test
    fixture eliminates the coupling regardless of execution model.
    """

    def __init__(self, queue_event, registry: dict) -> None:
        self._queue_event = queue_event
        self._registry = registry
        self.done = asyncio.Event()

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None):  # noqa: D401
        # Enqueue one event *and* complete the task in the same tick so
        # ``asyncio.wait({get_task, kickoff_task}, FIRST_COMPLETED)`` may
        # observe ``kickoff_task in done`` while ``get_task`` was also
        # delivered an item that the old code cancelled and dropped.
        q = self._registry.get(id(self))
        if q is not None:
            q.put_nowait(self._queue_event)
        self.done.set()


@pytest.fixture
def _race_queue_registry() -> dict:
    """Per-test queue registry for ``_RaceFlow`` (CR6-7 LOW #3).

    Replaces the prior module-level ``_MODULE_QUEUE_REGISTRY`` global.
    Scoping to the test fixture keeps the side-channel local to the
    test case — no cross-test mutation, no xdist-serialisation concern,
    and no surprise when a future regression deletes the clear() call.
    """
    return {}


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_cancel_race_does_not_drop_delivered_queue_item(
    monkeypatch, factory, _race_queue_registry
):
    """R4 HIGH #1: if ``get_task`` is cancelled but had already been
    delivered a queue item, the item must be yielded (or the cancel must
    not happen). The cancel-race guard in the ``finally`` clause harvests
    ``get_task.result()`` before discarding.

    This test wires a flow whose ``kickoff_async`` enqueues a recognisable
    RunStartedEvent ``item`` and returns immediately. The race is that
    both the queue delivery and the kickoff completion may land in the
    same scheduler tick; we simulate the timing by running many rounds
    and asserting that the event is delivered every time.

    Amplification-style probe (CR6-6 LOW #3): this test runs 10
    independent rounds to amplify the timing-dependent race signal.
    The regression it pins does not reproduce deterministically in
    userspace — the scheduler may or may not surface the race on any
    single round. If this test ever flakes in CI, INCREASE the round
    count rather than remove; a single flaky run means a regression
    just barely slipped through the probe's catch net, not that the
    test is unreliable.
    """
    from ag_ui_crewai import endpoint as ep

    # Give the flow a long ceiling so the outer wait_for does not mask a
    # regression by firing the timeout path.
    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "30")

    # Wrap create_queue so _RaceFlow.kickoff_async can reach the queue.
    original_create_queue = ep.create_queue

    async def _tracked_create_queue(flow_obj):
        q = await original_create_queue(flow_obj)
        _race_queue_registry[id(flow_obj)] = q
        return q

    monkeypatch.setattr(ep, "create_queue", _tracked_create_queue)

    # Run 10 independent requests; every single one must deliver the
    # enqueued custom event. A regression drops the event on SOME of the
    # runs (the race is timing-dependent) — so loop to amplify the
    # signal.
    for _round in range(10):
        _race_queue_registry.clear()
        event = RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id="?",
            run_id="?",
        )
        flow = _RaceFlow(event, _race_queue_registry)
        app = FastAPI()
        _register_with_factory(factory, app, flow, "/run", ep, monkeypatch)

        route = next(
            r for r in app.router.routes if getattr(r, "path", None) == "/run"
        )
        endpoint_fn = route.endpoint
        fake_request = _make_request()
        response = await endpoint_fn(_make_input(), fake_request)
        body_iter = response.body_iterator

        drained: list[bytes] = []

        async def _drain():
            async for chunk in body_iter:
                drained.append(chunk)

        await asyncio.wait_for(_drain(), timeout=5.0)

        parts = [
            p.decode("utf-8", errors="replace")
            if isinstance(p, (bytes, bytearray))
            else p
            for p in drained
        ]
        joined = "".join(parts)
        payloads = _parse_sse_payloads(joined)
        types = [p.get("type") for p in payloads]
        assert "RUN_STARTED" in types, (
            f"round {_round}: enqueued RUN_STARTED was dropped by cancel-race; "
            f"got types={types!r}"
        )


class _LateEnqueueFlow:
    """Flow that schedules a queue ``put_nowait`` via ``call_soon`` just
    before returning from ``kickoff_async``. The enqueue fires the scheduler
    tick AFTER ``kickoff_task.done()`` becomes true.

    Used to pin R4 MEDIUM #3: the happy-path drain must not break on the
    first empty probe — it must yield once and probe again to catch a
    late-arriving listener enqueue (otherwise the event is silently dropped).

    R5 MEDIUM #9: the deferred-put task is retained on the instance so
    tests can cancel/await it during cleanup; a fire-and-forget
    ``create_task`` would occasionally leave a pending task on the loop
    at test teardown.
    """

    def __init__(self, late_event, registry: dict, delay_ticks: int = 2) -> None:
        self._late_event = late_event
        self._registry = registry
        self._delay_ticks = delay_ticks
        self._deferred_task: asyncio.Task | None = None
        self.done = asyncio.Event()

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None):  # noqa: D401
        # Spawn a background task that will enqueue AFTER this coroutine
        # returns. Using ``create_task`` + ``sleep(0)`` guarantees the
        # enqueue lands on a subsequent scheduler tick — simulating a
        # listener callback that fires right after ``kickoff_async``
        # returns but before the drain loop runs.
        q = self._registry.get(id(self))
        event = self._late_event
        delay_ticks = self._delay_ticks

        async def _deferred_put():
            # Yield ``delay_ticks`` times so we are strictly after the
            # main loop has observed ``kickoff_task.done()`` and entered
            # the drain. ``delay_ticks=2`` hits the fast-path; higher
            # values exercise the drain's extended pass budget.
            for _ in range(delay_ticks):
                await asyncio.sleep(0)
            if q is not None:
                q.put_nowait(event)

        # Retain a handle so tests can tear the task down deterministically
        # (R5 MEDIUM #9: previously fire-and-forget).
        self._deferred_task = asyncio.create_task(_deferred_put())
        self.done.set()

    async def await_deferred(self) -> None:
        """Await the deferred enqueue task, if it was scheduled."""
        if self._deferred_task is not None and not self._deferred_task.done():
            try:
                await asyncio.wait_for(self._deferred_task, timeout=1.0)
            except (asyncio.TimeoutError, TimeoutError):
                self._deferred_task.cancel()
                try:
                    await self._deferred_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass


async def test_happy_path_drain_captures_late_listener_enqueue(
    monkeypatch, _race_queue_registry
):
    """R4 MEDIUM #3: after ``kickoff_task.done()``, the drain loop must
    yield to the event loop and re-probe the queue — otherwise a
    listener that enqueues in the tick immediately after kickoff's
    return is silently dropped.

    Pre-fix drain was ``while get_nowait() ... break on QueueEmpty`` —
    one-shot. Post-fix drain does two passes with an ``asyncio.sleep(0)``
    between them so a ``call_soon``-scheduled listener callback has a
    chance to run and enqueue before the drain concludes.
    """
    from ag_ui_crewai import endpoint as ep

    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "30")

    original_create_queue = ep.create_queue

    async def _tracked_create_queue(flow_obj):
        q = await original_create_queue(flow_obj)
        _race_queue_registry[id(flow_obj)] = q
        return q

    monkeypatch.setattr(ep, "create_queue", _tracked_create_queue)

    late_event = RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id="?",
        run_id="?",
    )
    flow = _LateEnqueueFlow(late_event, _race_queue_registry)
    app = FastAPI()
    ep.add_crewai_flow_fastapi_endpoint(app, flow, path="/run")

    route = next(r for r in app.router.routes if getattr(r, "path", None) == "/run")
    endpoint_fn = route.endpoint
    response = await endpoint_fn(_make_input(), _make_request())
    body_iter = response.body_iterator

    drained: list[bytes] = []

    async def _drain():
        async for chunk in body_iter:
            drained.append(chunk)

    await asyncio.wait_for(_drain(), timeout=5.0)
    parts = [
        p.decode("utf-8", errors="replace") if isinstance(p, (bytes, bytearray)) else p
        for p in drained
    ]
    joined = "".join(parts)
    payloads = _parse_sse_payloads(joined)
    types = [p.get("type") for p in payloads]

    assert "RUN_STARTED" in types, (
        f"late-arriving listener enqueue was silently dropped by the "
        f"happy-path drain; expected RUN_STARTED in payload types, "
        f"got types={types!r}"
    )

    # R5 MEDIUM #9: await the deferred enqueue task so it does not leak
    # into later tests as a pending task on the event loop.
    await flow.await_deferred()


@pytest.mark.parametrize("delay_ticks", [3, 4])
async def test_happy_path_drain_captures_multi_tick_late_enqueue(
    monkeypatch, delay_ticks, _race_queue_registry
):
    """R5 HIGH #3 red-green: a listener enqueue that needs >1 scheduler
    tick after ``kickoff_task.done()`` to materialise must still be
    delivered. Pre-R5 the drain performed at most 2 passes (with a
    single ``sleep(0)`` between) and early-returned on the first empty
    pass — a listener whose ``call_soon`` chain fires 3+ ticks later
    would be silently dropped.

    Post-R5 the drain keeps looping while any pass drains something OR
    while we have consecutive-empty budget (two empty passes) AND the
    ``_DRAIN_MAX_PASSES`` / ``_DRAIN_BUDGET_SECONDS`` caps remain.
    """
    from ag_ui_crewai import endpoint as ep

    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "30")

    original_create_queue = ep.create_queue

    async def _tracked_create_queue(flow_obj):
        q = await original_create_queue(flow_obj)
        _race_queue_registry[id(flow_obj)] = q
        return q

    monkeypatch.setattr(ep, "create_queue", _tracked_create_queue)

    late_event = RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id="?",
        run_id="?",
    )
    flow = _LateEnqueueFlow(late_event, _race_queue_registry, delay_ticks=delay_ticks)
    app = FastAPI()
    ep.add_crewai_flow_fastapi_endpoint(app, flow, path="/run")

    route = next(r for r in app.router.routes if getattr(r, "path", None) == "/run")
    endpoint_fn = route.endpoint
    response = await endpoint_fn(_make_input(), _make_request())
    body_iter = response.body_iterator

    drained: list[bytes] = []

    async def _drain():
        async for chunk in body_iter:
            drained.append(chunk)

    await asyncio.wait_for(_drain(), timeout=5.0)
    parts = [
        p.decode("utf-8", errors="replace") if isinstance(p, (bytes, bytearray)) else p
        for p in drained
    ]
    joined = "".join(parts)
    payloads = _parse_sse_payloads(joined)
    types = [p.get("type") for p in payloads]

    assert "RUN_STARTED" in types, (
        f"multi-tick late enqueue (delay_ticks={delay_ticks}) was silently "
        f"dropped by the happy-path drain; expected RUN_STARTED in payload "
        f"types, got types={types!r}"
    )

    await flow.await_deferred()


def test_flow_timeout_nan_falls_back_to_default(monkeypatch):
    """R4 LOW #17: ``float('nan') > 0`` is False, which would silently
    disable the ceiling. A NaN env var must fall back to the default.
    """
    from ag_ui_crewai import endpoint as ep

    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "nan")
    result = ep._flow_timeout_seconds()
    assert result == ep._DEFAULT_FLOW_TIMEOUT_SECONDS, (
        f"NaN env var must fall back to default, not disable the ceiling; "
        f"got {result!r}"
    )


def test_cancel_join_timeout_env_override(monkeypatch):
    """R4 MEDIUM #8: operators must be able to tune the teardown ceiling
    via AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS to reduce tail latency
    under disconnect-heavy load.
    """
    from ag_ui_crewai import endpoint as ep

    monkeypatch.delenv("AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS", raising=False)
    assert ep._cancel_join_timeout_seconds() == ep._CANCEL_JOIN_TIMEOUT_SECONDS

    monkeypatch.setenv("AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS", "3.5")
    assert ep._cancel_join_timeout_seconds() == pytest.approx(3.5)

    monkeypatch.setenv("AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS", "not-a-number")
    assert ep._cancel_join_timeout_seconds() == ep._CANCEL_JOIN_TIMEOUT_SECONDS

    monkeypatch.setenv("AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS", "0")
    assert ep._cancel_join_timeout_seconds() == ep._CANCEL_JOIN_TIMEOUT_SECONDS

    monkeypatch.setenv("AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS", "nan")
    assert ep._cancel_join_timeout_seconds() == ep._CANCEL_JOIN_TIMEOUT_SECONDS


async def test_get_flow_is_serialized_under_concurrent_first_requests(monkeypatch):
    """R4 MEDIUM #6: two concurrent first-requests must not both
    construct ``ChatWithCrewFlow`` — that constructor issues a real LLM
    call and is expensive.

    Scope note (R5 MEDIUM #5): ``ChatWithCrewFlow.__init__`` is
    synchronous. The constructor IS slow (it issues a real LLM call), but
    from the event loop's perspective it runs to completion on a single
    tick — it cannot interleave with another coroutine via ``await``.
    The serializing ``asyncio.Lock`` guarantees that even if two
    coroutines race into ``_get_flow()``, only one crosses the
    ``_cached_flow is None`` gate before releasing the lock. A TRUE
    async-init race (where one coroutine awaits inside its __init__
    while another races past) would require a module-level hook that
    inserts an ``await`` between the counter increment and the lock
    release; we don't have that surface here and reproducing it would
    require patching both ``asyncio.Lock`` and adding an async
    constructor shim.

    This test is therefore a POSITIVE-CONTRACT PIN only: it asserts that
    the "check then construct then cache" sequence under the lock
    produces exactly one constructor call even when two coroutines enter
    ``_get_flow()`` concurrently. A regression that removes the lock
    entirely would still be caught because the two coroutines can
    interleave their reads of ``_cached_flow`` if ``_get_flow`` awaits
    any schedulable work between the check and the cache write.

    CR6-7 LOW #5: reviewer suggested strengthening by monkeypatching
    ``__init__`` to include ``await asyncio.sleep(0)``. That is
    syntactically impossible — Python's sync ``__init__`` cannot host
    an ``await`` — and the ``_get_flow`` double-check inside the
    ``async with`` critical section means a no-op lock substitute still
    produces a single ctor call (the second caller, after any yield,
    observes ``_cached_flow is not None`` and returns the cache). The
    only regression shape this pin does NOT catch is one where both
    the lock AND the double-check are removed; that pattern is
    structurally distant from the current implementation and would be
    obvious on review. Kept as a positive pin rather than deleted
    (option b) because the exact-one-ctor-call property is itself a
    load-bearing contract worth keeping green through refactors.
    """
    from ag_ui_crewai import endpoint as ep

    # Fast ceiling so test finishes even if a regression holds one caller.
    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "5")

    construct_calls = 0

    class _SlowCtorFlow:
        def __init__(self, *_a, **_kw):  # noqa: D401
            nonlocal construct_calls
            construct_calls += 1
            # Synchronous constructor — the serializing behaviour the
            # test exercises is the lock around the cache check/write,
            # not a cross-await guard inside __init__ (see docstring).
            self.cancelled = asyncio.Event()
            self.started = asyncio.Event()

        def __deepcopy__(self, memo):
            return self

        async def kickoff_async(self, inputs=None):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    monkeypatch.setattr(ep, "ChatWithCrewFlow", _SlowCtorFlow)

    app = FastAPI()
    ep.add_crewai_crew_fastapi_endpoint(app, _FakeCrew(), path="/run")
    route = next(
        r for r in app.router.routes if getattr(r, "path", None) == "/run"
    )
    endpoint_fn = route.endpoint

    # Fire two concurrent first-requests. Both must see the SAME
    # cached flow; we assert the constructor ran exactly once.
    async def _one_request():
        response = await endpoint_fn(_make_input(), _make_request())
        return response

    # Kick off both responses in parallel; don't fully drain — we only
    # care that the endpoint returned a StreamingResponse each. Cancel
    # the bodies after to free resources.
    responses = await asyncio.gather(
        _one_request(), _one_request(), return_exceptions=True
    )

    assert construct_calls == 1, (
        f"ChatWithCrewFlow constructor should be called exactly once "
        f"under concurrent first-requests; got {construct_calls} calls"
    )

    # Drain/close both responses so teardown is clean.
    for r in responses:
        if hasattr(r, "body_iterator"):
            try:
                await r.body_iterator.aclose()
            except Exception:  # noqa: BLE001
                pass


async def test_cancel_and_join_outer_cancel_bounded_by_monotonic_deadline(monkeypatch):
    """Finding #2 + #7: verify the teardown window stays bounded when the
    outer scope is cancelled mid-teardown. The post-fix implementation
    uses a shared monotonic deadline, so the combined wait of the inner
    shielded ``wait_for`` plus the CancelledError-branch ``wait_for`` must
    not exceed a single ``_CANCEL_JOIN_TIMEOUT_SECONDS`` window (plus a
    small slack for scheduling jitter).
    """
    # Shrink the teardown ceiling so the test is fast and the regression
    # signature (2x → 1x) is clearly distinguishable from scheduling jitter.
    # CR7 MEDIUM: drive this via the public env var rather than stabbing
    # the module constant — that exercises the full parse pipeline in
    # ``_cancel_join_timeout_seconds`` (float(), isfinite, >0 guard)
    # rather than short-circuiting it at the constant-read site.
    monkeypatch.setenv("AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS", "0.5")

    from ag_ui_crewai import endpoint as ep

    stop_absorbing = asyncio.Event()

    async def _absorb_cancel():
        # Absorb cancellations until stop_absorbing fires, so the bounded
        # wait_for MUST time out on its own rather than finishing because
        # the task exited. We check ``stop_absorbing`` between sleeps so
        # the test can release the task during cleanup without warnings.
        while not stop_absorbing.is_set():
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                if stop_absorbing.is_set():
                    return
                continue

    task = asyncio.create_task(_absorb_cancel())
    await asyncio.sleep(0)

    async def _driver():
        await ep._cancel_and_join(
            task,
            thread_id="t-double",
            run_id="r-double",
            allow_grace=False,
        )

    driver = asyncio.create_task(_driver())
    # Let the driver reach the inner ``await asyncio.shield(teardown)``.
    await asyncio.sleep(0.05)

    # Outer cancel: triggers the CancelledError branch inside _cancel_and_join.
    driver.cancel()

    # R5 LOW #17: use ``time.monotonic()`` for parity with
    # ``_cancel_and_join``'s internal deadline math; ``loop.time()`` was
    # an unnecessary divergence and on some platforms has different
    # resolution characteristics than ``time.monotonic()``.
    # Resolve the ceiling through the helper that reads the env var, so
    # the bound matches what ``_cancel_and_join`` actually uses under
    # the monkeypatched env (CR7 MEDIUM).
    ceiling = ep._cancel_join_timeout_seconds()

    start = time.monotonic()
    try:
        # Bound = 1 × ceiling + generous slack. The pre-fix code produced
        # up to 2 × ceiling, so a regression would exceed this bound.
        bound = ceiling + 0.4
        await asyncio.wait_for(driver, timeout=bound + 0.5)
    except asyncio.CancelledError:
        pass  # expected propagation
    except (asyncio.TimeoutError, TimeoutError):  # pragma: no cover - regression
        pytest.fail(
            "_cancel_and_join exceeded even the generous outer wait; the "
            "ceiling regression is severe."
        )
    elapsed = time.monotonic() - start

    # Strict invariant: single-ceiling window.
    assert elapsed <= ceiling + 0.4, (
        f"_cancel_and_join outer-cancel teardown exceeded single-ceiling "
        f"window; elapsed={elapsed:.3f}s, "
        f"ceiling={ceiling}s "
        f"(finding #7 regression)"
    )

    # Clean up the absorb-cancel task cleanly.
    stop_absorbing.set()
    if not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, TimeoutError):
            pass


class _UpstreamTimeoutFlow:
    """A flow whose ``kickoff_async`` raises a bare ``TimeoutError``.

    Models the scenario where LiteLLM (or any other underlying library)
    surfaces a read-timeout as a plain ``TimeoutError`` that bubbles out
    of ``kickoff_async``. The endpoint's timeout handler must be robust
    to this exception even when the flow-ceiling is disabled
    (``timeout=None``) — otherwise the ``timeout:g`` formatting crashes
    with ``TypeError: unsupported format string passed to
    NoneType.__format__`` and the client sees an abruptly terminated
    stream instead of a correlated RunErrorEvent. CR6-4 LOW.
    """

    def __init__(self) -> None:
        self.done = asyncio.Event()

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None):  # noqa: D401
        self.done.set()
        # Bare TimeoutError — caught by the endpoint's
        # ``except (asyncio.TimeoutError, TimeoutError)`` handler.
        raise TimeoutError("simulated upstream read timeout")


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_upstream_timeout_distinct_from_ceiling_timeout_when_ceiling_disabled(
    monkeypatch, factory
):
    """CR7 CRITICAL: upstream ``TimeoutError`` bubbling out of
    ``kickoff_async`` must NOT be classified as a flow-ceiling timeout
    when the ceiling is disabled.

    Prior to the fix, ``except (asyncio.TimeoutError, TimeoutError)``
    emitted ``code=AGUI_CREWAI_FLOW_TIMEOUT`` with "exceeded ceiling=…"
    prose regardless of whether the ceiling fired or upstream raised.
    When the ceiling is disabled (``AGUI_CREWAI_FLOW_TIMEOUT_SECONDS=0``
    → ``timeout=None``), the message "exceeded ceiling=disabled" is
    self-contradictory, and downstream log consumers treating
    ``AGUI_CREWAI_FLOW_TIMEOUT`` as "we hit our configured ceiling" end
    up with a lying signal.

    Fix: ceiling-fired sites raise a sentinel ``_CeilingExceeded``;
    upstream timeouts are caught by a separate handler that emits
    ``AGUI_CREWAI_UPSTREAM_TIMEOUT`` with a message that explicitly
    notes the ceiling did not fire.
    """
    # Disable the flow ceiling — timeout is None in the generator.
    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "0")

    from ag_ui_crewai import endpoint as ep

    flow = _UpstreamTimeoutFlow()
    app = FastAPI()
    _register_with_factory(factory, app, flow, "/run", ep, monkeypatch)

    route = next(
        r for r in app.router.routes if getattr(r, "path", None) == "/run"
    )
    endpoint_fn = route.endpoint
    fake_request = _make_request()

    response = await endpoint_fn(_make_input(), fake_request)
    body_iter = response.body_iterator

    drained: list[bytes] = []

    async def _drain():
        async for chunk in body_iter:
            drained.append(chunk)

    # The stream must complete cleanly — a crash inside the handler would
    # either abort the generator mid-flight (our drain would surface the
    # TypeError here) or leave the stream hanging.
    await asyncio.wait_for(_drain(), timeout=10.0)

    parts = [
        p.decode("utf-8", errors="replace")
        if isinstance(p, (bytes, bytearray))
        else p
        for p in drained
    ]
    joined = "".join(parts)
    payloads = _parse_sse_payloads(joined)
    run_errors = [p for p in payloads if p.get("type") == "RUN_ERROR"]

    assert run_errors, (
        "expected a RUN_ERROR event when upstream TimeoutError bubbles out "
        f"with ceiling disabled; got payloads={payloads!r}"
    )
    err = run_errors[0]
    error_msg = err.get("message", "")

    # Correlation must still appear.
    assert "t-1" in error_msg and "r-1" in error_msg, (
        f"RunErrorEvent must carry thread/run correlation; got: {error_msg!r}"
    )
    assert err.get("threadId") == "t-1", err
    assert err.get("runId") == "r-1", err
    # Stack trace repr must not leak.
    assert "Traceback" not in error_msg, (
        f"RunErrorEvent must not leak a Python traceback; got: {error_msg!r}"
    )
    assert "NoneType" not in error_msg, (
        f"RunErrorEvent must not surface a NoneType-formatting error; "
        f"got: {error_msg!r}"
    )
    # CR7 CRITICAL: the code must distinguish upstream timeouts from
    # ceiling-fired timeouts. This is the load-bearing assertion —
    # pre-fix emitted AGUI_CREWAI_FLOW_TIMEOUT here, which is what
    # downstream alerting treats as "our configured ceiling tripped".
    assert err.get("code") == "AGUI_CREWAI_UPSTREAM_TIMEOUT", (
        f"upstream TimeoutError must surface as "
        f"AGUI_CREWAI_UPSTREAM_TIMEOUT (not AGUI_CREWAI_FLOW_TIMEOUT — "
        f"conflating them lies to downstream alerting); "
        f"got: {err.get('code')!r}"
    )
    # The prose must be compatible with both "ceiling disabled" and
    # "ceiling set but didn't fire" — the shared token is "did not
    # fire" so operators can grep uniformly.
    assert "did not fire" in error_msg, (
        f"upstream-timeout message must indicate the ceiling did not "
        f"fire (as opposed to the ceiling-fired message which uses "
        f"'exceeded ceiling='); got: {error_msg!r}"
    )
    # The ceiling descriptor still appears so operators can see the
    # configured ceiling at failure time.
    assert "disabled" in error_msg, (
        f"when ceiling is disabled, message should mention 'disabled' "
        f"as the ceiling descriptor; got: {error_msg!r}"
    )
    # Must NOT advertise ceiling-exceeded prose (that's the
    # ceiling-fired path).
    assert "exceeded ceiling=" not in error_msg, (
        f"upstream-timeout message must NOT use the ceiling-fired "
        f"'exceeded ceiling=' prose; got: {error_msg!r}"
    )


class _HangingLongerThanCeilingFlow:
    """A flow whose ``kickoff_async`` hangs longer than our ceiling.

    Used to pin the ceiling-fired path (CR7 CRITICAL): our monotonic
    deadline / ``asyncio.wait`` timeout must raise ``_CeilingExceeded``
    (not a bare ``TimeoutError``) so the dedicated handler emits
    ``AGUI_CREWAI_FLOW_TIMEOUT`` with "exceeded ceiling=<value>s".
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None):  # noqa: D401
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.parametrize("factory", ["flow", "crew"])
async def test_ceiling_fired_emits_flow_timeout_code_distinct_from_upstream(
    monkeypatch, factory
):
    """CR7 CRITICAL: the ceiling-fired path must keep emitting
    ``AGUI_CREWAI_FLOW_TIMEOUT`` with "exceeded ceiling=<value>s".

    Complement to ``test_upstream_timeout_distinct_from_ceiling_timeout``
    — together the two tests pin that:
    * ``_CeilingExceeded`` → ``AGUI_CREWAI_FLOW_TIMEOUT`` + "exceeded
      ceiling=<value>s" (this test).
    * bare ``TimeoutError`` from upstream → ``AGUI_CREWAI_UPSTREAM_TIMEOUT``
      + "did not fire" (sibling test).

    A regression that reconflates the two handlers would break exactly
    one of the two assertions — the aggregate red-green ensures the
    split is preserved.
    """
    # Short finite ceiling — ours fires, not upstream.
    monkeypatch.setenv("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS", "0.2")

    from ag_ui_crewai import endpoint as ep

    flow = _HangingLongerThanCeilingFlow()
    app = FastAPI()
    _register_with_factory(factory, app, flow, "/run", ep, monkeypatch)

    route = next(
        r for r in app.router.routes if getattr(r, "path", None) == "/run"
    )
    endpoint_fn = route.endpoint
    fake_request = _make_request()

    response = await endpoint_fn(_make_input(), fake_request)
    body_iter = response.body_iterator

    drained: list[bytes] = []

    async def _drain():
        async for chunk in body_iter:
            drained.append(chunk)

    await asyncio.wait_for(_drain(), timeout=15.0)

    parts = [
        p.decode("utf-8", errors="replace")
        if isinstance(p, (bytes, bytearray))
        else p
        for p in drained
    ]
    joined = "".join(parts)
    payloads = _parse_sse_payloads(joined)
    run_errors = [p for p in payloads if p.get("type") == "RUN_ERROR"]

    assert run_errors, (
        "expected a RUN_ERROR event on the ceiling-fired path; "
        f"got payloads={payloads!r}"
    )
    err = run_errors[0]
    error_msg = err.get("message", "")

    # Code must be FLOW_TIMEOUT (NOT the upstream variant).
    assert err.get("code") == "AGUI_CREWAI_FLOW_TIMEOUT", (
        f"ceiling-fired path should emit AGUI_CREWAI_FLOW_TIMEOUT; "
        f"got: {err.get('code')!r}"
    )
    # Prose must be the ceiling-exceeded variant.
    assert "exceeded ceiling=" in error_msg, (
        f"ceiling-fired path should emit 'exceeded ceiling=...' prose; "
        f"got: {error_msg!r}"
    )
    # And MUST NOT borrow the upstream-variant prose.
    assert "did not fire" not in error_msg, (
        f"ceiling-fired path must NOT use the upstream 'did not fire' "
        f"prose; got: {error_msg!r}"
    )
    # The configured ceiling value must appear.
    assert "0.2" in error_msg, (
        f"ceiling-fired path should include the configured ceiling "
        f"value (0.2); got: {error_msg!r}"
    )
