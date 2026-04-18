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

    We intentionally do NOT silently swallow JSON decode errors: a malformed
    SSE frame is a real bug and the test should fail loudly rather than
    hide it. Callers get a ``pytest.fail`` that pins the offending frame.
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
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            pytest.fail(
                f"Malformed SSE data frame could not be parsed as JSON: "
                f"{payload_text!r} ({exc})"
            )
    return payloads


def _extract_sse_event_names(raw: str) -> list[str]:
    """Return the ``event:`` names appearing in the SSE stream, in order.

    Used by the RUN_ERROR-is-terminal assertion: we want to confirm the
    stream contains no RUN_FINISHED frame alongside RUN_ERROR on the
    timeout path, regardless of payload parsing order.
    """

    names: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event:"):
            names.append(line[len("event:"):].strip())
    return names


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


class _DoubleCancelFlow:
    """Flow that cooperates with the inner CancelledError path under a
    simulated double-cancel (Python 3.11+ ``Task.cancelling`` > 1).

    The outer test cancels the generator twice: once to trigger the grace
    path, and once more to exercise the ``_cancel_and_join`` outer-cancel
    branch. Without an ``uncancel()`` call in the CancelledError handler,
    the bounded second ``await wait_for`` re-raises on entry, making the
    bounded wait a no-op.
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

    # Also verify at the SSE-frame layer that we don't even emit the
    # ``event: RUN_FINISHED`` header. This catches a regression where a
    # listener enqueues RunFinishedEvent *after* the timeout logic has
    # already yielded RunErrorEvent but before the generator's ``finally``
    # tears down the queue.
    frame_names = _extract_sse_event_names(joined)
    assert "RUN_FINISHED" not in frame_names, (
        f"RUN_FINISHED SSE frame must NOT appear on timeout; got={frame_names!r}"
    )


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

    # 5s is dramatically shorter than the 30s env-var ceiling; if the
    # kickoff race is missing or broken, this wait_for raises.
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

    # No RUN_ERROR should fire; happy-path exits cleanly.
    parts = [p.decode("utf-8", errors="replace") if isinstance(p, (bytes, bytearray)) else p
             for p in drained]
    joined = "".join(parts)
    payloads = _parse_sse_payloads(joined)
    types = [p.get("type") for p in payloads]
    assert "RUN_ERROR" not in types, (
        f"happy-path completion without sentinel should NOT emit RUN_ERROR; "
        f"got types={types!r}"
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


async def test_cancel_and_join_outer_cancel_bounded_by_monotonic_deadline(monkeypatch):
    """Finding #2 + #7: verify the teardown window stays bounded when the
    outer scope is cancelled mid-teardown. The post-fix implementation
    uses a shared monotonic deadline, so the combined wait of the inner
    shielded ``wait_for`` plus the CancelledError-branch ``wait_for`` must
    not exceed a single ``_CANCEL_JOIN_TIMEOUT_SECONDS`` window (plus a
    small slack for scheduling jitter).
    """
    from ag_ui_crewai import endpoint as ep

    # Shrink the teardown ceiling so the test is fast and the regression
    # signature (2x → 1x) is clearly distinguishable from scheduling jitter.
    monkeypatch.setattr(ep, "_CANCEL_JOIN_TIMEOUT_SECONDS", 0.5)

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

    loop = asyncio.get_event_loop()
    start = loop.time()
    try:
        # Bound = 1 × ceiling + generous slack. The pre-fix code produced
        # up to 2 × ceiling, so a regression would exceed this bound.
        bound = ep._CANCEL_JOIN_TIMEOUT_SECONDS + 0.4
        await asyncio.wait_for(driver, timeout=bound + 0.5)
    except asyncio.CancelledError:
        pass  # expected propagation
    except (asyncio.TimeoutError, TimeoutError):  # pragma: no cover - regression
        pytest.fail(
            "_cancel_and_join exceeded even the generous outer wait; the "
            "ceiling regression is severe."
        )
    elapsed = loop.time() - start

    # Strict invariant: single-ceiling window.
    assert elapsed <= ep._CANCEL_JOIN_TIMEOUT_SECONDS + 0.4, (
        f"_cancel_and_join outer-cancel teardown exceeded single-ceiling "
        f"window; elapsed={elapsed:.3f}s, "
        f"ceiling={ep._CANCEL_JOIN_TIMEOUT_SECONDS}s "
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
