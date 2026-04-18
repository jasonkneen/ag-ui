"""
AG-UI FastAPI server for CrewAI.
"""
import copy
import asyncio
import logging
import math
import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from crewai.utilities.events import (
    FlowStartedEvent,
    FlowFinishedEvent,
    MethodExecutionStartedEvent,
    MethodExecutionFinishedEvent,
)
from crewai.flow.flow import Flow
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai import Crew

from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    Message,
    Tool
)
from ag_ui.core.events import (
  TextMessageChunkEvent,
  ToolCallChunkEvent,
  StepStartedEvent,
  StepFinishedEvent,
  MessagesSnapshotEvent,
  StateSnapshotEvent,
  CustomEvent,
)
from ag_ui.encoder import EventEncoder

from .events import (
  BridgedTextMessageChunkEvent,
  BridgedToolCallChunkEvent,
  BridgedCustomEvent,
  BridgedStateSnapshotEvent
)
from .context import flow_context
from .sdk import litellm_messages_to_ag_ui_messages
from .crews import ChatWithCrewFlow

_LOGGER = logging.getLogger(__name__)

# Sentinel to distinguish "no item delivered" from a legitimate ``None``
# queue payload (the happy-path stream-end sentinel). Used by the
# cancel-race guard in ``_run_flow_event_stream`` (finding #1 HIGH H1)
# where an item may have been delivered to ``get_task`` between
# ``asyncio.wait`` returning and the ``finally`` clause cancelling it.
_UNSET = object()

QUEUES = {}
QUEUES_LOCK = asyncio.Lock()

# Hard wall-clock ceiling on a single flow run. A runaway flow (e.g. a hung
# LiteLLM stream or an infinite loop in a user task) must not be able to pin
# the process indefinitely. Override via the ``AGUI_CREWAI_FLOW_TIMEOUT_SECONDS``
# environment variable; defaults to 10 minutes. Deployments with legitimately
# long-running crews should set the env var explicitly or use a non-positive
# value to disable the ceiling.
_DEFAULT_FLOW_TIMEOUT_SECONDS = 600.0

# When we see a FlowFinishedEvent the listener puts ``None`` on the queue
# *before* kickoff_async has actually returned. Give the task a short grace
# period to complete cleanly before we force-cancel it in _cancel_and_join.
# This grace window is drawn from the SHARED ``_cancel_join_timeout_seconds``
# teardown budget (finding #7): total upper bound on teardown from entry to
# ``_cancel_and_join`` is one ceiling window, not ``grace + join``.
_CANCEL_GRACE_SECONDS = 1.0

# If a cancelled task refuses to terminate within this window, log a warning
# so operators have visibility into stuck cancellations instead of a silent
# swallow. Default override-able via ``AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS``
# so operators can tune it under disconnect-heavy load (finding #8).
_CANCEL_JOIN_TIMEOUT_SECONDS = 10.0


def _flow_timeout_seconds() -> float | None:
    """Return the configured flow-execution ceiling in seconds.

    A non-positive value (e.g. ``0`` or ``-1``) disables the ceiling. A
    NaN or any other non-finite value is treated as unparseable and falls
    back to the default — ``float('nan') > 0`` is False, which would
    otherwise silently disable the ceiling (finding #17).
    """
    raw = os.environ.get("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS")
    if raw is None:
        return _DEFAULT_FLOW_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_FLOW_TIMEOUT_SECONDS
    if not math.isfinite(value):
        return _DEFAULT_FLOW_TIMEOUT_SECONDS
    if value <= 0:
        return None
    return value


def _cancel_join_timeout_seconds() -> float:
    """Return the configured cancel-and-join teardown ceiling in seconds.

    Exists so that operators running disconnect-heavy workloads can tune
    the per-request teardown window via
    ``AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS`` without redeploying code
    (finding #8). Non-finite or non-positive values fall back to the
    conservative default so a fat-fingered env var cannot disable the
    ceiling entirely.
    """
    raw = os.environ.get("AGUI_CREWAI_CANCEL_JOIN_TIMEOUT_SECONDS")
    if raw is None:
        return _CANCEL_JOIN_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _CANCEL_JOIN_TIMEOUT_SECONDS
    if not math.isfinite(value) or value <= 0:
        return _CANCEL_JOIN_TIMEOUT_SECONDS
    return value


async def _cancel_and_join(
    task: asyncio.Task | None,
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    allow_grace: bool = True,
) -> None:
    """Cancel ``task`` and await its completion, letting CancelledError propagate.

    Used in the ``finally`` block of the event generators so that a client
    disconnect (which closes the generator) tears down the kickoff coroutine
    instead of leaking it.

    Semantics:
    - A SINGLE shared monotonic deadline (``_cancel_join_timeout_seconds``)
      bounds the combined wait of (grace window + force-cancel join +
      outer-cancel recovery). There is one ceiling window for the entire
      teardown, not three (finding #7).
    - If ``allow_grace`` and the task is mid-flight on a happy path, wait up
      to ``min(_CANCEL_GRACE_SECONDS, remaining-budget)`` for it to finish
      on its own (the FlowFinishedEvent listener enqueues ``None``
      microseconds before ``kickoff_async`` actually returns). A quick
      ``sleep(0)`` + ``task.done()`` probe fast-paths the common case where
      the task is microseconds from returning, so happy-path requests do
      NOT systematically pay the 1s grace latency tax (finding #9).
    - The grace wait is SHIELDED and protected by the same outer-cancel
      recovery pattern used post-grace (finding #5). If the caller is
      cancelled during the grace wait, ``task.cancel()`` still fires via
      the ``finally`` and the task is cleanly unwound within the remaining
      budget; we don't leave a cancelled-but-unjoined task behind.
    - We deliberately do NOT catch ``BaseException``. ``SystemExit`` /
      ``KeyboardInterrupt`` / ``CancelledError`` must propagate; we only
      swallow ``TimeoutError`` (explicitly) and recoverable ``Exception``
      subclasses from the task itself.
    - On Python 3.11+, catching ``CancelledError`` does NOT decrement
      ``Task.cancelling()``: any subsequent ``await`` re-raises immediately
      unless we call ``asyncio.current_task().uncancel()``. Without that,
      the bounded recovery wait in the CancelledError branch is defeated
      (re-raises on entry). We invoke ``uncancel`` via ``getattr`` so the
      implementation remains compatible with 3.10 (where the method does
      not exist).
    """
    if task is None or task.done():
        return

    # Shared monotonic deadline covering the ENTIRE teardown — grace
    # window, force-cancel join, and outer-cancel recovery (finding #7).
    ceiling = _cancel_join_timeout_seconds()
    deadline = time.monotonic() + ceiling

    def _remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    cancellation_scheduled = False
    try:
        if allow_grace:
            # Fast-path probe (finding #9): let the task advance a tick
            # before paying the 1s grace wait. The common case is that
            # ``kickoff_async`` is microseconds from returning once the
            # listener has enqueued the ``None`` sentinel; yielding once
            # usually lets the task complete without blocking.
            await asyncio.sleep(0)
            if task.done():
                return

            # Grace period for happy-path completion. ``shield`` keeps the
            # task alive if our wait_for is itself cancelled. Note (3.10
            # compatibility): ``asyncio.TimeoutError`` is aliased to the
            # builtin ``TimeoutError`` on 3.11+, but the dual tuple is
            # load-bearing on 3.10 where they are distinct classes.
            grace_budget = min(_CANCEL_GRACE_SECONDS, _remaining())
            if grace_budget > 0:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=grace_budget
                    )
                    return
                except (asyncio.TimeoutError, TimeoutError):
                    # Happy path did not complete in time; fall through to
                    # force-cancel below.
                    pass
                except asyncio.CancelledError:
                    # Outer-cancel during the grace wait. Mirror the
                    # post-grace recovery pattern (finding #5): ensure
                    # task.cancel() fires within the remaining budget and
                    # await its unwind so we do not leave a
                    # cancelled-but-unjoined task behind.
                    current = asyncio.current_task()
                    uncancel = getattr(current, "uncancel", None)
                    if callable(uncancel):
                        uncancel()
                    if not task.done():
                        task.cancel()
                        cancellation_scheduled = True
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(
                                asyncio.gather(task, return_exceptions=True)
                            ),
                            timeout=_remaining(),
                        )
                    except (asyncio.TimeoutError, TimeoutError):
                        _log_stuck_cancel(
                            thread_id, run_id, after_outer_cancel=True
                        )
                    except asyncio.CancelledError:
                        # Recovery wait itself cancelled; there is nothing
                        # further we can do within budget — propagate.
                        raise
                    # Re-raise the original outer cancel so semantics
                    # propagate to the caller.
                    raise asyncio.CancelledError
                except Exception as grace_exc:  # pylint: disable=broad-exception-caught
                    # The task itself raised during the grace wait. It has
                    # finished — nothing left to clean up. Log the
                    # exception rather than silently swallowing it so that
                    # operators can diagnose teardown surprises.
                    if task.done():
                        return
                    # Unusual ordering: log loudly and fall through to
                    # force-cancel.
                    _LOGGER.warning(
                        "CrewAI grace-period wait raised a non-Timeout error "
                        "while task is not done; proceeding to force-cancel "
                        "thread=%s run=%s cause=%s",
                        thread_id,
                        run_id,
                        type(grace_exc).__name__,
                    )

        if task.done():
            return

        # Force-cancel from here on out; the finally clause guarantees
        # task.cancel() runs exactly once even if we are cancelled mid-flight.
        task.cancel()
        cancellation_scheduled = True

        # Build a teardown coroutine and shield it so outer cancellation
        # cannot abandon the task mid-teardown. We want resources (httpx
        # clients, file descriptors, LLM subscriptions) to actually unwind.
        teardown = asyncio.ensure_future(
            asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True),
                timeout=_remaining(),
            )
        )
        try:
            await asyncio.shield(teardown)
        except asyncio.CancelledError as outer_cancel:
            # Outer scope was cancelled. On Python 3.11+, we must uncancel
            # the current task before issuing another ``await`` — otherwise
            # the next ``await`` re-raises CancelledError immediately and
            # the bounded recovery wait is a no-op (finding #2).
            current = asyncio.current_task()
            uncancel = getattr(current, "uncancel", None)
            if callable(uncancel):
                uncancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(teardown),
                    timeout=_remaining(),
                )
            except (asyncio.TimeoutError, TimeoutError):
                _log_stuck_cancel(thread_id, run_id, after_outer_cancel=True)
            except Exception as recov_exc:  # pylint: disable=broad-exception-caught
                # A non-timeout, non-cancel error surfaced from the
                # recovery wait; surface it in DEBUG logs rather than
                # swallowing silently (finding #10).
                _LOGGER.debug(
                    "CrewAI cancel-recovery wait swallowed %s "
                    "(thread=%s run=%s)",
                    type(recov_exc).__name__,
                    thread_id,
                    run_id,
                )
            # Retrieve any exception on ``teardown`` so it does not surface
            # as ``Task exception was never retrieved`` during GC. If the
            # task is still pending (recovery wait_for timed out), detach
            # a callback that drains its eventual result — we've already
            # spent our full ceiling budget and must not block further.
            def _drain(fut):
                if fut.cancelled():
                    return
                # ``.exception()`` marks the exception retrieved.
                exc = fut.exception()
                if exc is not None:
                    _LOGGER.debug(
                        "CrewAI kickoff teardown completed post-cancel "
                        "with %s (thread=%s run=%s)",
                        type(exc).__name__,
                        thread_id,
                        run_id,
                    )

            if teardown.done():
                _drain(teardown)
            else:
                teardown.add_done_callback(_drain)
            # Re-raise the original CancelledError so traceback and
            # ``.args`` context propagate intact to the outer scope
            # (finding #14). A bare ``raise`` would reference ``outer_cancel``
            # via the active handler; using the captured name is explicit.
            raise outer_cancel
        except (asyncio.TimeoutError, TimeoutError):
            _log_stuck_cancel(thread_id, run_id, after_outer_cancel=False)
    finally:
        # Last-ditch: if we scheduled cancellation and the task still isn't
        # done (e.g. we were cancelled before reaching task.cancel()), ensure
        # we don't leak a running kickoff_async.
        if task is not None and not task.done() and not cancellation_scheduled:
            task.cancel()


def _log_stuck_cancel(
    thread_id: str | None, run_id: str | None, *, after_outer_cancel: bool
) -> None:
    """Emit a single consolidated warning when a cancelled task won't terminate.

    Centralised so the message format, fields, and distinguishing context are
    identical at both call sites.
    """
    suffix = " (after outer cancel)" if after_outer_cancel else ""
    _LOGGER.warning(
        "CrewAI kickoff task did not terminate within %.1fs of cancel%s"
        " thread=%s run=%s",
        _cancel_join_timeout_seconds(),
        suffix,
        thread_id,
        run_id,
    )


async def create_queue(flow: object) -> asyncio.Queue:
    """Create a queue for a flow."""
    queue_id = id(flow)
    async with QUEUES_LOCK:
        queue = asyncio.Queue()
        QUEUES[queue_id] = queue
        return queue


def get_queue(flow: object) -> asyncio.Queue | None:
    """Get the queue for a flow."""
    queue_id = id(flow)
    # not using a lock here should be fine
    return QUEUES.get(queue_id)

async def delete_queue(flow: object) -> None:
    """Delete the queue for a flow."""
    queue_id = id(flow)
    async with QUEUES_LOCK:
        if queue_id in QUEUES:
            del QUEUES[queue_id]

GLOBAL_EVENT_LISTENER = None

class FastAPICrewFlowEventListener(BaseEventListener):
    """FastAPI CrewFlow event listener"""

    def setup_listeners(self, crewai_event_bus):
        """Setup listeners for the FastAPI CrewFlow event listener"""
        @crewai_event_bus.on(FlowStartedEvent)
        def _(source, event):  # pylint: disable=unused-argument
            queue = get_queue(source)
            if queue is not None:
                queue.put_nowait(
                    RunStartedEvent(
                        type=EventType.RUN_STARTED,
                         # will be replaced by the correct thread_id/run_id when sending the event
                        thread_id="?",
                        run_id="?",
                    ),
                )
        @crewai_event_bus.on(FlowFinishedEvent)
        def _(source, event):  # pylint: disable=unused-argument
            queue = get_queue(source)
            if queue is not None:
                queue.put_nowait(
                    RunFinishedEvent(
                        type=EventType.RUN_FINISHED,
                        thread_id="?",
                        run_id="?",
                    ),
                )
                queue.put_nowait(None)
        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def _(source, event):
            queue = get_queue(source)
            if queue is not None:
                queue.put_nowait(
                    StepStartedEvent(
                        type=EventType.STEP_STARTED,
                        step_name=event.method_name
                    )
                )
        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def _(source, event):
            queue = get_queue(source)
            if queue is not None:
                # source.state may be a Pydantic model (with .messages attr) or a plain dict
                state = source.state
                raw_messages = getattr(state, "messages", None) or (state.get("messages") if isinstance(state, dict) else None) or []
                messages = litellm_messages_to_ag_ui_messages(raw_messages)

                queue.put_nowait(
                    MessagesSnapshotEvent(
                        type=EventType.MESSAGES_SNAPSHOT,
                        messages=messages
                    )
                )
                queue.put_nowait(
                    StateSnapshotEvent(
                        type=EventType.STATE_SNAPSHOT,
                        snapshot=state if isinstance(state, dict) else state.model_dump() if hasattr(state, "model_dump") else {}
                    )
                )
                queue.put_nowait(
                    StepFinishedEvent(
                        type=EventType.STEP_FINISHED,
                        step_name=event.method_name
                    )
                )
        @crewai_event_bus.on(BridgedTextMessageChunkEvent)
        def _(source, event):
            queue = get_queue(source)
            if queue is not None:
                queue.put_nowait(
                    TextMessageChunkEvent(
                        type=EventType.TEXT_MESSAGE_CHUNK,
                        message_id=event.message_id,
                        role=event.role,
                        delta=event.delta,
                    )
                )
        @crewai_event_bus.on(BridgedToolCallChunkEvent)
        def _(source, event):
            queue = get_queue(source)
            if queue is not None:
                queue.put_nowait(
                    ToolCallChunkEvent(
                        type=EventType.TOOL_CALL_CHUNK,
                        tool_call_id=event.tool_call_id,
                        tool_call_name=event.tool_call_name,
                        delta=event.delta,
                    )
                )
        @crewai_event_bus.on(BridgedCustomEvent)
        def _(source, event):
            queue = get_queue(source)
            if queue is not None:
                queue.put_nowait(
                    CustomEvent(
                        type=EventType.CUSTOM,
                        name=event.name,
                        value=event.value
                    )
                )
        @crewai_event_bus.on(BridgedStateSnapshotEvent)
        def _(source, event):
            queue = get_queue(source)
            if queue is not None:
                queue.put_nowait(
                    StateSnapshotEvent(
                        type=EventType.STATE_SNAPSHOT,
                        snapshot=event.snapshot
                    )
                )


def _format_timeout_message(timeout: float | None) -> str:
    """Build the ``TimeoutError`` message for the flow-ceiling path.

    Extracted (finding #13) so the two TimeoutError construction sites and
    the client-facing error message derive from a single source of truth.

    ``timeout`` is always a finite positive value here — the flow-ceiling
    code paths that raise ``TimeoutError`` are guarded by ``timeout is not
    None``. Using ``%g`` (up to 6 significant digits, no trailing zeros)
    avoids the truncation of sub-decisecond values that ``%.1f`` produces
    (finding #14). For ``0.2``, ``%g`` renders ``0.2``; for ``0.25``,
    ``0.25``; for ``600``, ``600``.
    """
    return f"CrewAI flow exceeded {timeout:g}s ceiling"


def _field_alias(model_cls, field_name: str, default: str) -> str:
    """Return the serialization alias for ``field_name`` on ``model_cls``.

    Pydantic models in ag-ui.core set camelCase aliases via an alias
    generator; we derive the wire name here so a future rename of the
    alias policy propagates automatically (finding #30) instead of
    silently diverging from this module's hardcoded camelCase literals.
    Falls back to ``default`` if the model does not declare the field
    (keeps the code path stable under library upgrades).
    """
    try:
        field = model_cls.model_fields[field_name]
    except (AttributeError, KeyError):
        return default
    # Pydantic v2 exposes the alias either as ``alias`` (explicit) or via
    # ``serialization_alias``; prefer the latter if set.
    alias = (
        getattr(field, "serialization_alias", None)
        or getattr(field, "alias", None)
    )
    return alias or default


def _run_error_extras(input_data: RunAgentInput) -> dict:
    """Return the extras kwargs for a RunErrorEvent, camelCased to match
    peer events' wire format.

    ``ConfiguredBaseModel`` uses ``extra="allow"`` — extras bypass the
    alias generator, so pre-camelCased keys are required to line up with
    declared-field peers (``RunStartedEvent.thread_id`` / ``run_id`` emit
    as ``threadId`` / ``runId`` via the alias generator). Finding #3.

    The alias names are derived from ``RunStartedEvent.model_fields``
    (finding #30) so a rename of the alias policy in ag-ui.core does not
    silently regress this module.
    """
    thread_alias = _field_alias(RunStartedEvent, "thread_id", "threadId")
    run_alias = _field_alias(RunStartedEvent, "run_id", "runId")
    return {
        thread_alias: input_data.thread_id,
        run_alias: input_data.run_id,
    }


async def _run_flow_event_stream(
    *,
    flow_copy: object,
    encoder: EventEncoder,
    input_data: RunAgentInput,
    inputs: dict,
    timeout: float | None,
):
    """Drive a single flow kickoff and yield encoded AG-UI events.

    Extracted from the flow and crew endpoints so they share identical
    cancellation, timeout, and error-reporting semantics. The generator:

    * spawns ``kickoff_async`` as a task (kept in scope so it can be torn
      down on client disconnect);
    * reads from the per-flow queue with a wall-clock deadline;
    * surfaces timeouts and other exceptions as a ``RunErrorEvent`` whose
      ``message`` carries thread/run correlation AND whose event-level
      extras (``threadId`` / ``runId``) mirror the peer events' wire
      format (finding #3);
    * on exit, cancels the kickoff task, drops the queue, and resets the
      context var — unconditionally, even if the outer scope is cancelled.
    """
    queue = await create_queue(flow_copy)
    token = flow_context.set(flow_copy)
    # Hold a reference to the kickoff task so we can cancel it on
    # client disconnect. Without this reference the task can outlive
    # the request (orphaned), continuing to drive LiteLLM / tools
    # after nobody is listening.
    kickoff_task: asyncio.Task | None = None
    # ``allow_grace`` controls whether _cancel_and_join waits up to
    # _CANCEL_GRACE_SECONDS for a happy-path completion. Only the normal
    # ``None`` sentinel exit sets this to True; disconnect / timeout /
    # exception paths force an immediate cancel to keep teardown snappy.
    allow_grace = False
    try:
        try:
            kickoff_task = asyncio.create_task(
                flow_copy.kickoff_async(inputs=inputs)  # type: ignore[attr-defined]
            )

            deadline = (
                time.monotonic() + timeout
                if timeout is not None
                else None
            )

            async def _drain_queue_until_sentinel_or_empty():
                """Drain remaining queued items into the output stream.

                Returns ``True`` if the ``None`` sentinel was observed.
                After a first pass that returns ``QueueEmpty`` we yield
                the event loop once and probe again — this catches
                late-arriving listener ``put_nowait`` callbacks that
                fired microseconds after ``kickoff_task`` completed
                (finding #3: a listener enqueueing right after
                ``kickoff_task.done()`` would otherwise be silently
                dropped).
                """
                drained_sentinel_local = False
                for _pass in range(2):
                    drained_anything = False
                    while True:
                        try:
                            item_local = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        drained_anything = True
                        if item_local is None:
                            drained_sentinel_local = True
                            break
                        if item_local.type in (
                            EventType.RUN_STARTED,
                            EventType.RUN_FINISHED,
                        ):
                            item_local.thread_id = input_data.thread_id
                            item_local.run_id = input_data.run_id
                        yield encoder.encode(item_local)
                    if drained_sentinel_local:
                        return
                    # Yield once so any listener callback enqueued via
                    # ``call_soon`` has a chance to run before we conclude
                    # the queue is empty.
                    await asyncio.sleep(0)
                    if not drained_anything and queue.empty():
                        # Two passes found nothing new — we are consistent.
                        return
                # (unreachable; two passes is enough)

            while True:
                # Surface kickoff exceptions promptly. Without this race, a
                # crash inside ``kickoff_async`` (auth failure, library
                # assertion) would leave the main loop blocked on
                # ``queue.get()`` until the flow-timeout ceiling, and users
                # would see ``AGUI_CREWAI_FLOW_TIMEOUT`` instead of the real
                # traceback. We use ``await kickoff_task`` (rather than
                # ``raise kickoff_task.exception()``) so the original
                # traceback is preserved — finding #4: re-raising the
                # stored exception via ``raise exc`` starts a new
                # traceback chain whose innermost frame is this ``raise``
                # line, hiding the real origin.
                if kickoff_task.done():
                    # Guard against ``.exception()`` raising
                    # CancelledError if the task was cancelled externally
                    # (finding #2): only read ``.exception()`` on a
                    # non-cancelled task.
                    if not kickoff_task.cancelled():
                        kickoff_exc = kickoff_task.exception()
                        if kickoff_exc is not None:
                            # ``await`` re-raises the stored exception
                            # WITH its original traceback intact.
                            await kickoff_task
                    # Happy path: task finished without error. Drain any
                    # remaining queue items (for example the ``None``
                    # sentinel enqueued by the FlowFinishedEvent listener),
                    # then break. Critically we do NOT fall through to
                    # ``asyncio.wait({get_task, kickoff_task}, ...)``
                    # below, because that wait would return immediately
                    # (kickoff_task is already done) and cause a CPU spin
                    # (finding #1).
                    async for encoded in _drain_queue_until_sentinel_or_empty():
                        yield encoded
                    # ``allow_grace`` only matters while the task is in
                    # flight (`_cancel_and_join` short-circuits if the
                    # task is already done). We leave the default False
                    # here rather than setting True on the inline-sentinel
                    # branch — the value is dead either way (finding
                    # #15), and an explicit False is less misleading.
                    break

                get_task = asyncio.ensure_future(queue.get())
                item: object = _UNSET  # sentinel: not yet populated
                try:
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(_format_timeout_message(timeout))
                        done, _pending = await asyncio.wait(
                            {get_task, kickoff_task},
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=remaining,
                        )
                    else:
                        done, _pending = await asyncio.wait(
                            {get_task, kickoff_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                    if not done:
                        raise TimeoutError(_format_timeout_message(timeout))

                    # Prefer propagating the kickoff exception (if any)
                    # over consuming a queued event — the exception is
                    # the real story. Bind the result once (finding #16)
                    # and guard against CancelledError (finding #2).
                    if kickoff_task in done and not kickoff_task.cancelled():
                        kickoff_exc = kickoff_task.exception()
                        if kickoff_exc is not None:
                            await kickoff_task

                    if get_task in done:
                        item = get_task.result()
                    else:
                        # kickoff finished without error but no item was
                        # enqueued yet; the top-of-loop guard on the next
                        # iteration will observe ``kickoff_task.done()``
                        # and drain via the fast path above (no spin —
                        # finding #1).
                        pass
                finally:
                    # Cancel-race guard (finding #1 HIGH H1): between
                    # ``asyncio.wait`` returning and us cancelling
                    # ``get_task``, the queue may have delivered an item
                    # to the getter. If we blindly cancel, that item is
                    # dropped. Check ``get_task.done()`` first and, if so,
                    # harvest the result (even when the primary branch
                    # above did not because ``get_task`` was not in
                    # ``done`` — e.g. it completed between ``asyncio.wait``
                    # returning and this ``finally``).
                    if not get_task.done():
                        get_task.cancel()
                    elif item is _UNSET and not get_task.cancelled():
                        try:
                            pending_item = get_task.result()
                        except BaseException:  # noqa: BLE001
                            pending_item = _UNSET
                        if pending_item is not _UNSET:
                            item = pending_item

                if item is _UNSET:
                    # No item to yield — either kickoff exited without
                    # enqueueing, or only kickoff was in ``done`` and
                    # ``get_task`` was cleanly cancelled. Loop back to the
                    # top to hit the ``kickoff_task.done()`` fast path.
                    continue

                if item is None:
                    # Happy-path sentinel: grant the kickoff task a short
                    # grace period so a task that is microseconds from
                    # returning does not get needlessly cancelled.
                    allow_grace = True
                    break

                if item.type in (EventType.RUN_STARTED, EventType.RUN_FINISHED):
                    item.thread_id = input_data.thread_id
                    item.run_id = input_data.run_id

                yield encoder.encode(item)

        except (asyncio.TimeoutError, TimeoutError):
            # Log full context server-side; keep the client message tight
            # and correlated. Extras expose ``threadId`` / ``runId`` in
            # camelCase to match peer events' wire format (finding #3).
            # ``timeout`` is always non-None here — the ``deadline``
            # computation above only produces TimeoutError when ``timeout``
            # was configured (the unreachable ``"configured"`` fallback
            # branch has been removed, finding #11).
            _LOGGER.warning(
                "CrewAI flow exceeded ceiling thread=%s run=%s ceiling=%gs",
                input_data.thread_id,
                input_data.run_id,
                timeout,
            )
            message = (
                f"thread={input_data.thread_id} run={input_data.run_id}: "
                f"CrewAI flow exceeded {timeout:g}s ceiling"
            )
            yield encoder.encode(
                RunErrorEvent(
                    message=message,
                    code="AGUI_CREWAI_FLOW_TIMEOUT",
                    **_run_error_extras(input_data),
                )
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Log full traceback server-side; send a coarse, correlated
            # message to the client (do not leak internal repr of e).
            _LOGGER.exception(
                "CrewAI flow failed thread=%s run=%s cause=%s",
                input_data.thread_id,
                input_data.run_id,
                type(e).__name__,
            )
            # Tight message (finding #5): the exception class name already
            # lives in ``code`` (AGUI_CREWAI_FLOW_ERROR:<Class>); the
            # run_id already appears once as a prefix — do not duplicate.
            message = (
                f"thread={input_data.thread_id} run={input_data.run_id}: "
                f"CrewAI flow failed; see server logs"
            )
            yield encoder.encode(
                RunErrorEvent(
                    message=message,
                    code=f"AGUI_CREWAI_FLOW_ERROR:{type(e).__name__}",
                    **_run_error_extras(input_data),
                )
            )
    finally:
        # Teardown must run unconditionally — including when the outer
        # scope has been cancelled. Nested try/finally ensures that even if
        # _cancel_and_join raises CancelledError, we still drop the queue
        # and reset the context var.
        try:
            await _cancel_and_join(
                kickoff_task,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
                allow_grace=allow_grace,
            )
        finally:
            try:
                await delete_queue(flow_copy)
            finally:
                flow_context.reset(token)


def add_crewai_flow_fastapi_endpoint(app: FastAPI, flow: Flow, path: str = "/"):
    """Adds a CrewAI endpoint to the FastAPI app."""
    global GLOBAL_EVENT_LISTENER # pylint: disable=global-statement

    # Set up the global event listener singleton
    # we are doing this here because calling add_crewai_flow_fastapi_endpoint is a clear indicator
    # that we are not running on CrewAI enterprise
    if GLOBAL_EVENT_LISTENER is None:
        GLOBAL_EVENT_LISTENER = FastAPICrewFlowEventListener()

    @app.post(path)
    async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
        """Agentic chat endpoint"""

        flow_copy = copy.deepcopy(flow)

        # Get the accept header from the request
        accept_header = request.headers.get("accept")

        # Create an event encoder to properly format SSE events
        encoder = EventEncoder(accept=accept_header)

        inputs = crewai_prepare_inputs(
            state=input_data.state,
            messages=input_data.messages,
            tools=input_data.tools,
        )
        inputs["id"] = input_data.thread_id

        timeout = _flow_timeout_seconds()

        return StreamingResponse(
            _run_flow_event_stream(
                flow_copy=flow_copy,
                encoder=encoder,
                input_data=input_data,
                inputs=inputs,
                timeout=timeout,
            ),
            media_type=encoder.get_content_type(),
        )


def add_crewai_crew_fastapi_endpoint(app: FastAPI, crew: Crew, path: str = "/"):
    """Adds a CrewAI crew endpoint to the FastAPI app.

    ChatWithCrewFlow construction is deferred to first request because the
    constructor calls crew_chat_generate_crew_chat_inputs which makes an LLM
    call. At import time the LLM mock server may not be running yet.
    """
    global GLOBAL_EVENT_LISTENER # pylint: disable=global-statement
    if GLOBAL_EVENT_LISTENER is None:
        GLOBAL_EVENT_LISTENER = FastAPICrewFlowEventListener()

    _cached_flow = None
    # Dedicated per-endpoint lock so two concurrent first-requests cannot
    # both call ``ChatWithCrewFlow(crew=crew)`` — which issues a real LLM
    # call — and waste API budget / memory (finding #6). Not sharing
    # QUEUES_LOCK: the flow-construction critical section is independent
    # of queue lifecycle and should not serialise per-request queue
    # teardown.
    _flow_lock = asyncio.Lock()

    async def _get_flow():
        nonlocal _cached_flow
        if _cached_flow is not None:
            return _cached_flow
        async with _flow_lock:
            if _cached_flow is None:
                _cached_flow = ChatWithCrewFlow(crew=crew)
            return _cached_flow

    @app.post(path)
    async def crew_endpoint(input_data: RunAgentInput, request: Request):
        """Crew chat endpoint with deferred initialization."""
        flow = await _get_flow()
        flow_copy = copy.deepcopy(flow)

        accept_header = request.headers.get("accept")
        encoder = EventEncoder(accept=accept_header)

        inputs = crewai_prepare_inputs(
            state=input_data.state,
            messages=input_data.messages,
            tools=input_data.tools,
        )
        inputs["id"] = input_data.thread_id

        timeout = _flow_timeout_seconds()

        return StreamingResponse(
            _run_flow_event_stream(
                flow_copy=flow_copy,
                encoder=encoder,
                input_data=input_data,
                inputs=inputs,
                timeout=timeout,
            ),
            media_type=encoder.get_content_type(),
        )


def crewai_prepare_inputs(  # pylint: disable=unused-argument, too-many-arguments
    *,
    state: dict,
    messages: list[Message],
    tools: list[Tool],
):
    """Default merge state for CrewAI"""
    messages = [message.model_dump() for message in messages]

    if len(messages) > 0:
        if "role" in messages[0] and messages[0]["role"] == "system":
            messages = messages[1:]

    actions = [{
        "type": "function",
        "function": {
            **tool.model_dump(),
        }
    } for tool in tools]

    new_state = {
        **state,
        "messages": messages,
        "copilotkit": {
            "actions": actions
        }
    }

    return new_state
