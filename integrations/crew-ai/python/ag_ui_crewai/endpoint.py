"""
AG-UI FastAPI server for CrewAI.
"""
import copy
import asyncio
import logging
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
_CANCEL_GRACE_SECONDS = 1.0

# If a cancelled task refuses to terminate within this window, log a warning
# so operators have visibility into stuck cancellations instead of a silent
# swallow.
_CANCEL_JOIN_TIMEOUT_SECONDS = 10.0


def _flow_timeout_seconds() -> float | None:
    """Return the configured flow-execution ceiling in seconds.

    A non-positive value (e.g. ``0`` or ``-1``) disables the ceiling. Any
    unparseable value falls back to the default.
    """
    raw = os.environ.get("AGUI_CREWAI_FLOW_TIMEOUT_SECONDS")
    if raw is None:
        return _DEFAULT_FLOW_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_FLOW_TIMEOUT_SECONDS
    if value <= 0:
        return None
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
    - If ``allow_grace`` and the task is mid-flight on a happy path, wait up to
      ``_CANCEL_GRACE_SECONDS`` for it to finish on its own (the
      FlowFinishedEvent listener enqueues ``None`` microseconds before
      ``kickoff_async`` actually returns).
    - If the caller is cancelled during that grace-period wait, the shielded
      inner task is left intact; the caller's ``CancelledError`` propagates
      (we intentionally do NOT swallow it) and a ``finally`` guarantees
      ``task.cancel()`` still fires so the task is not leaked.
    - After grace (or if grace is disabled), cancel the task and wait for it
      to finish unwinding. We use a detached teardown task wrapped in
      ``asyncio.shield`` so outer cancellation does not abandon it; if the
      outer scope is cancelled we still log a warning if the task is stuck.
    - We deliberately do NOT catch ``BaseException``. ``SystemExit`` /
      ``KeyboardInterrupt`` / ``CancelledError`` must propagate; we only
      swallow ``TimeoutError`` (explicitly) and recoverable ``Exception``
      subclasses from the task itself.
    - On Python 3.11+, catching ``CancelledError`` does NOT decrement
      ``Task.cancelling()``: any subsequent ``await`` re-raises immediately
      unless we call ``asyncio.current_task().uncancel()``. Without that,
      the bounded second wait in the CancelledError branch is defeated
      (re-raises on entry). We invoke ``uncancel`` via ``getattr`` so the
      implementation remains compatible with 3.10 (where the method does
      not exist). The total teardown window is capped at a SHARED
      ``_CANCEL_JOIN_TIMEOUT_SECONDS`` monotonic deadline — the inner wait
      and the outer-cancel wait draw from the same budget — so the
      effective ceiling is one window, not two (finding #7).
    """
    if task is None or task.done():
        return

    cancellation_scheduled = False
    try:
        if allow_grace:
            # Grace period for happy-path completion. ``shield`` keeps the
            # task alive if our wait_for is itself cancelled. Note (3.10
            # compatibility): ``asyncio.TimeoutError`` is aliased to the
            # builtin ``TimeoutError`` on 3.11+, but the dual tuple is
            # load-bearing on 3.10 where they are distinct classes.
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=_CANCEL_GRACE_SECONDS
                )
                return
            except (asyncio.TimeoutError, TimeoutError):
                # Happy path did not complete in time; fall through to
                # force-cancel below.
                pass
            except Exception as grace_exc:  # pylint: disable=broad-exception-caught
                # The task itself raised during the grace wait. It has
                # finished — nothing left to clean up. Log the exception
                # rather than silently swallowing it so that operators can
                # diagnose teardown surprises (finding #6).
                if task.done():
                    return
                # This branch is reachable only in unusual orderings
                # (e.g. a TimeoutError raised from inside the task that
                # isn't caught by the dual-class tuple above). Log loudly
                # and fall through to force-cancel.
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

        # Shared monotonic deadline: the combined budget for the inner
        # wait and any CancelledError-branch recovery wait is ONE ceiling
        # window, not two (finding #7). ``max(0, deadline - now)`` is used
        # at each await-site so the remaining budget shrinks monotonically.
        deadline = time.monotonic() + _CANCEL_JOIN_TIMEOUT_SECONDS

        def _remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        # Build a teardown coroutine and shield it so outer cancellation
        # cannot abandon the task mid-teardown. We want resources (httpx
        # clients, file descriptors, LLM subscriptions) to actually unwind.
        teardown = asyncio.ensure_future(
            asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True),
                timeout=_CANCEL_JOIN_TIMEOUT_SECONDS,
            )
        )
        try:
            await asyncio.shield(teardown)
        except asyncio.CancelledError:
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
            except Exception:  # pylint: disable=broad-exception-caught
                pass
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
            # Re-raise CancelledError so the outer scope still observes the
            # cancellation (finding #2: we uncancel only to permit the
            # bounded recovery wait; cancellation semantics for the caller
            # must still propagate).
            raise asyncio.CancelledError()
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
        _CANCEL_JOIN_TIMEOUT_SECONDS,
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


def _run_error_extras(input_data: RunAgentInput) -> dict:
    """Return the extras kwargs for a RunErrorEvent, camelCased to match
    peer events' wire format.

    ``ConfiguredBaseModel`` uses ``extra="allow"`` — extras bypass the
    alias generator, so pre-camelCased keys are required to line up with
    declared-field peers (``RunStartedEvent.thread_id`` / ``run_id`` emit
    as ``threadId`` / ``runId`` via the alias generator). Finding #3.
    """
    return {
        "threadId": input_data.thread_id,
        "runId": input_data.run_id,
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
                    if kickoff_task.exception() is not None:
                        # ``await`` re-raises the stored exception WITH its
                        # original traceback intact.
                        await kickoff_task
                    # Happy path: task finished without error. Drain any
                    # remaining queue items (for example the ``None``
                    # sentinel enqueued by the FlowFinishedEvent listener),
                    # then break. Critically we do NOT fall through to
                    # ``asyncio.wait({get_task, kickoff_task}, ...)``
                    # below, because that wait would return immediately
                    # (kickoff_task is already done) and cause a CPU spin
                    # (finding #1).
                    drained_sentinel = False
                    while True:
                        try:
                            item = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if item is None:
                            drained_sentinel = True
                            break
                        if item.type in (EventType.RUN_STARTED, EventType.RUN_FINISHED):
                            item.thread_id = input_data.thread_id
                            item.run_id = input_data.run_id
                        yield encoder.encode(item)
                    if drained_sentinel:
                        # Sentinel observed inline; still grant the tiny
                        # grace window for any in-flight teardown.
                        allow_grace = True
                    break

                get_task = asyncio.ensure_future(queue.get())
                try:
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            get_task.cancel()
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
                        get_task.cancel()
                        raise TimeoutError(_format_timeout_message(timeout))

                    # Prefer propagating the kickoff exception (if any) over
                    # consuming a queued event — the exception is the real
                    # story the operator needs. ``await`` preserves the
                    # original traceback (finding #4).
                    if kickoff_task in done and kickoff_task.exception() is not None:
                        get_task.cancel()
                        await kickoff_task

                    if get_task in done:
                        item = get_task.result()
                    else:
                        # kickoff finished without error but no item was
                        # enqueued yet; the top-of-loop guard on the next
                        # iteration will observe ``kickoff_task.done()``
                        # and drain via the fast path above (no spin —
                        # finding #1).
                        continue
                finally:
                    if not get_task.done():
                        get_task.cancel()

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

    def _get_flow():
        nonlocal _cached_flow
        if _cached_flow is None:
            _cached_flow = ChatWithCrewFlow(crew=crew)
        return _cached_flow

    @app.post(path)
    async def crew_endpoint(input_data: RunAgentInput, request: Request):
        """Crew chat endpoint with deferred initialization."""
        flow = _get_flow()
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
