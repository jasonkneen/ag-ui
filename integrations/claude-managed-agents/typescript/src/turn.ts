import type Anthropic from "@anthropic-ai/sdk";
import type {
  BetaManagedAgentsEventParams,
  BetaManagedAgentsStreamSessionEvents,
} from "@anthropic-ai/sdk/resources/beta/sessions/events";
import { accumulateManagedAgentsEvent } from "@anthropic-ai/sdk/lib/sessions/accumulate";
import type { AccumulatedEvent } from "@anthropic-ai/sdk/lib/sessions/accumulate";
import { EventType } from "@ag-ui/client";
import type { BaseEvent } from "@ag-ui/client";
import { BEST_EFFORT_SEND_TIMEOUT_MS, PARKED_RETRY_DELAYS_MS, TOOL_RESULT_MAX_CHARS } from "./constants";
import { describeToolResult, textOf } from "./text";
import type { BackendCustomTool, ManagedAgentsErrorHandler } from "./types";

export interface TurnOptions {
  client: Anthropic;
  sessionId: string;
  /** Events posted into the session once the stream is open (user message, tool results). */
  outbound: BetaManagedAgentsEventParams[];
  /** Frontend tools, keyed by their managed-agent (normalized) name → original AG-UI name. Calls to these park the session. */
  clientTools: Map<string, string>;
  /** Custom tools executed on this server, keyed by their managed-agent (normalized) name. */
  backendTools: Map<string, BackendCustomTool>;
  /** How to answer built-in tools gated on confirmation; undefined = fail the run. */
  toolConfirmation: "allow" | "deny" | undefined;
  streamDeltas: boolean;
  /** Notified when a best-effort operation fails. Never throws into the turn. */
  onError?: ManagedAgentsErrorHandler;
  /** Called once the outbound events have been posted into the session. */
  onSent?: () => void | Promise<void>;
  emit: (event: BaseEvent) => void;
  signal: AbortSignal;
}

export type TurnOutcome =
  | { status: "finished" }
  /** The session is parked on custom tool calls the frontend must answer. */
  | { status: "parked"; clientToolUseIds: string[] }
  /** A RUN_ERROR was already emitted. */
  | { status: "errored"; sessionEnded?: boolean };

type StreamEvent = BetaManagedAgentsStreamSessionEvents;

/** Fallback text when the API reports a session error without a message. */
const SESSION_ERROR_FALLBACK = "The session reported an error.";
/** Posted for a backend tool whose handler was cut off by a timeout or disconnect. */
const INTERRUPTED_TOOL_TEXT = "Tool execution was interrupted.";

/**
 * A live text preview: the SDK's accumulated snapshot (best-effort, dropped
 * if it errors) plus the text already emitted to the UI, which the buffered
 * `agent.message` reconciles against.
 */
interface Preview {
  snapshot: AccumulatedEvent | undefined;
  emitted: string;
}

/**
 * Drive one turn of a managed session: open the event stream, post the
 * outbound events, and translate the session's events into AG-UI events
 * until the session goes idle.
 *
 * Invariant: no TEXT_MESSAGE or REASONING block is left open when this
 * returns or throws — every exit path closes them.
 */
export async function runTurn(opts: TurnOptions): Promise<TurnOutcome> {
  const { client, sessionId, emit, signal } = opts;

  /** Report a swallowed failure. A broken hook must never break the turn. */
  const report = (operation: string, error: unknown) => {
    try {
      opts.onError?.(error, { operation, sessionId });
    } catch {
      // ignored on purpose
    }
  };

  // Open the stream before sending so no early events are missed.
  const stream = await client.beta.sessions.events.stream(
    sessionId,
    // "agent.thinking" opts into the live thinking indicator (event_start);
    // thinking carries no text deltas today.
    opts.streamDeltas ? { event_deltas: ["agent.message", "agent.thinking"] } : {},
    { signal },
  );

  // A parked session accepts only tool results, so post those first (which
  // resumes it) and any user messages in a second call. The API validates a
  // whole batch against the session's current state, so mixing them fails.
  const isFollowUp = (event: (typeof opts.outbound)[number]) =>
    event.type === "user.message" || event.type === "system.message";
  const followUps = opts.outbound.filter(isFollowUp);
  const results = opts.outbound.filter((event) => !isFollowUp(event));

  // The session un-parks asynchronously after a tool result is posted, so a
  // follow-up user message can race ahead of that transition and be rejected
  // as sent-while-parked. Retry it briefly on that specific error.
  const sendFollowUps = async () => {
    for (let attempt = 0; ; attempt++) {
      try {
        await client.beta.sessions.events.send(sessionId, { events: followUps });
        return;
      } catch (err) {
        const delayMs = PARKED_RETRY_DELAYS_MS[attempt];
        if (delayMs === undefined || !isSentWhileParked(err)) throw err;
        await sleep(delayMs, signal);
      }
    }
  };

  try {
    if (results.length > 0) await client.beta.sessions.events.send(sessionId, { events: results });
    if (followUps.length > 0) await sendFollowUps();
  } catch (err) {
    stream.controller.abort();
    throw err;
  }
  await opts.onSent?.();

  const previews = new Map<string, Preview>();
  const closedMessages = new Set<string>();
  const openReasoning = new Set<string>();
  const ackedToolUses = new Set<string>();
  const clientParks = new Set<string>();
  const askedConfirmations = new Set<string>();

  const emitTextStart = (messageId: string) =>
    emit({ type: EventType.TEXT_MESSAGE_START, messageId, role: "assistant" } as BaseEvent);
  // Never emit an empty content delta: it is a no-op the schema rejects.
  const emitTextContent = (messageId: string, delta: string) => {
    if (delta) emit({ type: EventType.TEXT_MESSAGE_CONTENT, messageId, delta } as BaseEvent);
  };
  const emitTextEnd = (messageId: string) => emit({ type: EventType.TEXT_MESSAGE_END, messageId } as BaseEvent);

  const openMessage = (messageId: string): Preview => {
    emitTextStart(messageId);
    const preview: Preview = { snapshot: undefined, emitted: "" };
    previews.set(messageId, preview);
    return preview;
  };
  const closeMessage = (messageId: string) => {
    emitTextEnd(messageId);
    previews.delete(messageId);
    closedMessages.add(messageId);
  };
  const closeReasoning = (messageId: string) => {
    emit({ type: EventType.REASONING_MESSAGE_END, messageId } as BaseEvent);
    emit({ type: EventType.REASONING_END, messageId } as BaseEvent);
    openReasoning.delete(messageId);
  };
  const closeAll = () => {
    for (const messageId of [...previews.keys()]) closeMessage(messageId);
    for (const reasoningId of [...openReasoning]) closeReasoning(reasoningId);
  };

  const emitToolCall = (id: string, name: string, input: unknown) => {
    emit({ type: EventType.TOOL_CALL_START, toolCallId: id, toolCallName: name } as BaseEvent);
    emit({ type: EventType.TOOL_CALL_ARGS, toolCallId: id, delta: JSON.stringify(input ?? {}) } as BaseEvent);
    emit({ type: EventType.TOOL_CALL_END, toolCallId: id } as BaseEvent);
  };
  const emitToolResult = (toolUseId: string, content: string) => {
    emit({
      type: EventType.TOOL_CALL_RESULT,
      messageId: `result_${toolUseId}`,
      toolCallId: toolUseId,
      content,
      role: "tool",
    } as BaseEvent);
  };

  const interrupt = () =>
    client.beta.sessions.events
      .send(sessionId, { events: [{ type: "user.interrupt" }] })
      .catch((error: unknown) => report("interrupt", error));

  const fail = (message: string, code?: string): TurnOutcome => {
    closeAll();
    emit({ type: EventType.RUN_ERROR, message, ...(code ? { code } : {}) } as BaseEvent);
    return { status: "errored" };
  };

  /**
   * Post a custom tool's result into the session. Deliberately not tied to the
   * run's abort signal (the session is waiting on the result even when the run
   * is being torn down) but bounded by its own timeout so a stalled connection
   * cannot hold the thread's run gate open.
   */
  const postCustomToolResult = async (toolUseId: string, text: string, isError: boolean) => {
    await client.beta.sessions.events.send(
      sessionId,
      {
        events: [{ type: "user.custom_tool_result", custom_tool_use_id: toolUseId, content: [{ type: "text", text }], is_error: isError }],
      },
      { signal: AbortSignal.timeout(BEST_EFFORT_SEND_TIMEOUT_MS) },
    );
    ackedToolUses.add(toolUseId);
  };

  /** Surface a custom tool result to the UI and answer it in the session. */
  const answerCustomToolUse = async (toolUseId: string, text: string, isError: boolean) => {
    emitToolResult(toolUseId, text);
    await postCustomToolResult(toolUseId, text, isError);
  };

  /**
   * Run a backend custom tool and post its result back into the session.
   * The handler races the turn's abort signal: on a timeout or disconnect
   * the tool is answered with an error (best-effort, non-cancellable) so
   * the session is never left parked, then the abort proceeds.
   */
  const runBackendTool = async (id: string, tool: BackendCustomTool, input: unknown) => {
    let text: string;
    let isError = false;
    try {
      text = await raceAbort(() => tool.handler(input), signal);
    } catch (err) {
      if (signal.aborted) {
        await postCustomToolResult(id, INTERRUPTED_TOOL_TEXT, true).catch((error: unknown) =>
          report("post_interrupted_tool_result", error),
        );
        throw err;
      }
      text = err instanceof Error ? err.message : String(err);
      isError = true;
    }
    await answerCustomToolUse(id, text, isError);
  };

  const consume = async (): Promise<TurnOutcome> => {
    for await (const event of stream as AsyncIterable<StreamEvent>) {
      switch (event.type) {
        case "event_start": {
          if (event.event.type === "agent.message") {
            openMessage(event.event.id).snapshot = accumulateManagedAgentsEvent(undefined, event);
          } else if (event.event.type === "agent.thinking") {
            openReasoning.add(event.event.id);
            emit({ type: EventType.REASONING_START, messageId: event.event.id } as BaseEvent);
            emit({ type: EventType.REASONING_MESSAGE_START, messageId: event.event.id, role: "reasoning" } as BaseEvent);
          }
          break;
        }

        case "event_delta": {
          const preview = previews.get(event.event_id);
          if (!preview) break; // best-effort; the buffered agent.message is canonical
          try {
            preview.snapshot = accumulateManagedAgentsEvent(preview.snapshot, event);
          } catch (error) {
            // A gapped or out-of-order delta: drop this preview and let the
            // buffered agent.message reconcile against what was emitted.
            report("accumulate_preview", error);
            preview.snapshot = undefined;
            break;
          }
          if (event.delta.type === "content_delta" && event.delta.content.type === "text") {
            preview.emitted += event.delta.content.text;
            emitTextContent(event.event_id, event.delta.content.text);
          }
          break;
        }

        case "agent.thinking": {
          // The thinking stretch finished. Its text is not exposed by the API today,
          // so this is a progress signal: close the reasoning block we opened.
          if (openReasoning.has(event.id)) {
            closeReasoning(event.id);
          } else {
            emit({ type: EventType.REASONING_START, messageId: event.id } as BaseEvent);
            emit({ type: EventType.REASONING_END, messageId: event.id } as BaseEvent);
          }
          break;
        }

        case "agent.message": {
          if (closedMessages.has(event.id)) break;
          reconcileMessage(event.id, textOf(event.content));
          break;
        }

        case "agent.custom_tool_use": {
          // Report the frontend's original tool name, which may differ from
          // the normalized name registered on the managed agent.
          emitToolCall(event.id, opts.clientTools.get(event.name) ?? event.name, event.input);
          if (opts.clientTools.has(event.name)) {
            // The frontend executes this tool. Leave it unanswered; the session
            // will park on it and the next run supplies the result.
            clientParks.add(event.id);
            break;
          }
          const backend = opts.backendTools.get(event.name);
          if (backend) {
            await runBackendTool(event.id, backend, event.input);
            break;
          }
          // Nothing can execute this tool. Answer with an error so the agent recovers.
          await answerCustomToolUse(event.id, `No handler is registered for tool "${event.name}".`, true);
          break;
        }

        case "agent.tool_use": {
          emitToolCall(event.id, event.name, event.input);
          if (event.evaluated_permission === "ask") askedConfirmations.add(event.id);
          break;
        }

        case "agent.mcp_tool_use": {
          emitToolCall(event.id, `${event.mcp_server_name}: ${event.name}`, event.input);
          if (event.evaluated_permission === "ask") askedConfirmations.add(event.id);
          break;
        }

        case "agent.tool_result": {
          emitToolResult(event.tool_use_id, describeToolResult(event.content).slice(0, TOOL_RESULT_MAX_CHARS));
          break;
        }

        case "agent.mcp_tool_result": {
          emitToolResult(event.mcp_tool_use_id, describeToolResult(event.content).slice(0, TOOL_RESULT_MAX_CHARS));
          break;
        }

        case "span.model_request_end": {
          // A failed model request produces no buffered agent.message, so its
          // dangling preview must be closed here. A successful one is left
          // to the buffered message, which may arrive after this event.
          if (event.is_error === true) closeAll();
          break;
        }

        case "session.error": {
          const { type, message, retry_status } = event.error;
          if (retry_status.type === "retrying") break; // transient; the session recovers on its own
          return fail(message || SESSION_ERROR_FALLBACK, type);
        }

        case "session.status_idle": {
          const { stop_reason } = event;
          if (stop_reason.type === "end_turn") {
            closeAll();
            return { status: "finished" };
          }
          if (stop_reason.type === "retries_exhausted") {
            return fail("The session gave up after exhausting its retries.", "retries_exhausted");
          }
          // requires_action: work out what the session is blocked on.
          const blockedOn = stop_reason.event_ids.filter((id) => !ackedToolUses.has(id));
          if (blockedOn.length === 0) break; // everything is already answered; wait for it to resume

          const confirmations = blockedOn.filter((id) => askedConfirmations.has(id));
          if (confirmations.length > 0) {
            if (!opts.toolConfirmation) {
              await interrupt();
              return fail(
                "A tool requires confirmation but no confirmation policy is configured. " +
                  "Set `toolConfirmation` to \"allow\" or \"deny\", or use a permission policy that does not ask.",
                "tool_confirmation_required",
              );
            }
            // Bounded like tool-result posts: the session is parked on these answers.
            await client.beta.sessions.events.send(
              sessionId,
              {
                events: confirmations.map((tool_use_id) => ({
                  type: "user.tool_confirmation" as const,
                  tool_use_id,
                  result: opts.toolConfirmation!,
                })),
              },
              { signal: AbortSignal.timeout(BEST_EFFORT_SEND_TIMEOUT_MS) },
            );
            for (const id of confirmations) ackedToolUses.add(id);
            if (confirmations.length === blockedOn.length) break;
          }

          const clientToolUseIds = blockedOn.filter((id) => clientParks.has(id));
          const unknown = blockedOn.filter((id) => !askedConfirmations.has(id) && !clientParks.has(id));
          if (unknown.length > 0) {
            await interrupt();
            return fail("The agent is waiting on an action this integration cannot answer.", "unsupported_action");
          }
          if (clientToolUseIds.length > 0) {
            // Hand control back to the frontend to execute its tools.
            closeAll();
            return { status: "parked", clientToolUseIds };
          }
          break;
        }

        case "session.status_terminated":
        case "session.deleted": {
          closeAll();
          emit({ type: EventType.RUN_ERROR, message: "The managed session ended on the server. Send another message to start a fresh one.", code: "session_ended" } as BaseEvent);
          return { status: "errored", sessionEnded: true };
        }

        default:
          break; // status_running, rescheduled, spans, thread events, echoed user events
      }
    }
    closeAll();
    if (signal.aborted) throw new Error("turn aborted");
    return fail("The session event stream ended before the reply completed.", "stream_ended");
  };

  /** Fold the buffered final `agent.message` into what the preview emitted. */
  const reconcileMessage = (messageId: string, finalText: string) => {
    const preview = previews.get(messageId);
    if (!preview) {
      emitTextStart(messageId);
      emitTextContent(messageId, finalText);
    } else if (finalText.startsWith(preview.emitted)) {
      emitTextContent(messageId, finalText.slice(preview.emitted.length));
    } else {
      // Preview diverged from the final text: close it and re-emit the corrected whole.
      closeMessage(messageId);
      if (finalText) {
        const correctedId = `corrected_${messageId}`;
        emitTextStart(correctedId);
        emitTextContent(correctedId, finalText);
        emitTextEnd(correctedId);
      }
      return;
    }
    closeMessage(messageId);
  };

  try {
    return await consume();
  } finally {
    closeAll();
  }
}

/** The API rejects user messages while a session is parked on tool results. */
const isSentWhileParked = (err: unknown): boolean =>
  typeof err === "object" &&
  err !== null &&
  (err as { status?: unknown }).status === 400 &&
  String((err as { message?: unknown }).message ?? "").includes("waiting on responses");

/** Rejects when `signal` aborts, whether it aborts before or during the wait. */
const abortRejection = <T>(signal: AbortSignal): { promise: Promise<T>; dispose: () => void } => {
  let onAbort: () => void = () => {};
  const promise = new Promise<T>((_, reject) => {
    onAbort = () => reject(new Error("turn aborted"));
    if (signal.aborted) onAbort();
    else signal.addEventListener("abort", onAbort, { once: true });
  });
  return { promise, dispose: () => signal.removeEventListener("abort", onAbort) };
};

/** Await `work`, but give up as soon as `signal` aborts. */
const raceAbort = async <T>(work: () => T | Promise<T>, signal: AbortSignal): Promise<T> => {
  const aborted = abortRejection<T>(signal);
  const working = Promise.resolve().then(work);
  working.catch(() => {}); // an abandoned handler that rejects later must not go unhandled
  try {
    return await Promise.race([working, aborted.promise]);
  } finally {
    aborted.dispose();
  }
};

const sleep = async (ms: number, signal: AbortSignal): Promise<void> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await raceAbort(() => new Promise<void>((resolve) => (timer = setTimeout(resolve, ms))), signal);
  } finally {
    clearTimeout(timer);
  }
};
