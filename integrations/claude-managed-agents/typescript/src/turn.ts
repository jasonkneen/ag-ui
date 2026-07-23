import type Anthropic from "@anthropic-ai/sdk";
import type {
  BetaManagedAgentsEventParams,
  BetaManagedAgentsStreamSessionEvents,
} from "@anthropic-ai/sdk/resources/beta/sessions/events";
import { accumulateManagedAgentsEvent } from "@anthropic-ai/sdk/lib/sessions/accumulate";
import type { AccumulatedEvent } from "@anthropic-ai/sdk/lib/sessions/accumulate";
import { EventType } from "@ag-ui/client";
import type { BaseEvent } from "@ag-ui/client";
import { describeToolResult, textOf } from "./text";
import type { BackendCustomTool } from "./types";

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
    const delaysMs = [150, 300, 600, 1000, 1500, 2000];
    for (let attempt = 0; ; attempt++) {
      try {
        await client.beta.sessions.events.send(sessionId, { events: followUps });
        return;
      } catch (err) {
        if (attempt >= delaysMs.length || !isSentWhileParked(err)) throw err;
        await sleep(delaysMs[attempt]!, signal);
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

  const previews = new Map<string, AccumulatedEvent>();
  const closedMessages = new Set<string>();
  const openReasoning = new Set<string>();
  const ackedToolUses = new Set<string>();
  const clientParks = new Set<string>();
  const askedConfirmations = new Set<string>();

  const closeMessage = (messageId: string) => {
    emit({ type: EventType.TEXT_MESSAGE_END, messageId } as BaseEvent);
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
      .catch(() => {});

  const fail = (message: string, code?: string): TurnOutcome => {
    closeAll();
    emit({ type: EventType.RUN_ERROR, message, ...(code ? { code } : {}) } as BaseEvent);
    return { status: "errored" };
  };

  /** Run a backend custom tool and post its result back into the session. */
  const runBackendTool = async (id: string, tool: BackendCustomTool, input: unknown) => {
    let text: string;
    let isError = false;
    try {
      text = await tool.handler(input);
    } catch (err) {
      isError = true;
      text = err instanceof Error ? err.message : String(err);
    }
    emitToolResult(id, text);
    await client.beta.sessions.events.send(sessionId, {
      events: [
        {
          type: "user.custom_tool_result",
          custom_tool_use_id: id,
          content: [{ type: "text", text }],
          is_error: isError,
        },
      ],
    });
    ackedToolUses.add(id);
  };

  const consume = async (): Promise<TurnOutcome> => {
    for await (const event of stream as AsyncIterable<StreamEvent>) {
      switch (event.type) {
        case "event_start": {
          if (event.event.type === "agent.message") {
            emit({ type: EventType.TEXT_MESSAGE_START, messageId: event.event.id, role: "assistant" } as BaseEvent);
            const seed = accumulateManagedAgentsEvent(undefined, event);
            if (seed) previews.set(event.event.id, seed);
          } else if (event.event.type === "agent.thinking") {
            openReasoning.add(event.event.id);
            emit({ type: EventType.REASONING_START, messageId: event.event.id } as BaseEvent);
            emit({ type: EventType.REASONING_MESSAGE_START, messageId: event.event.id, role: "reasoning" } as BaseEvent);
          }
          break;
        }

        case "event_delta": {
          const snapshot = previews.get(event.event_id);
          if (!snapshot) break; // best-effort; the buffered agent.message is canonical
          previews.set(event.event_id, accumulateManagedAgentsEvent(snapshot, event) ?? snapshot);
          if (event.delta.type === "content_delta" && event.delta.content.type === "text") {
            emit({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: event.event_id, delta: event.delta.content.text } as BaseEvent);
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
          const snapshot = previews.get(event.id);
          const finalText = textOf(event.content);
          if (!snapshot) {
            emit({ type: EventType.TEXT_MESSAGE_START, messageId: event.id, role: "assistant" } as BaseEvent);
            if (finalText) emit({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: event.id, delta: finalText } as BaseEvent);
          } else {
            const previewed = textOf(snapshot.content);
            if (finalText.startsWith(previewed)) {
              if (finalText.length > previewed.length) {
                emit({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: event.id, delta: finalText.slice(previewed.length) } as BaseEvent);
              }
            } else {
              // Preview diverged from the final text: close it and re-emit the corrected whole.
              closeMessage(event.id);
              if (finalText) {
                const messageId = `corrected_${event.id}`;
                emit({ type: EventType.TEXT_MESSAGE_START, messageId, role: "assistant" } as BaseEvent);
                emit({ type: EventType.TEXT_MESSAGE_CONTENT, messageId, delta: finalText } as BaseEvent);
                emit({ type: EventType.TEXT_MESSAGE_END, messageId } as BaseEvent);
              }
              break;
            }
          }
          closeMessage(event.id);
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
          const text = `No handler is registered for tool "${event.name}".`;
          emitToolResult(event.id, text);
          await client.beta.sessions.events.send(sessionId, {
            events: [
              { type: "user.custom_tool_result", custom_tool_use_id: event.id, content: [{ type: "text", text }], is_error: true },
            ],
          });
          ackedToolUses.add(event.id);
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
          emitToolResult(event.tool_use_id, describeToolResult(event.content ?? undefined).slice(0, 4000));
          break;
        }

        case "agent.mcp_tool_result": {
          emitToolResult(event.mcp_tool_use_id, describeToolResult(event.content ?? undefined).slice(0, 4000));
          break;
        }

        case "span.model_request_end": {
          // Closes any preview whose buffered agent.message never arrived
          // (e.g. an interrupted or errored model request).
          closeAll();
          break;
        }

        case "session.error": {
          const { type, message, retry_status } = event.error;
          if (retry_status.type === "retrying") break; // transient; the session recovers on its own
          return fail(message, type);
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
            await client.beta.sessions.events.send(sessionId, {
              events: confirmations.map((tool_use_id) => ({
                type: "user.tool_confirmation" as const,
                tool_use_id,
                result: opts.toolConfirmation!,
              })),
            });
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

const sleep = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new Error("turn aborted"));
      },
      { once: true },
    );
  });
