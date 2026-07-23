import Anthropic from "@anthropic-ai/sdk";
import type { BetaManagedAgentsEventParams } from "@anthropic-ai/sdk/resources/beta/sessions/events";
import type { SessionCreateParams } from "@anthropic-ai/sdk/resources/beta/sessions/sessions";
import { AbstractAgent, EventType } from "@ag-ui/client";
import type { BaseEvent, Message, RunAgentInput, Tool } from "@ag-ui/client";
import { Observable } from "rxjs";
import { InMemorySessionStore } from "./sessions";
import { customToolFrom, normalizeToolName, type CustomToolParams } from "./tools";
import { runTurn, type TurnOutcome } from "./turn";
import type { BackendCustomTool, ManagedAgentsAgentConfig, SessionRecord, SessionStore } from "./types";

const DEFAULT_TURN_TIMEOUT_MS = 5 * 60 * 1000;

type OverrideTools = NonNullable<
  Extract<SessionCreateParams["agent"], { type: "agent_with_overrides" }>["tools"]
>;

/** The text of the given user message. */
const userText = (message: Extract<Message, { role: "user" }>): string => {
  if (typeof message.content === "string") return message.content;
  return (message.content ?? [])
    .map((part) => ("text" in part && typeof part.text === "string" ? part.text : ""))
    .join("");
};

/**
 * An AG-UI agent backed by Claude Managed Agents. Each AG-UI thread maps to
 * one managed session; each run drives one turn of that session.
 */
export class ManagedAgentsAgent extends AbstractAgent {
  private readonly client: Anthropic;
  private readonly store: SessionStore;
  private readonly backendTools: Map<string, BackendCustomTool>;
  private currentRunAbort: AbortController | null = null;

  // Shared across clones so per-run instances see the same state.
  // Keys are scoped to this managed agent so distinct agents never collide.
  private static busyThreads = new Set<string>();

  constructor(private config: ManagedAgentsAgentConfig) {
    super(config);
    this.client = config.client ?? new Anthropic();
    this.store = config.sessionStore ?? new InMemorySessionStore();
    this.backendTools = new Map((config.backendTools ?? []).map((tool) => [normalizeToolName(tool.name), tool]));
  }

  public clone() {
    // Clones share the client and session store so per-run copies see the
    // same thread↔session mappings.
    return new ManagedAgentsAgent({ ...this.config, client: this.client, sessionStore: this.store });
  }

  public abortRun() {
    this.currentRunAbort?.abort();
  }

  run(input: RunAgentInput): Observable<BaseEvent> {
    return new Observable<BaseEvent>((subscriber) => {
      const disconnect = new AbortController();
      this.currentRunAbort = disconnect;
      const timeoutMs = this.config.turnTimeoutMs ?? DEFAULT_TURN_TIMEOUT_MS;
      const timeout = AbortSignal.timeout(timeoutMs);
      const signal = AbortSignal.any([disconnect.signal, timeout]);
      const emit = (event: BaseEvent) => subscriber.next(event);
      let sessionId: string | undefined;

      this.runTurnForInput(input, emit, signal, (id) => (sessionId = id))
        .catch((err) => {
          if (signal.aborted && sessionId) {
            this.client.beta.sessions.events
              .send(sessionId, { events: [{ type: "user.interrupt" }] })
              .catch(() => {});
          }
          if (disconnect.signal.aborted) return; // client went away; nothing left to tell
          const message = timeout.aborted
            ? `The turn exceeded the ${Math.round(timeoutMs / 1000)}s limit and was interrupted.`
            : err instanceof Error
              ? err.message
              : "The run failed.";
          emit({ type: EventType.RUN_ERROR, message } as BaseEvent);
        })
        .finally(() => subscriber.complete());

      return () => {
        disconnect.abort();
        if (this.currentRunAbort === disconnect) this.currentRunAbort = null;
      };
    });
  }

  private async runTurnForInput(
    input: RunAgentInput,
    emit: (event: BaseEvent) => void,
    signal: AbortSignal,
    onSession: (sessionId: string) => void,
  ): Promise<void> {
    const { threadId, runId } = input;
    const busyKey = this.threadKey(threadId);
    emit({ type: EventType.RUN_STARTED, threadId, runId } as BaseEvent);
    if (input.state !== undefined && input.state !== null) {
      emit({ type: EventType.STATE_SNAPSHOT, snapshot: input.state } as BaseEvent);
    }

    if (ManagedAgentsAgent.busyThreads.has(busyKey)) {
      emit({ type: EventType.RUN_ERROR, message: "A run is already in progress on this thread." } as BaseEvent);
      return;
    }
    // Check for something sendable before touching the API, so a malformed
    // run does not create an orphan session.
    if (!this.hasSendableContent(input.messages)) {
      emit({ type: EventType.RUN_ERROR, message: "There is nothing to send: this run has no user message or tool result." } as BaseEvent);
      return;
    }
    ManagedAgentsAgent.busyThreads.add(busyKey);
    try {
      const record = await this.getOrCreateSession(threadId, input, emit);
      if (!record) {
        emit({ type: EventType.RUN_ERROR, message: "There is nothing to send: a tool result arrived for a thread with no session." } as BaseEvent);
        return;
      }
      onSession(record.sessionId);
      await this.syncClientTools(record, input.tools);

      const outbound = this.outboundEvents(record, input.messages);
      if (outbound.events.length === 0) {
        emit({ type: EventType.RUN_ERROR, message: "There is nothing new to send: no user message or tool result in this run." } as BaseEvent);
        return;
      }

      // Some parked tool calls are still unanswered: post what we have and
      // stay parked instead of waiting on a session that will not resume.
      if (outbound.stillParked.length > 0) {
        await this.client.beta.sessions.events.send(record.sessionId, { events: outbound.events }, { signal });
        record.pendingClientToolUseIds = outbound.stillParked;
        record.lastUserMessageId = outbound.lastUserMessageId ?? record.lastUserMessageId;
        await this.store.set(this.storeKey(threadId), record);
        emit({ type: EventType.RUN_FINISHED, threadId, runId } as BaseEvent);
        return;
      }

      const outcome = await runTurn({
        client: this.client,
        sessionId: record.sessionId,
        outbound: outbound.events,
        // Persist delivery as soon as the events land, so a timeout or
        // disconnect later in the turn does not re-post them next run.
        onSent: async () => {
          record.pendingClientToolUseIds = [];
          if (outbound.lastUserMessageId) record.lastUserMessageId = outbound.lastUserMessageId;
          await this.store.set(this.storeKey(threadId), record);
        },
        clientTools: new Map((input.tools ?? []).map((tool) => [normalizeToolName(tool.name), tool.name])),
        backendTools: this.backendTools,
        toolConfirmation: this.config.toolConfirmation,
        streamDeltas: this.config.streamDeltas ?? true,
        emit,
        signal,
      });

      await this.recordOutcome(threadId, record, outcome);
      if (outcome.status !== "errored") {
        emit({ type: EventType.RUN_FINISHED, threadId, runId } as BaseEvent);
      }
    } finally {
      ManagedAgentsAgent.busyThreads.delete(busyKey);
    }
  }

  /** Whether the run carries a user message with text or a tool result. */
  private hasSendableContent(messages: Message[]): boolean {
    return messages.some(
      (message) =>
        message.role === "tool" ||
        (message.role === "user" && userText(message).trim().length > 0),
    );
  }

  /** Shared-state key scoped to this managed agent (and scope, if set). */
  private threadKey(threadId: string): string {
    return `${this.config.agentId}:${this.storeKey(threadId)}`;
  }

  /** Store key: the thread ID, partitioned by scope when configured. */
  private storeKey(threadId: string): string {
    return this.config.scope ? `${this.config.scope}:${threadId}` : threadId;
  }

  private async recordOutcome(threadId: string, record: SessionRecord, outcome: TurnOutcome): Promise<void> {
    if (outcome.status === "errored" && outcome.sessionEnded) {
      await this.store.delete(this.storeKey(threadId));
      return;
    }
    if (outcome.status !== "parked") return;
    record.pendingClientToolUseIds = outcome.clientToolUseIds;
    await this.store.set(this.storeKey(threadId), record);
  }

  /**
   * Work out what to post into the session for this run: results for any
   * tool calls the frontend was asked to run, plus every user message not
   * yet delivered (in order).
   */
  private outboundEvents(
    record: SessionRecord,
    messages: Message[],
  ): { events: BetaManagedAgentsEventParams[]; stillParked: string[]; lastUserMessageId?: string } {
    const events: BetaManagedAgentsEventParams[] = [];
    const pending = new Set(record.pendingClientToolUseIds);

    for (const message of messages) {
      if (message.role !== "tool" || !pending.has(message.toolCallId)) continue;
      events.push({
        type: "user.custom_tool_result",
        custom_tool_use_id: message.toolCallId,
        content: [{ type: "text", text: message.content }],
        is_error: Boolean(message.error),
      });
      pending.delete(message.toolCallId);
    }

    // User messages after the last delivered one; on first contact, just the newest.
    let lastUserMessageId: string | undefined;
    const userMessages = messages.filter((message) => message.role === "user");
    const deliveredIndex = userMessages.findIndex((message) => message.id === record.lastUserMessageId);
    const undelivered = deliveredIndex >= 0 ? userMessages.slice(deliveredIndex + 1) : userMessages.slice(-1);
    for (const message of undelivered) {
      const text = userText(message).trim();
      if (!text) continue;
      events.push({ type: "user.message", content: [{ type: "text", text }] });
      lastUserMessageId = message.id;
    }

    // The user moved on without answering the tools the frontend was asked
    // to run: fail those calls so the agent can respond to the new message.
    if (lastUserMessageId !== undefined && pending.size > 0) {
      for (const toolUseId of pending) {
        events.unshift({
          type: "user.custom_tool_result",
          custom_tool_use_id: toolUseId,
          content: [{ type: "text", text: "The user did not provide a result for this tool call." }],
          is_error: true,
        });
      }
      pending.clear();
    }

    return { events, stillParked: [...pending], lastUserMessageId };
  }

  private async getOrCreateSession(
    threadId: string,
    input: RunAgentInput,
    emit: (event: BaseEvent) => void,
  ): Promise<SessionRecord | undefined> {
    const existing = await this.store.get(this.storeKey(threadId));
    if (existing) return existing;

    // A tool result only answers a pending call on an existing session;
    // never create a session to receive one.
    const hasUserText = input.messages.some((m) => m.role === "user" && userText(m).trim().length > 0);
    if (!hasUserText) return undefined;

    // The busy-thread gate serializes runs per thread, so creation cannot race.
    const record = await this.createSession(threadId, input.tools ?? []);
    await this.store.set(this.storeKey(threadId), record);
    emit({
      type: EventType.CUSTOM,
      name: "managed_agents.session",
      value: { sessionId: record.sessionId, threadId },
    } as BaseEvent);
    return record;
  }

  private async createSession(threadId: string, clientTools: Tool[]): Promise<SessionRecord> {
    const { agentId, agentVersion, environmentId } = this.config;
    const customTools = this.customTools(clientTools);
    const title = this.config.sessionTitle?.(threadId) ?? `AG-UI thread ${threadId}`;

    const agent: SessionCreateParams["agent"] =
      customTools.length === 0
        ? { type: "agent", id: agentId, ...(agentVersion !== undefined && { version: agentVersion }) }
        : {
            type: "agent_with_overrides",
            id: agentId,
            ...(agentVersion !== undefined && { version: agentVersion }),
            // Overrides replace the tool list, so keep the agent's own tools.
            tools: [...(await this.baseTools()), ...customTools],
          };

    const session = await this.client.beta.sessions.create({ agent, environment_id: environmentId, title });
    return {
      sessionId: session.id,
      toolNames: customTools.map((tool) => tool.name),
      pendingClientToolUseIds: [],
    };
  }

  /**
   * Frontend tools plus configured backend tools, as custom tool definitions.
   * Keyed by normalized name; on a collision the frontend tool wins, matching
   * dispatch order in the turn loop.
   */
  private customTools(clientTools: Tool[]): CustomToolParams[] {
    const byName = new Map<string, CustomToolParams>();
    for (const tool of this.config.backendTools ?? []) {
      byName.set(normalizeToolName(tool.name), customToolFrom(tool));
    }
    for (const tool of clientTools) {
      const custom = customToolFrom(tool);
      byName.set(custom.name, custom);
    }
    return [...byName.values()];
  }

  /**
   * Register any client tools the session's agent does not yet have.
   * The tool list is a full replacement, so we merge with what the agent has.
   */
  private async syncClientTools(record: SessionRecord, clientTools: Tool[]): Promise<void> {
    const desired = this.customTools(clientTools ?? []);
    const known = new Set(record.toolNames);
    if (desired.every((tool) => known.has(tool.name))) return;

    await this.client.beta.sessions.update(record.sessionId, {
      agent: { tools: [...(await this.baseTools()), ...desired] },
    });
    record.toolNames = desired.map((tool) => tool.name);
  }

  /** The tools defined on the managed agent itself, fetched fresh so console edits apply. */
  private async baseTools(): Promise<OverrideTools> {
    const agent = await this.client.beta.agents.retrieve(
      this.config.agentId,
      this.config.agentVersion !== undefined ? { version: this.config.agentVersion } : undefined,
    );
    // The read shape is structurally compatible with the params shape.
    return agent.tools as unknown as OverrideTools;
  }
}
