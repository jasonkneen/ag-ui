import Anthropic from "@anthropic-ai/sdk";
import type { BetaManagedAgentsEventParams } from "@anthropic-ai/sdk/resources/beta/sessions/events";
import type { SessionCreateParams } from "@anthropic-ai/sdk/resources/beta/sessions/sessions";
import { AbstractAgent, EventType } from "@ag-ui/client";
import type { BaseEvent, Message, RunAgentInput, Tool } from "@ag-ui/client";
import { Observable } from "rxjs";
import { BEST_EFFORT_SEND_TIMEOUT_MS, DEFAULT_TURN_TIMEOUT_MS } from "./constants";
import { InMemorySessionStore } from "./sessions";
import { customToolFrom, customToolsFingerprint, normalizeToolName, type CustomToolParams } from "./tools";
import { runTurn, type TurnOutcome } from "./turn";
import type { BackendCustomTool, ManagedAgentsAgentConfig, SessionRecord, SessionStore } from "./types";

type OverrideTools = NonNullable<
  Extract<SessionCreateParams["agent"], { type: "agent_with_overrides" }>["tools"]
>;

/** The text of the given user message (string or multimodal content). */
const userText = (message: Extract<Message, { role: "user" }>): string => {
  if (typeof message.content === "string") return message.content;
  return (message.content ?? [])
    .map((part) => ("text" in part && typeof part.text === "string" ? part.text : ""))
    .join("");
};

/** Whether any user message in the run carries text. */
const hasUserText = (messages: Message[]): boolean =>
  messages.some((message) => message.role === "user" && userText(message).trim().length > 0);

/** A tool message's payload: its content plus any error text. */
const toolResultText = (message: Extract<Message, { role: "tool" }>): string =>
  [message.content, message.error].filter(Boolean).join("\n");

const runError = (message: string, code: string): BaseEvent =>
  ({ type: EventType.RUN_ERROR, message, code }) as BaseEvent;

const ABANDONED_TOOL_TEXT = "The user did not provide a result for this tool call.";

/**
 * An AG-UI agent backed by Claude Managed Agents. Each AG-UI thread maps to
 * one managed session; each run drives one turn of that session.
 */
export class ManagedAgentsAgent extends AbstractAgent {
  private readonly client: Anthropic;
  private readonly store: SessionStore;
  private readonly backendTools: Map<string, BackendCustomTool>;
  private currentRunAbort: AbortController | null = null;

  // Keyed by session-store identity: the store is the unit of tenancy, so
  // agents (and clones) sharing a store serialize runs per thread, while
  // per-caller stores keep one caller's runs from blocking another's.
  // Keys within a store's set are scoped to this managed agent.
  private static busyThreadsByStore = new WeakMap<SessionStore, Set<string>>();

  private get busyThreads(): Set<string> {
    let set = ManagedAgentsAgent.busyThreadsByStore.get(this.store);
    if (!set) {
      set = new Set<string>();
      ManagedAgentsAgent.busyThreadsByStore.set(this.store, set);
    }
    return set;
  }

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

  /** Report a swallowed failure. A broken hook must never break the run. */
  private report(operation: string, error: unknown, ids: { sessionId?: string; threadId?: string } = {}) {
    try {
      this.config.onError?.(error, { operation, ...ids });
    } catch {
      // ignored on purpose
    }
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

      this.runTurnForInput(input, emit, signal)
        .catch((err) => {
          if (disconnect.signal.aborted) return; // client went away; nothing left to tell
          if (timeout.aborted) {
            emit(runError(`The turn exceeded the ${timeoutMs / 1000}s limit and was interrupted.`, "turn_timeout"));
            return;
          }
          emit(runError(err instanceof Error && err.message ? err.message : "The run failed.", "run_failed"));
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
  ): Promise<void> {
    // RunAgentInput is not validated at runtime; a body without `messages` or
    // `tools` must read as an empty run, not a TypeError.
    input = { ...input, messages: input.messages ?? [], tools: input.tools ?? [] };
    const { threadId, runId } = input;
    const busyKey = this.threadKey(threadId);
    let sessionId: string | undefined;
    emit({ type: EventType.RUN_STARTED, threadId, runId } as BaseEvent);
    if (input.state !== undefined && input.state !== null) {
      emit({ type: EventType.STATE_SNAPSHOT, snapshot: input.state } as BaseEvent);
    }

    if (this.busyThreads.has(busyKey)) {
      emit(runError("A run is already in progress on this thread.", "run_in_progress"));
      return;
    }
    // Check for something sendable before touching the API, so a malformed
    // run does not create an orphan session.
    if (!this.hasSendableContent(input.messages)) {
      emit(runError("There is nothing to send: this run has no user message or tool result.", "empty_run"));
      return;
    }
    this.busyThreads.add(busyKey);
    try {
      const record = await this.getOrCreateSession(threadId, input, emit);
      if (!record) {
        emit(runError("There is nothing to send: a tool result arrived for a thread with no session.", "tool_result_without_session"));
        return;
      }
      sessionId = record.sessionId;
      await this.syncClientTools(record, input.tools);

      const outbound = this.outboundEvents(record, input.messages);
      if (outbound.events.length === 0) {
        emit(runError("There is nothing new to send: no user message or tool result in this run.", "nothing_to_send"));
        return;
      }

      // Some parked tool calls are still unanswered: post what we have and
      // stay parked instead of waiting on a session that will not resume.
      if (outbound.stillParked.length > 0) {
        await this.client.beta.sessions.events.send(record.sessionId, { events: outbound.events }, { signal });
        record.pendingClientToolUseIds = outbound.stillParked;
        record.lastUserMessageId = outbound.lastUserMessageId ?? record.lastUserMessageId;
        await this.store.set(threadId, record);
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
          await this.store.set(threadId, record);
        },
        clientTools: new Map((input.tools ?? []).map((tool) => [normalizeToolName(tool.name), tool.name])),
        backendTools: this.backendTools,
        toolConfirmation: this.config.toolConfirmation,
        streamDeltas: this.config.streamDeltas ?? true,
        onError: this.config.onError,
        emit,
        signal,
      });

      await this.recordOutcome(threadId, record, outcome);
      if (outcome.status !== "errored") {
        emit({ type: EventType.RUN_FINISHED, threadId, runId } as BaseEvent);
      }
    } catch (err) {
      // Interrupt the session while the busy gate is still held, so a user
      // who resends right away is not interrupted by this run's teardown.
      if (signal.aborted && sessionId) {
        await this.client.beta.sessions.events
          .send(sessionId, { events: [{ type: "user.interrupt" }] }, { signal: AbortSignal.timeout(BEST_EFFORT_SEND_TIMEOUT_MS) })
          .catch((error: unknown) => this.report("interrupt", error, { sessionId, threadId }));
      }
      throw err;
    } finally {
      this.busyThreads.delete(busyKey);
    }
  }

  /** Whether the run carries a user message with text or a tool result. */
  private hasSendableContent(messages: Message[]): boolean {
    return hasUserText(messages) || messages.some((message) => message.role === "tool");
  }

  /** Shared-state key scoped to this managed agent so distinct agents never collide. */
  private threadKey(threadId: string): string {
    return `${this.config.managedAgentId}:${threadId}`;
  }

  private async recordOutcome(threadId: string, record: SessionRecord, outcome: TurnOutcome): Promise<void> {
    if (outcome.status === "errored" && outcome.sessionEnded) {
      await this.store.delete(threadId);
      return;
    }
    if (outcome.status !== "parked") return;
    record.pendingClientToolUseIds = outcome.clientToolUseIds;
    await this.store.set(threadId, record);
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
    const toolResult = (toolUseId: string, text: string, isError: boolean): BetaManagedAgentsEventParams => ({
      type: "user.custom_tool_result",
      custom_tool_use_id: toolUseId,
      content: [{ type: "text", text }],
      is_error: isError,
    });

    const answered: BetaManagedAgentsEventParams[] = [];
    const pending = new Set(record.pendingClientToolUseIds);
    for (const message of messages) {
      if (message.role !== "tool" || !pending.has(message.toolCallId)) continue;
      answered.push(toolResult(message.toolCallId, toolResultText(message), Boolean(message.error)));
      pending.delete(message.toolCallId);
    }

    // User messages after the last delivered one; on first contact, just the newest.
    let lastUserMessageId: string | undefined;
    const followUps: BetaManagedAgentsEventParams[] = [];
    const userMessages = messages.filter((message) => message.role === "user");
    const deliveredIndex = userMessages.findIndex((message) => message.id === record.lastUserMessageId);
    const undelivered = deliveredIndex >= 0 ? userMessages.slice(deliveredIndex + 1) : userMessages.slice(-1);
    for (const message of undelivered) {
      const text = userText(message).trim();
      if (!text) continue;
      followUps.push({ type: "user.message", content: [{ type: "text", text }] });
      lastUserMessageId = message.id;
    }

    // The user moved on without answering the tools the frontend was asked
    // to run: fail those calls (in their original order) so the agent can
    // respond to the new message.
    const abandoned: BetaManagedAgentsEventParams[] = [];
    if (lastUserMessageId !== undefined && pending.size > 0) {
      for (const toolUseId of pending) abandoned.push(toolResult(toolUseId, ABANDONED_TOOL_TEXT, true));
      pending.clear();
    }

    return { events: [...abandoned, ...answered, ...followUps], stillParked: [...pending], lastUserMessageId };
  }

  private async getOrCreateSession(
    threadId: string,
    input: RunAgentInput,
    emit: (event: BaseEvent) => void,
  ): Promise<SessionRecord | undefined> {
    const existing = await this.store.get(threadId);
    if (existing) return existing;

    // A tool result only answers a pending call on an existing session;
    // never create a session to receive one.
    if (!hasUserText(input.messages)) return undefined;

    // The busy-thread gate serializes runs per thread, so creation cannot race.
    const record = await this.createSession(threadId, input.tools ?? []);
    await this.store.set(threadId, record);
    emit({
      type: EventType.CUSTOM,
      name: "managed_agents.session",
      value: { sessionId: record.sessionId, threadId },
    } as BaseEvent);
    return record;
  }

  private async createSession(threadId: string, clientTools: Tool[]): Promise<SessionRecord> {
    const { managedAgentId, agentVersion, environmentId } = this.config;
    const customTools = this.customTools(clientTools);
    const title = this.config.sessionTitle?.(threadId) ?? `AG-UI thread ${threadId}`;
    const agentRef = { id: managedAgentId, ...(agentVersion !== undefined && { version: agentVersion }) };

    const agent: SessionCreateParams["agent"] =
      customTools.length === 0
        ? { type: "agent", ...agentRef }
        : { type: "agent_with_overrides", ...agentRef, tools: await this.mergedTools(customTools) };

    const session = await this.client.beta.sessions.create({ agent, environment_id: environmentId, title });
    return {
      sessionId: session.id,
      toolNames: customTools.map((tool) => tool.name),
      toolDefinitionsFingerprint: customToolsFingerprint(customTools),
      pendingClientToolUseIds: [],
    };
  }

  /**
   * Frontend tools plus configured backend tools, as custom tool definitions.
   * Keyed by normalized name: on a collision the last definition wins, and
   * a frontend tool always beats a backend tool of the same name, matching
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

  /** Keep the session's full replacement tool list aligned with this run. */
  private async syncClientTools(record: SessionRecord, clientTools: Tool[]): Promise<void> {
    const desired = this.customTools(clientTools);
    const fingerprint = customToolsFingerprint(desired);
    if (record.toolDefinitionsFingerprint === fingerprint) return;

    await this.client.beta.sessions.update(record.sessionId, {
      agent: { tools: await this.mergedTools(desired) },
    });
    record.toolNames = desired.map((tool) => tool.name);
    record.toolDefinitionsFingerprint = fingerprint;
  }

  /**
   * The agent's own tools plus `custom` tools, without duplicate names.
   * Overrides replace the whole list, so the agent's tools are carried
   * along, but a custom tool of the same name wins over the agent's copy.
   */
  private async mergedTools(custom: CustomToolParams[]): Promise<OverrideTools> {
    const names = new Set(custom.map((tool) => tool.name));
    const base = (await this.baseTools()).filter((tool) => tool.type !== "custom" || !names.has(tool.name));
    return [...base, ...custom];
  }

  /** The tools defined on the managed agent itself, fetched fresh so console edits apply. */
  private async baseTools(): Promise<OverrideTools> {
    const agent = await this.client.beta.agents.retrieve(
      this.config.managedAgentId,
      this.config.agentVersion !== undefined ? { version: this.config.agentVersion } : undefined,
    );
    // The read shape is structurally compatible with the params shape.
    return agent.tools as unknown as OverrideTools;
  }
}
