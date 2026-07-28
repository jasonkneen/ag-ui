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
    // One key for the session store and the busy-run gate, so a stored session
    // and the gate that serializes access to it can never disagree.
    const key = this.sessionKey(threadId);
    let sessionId: string | undefined;
    emit({ type: EventType.RUN_STARTED, threadId, runId } as BaseEvent);
    if (input.state !== undefined && input.state !== null) {
      emit({ type: EventType.STATE_SNAPSHOT, snapshot: input.state } as BaseEvent);
    }

    if (this.busyThreads.has(key)) {
      emit(runError("A run is already in progress on this thread.", "run_in_progress"));
      return;
    }
    // Check for something sendable before touching the API, so a malformed
    // run does not create an orphan session.
    if (!this.hasSendableContent(input.messages)) {
      emit(runError("There is nothing to send: this run has no user message or tool result.", "empty_run"));
      return;
    }
    this.busyThreads.add(key);
    try {
      const record = await this.getOrCreateSession(key, threadId, input, emit);
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
        await this.store.set(key, record);
        emit({ type: EventType.RUN_FINISHED, threadId, runId } as BaseEvent);
        return;
      }

      const outcome = await runTurn({
        client: this.client,
        sessionId: record.sessionId,
        outbound: outbound.events,
        // Persist each delivery as soon as it lands, so a failure or
        // interruption later in the turn does not re-post it next run: the
        // tool results resume the session even if the follow-ups then fail.
        onResultsSent: async () => {
          record.pendingClientToolUseIds = [];
          await this.store.set(key, record);
        },
        onFollowUpsSent: async () => {
          if (outbound.lastUserMessageId) record.lastUserMessageId = outbound.lastUserMessageId;
          await this.store.set(key, record);
        },
        // Persist a park the moment the call is handed to the UI. A later
        // event can fail the turn before the session confirms the park, and
        // the remote session would then wait on an ID nothing remembers.
        onClientPark: async (toolUseId) => {
          if (record.pendingClientToolUseIds.includes(toolUseId)) return;
          record.pendingClientToolUseIds = [...record.pendingClientToolUseIds, toolUseId];
          await this.store.set(key, record);
        },
        clientTools: new Map((input.tools ?? []).map((tool) => [normalizeToolName(tool.name), tool.name])),
        backendTools: this.backendTools,
        toolConfirmation: this.config.toolConfirmation,
        streamDeltas: this.config.streamDeltas ?? true,
        onError: this.config.onError,
        emit,
        signal,
      });

      await this.recordOutcome(key, record, outcome);
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
      this.busyThreads.delete(key);
    }
  }

  /** Whether the run carries a user message with text or a tool result. */
  private hasSendableContent(messages: Message[]): boolean {
    return hasUserText(messages) || messages.some((message) => message.role === "tool");
  }

  /**
   * The key that identifies this thread's state, in the session store and in
   * the busy-run gate alike, so two agents sharing one store neither adopt each
   * other's sessions nor serialize against each other's threads.
   *
   * Every field baked into the remote session at creation is part of the key:
   * none of them can be re-checked or changed on resume, so an agent must never
   * inherit a session created with a different environment, pinned version or
   * vault set. Each is length-prefixed so no two combinations can collide —
   * plain concatenation would let a `managedAgentId` of `support:beta` with
   * thread `t1` and one of `support` with thread `beta:t1` share one record.
   * The thread id is last, so it needs no prefix and may contain anything.
   */
  private sessionKey(threadId: string): string {
    const { managedAgentId, agentVersion, environmentId } = this.config;
    const field = (value: string) => `${value.length}:${value}|`;
    return (
      field(managedAgentId) +
      field(agentVersion === undefined ? "" : String(agentVersion)) +
      field(environmentId) +
      threadId
    );
  }

  /**
   * Reconcile the record with how the turn ended. An errored turn keeps
   * whatever `onClientPark` already persisted: the remote session is still
   * parked on those calls and the next run has to answer them.
   */
  private async recordOutcome(key: string, record: SessionRecord, outcome: TurnOutcome): Promise<void> {
    if (outcome.status === "errored") {
      if (outcome.sessionEnded) await this.store.delete(key);
      return;
    }
    if (outcome.status === "parked") {
      record.pendingClientToolUseIds = outcome.clientToolUseIds;
      await this.store.set(key, record);
      return;
    }
    // The session went idle on end_turn: nothing is awaited any more.
    if (record.pendingClientToolUseIds.length > 0) {
      record.pendingClientToolUseIds = [];
      await this.store.set(key, record);
    }
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
    key: string,
    threadId: string,
    input: RunAgentInput,
    emit: (event: BaseEvent) => void,
  ): Promise<SessionRecord | undefined> {
    const existing = await this.store.get(key);
    if (existing) return existing;

    // A tool result only answers a pending call on an existing session;
    // never create a session to receive one.
    if (!hasUserText(input.messages)) return undefined;

    // The busy-thread gate serializes runs per thread, so creation cannot race.
    const record = await this.createSession(threadId, input.tools ?? []);
    await this.store.set(key, record);
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
