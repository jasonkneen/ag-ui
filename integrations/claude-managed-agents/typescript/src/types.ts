import type Anthropic from "@anthropic-ai/sdk";
import type { AgentConfig } from "@ag-ui/client";

/**
 * A tool the agent may call, executed on this server rather than in the
 * browser. Registered on the managed agent as a `custom` tool; when the
 * agent calls it we run `handler`, stream the call and result to the UI,
 * and post the result back into the session.
 */
export interface BackendCustomTool {
  name: string;
  description: string;
  /** JSON Schema for the tool input. */
  parameters: Record<string, unknown>;
  handler: (input: unknown) => Promise<string> | string;
}

/** Persistent mapping between an AG-UI thread and a managed session. */
export interface SessionRecord {
  sessionId: string;
  /** Custom tool names currently registered on the session's agent. */
  toolNames: string[];
  /** Fingerprint of the canonical custom tool definitions registered on the session's agent. */
  toolDefinitionsFingerprint?: string;
  /** ID of the last user message forwarded into the session. */
  lastUserMessageId?: string;
  /**
   * Custom tool calls handed to the frontend that the session is parked on.
   * The next run must answer them with `role: "tool"` messages.
   */
  pendingClientToolUseIds: string[];
}

/**
 * Where thread↔session mappings live. The default is in-memory (lost on
 * restart, in which case a fresh session is created). Provide your own to
 * survive restarts or run multiple replicas.
 */
export interface SessionStore {
  get(threadId: string): Promise<SessionRecord | undefined> | SessionRecord | undefined;
  set(threadId: string, record: SessionRecord): Promise<void> | void;
  delete(threadId: string): Promise<void> | void;
}

export interface ManagedAgentsAgentConfig extends AgentConfig {
  /**
   * ID of the Anthropic managed agent (`agent_...`) that powers each session.
   * Named apart from AG-UI's own `agentId` so the two never collide.
   */
  managedAgentId: string;
  /** Pin an agent version; omit to use the latest at session creation. */
  agentVersion?: number;
  /** ID of the environment the agent runs in. */
  environmentId: string;
  /** Anthropic client. Defaults to `new Anthropic()`, which reads `ANTHROPIC_API_KEY`. */
  client?: Anthropic;
  /**
   * Thread↔session store, keyed by AG-UI thread ID. Defaults to an in-memory
   * store. Thread IDs are client-supplied, so put the endpoint behind your own
   * authentication and supply a store that partitions by caller if you need
   * multi-tenant isolation.
   */
  sessionStore?: SessionStore;
  /** Tools the agent can call that this server executes. */
  backendTools?: BackendCustomTool[];
  /** Title for newly created sessions. Defaults to `AG-UI thread <threadId>`. */
  sessionTitle?: (threadId: string) => string;
  /**
   * When a built-in tool is gated on user confirmation (`evaluated_permission: "ask"`),
   * answer it automatically. `undefined` (default) ends the run with an error
   * instead, since there is no confirmation UI wired up yet.
   */
  toolConfirmation?: "allow" | "deny";
  /** Abort a turn that runs longer than this. Defaults to 5 minutes. */
  turnTimeoutMs?: number;
  /**
   * Request text and thinking previews so replies stream incrementally.
   * Set to false to receive each reply as a whole message only. Default true.
   */
  streamDeltas?: boolean;
}
