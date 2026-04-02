import { Message, RunAgentInput, State } from "@ag-ui/core";

/** Normalized debug configuration for the AG-UI agent. */
export interface ResolvedAgentDebugConfig {
  enabled: boolean;
  events: boolean;
  lifecycle: boolean;
  verbose: boolean;
}

/** Debug input — boolean shorthand or granular config. */
export type AgentDebugConfig =
  | boolean
  | {
      events?: boolean;
      lifecycle?: boolean;
      verbose?: boolean;
    };

/** Resolves an AgentDebugConfig into a normalized ResolvedAgentDebugConfig. */
export function resolveAgentDebugConfig(
  debug: AgentDebugConfig | undefined,
): ResolvedAgentDebugConfig {
  if (!debug) return { enabled: false, events: false, lifecycle: false, verbose: false };
  if (debug === true) return { enabled: true, events: true, lifecycle: true, verbose: true };

  const events = debug.events ?? true;
  const lifecycle = debug.lifecycle ?? true;
  const verbose = debug.verbose ?? false;
  return { enabled: events || lifecycle, events, lifecycle, verbose };
}

/**
 * Input configuration for throttling subscriber notifications during streaming.
 *
 * Mutations are always applied immediately (`agent.messages`/`agent.state`
 * stay current); only subscriber notifications (`onMessagesChanged`,
 * `onStateChanged`) are coalesced.
 *
 * The first event always fires immediately (leading edge). Subsequent
 * notifications fire when either threshold is met. A trailing timer
 * ensures pending notifications are flushed; the finalize flush supersedes
 * any pending trailing timer (the timer is cleared before final delivery).
 * On stream completion, any remaining pending notification is always delivered.
 */
export interface NotificationThrottleConfig {
  /**
   * Time-based throttle window in milliseconds.
   * Notifications are suppressed for this duration after each delivery;
   * when the window expires, a single coalesced notification is delivered
   * reflecting all mutations that accumulated during the window.
   * Must be a non-negative finite number. Example: `16` sets a minimum
   * ~16 ms gap between notifications (comparable to a 60 fps cadence).
   */
  readonly intervalMs: number;
  /**
   * Minimum new characters to accumulate before firing a notification.
   * When set, a notification also fires when this many new characters
   * have been appended to the trailing assistant message (the last message
   * in the array, when it has role `"assistant"` and string content),
   * even if the time window has not yet elapsed.
   * Must be a non-negative finite number. Default: `0` (no minimum).
   */
  readonly minChunkSize?: number;
}

/** Normalized throttle config — `minChunkSize` is always a definite number. */
export interface ResolvedNotificationThrottleConfig {
  readonly intervalMs: number;
  readonly minChunkSize: number;
}

/**
 * Validates and normalizes a NotificationThrottleConfig.
 * Returns `undefined` when both thresholds are zero (no-op).
 */
export function resolveNotificationThrottleConfig(
  config: NotificationThrottleConfig | undefined,
): ResolvedNotificationThrottleConfig | undefined {
  if (!config) return undefined;

  const { intervalMs, minChunkSize } = config;
  if (!Number.isFinite(intervalMs) || intervalMs < 0) {
    throw new Error(
      `notificationThrottle.intervalMs must be a non-negative finite number, got ${intervalMs}`,
    );
  }
  if (minChunkSize !== undefined && (!Number.isFinite(minChunkSize) || minChunkSize < 0)) {
    throw new Error(
      `notificationThrottle.minChunkSize must be a non-negative finite number, got ${minChunkSize}`,
    );
  }

  const resolvedMinChunkSize = minChunkSize ?? 0;
  // If both thresholds are zero, throttling is a no-op — the agent will
  // use the immediate (non-throttled) notification path.
  if (intervalMs <= 0 && resolvedMinChunkSize <= 0) return undefined;

  return { intervalMs, minChunkSize: resolvedMinChunkSize };
}

export interface AgentConfig {
  agentId?: string;
  description?: string;
  threadId?: string;
  initialMessages?: Message[];
  initialState?: State;
  debug?: AgentDebugConfig;
  /**
   * Throttle subscriber notifications during streaming.
   * When omitted, every mutation fires a notification immediately.
   */
  notificationThrottle?: NotificationThrottleConfig;
}

export interface HttpAgentConfig extends AgentConfig {
  url: string;
  headers?: Record<string, string>;
}

export type RunAgentParameters = Partial<
  Pick<RunAgentInput, "runId" | "tools" | "context" | "forwardedProps">
>;
