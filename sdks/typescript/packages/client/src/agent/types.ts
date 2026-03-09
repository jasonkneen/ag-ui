import { Message, RunAgentInput, State } from "@ag-ui/core";

export interface AgentConfig {
  agentId?: string;
  description?: string;
  threadId?: string;
  initialMessages?: Message[];
  initialState?: State;
  debug?: boolean;
}

/**
 * Override specific endpoint URLs used by `HttpAgent`.
 * Each defaults to `{url}/{name}` when not provided.
 */
export interface HttpAgentEndpoints {
  /** Override the capabilities discovery endpoint. Defaults to `{url}/capabilities`. */
  capabilities?: string;
}

export interface HttpAgentConfig extends AgentConfig {
  url: string;
  headers?: Record<string, string>;
  /** Override specific endpoint URLs used by the agent. */
  endpoints?: HttpAgentEndpoints;
}

export type RunAgentParameters = Partial<
  Pick<RunAgentInput, "runId" | "tools" | "context" | "forwardedProps">
>;
