/**
 * Agentic chat with reasoning example.
 *
 * This example shows how to create an agentic chat adapter with
 * extended thinking/reasoning enabled using Claude Haiku 4.5.
 */

import { ClaudeAgentAdapter } from "@ag-ui/claude-agent-sdk";
import { DEFAULT_DISALLOWED_TOOLS } from "./constants";

/**
 * Create adapter for agentic chat with reasoning enabled.
 *
 * Uses maxThinkingTokens to enable extended thinking so Claude
 * can reason through complex questions step by step.
 */
export function createAgenticChatReasoningAdapter(): ClaudeAgentAdapter {
  return new ClaudeAgentAdapter({
    agentId: "agentic_chat_reasoning",
    description: "Chat assistant with extended thinking/reasoning",
    model: "sonnet",
    systemPrompt: "You are a helpful assistant. Think step by step when answering questions.",
    thinking: { type: "enabled", budgetTokens: 10000 },
    includePartialMessages: true,
    disallowedTools: [...DEFAULT_DISALLOWED_TOOLS],
  });
}
