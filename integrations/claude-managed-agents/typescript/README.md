# @ag-ui/claude-managed-agents

Connect an [AG-UI](https://ag-ui.com) frontend to [Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview), Anthropic's hosted agent runtime. Each AG-UI thread maps to one managed session. Each run drives one turn of that session and streams the agent's events back as AG-UI events.

## Installation

```bash
npm install @ag-ui/claude-managed-agents @anthropic-ai/sdk
```

## Usage

Create a managed agent and an environment once (in the Console, or via the SDK), then wrap them:

```typescript
import { ManagedAgentsAgent } from "@ag-ui/claude-managed-agents";

const agent = new ManagedAgentsAgent({
  agentId: "agent_...",
  environmentId: "env_...",
});

agent.run({ threadId, runId, messages, tools, state, context, forwardedProps }).subscribe({
  next: (event) => console.log(event.type),
});
```

The Anthropic client reads `ANTHROPIC_API_KEY` from the environment. Pass `client` to supply your own.

## What it does

| Managed Agents | AG-UI |
| --- | --- |
| `agent.message` (with `event_delta` previews) | `TEXT_MESSAGE_START` / `CONTENT` / `END` |
| `agent.thinking` | `REASONING_START` / `REASONING_END` |
| `agent.tool_use`, `agent.mcp_tool_use` + results | `TOOL_CALL_*` + `TOOL_CALL_RESULT` (server-executed, display only) |
| `agent.custom_tool_use` for a frontend tool | `TOOL_CALL_*`, then the run ends so the client can run the tool |
| `agent.custom_tool_use` for a backend tool | `TOOL_CALL_*` + `TOOL_CALL_RESULT`, and the handler's result is posted back |
| `session.error` (terminal) | `RUN_ERROR` with the error type as `code` |
| `session.status_idle` (`end_turn`) | `RUN_FINISHED` |

### Frontend tools (human-in-the-loop)

Tools passed in `RunAgentInput.tools` are registered on the session as custom tools. When the agent calls one, the run emits the tool call and finishes, leaving the session parked. The client executes the tool and starts the next run with a `role: "tool"` message carrying `toolCallId`. The adapter forwards it into the session as the tool result and resumes streaming.

### Backend tools

Tools your server executes go in `backendTools`:

```typescript
new ManagedAgentsAgent({
  agentId,
  environmentId,
  backendTools: [
    {
      name: "get_weather",
      description: "Get the weather for a location.",
      parameters: { type: "object", properties: { location: { type: "string" } } },
      handler: async (input) => JSON.stringify({ temperature: 21 }),
    },
  ],
});
```

The tool call and its result stream to the UI, and the result is returned to the agent.

## Options

| Option | Default | |
| --- | --- | --- |
| `agentId`, `environmentId` | required | The managed agent and environment behind each session. |
| `agentVersion` | latest | Pin an agent version. |
| `client` | `new Anthropic()` | Bring your own Anthropic client. |
| `sessionStore` | in-memory | Thread↔session mapping. Provide your own to survive restarts. |
| `scope` | none | Partitions thread↔session state, for example by authenticated user. |
| `backendTools` | `[]` | Server-executed custom tools. |
| `sessionTitle` | `AG-UI thread <id>` | Title for created sessions. |
| `toolConfirmation` | error | `"allow"`/`"deny"` to answer built-in tools whose permission policy asks. |
| `turnTimeoutMs` | 300000 | Interrupt turns that run longer. |
| `streamDeltas` | `true` | Request text/thinking previews for token streaming. |

## Notes

- Thread IDs come from the client. Without `scope`, any caller that knows a thread ID resumes that thread's session, so put the endpoint behind your own authentication and construct one agent (or scope) per caller.
- The default session store is in-memory: restarting the process starts new sessions. Managed sessions themselves persist server-side.
- Turns are serial per thread. A second run on a busy thread errors.
- Built-in tools (bash, file editing, web) execute inside the managed environment. This adapter surfaces them for display, so enable them on your agent as usual.

## Running the examples

```bash
cd integrations/claude-managed-agents/typescript
pnpm install
export ANTHROPIC_API_KEY=sk-ant-...   # or ANTHROPIC_AUTH_TOKEN
pnpm setup:examples   # provisions an environment + one agent per Dojo feature (idempotent)
pnpm dev:examples     # http://localhost:8024
```

Setup writes the provisioned IDs to `examples/.managed-agents.json` (gitignored). It reuses existing agents by name and does not modify them: to apply prompt changes from `examples/agents.ts`, archive the agent and re-run setup.
