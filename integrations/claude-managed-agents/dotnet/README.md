# AGUI.ClaudeManagedAgents

Connect an [AG-UI](https://ag-ui.com) frontend to [Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview), Anthropic's hosted agent runtime, from .NET. Each AG-UI thread maps to one managed session. Each run drives one turn of that session and streams the agent's events back as AG-UI events.

## Installation

The library targets `net10.0`, `net9.0`, and `net8.0`, and depends on the [`Anthropic`](https://www.nuget.org/packages/Anthropic) NuGet package (12.34.0 or later, which carries the Managed Agents surface), `AGUI.Abstractions`, and `AGUI.Formatting`.

```bash
dotnet add package AGUI.ClaudeManagedAgents
```

Until the packages ship, reference the projects directly:

```xml
<ProjectReference Include="path/to/integrations/claude-managed-agents/dotnet/src/AGUI.ClaudeManagedAgents/AGUI.ClaudeManagedAgents.csproj" />
```

## Usage

Create a managed agent and an environment once (in the Console, or via the SDK), then map a route onto them in your ASP.NET Core app:

```csharp
using AGUI.ClaudeManagedAgents;

var app = WebApplication.CreateBuilder(args).Build();

var agent = new ManagedAgentsAgent(new ManagedAgentsAgentOptions
{
    AgentId = "agent_...",
    EnvironmentId = "env_...",
});

app.MapManagedAgentsAgent("/chat", agent);   // POST /chat streams AG-UI events over SSE
app.Run();
```

`MapManagedAgentsAgent` deserializes the posted AG-UI `RunAgentInput`, runs one turn, and writes the events as Server-Sent Events. Pass an `ownerId` selector to scope threads by the authenticated caller: `app.MapManagedAgentsAgent("/chat", agent, ctx => ctx.User.Identity?.Name)`.

To drive a run yourself, call `agent.RunAsync(runAgentInput, cancellationToken)`. It returns an `IAsyncEnumerable<BaseEvent>` of AG-UI events you can format however your host needs. The package references the ASP.NET Core shared framework (`Microsoft.AspNetCore.App`) for the endpoint helper, so it targets ASP.NET Core apps.

The Anthropic client reads `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) from the environment. Set `AnthropicClient` to supply your own, or `Client` to swap the whole API surface.

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

Tools passed in `RunAgentInput.Tools` are registered on the session as custom tools. When the agent calls one, the run emits the tool call and finishes, leaving the session parked. The client executes the tool and starts the next run with a `role: "tool"` message carrying `toolCallId`. The adapter forwards it into the session as the tool result and resumes streaming.

### Backend tools

Tools your server executes go in `BackendTools`:

```csharp
var options = new ManagedAgentsAgentOptions
{
    AgentId = agentId,
    EnvironmentId = environmentId,
    BackendTools =
    {
        new ManagedAgentsBackendTool
        {
            Name = "get_weather",
            Description = "Get the weather for a location.",
            Parameters = JsonSerializer.SerializeToElement(new
            {
                type = "object",
                properties = new { location = new { type = "string" } },
            }),
            Handler = input => Task.FromResult("{\"temperature\":21}"),
        },
    },
};
```

The tool call and its result stream to the UI, and the result is returned to the agent.

## Options

| Option | Default | |
| --- | --- | --- |
| `AgentId`, `EnvironmentId` | required | The managed agent and environment behind each session. |
| `AgentVersion` | latest | Pin an agent version. |
| `AnthropicClient` | `new AnthropicClient()` | Bring your own Anthropic SDK client. |
| `Client` | `AnthropicManagedAgentsClient` | Replace the Managed Agents API surface, for example in tests. |
| `SessionStore` | in-memory | Thread↔session mapping. Provide your own to survive restarts. |
| `BackendTools` | `[]` | Server-executed custom tools. |
| `SessionTitle` | `AG-UI thread <id>` | Title for created sessions. |
| `ToolConfirmation` | error | `ToolConfirmationPolicy.Allow`/`Deny` to answer built-in tools whose permission policy asks. |
| `TurnTimeout` | 5 minutes | Interrupt turns that run longer. |
| `StreamDeltas` | `true` | Request text and thinking previews for token streaming. |

## Notes

- The default session store is in-memory: restarting the process starts new sessions. Managed sessions themselves persist server-side.
- Turns are serial per thread. A second run on a busy thread errors.
- Built-in tools (bash, file editing, web) execute inside the managed environment. This adapter shows them for display, so enable them on your agent as usual.
- `RunAsync` and `MapManagedAgentsAgent` accept an optional `ownerId`. Pass the host's authenticated caller identity (never a client-supplied value) so thread↔session mappings are scoped per caller, and one caller cannot resume or evict another caller's session by reusing a thread ID.

## Running the example server

```bash
cd integrations/claude-managed-agents/dotnet/examples/AGUIDojoServer
export ANTHROPIC_API_KEY=sk-ant-...   # or ANTHROPIC_AUTH_TOKEN
dotnet run -- setup   # provisions an environment plus one agent per route (idempotent)
dotnet run            # http://localhost:8026
```

Setup writes the provisioned IDs to `.managed-agents.json` next to the built assembly (gitignored). It reuses existing agents by name and does not modify them: to apply prompt changes from `AgentSpecs.cs`, archive the agent and re-run setup.

## Development

```bash
cd integrations/claude-managed-agents/dotnet
dotnet build
dotnet test
```

The solution file is `AGUI.ClaudeManagedAgents.slnx`. The library and example reference the ag-ui .NET SDK by project reference (`sdks/dotnet/src`) until those packages are published.
