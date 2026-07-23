# ag-ui-claude-managed-agents

Connect an [AG-UI](https://ag-ui.com) frontend to [Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview), Anthropic's hosted agent runtime. Each AG-UI thread maps to one managed session. Each run drives one turn of that session and streams the agent's events back as AG-UI events.

## Installation

```bash
pip install ag-ui-claude-managed-agents
```

## Usage

Create a managed agent and an environment once (in the Console, or via the SDK), then wrap them:

```python
from fastapi import FastAPI
from ag_ui_claude_managed_agents import ManagedAgentsAgent, add_managed_agents_fastapi_endpoint

app = FastAPI()

agent = ManagedAgentsAgent(agent_id="agent_...", environment_id="env_...")
add_managed_agents_fastapi_endpoint(app=app, agent=agent, path="/agent")
```

The endpoint helper accepts a POST with a `RunAgentInput` body and streams the encoded AG-UI events. Without FastAPI, iterate the events yourself:

```python
async for event in agent.run(run_input):
    print(event.type)
```

The Anthropic client reads `ANTHROPIC_API_KEY` from the environment. Pass `client` to supply your own `AsyncAnthropic`.

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

### Frontend tools (human in the loop)

Tools passed in `RunAgentInput.tools` are registered on the session as custom tools. When the agent calls one, the run emits the tool call and finishes, leaving the session parked. The client executes the tool and starts the next run with a `role: "tool"` message carrying `toolCallId`. The adapter forwards it into the session as the tool result and resumes streaming.

### Backend tools

Tools your server executes go in `backend_tools`:

```python
from ag_ui_claude_managed_agents import BackendTool, ManagedAgentsAgent

agent = ManagedAgentsAgent(
    agent_id=agent_id,
    environment_id=environment_id,
    backend_tools=[
        BackendTool(
            name="get_weather",
            description="Get the weather for a location.",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            handler=lambda tool_input: json.dumps({"temperature": 21}),
        ),
    ],
)
```

The handler may be a plain function or a coroutine. The tool call and its result stream to the UI, and the result is returned to the agent.

## Options

| Option | Default | |
| --- | --- | --- |
| `agent_id`, `environment_id` | required | The managed agent and environment behind each session. |
| `agent_version` | latest | Pin an agent version. |
| `client` | `AsyncAnthropic()` | Bring your own Anthropic client. |
| `session_store` | in-memory | Thread to session mapping. Provide your own to survive restarts. |
| `backend_tools` | `[]` | Server-executed custom tools. |
| `session_title` | `AG-UI thread <id>` | Callable returning the title for created sessions. |
| `tool_confirmation` | error | `"allow"`/`"deny"` to answer built-in tools whose permission policy asks. |
| `turn_timeout_s` | 300 | Interrupt turns that run longer. |
| `stream_deltas` | `True` | Request text/thinking previews for token streaming. |

## Notes

- The default session store is in-memory: restarting the process starts new sessions. Managed sessions themselves persist server-side.
- Turns are serial per thread. A second run on a busy thread errors.
- Built-in tools (bash, file editing, web) execute inside the managed environment. This adapter surfaces them for display, so enable them on your agent as usual.
- Sessions are keyed by the AG-UI `thread_id`, which is client-supplied, and whoever supplies a thread id resumes its session. Pass `scope` (for example the authenticated user) to partition thread state per caller. Authenticate the endpoint and scope thread ids to the caller in production (for example with a `session_store` that keys records by user).

## Running the examples

```bash
cd integrations/claude-managed-agents/python/examples
uv sync
export ANTHROPIC_API_KEY=sk-ant-...   # or ANTHROPIC_AUTH_TOKEN
uv run python setup.py               # provisions an environment plus one agent per Dojo feature (idempotent)
uv run dev                           # http://localhost:8025
```

Setup writes the provisioned IDs to `examples/.managed-agents.json` (gitignored). It reuses existing agents by name and does not modify them. To apply prompt changes from `examples/agents.py`, archive the agent and re-run setup.
