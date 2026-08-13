# AWS Strands Integration for AG-UI

This package exposes a lightweight wrapper that lets any `strands.Agent` speak the AG-UI protocol. It mirrors the developer experience of the other integrations: give us a Strands agent instance, plug it into `StrandsAgent`, and wire it to FastAPI via `create_strands_app` (or `add_strands_fastapi_endpoint`).

## Prerequisites

- Python 3.10+
- `poetry` (recommended) or `pip`
- A Strands-compatible model key (e.g., `GOOGLE_API_KEY` for Gemini)

## Quick Start

The `examples/server/__main__.py` module mounts all demo routes behind a single FastAPI app. Run:

```bash
cd integrations/aws-strands/python/examples
poetry install
poetry run python -m server
```

It exposes:

| Route                     | Description                  |
| ------------------------- | ---------------------------- |
| `/agentic-chat`           | Frontend tool demo           |
| `/backend-tool-rendering` | Backend tool rendering demo  |
| `/shared-state`           | Shared recipe state          |
| `/agentic-generative-ui`  | Agentic UI with PredictState |

This is the easiest way to test multiple flows locally. Each route still follows the pattern described below (Strands agent → wrapper → FastAPI).

## Architecture Overview

The integration has three main layers:

- **StrandsAgent** – wraps `strands.Agent.stream_async`. It translates Strands events into AG-UI events (text chunks, tool calls, PredictState, snapshots, reasoning/thinking, multi-agent steps, etc.).
- **Configuration** – `StrandsAgentConfig` + `ToolBehavior` + `PredictStateMapping` let you describe tool-specific quirks declaratively (skip message snapshots, emit state, stream args, send confirm actions, etc.).
- **Transport helpers** – `create_strands_app` and `add_strands_fastapi_endpoint` expose the agent via SSE. They are thin shells over the shared `ag_ui.encoder.EventEncoder`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for diagrams and a deeper dive.

## Key Files

| File                            | Description                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `src/ag_ui_strands/agent.py`    | Core wrapper translating Strands streams into AG-UI events                      |
| `src/ag_ui_strands/config.py`   | Config primitives (`StrandsAgentConfig`, `ToolBehavior`, `PredictStateMapping`) |
| `src/ag_ui_strands/endpoint.py` | FastAPI endpoint helper                                                         |
| `examples/server/api/*.py`      | Ready-to-run demo apps                                                          |

## Amazon Bedrock AgentCore considerations

If you are planning to deploy your agent into Amazon Bedrock AgentCore (AC), please note that AC expects the following:

- The server is running on port 8080.
- The path `/invocations - POST` is implemented and can be used for interacting with the agent.
- The path `/ping - GET` is implemented and can be used for verifying that the agent is operational and ready to handle requests.

To implement the path mentioned above, you can use the helper function `create_strands_app` and pass the agent interaction path and the ping path as shown below:

```python
    create_strands_app(agui_agent, "/invocations", "/ping")
```

You can also use the helper functions `add_strands_fastapi_endpoint` and `add_ping` for adding the mentioned paths to a FastAPI app that you are creating separately:

```python
    add_strands_fastapi_endpoint(app, agent, "/invocations")
    add_ping(app, "/ping")
```

Requests to the AC endpoint must be authenticated. You can configure your agent runtime to accept JWT bearer tokens (via Amazon Cognito) or use SigV4. See [Set up authentication](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui.html) in the AgentCore documentation.

For details on how AgentCore handles AG-UI requests, event streaming, and error formatting, see the [AG-UI protocol contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui-protocol-contract.html).

To deploy, use the [AgentCore Starter Toolkit](https://github.com/awslabs/bedrock-agentcore-starter-toolkit):

```bash
pip install bedrock-agentcore-starter-toolkit
agentcore configure -e my_agui_server.py --protocol AGUI
agentcore deploy
```

For the complete deployment walkthrough, see [Deploy AG-UI servers in AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui.html).

## Human-in-the-loop (native Strands interrupts)

Tools that pause with `tool_context.interrupt(...)` are bridged to the AG-UI
interrupt round-trip:

- When a run pauses, it finishes with `RUN_FINISHED` carrying a
  `RunFinishedInterruptOutcome` (`outcome.type == "interrupt"`) and one AG-UI
  `Interrupt` per reported Strands interrupt. Native interrupts use the AG-UI
  reason `tool_call`; the Strands name and free-form reason are preserved in
  JSON-safe form as `metadata.strands_name` and `metadata.strands_reason`.
  JSON-native values remain unchanged, while `bytes` use Strands' deterministic
  base64 marker (`{"__bytes_encoded__": true, "data": "..."}`).
- To resume, the client sends the next `RunAgentInput` on the same `thread_id`
  with `resume=[ResumeEntry(interrupt_id=..., status="resolved", payload=...)]`.
  Strands' resume gate is truthiness-based (`if interrupt_.response:`), so a
  falsy `payload` (`None`, `False`, `""`, `0`,
  `[]`, `{}`) would otherwise re-raise the same interrupt and re-run the tool
  body forever. To prevent that, `interrupt()` does **not** return `payload`
  directly — it returns a truthy envelope: `{"response": payload}` on
  resolve, `{"cancelled": True}` on cancel. Destructure it with
  `.get("response")` / `.get("cancelled")`.
- `status="cancelled"` resumes the tool with the sentinel
  `{"cancelled": True}` (`ag_ui_strands.INTERRUPT_CANCELLED`) so it can
  treat the pause as a denial.
- **Re-execution on resume:** resuming a paused tool re-runs its body from
  the top — any code before the `interrupt()` call executes again. Guard
  side effects that must not repeat:

  ```python
  from strands import ToolContext, tool


  @tool(context=True)
  def charge_card(tool_context: ToolContext, amount: float) -> str:
      # Unsafe: re-runs (and re-charges) on every resume.
      charge(amount)
      envelope = tool_context.interrupt("confirm_charge", reason={"amount": amount})
      return "cancelled" if envelope.get("cancelled") or not envelope.get("response") else "charged"


  @tool(context=True)
  def charge_card(tool_context: ToolContext, amount: float) -> str:
      # Safe: side effect happens only after the pause resolves.
      envelope = tool_context.interrupt("confirm_charge", reason={"amount": amount})
      if envelope.get("cancelled") or not envelope.get("response"):
          return "cancelled"
      charge(amount)
      return "charged"
  ```

### Persistence and proxy-tool boundaries

| Scenario                                                                      | Support boundary                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native interrupts in the same live wrapper instance, process, and `thread_id` | Supported without a `SessionManager`: the cached per-thread agent retains the live interrupt checkpoint and tool metadata.                                                                                                                                                                                                                                                                                     |
| Wrapper/agent recreation in the same process or host                          | Can resume through `StrandsAgentConfig.session_manager_provider` only when the returned `SessionManager` restores the Strands checkpoint and metadata for the same thread/session and Strands `agent_id`. Implementing the `SessionManager` interface alone does not guarantee restoration.                                                                                                                |
| Cross-process or replica resume                                               | Requires the same AG-UI `thread_id`, stable Strands session and `agent_id` identities, and a compatible durable session repository shared by every process or replica. The provider must return managers connected to that same repository and session; merely passing an arbitrary per-thread `SessionManager` instance is not sufficient.                                                                  |
| Frontend proxy tools only                                                     | Uses the legacy client handoff and normal AG-UI tool-call wire flow; it is not a canonical interrupt.                                                                                                                                                                                                                                                                                                          |
| Frontend proxy and native interrupt in the same turn                          | Requires repository-backed reconciliation: the configured manager must expose `session_id` and `session_repository` with `list_messages()` and `update_message()`. Without a manager, the run emits one `RUN_ERROR` with code `INTERRUPT_SESSION_REQUIRED`; a manager without that capability emits `INTERRUPT_SESSION_CAPABILITY_ERROR`. The run does not advertise or continue the interrupt in either case. |

Every durable resume attempt must therefore recreate the Strands agent with the
same `agent_id` used when the interrupt was stored. The adapter default,
`agent_id="default"`, is stable when callers do not override it. If you set a
custom `agent_id` on the template Strands agent, reuse that exact value for each
recreated wrapper, process, and replica; otherwise the session repository looks
under a different agent record and cannot restore the interrupt checkpoint.

While a native interrupt is active, any failure to reconcile a frontend proxy
result emits `RUN_ERROR` with code `INTERRUPT_RECONCILIATION_ERROR` and stops
that run. The interrupt and reconciliation metadata remain intact so the client
can retry.

This bridge does not promise canonical all-open interrupt batching, automatic
event replay, end-to-end idempotency, or execution of frontend proxy tools as
part of native-interrupt semantics. Proxy calls remain ordinary AG-UI tool
calls even when they share a turn with a native interrupt. Because the native
tool body re-executes on resume, make any pre-interrupt side effects idempotent
or move them after the interrupt, as shown above.

When using a `SessionManager`, keep interrupt payloads and tool results
JSON-safe (no raw `bytes`): Strands' `SessionAgent.to_dict()` — unlike
`SessionMessage.to_dict()` — does not base64-encode `bytes` values, so a
`bytes`-bearing interrupt `reason`/`response`/resume `payload`, or a sibling
`ToolResult` in the same turn, raises `TypeError: Object of type bytes is not
JSON serializable` from `FileSessionManager`/`S3SessionManager` and aborts the
run.

## Supported AG-UI Events

The integration supports the following AG-UI event families:

- **Lifecycle**: `RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`
- **Text streaming**: `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END`
- **Reasoning**: `REASONING_*` events for models with extended thinking
- **Tool calls**: `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, `TOOL_CALL_RESULT`
- **State management**: `STATE_SNAPSHOT`
- **Multi-agent**: `STEP_STARTED`, `STEP_FINISHED`, and `MultiAgentHandoff` custom events
- **Generative UI**: `PredictState` custom events for optimistic UI updates
- **Multimodal**: Image, document, and video content in user messages (converted to Strands ContentBlock format)

## Next Steps

- Add an event queue layer (like the ADK middleware) for resumable streams and non-HTTP transports.
- Expand the test suite as new behaviors land.
