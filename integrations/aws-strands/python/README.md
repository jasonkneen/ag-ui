# AWS Strands Integration for AG-UI

This package exposes a lightweight wrapper that lets any `strands.Agent` speak the AG-UI protocol. It mirrors the developer experience of the other integrations: give us a Strands agent instance, plug it into `StrandsAgent`, and wire it to FastAPI via `create_strands_app` (or `add_strands_fastapi_endpoint`).

## Prerequisites

- Python 3.10+
- `poetry` (recommended) or `pip`
- A model key for the provider `MODEL_PROVIDER` selects. It defaults to
  `openai`, which requires `OPENAI_API_KEY`; `anthropic` and `gemini` need
  `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` instead.

## Quick Start

The `examples/server` package mounts all demo routes behind a single FastAPI app. Run:

```bash
cd integrations/aws-strands/python/examples
poetry install
poetry run python -m server
```

`PORT` selects the port and defaults to 8000. It must be written in plain
decimal digits with no leading zero and no sign, giving a number between 1 and 65535. Anything else is refused at startup, naming the variable and the value,
rather than silently binding somewhere unreachable: `0` binds an arbitrary free
port, and Python would otherwise read `0100` as 100 and `1_0` as 10.

`CORS_ALLOW_ORIGINS` is a comma-separated list of browser origins to allow. It
is applied to the dojo app and to every demo mounted inside it, because both
install CORS middleware and the mounted one answers first.

Entries are matched against the `Origin` header exactly. A trailing slash and
letter case are repaired, since a browser sends neither, but nothing else is
validated: an entry that is not an origin stays in the list and simply matches
nothing. `null`, which a sandboxed iframe or a `file://` page sends, is matched
like any other entry, but it names no site, so its presence disables
credentials for the whole list, exactly as `*` does.

Only an unset or blank value allows every origin. That is the local-development
default, and the server says so once at startup. A value that was written but
holds nothing a browser could send, `/` or `*/` for instance, refuses every
cross-origin request instead of widening to allow all of them, and is reported
separately at startup. Setting the variable is a request to restrict, so a typo
in it must never grant more than was asked for.

It exposes:

| Route                       | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `/agentic-chat`             | Frontend tool demo                             |
| `/agentic-chat-reasoning`   | Reasoning / thinking event streaming           |
| `/agentic-chat-multimodal`  | Multimodal image / document analysis           |
| `/backend-tool-rendering`   | Backend tool rendering demo                    |
| `/shared-state`             | Shared recipe state                            |
| `/agentic-generative-ui`    | Agentic UI with PredictState                   |
| `/human-in-the-loop`        | Frontend proxy tool with halt-after-call       |
| `/interrupt`                | Tool pauses to ask the user for a meeting time |
| `/predictive-state-updates` | Document editor driven by streaming tool args  |
| `/tool-based-generative-ui` | Frontend-rendered tool (`generate_haiku`)      |
| `/multi-agent`              | Strands graph of agents, streamed as steps     |
| `/a2ui-dynamic-schema`      | A2UI surfaces composed on the fly              |
| `/a2ui-fixed-schema`        | A2UI from fixed-layout backend tools           |
| `/a2ui-recovery`            | A2UI validate-and-retry recovery loop          |

This is the easiest way to test multiple flows locally. Each route still follows the pattern described below (Strands agent → wrapper → FastAPI).

## Architecture Overview

The integration has three main layers:

- **StrandsAgent** – wraps `strands.Agent.stream_async`. It translates Strands events into AG-UI events (text chunks, tool calls, PredictState, snapshots, reasoning/thinking, multi-agent steps, etc.).
- **Configuration** – `StrandsAgentConfig` + `ToolBehavior` + `PredictStateMapping` let you describe tool-specific quirks declaratively (skip message snapshots, emit state, stream args, send confirm actions, etc.).
- **Transport helpers** – `create_strands_app` and `add_strands_fastapi_endpoint` expose the agent via SSE. They are thin shells over the shared `ag_ui.encoder.EventEncoder`.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for diagrams and a deeper dive.

## Key Files

| File                            | Description                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `src/ag_ui_strands/agent.py`    | Core wrapper translating Strands streams into AG-UI events                      |
| `src/ag_ui_strands/config.py`   | Config primitives (`StrandsAgentConfig`, `ToolBehavior`, `PredictStateMapping`) |
| `src/ag_ui_strands/endpoint.py` | FastAPI endpoint helper                                                         |
| `src/ag_ui_strands/utils.py`    | `create_strands_app`, multimodal conversion, and `UrlFetchPolicy`               |
| `examples/server/api/*.py`      | Ready-to-run demo apps                                                          |

## Amazon Bedrock AgentCore considerations

If you are planning to deploy your agent into Amazon Bedrock AgentCore (AC), please note that AC expects the following:

- The server is running on port 8080.
- The path `/invocations - POST` is implemented and can be used for interacting with the agent.
- The path `/ping - GET` is implemented and can be used for verifying that the agent is operational and ready to handle requests.

To implement the path mentioned above, you can use the helper function `create_strands_app` and pass the agent interaction path and the ping path as shown below. Pass `origins` too: omitting every CORS option selects the deprecated implicit wildcard and emits a `FutureWarning`.

```python
    create_strands_app(agui_agent, "/invocations", "/ping", origins=["https://app.example"])
```

You can also use the helper functions `add_strands_fastapi_endpoint` and `add_ping` for adding the mentioned paths to a FastAPI app that you are creating separately:

```python
    add_strands_fastapi_endpoint(app, agent, "/invocations")
    add_ping(app, "/ping")
```

## Securing the endpoint

`create_strands_app` remains backward-compatible with earlier releases: when no
CORS option is supplied it installs permissive wildcard CORS and emits a
`FutureWarning`. Choose the intended policy explicitly to silence the warning:

- Prefer an exact browser allowlist, e.g. `create_strands_app(agui_agent, origins=["http://localhost:3000"])`.
- Pass `cors_enabled=False` for same-origin or server-to-server deployments that need no CORS middleware.
- Pass `origins=["*"]` (or `cors_enabled=True`) to explicitly retain wildcard CORS for local development.
- The implicit wildcard fallback will be removed in a future release.
- The agent route has no authentication unless you pass an `auth` dependency:

```python
import os
from fastapi import Header, HTTPException

def require_token(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {os.environ['AGENT_TOKEN']}":
        raise HTTPException(status_code=401, detail="Unauthorized")

app = create_strands_app(
    agui_agent,
    origins=["https://app.example"],
    auth=require_token,
)
```

Pass `origins` to every app you mount as well as to the parent. A mounted app
installs its own CORS middleware and answers first, so one left on the wildcard
default replies `Access-Control-Allow-Origin: *` to an origin the parent would
have refused, and the parent's middleware then adds
`Access-Control-Allow-Credentials: true` on the way out. Preflighted requests
are still refused by the parent, but any route reachable as a simple request,
including `/ping`, is readable by any origin.

The same `auth` argument is accepted by `add_strands_fastapi_endpoint` and is
evaluated before JSON decoding or model validation. The ping endpoint is left
unauthenticated so load balancer and AgentCore health probes keep working.

Agent POST requests must send a JSON-compatible `Content-Type`: either
`application/json` or an `application/*+json` media type. Requests with a
missing or non-JSON `Content-Type` are rejected with HTTP 415 before the agent
runs.

Requests to the AC endpoint must be authenticated. You can configure your agent runtime to accept JWT bearer tokens (via Amazon Cognito) or use SigV4. See [Set up authentication](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui.html) in the AgentCore documentation.

For details on how AgentCore handles AG-UI requests, event streaming, and error formatting, see the [AG-UI protocol contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui-protocol-contract.html).

To deploy, use the [AgentCore Starter Toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit):

```bash
pip install bedrock-agentcore-starter-toolkit
agentcore configure -e my_agui_server.py --protocol AGUI
agentcore deploy
```

For the complete deployment walkthrough, see [Deploy AG-UI servers in AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui.html).

## Request-scoped invocation state

Use `invocation_state_provider` to make trusted server context available to
Strands hooks and tools for one request. The provider may be synchronous or
asynchronous and receives both the FastAPI `Request` and the validated
`RunAgentInput`:

```python
from fastapi import Request
from ag_ui.core import RunAgentInput
from ag_ui_strands import create_strands_app

async def invocation_state(
    request: Request,
    input_data: RunAgentInput,
) -> dict[str, object]:
    return {
        "tenant_id": request.state.tenant_id,
        "run_id": input_data.run_id,
    }

app = create_strands_app(
    agui_agent,
    auth=require_token,
    invocation_state_provider=invocation_state,
)
```

The adapter shallow-copies the returned dictionary before each invocation,
because Strands adds its own runtime entries to that dictionary. Do not source
trusted values from client-controlled `forwarded_props`; derive them from
authenticated request context instead. Custom routes can pass the same state
directly with `agent.run(input_data, invocation_state={...})`.

## Human-in-the-loop (native Strands interrupts)

Python frontend tools explicitly configured with
`ToolBehavior(continue_after_frontend_call=False)` wait in Strands' native
interrupt checkpoint. This is an internal implementation detail; the AG-UI
client contract remains `TOOL_CALL_*` -> successful `RUN_FINISHED` -> an
ordinary `ToolMessage` on the next request. The client does not receive a
frontend-tool interrupt outcome, does not send `resume[]`, and does not receive
a duplicate `TOOL_CALL_RESULT` for its own result.

A waiting frontend tool is not an AG-UI interrupt. An interrupt means the agent
itself paused and is waiting on `resume[]`; a waiting frontend tool is the
ordinary tool-call round trip, and the run still finishes successfully so a
generic interrupt handler does not fire on a tool card it does not own. Native
waiting is only how the adapter parks the call.

Retries are idempotent. Re-sending an answer the checkpoint already holds
verbatim neither resumes Strands nor re-invokes the model, including when a
client replays its full history and repeats an answer alongside a new one. A
_different_ answer for the same call fails with
`FRONTEND_TOOL_RESULT_CONFLICT`.

Strands is the source of truth for active calls, answered calls, partial
responses, mixed checkpoints, and restart recovery. The adapter reads that
checkpoint only to correlate the client's `ToolMessage` by the native Strands
`toolUseId`, which is also the AG-UI `tool_call_id`. Missing, blank, duplicate,
or reused native IDs fail loudly; affected model providers should upgrade to a
Strands/provider version that supplies stable IDs or avoid parallel frontend
calls. Unconfigured tools and explicit `True` retain the legacy placeholder
path. This native frontend-wait bridge is currently Python-specific; it does
not claim TypeScript parity.

Tools that pause with `tool_context.interrupt(...)` are bridged to the AG-UI
interrupt round-trip:

- When a run pauses, it finishes with `RUN_FINISHED` carrying a
  `RunFinishedInterruptOutcome` (`outcome.type == "interrupt"`) and one AG-UI
  `Interrupt` per Strands interrupt. Generic native interrupts preserve the
  Strands name as the AG-UI reason and the free-form Strands reason under
  `metadata.reason`. Tools configured with `ToolBehavior(interrupt_on_call=True)`
  instead emit a `tool_call` approval interrupt with an `approved` response
  schema. Applies to server-executed tools only. For client-provided tools, gate
  execution in the client — define the tool with a `render` that calls `respond`,
  not a `handler` — since the tool runs in the browser and the adapter has already
  finished the public AG-UI run.
- To resume, the client sends the next `RunAgentInput` on the **same
  `thread_id`** with `resume=[ResumeEntry(interrupt_id=..., status="resolved",
payload=...)]`. Strands' resume gate is truthiness-based (`if
interrupt_.response:`), so a falsy `payload` (`None`, `False`, `""`, `0`,
  `[]`, `{}`) would otherwise re-raise the same interrupt and re-run the tool
  body forever. To prevent that, `interrupt()` does **not** return `payload`
  directly — it returns a truthy envelope: `{"response": payload}` on
  resolve, `{"cancelled": True}` on cancel. Destructure it with
  `.get("response")` / `.get("cancelled")`. Adapter-managed
  `interrupt_on_call` approvals are the exception: their
  `{"approved": bool}` payload is passed through directly.
- For generic native interrupts, `status="cancelled"` resumes the tool with
  the sentinel `{"cancelled": True}` (`ag_ui_strands.INTERRUPT_CANCELLED`)
  so it can treat the pause as a denial. An adapter-managed approval receives
  `{"approved": False}` instead.
- **Re-execution on resume:** resuming a paused tool re-runs its body from
  the top — any code before the `interrupt()` call executes again. Guard
  side effects that must not repeat:

  ```python
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

| Scenario                                                                        | Support boundary                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Native-only pause and resume on the same live wrapper, process, and `thread_id` | Supported without a `SessionManager`; the cached per-thread Strands agent is the checkpoint.                                                                                                                                                                                                     |
| Wrapper recreation or cross-process resume                                      | Requires a compatible durable `SessionManager` that restores the same session and stable Strands `agent_id`.                                                                                                                                                                                     |
| Legacy placeholder proxy and native interrupt in the same checkpoint            | Requires `session_id` plus `session_repository.list_messages()` and `session_repository.update_message()`. Without a manager the run emits `INTERRUPT_SESSION_REQUIRED`; without those capabilities it emits `INTERRUPT_SESSION_CAPABILITY_ERROR`. The checkpoint is not advertised or consumed. |
| Explicitly waiting frontend tools, alone or mixed with ordinary interrupts      | Uses the native Strands checkpoint. Frontend answers arrive as `ToolMessage`s; ordinary interrupt answers retain `resume[]`. Partial batches are passed through and remain paused until Strands reports the checkpoint complete.                                                                 |

Submitted resume batches are validated before streaming or reconciliation.
They must contain unique, non-blank, currently open interrupt ids. An
ordinary-only checkpoint still requires every open interrupt in one batch. A
checkpoint containing explicitly waiting frontend tools may be answered
partially; Strands records the supplied responses and remains paused on its
unanswered siblings. Malformed or unopened entries emit
`INTERRUPT_RESUME_ERROR`; incomplete ordinary-only batches emit
`PARTIAL_RESUME`. These failures leave the checkpoint retryable. If
reconciliation fails while a legacy proxy/native interrupt checkpoint is
active, the run emits `INTERRUPT_RECONCILIATION_ERROR` without finishing or
consuming the checkpoint.

When using a `SessionManager`, keep interrupt payloads and tool results
JSON-safe (no raw `bytes`): Strands' `SessionAgent.to_dict()` — unlike
`SessionMessage.to_dict()` — does not base64-encode `bytes` values, so a
`bytes`-bearing interrupt `reason`/`response`/resume `payload`, or a sibling
`ToolResult` in the same turn, raises `TypeError: Object of type bytes is not
JSON serializable` from `FileSessionManager`/`S3SessionManager` and aborts the
run.

## Fetching URL content sources

A user message may carry an image, document or video as a URL rather than
inline data. The adapter fetches those server-side, so every fetch runs under
a `UrlFetchPolicy`. The default refuses everything but `http`/`https`, refuses
any host that resolves outside the public internet (loopback, private,
link-local, including the cloud metadata endpoints), pins the connection to
the address it validated so a second DNS answer cannot redirect it, re-checks
every redirect hop, refuses a redirect that drops TLS, and bounds both one
attachment and everything a single run fetches.

A deployment whose attachments live on a private CDN or behind split DNS opts
in explicitly:

```python
from ag_ui_strands import StrandsAgent, StrandsAgentConfig, UrlFetchPolicy

agent = StrandsAgent(
    strands_agent,
    name="my-agent",
    config=StrandsAgentConfig(
        url_fetch_policy=UrlFetchPolicy(
            allow_private_networks=True,
            max_attachments=20,
            max_total_bytes=100 * 1024 * 1024,
            max_total_seconds=120.0,
        ),
    ),
)
```

Link-local addresses stay blocked under `allow_private_networks`, and
`allowed_schemes` can only be narrowed, never widened: a scheme with no pinned
transport would resolve the host again at connection time.

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
- **Citations**: source passages attached to the assistant message's `metadata` (see below)

## Citations

When you give a model documents and turn citations on, its answer comes back
with the passages it drew from: which document, where in that document, and the
text of the passage itself. That is what lets an interface show "according to
quarterly-report.pdf" next to a claim instead of asking the reader to take the
answer on trust. Bedrock calls these citations. Strands documents them only as
an API reference, and only for its TypeScript SDK, at
[`CitationsBlock`](https://strandsagents.com/docs/api/typescript/CitationsBlock/).
The Python SDK models the same concept in `strands.types.citations`, with the
document and search-result location kinds; some fields on that page exist only
on the TypeScript side, and the table below says which.

The model emits them between the text deltas of the answer, so a citation
arrives in the middle of the message it belongs to. This adapter attaches them
to that message rather than emitting them separately, which is what keeps a
citation joined to the thing it annotates.

### Where they arrive

Under the `citations` key of the assistant message's `metadata`, as a list:

```json
{
  "citations": [
    {
      "title": "quarterly-report.pdf",
      "sourceContent": [{ "text": "revenue grew 12%" }],
      "location": {
        "type": "documentChar",
        "documentIndex": 0,
        "start": 10,
        "end": 26
      },
      "textOffset": 17
    }
  ]
}
```

| Field           | Meaning                                                           | Bedrock |
| --------------- | ----------------------------------------------------------------- | ------- |
| `title`         | Title of the cited source                                         | yes     |
| `sourceContent` | The passage in the source document that supports the answer       | yes     |
| `location`      | Where that passage sits in the source, discriminated by `type`    | yes     |
| `source`        | Source identifier, typically a URL                                | no      |
| `content`       | The generated text the citation supports                          | no      |
| `textOffset`    | UTF-16 code units of this message's text streamed when it arrived | derived |

The `Bedrock` column matters because Strands passes the provider's own citation
through. Bedrock's streaming citation carries `title`, `sourceContent` and
`location` and nothing else, so `source` and `content` are simply absent on that
path. They are in the shape because a provider that does supply them reaches
this key by the same route, and because the TypeScript adapter emits them where
its SDK produces them.

`location` is `{ "type": "documentChar" \| "documentPage" \| "documentChunk",
"documentIndex", "start", "end" }` for document sources,
`{ "type": "searchResult", "searchResultIndex", "start", "end" }` for search
results, and `{ "type": "web", "url", "domain" }` for web ones.

Bedrock names the search-result kind `searchResultLocation` and the Strands
TypeScript SDK renames it to `searchResult`; this adapter applies the same
rename so both bridges agree on it. A kind neither SDK names yet is passed
through here with its own name, and does **not** reach a TypeScript client at
all: that SDK's Bedrock mapper logs an unknown location and drops the citation
with it. The asymmetry is upstream and cannot be normalised away.

A field the provider did not supply is absent rather than empty, and a citation
that names no source is dropped rather than emitted as a bare `textOffset`. The
generated span does not count as naming one, since it is the text being
annotated rather than the thing annotating it. One that will not survive JSON encoding is dropped too, with a
warning: metadata rides an event that is encoded for the stream, and a value
that fails to encode would end the run early.

The key is a plain metadata key, not AG-UI's reserved `ag-ui` one. Metadata is
open by key and user space is yours, so an application already storing something
under `citations` should rename it.

### Where the two adapters agree, and where they do not

For a Bedrock response the Python and TypeScript adapters emit equal citation
objects. That is what the normalisation is for: the TypeScript Strands
SDK coalesces a missing `source` or `title` to `""` and wraps nothing, Python
receives Bedrock's key-wrapped `location` and omits absent fields, and both
adapters converge on the same discriminated, empty-free shape.

They do not agree for every provider. Strands reports the generated span on the
delta rather than on the citation, and only some providers fill it: the
TypeScript SDK's OpenAI Responses adapter supplies `content` and `source`, and
the Python SDK's stream shape has no equivalent field. A provider that supplies
a generated span therefore reaches a TypeScript client with `content` and a
Python client without it.

### How precisely they can be placed

**Message level is the ceiling, and it bounds what a frontend can render.** A
citation locates a span in the _source_ document. It carries no offset into the
answer, and AG-UI has no anchor for a span inside a message, so nothing here can
promise "these words came from that passage".

`textOffset` is the adapter's best effort at closing that gap: it records how
much of the message had been streamed when the citation arrived. Bedrock emits a
citation after the text it supports, which makes the offset the end of the
annotated span in practice, but that is the provider's ordering rather than a
guarantee, so treat a marker placed with it as approximate.

Where a provider reports `content`, that is the generated span itself and is
exact, but no provider reaches this adapter with one: Strands' Python stream
shape has no field for it. A TypeScript client can get it; a Python one cannot.

`textOffset` is counted in **UTF-16 code units**, not characters. The number is
an index into the message text a client holds, and the clients that will slice
with it are browsers, where string indices are UTF-16 units. Both adapters count
the same units, so an answer containing an emoji does not shift the marker on
one side and not the other.

### What a client sees while streaming

The list is republished as it grows, so a client holds a whole prefix at every
point rather than a fragment:

1. A citation arrives and is attached to the next `TEXT_MESSAGE_CONTENT`, so it
   is visible while the answer is still being written.
2. Each publish carries every citation seen so far for that message. Metadata
   merging replaces a key's value rather than appending to it, so the complete
   list is the only correct thing to send.
3. `TEXT_MESSAGE_END` carries the final list, which is how a citation with no
   text after it reaches the client at all.
4. The assistant message inside the following `MESSAGES_SNAPSHOT` carries the
   same list. A snapshot replaces the message a client assembled, so without it
   the citations would vanish the moment one arrived. That also applies to the
   snapshot seeded from `RunAgentInput.messages` at the start of a later turn,
   which is why the rebuild preserves a message's existing metadata.

Citations belong to the message that was open when they arrived. A tool call
closes that message and rotates its id, and the next message starts with none. A
citation that arrives when no message is open has nothing to annotate, so it is
dropped rather than carried onto whatever message comes next, and every drop is
logged: at warning level when it was orphaned, at debug when it arrived after a
tool result already stopped the text stream.

On the multi-agent orchestrator path, each node's citations ride that node's own
message, and there is no `MESSAGES_SNAPSHOT` at all: point 4 above does not
apply there, so the node's `TEXT_MESSAGE_END` is the final carrier.

## Next Steps

- Add an event queue layer (like the ADK middleware) for resumable streams and non-HTTP transports.
- Expand the test suite as new behaviors land.
