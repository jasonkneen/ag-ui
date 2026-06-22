# ag-ui-langgraph

Implementation of the AG-UI protocol for LangGraph.

Provides a complete Python integration for LangGraph agents with the AG-UI protocol, including FastAPI endpoint creation and comprehensive event streaming.

## Installation

```bash
pip install ag-ui-langgraph
```

## Usage

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from fastapi import FastAPI
from my_langgraph_workflow import graph

# Add to FastAPI
app = FastAPI()
add_langgraph_fastapi_endpoint(app, graph, "/agent")
```

## Features

- **Native LangGraph integration** – Direct support for LangGraph workflows and state management
- **FastAPI endpoint creation** – Automatic HTTP endpoint generation with proper event streaming
- **Advanced event handling** – Comprehensive support for all AG-UI events including thinking, tool calls, and state updates
- **Message translation** – Seamless conversion between AG-UI and LangChain message formats

## Resuming via AG-UI standard `resume[]`

When a client uses `RunAgentInput.resume = [ResumeEntry, ...]` instead of
the legacy `forwardedProps.command.resume`, the integration converts the
array into a single `Command(resume=...)` value (LangGraph's resume
channel is per-task, not per-interrupt). The shape your graph receives:

- **Single `resolved` entry** → `interrupt()` returns `entry.payload`
  verbatim. Existing graphs that consumed `Command(resume=<payload>)`
  keep working.
- **Single `cancelled` entry** → `interrupt()` returns the sentinel
  `{"__agui_cancelled__": true, "interrupt_id": "..."}`.
  Your graph should branch on this key.
- **Multiple entries** (parallel interrupts) → `interrupt()` returns
  `{"__agui_resume_map__": { interruptId: {status, payload}, ... }}`.

These sentinels live in the AG-UI integration only — they do **not**
leak into transport-level events.

## Migrating to AG-UI standard interrupts

The LangGraph integration now supports the AG-UI standard interrupt protocol. Key changes:

### Detecting a paused run

`RunFinishedEvent.outcome.type == "interrupt"` is the canonical signal that a run has paused for human input. The `outcome.interrupts` list contains AG-UI `Interrupt` objects with `id`, `reason`, `message`, `tool_call_id`, `response_schema`, `expires_at`, and `metadata` fields. LangGraph-specific data (raw interrupt value, `ns`, `resumable`, `when`) is preserved under `metadata["langgraph"]`.

```python
# New: read interrupts from outcome
if event.type == EventType.RUN_FINISHED and getattr(event, "outcome", None) and event.outcome.type == "interrupt":
    for interrupt in event.outcome.interrupts:
        print(interrupt.id, interrupt.reason, interrupt.message)
```

### Resuming a run

Send `RunAgentInput.resume` (recommended) instead of `forwardedProps.command.resume`:

```python
# New (recommended)
input = RunAgentInput(
    thread_id="t1",
    run_id="r2",
    messages=[],
    resume=[
        ResumeEntry(interrupt_id="int-abc", status="resolved", payload={"approved": True}),
    ],
)

# Old (still works, but deprecated)
input = RunAgentInput(
    thread_id="t1",
    run_id="r2",
    messages=[],
    forwarded_props={"command": {"resume": {"approved": True}}},
)
```

If both `input.resume` and `forwarded_props["command"]["resume"]` are provided, `input.resume` takes precedence and a warning is logged.

### Legacy `on_interrupt` custom event

By default the integration still emits `CustomEvent(name="on_interrupt")` alongside the new `RunFinishedEvent.outcome` for backward compatibility. To suppress the legacy event:

```python
agent = LangGraphAgent(
    name="my-agent",
    graph=graph,
    enable_legacy_on_interrupt_event=False,
)
```

Consumers should migrate to reading `outcome` from `RunFinishedEvent` rather than listening for `CustomEvent(name="on_interrupt")`.

### Capabilities

`LangGraphAgent.get_capabilities()` returns `{"humanInTheLoop": {"supported": True, "interrupts": True, "approveWithEdits": True}}`.

### Customising the HITL bridge (subclass hooks)

If your graph uses a middleware whose interrupt value carries structured payloads (e.g. LangChain's `HumanInTheLoopMiddleware` with `action_requests` / `review_configs`), you can override two protected methods instead of monkey-patching the run loop:

```python
from ag_ui_langgraph import LangGraphAgent
from ag_ui_langgraph.interrupts import lg_interrupt_to_agui
from ag_ui.core import Interrupt as AGUIInterrupt
from langgraph.types import Command

class HITLLangGraphAgent(LangGraphAgent):
    def _interrupts_to_agui(self, lg_interrupts):
        out = []
        for lg in lg_interrupts:
            value = lg.value
            if isinstance(value, dict) and "action_requests" in value:
                out.extend(my_action_requests_to_agui(value))
            else:
                out.append(lg_interrupt_to_agui(lg))
        return out

    def _build_command_from_agui_resume(self, entries, *, open_interrupts=None):
        return Command(
            resume=my_resume_to_decisions(entries, open_interrupts),
        )
```

The base class still handles `STATE_SNAPSHOT` / `MESSAGES_SNAPSHOT` ordering, legacy `CustomEvent(on_interrupt)` emission, the `prepare_stream` short-circuit, and `forwarded_props.command.resume` deprecation — your subclass only needs to care about the HITL-specific translation.

## To run the dojo examples

```bash
cd python/ag_ui_langgraph/examples
poetry install
poetry run dev
```
