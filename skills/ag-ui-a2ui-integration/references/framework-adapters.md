# Framework Adapters

Start from the app's existing agent framework. Use the relevant AG-UI adapter
instead of hand-rolling event translation when an adapter exists.

The examples below are common adapter patterns, not the complete AG-UI support
matrix. For any other AG-UI-supported framework, search the AG-UI repo's
`integrations/` directory, the AG-UI docs, and the current CLI source before
choosing an implementation path. Use the framework's own AG-UI package,
endpoint helper, or example when one exists.

## ADK

AG-UI provides Python ADK middleware. For a FastAPI endpoint, use
`ADKAgent` and `add_adk_fastapi_endpoint`.

```python
from fastapi import FastAPI
from ag_ui_adk import ADKAgent, AGUIToolset, add_adk_fastapi_endpoint
from google.adk.agents import Agent

my_agent = Agent(
    name="assistant",
    instruction="You are a helpful assistant.",
    tools=[
        AGUIToolset(),
    ],
)

agent = ADKAgent(
    adk_agent=my_agent,
    app_name="my_app",
    user_id="user123",
)

app = FastAPI()
add_adk_fastapi_endpoint(app, agent, path="/chat")
```

Use the ADK resumability path for human-in-the-loop flows that must pause and
resume around frontend tools.

## LangGraph

AG-UI provides `ag-ui-langgraph` for Python LangGraph apps.

```python
from fastapi import FastAPI
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from my_langgraph_workflow import graph

app = FastAPI()
add_langgraph_fastapi_endpoint(app, graph, "/agent")
```

Keep the LangGraph state model intact. Add A2UI instructions and tools around
the graph rather than replacing graph nodes with AG-UI-specific code.

## Mastra

AG-UI provides `@ag-ui/mastra` for TypeScript Mastra agents.

```ts
import { MastraAgent } from "@ag-ui/mastra";
import { mastra } from "./mastra";

const agent = new MastraAgent({
  agent: mastra.getAgent("weather-agent"),
  resourceId: "user-123",
});

const result = await agent.runAgent({
  messages: [{ role: "user", content: "What's the weather like?" }],
});
```

Wire this into the app's existing AG-UI server route or scaffolded runtime.

## CrewAI

AG-UI provides `ag-ui-crewai` for Python CrewAI flows.

```python
from crewai.flow.flow import Flow, start
from litellm import acompletion
from ag_ui_crewai import (
    add_crewai_flow_fastapi_endpoint,
    copilotkit_stream,
    CopilotKitState,
)
from fastapi import FastAPI

class MyFlow(Flow[CopilotKitState]):
    @start()
    async def chat(self):
        response = await copilotkit_stream(
            await acompletion(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *self.state.messages,
                ],
                tools=self.state.copilotkit.actions,
                stream=True,
            )
        )
        self.state.messages.append(response.choices[0].message)

app = FastAPI()
add_crewai_flow_fastapi_endpoint(app, MyFlow(), "/flow")
```

The CrewAI adapter currently exposes some legacy CopilotKit names. Keep the
source package names exactly as documented by AG-UI.

## Strands

AG-UI provides `@ag-ui/aws-strands` for TypeScript Strands agents.

```ts
import { Agent } from "@strands-agents/sdk";
import { StrandsAgent } from "@ag-ui/aws-strands";
import { createStrandsApp } from "@ag-ui/aws-strands/server";

const strandsAgent = new Agent({
  model: "anthropic.claude-sonnet-4-5-20250929-v1:0",
});

const aguiAgent = new StrandsAgent({ agent: strandsAgent });
const app = await createStrandsApp(aguiAgent, { path: "/invocations" });
app.listen(8080);
```

There is no Strands flag in the current AG-UI CLI source. Use the integration
package and examples directly.

## Other AG-UI-Supported Frameworks

For frameworks not shown above, follow this order:

1. Search `integrations/` for the framework name.
2. Check `sdks/typescript/packages/cli/src/index.ts` for a scaffold flag.
3. Check the AG-UI docs for the framework's package name and endpoint helper.
4. Reuse the closest documented integration pattern, preserving the target
   framework's normal agent lifecycle, state model, and tool conventions.
5. Apply the shared A2UI runtime, renderer, catalog, and verification steps
   from the other references in this skill.

## Custom AG-UI Agents

For a custom backend, keep the protocol stream valid:

- Start each run with `RUN_STARTED`.
- Emit text, tool, state, and A2UI-related events in order.
- End with `RUN_FINISHED` or `RUN_ERROR`.
- Preserve `threadId`, `runId`, message ids, and tool call ids consistently.
- Use the encoder packages where available instead of ad hoc SSE formatting.
