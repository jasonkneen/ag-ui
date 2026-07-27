import type { RunAgentInput, Tool } from "@ag-ui/client";
import { lastValueFrom, toArray } from "rxjs";
import { describe, expect, it } from "vitest";
import { ManagedAgentsAgent } from "../agent";
import { InMemorySessionStore } from "../sessions";
import { createFakeClient } from "./fake-client";

const idleEndTurn = {
  type: "session.status_idle",
  id: "idle_1",
  stop_reason: { type: "end_turn" },
};
const baseAgentTool = {
  type: "agent_toolset_20260401",
  configs: [],
  default_config: {},
};

const tool = (name: string, description: string, properties: object = {}): Tool => ({
  name,
  description,
  parameters: { type: "object", properties },
});

const input = (runId: string, messageId: string, tools: Tool[]): RunAgentInput => ({
  threadId: "thread_1",
  runId,
  state: {},
  messages: [
    {
      id: messageId,
      role: "user",
      content: messageId === "u1" ? "Hello" : "Follow-up",
    },
  ],
  tools,
  context: [],
  forwardedProps: {},
});

const runToolTransition = async (initialTools: Tool[], nextTools: Tool[]) => {
  const fake = createFakeClient({
    streams: [[idleEndTurn], [idleEndTurn]],
    agentTools: [baseAgentTool],
  });
  const store = new InMemorySessionStore();
  const agent = new ManagedAgentsAgent({
    managedAgentId: "agent_1",
    environmentId: "env_1",
    client: fake.client,
    sessionStore: store,
  });

  await lastValueFrom(agent.run(input("run_1", "u1", initialTools)).pipe(toArray()));
  await lastValueFrom(agent.run(input("run_2", "u2", nextTools)).pipe(toArray()));
  return fake;
};

const customTool = (name: string, description: string, properties: object = {}) => ({
  type: "custom",
  name,
  description,
  input_schema: { type: "object", properties, required: [] },
});

describe("dynamic frontend tools", () => {
  it("removes tools that are absent from the next run", async () => {
    const showChart = tool("show_chart", "Render a chart");
    const fake = await runToolTransition([showChart, tool("export_csv", "Export a CSV")], [showChart]);

    expect(fake.spies.update).toHaveBeenCalledWith("sesn_1", {
      agent: {
        tools: [baseAgentTool, customTool("show_chart", "Render a chart")],
      },
    });
  });

  it("clears frontend tools when the next run has none", async () => {
    const fake = await runToolTransition([tool("show_chart", "Render a chart")], []);

    expect(fake.spies.update).toHaveBeenCalledWith("sesn_1", {
      agent: { tools: [baseAgentTool] },
    });
  });

  it("updates a same-named tool when its definition changes", async () => {
    const fake = await runToolTransition(
      [tool("show_chart", "Render a chart", { title: { type: "string" } })],
      [
        tool("show_chart", "Render a visualization", {
          series: { type: "array" },
        }),
      ],
    );

    expect(fake.spies.update).toHaveBeenCalledWith("sesn_1", {
      agent: {
        tools: [
          baseAgentTool,
          customTool("show_chart", "Render a visualization", {
            series: { type: "array" },
          }),
        ],
      },
    });
  });

  it("does not update the session when tool definitions are unchanged", async () => {
    const tools = [tool("show_chart", "Render a chart", { title: { type: "string" } })];
    const fake = await runToolTransition(tools, tools);

    expect(fake.spies.update).not.toHaveBeenCalled();
  });
});
