/**
 * Dojo example server: one AG-UI endpoint per feature, each backed by a
 * managed agent. Provision the agents first with `pnpm setup:examples`.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=sk-ant-xxx pnpm dev:examples
 */

import http from "node:http";
import { readFileSync } from "node:fs";
import { EventEncoder } from "@ag-ui/encoder";
import type { RunAgentInput } from "@ag-ui/core";
import { ManagedAgentsAgent } from "../src";
import type { BackendCustomTool } from "../src";
import { FEATURE_AGENTS } from "./agents";
import { IDS_PATH, type ProvisionedIds } from "./setup";

const loadIds = (): ProvisionedIds | undefined => {
  try {
    return JSON.parse(readFileSync(IDS_PATH, "utf-8")) as ProvisionedIds;
  } catch {
    console.warn(`No provisioned agents (${IDS_PATH} missing); run \`pnpm setup:examples\`. Serving no routes.`);
    return undefined;
  }
};

const getWeather: BackendCustomTool = {
  name: "get_weather",
  description: "Get the current weather for a location.",
  parameters: {
    type: "object",
    properties: { location: { type: "string", description: "City name" } },
    required: ["location"],
  },
  handler: (input) => {
    const location = (input as { location?: string }).location ?? "somewhere";
    return JSON.stringify({ location, temperature: 21, conditions: "sunny", humidity: 48, windSpeed: 12 });
  },
};

const BACKEND_TOOLS: Record<string, BackendCustomTool[]> = {
  backend_tool_rendering: [getWeather],
};

const buildAgents = (): Record<string, ManagedAgentsAgent> => {
  const ids = loadIds();
  const agents: Record<string, ManagedAgentsAgent> = {};
  if (!ids) return agents;
  const { environmentId, agents: agentIds } = ids;
  for (const spec of FEATURE_AGENTS) {
    const agentId = agentIds[spec.feature];
    if (!agentId) {
      console.warn(`No agent provisioned for ${spec.feature}; skipping. Re-run setup.`);
      continue;
    }
    agents[spec.feature] = new ManagedAgentsAgent({
      agentId,
      environmentId,
      backendTools: BACKEND_TOOLS[spec.feature],
    });
  }
  return agents;
};

const agents = buildAgents();

async function handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "*");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const path = new URL(req.url ?? "/", `http://${req.headers.host}`).pathname.replace(/^\//, "");

  if (req.method === "GET" && path === "health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "healthy", agents: Object.keys(agents) }));
    return;
  }

  const agent = Object.hasOwn(agents, path) ? agents[path] : undefined;
  if (req.method !== "POST" || !agent) {
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not found", availableRoutes: Object.keys(agents) }));
    return;
  }

  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);

  let input: RunAgentInput;
  try {
    input = JSON.parse(Buffer.concat(chunks).toString("utf-8")) as RunAgentInput;
  } catch {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Invalid JSON body" }));
    return;
  }

  const encoder = new EventEncoder({ accept: req.headers.accept ?? "text/event-stream" });
  res.writeHead(200, {
    "Content-Type": encoder.getContentType(),
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const subscription = agent
    .clone()
    .run(input)
    .subscribe({
      next: (event) => res.write(encoder.encode(event)),
      error: () => res.end(),
      complete: () => res.end(),
    });
  res.on("close", () => subscription.unsubscribe());
}

function main() {
  if (!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_AUTH_TOKEN) {
    console.error("Error: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)");
    process.exit(1);
  }
  const port = parseInt(process.env.PORT ?? "8024", 10);
  http.createServer(handleRequest).listen(port, "0.0.0.0", () => {
    console.log(`Claude Managed Agents server running on http://localhost:${port}`);
    for (const name of Object.keys(agents)) console.log(`  POST http://localhost:${port}/${name}`);
    console.log(`  GET  http://localhost:${port}/health`);
  });
}

main();
