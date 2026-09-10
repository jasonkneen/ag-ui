import { createServer } from "node:http";
import { once } from "node:events";
import { expect, test } from "vitest";
import { firstValueFrom, toArray } from "rxjs";
import { MCPAppsMiddleware } from "../src/index";
import { MockAgent, createRunAgentInput } from "./test-utils";

/** Serve the MCP HTTP protocol and record transport effects on real sockets. */
async function setup() {
  const requests: Array<{
    method: string;
    authorization?: string;
    rpc?: string;
  }> = [];
  const server = createServer(async (request, response) => {
    const entry = {
      method: request.method!,
      authorization: request.headers.authorization,
      rpc: undefined as string | undefined,
    };
    requests.push(entry);
    if (request.method === "DELETE") {
      response.writeHead(204).end();
      return;
    }
    if (request.method === "GET") {
      response.writeHead(405).end();
      return;
    }
    let raw = "";
    for await (const chunk of request) raw += chunk;
    const message = JSON.parse(raw);
    entry.rpc = message.method;
    if (message.id === undefined) {
      response.writeHead(202).end();
      return;
    }
    const result =
      message.method === "initialize"
        ? {
            protocolVersion: message.params.protocolVersion,
            capabilities: { resources: {} },
            serverInfo: { name: "fixture", version: "1" },
          }
        : {
            contents: [
              { uri: "ui://card", text: "Card", mimeType: "text/html+mcp" },
            ],
          };
    response.writeHead(200, {
      "content-type": "application/json",
      "mcp-session-id": "fixture-session",
    });
    response.end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string")
    throw new Error("Fixture did not listen");
  const middleware = new MCPAppsMiddleware({
    mcpServers: [
      {
        type: "http",
        url: `http://127.0.0.1:${address.port}`,
        serverId: "cards",
        headers: { Authorization: "Bearer fixture-token" },
      },
    ],
  });
  const agent = new MockAgent();
  return {
    requests,
    agent,
    run: (method: string) =>
      firstValueFrom(
        middleware
          .run(
            createRunAgentInput({
              forwardedProps: {
                __proxiedMCPRequest: {
                  serverId: "cards",
                  method,
                  params: { uri: "ui://card" },
                },
              },
            }),
            agent,
          )
          .pipe(toArray()),
      ),
    teardown: async () => {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}

test("blocked proxy methods do not initialize an MCP session", async () => {
  const { run, requests, agent, teardown } = await setup();
  try {
    const events = await run("tools/list");
    expect(events.at(-1)).toMatchObject({
      result: { error: expect.stringContaining("not allowed") },
    });
    expect(requests).toEqual([]);
    expect(agent.runCalls).toEqual([]);
  } finally {
    await teardown();
  }
});

test("successful HTTP proxy requests delete their authenticated MCP session", async () => {
  const { run, requests, teardown } = await setup();
  try {
    const events = await run("resources/read");
    expect(events.at(-1)).toMatchObject({
      result: { contents: [{ text: "Card" }] },
    });
    expect(
      requests.filter((request) => request.method === "DELETE"),
    ).toHaveLength(1);
    expect(
      requests.every(
        (request) => request.authorization === "Bearer fixture-token",
      ),
    ).toBe(true);
  } finally {
    await teardown();
  }
});
