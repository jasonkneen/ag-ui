import { createServer } from "node:http";
import { once } from "node:events";
import { expect, test } from "vitest";
import { firstValueFrom, toArray } from "rxjs";
import { MCPAppsMiddleware, getServerHash } from "../src/index";
import { MockAgent, createRunAgentInput } from "./test-utils";

/** Serve the MCP HTTP protocol and record transport effects on real sockets. */
async function setup(sharedEndpoint = false) {
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
  const url = `http://127.0.0.1:${address.port}`;
  const middleware = new MCPAppsMiddleware({
    mcpServers: [
      {
        type: "http",
        url,
        serverId: "cards",
        headers: { Authorization: "Bearer fixture-token" },
      },
      ...(sharedEndpoint
        ? [
            {
              type: "http" as const,
              url,
              serverId: "other",
              headers: { Authorization: "Bearer other-token" },
            },
          ]
        : []),
    ],
  });
  const agent = new MockAgent();
  return {
    requests,
    agent,
    run: (
      method: string,
      serverId: string | undefined = "cards",
      hashOnly = false,
    ) =>
      firstValueFrom(
        middleware
          .run(
            createRunAgentInput({
              forwardedProps: {
                __proxiedMCPRequest: {
                  serverId: hashOnly ? undefined : serverId,
                  serverHash: getServerHash({ type: "http", url }),
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

test.each(["http", "sse"] as const)(
  "%s server hashes do not expose a credential checksum",
  (type) => {
    const publicServer = { type, url: "https://mcp.example.test" };
    expect(
      getServerHash({
        ...publicServer,
        headers: { Authorization: "guessable-token" },
      }),
    ).toBe(getServerHash(publicServer));
  },
);

test("servers sharing a public endpoint require distinct server IDs", () => {
  expect(
    () =>
      new MCPAppsMiddleware({
        mcpServers: [
          {
            type: "http",
            url: "https://mcp.example.test",
            headers: { Authorization: "first" },
          },
          {
            type: "http",
            url: "https://mcp.example.test",
            headers: { Authorization: "second" },
          },
        ],
      }),
  ).toThrow("distinct serverId");
});

test("explicit IDs select the right credentials on a shared endpoint", async () => {
  const { run, requests, teardown } = await setup(true);
  try {
    const events = await run("resources/read", "other");
    expect(events.at(-1)).toMatchObject({
      result: { contents: [{ text: "Card" }] },
    });
    expect(requests.length).toBeGreaterThan(0);
    expect(
      requests.every(
        (request) => request.authorization === "Bearer other-token",
      ),
    ).toBe(true);
  } finally {
    await teardown();
  }
});

test("hash-only requests cannot select an ambiguous credential scope", async () => {
  const { run, requests, teardown } = await setup(true);
  try {
    const events = await run("resources/read", undefined, true);
    expect(events.at(-1)).toMatchObject({
      result: { error: expect.stringContaining("Unknown server") },
    });
    expect(requests).toEqual([]);
  } finally {
    await teardown();
  }
});

test("duplicate explicit server IDs are rejected", () => {
  expect(
    () =>
      new MCPAppsMiddleware({
        mcpServers: [
          { type: "http", url: "https://one.example.test", serverId: "cards" },
          { type: "http", url: "https://two.example.test", serverId: "cards" },
        ],
      }),
  ).toThrow("distinct serverId");
});
