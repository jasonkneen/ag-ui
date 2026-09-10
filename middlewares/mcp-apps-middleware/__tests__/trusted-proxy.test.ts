import { createServer } from "node:http";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { once } from "node:events";
import { expect, test, vi } from "vitest";
import { firstValueFrom, toArray } from "rxjs";
import { MCPAppsMiddleware, getServerHash } from "../src/index";
import {
  MockAgent,
  createRunAgentInput,
  createRunStartedEvent,
  createRunFinishedEvent,
  createToolCallStartEvent,
  createToolCallArgsEvent,
  createToolCallEndEvent,
} from "./test-utils";

/** Serve the MCP HTTP protocol and record transport effects on real sockets. */
async function setup(
  sharedEndpoint = false,
  stallDelete = false,
  redirect = false,
  rejectAuth = false,
  metadata: "legacy" | "nested" | "both" = "legacy",
) {
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
    if (rejectAuth) {
      response.writeHead(401).end("private-auth-diagnostic");
      return;
    }
    if (redirect && request.url !== "/capture") {
      response.writeHead(307, { location: "/capture" }).end();
      return;
    }
    if (request.method === "DELETE") {
      if (stallDelete) return;
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
            capabilities: { resources: {}, tools: {} },
            serverInfo: { name: "fixture", version: "1" },
          }
        : message.method === "tools/list"
          ? {
              tools: [
                {
                  name: "card",
                  description: "Card",
                  inputSchema: { type: "object", properties: {} },
                  _meta:
                    metadata === "legacy"
                      ? { "ui/resourceUri": "ui://card" }
                      : {
                          ui: { resourceUri: "ui://card" },
                          ...(metadata === "both"
                            ? { "ui/resourceUri": "ui://legacy" }
                            : {}),
                        },
                },
              ],
            }
          : message.method === "tools/call"
            ? { content: [{ type: "text", text: "Card result" }] }
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
  const config = {
    discoveryFailureMode: "throw" as const,
    mcpServers: [
      {
        type: "http" as const,
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
  };
  const middleware = new MCPAppsMiddleware(config);
  const agent = new MockAgent();
  return {
    requests,
    agent,
    discover: () =>
      firstValueFrom(
        middleware.run(createRunAgentInput(), agent).pipe(toArray()),
      ),
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

test("an unresponsive DELETE cannot hold a completed proxy result", async () => {
  const { run, requests, teardown } = await setup(false, true);
  let deadline: ReturnType<typeof setTimeout> | undefined;
  try {
    const events = await Promise.race([
      run("resources/read"),
      new Promise<never>((_, reject) => {
        deadline = setTimeout(
          () => reject(new Error("session cleanup did not finish")),
          4500,
        );
      }),
    ]);
    expect(events.at(-1)).toMatchObject({
      result: { contents: [{ text: "Card" }] },
    });
    expect(
      requests.filter((request) => request.method === "DELETE"),
    ).toHaveLength(1);
  } finally {
    clearTimeout(deadline);
    await teardown();
  }
}, 6000);

test("MCP redirects cannot forward configured credentials", async () => {
  const { run, requests, teardown } = await setup(false, false, true);
  try {
    await run("resources/read");
    expect(requests).toHaveLength(1);
  } finally {
    await teardown();
  }
});

test("strict discovery failures stop the agent without exposing upstream diagnostics", async () => {
  const { discover, agent, teardown } = await setup(false, false, false, true);
  const log = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    await expect(discover()).rejects.toThrow("MCP tool discovery failed");
    expect(agent.runCalls).toEqual([]);
    expect(JSON.stringify(log.mock.calls)).not.toContain(
      "private-auth-diagnostic",
    );
  } finally {
    log.mockRestore();
    await teardown();
  }
});

test("proxy errors do not expose upstream diagnostics", async () => {
  const { run, teardown } = await setup(false, false, false, true);
  try {
    const events = await run("resources/read");
    expect(events.at(-1)).toMatchObject({
      result: { error: "Error: MCP request failed" },
    });
    expect(JSON.stringify(events)).not.toContain("private-auth-diagnostic");
  } finally {
    await teardown();
  }
});

test("legacy SSE reentry keeps trusted authentication on GET and POST", async () => {
  const requests: Array<{ method?: string; authorization?: string }> = [];
  const sdkServer = new Server(
    { name: "legacy-fixture", version: "1" },
    { capabilities: {} },
  );
  let transport: SSEServerTransport | undefined;
  const server = createServer(async (request, response) => {
    requests.push({
      method: request.method,
      authorization: request.headers.authorization,
    });
    if (request.headers.authorization !== "Bearer legacy-secret") {
      response.writeHead(401).end();
      return;
    }
    if (request.method === "GET") {
      transport = new SSEServerTransport("/messages", response);
      await sdkServer.connect(transport);
    } else if (transport) {
      await transport.handlePostMessage(request, response);
    } else {
      response.writeHead(404).end();
    }
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  try {
    const address = server.address();
    if (!address || typeof address === "string")
      throw new Error("No fixture address");
    const middleware = new MCPAppsMiddleware({
      mcpServers: [
        {
          type: "sse",
          url: `http://127.0.0.1:${address.port}/sse`,
          serverId: "legacy",
          headers: { Authorization: "Bearer legacy-secret" },
        },
      ],
    });
    const agent = new MockAgent();
    const events = await firstValueFrom(
      middleware
        .run(
          createRunAgentInput({
            forwardedProps: {
              __proxiedMCPRequest: { serverId: "legacy", method: "ping" },
            },
          }),
          agent,
        )
        .pipe(toArray()),
    );
    expect(events.at(-1)).toMatchObject({ type: "RUN_FINISHED", result: {} });
    expect(agent.runCalls).toEqual([]);
    expect(requests.some((request) => request.method === "GET")).toBe(true);
    expect(requests.some((request) => request.method === "POST")).toBe(true);
    expect(
      requests.every(
        (request) => request.authorization === "Bearer legacy-secret",
      ),
    ).toBe(true);
  } finally {
    await sdkServer.close();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("authenticated discovery and tool execution delete both sessions", async () => {
  const { discover, agent, requests, teardown } = await setup();
  agent.setEvents([
    createRunStartedEvent(),
    createToolCallStartEvent("call", "card"),
    createToolCallArgsEvent("call", "{}"),
    createToolCallEndEvent("call"),
    createRunFinishedEvent(),
  ]);
  try {
    const events = await discover();
    expect(agent.runCalls[0].tools.map((tool) => tool.name)).toContain("card");
    expect(events.some((event) => event.type === "ACTIVITY_SNAPSHOT")).toBe(
      true,
    );
    expect(events.at(-1)).toMatchObject({ type: "RUN_FINISHED" });
    expect(
      requests.filter((request) => request.method === "DELETE"),
    ).toHaveLength(2);
    expect(
      requests.every(
        (request) => request.authorization === "Bearer fixture-token",
      ),
    ).toBe(true);
  } finally {
    await teardown();
  }
});

test.each(["nested", "both"] as const)(
  "discovery uses current MCP metadata (%s)",
  async (metadata) => {
    const { discover, agent, teardown } = await setup(
      false,
      false,
      false,
      false,
      metadata,
    );
    agent.setEvents([
      createRunStartedEvent(),
      createToolCallStartEvent("call", "card"),
      createToolCallArgsEvent("call", "{}"),
      createToolCallEndEvent("call"),
      createRunFinishedEvent(),
    ]);
    try {
      const events = await discover();
      expect(agent.runCalls[0].tools.map((tool) => tool.name)).toContain(
        "card",
      );
      expect(
        events.find((event) => event.type === "ACTIVITY_SNAPSHOT"),
      ).toMatchObject({ content: { resourceUri: "ui://card" } });
    } finally {
      await teardown();
    }
  },
);
