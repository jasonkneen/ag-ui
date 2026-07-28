import { EventType } from "@ag-ui/client";
import type { BaseEvent } from "@ag-ui/client";
import { describe, expect, it } from "vitest";
import { ManagedAgentsAgent } from "../agent";
import type { ManagedAgentsErrorContext } from "../types";
import { createFakeClient } from "./fake-client";

const baseInput = () =>
  ({
    threadId: "thread_1",
    runId: "run_1",
    state: null,
    messages: [{ id: "u1", role: "user", content: "Hello" }],
    tools: [],
    context: [],
    forwardedProps: {},
  }) as never;

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("onError", () => {
  it("reports an interrupt that could not be posted", async () => {
    const reported: { error: unknown; context: ManagedAgentsErrorContext }[] = [];
    // The stream hangs so the run is still open when the client leaves; the
    // interrupt is then the send that fails.
    const fake = createFakeClient({
      streams: [[new Promise<void>(() => {})]],
      sendResults: [undefined, new Error("interrupt rejected")],
    });
    const agent = new ManagedAgentsAgent({
      managedAgentId: "agent_1",
      environmentId: "env_1",
      client: fake.client,
      onError: (error, context) => reported.push({ error, context }),
    });

    const subscription = agent.run(baseInput()).subscribe({ next: () => {} });
    await settle();
    subscription.unsubscribe();
    await settle();

    // The abandoned run is reported too (see below); the interrupt failure is
    // the one this test is about.
    const interrupt = reported.find((entry) => entry.context.operation === "interrupt");
    expect(interrupt).toBeDefined();
    expect((interrupt!.error as Error).message).toBe("interrupt rejected");
    expect(interrupt!.context.sessionId).toBe("sesn_1");
  });

  it("reports a backend tool that fails after the run walked away", async () => {
    // The handler is abandoned when the client leaves; its later rejection has
    // nowhere to go, and consuming it silently is the only reason a failing
    // backend tool could go unnoticed entirely.
    const reported: ManagedAgentsErrorContext[] = [];
    let rejectHandler!: (error: Error) => void;
    const fake = createFakeClient({ streams: [[{ type: "agent.custom_tool_use", id: "ctu_1", name: "slow_tool", input: {} }, new Promise<void>(() => {})]] });
    const agent = new ManagedAgentsAgent({
      managedAgentId: "agent_1",
      environmentId: "env_1",
      client: fake.client,
      backendTools: [
        {
          name: "slow_tool",
          description: "",
          parameters: {},
          handler: () => new Promise<string>((_resolve, reject) => (rejectHandler = reject)),
        },
      ],
      onError: (_error, context) => reported.push(context),
    });

    const subscription = agent.run(baseInput()).subscribe({ next: () => {} });
    await settle();
    subscription.unsubscribe();
    await settle();

    // The handler only fails after the run is gone.
    rejectHandler(new Error("tool blew up late"));
    await settle();

    expect(reported.map((context) => context.operation)).toContain("abandoned_backend_tool");
  });

  it("reports a run that failed after the client disconnected", async () => {
    // Nothing can be emitted to a client that left, so the hook is the only
    // place this failure can show up.
    const reported: { error: unknown; context: ManagedAgentsErrorContext }[] = [];
    const fake = createFakeClient({ streams: [[new Promise<void>(() => {})]] });
    // Fail the session-create so the run rejects the moment it is abandoned.
    const originalStream = fake.client.beta.sessions.events.stream;
    fake.client.beta.sessions.events.stream = (async (...args: unknown[]) => {
      const stream = await (originalStream as (...a: unknown[]) => Promise<{ [Symbol.asyncIterator]: unknown }>)(...args);
      return {
        ...stream,
        async *[Symbol.asyncIterator]() {
          await settle();
          throw new Error("stream died");
        },
      };
    }) as typeof originalStream;
    const agent = new ManagedAgentsAgent({
      managedAgentId: "agent_1",
      environmentId: "env_1",
      client: fake.client,
      onError: (error, context) => reported.push({ error, context }),
    });

    const subscription = agent.run(baseInput()).subscribe({ next: () => {} });
    await settle();
    subscription.unsubscribe();
    await settle();
    await settle();

    expect(reported.map((entry) => entry.context.operation)).toContain("run_after_disconnect");
  });

  it("a throwing hook does not break the run", async () => {
    const fake = createFakeClient({
      streams: [[{ type: "session.status_idle", id: "idle_1", stop_reason: { type: "end_turn" } }]],
    });
    const agent = new ManagedAgentsAgent({
      managedAgentId: "agent_1",
      environmentId: "env_1",
      client: fake.client,
      onError: () => {
        throw new Error("hook is broken");
      },
    });

    const events: BaseEvent[] = [];
    await new Promise<void>((resolve) =>
      agent.run(baseInput()).subscribe({ next: (event) => events.push(event), complete: () => resolve() }),
    );
    expect(events.at(-1)?.type).toBe(EventType.RUN_FINISHED);
  });
});
