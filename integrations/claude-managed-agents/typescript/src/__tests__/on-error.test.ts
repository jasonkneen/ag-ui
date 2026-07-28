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

    expect(reported).toHaveLength(1);
    expect(reported[0]!.context.operation).toBe("interrupt");
    expect((reported[0]!.error as Error).message).toBe("interrupt rejected");
    expect(reported[0]!.context.sessionId).toBe("sesn_1");
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
