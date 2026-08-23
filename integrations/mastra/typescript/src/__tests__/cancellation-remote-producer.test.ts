import { describe, it, expect, vi, afterEach } from "vitest";
import { EventType, type BaseEvent } from "@ag-ui/client";

// ---------------------------------------------------------------------------
// #2288, remote half: cancelling a remote run must stop the PRODUCER, not just
// our consumption of it.
//
// @mastra/client-js Omits `abortSignal` from its stream params and reads the
// fetch signal from construction-time `ClientOptions.abortSignal`, so the
// bridge binds each run's signal to a per-run client (see remoteAgentForRun).
// That client passes the signal to fetch and tears down processDataStream on
// abort, which is what actually stops the server-side run.
//
// The client is mocked here so the assertions can be about the contract the
// bridge relies on — "the signal reaches client construction, and production
// stops once it fires" — without a network. The mock is kept in its own file so
// it cannot leak into the rest of the cancellation suite.
// ---------------------------------------------------------------------------

const constructorOptions: Array<Record<string, any>> = [];
const producer = { delivered: 0, stoppedEarly: false };

vi.mock("@mastra/client-js", () => {
  class FakeMastraClient {
    readonly options: Record<string, any>;

    constructor(options: Record<string, any>) {
      this.options = options;
      constructorOptions.push(options);
    }

    getAgent() {
      const signal: AbortSignal | undefined = this.options.abortSignal;
      return {
        async listTools() {
          return {};
        },
        async stream() {
          return {
            processDataStream: async ({
              onChunk,
            }: {
              onChunk: (chunk: any) => Promise<void>;
            }) => {
              producer.delivered++;
              await onChunk({
                type: "text-delta",
                payload: { text: "first" },
              });

              // A real client aborts the fetch here, so the server stops
              // producing and the loop ends. Modelled by refusing to emit any
              // further chunk once the run's signal has fired.
              for (let i = 0; i < 10; i++) {
                if (signal?.aborted) {
                  producer.stoppedEarly = true;
                  return;
                }
                producer.delivered++;
                await onChunk({
                  type: "text-delta",
                  payload: { text: `more-${i}` },
                });
              }
              producer.delivered++;
              await onChunk({ type: "finish", payload: {} });
            },
          };
        },
      };
    }
  }

  return { MastraClient: FakeMastraClient };
});

const { MastraClient } = await import("@mastra/client-js");
const { MastraAgent } = await import("../mastra");
const { makeInput } = await import("./helpers");

const tick = () => new Promise((resolve) => setTimeout(resolve, 20));

const STREAM_INPUT = makeInput({
  messages: [{ id: "1", role: "user", content: "Hi" }] as any,
});

afterEach(() => {
  constructorOptions.length = 0;
  producer.delivered = 0;
  producer.stoppedEarly = false;
  vi.restoreAllMocks();
});

describe("remote run cancellation reaches the producer (#2288)", () => {
  function remoteAgent() {
    const client = new MastraClient({ baseUrl: "http://localhost:4111" });
    return new MastraAgent({
      agentId: "test-agent",
      agent: client.getAgent("test-agent") as any,
      resourceId: "resource-1",
      remoteClient: client as any,
    });
  }

  it("binds the run's abort signal to a per-run client", async () => {
    const agent = remoteAgent();
    const events: BaseEvent[] = [];
    const firstChunk = { release: () => {} } as { release: () => void };
    const gotFirst = new Promise<void>((resolve) => {
      firstChunk.release = resolve;
    });

    const subscription = agent.run(STREAM_INPUT).subscribe({
      next: (event) => {
        events.push(event);
        if (event.type === EventType.TEXT_MESSAGE_CHUNK) firstChunk.release();
      },
      error: () => firstChunk.release(),
      complete: () => firstChunk.release(),
    });

    await gotFirst;

    // One client for the agent handle the test built, one per-run client that
    // carries this run's signal.
    const perRun = constructorOptions.filter((o) => o.abortSignal);
    expect(perRun).toHaveLength(1);
    expect(perRun[0].abortSignal).toBeInstanceOf(AbortSignal);
    expect(perRun[0].abortSignal.aborted).toBe(false);
    // The rest of the client config has to survive the clone, or a per-run
    // client would talk to the wrong server.
    expect(perRun[0].baseUrl).toBe("http://localhost:4111");

    subscription.unsubscribe();
    await tick();

    expect(perRun[0].abortSignal.aborted).toBe(true);
  });

  it("stops upstream production once the run is cancelled", async () => {
    const agent = remoteAgent();
    const firstChunk = { release: () => {} } as { release: () => void };
    const gotFirst = new Promise<void>((resolve) => {
      firstChunk.release = resolve;
    });

    const subscription = agent.run(STREAM_INPUT).subscribe({
      next: (event) => {
        if (event.type === EventType.TEXT_MESSAGE_CHUNK) firstChunk.release();
      },
      error: () => firstChunk.release(),
      complete: () => firstChunk.release(),
    });

    await gotFirst;
    subscription.unsubscribe();
    await tick();

    // The point of the fix: the producer stopped instead of running to
    // completion. Before it, the signal never reached the client and all 12
    // chunks were delivered (and billed) with only our consumption silenced.
    expect(producer.stoppedEarly).toBe(true);
    expect(producer.delivered).toBeLessThan(12);
  });

  it("settles the Observable on abortRun() without an unsubscribe", async () => {
    const agent = remoteAgent();
    const firstChunk = { release: () => {} } as { release: () => void };
    const gotFirst = new Promise<void>((resolve) => {
      firstChunk.release = resolve;
    });
    let outcome: "complete" | "error" | null = null;
    const settled = new Promise<void>((resolve) => {
      agent.run(STREAM_INPUT).subscribe({
        next: (event) => {
          if (event.type === EventType.TEXT_MESSAGE_CHUNK) firstChunk.release();
        },
        error: () => {
          outcome = "error";
          resolve();
        },
        complete: () => {
          outcome = "complete";
          resolve();
        },
      });
    });

    await gotFirst;
    agent.abortRun();

    await Promise.race([
      settled,
      new Promise<void>((_, reject) =>
        setTimeout(() => reject(new Error("run() never settled")), 1000),
      ),
    ]);

    expect(outcome).toBe("complete");
  });
});
