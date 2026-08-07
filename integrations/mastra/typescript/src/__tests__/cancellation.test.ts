import { describe, it, expect } from "vitest";
import { EventType, type BaseEvent } from "@ag-ui/client";
import { FakeMemory, makeInput } from "./helpers";
import { MastraAgent } from "../mastra";

// ---------------------------------------------------------------------------
// Regression tests for #2288: unsubscribing from the run() Observable must
// propagate cancellation into the underlying Mastra stream. Before the fix the
// teardown was `() => {}`, so an aborted run kept generating (and billing)
// tokens to completion.
//
// Each fake below hands out a gate the test opens only AFTER unsubscribing, so
// the stream is provably still mid-flight when teardown fires.
// ---------------------------------------------------------------------------

function deferred() {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 20));

const RESUME_INPUT = makeInput({
  messages: [{ id: "1", role: "user", content: "Hi" }] as any,
  forwardedProps: {
    command: {
      resume: { approved: true },
      interruptEvent: { toolCallId: "call-1", runId: "mastra-run-1" },
    },
  },
});

const STREAM_INPUT = makeInput({
  messages: [{ id: "1", role: "user", content: "Hi" }] as any,
});

/** Subscribes, waits for the first event, unsubscribes, then opens the gate. */
async function runUntilFirstEventThenUnsubscribe(
  agent: MastraAgent,
  input = STREAM_INPUT,
  gate?: { release: () => void },
) {
  const events: BaseEvent[] = [];
  const firstChunk = deferred();

  const subscription = agent.run(input).subscribe({
    next: (event) => {
      events.push(event);
      if (event.type === EventType.TEXT_MESSAGE_CHUNK) firstChunk.release();
    },
    error: () => firstChunk.release(),
    complete: () => firstChunk.release(),
  });

  await firstChunk.promise;
  subscription.unsubscribe();
  const countAtUnsubscribe = events.length;

  gate?.release();
  await tick();

  return { events, countAtUnsubscribe };
}

function chunksBeforeAndAfter(gate: Promise<void>) {
  return async function* () {
    yield { type: "text-delta", payload: { text: "first" } };
    await gate;
    yield { type: "text-delta", payload: { text: "second" } };
    yield { type: "finish", payload: {} };
  };
}

describe("run() cancellation propagation (#2288)", () => {
  describe("local agent stream()", () => {
    it("aborts the signal forwarded to agent.stream() when unsubscribed", async () => {
      const gate = deferred();
      let capturedOpts: any = null;

      const fakeAgent = {
        memory: new FakeMemory(),
        async getMemory() {
          return this.memory;
        },
        async listTools() {
          return {};
        },
        async stream(_messages: any, opts: any) {
          capturedOpts = opts;
          return { fullStream: chunksBeforeAndAfter(gate.promise)() };
        },
      };

      const agent = new MastraAgent({
        agentId: "test-agent",
        agent: fakeAgent as any,
        resourceId: "resource-1",
      });

      const { events, countAtUnsubscribe } =
        await runUntilFirstEventThenUnsubscribe(agent, STREAM_INPUT, gate);

      expect(capturedOpts?.abortSignal).toBeInstanceOf(AbortSignal);
      expect(capturedOpts.abortSignal.aborted).toBe(true);
      expect(events).toHaveLength(countAtUnsubscribe);
    });
  });

  describe("remote agent stream()", () => {
    it("aborts the signal forwarded to remote agent.stream() when unsubscribed", async () => {
      const gate = deferred();
      let capturedOpts: any = null;

      const fakeAgent = {
        async stream(_messages: any, opts: any) {
          capturedOpts = opts;
          return {
            processDataStream: async ({
              onChunk,
            }: {
              onChunk: (chunk: any) => Promise<void>;
            }) => {
              await onChunk({ type: "text-delta", payload: { text: "first" } });
              await gate.promise;
              await onChunk({
                type: "text-delta",
                payload: { text: "second" },
              });
              await onChunk({ type: "finish", payload: {} });
            },
          };
        },
      };

      const agent = new MastraAgent({
        agentId: "test-agent",
        agent: fakeAgent as any,
        resourceId: "resource-1",
      });

      const { events, countAtUnsubscribe } =
        await runUntilFirstEventThenUnsubscribe(agent, STREAM_INPUT, gate);

      expect(capturedOpts?.abortSignal).toBeInstanceOf(AbortSignal);
      expect(capturedOpts.abortSignal.aborted).toBe(true);
      expect(events).toHaveLength(countAtUnsubscribe);
    });
  });

  describe("local agent resumeStream()", () => {
    it("aborts the signal forwarded via resume options when unsubscribed", async () => {
      const gate = deferred();
      let capturedOpts: any = null;

      const fakeAgent = {
        memory: new FakeMemory(),
        async getMemory() {
          return this.memory;
        },
        async listTools() {
          return {};
        },
        async stream() {
          return { fullStream: (async function* () {})() };
        },
        async resumeStream(_resumeData: any, opts: any) {
          capturedOpts = opts;
          return { fullStream: chunksBeforeAndAfter(gate.promise)() };
        },
      };

      const agent = new MastraAgent({
        agentId: "test-agent",
        agent: fakeAgent as any,
        resourceId: "resource-1",
      });

      const { events, countAtUnsubscribe } =
        await runUntilFirstEventThenUnsubscribe(agent, RESUME_INPUT, gate);

      expect(capturedOpts?.abortSignal).toBeInstanceOf(AbortSignal);
      expect(capturedOpts.abortSignal.aborted).toBe(true);
      expect(events).toHaveLength(countAtUnsubscribe);
    });
  });

  describe("remote agent resumeStream()", () => {
    it("aborts the signal forwarded via resume options when unsubscribed", async () => {
      const gate = deferred();
      let capturedOpts: any = null;

      const fakeAgent = {
        async stream() {
          return { processDataStream: async () => {} };
        },
        async resumeStream(_resumeData: any, opts: any) {
          capturedOpts = opts;
          return {
            processDataStream: async ({
              onChunk,
            }: {
              onChunk: (chunk: any) => Promise<void>;
            }) => {
              await onChunk({ type: "text-delta", payload: { text: "first" } });
              await gate.promise;
              await onChunk({
                type: "text-delta",
                payload: { text: "second" },
              });
              await onChunk({ type: "finish", payload: {} });
            },
          };
        },
      };

      const agent = new MastraAgent({
        agentId: "test-agent",
        agent: fakeAgent as any,
        resourceId: "resource-1",
      });

      const { events, countAtUnsubscribe } =
        await runUntilFirstEventThenUnsubscribe(agent, RESUME_INPUT, gate);

      expect(capturedOpts?.abortSignal).toBeInstanceOf(AbortSignal);
      expect(capturedOpts.abortSignal.aborted).toBe(true);
      expect(events).toHaveLength(countAtUnsubscribe);
    });
  });

  describe("abortRun()", () => {
    it("aborts the in-flight run's signal", async () => {
      const gate = deferred();
      let capturedOpts: any = null;

      const fakeAgent = {
        memory: new FakeMemory(),
        async getMemory() {
          return this.memory;
        },
        async listTools() {
          return {};
        },
        async stream(_messages: any, opts: any) {
          capturedOpts = opts;
          return { fullStream: chunksBeforeAndAfter(gate.promise)() };
        },
      };

      const agent = new MastraAgent({
        agentId: "test-agent",
        agent: fakeAgent as any,
        resourceId: "resource-1",
      });

      const firstChunk = deferred();
      const subscription = agent.run(STREAM_INPUT).subscribe({
        next: (event) => {
          if (event.type === EventType.TEXT_MESSAGE_CHUNK) firstChunk.release();
        },
        error: () => firstChunk.release(),
        complete: () => firstChunk.release(),
      });

      await firstChunk.promise;
      agent.abortRun();

      expect(capturedOpts?.abortSignal?.aborted).toBe(true);

      gate.release();
      subscription.unsubscribe();
      await tick();
    });
  });
});
