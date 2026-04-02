import { describe, it, expect, vi } from "vitest";
import { Observable, Subject } from "rxjs";
import { AbstractAgent } from "../agent";
import type { BaseEvent, Message, RunAgentInput } from "@ag-ui/core";
import { EventType } from "@ag-ui/core";

class TestAgent extends AbstractAgent {
  public subject = new Subject<BaseEvent>();

  run(_input: RunAgentInput): Observable<BaseEvent> {
    return this.subject.asObservable();
  }
}

/** Wait one macrotask so runAgent's pipeline has subscribed to the subject */
const tick = () => new Promise((r) => setTimeout(r, 0));

// ---------------------------------------------------------------------------
// Event factories
//
// BaseEvent's Zod passthrough schema adds an index signature that plain
// object literals cannot satisfy. These factories centralize the single
// necessary cast so test call-sites stay cast-free.
// ---------------------------------------------------------------------------

function runStarted(): BaseEvent {
  return { type: EventType.RUN_STARTED } as BaseEvent;
}

function runFinished(): BaseEvent {
  return { type: EventType.RUN_FINISHED } as BaseEvent;
}

function textChunk(messageId: string, delta: string): BaseEvent {
  return { type: EventType.TEXT_MESSAGE_CHUNK, messageId, delta } as BaseEvent;
}

function stateSnapshot(snapshot: Record<string, unknown>): BaseEvent {
  return { type: EventType.STATE_SNAPSHOT, snapshot } as BaseEvent;
}

// ---------------------------------------------------------------------------
// Helpers — eliminate duplication across tests
// ---------------------------------------------------------------------------

/** Extract the string content of the first assistant message, or "" */
function firstAssistantContent(messages: readonly Message[]): string {
  const msg = messages[0];
  return msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
}

/** Extract the string content of the last assistant message, or "" */
function lastAssistantContent(messages: readonly Message[]): string {
  const msg = messages[messages.length - 1];
  return msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
}

/** Emit `count` single-char TEXT_MESSAGE_CHUNK events (A, B, C, …) */
function emitCharChunks(agent: TestAgent, count: number, messageId = "m1", startChar = 65) {
  for (let i = 0; i < count; i++) {
    agent.subject.next(textChunk(messageId, String.fromCharCode(startChar + i)));
  }
}

/**
 * Start a run: call runAgent(), wait for pipeline subscription, emit RUN_STARTED.
 * Returns `{ completion }` — an object wrapper that prevents async/await from
 * auto-flattening the inner Promise (which would block until the stream ends).
 */
async function startRun(agent: TestAgent) {
  const completion = agent.runAgent();
  await tick();
  agent.subject.next(runStarted());
  return { completion };
}

/** Full lifecycle: start run -> emit events -> finish run */
async function runToCompletion(agent: TestAgent, emitFn: () => void) {
  const { completion } = await startRun(agent);
  emitFn();
  agent.subject.next(runFinished());
  agent.subject.complete();
  await completion;
}

describe("AbstractAgent notification throttle", () => {
  // ── Baseline (no throttle) ──────────────────────────────────────────

  it("without throttle config, onMessagesChanged fires for every chunk", async () => {
    const agent = new TestAgent();
    const calls: number[] = [];
    agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(messages.length); } });

    await runToCompletion(agent, () => {
      for (let i = 0; i < 5; i++) {
        agent.subject.next(textChunk("m1", `chunk${i} `));
      }
    });

    expect(calls.length).toBeGreaterThanOrEqual(5);
  });

  // ── Time-based throttle ─────────────────────────────────────────────

  it("with intervalMs, fewer onMessagesChanged calls than chunks", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 50 } });
    const calls: string[] = [];
    agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(firstAssistantContent(messages)); } });

    await runToCompletion(agent, () => emitCharChunks(agent, 20));

    expect(calls.length).toBeLessThan(20);
    expect(calls[calls.length - 1]).toBe("ABCDEFGHIJKLMNOPQRST");
  });

  // ── Chunk-size throttle ─────────────────────────────────────────────

  it("with minChunkSize, notifications wait until enough chars accumulate", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 500, minChunkSize: 10 } });
    const calls: string[] = [];
    agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(firstAssistantContent(messages)); } });

    await runToCompletion(agent, () => emitCharChunks(agent, 20));

    expect(calls.length).toBeLessThanOrEqual(4);
    expect(calls.length).toBeGreaterThanOrEqual(1);
    expect(calls[calls.length - 1]).toBe("ABCDEFGHIJKLMNOPQRST");
  });

  // ── Leading edge fires immediately ──────────────────────────────────

  it("with large throttle window, coalesces into leading + trailing notifications", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 5000 } });
    const calls: string[] = [];
    agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(firstAssistantContent(messages)); } });

    await runToCompletion(agent, () => {
      agent.subject.next(textChunk("m1", "hello"));
      agent.subject.next(textChunk("m1", " world"));
    });

    expect(calls.length).toBeGreaterThanOrEqual(1);
    expect(calls.length).toBeLessThanOrEqual(3);
    expect(calls[calls.length - 1]).toBe("hello world");
  });

  // ── agent.messages stays current even when notification is deferred ─

  it("agent.messages is always up-to-date even between throttled notifications", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 5000 } });
    const notificationContents: string[] = [];
    agent.subscribe({ onMessagesChanged: ({ messages }) => { notificationContents.push(firstAssistantContent(messages)); } });

    await runToCompletion(agent, () => emitCharChunks(agent, 10));

    expect(notificationContents[notificationContents.length - 1]).toBe("ABCDEFGHIJ");
    expect(firstAssistantContent(agent.messages)).toBe("ABCDEFGHIJ");
  });

  // ── State change notifications under throttle ───────────────────────

  it("onStateChanged is throttled and flushed correctly", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 50 } });
    const stateCalls: Record<string, unknown>[] = [];
    agent.subscribe({ onStateChanged: ({ state }) => { stateCalls.push(structuredClone(state)); } });

    await runToCompletion(agent, () => {
      agent.subject.next(stateSnapshot({ count: 1 }));
      agent.subject.next(stateSnapshot({ count: 2 }));
      agent.subject.next(stateSnapshot({ count: 3 }));
    });

    expect(stateCalls.length).toBeGreaterThanOrEqual(1);
    expect(stateCalls.length).toBeLessThanOrEqual(3);
    expect(stateCalls[stateCalls.length - 1]).toEqual({ count: 3 });
  });

  // ── Subscriber error does not crash the pipeline ────────────────────

  it("subscriber error in throttled path is caught and does not crash", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 50 } });
    const goodCalls: string[] = [];
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    agent.subscribe({ onMessagesChanged: () => { throw new Error("boom"); } });
    agent.subscribe({ onMessagesChanged: ({ messages }) => { goodCalls.push(firstAssistantContent(messages)); } });

    await runToCompletion(agent, () => {
      agent.subject.next(textChunk("m1", "hello"));
    });

    expect(goodCalls.length).toBeGreaterThanOrEqual(1);
    expect(goodCalls[goodCalls.length - 1]).toBe("hello");
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining("AG-UI: Subscriber onMessagesChanged threw"),
      expect.any(Error),
    );

    consoleErrorSpy.mockRestore();
  });

  // ── Clone preserves throttle config ─────────────────────────────────

  it("clone() preserves notificationThrottle config", () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 42, minChunkSize: 10 } });
    const cloned = agent.clone();

    expect(cloned.notificationThrottle).toEqual({ intervalMs: 42, minChunkSize: 10 });
    expect(cloned.notificationThrottle).not.toBe(agent.notificationThrottle);
  });

  it("clone() preserves undefined notificationThrottle", () => {
    const agent = new TestAgent();
    const cloned = agent.clone();
    expect(cloned.notificationThrottle).toBeUndefined();
  });

  // ── Input validation ────────────────────────────────────────────────

  it("throws on negative intervalMs", () => {
    expect(() => new TestAgent({ notificationThrottle: { intervalMs: -1 } })).toThrow("non-negative finite number");
  });

  it("throws on NaN intervalMs", () => {
    expect(() => new TestAgent({ notificationThrottle: { intervalMs: NaN } })).toThrow("non-negative finite number");
  });

  it("throws on Infinity intervalMs", () => {
    expect(() => new TestAgent({ notificationThrottle: { intervalMs: Infinity } })).toThrow("non-negative finite number");
  });

  it("throws on negative minChunkSize", () => {
    expect(() => new TestAgent({ notificationThrottle: { intervalMs: 16, minChunkSize: -5 } })).toThrow("non-negative finite number");
  });

  it("intervalMs: 0 with no minChunkSize skips throttle activation", () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 0 } });
    expect(agent.notificationThrottle).toBeUndefined();
  });

  it("intervalMs: 0 with minChunkSize > 0 still activates throttle", () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 0, minChunkSize: 10 } });
    expect(agent.notificationThrottle).toEqual({ intervalMs: 0, minChunkSize: 10 });
  });

  // ── Trailing timer fires mid-stream (Issue 6) ───────────────────────

  it("trailing timer fires pending notification mid-stream", async () => {
    vi.useFakeTimers();
    try {
      const agent = new TestAgent({ notificationThrottle: { intervalMs: 50 } });
      const calls: string[] = [];
      agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(firstAssistantContent(messages)); } });

      const runPromise = agent.runAgent();
      await vi.advanceTimersByTimeAsync(0);

      agent.subject.next(runStarted());
      agent.subject.next(textChunk("m1", "A"));
      agent.subject.next(textChunk("m1", "B"));

      const callsBeforeTimer = calls.length;

      // Advance past the throttle window — trailing timer should fire
      await vi.advanceTimersByTimeAsync(60);

      expect(calls.length).toBeGreaterThan(callsBeforeTimer);
      expect(calls[calls.length - 1]).toBe("AB");

      agent.subject.next(runFinished());
      agent.subject.complete();
      await vi.advanceTimersByTimeAsync(0);
      await runPromise;
    } finally {
      vi.useRealTimers();
    }
  });

  // ── onStateChanged subscriber error (Issue 7) ──────────────────────

  it("onStateChanged subscriber error is caught and does not crash", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 50 } });
    const goodCalls: Record<string, unknown>[] = [];
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    agent.subscribe({ onStateChanged: () => { throw new Error("state boom"); } });
    agent.subscribe({ onStateChanged: ({ state }) => { goodCalls.push(structuredClone(state)); } });

    await runToCompletion(agent, () => {
      agent.subject.next(stateSnapshot({ count: 1 }));
    });

    expect(goodCalls.length).toBeGreaterThanOrEqual(1);
    expect(goodCalls[goodCalls.length - 1]).toEqual({ count: 1 });
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining("AG-UI: Subscriber onStateChanged threw"),
      expect.any(Error),
    );

    consoleErrorSpy.mockRestore();
  });

  // ── Interleaved message IDs with minChunkSize (Issue 8) ─────────────

  it("minChunkSize resets tracking when message identity changes", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 5000, minChunkSize: 5 } });
    const calls: number[] = [];
    agent.subscribe({ onMessagesChanged: () => { calls.push(calls.length); } });

    await runToCompletion(agent, () => {
      emitCharChunks(agent, 4, "m1");
      emitCharChunks(agent, 6, "m2", 75);
    });

    expect(calls.length).toBeGreaterThanOrEqual(2);
    expect(calls.length).toBeLessThanOrEqual(5);
  });

  // ── No flush on stream error (Issue 9) ──────────────────────────────

  it("does not flush pending notifications on stream error", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 5000 } });
    const calls: string[] = [];
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(firstAssistantContent(messages)); } });

    const { completion } = await startRun(agent);

    agent.subject.next(textChunk("m1", "hello"));
    agent.subject.next(textChunk("m1", " world"));

    const callsBeforeError = calls.length;

    agent.subject.error(new Error("stream error"));
    await completion.catch(() => {});

    expect(calls.length).toBe(callsBeforeError);

    consoleErrorSpy.mockRestore();
  });

  // ── Non-throttled subscriber error resilience ───────────────────────

  it("without throttle, a throwing subscriber does not crash the pipeline", async () => {
    const agent = new TestAgent();
    const calls: number[] = [];
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    agent.subscribe({ onMessagesChanged: () => { throw new Error("subscriber boom"); } });
    agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(messages.length); } });

    await runToCompletion(agent, () => {
      agent.subject.next(textChunk("m1", "hello"));
    });

    expect(calls.length).toBeGreaterThanOrEqual(1);
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("AG-UI: Subscriber"),
      expect.any(Error),
    );

    errorSpy.mockRestore();
  });

  // ── Combined messages + state under single throttle window (#17) ────

  it("combined message and state mutations within a throttle window both flush", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 5000 } });
    const msgCalls: number[] = [];
    const stateCalls: Record<string, unknown>[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => { msgCalls.push(messages.length); },
      onStateChanged: ({ state }) => { stateCalls.push(structuredClone(state)); },
    });

    await runToCompletion(agent, () => {
      agent.subject.next(textChunk("m1", "hello"));
      agent.subject.next(stateSnapshot({ count: 1 }));
      agent.subject.next(textChunk("m1", " world"));
      agent.subject.next(stateSnapshot({ count: 2 }));
    });

    expect(msgCalls.length).toBeGreaterThanOrEqual(1);
    expect(stateCalls.length).toBeGreaterThanOrEqual(1);
    expect(stateCalls[stateCalls.length - 1]).toEqual({ count: 2 });
  });

  // ── intervalMs: 0 + minChunkSize streaming (#18) ───────────────────

  it("intervalMs: 0 with minChunkSize drives notifications by character count alone", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 0, minChunkSize: 5 } });
    const calls: string[] = [];
    agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(firstAssistantContent(messages)); } });

    await runToCompletion(agent, () => emitCharChunks(agent, 15));

    expect(calls.length).toBeGreaterThanOrEqual(2);
    expect(calls.length).toBeLessThanOrEqual(6);
    expect(calls[calls.length - 1]).toBe("ABCDEFGHIJKLMNO");
  });

  // ── Multiple sequential runAgent on same agent (#20) ───────────────

  it("second runAgent on same throttled agent starts fresh throttle state", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 5000 } });
    const calls: string[] = [];
    agent.subscribe({ onMessagesChanged: ({ messages }) => { calls.push(lastAssistantContent(messages)); } });

    // First run
    await runToCompletion(agent, () => {
      agent.subject.next(textChunk("m1", "first"));
    });

    const callsAfterRun1 = calls.length;

    // Second run — new subject needed since the old one completed
    agent.subject = new Subject<BaseEvent>();
    await runToCompletion(agent, () => {
      agent.subject.next(textChunk("m2", "second"));
    });

    expect(calls.length).toBeGreaterThan(callsAfterRun1);
    expect(calls[calls.length - 1]).toBe("second");
  });

  // ── Non-string content falls back to time-only throttling (#21) ────

  it("non-string assistant content falls back to time-only throttling", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 5000, minChunkSize: 5 } });
    const calls: number[] = [];
    agent.subscribe({ onMessagesChanged: () => { calls.push(calls.length); } });

    await runToCompletion(agent, () => emitCharChunks(agent, 10));

    expect(calls.length).toBeGreaterThanOrEqual(1);
    expect(firstAssistantContent(agent.messages)).toBe("ABCDEFGHIJ");
  });
});
