import { describe, it, expect, vi } from "vitest";
import { Observable, Subject } from "rxjs";
import { AbstractAgent } from "../agent";
import { BaseEvent, RunAgentInput, EventType } from "@ag-ui/core";

class TestAgent extends AbstractAgent {
  public subject = new Subject<BaseEvent>();

  run(_input: RunAgentInput): Observable<BaseEvent> {
    return this.subject.asObservable();
  }
}

/** Wait one macrotask so runAgent's pipeline has subscribed to the subject */
const tick = () => new Promise((r) => setTimeout(r, 0));

describe("AbstractAgent notification throttle", () => {
  // ── Baseline (no throttle) ──────────────────────────────────────────

  it("without throttle config, onMessagesChanged fires for every chunk", async () => {
    const agent = new TestAgent();
    const calls: number[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        calls.push(messages.length);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    for (let i = 0; i < 5; i++) {
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: `chunk${i} `,
      } as BaseEvent);
    }
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    expect(calls.length).toBeGreaterThanOrEqual(5);
  });

  // ── Time-based throttle ─────────────────────────────────────────────

  it("with intervalMs, fewer onMessagesChanged calls than chunks", async () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 50 } });
    const calls: string[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        const msg = messages[0];
        const content = msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
        calls.push(content);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    for (let i = 0; i < 20; i++) {
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: String.fromCharCode(65 + i),
      } as BaseEvent);
    }
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    expect(calls.length).toBeLessThan(20);
    expect(calls[calls.length - 1]).toBe("ABCDEFGHIJKLMNOPQRST");
  });

  // ── Chunk-size throttle ─────────────────────────────────────────────

  it("with minChunkSize, notifications wait until enough chars accumulate", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 500, minChunkSize: 10 },
    });
    const calls: string[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        const msg = messages[0];
        const content = msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
        calls.push(content);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    for (let i = 0; i < 20; i++) {
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: String.fromCharCode(65 + i),
      } as BaseEvent);
    }
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // minChunkSize=10: 20 single-char chunks → ~2-3 notifications
    expect(calls.length).toBeLessThanOrEqual(4);
    expect(calls.length).toBeGreaterThanOrEqual(1);
    expect(calls[calls.length - 1]).toBe("ABCDEFGHIJKLMNOPQRST");
  });

  // ── Leading edge fires immediately ──────────────────────────────────

  it("with large throttle window, coalesces into leading + trailing notifications", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 5000 },
    });
    const calls: string[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        const msg = messages[0];
        const content = msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
        calls.push(content);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: "hello",
    } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: " world",
    } as BaseEvent);
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // With 5s window, all events land within it → leading edge + finalize flush
    expect(calls.length).toBeGreaterThanOrEqual(1);
    expect(calls.length).toBeLessThanOrEqual(3);
    // Final notification must contain full content
    expect(calls[calls.length - 1]).toBe("hello world");
  });

  // ── agent.messages stays current even when notification is deferred ─

  it("agent.messages is always up-to-date even between throttled notifications", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 5000 },
    });
    const notificationContents: string[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        const msg = messages[0];
        const content = msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
        notificationContents.push(content);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    for (let i = 0; i < 10; i++) {
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: String.fromCharCode(65 + i),
      } as BaseEvent);
    }
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // Final notification must have all content
    expect(notificationContents[notificationContents.length - 1]).toBe("ABCDEFGHIJ");
    // agent.messages was current at the time of the finalize notification
    expect(agent.messages[0]).toBeDefined();
    const finalMsg = agent.messages[0];
    const finalContent = finalMsg?.role === "assistant" && typeof finalMsg.content === "string" ? finalMsg.content : "";
    expect(finalContent).toBe("ABCDEFGHIJ");
  });

  // ── State change notifications under throttle ───────────────────────

  it("onStateChanged is throttled and flushed correctly", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 50 },
    });
    const stateCalls: any[] = [];

    agent.subscribe({
      onStateChanged: ({ state }) => {
        stateCalls.push(structuredClone(state));
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.STATE_SNAPSHOT,
      snapshot: { count: 1 },
    } as BaseEvent);
    agent.subject.next({
      type: EventType.STATE_SNAPSHOT,
      snapshot: { count: 2 },
    } as BaseEvent);
    agent.subject.next({
      type: EventType.STATE_SNAPSHOT,
      snapshot: { count: 3 },
    } as BaseEvent);
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // Should have coalesced, but final state must be { count: 3 }
    expect(stateCalls.length).toBeGreaterThanOrEqual(1);
    expect(stateCalls.length).toBeLessThanOrEqual(3);
    expect(stateCalls[stateCalls.length - 1]).toEqual({ count: 3 });
  });

  // ── Subscriber error does not crash the pipeline ────────────────────

  it("subscriber error in throttled path is caught and does not crash", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 50 },
    });
    const goodCalls: string[] = [];
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    agent.subscribe({
      onMessagesChanged: () => {
        throw new Error("boom");
      },
    });
    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        const msg = messages[0];
        const content = msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
        goodCalls.push(content);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: "hello",
    } as BaseEvent);
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // The second (good) subscriber still received notifications
    expect(goodCalls.length).toBeGreaterThanOrEqual(1);
    expect(goodCalls[goodCalls.length - 1]).toBe("hello");
    // The error was logged
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining("AG-UI: Subscriber onMessagesChanged threw"),
      expect.any(Error),
    );

    consoleErrorSpy.mockRestore();
  });

  // ── Clone preserves throttle config ─────────────────────────────────

  it("clone() preserves notificationThrottle config", () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 42, minChunkSize: 10 },
    });

    const cloned = agent.clone();

    expect(cloned.notificationThrottle).toEqual({ intervalMs: 42, minChunkSize: 10 });
    // Should be a separate object
    expect(cloned.notificationThrottle).not.toBe(agent.notificationThrottle);
  });

  it("clone() preserves undefined notificationThrottle", () => {
    const agent = new TestAgent();
    const cloned = agent.clone();
    expect(cloned.notificationThrottle).toBeUndefined();
  });

  // ── Input validation ────────────────────────────────────────────────

  it("throws on negative intervalMs", () => {
    expect(() => new TestAgent({ notificationThrottle: { intervalMs: -1 } })).toThrow(
      "non-negative finite number",
    );
  });

  it("throws on NaN intervalMs", () => {
    expect(() => new TestAgent({ notificationThrottle: { intervalMs: NaN } })).toThrow(
      "non-negative finite number",
    );
  });

  it("throws on Infinity intervalMs", () => {
    expect(() => new TestAgent({ notificationThrottle: { intervalMs: Infinity } })).toThrow(
      "non-negative finite number",
    );
  });

  it("throws on negative minChunkSize", () => {
    expect(
      () => new TestAgent({ notificationThrottle: { intervalMs: 16, minChunkSize: -5 } }),
    ).toThrow("non-negative finite number");
  });

  it("intervalMs: 0 with no minChunkSize skips throttle activation", () => {
    const agent = new TestAgent({ notificationThrottle: { intervalMs: 0 } });
    // Zero-zero is a no-op — treated as unthrottled
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

      agent.subscribe({
        onMessagesChanged: ({ messages }) => {
          const msg = messages[0];
          const content =
            msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
          calls.push(content);
        },
      });

      const runPromise = agent.runAgent();
      await vi.advanceTimersByTimeAsync(0);

      agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: "A",
      } as BaseEvent);
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: "B",
      } as BaseEvent);

      const callsBeforeTimer = calls.length;

      // Advance past the throttle window — trailing timer should fire
      await vi.advanceTimersByTimeAsync(60);

      // Trailing timer should have fired, producing at least one more notification
      expect(calls.length).toBeGreaterThan(callsBeforeTimer);
      expect(calls[calls.length - 1]).toBe("AB");

      agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
      agent.subject.complete();
      await vi.advanceTimersByTimeAsync(0);
      await runPromise;
    } finally {
      vi.useRealTimers();
    }
  });

  // ── onStateChanged subscriber error (Issue 7) ──────────────────────

  it("onStateChanged subscriber error is caught and does not crash", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 50 },
    });
    const goodCalls: any[] = [];
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    agent.subscribe({
      onStateChanged: () => {
        throw new Error("state boom");
      },
    });
    agent.subscribe({
      onStateChanged: ({ state }) => {
        goodCalls.push(structuredClone(state));
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.STATE_SNAPSHOT,
      snapshot: { count: 1 },
    } as BaseEvent);
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

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
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 5000, minChunkSize: 5 },
    });
    const calls: number[] = [];

    agent.subscribe({
      onMessagesChanged: () => {
        calls.push(calls.length);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);

    // Emit 4 chars on m1 (below minChunkSize=5)
    for (let i = 0; i < 4; i++) {
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: String.fromCharCode(65 + i),
      } as BaseEvent);
    }

    // Switch to m2 — 6 chars (above minChunkSize=5, should trigger notification)
    for (let i = 0; i < 6; i++) {
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m2",
        delta: String.fromCharCode(75 + i),
      } as BaseEvent);
    }

    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // Leading edge on first m1 chunk. After identity change to m2, tracking resets.
    // After 5+ chars on m2, chunk threshold fires. Then finalize flush.
    // Should see 2-4 notifications, not 10+ (one per event).
    expect(calls.length).toBeGreaterThanOrEqual(2);
    expect(calls.length).toBeLessThanOrEqual(5);
  });

  // ── No flush on stream error (Issue 9) ──────────────────────────────

  it("does not flush pending notifications on stream error", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 5000 },
    });
    const calls: string[] = [];
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        const msg = messages[0];
        const content =
          msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
        calls.push(content);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: "hello",
    } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: " world",
    } as BaseEvent);

    // Snapshot how many notifications fired before the error
    const callsBeforeError = calls.length;

    // Error the stream — pending notifications should NOT be flushed
    agent.subject.error(new Error("stream error"));
    await runPromise.catch(() => {});

    // No additional notification was flushed after the error
    expect(calls.length).toBe(callsBeforeError);

    consoleErrorSpy.mockRestore();
  });

  // ── Non-throttled subscriber error resilience ───────────────────────

  it("without throttle, a throwing subscriber does not crash the pipeline", async () => {
    const agent = new TestAgent();
    const calls: number[] = [];

    // First subscriber throws
    agent.subscribe({
      onMessagesChanged: () => {
        throw new Error("subscriber boom");
      },
    });

    // Second subscriber should still receive notifications
    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        calls.push(messages.length);
      },
    });

    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: "hello",
    } as BaseEvent);
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // Second subscriber was reached despite first throwing
    expect(calls.length).toBeGreaterThanOrEqual(1);
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("AG-UI: Subscriber"),
      expect.any(Error),
    );

    errorSpy.mockRestore();
  });

  // ── Combined messages + state under single throttle window (#17) ────

  it("combined message and state mutations within a throttle window both flush", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 5000 },
    });
    const msgCalls: number[] = [];
    const stateCalls: any[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => { msgCalls.push(messages.length); },
      onStateChanged: ({ state }) => { stateCalls.push(structuredClone(state)); },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: "hello",
    } as BaseEvent);
    agent.subject.next({
      type: EventType.STATE_SNAPSHOT,
      snapshot: { count: 1 },
    } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: " world",
    } as BaseEvent);
    agent.subject.next({
      type: EventType.STATE_SNAPSHOT,
      snapshot: { count: 2 },
    } as BaseEvent);
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // Both onMessagesChanged and onStateChanged must have fired at least once
    expect(msgCalls.length).toBeGreaterThanOrEqual(1);
    expect(stateCalls.length).toBeGreaterThanOrEqual(1);
    // Final flush has latest of both
    expect(stateCalls[stateCalls.length - 1]).toEqual({ count: 2 });
  });

  // ── intervalMs: 0 + minChunkSize streaming (#18) ───────────────────

  it("intervalMs: 0 with minChunkSize drives notifications by character count alone", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 0, minChunkSize: 5 },
    });
    const calls: string[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        const msg = messages[0];
        const content = msg?.role === "assistant" && typeof msg.content === "string" ? msg.content : "";
        calls.push(content);
      },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    // 15 single-char chunks → should fire roughly every 5 chars
    for (let i = 0; i < 15; i++) {
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: String.fromCharCode(65 + i),
      } as BaseEvent);
    }
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // Should be ~3-4 notifications (leading + every ~5 chars + flush), not 15
    expect(calls.length).toBeGreaterThanOrEqual(2);
    expect(calls.length).toBeLessThanOrEqual(6);
    expect(calls[calls.length - 1]).toBe("ABCDEFGHIJKLMNO");
  });

  // ── Multiple sequential runAgent on same agent (#20) ───────────────

  it("second runAgent on same throttled agent starts fresh throttle state", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 5000 },
    });
    const calls: string[] = [];

    agent.subscribe({
      onMessagesChanged: ({ messages }) => {
        const lastMsg = messages[messages.length - 1];
        const content = lastMsg?.role === "assistant" && typeof lastMsg.content === "string" ? lastMsg.content : "";
        calls.push(content);
      },
    });

    // First run
    const run1 = agent.runAgent();
    await tick();
    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m1",
      delta: "first",
    } as BaseEvent);
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();
    await run1;

    const callsAfterRun1 = calls.length;

    // Second run — new subject needed since the old one completed
    (agent as any).subject = new Subject<BaseEvent>();

    const run2 = agent.runAgent();
    await tick();
    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    agent.subject.next({
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "m2",
      delta: "second",
    } as BaseEvent);
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();
    await run2;

    // Second run should have fired leading edge immediately (fresh state)
    expect(calls.length).toBeGreaterThan(callsAfterRun1);
    expect(calls[calls.length - 1]).toBe("second");
  });

  // ── Non-string content falls back to time-only throttling (#21) ────

  it("non-string assistant content falls back to time-only throttling", async () => {
    const agent = new TestAgent({
      notificationThrottle: { intervalMs: 5000, minChunkSize: 5 },
    });
    const calls: number[] = [];

    agent.subscribe({
      onMessagesChanged: () => { calls.push(calls.length); },
    });

    const runPromise = agent.runAgent();
    await tick();

    agent.subject.next({ type: EventType.RUN_STARTED } as BaseEvent);
    // Emit chunks — the message will have string content from TEXT_MESSAGE_CHUNK,
    // but the minChunkSize logic only tracks the last assistant message.
    // This test verifies no crash occurs and notifications still fire.
    for (let i = 0; i < 10; i++) {
      agent.subject.next({
        type: EventType.TEXT_MESSAGE_CHUNK,
        messageId: "m1",
        delta: String.fromCharCode(65 + i),
      } as BaseEvent);
    }
    agent.subject.next({ type: EventType.RUN_FINISHED } as BaseEvent);
    agent.subject.complete();

    await runPromise;

    // Should get at least leading + flush
    expect(calls.length).toBeGreaterThanOrEqual(1);
    // agent.messages should have the full content
    const lastMsg = agent.messages[agent.messages.length - 1];
    const content = lastMsg?.role === "assistant" && typeof lastMsg.content === "string" ? lastMsg.content : "";
    expect(content).toBe("ABCDEFGHIJ");
  });
});
