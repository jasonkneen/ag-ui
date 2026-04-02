import { describe, it, expect, vi } from "vitest";
import { Observable, Subject } from "rxjs";
import {
  AbstractAgent,
  Middleware,
  BaseEvent,
  EventType,
  RunAgentInput,
} from "@ag-ui/client";

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/** A minimal agent whose event stream we control via a Subject. */
class TestAgent extends AbstractAgent {
  public subject = new Subject<BaseEvent>();

  run(_input: RunAgentInput): Observable<BaseEvent> {
    return this.subject.asObservable();
  }
}

/** Cast helper — BaseEvent's Zod passthrough schema adds an index signature. */
function ev(partial: { type: EventType; [key: string]: unknown }): BaseEvent {
  return partial as BaseEvent;
}

const runStarted = () => ev({ type: EventType.RUN_STARTED });
const runFinished = () => ev({ type: EventType.RUN_FINISHED });
const textChunk = (messageId: string, delta: string) =>
  ev({ type: EventType.TEXT_MESSAGE_CHUNK, messageId, delta });
const textStart = (messageId: string) =>
  ev({ type: EventType.TEXT_MESSAGE_START, messageId, role: "assistant" });
const textEnd = (messageId: string) =>
  ev({ type: EventType.TEXT_MESSAGE_END, messageId });
const stateSnapshot = (snapshot: Record<string, unknown>) =>
  ev({ type: EventType.STATE_SNAPSHOT, snapshot });
const toolCallStart = (toolCallId: string) =>
  ev({ type: EventType.TOOL_CALL_START, toolCallId, toolCallName: "test" });

/** Collect all events emitted by an observable into an array. */
function collectEvents(obs$: Observable<BaseEvent>): {
  events: BaseEvent[];
  done: Promise<void>;
} {
  const events: BaseEvent[] = [];
  const done = new Promise<void>((resolve, reject) => {
    obs$.subscribe({
      next: (e) => events.push(e),
      error: reject,
      complete: resolve,
    });
  });
  return { events, done };
}

/**
 * Run the middleware (or raw agent) by piping events through,
 * collecting what comes out the other side.
 */
function setup(middleware?: Middleware) {
  const agent = new TestAgent();
  const input: RunAgentInput = {
    threadId: "t1",
    runId: "r1",
    messages: [],
    tools: [],
    context: [],
    forwardedProps: {},
  };

  let events$: Observable<BaseEvent>;
  if (middleware) {
    events$ = middleware.run(input, agent);
  } else {
    events$ = agent.run(input);
  }

  const { events, done } = collectEvents(events$);
  return { agent, events, done };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("NotificationThrottleMiddleware", () => {
  describe("baseline (no middleware)", () => {
    it("without middleware, every event passes through 1:1", async () => {
      const { agent, events, done } = setup();

      agent.subject.next(runStarted());
      agent.subject.next(textChunk("m1", "A"));
      agent.subject.next(textChunk("m1", "B"));
      agent.subject.next(textChunk("m1", "C"));
      agent.subject.next(runFinished());
      agent.subject.complete();

      await done;
      expect(events).toHaveLength(5);
    });
  });

  describe("time-based throttle", () => {
    it("with intervalMs, fewer events emitted than chunks sent", async () => {
      const { NotificationThrottleMiddleware } = await import("../index");
      const mw = new NotificationThrottleMiddleware({ intervalMs: 50 });
      const { agent, events, done } = setup(mw);

      agent.subject.next(runStarted());
      for (let i = 0; i < 20; i++) {
        agent.subject.next(textChunk("m1", String.fromCharCode(65 + i)));
      }
      agent.subject.next(runFinished());
      agent.subject.complete();

      await done;

      const chunkEvents = events.filter(
        (e) => e.type === EventType.TEXT_MESSAGE_CHUNK,
      );
      expect(chunkEvents.length).toBeLessThan(20);
      expect(chunkEvents.length).toBeGreaterThanOrEqual(1);

      const allDeltas = chunkEvents.map((e) => (e as any).delta).join("");
      expect(allDeltas).toBe("ABCDEFGHIJKLMNOPQRST");
    });
  });
});
