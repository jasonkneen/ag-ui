import {
  Middleware,
  AbstractAgent,
  BaseEvent,
  EventType,
  RunAgentInput,
} from "@ag-ui/client";
import { Observable } from "rxjs";

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

export interface EventThrottleConfig {
  /** Time-based throttle window in ms (e.g. 16 = ~60fps). */
  readonly intervalMs: number;
  /** Min new characters to accumulate before flushing. Default: 0. */
  readonly minChunkSize?: number;
}

// ---------------------------------------------------------------------------
// Event classification
// ---------------------------------------------------------------------------

/**
 * Events that must pass through immediately (flushing any pending buffer first).
 * Everything NOT in this set is buffered.
 */
const IMMEDIATE_EVENT_TYPES: ReadonlySet<string> = new Set([
  EventType.RUN_STARTED,
  EventType.RUN_FINISHED,
  EventType.RUN_ERROR,
  EventType.TOOL_CALL_START,
  EventType.TOOL_CALL_ARGS,
  EventType.TOOL_CALL_END,
  EventType.TOOL_CALL_RESULT,
  EventType.TEXT_MESSAGE_START,
  EventType.TEXT_MESSAGE_END,
  EventType.CUSTOM,
]);

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

export class EventThrottleMiddleware extends Middleware {
  private readonly intervalMs: number;
  private readonly minChunkSize: number;
  private readonly isNoop: boolean;

  constructor(config: EventThrottleConfig) {
    super();

    const { intervalMs, minChunkSize } = config;
    if (!Number.isFinite(intervalMs) || intervalMs < 0) {
      throw new Error(
        `intervalMs must be a non-negative finite number, got ${intervalMs}`,
      );
    }
    if (
      minChunkSize !== undefined &&
      (!Number.isFinite(minChunkSize) || minChunkSize < 0)
    ) {
      throw new Error(
        `minChunkSize must be a non-negative finite number, got ${minChunkSize}`,
      );
    }

    this.intervalMs = intervalMs;
    this.minChunkSize = minChunkSize ?? 0;
    this.isNoop = intervalMs <= 0 && this.minChunkSize <= 0;
  }

  run(input: RunAgentInput, next: AbstractAgent): Observable<BaseEvent> {
    // Use next.run() directly instead of this.runNext() because runNext applies
    // transformChunks, which converts TEXT_MESSAGE_CHUNK into TEXT_MESSAGE_START/
    // CONTENT/END sequences — eliminating the chunk events we need to buffer.
    // The agent's own apply() pipeline normalizes chunks downstream.
    const events$ = next.run(input);

    if (this.isNoop) {
      return events$;
    }

    const intervalMs = this.intervalMs;
    const minChunkSize = this.minChunkSize;

    return new Observable<BaseEvent>((subscriber) => {
      let buffer: BaseEvent[] = [];
      let lastFlushTime = 0;
      let charsSinceFlush = 0;
      let lastTrackedMessageId: string | null = null;
      let timerId: ReturnType<typeof setTimeout> | null = null;

      const flush = () => {
        if (timerId !== null) {
          clearTimeout(timerId);
          timerId = null;
        }
        if (buffer.length === 0) return;

        const batch = buffer;
        buffer = [];
        charsSinceFlush = 0;
        lastFlushTime = Date.now();

        // Coalesce consecutive TEXT_MESSAGE_CHUNK events for the same messageId
        // into a single chunk with combined delta. Non-chunk events pass through as-is.
        const coalesced: BaseEvent[] = [];
        for (const event of batch) {
          if (event.type === EventType.TEXT_MESSAGE_CHUNK) {
            const last = coalesced[coalesced.length - 1];
            if (
              last &&
              last.type === EventType.TEXT_MESSAGE_CHUNK &&
              (last as any).messageId === (event as any).messageId
            ) {
              // Merge delta into the previous chunk
              (last as any).delta =
                ((last as any).delta ?? "") + ((event as any).delta ?? "");
            } else {
              // Push a shallow copy so we don't mutate the original event
              coalesced.push({ ...event } as BaseEvent);
            }
          } else {
            coalesced.push(event);
          }
        }

        for (const event of coalesced) {
          subscriber.next(event);
        }
      };

      const scheduleTrailing = () => {
        if (timerId !== null) return;
        if (intervalMs <= 0) return;
        const elapsed = Date.now() - lastFlushTime;
        const remaining = Math.max(0, intervalMs - elapsed);
        timerId = setTimeout(() => {
          timerId = null;
          flush();
        }, remaining);
      };

      const sub = events$.subscribe({
        next: (event) => {
          // Immediate events flush the buffer first, then pass through directly
          if (IMMEDIATE_EVENT_TYPES.has(event.type)) {
            flush();
            subscriber.next(event);
            // Reset lastFlushTime so next buffered event gets leading-edge treatment
            lastFlushTime = Date.now();
            return;
          }

          // Buffer this event
          buffer.push(event);

          // Track character accumulation for TEXT_MESSAGE_CHUNK
          if (
            event.type === EventType.TEXT_MESSAGE_CHUNK &&
            minChunkSize > 0
          ) {
            const messageId = (event as any).messageId ?? null;
            if (messageId !== lastTrackedMessageId) {
              lastTrackedMessageId = messageId;
              charsSinceFlush = 0;
            }
            charsSinceFlush += ((event as any).delta ?? "").length;
          }

          // Check thresholds
          const isLeading = lastFlushTime === 0;
          const timeThresholdMet =
            intervalMs > 0 && Date.now() - lastFlushTime >= intervalMs;
          const chunkThresholdMet =
            minChunkSize > 0 && charsSinceFlush >= minChunkSize;

          if (isLeading || timeThresholdMet || chunkThresholdMet) {
            flush();
          } else {
            scheduleTrailing();
          }
        },
        error: (err) => {
          // Discard buffer on error — don't deliver inconsistent state
          buffer = [];
          if (timerId !== null) {
            clearTimeout(timerId);
            timerId = null;
          }
          subscriber.error(err);
        },
        complete: () => {
          // Flush remaining on completion
          flush();
          subscriber.complete();
        },
      });

      // Teardown
      return () => {
        if (timerId !== null) {
          clearTimeout(timerId);
          timerId = null;
        }
        sub.unsubscribe();
      };
    });
  }
}
