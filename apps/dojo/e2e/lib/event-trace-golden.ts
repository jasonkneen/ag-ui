import type { TraceEvent } from "./event-trace-events";

type EventTraceShape = {
  readonly [journeyKey: string]: readonly TraceEvent[];
};

export type EventTraceDestination = {
  sourceUrl: string;
  journeyKey: string;
};

const destinations = new WeakMap<object, EventTraceDestination>();

export function defineEventTrace<const Golden extends EventTraceShape>(
  sourceUrl: string,
  golden: Golden,
): Golden {
  for (const [journeyKey, journey] of Object.entries(golden)) {
    destinations.set(journey, { sourceUrl, journeyKey });
  }

  return golden;
}

export function getEventTraceDestination(
  journey: readonly TraceEvent[],
): EventTraceDestination | undefined {
  return destinations.get(journey);
}
