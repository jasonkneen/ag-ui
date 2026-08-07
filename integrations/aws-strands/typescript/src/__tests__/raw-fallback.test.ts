/**
 * Issue #2291 — the adapter's event dispatch has no terminal fallback, so any
 * Strands event it does not translate is dropped with no error and no log.
 * Provider extensions the adapter predates (Bedrock citations being the
 * reported case) vanish before they reach the frontend.
 *
 * The adapter must forward anything it does not map as a RAW event carrying
 * the original Strands payload, while the deliberate lifecycle skips stay
 * silent.
 */

import { describe, it, expect } from "vitest";
import type { AgentStreamEvent } from "@strands-agents/sdk";
import { EventType } from "@ag-ui/core";

import { collect, scriptedStrandsAgent } from "./helpers";

const CITATION_EVENT = {
  type: "modelCitationEvent",
  citation: {
    title: "quarterly-report.pdf",
    sourceContent: [{ text: "revenue grew 12%" }],
    location: { documentChar: { documentIndex: 0, start: 10, end: 26 } },
  },
} as unknown as AgentStreamEvent;

function rawEvents(events: Array<{ type: string }>) {
  return events.filter((e) => e.type === EventType.RAW) as unknown as Array<{
    event: { type?: string };
    source?: string;
  }>;
}

describe("RAW fallback for unmapped Strands events", () => {
  it("forwards an unmapped provider event as RAW with the original payload", async () => {
    const agent = scriptedStrandsAgent([CITATION_EVENT]);
    const events = await collect(agent);

    const raws = rawEvents(events as Array<{ type: string }>);
    const citationRaws = raws.filter(
      (e) => e.event?.type === "modelCitationEvent",
    );

    expect(citationRaws).toHaveLength(1);
    expect(citationRaws[0].source).toBe("strands");
    expect(citationRaws[0].event).toEqual(CITATION_EVENT);
  });

  it("unwraps the modelStreamUpdateEvent envelope before emitting RAW", async () => {
    // Strands v1 decorates raw model events; the RAW payload must be the inner
    // event the adapter actually failed to map, not the wrapper.
    const agent = scriptedStrandsAgent([
      {
        type: "modelStreamUpdateEvent",
        event: CITATION_EVENT,
      } as unknown as AgentStreamEvent,
    ]);
    const events = await collect(agent);

    const raws = rawEvents(events as Array<{ type: string }>);
    expect(raws.map((e) => e.event?.type)).toContain("modelCitationEvent");
    expect(raws.map((e) => e.event?.type)).not.toContain(
      "modelStreamUpdateEvent",
    );
  });

  it("keeps lifecycle plumbing events silent", async () => {
    const agent = scriptedStrandsAgent([
      { type: "initializedEvent" } as unknown as AgentStreamEvent,
      { type: "beforeInvocationEvent" } as unknown as AgentStreamEvent,
      { type: "beforeModelCallEvent" } as unknown as AgentStreamEvent,
      { type: "afterModelCallEvent" } as unknown as AgentStreamEvent,
      { type: "modelMessageStartEvent" } as unknown as AgentStreamEvent,
      { type: "modelMessageStopEvent" } as unknown as AgentStreamEvent,
      { type: "afterInvocationEvent" } as unknown as AgentStreamEvent,
    ]);
    const events = await collect(agent);

    expect(rawEvents(events as Array<{ type: string }>)).toEqual([]);
  });

  it("does not disturb events the adapter already maps", async () => {
    const agent = scriptedStrandsAgent([
      {
        type: "modelContentBlockDeltaEvent",
        delta: { type: "textDelta", text: "hello" },
      } as unknown as AgentStreamEvent,
      CITATION_EVENT,
    ]);
    const events = await collect(agent);

    const deltas = (
      events as unknown as Array<{ type: string; delta?: string }>
    )
      .filter((e) => e.type === EventType.TEXT_MESSAGE_CONTENT)
      .map((e) => e.delta)
      .join("");
    expect(deltas).toBe("hello");
    expect(rawEvents(events as Array<{ type: string }>)).toHaveLength(1);
  });
});
