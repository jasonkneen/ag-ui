import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Message, State } from "@ag-ui/core";

// Spy on the clone helper so we can COUNT how many full structuredClone_ calls
// runSubscribersWithMutation makes per invocation. This is the cost that, when
// paid on every streamed event over a large messages/state, exhausts the
// renderer heap (DataCloneError: structuredClone … out of memory).
const { cloneSpy } = vi.hoisted(() => ({
  cloneSpy: vi.fn((obj: any) => (obj === undefined ? undefined : JSON.parse(JSON.stringify(obj)))),
}));
vi.mock("@/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/utils")>();
  return { ...actual, structuredClone_: cloneSpy };
});

import { type AgentSubscriber, runSubscribersWithMutation } from "../subscriber";

describe("runSubscribersWithMutation clone cost", () => {
  beforeEach(() => cloneSpy.mockClear());

  const noopSubscriber: AgentSubscriber = { onEvent: () => undefined };

  const run = (messages: Message[], state: State) =>
    runSubscribersWithMutation([noopSubscriber], messages, state, (s, m, st) =>
      s.onEvent?.({
        messages: m,
        state: st,
        agent: {} as any,
        input: {} as any,
        event: { type: "RUN_STARTED" } as any,
      }),
    );

  it("clones baseline messages+state for SMALL payloads (dev freeze guard active)", async () => {
    await run([{ id: "m", role: "user", content: "hi" }], { counter: 1 });
    // Freeze path: baseline messages + baseline state are cloned.
    expect(cloneSpy).toHaveBeenCalledTimes(2);
  });

  it("makes ZERO clones for a LARGE payload with no mutation (the fix)", async () => {
    const bigArgs = "x".repeat(600_000); // > DEV_FREEZE_CHAR_LIMIT (512K)
    const messages: Message[] = [
      {
        id: "m",
        role: "assistant",
        toolCalls: [{ id: "tc", type: "function", function: { name: "write_file", arguments: bigArgs } }],
      } as unknown as Message,
    ];
    await run(messages, {});
    // Large payload skips the dev clone+freeze; no subscriber mutation ⇒ no clone.
    // Before the fix this was 2 full clones of a ~600KB structure on EVERY event.
    expect(cloneSpy).not.toHaveBeenCalled();
  });

  it("still defensively clones a subscriber's returned mutation on the large path", async () => {
    const bigArgs = "x".repeat(600_000);
    const mutating: AgentSubscriber = {
      onEvent: ({ messages }) => ({ messages: [...messages] as Message[] }),
    };
    const messages: Message[] = [
      {
        id: "m",
        role: "assistant",
        toolCalls: [{ id: "tc", type: "function", function: { name: "write_file", arguments: bigArgs } }],
      } as unknown as Message,
    ];
    const result = await runSubscribersWithMutation([mutating], messages, {}, (s, m, st) =>
      s.onEvent?.({ messages: m, state: st, agent: {} as any, input: {} as any, event: { type: "RUN_STARTED" } as any }),
    );
    // Exactly one clone — the defensive copy of the returned mutation (isolation
    // contract preserved), not a per-event baseline clone.
    expect(cloneSpy).toHaveBeenCalledTimes(1);
    expect(result.messages).toBeDefined();
  });
});
