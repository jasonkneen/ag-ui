import { describe, expect, it } from "vitest";
import { InMemorySessionStore } from "../sessions";
import type { SessionRecord } from "../types";

const record = (): SessionRecord => ({
  sessionId: "sesn_1",
  toolNames: ["a"],
  pendingClientToolUseIds: ["ctu_1"],
});

describe("InMemorySessionStore", () => {
  it("does not alias the record it was given", () => {
    const store = new InMemorySessionStore();
    const original = record();
    store.set("t", original);

    // Mutating the caller's copy after the write must not reach the store.
    original.pendingClientToolUseIds.push("ctu_2");
    original.sessionId = "sesn_mutated";

    const read = store.get("t")!;
    expect(read.sessionId).toBe("sesn_1");
    expect(read.pendingClientToolUseIds).toEqual(["ctu_1"]);
  });

  it("does not alias the record it hands out", () => {
    const store = new InMemorySessionStore();
    store.set("t", record());

    // The agent mutates records in place between persists; those mutations
    // must not be visible until they are actually written back.
    const first = store.get("t")!;
    first.pendingClientToolUseIds.push("ctu_2");
    first.lastUserMessageId = "m_unpersisted";

    const second = store.get("t")!;
    expect(second.pendingClientToolUseIds).toEqual(["ctu_1"]);
    expect(second.lastUserMessageId).toBeUndefined();
  });

  it("returns undefined for an unknown thread and forgets deleted ones", () => {
    const store = new InMemorySessionStore();
    expect(store.get("nope")).toBeUndefined();
    store.set("t", record());
    store.delete("t");
    expect(store.get("t")).toBeUndefined();
  });
});
