import { describe, it, expect } from "vitest";
import {
  MessageSchema,
  AssistantMessageSchema,
  ToolMessageSchema,
  ActivityMessageSchema,
  ReasoningMessageSchema,
} from "../types";

describe("message subagentRunId attribution", () => {
  it("accepts subagentRunId on an assistant message", () => {
    const parsed = AssistantMessageSchema.parse({
      id: "m1",
      role: "assistant",
      content: "hi",
      subagentRunId: "sub-1",
    });
    expect(parsed.subagentRunId).toBe("sub-1");
  });

  it("accepts subagentRunId on tool, activity, and reasoning messages", () => {
    expect(
      ToolMessageSchema.parse({
        id: "t1",
        role: "tool",
        content: "ok",
        toolCallId: "tc1",
        subagentRunId: "sub-2",
      }).subagentRunId,
    ).toBe("sub-2");
    expect(
      ActivityMessageSchema.parse({
        id: "a1",
        role: "activity",
        activityType: "x",
        content: {},
        subagentRunId: "sub-3",
      }).subagentRunId,
    ).toBe("sub-3");
    expect(
      ReasoningMessageSchema.parse({
        id: "r1",
        role: "reasoning",
        content: "think",
        subagentRunId: "sub-4",
      }).subagentRunId,
    ).toBe("sub-4");
  });

  it("treats subagentRunId as optional (omitted => undefined)", () => {
    const parsed = MessageSchema.parse({ id: "m2", role: "assistant", content: "hi" });
    expect(parsed.subagentRunId).toBeUndefined();
  });
});
