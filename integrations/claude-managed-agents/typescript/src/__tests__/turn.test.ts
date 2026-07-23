import { EventType } from "@ag-ui/client";
import type { BaseEvent } from "@ag-ui/client";
import { describe, expect, it } from "vitest";
import { runTurn } from "../turn";
import { createFakeClient } from "./fake-client";

const idleEndTurn = { type: "session.status_idle", id: "idle_1", stop_reason: { type: "end_turn" } };

const collect = async (
  streamEvents: unknown[],
  overrides: Partial<Parameters<typeof runTurn>[0]> = {},
  clientOptions: Parameters<typeof createFakeClient>[0] = {},
) => {
  const fake = createFakeClient({ streams: [streamEvents], ...clientOptions });
  const emitted: BaseEvent[] = [];
  const outcome = await runTurn({
    client: fake.client,
    sessionId: "sesn_1",
    outbound: [{ type: "user.message", content: [{ type: "text", text: "hi" }] }],
    clientTools: new Map(),
    backendTools: new Map(),
    toolConfirmation: undefined,
    streamDeltas: true,
    emit: (event) => emitted.push(event),
    signal: new AbortController().signal,
    ...overrides,
  });
  return { emitted, outcome, fake };
};

const types = (events: BaseEvent[]) => events.map((event) => event.type);

describe("runTurn", () => {
  it("streams a text preview, tops it up from the buffered message, and finishes", async () => {
    const { emitted, outcome, fake } = await collect([
      { type: "session.status_running", id: "run_1" },
      { type: "event_start", event: { type: "agent.message", id: "msg_1" } },
      { type: "event_delta", event_id: "msg_1", delta: { type: "content_delta", index: 0, content: { type: "text", text: "Hel" } } },
      { type: "event_delta", event_id: "msg_1", delta: { type: "content_delta", index: 0, content: { type: "text", text: "lo" } } },
      { type: "agent.message", id: "msg_1", content: [{ type: "text", text: "Hello there" }] },
      idleEndTurn,
    ]);

    expect(outcome).toEqual({ status: "finished" });
    expect(fake.sent[0].events).toEqual([{ type: "user.message", content: [{ type: "text", text: "hi" }] }]);
    expect(emitted).toEqual([
      { type: EventType.TEXT_MESSAGE_START, messageId: "msg_1", role: "assistant" },
      { type: EventType.TEXT_MESSAGE_CONTENT, messageId: "msg_1", delta: "Hel" },
      { type: EventType.TEXT_MESSAGE_CONTENT, messageId: "msg_1", delta: "lo" },
      { type: EventType.TEXT_MESSAGE_CONTENT, messageId: "msg_1", delta: " there" },
      { type: EventType.TEXT_MESSAGE_END, messageId: "msg_1" },
    ]);
  });

  it("emits a whole message when there was no preview", async () => {
    const { emitted } = await collect([
      { type: "agent.message", id: "msg_1", content: [{ type: "text", text: "All at once" }] },
      idleEndTurn,
    ]);
    expect(emitted).toEqual([
      { type: EventType.TEXT_MESSAGE_START, messageId: "msg_1", role: "assistant" },
      { type: EventType.TEXT_MESSAGE_CONTENT, messageId: "msg_1", delta: "All at once" },
      { type: EventType.TEXT_MESSAGE_END, messageId: "msg_1" },
    ]);
  });

  it("re-emits a corrected message when the preview diverges", async () => {
    const { emitted } = await collect([
      { type: "event_start", event: { type: "agent.message", id: "msg_1" } },
      { type: "event_delta", event_id: "msg_1", delta: { type: "content_delta", index: 0, content: { type: "text", text: "Draft" } } },
      { type: "agent.message", id: "msg_1", content: [{ type: "text", text: "Final" }] },
      idleEndTurn,
    ]);
    expect(emitted).toEqual([
      { type: EventType.TEXT_MESSAGE_START, messageId: "msg_1", role: "assistant" },
      { type: EventType.TEXT_MESSAGE_CONTENT, messageId: "msg_1", delta: "Draft" },
      { type: EventType.TEXT_MESSAGE_END, messageId: "msg_1" },
      { type: EventType.TEXT_MESSAGE_START, messageId: "corrected_msg_1", role: "assistant" },
      { type: EventType.TEXT_MESSAGE_CONTENT, messageId: "corrected_msg_1", delta: "Final" },
      { type: EventType.TEXT_MESSAGE_END, messageId: "corrected_msg_1" },
    ]);
  });

  it("maps a thinking stretch to reasoning start and end", async () => {
    const { emitted } = await collect([
      { type: "event_start", event: { type: "agent.thinking", id: "think_1" } },
      { type: "agent.thinking", id: "think_1" },
      idleEndTurn,
    ]);
    expect(types(emitted)).toEqual([
      EventType.REASONING_START,
      EventType.REASONING_MESSAGE_START,
      EventType.REASONING_MESSAGE_END,
      EventType.REASONING_END,
    ]);
  });

  it("streams built-in tool calls and their results", async () => {
    const { emitted } = await collect([
      { type: "agent.tool_use", id: "tu_1", name: "bash", input: { command: "ls" } },
      { type: "agent.tool_result", id: "tr_1", tool_use_id: "tu_1", content: [{ type: "text", text: "file.txt" }] },
      idleEndTurn,
    ]);
    expect(emitted).toEqual([
      { type: EventType.TOOL_CALL_START, toolCallId: "tu_1", toolCallName: "bash" },
      { type: EventType.TOOL_CALL_ARGS, toolCallId: "tu_1", delta: '{"command":"ls"}' },
      { type: EventType.TOOL_CALL_END, toolCallId: "tu_1" },
      { type: EventType.TOOL_CALL_RESULT, messageId: "result_tu_1", toolCallId: "tu_1", content: "file.txt", role: "tool" },
    ]);
  });

  it("maps MCP tool calls with a server-qualified name", async () => {
    const { emitted } = await collect([
      { type: "agent.mcp_tool_use", id: "mcp_1", name: "search", mcp_server_name: "docs", input: { q: "x" } },
      { type: "agent.mcp_tool_result", id: "mr_1", mcp_tool_use_id: "mcp_1", content: [{ type: "text", text: "found" }] },
      idleEndTurn,
    ]);
    expect(emitted[0]).toEqual({ type: EventType.TOOL_CALL_START, toolCallId: "mcp_1", toolCallName: "docs: search" });
    expect(emitted[3]).toMatchObject({ type: EventType.TOOL_CALL_RESULT, toolCallId: "mcp_1", content: "found" });
  });

  it("runs a backend tool and posts its result back into the session", async () => {
    const backend = { name: "get_time", description: "", parameters: {}, handler: async () => "noon" };
    const { emitted, fake } = await collect(
      [
        { type: "agent.custom_tool_use", id: "ctu_1", name: "get_time", input: {} },
        { type: "session.status_idle", id: "idle_1", stop_reason: { type: "requires_action", event_ids: ["ctu_1"] } },
        idleEndTurn,
      ],
      { backendTools: new Map([["get_time", backend]]) },
    );
    expect(emitted).toContainEqual({
      type: EventType.TOOL_CALL_RESULT,
      messageId: "result_ctu_1",
      toolCallId: "ctu_1",
      content: "noon",
      role: "tool",
    });
    expect(fake.sent[1].events).toEqual([
      { type: "user.custom_tool_result", custom_tool_use_id: "ctu_1", content: [{ type: "text", text: "noon" }], is_error: false },
    ]);
  });

  it("posts an error result for a tool nothing can execute", async () => {
    const { fake } = await collect([
      { type: "agent.custom_tool_use", id: "ctu_1", name: "mystery", input: {} },
      idleEndTurn,
    ]);
    expect(fake.sent[1].events[0]).toMatchObject({
      type: "user.custom_tool_result",
      custom_tool_use_id: "ctu_1",
      is_error: true,
    });
  });

  it("parks the turn when the frontend must execute a tool", async () => {
    const { emitted, outcome, fake } = await collect(
      [
        { type: "agent.custom_tool_use", id: "ctu_1", name: "confirm_purchase", input: { amount: 5 } },
        { type: "session.status_idle", id: "idle_1", stop_reason: { type: "requires_action", event_ids: ["ctu_1"] } },
      ],
      { clientTools: new Map([["confirm_purchase", "confirm_purchase"]]) },
    );
    expect(outcome).toEqual({ status: "parked", clientToolUseIds: ["ctu_1"] });
    expect(fake.sent).toHaveLength(1); // only the user message; no result posted
    expect(types(emitted)).toEqual([EventType.TOOL_CALL_START, EventType.TOOL_CALL_ARGS, EventType.TOOL_CALL_END]);
  });

  it("reports the frontend's original name for a normalized tool", async () => {
    const { emitted, outcome } = await collect(
      [
        { type: "agent.custom_tool_use", id: "ctu_1", name: "search_web", input: {} },
        { type: "session.status_idle", id: "idle_1", stop_reason: { type: "requires_action", event_ids: ["ctu_1"] } },
      ],
      { clientTools: new Map([["search_web", "search web"]]) },
    );
    expect(outcome).toEqual({ status: "parked", clientToolUseIds: ["ctu_1"] });
    expect(emitted[0]).toEqual({ type: EventType.TOOL_CALL_START, toolCallId: "ctu_1", toolCallName: "search web" });
  });

  it("answers a confirmation-gated tool when a policy is configured", async () => {
    const { outcome, fake } = await collect(
      [
        { type: "agent.tool_use", id: "tu_1", name: "bash", input: {}, evaluated_permission: "ask" },
        { type: "session.status_idle", id: "idle_1", stop_reason: { type: "requires_action", event_ids: ["tu_1"] } },
        { type: "agent.tool_result", id: "tr_1", tool_use_id: "tu_1", content: [] },
        idleEndTurn,
      ],
      { toolConfirmation: "allow" },
    );
    expect(outcome).toEqual({ status: "finished" });
    expect(fake.sent[1].events).toEqual([{ type: "user.tool_confirmation", tool_use_id: "tu_1", result: "allow" }]);
  });

  it("fails the run on a confirmation-gated tool with no policy", async () => {
    const { emitted, outcome } = await collect([
      { type: "agent.tool_use", id: "tu_1", name: "bash", input: {}, evaluated_permission: "ask" },
      { type: "session.status_idle", id: "idle_1", stop_reason: { type: "requires_action", event_ids: ["tu_1"] } },
    ]);
    expect(outcome).toEqual({ status: "errored" });
    expect(emitted.at(-1)).toMatchObject({ type: EventType.RUN_ERROR, code: "tool_confirmation_required" });
  });

  it("surfaces a terminal session error with its type as the code", async () => {
    const { emitted, outcome } = await collect([
      { type: "session.error", id: "err_1", error: { type: "billing_error", message: "Out of credits", retry_status: { type: "terminal" } } },
    ]);
    expect(outcome).toEqual({ status: "errored" });
    expect(emitted).toEqual([{ type: EventType.RUN_ERROR, message: "Out of credits", code: "billing_error" }]);
  });

  it("ignores a retrying session error and completes", async () => {
    const { outcome } = await collect([
      { type: "session.error", id: "err_1", error: { type: "model_overloaded_error", message: "busy", retry_status: { type: "retrying" } } },
      { type: "agent.message", id: "msg_1", content: [{ type: "text", text: "ok" }] },
      idleEndTurn,
    ]);
    expect(outcome).toEqual({ status: "finished" });
  });

  it("treats retries_exhausted as an error, not a clean finish", async () => {
    const { emitted, outcome } = await collect([
      { type: "session.status_idle", id: "idle_1", stop_reason: { type: "retries_exhausted" } },
    ]);
    expect(outcome).toEqual({ status: "errored" });
    expect(emitted.at(-1)).toMatchObject({ type: EventType.RUN_ERROR, code: "retries_exhausted" });
  });

  it("reports a terminated session as ended", async () => {
    const { outcome } = await collect([{ type: "session.status_terminated", id: "term_1" }]);
    expect(outcome).toEqual({ status: "errored", sessionEnded: true });
  });

  it("closes a dangling preview when the model request ends without a message", async () => {
    const { emitted } = await collect([
      { type: "event_start", event: { type: "agent.message", id: "msg_1" } },
      { type: "event_delta", event_id: "msg_1", delta: { type: "content_delta", index: 0, content: { type: "text", text: "partia" } } },
      { type: "span.model_request_end", id: "span_1", model_request_start_id: "s_1", is_error: true, model_usage: {} },
      idleEndTurn,
    ]);
    expect(types(emitted)).toEqual([
      EventType.TEXT_MESSAGE_START,
      EventType.TEXT_MESSAGE_CONTENT,
      EventType.TEXT_MESSAGE_END,
    ]);
  });

  it("errors when the stream ends before the turn completes", async () => {
    const { emitted, outcome } = await collect([
      { type: "event_start", event: { type: "agent.message", id: "msg_1" } },
    ]);
    expect(outcome).toEqual({ status: "errored" });
    // The open message is closed before the error.
    expect(types(emitted)).toEqual([EventType.TEXT_MESSAGE_START, EventType.TEXT_MESSAGE_END, EventType.RUN_ERROR]);
  });
});
