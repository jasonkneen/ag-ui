/**
 * Repro for the SSE-stream-drop bug (OSS-28 / GitHub #1278) on the
 * TypeScript LangGraph integration.
 *
 * The Python integration was fixed by an ID guard in `prepare_stream`
 * (regenerate only when the last user message id is present in the
 * checkpoint). The TypeScript `prepareStream` has NO such guard: it routes
 * into `prepareRegenerateStream` on any non-system count mismatch
 * (`stateNonSystemCount > inputNonSystemCount`, agent.ts), then
 * `getCheckpointByMessage` throws `Error("Message not found")` because the
 * client's freshly generated UUID was never persisted.
 *
 * The guard has now been ported to agent.ts: regenerate is only taken when
 * the incoming IDs are not already a subset of the checkpoint AND the last
 * user message's ID exists in the checkpoint. These tests assert recovery.
 */
import { describe, it, expect, vi } from "vitest";
import { LangGraphAgent } from "../agent";

function buildAgent(checkpointMessages: any[], history: any[]) {
  const agent = new LangGraphAgent({
    graphId: "test-graph",
    deploymentUrl: "http://localhost:8000",
  });

  (agent as any).activeRun = {
    id: "run-1",
    threadId: "thread-1",
    hasFunctionStreaming: false,
    modelMadeToolCall: false,
  };
  // Pre-set assistant so prepareStream doesn't need a live search.
  (agent as any).assistant = {
    assistant_id: "asst-1",
    graph_id: "test-graph",
    config: { configurable: {} },
  };

  const streamCalls: any[] = [];
  (agent as any).client = {
    threads: {
      get: vi.fn().mockResolvedValue({ thread_id: "thread-1" }),
      create: vi.fn().mockResolvedValue({ thread_id: "thread-1" }),
      getState: vi
        .fn()
        .mockResolvedValue({ values: { messages: checkpointMessages }, tasks: [] }),
      getHistory: vi.fn().mockResolvedValue(history),
      updateState: vi
        .fn()
        .mockResolvedValue({ checkpoint: { checkpoint_id: "ck-fork" } }),
    },
    assistants: {
      search: vi.fn().mockResolvedValue([
        { assistant_id: "asst-1", graph_id: "test-graph", config: { configurable: {} } },
      ]),
      getGraph: vi.fn().mockResolvedValue({ nodes: [], edges: [] }),
      getSchemas: vi.fn().mockResolvedValue({
        input_schema: { properties: { messages: {}, tools: {} } },
        output_schema: { properties: { messages: {}, tools: {} } },
      }),
    },
    runs: {
      stream: vi.fn().mockImplementation((_t: string, _a: string, payload: any) => {
        streamCalls.push(payload);
        return {
          [Symbol.asyncIterator]() {
            return { next: async () => ({ done: true, value: undefined }) };
          },
        };
      }),
    },
  };

  const events: any[] = [];
  (agent as any).subscriber = {
    next: (e: any) => events.push(e),
    error: vi.fn(),
    complete: vi.fn(),
    closed: false,
  };

  return { agent, events, streamCalls };
}

const STREAM_MODE = ["events", "values", "updates", "messages-tuple"] as const;

describe("OSS-28 / #1278 SSE-drop recovery (TypeScript)", () => {
  it("recovers from a fresh-UUID resend as a continuation (no throw, no regenerate)", async () => {
    // Server finished the previous turn: checkpoint has Human + AI (2 non-system).
    const checkpointMessages = [
      { type: "human", id: "h1", content: "first question" },
      { type: "ai", id: "ai1", content: "first answer" },
    ];
    // Realistic history: only h1/ai1 were ever persisted -- the fresh client
    // UUID is nowhere in it. (If regenerate were taken, getCheckpointByMessage
    // would walk this and throw "Message not found".)
    const history = [
      {
        values: { messages: checkpointMessages },
        checkpoint: { checkpoint_id: "ck-1", checkpoint_ns: "" },
        parent_checkpoint: null,
        next: [],
      },
    ];
    const { agent } = buildAgent(checkpointMessages, history);

    // SSE dropped before MESSAGES_SNAPSHOT, so the client resends only the new
    // user message with a freshly generated UUID (1 non-system message).
    // 2 > 1, but the fresh UUID isn't in the checkpoint -> continuation, not
    // regeneration. Must not throw; must produce a normal stream.
    const input = {
      runId: "run-1",
      threadId: "thread-1",
      messages: [
        { id: "fresh-uuid-never-persisted", role: "user", content: "second question" },
      ],
      tools: [],
      context: [],
      forwardedProps: {},
    };

    const prepared = await agent.prepareStream(input as any, STREAM_MODE as any);

    expect(prepared).toBeTruthy();
    // Regenerate path not taken: the history lookup never happened.
    expect((agent as any).client.threads.getHistory).not.toHaveBeenCalled();
  });

  it("a genuine edit still routes into regenerate", async () => {
    // checkpoint: 4 non-system messages.
    const checkpointMessages = [
      { type: "human", id: "h1", content: "original" },
      { type: "ai", id: "ai1", content: "answer" },
      { type: "human", id: "h2", content: "regenerate from here" },
      { type: "ai", id: "ai2", content: "second answer" },
    ];
    const { agent } = buildAgent(checkpointMessages, []);
    // Spy out the regenerate machinery; we only assert routing here.
    const regenSpy = vi
      .fn()
      .mockResolvedValue({ streamResponse: {}, state: {}, streamMode: STREAM_MODE });
    (agent as any).prepareRegenerateStream = regenSpy;

    // An incoming id (h-edited) is NOT in the checkpoint -> not a plain
    // continuation; the LAST user id (h2) IS in the checkpoint -> genuine edit.
    const input = {
      runId: "run-1",
      threadId: "thread-1",
      messages: [
        { id: "h1", role: "user", content: "original" },
        { id: "h-edited", role: "user", content: "edited earlier turn" },
        { id: "h2", role: "user", content: "regenerate from here" },
      ],
      tools: [],
      context: [],
      forwardedProps: {},
    };

    await agent.prepareStream(input as any, STREAM_MODE as any);

    expect(regenSpy).toHaveBeenCalledTimes(1);
  });

  it("a genuine continuation (no count mismatch) does NOT throw", async () => {
    // Control: when the client is in sync (checkpoint count == input count),
    // there's no regenerate routing and no throw.
    const checkpointMessages = [
      { type: "human", id: "h1", content: "first question" },
    ];
    const { agent } = buildAgent(checkpointMessages, []);

    const input = {
      runId: "run-1",
      threadId: "thread-1",
      messages: [{ id: "h1", role: "user", content: "first question" }],
      tools: [],
      context: [],
      forwardedProps: {},
    };

    const prepared = await agent.prepareStream(input as any, STREAM_MODE as any);
    expect(prepared).toBeTruthy();
  });
});
