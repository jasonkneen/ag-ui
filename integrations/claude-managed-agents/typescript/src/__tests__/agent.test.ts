import { EventType } from "@ag-ui/client";
import type { BaseEvent, RunAgentInput } from "@ag-ui/client";
import { lastValueFrom, toArray } from "rxjs";
import { describe, expect, it } from "vitest";
import { ManagedAgentsAgent } from "../agent";
import { InMemorySessionStore } from "../sessions";
import { createFakeClient } from "./fake-client";

const idleEndTurn = { type: "session.status_idle", id: "idle_1", stop_reason: { type: "end_turn" } };

const baseInput = (overrides: Partial<RunAgentInput> = {}): RunAgentInput => ({
  threadId: "thread_1",
  runId: "run_1",
  state: {},
  messages: [{ id: "u1", role: "user", content: "Hello" }],
  tools: [],
  context: [],
  forwardedProps: {},
  ...overrides,
});

const collect = async (agent: ManagedAgentsAgent, input: RunAgentInput): Promise<BaseEvent[]> =>
  (await lastValueFrom(agent.run(input).pipe(toArray()))) as BaseEvent[];

const types = (events: BaseEvent[]) => events.map((event) => event.type);

const newAgent = (fake: ReturnType<typeof createFakeClient>, store = new InMemorySessionStore()) =>
  new ManagedAgentsAgent({
    agentId: "agent_1",
    environmentId: "env_1",
    client: fake.client,
    sessionStore: store,
  });

describe("ManagedAgentsAgent", () => {
  it("creates a session for a new thread and streams a reply", async () => {
    const fake = createFakeClient({
      streams: [[{ type: "agent.message", id: "msg_1", content: [{ type: "text", text: "Hi!" }] }, idleEndTurn]],
    });
    const events = await collect(newAgent(fake), baseInput());

    expect(fake.spies.create).toHaveBeenCalledWith({
      agent: { type: "agent", id: "agent_1" },
      environment_id: "env_1",
      title: "AG-UI thread thread_1",
    });
    expect(types(events)).toEqual([
      EventType.RUN_STARTED,
      EventType.STATE_SNAPSHOT,
      EventType.CUSTOM,
      EventType.TEXT_MESSAGE_START,
      EventType.TEXT_MESSAGE_CONTENT,
      EventType.TEXT_MESSAGE_END,
      EventType.RUN_FINISHED,
    ]);
    expect(events[2]).toMatchObject({ name: "managed_agents.session", value: { sessionId: "sesn_1", threadId: "thread_1" } });
    expect(fake.sent[0].events).toEqual([{ type: "user.message", content: [{ type: "text", text: "Hello" }] }]);
  });

  it("reuses the session on the thread's next run and sends only the new message", async () => {
    const fake = createFakeClient({
      streams: [
        [{ type: "agent.message", id: "msg_1", content: [{ type: "text", text: "one" }] }, idleEndTurn],
        [{ type: "agent.message", id: "msg_2", content: [{ type: "text", text: "two" }] }, idleEndTurn],
      ],
    });
    const store = new InMemorySessionStore();
    await collect(newAgent(fake, store), baseInput());
    await collect(
      newAgent(fake, store),
      baseInput({
        runId: "run_2",
        messages: [
          { id: "u1", role: "user", content: "Hello" },
          { id: "a1", role: "assistant", content: "one" },
          { id: "u2", role: "user", content: "Follow-up" },
        ],
      }),
    );

    expect(fake.spies.create).toHaveBeenCalledTimes(1);
    expect(fake.sent[1].events).toEqual([{ type: "user.message", content: [{ type: "text", text: "Follow-up" }] }]);
  });

  it("registers frontend tools as custom tools when creating the session", async () => {
    const fake = createFakeClient({
      streams: [[idleEndTurn]],
      agentTools: [{ type: "agent_toolset_20260401", configs: [], default_config: {} }],
    });
    await collect(
      newAgent(fake),
      baseInput({
        tools: [{ name: "show_chart", description: "Render a chart", parameters: { type: "object", properties: { title: { type: "string" } } } }],
      }),
    );

    expect(fake.spies.create).toHaveBeenCalledWith(
      expect.objectContaining({
        agent: {
          type: "agent_with_overrides",
          id: "agent_1",
          tools: [
            { type: "agent_toolset_20260401", configs: [], default_config: {} },
            {
              type: "custom",
              name: "show_chart",
              description: "Render a chart",
              input_schema: { type: "object", properties: { title: { type: "string" } }, required: [] },
            },
          ],
        },
      }),
    );
  });

  it("round-trips a frontend tool: park, then resume with the client's result", async () => {
    const fake = createFakeClient({
      streams: [
        [
          { type: "agent.custom_tool_use", id: "ctu_1", name: "show_chart", input: { title: "Sales" } },
          { type: "session.status_idle", id: "idle_1", stop_reason: { type: "requires_action", event_ids: ["ctu_1"] } },
        ],
        [{ type: "agent.message", id: "msg_1", content: [{ type: "text", text: "Chart shown." }] }, idleEndTurn],
      ],
    });
    const store = new InMemorySessionStore();
    const tools = [{ name: "show_chart", description: "Render a chart", parameters: { type: "object" } }];

    const first = await collect(newAgent(fake, store), baseInput({ tools }));
    expect(types(first)).toEqual([
      EventType.RUN_STARTED,
      EventType.STATE_SNAPSHOT,
      EventType.CUSTOM,
      EventType.TOOL_CALL_START,
      EventType.TOOL_CALL_ARGS,
      EventType.TOOL_CALL_END,
      EventType.RUN_FINISHED,
    ]);
    expect(await store.get("thread_1")).toMatchObject({ pendingClientToolUseIds: ["ctu_1"] });

    const second = await collect(
      newAgent(fake, store),
      baseInput({
        runId: "run_2",
        tools,
        messages: [
          { id: "u1", role: "user", content: "Hello" },
          { id: "t1", role: "tool", toolCallId: "ctu_1", content: "rendered" },
        ],
      }),
    );

    expect(fake.sent[1].events).toEqual([
      { type: "user.custom_tool_result", custom_tool_use_id: "ctu_1", content: [{ type: "text", text: "rendered" }], is_error: false },
    ]);
    expect(types(second)).toContain(EventType.TEXT_MESSAGE_CONTENT);
    expect(second.at(-1)?.type).toBe(EventType.RUN_FINISHED);
    expect(await store.get("thread_1")).toMatchObject({ pendingClientToolUseIds: [] });
  });

  it("shares the session store across clones so a resumed run finds its parked session", async () => {
    const fake = createFakeClient({
      streams: [
        [
          { type: "agent.custom_tool_use", id: "ctu_1", name: "show_chart", input: {} },
          { type: "session.status_idle", id: "idle_1", stop_reason: { type: "requires_action", event_ids: ["ctu_1"] } },
        ],
        [{ type: "agent.message", id: "msg_1", content: [{ type: "text", text: "Done." }] }, idleEndTurn],
      ],
    });
    // No sessionStore passed: the default in-memory store must be shared by clones.
    const parent = new ManagedAgentsAgent({ agentId: "agent_1", environmentId: "env_1", client: fake.client });
    const tools = [{ name: "show_chart", description: "Render a chart", parameters: { type: "object" } }];

    await collect(parent.clone(), baseInput({ tools }));
    await collect(
      parent.clone(),
      baseInput({
        runId: "run_2",
        tools,
        messages: [
          { id: "u1", role: "user", content: "Hello" },
          { id: "t1", role: "tool", toolCallId: "ctu_1", content: "rendered" },
        ],
      }),
    );

    expect(fake.spies.create).toHaveBeenCalledTimes(1);
    expect(fake.sent[1].events).toEqual([
      { type: "user.custom_tool_result", custom_tool_use_id: "ctu_1", content: [{ type: "text", text: "rendered" }], is_error: false },
    ]);
  });

  it("forwards every undelivered user message in order", async () => {
    const fake = createFakeClient({ streams: [[idleEndTurn]] });
    const store = new InMemorySessionStore();
    await store.set("thread_1", { sessionId: "sesn_1", toolNames: [], pendingClientToolUseIds: [], lastUserMessageId: "u1" });

    await collect(
      newAgent(fake, store),
      baseInput({
        messages: [
          { id: "u1", role: "user", content: "delivered" },
          { id: "u2", role: "user", content: "second" },
          { id: "u3", role: "user", content: "third" },
        ],
      }),
    );

    expect(fake.sent[0].events).toEqual([
      { type: "user.message", content: [{ type: "text", text: "second" }] },
      { type: "user.message", content: [{ type: "text", text: "third" }] },
    ]);
    expect(await store.get("thread_1")).toMatchObject({ lastUserMessageId: "u3" });
  });

  it("abandons parked tool calls when the user sends a new message instead", async () => {
    const fake = createFakeClient({
      streams: [[{ type: "agent.message", id: "msg_1", content: [{ type: "text", text: "Moving on." }] }, idleEndTurn]],
    });
    const store = new InMemorySessionStore();
    await store.set("thread_1", { sessionId: "sesn_1", toolNames: [], pendingClientToolUseIds: ["ctu_1"], lastUserMessageId: "u1" });

    const events = await collect(
      newAgent(fake, store),
      baseInput({
        messages: [
          { id: "u1", role: "user", content: "old" },
          { id: "u2", role: "user", content: "never mind" },
        ],
      }),
    );

    // Tool results go first (resuming the parked session), then the message,
    // as two separate sends: the API rejects a user.message in the same batch
    // as the results while the session is still parked.
    expect(fake.sent[0].events).toEqual([
      {
        type: "user.custom_tool_result",
        custom_tool_use_id: "ctu_1",
        content: [{ type: "text", text: "The user did not provide a result for this tool call." }],
        is_error: true,
      },
    ]);
    expect(fake.sent[1].events).toEqual([{ type: "user.message", content: [{ type: "text", text: "never mind" }] }]);
    expect(events.at(-1)?.type).toBe(EventType.RUN_FINISHED);
    expect(await store.get("thread_1")).toMatchObject({ pendingClientToolUseIds: [], lastUserMessageId: "u2" });
  });

  it("partitions sessions by scope so identical thread ids do not collide", async () => {
    const fake = createFakeClient({
      streams: [
        [{ type: "agent.message", id: "msg_1", content: [{ type: "text", text: "a" }] }, idleEndTurn],
        [{ type: "agent.message", id: "msg_2", content: [{ type: "text", text: "b" }] }, idleEndTurn],
      ],
    });
    const store = new InMemorySessionStore();
    const scoped = (scope: string) =>
      new ManagedAgentsAgent({ agentId: "agent_1", environmentId: "env_1", client: fake.client, sessionStore: store, scope });

    await collect(scoped("alice"), baseInput());
    await collect(scoped("bob"), baseInput());

    // Same threadId, different scopes: two sessions and two store entries.
    expect(fake.spies.create).toHaveBeenCalledTimes(2);
    expect(await store.get("alice:thread_1")).toBeDefined();
    expect(await store.get("bob:thread_1")).toBeDefined();
    expect(await store.get("thread_1")).toBeUndefined();
  });

  it("does not create a session for a tool result on an unknown thread", async () => {
    const fake = createFakeClient({ streams: [[idleEndTurn]] });
    const events = await collect(
      newAgent(fake),
      baseInput({ messages: [{ id: "t1", role: "tool", toolCallId: "ctu_ghost", content: "late result" }] }),
    );
    expect(fake.spies.create).not.toHaveBeenCalled();
    expect(events.at(-1)).toMatchObject({ type: EventType.RUN_ERROR });
  });

  it("errors when a run has nothing new to send", async () => {
    const fake = createFakeClient({ streams: [[idleEndTurn]] });
    const store = new InMemorySessionStore();
    await store.set("thread_1", { sessionId: "sesn_1", toolNames: [], pendingClientToolUseIds: [], lastUserMessageId: "u1" });

    const events = await collect(newAgent(fake, store), baseInput());
    expect(events.at(-1)).toMatchObject({ type: EventType.RUN_ERROR });
  });

  it("updates the session's tools when the frontend adds a new one", async () => {
    const fake = createFakeClient({ streams: [[idleEndTurn], [idleEndTurn]] });
    const store = new InMemorySessionStore();
    await collect(newAgent(fake, store), baseInput());
    expect(fake.spies.update).not.toHaveBeenCalled();

    await collect(
      newAgent(fake, store),
      baseInput({
        runId: "run_2",
        messages: [
          { id: "u1", role: "user", content: "Hello" },
          { id: "u2", role: "user", content: "Show me a chart" },
        ],
        tools: [{ name: "show_chart", description: "Render a chart", parameters: { type: "object" } }],
      }),
    );

    expect(fake.spies.update).toHaveBeenCalledWith("sesn_1", {
      agent: {
        tools: [
          { type: "agent_toolset_20260401", configs: [], default_config: {} },
          { type: "custom", name: "show_chart", description: "Render a chart", input_schema: { type: "object", properties: {}, required: [] } },
        ],
      },
    });
  });
});
