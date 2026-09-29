import { describe, it, expect, vi } from "vitest";
import { EventType } from "@ag-ui/client";
import type { BaseEvent, Interrupt } from "@ag-ui/client";
import { RunFinishedEventSchema } from "@ag-ui/core/schemas";
import { Agent } from "@mastra/core/agent";
import { Mastra } from "@mastra/core";
import { InMemoryStore } from "@mastra/core/storage";
import { MockMemory } from "@mastra/core/memory";
import { createTool } from "@mastra/core/tools";
import { MastraLanguageModelV2Mock } from "@mastra/core/test-utils/llm-mock";
import { z } from "zod";
import {
  FakeLocalAgent,
  FakeRemoteAgent,
  makeInput,
  collectEvents,
  collectError,
} from "./helpers";
import { MastraAgent } from "../mastra";

// Mastra pauses a tool that needs approval (`requireApproval` on the tool, or
// `requireToolApproval` on the agent) with a `tool-call-approval` chunk, not
// `tool-call-suspended`. The pending call lives in Mastra's snapshot and is
// completed by Mastra's own approve / decline, keyed by the original runId and
// toolCallId.

const APPROVAL_SCHEMA = JSON.stringify({
  type: "object",
  properties: { approved: { type: "boolean" } },
  required: ["approved"],
});

function approvalChunks(toolCallId = "tc-1", runId = "mastra-run-1") {
  return [
    {
      type: "tool-call",
      runId,
      payload: {
        toolCallId,
        toolName: "record_expense",
        args: { amount: 250, description: "team dinner" },
      },
    },
    {
      type: "tool-call-approval",
      runId,
      payload: {
        toolCallId,
        toolName: "record_expense",
        args: { amount: 250, description: "team dinner" },
        resumeSchema: APPROVAL_SCHEMA,
      },
    },
  ];
}

// What Mastra streams after an approved call: the tool's real result (without
// re-emitting the tool-call) and the model's follow-up.
function approvedResumeChunks(toolCallId = "tc-1") {
  return [
    {
      type: "tool-result",
      payload: {
        toolCallId,
        toolName: "record_expense",
        args: { amount: 250, description: "team dinner" },
        result: { recorded: true, amount: 250 },
      },
    },
    { type: "text-delta", payload: { text: "Recorded the expense." } },
  ];
}

function makeLocal(
  opts: { streamChunks?: any[]; resumeChunks?: any[] } = {},
  emitInterruptOutcome = true,
) {
  const fake = new FakeLocalAgent(opts);
  const agent = new MastraAgent({
    agentId: "test-agent",
    agent: fake as any,
    resourceId: "resource-1",
    emitInterruptOutcome,
  });
  return { agent, fake };
}

function makeRemote(opts: { streamChunks?: any[]; resumeChunks?: any[] } = {}) {
  const fake = new FakeRemoteAgent(opts);
  const agent = new MastraAgent({
    agentId: "test-agent",
    agent: fake as any,
    resourceId: "resource-1",
    emitInterruptOutcome: true,
  });
  return { agent, fake };
}

function legacyValue(events: BaseEvent[]) {
  const custom = events.find(
    (e) => e.type === EventType.CUSTOM && (e as any).name === "on_interrupt",
  ) as any;
  return custom ? JSON.parse(custom.value) : undefined;
}

function outcomeInterrupts(events: BaseEvent[]): Interrupt[] {
  const finished = events[events.length - 1] as any;
  expect(finished.type).toBe(EventType.RUN_FINISHED);
  return finished.outcome?.interrupts ?? [];
}

function legacyResume(value: unknown, resume: unknown) {
  return makeInput({
    runId: "run-2",
    forwardedProps: {
      command: { resume, interruptEvent: JSON.stringify(value) },
    },
  });
}

function toolEventsFor(events: BaseEvent[], toolCallId: string) {
  return events
    .filter(
      (e) =>
        [
          EventType.TOOL_CALL_START,
          EventType.TOOL_CALL_ARGS,
          EventType.TOOL_CALL_END,
          EventType.TOOL_CALL_RESULT,
        ].includes(e.type) && (e as any).toolCallId === toolCallId,
    )
    .map((e) => e.type);
}

describe("tool approval: pause surfaces an interrupt", () => {
  it("emits the legacy on_interrupt marked as an approval request", async () => {
    const { agent } = makeLocal({ streamChunks: approvalChunks() });
    const events = await collectEvents(agent, makeInput({ runId: "agui-1" }));

    expect(legacyValue(events)).toEqual({
      type: "mastra_tool_approval",
      toolCallId: "tc-1",
      toolName: "record_expense",
      args: { amount: 250, description: "team dinner" },
      resumeSchema: APPROVAL_SCHEMA,
      runId: "mastra-run-1",
    });
  });

  it("carries a canonical approval Interrupt on RUN_FINISHED.outcome", async () => {
    const { agent } = makeLocal({ streamChunks: approvalChunks() });
    const events = await collectEvents(agent, makeInput({ runId: "agui-1" }));

    const finished = events[events.length - 1];
    expect(RunFinishedEventSchema.safeParse(finished).success).toBe(true);
    const interrupts = outcomeInterrupts(events);
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]).toMatchObject({
      reason: "mastra:tool_approval",
      toolCallId: "tc-1",
      responseSchema: JSON.parse(APPROVAL_SCHEMA),
      metadata: {
        mastra: {
          type: "mastra_tool_approval",
          toolName: "record_expense",
          args: { amount: 250, description: "team dinner" },
          resumeSchema: APPROVAL_SCHEMA,
          runId: "mastra-run-1",
        },
      },
    });
  });

  it("suppresses the buffered tool call so no call without a result is emitted", async () => {
    const { agent } = makeLocal({ streamChunks: approvalChunks() });
    const events = await collectEvents(agent, makeInput());

    expect(toolEventsFor(events, "tc-1")).toEqual([]);
  });

  it("closes a live-streamed server call instead of leaving it open", async () => {
    const fake = new FakeLocalAgent({
      streamChunks: [
        {
          type: "tool-call-input-streaming-start",
          payload: { toolCallId: "tc-1", toolName: "record_expense" },
        },
        {
          type: "tool-call-delta",
          payload: { toolCallId: "tc-1", argsTextDelta: '{"amount":250}' },
        },
        ...approvalChunks(),
      ],
    });
    const agent = new MastraAgent({
      agentId: "test-agent",
      agent: fake as any,
      resourceId: "resource-1",
      streamServerToolCalls: true,
    });
    const events = await collectEvents(agent, makeInput());

    const types = toolEventsFor(events, "tc-1");
    expect(types[0]).toBe(EventType.TOOL_CALL_START);
    expect(types[types.length - 1]).toBe(EventType.TOOL_CALL_END);
    expect(types).not.toContain(EventType.TOOL_CALL_RESULT);
    expect(legacyValue(events)?.type).toBe("mastra_tool_approval");
  });

  it("works the same for remote agents", async () => {
    const { agent } = makeRemote({ streamChunks: approvalChunks() });
    const events = await collectEvents(agent, makeInput());

    expect(legacyValue(events)?.type).toBe("mastra_tool_approval");
    expect(outcomeInterrupts(events)[0]?.reason).toBe("mastra:tool_approval");
    expect(toolEventsFor(events, "tc-1")).toEqual([]);
  });

  it("errors on an approval chunk without a toolCallId", async () => {
    const { agent } = makeLocal({
      streamChunks: [
        {
          type: "tool-call-approval",
          payload: { toolName: "record_expense", args: {}, resumeSchema: "{}" },
        },
      ],
    });
    const { error } = await collectError(agent, makeInput());
    expect(error.message).toMatch(/tool-call-approval/);
  });
});

describe("tool approval: resume completes the original call natively", () => {
  it("approve calls Mastra's approveToolCall with the original runId and toolCallId", async () => {
    const { agent, fake } = makeLocal({
      streamChunks: approvalChunks(),
      resumeChunks: approvedResumeChunks(),
    });
    const value = legacyValue(await collectEvents(agent, makeInput()));
    fake.lastStreamOpts = null;

    await collectEvents(agent, legacyResume(value, { approved: true }));

    expect(fake.toolApprovalCalls).toHaveLength(1);
    expect(fake.toolApprovalCalls[0].approved).toBe(true);
    expect(fake.toolApprovalCalls[0].opts).toMatchObject({
      runId: "mastra-run-1",
      toolCallId: "tc-1",
      memory: { thread: "thread-1", resource: "resource-1" },
    });
    // No fresh run: the paused call is completed, not re-requested.
    expect(fake.lastStreamOpts).toBeNull();
  });

  it("decline still goes to Mastra's declineToolCall", async () => {
    const { agent, fake } = makeLocal({
      streamChunks: approvalChunks(),
      resumeChunks: [
        {
          type: "tool-result",
          payload: {
            toolCallId: "tc-1",
            toolName: "record_expense",
            result: "Tool call was not approved by the user",
          },
        },
        { type: "text-delta", payload: { text: "Okay, not recorded." } },
      ],
    });
    const value = legacyValue(await collectEvents(agent, makeInput()));

    const events = await collectEvents(agent, legacyResume(value, false));

    expect(fake.toolApprovalCalls).toEqual([
      {
        approved: false,
        opts: expect.objectContaining({
          runId: "mastra-run-1",
          toolCallId: "tc-1",
        }),
      },
    ]);
    const text = events.find(
      (e) => e.type === EventType.TEXT_MESSAGE_CHUNK,
    ) as any;
    expect(text.delta).toBe("Okay, not recorded.");
  });

  it("an explicit { approved: false } payload declines", async () => {
    const { agent, fake } = makeLocal({ streamChunks: approvalChunks() });
    const value = legacyValue(await collectEvents(agent, makeInput()));

    await collectEvents(agent, legacyResume(value, { approved: false }));

    expect(fake.toolApprovalCalls.map((c) => c.approved)).toEqual([false]);
  });

  it("approved resume emits START/ARGS/END for the original id before its RESULT", async () => {
    const { agent } = makeLocal({
      streamChunks: approvalChunks(),
      resumeChunks: approvedResumeChunks(),
    });
    const value = legacyValue(await collectEvents(agent, makeInput()));

    const events = await collectEvents(
      agent,
      legacyResume(value, { approved: true }),
    );

    expect(toolEventsFor(events, "tc-1")).toEqual([
      EventType.TOOL_CALL_START,
      EventType.TOOL_CALL_ARGS,
      EventType.TOOL_CALL_END,
      EventType.TOOL_CALL_RESULT,
    ]);
    const start = events.find(
      (e) => e.type === EventType.TOOL_CALL_START,
    ) as any;
    expect(start.toolCallName).toBe("record_expense");
    const args = events.find((e) => e.type === EventType.TOOL_CALL_ARGS) as any;
    expect(JSON.parse(args.delta)).toEqual({
      amount: 250,
      description: "team dinner",
    });
    const result = events.find(
      (e) => e.type === EventType.TOOL_CALL_RESULT,
    ) as any;
    expect(JSON.parse(result.content)).toEqual({ recorded: true, amount: 250 });
    expect(events[events.length - 1].type).toBe(EventType.RUN_FINISHED);
  });

  it("the canonical resume channel approves via the emitted interrupt id", async () => {
    const { agent, fake } = makeLocal({
      streamChunks: approvalChunks(),
      resumeChunks: approvedResumeChunks(),
    });
    const [interrupt] = outcomeInterrupts(
      await collectEvents(agent, makeInput()),
    );

    const events = await collectEvents(
      agent,
      makeInput({
        runId: "run-2",
        resume: [
          {
            interruptId: interrupt.id,
            status: "resolved",
            payload: { approved: true },
          },
        ],
      } as any),
    );

    expect(fake.toolApprovalCalls).toHaveLength(1);
    expect(fake.toolApprovalCalls[0]).toMatchObject({
      approved: true,
      opts: { runId: "mastra-run-1", toolCallId: "tc-1" },
    });
    expect(toolEventsFor(events, "tc-1")).toEqual([
      EventType.TOOL_CALL_START,
      EventType.TOOL_CALL_ARGS,
      EventType.TOOL_CALL_END,
      EventType.TOOL_CALL_RESULT,
    ]);
  });

  it("a cancelled canonical entry declines through Mastra", async () => {
    const { agent, fake } = makeLocal({ streamChunks: approvalChunks() });
    const [interrupt] = outcomeInterrupts(
      await collectEvents(agent, makeInput()),
    );

    await collectEvents(
      agent,
      makeInput({
        runId: "run-2",
        resume: [
          { interruptId: interrupt.id, status: "cancelled", payload: null },
        ],
      } as any),
    );

    expect(fake.toolApprovalCalls).toHaveLength(1);
    expect(fake.toolApprovalCalls[0]).toMatchObject({
      approved: false,
      opts: { runId: "mastra-run-1", toolCallId: "tc-1" },
    });
  });

  it("remote approve resumes the original call with { approved: true }", async () => {
    const { agent, fake } = makeRemote({
      streamChunks: approvalChunks(),
      resumeChunks: approvedResumeChunks(),
    });
    const value = legacyValue(await collectEvents(agent, makeInput()));

    const events = await collectEvents(
      agent,
      legacyResume(value, { approved: true }),
    );

    expect(fake.resumeCalls).toHaveLength(1);
    expect(fake.resumeCalls[0].resumeData).toEqual({ approved: true });
    expect(fake.resumeCalls[0].opts).toMatchObject({
      runId: "mastra-run-1",
      toolCallId: "tc-1",
    });
    expect(toolEventsFor(events, "tc-1")).toEqual([
      EventType.TOOL_CALL_START,
      EventType.TOOL_CALL_ARGS,
      EventType.TOOL_CALL_END,
      EventType.TOOL_CALL_RESULT,
    ]);
  });

  it("remote decline resumes the original call with { approved: false }", async () => {
    const { agent, fake } = makeRemote({
      streamChunks: approvalChunks(),
      resumeChunks: [],
    });
    const value = legacyValue(await collectEvents(agent, makeInput()));

    await collectEvents(agent, legacyResume(value, false));

    expect(fake.resumeCalls).toHaveLength(1);
    expect(fake.resumeCalls[0].resumeData).toEqual({ approved: false });
    expect(fake.resumeCalls[0].opts).toMatchObject({
      runId: "mastra-run-1",
      toolCallId: "tc-1",
    });
  });
});

describe("tool approval: ordinary suspend is unchanged", () => {
  it("resume: false on a suspend still closes the run without calling Mastra", async () => {
    const { agent, fake } = makeLocal({ streamChunks: [] });
    const resumeSpy = vi.spyOn(fake, "resumeStream");

    const events = await collectEvents(
      agent,
      legacyResume(
        { type: "mastra_suspend", toolCallId: "tc-1", runId: "r" },
        false,
      ),
    );

    expect(resumeSpy).not.toHaveBeenCalled();
    expect(fake.toolApprovalCalls).toHaveLength(0);
    expect(events.map((e) => e.type)).toEqual([
      EventType.RUN_STARTED,
      EventType.RUN_FINISHED,
    ]);
  });

  it("a frontend tool call still streams as a tool call with no interrupt", async () => {
    const { agent } = makeLocal({
      streamChunks: [
        {
          type: "tool-call-input-streaming-start",
          payload: { toolCallId: "tc-ui", toolName: "confirm_expense" },
        },
        {
          type: "tool-call-delta",
          payload: { toolCallId: "tc-ui", argsTextDelta: '{"amount":250}' },
        },
        {
          type: "tool-call",
          payload: {
            toolCallId: "tc-ui",
            toolName: "confirm_expense",
            args: { amount: 250 },
          },
        },
      ],
    });

    const events = await collectEvents(
      agent,
      makeInput({
        tools: [
          {
            name: "confirm_expense",
            description: "Ask the user to confirm",
            parameters: { type: "object", properties: {} },
          },
        ],
      }),
    );

    expect(toolEventsFor(events, "tc-ui")).toEqual([
      EventType.TOOL_CALL_START,
      EventType.TOOL_CALL_ARGS,
      EventType.TOOL_CALL_END,
    ]);
    expect(legacyValue(events)).toBeUndefined();
    expect(outcomeInterrupts(events)).toEqual([]);
  });

  it("a suspend resume passes its payload to resumeStream, not approveToolCall", async () => {
    const { agent, fake } = makeLocal({ streamChunks: [] });
    const resumeSpy = vi.spyOn(fake, "resumeStream");

    await collectEvents(
      agent,
      legacyResume(
        { type: "mastra_suspend", toolCallId: "tc-1", runId: "r" },
        { chosen_time: "2pm" },
      ),
    );

    expect(fake.toolApprovalCalls).toHaveLength(0);
    expect(resumeSpy).toHaveBeenCalledWith(
      { chosen_time: "2pm" },
      expect.objectContaining({ runId: "r", toolCallId: "tc-1" }),
    );
  });
});

// ---------------------------------------------------------------------------
// Real @mastra/core round trip: a genuine pending approval in Mastra storage,
// completed through the adapter. The model stand-in answers from the prompt it
// is given, so the follow-up only appears once the real tool result is in it.
// ---------------------------------------------------------------------------

function promptHasToolResult(prompt: unknown): boolean {
  return Array.isArray(prompt) && prompt.some((m: any) => m?.role === "tool");
}

function approvalModel() {
  return new MastraLanguageModelV2Mock({
    doStream: async ({ prompt }: { prompt: unknown }) => {
      const chunks = promptHasToolResult(prompt)
        ? [
            { type: "text-start", id: "t" },
            { type: "text-delta", id: "t", delta: "All done." },
            { type: "text-end", id: "t" },
            {
              type: "finish",
              usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
              finishReason: "stop",
            },
          ]
        : [
            {
              type: "tool-call",
              toolCallId: "tc-real",
              toolName: "record_expense",
              input: JSON.stringify({ amount: 250 }),
            },
            {
              type: "finish",
              usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
              finishReason: "tool-calls",
            },
          ];
      return {
        stream: new ReadableStream({
          start(controller) {
            for (const chunk of chunks) controller.enqueue(chunk);
            controller.close();
          },
        }),
        request: { body: {} },
        response: undefined,
      };
    },
  });
}

function realApprovalAgent(level: "tool" | "agent") {
  const execute = vi.fn(async (input: { amount: number }) => ({
    recordId: `expense-${input.amount}`,
    amount: input.amount,
  }));
  const tool = createTool({
    id: "record_expense",
    description: "Record an expense",
    inputSchema: z.object({ amount: z.number() }),
    ...(level === "tool" ? { requireApproval: true } : {}),
    execute,
  });
  const mastraAgent = new Agent({
    id: "approval-agent",
    name: "approval-agent",
    instructions: "Record expenses when asked.",
    model: approvalModel() as any,
    tools: { record_expense: tool },
    memory: new MockMemory() as any,
    ...(level === "agent"
      ? { defaultOptions: { requireToolApproval: true } }
      : {}),
  });
  const mastra = new Mastra({
    agents: { approval: mastraAgent },
    storage: new InMemoryStore(),
    logger: false,
  });
  const agent = new MastraAgent({
    agentId: "approval",
    agent: mastra.getAgent("approval") as any,
    resourceId: "resource-1",
  });
  return { agent, execute };
}

const userTurn = makeInput({
  runId: "run-1",
  messages: [{ id: "u1", role: "user", content: "Record a $250 expense" }],
});

describe.each(["tool", "agent"] as const)(
  "tool approval: real @mastra/core round trip (%s-level approval)",
  (level) => {
    it("pauses with an interrupt, then approve executes the original call", async () => {
      const { agent, execute } = realApprovalAgent(level);

      const first = await collectEvents(agent, userTurn);
      expect(execute).not.toHaveBeenCalled();
      const [interrupt] = outcomeInterrupts(first);
      expect(interrupt).toMatchObject({
        reason: "mastra:tool_approval",
        toolCallId: "tc-real",
      });
      expect(legacyValue(first)?.type).toBe("mastra_tool_approval");

      const resumed = await collectEvents(
        agent,
        makeInput({
          runId: "run-2",
          resume: [
            {
              interruptId: interrupt.id,
              status: "resolved",
              payload: { approved: true },
            },
          ],
        } as any),
      );

      expect(execute).toHaveBeenCalledTimes(1);
      expect(execute.mock.calls[0][0]).toEqual({ amount: 250 });
      expect(toolEventsFor(resumed, "tc-real")).toEqual([
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_RESULT,
      ]);
      const result = resumed.find(
        (e) => e.type === EventType.TOOL_CALL_RESULT,
      ) as any;
      expect(JSON.parse(result.content)).toEqual({
        recordId: "expense-250",
        amount: 250,
      });
      const text = resumed
        .filter((e) => e.type === EventType.TEXT_MESSAGE_CHUNK)
        .map((e: any) => e.delta)
        .join("");
      expect(text).toBe("All done.");
      expect(resumed[resumed.length - 1].type).toBe(EventType.RUN_FINISHED);
    });

    it("decline resolves the pending approval in Mastra without executing", async () => {
      const { agent, execute } = realApprovalAgent(level);
      const value = legacyValue(await collectEvents(agent, userTurn));

      const resumed = await collectEvents(agent, legacyResume(value, false));

      expect(execute).not.toHaveBeenCalled();
      const result = resumed.find(
        (e) => e.type === EventType.TOOL_CALL_RESULT,
      ) as any;
      expect(result.toolCallId).toBe("tc-real");
      expect(JSON.parse(result.content)).toMatch(/not approved/);
      const text = resumed
        .filter((e) => e.type === EventType.TEXT_MESSAGE_CHUNK)
        .map((e: any) => e.delta)
        .join("");
      expect(text).toBe("All done.");
    });
  },
);
