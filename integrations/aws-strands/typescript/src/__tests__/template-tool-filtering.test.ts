/**
 * Per-request filtering of the tools the template agent contributed.
 *
 * The adapter keeps one Strands `Agent` per thread and that instance is
 * load-bearing: it holds the thread's `SessionManager`, its native interrupt
 * checkpoint and its conversation history. So the property these tests exist to
 * pin is not only that a filter takes effect, but that it takes effect on the
 * registry the live instance already owns. A filter applied by rebuilding the
 * thread's agent would pass a "the model saw fewer tools" assertion while
 * silently discarding a conversation and any approval waiting inside it, which
 * is why the identity of the cached agent is asserted alongside the tool specs.
 *
 * Driven through the real SDK: a genuine `Agent`, its real `ToolRegistry`, and
 * a scripted model that records the tool specs each turn was offered. The
 * offered specs are the only place the filter is observable from outside, so a
 * registry assertion alone would not show the model was actually affected.
 */

import { describe, it, expect, vi } from "vitest";
import { EventType, type BaseEvent, type RunAgentInput } from "@ag-ui/core";
import { Agent as StrandsAgentCore, type Tool } from "@strands-agents/sdk";

import { StrandsAgent } from "../agent";
import type { StrandsAgentConfig } from "../config";
import {
  indexTemplateTools,
  parkedBatchToolNames,
  resolveTemplateToolSelection,
  syncTemplateTools,
} from "../template-tools";
import type { StrandsToolRegistry } from "../client-proxy-tool";
import {
  collect,
  errorCodes,
  fakeTool,
  finishedOf,
  interruptsOf,
  minimalRunInput,
  modelTurn,
  realStrandsAgent,
  recordingTool,
  threadAgent,
  ScriptedModel,
} from "./helpers";

const READ = "read_docs";
const DELETE = "delete_record";

/** Two template tools and a two-turn script that answers with text. */
function twoToolAgent(config: StrandsAgentConfig) {
  const read = recordingTool(READ);
  const del = recordingTool(DELETE);
  const { agent, model } = realStrandsAgent(
    [modelTurn.text("first"), modelTurn.text("second")],
    { tools: [read.tool, del.tool], config },
  );
  return { agent, model, readCalls: read.calls, deleteCalls: del.calls };
}

function userTurn(overrides: Partial<RunAgentInput> = {}): RunAgentInput {
  return minimalRunInput({
    messages: [{ id: "u1", role: "user", content: "do the thing" } as never],
    ...overrides,
  });
}

function registryNames(
  agent: StrandsAgent,
  threadId = "thread-1",
): Set<string> {
  const core = threadAgent(agent, threadId);
  if (!core) throw new Error(`no per-thread agent for "${threadId}"`);
  return new Set(core.toolRegistry.list().map((t) => t.name));
}

function offered(model: ScriptedModel): Set<string> {
  const last = model.offeredToolNames.at(-1);
  expect(last, "the model was never invoked").toBeDefined();
  return last!;
}

// ---------------------------------------------------------------------------
// The rescope: filtering happens on the live agent, not by replacing it
// ---------------------------------------------------------------------------

describe("filtering without rebuilding the thread agent", () => {
  it("varies the filtered set between two requests on one thread", async () => {
    const byRun: Record<string, string[]> = {
      "run-1": [READ],
      "run-2": [READ, DELETE],
    };
    const { agent, model } = twoToolAgent({
      templateToolsProvider: (input) => byRun[input.runId],
    });

    await collect(agent, userTurn());
    expect(offered(model)).toEqual(new Set([READ]));
    expect(registryNames(agent)).toEqual(new Set([READ]));

    await collect(agent, userTurn({ runId: "run-2" }));
    expect(offered(model)).toEqual(new Set([READ, DELETE]));
    expect(registryNames(agent)).toEqual(new Set([READ, DELETE]));
  });

  it("keeps the same cached thread agent across the change", async () => {
    // The whole point of applying the filter to the registry. Recreating the
    // per-thread agent whenever the resolved tool set changed would satisfy the
    // assertion above and still be wrong, so identity is asserted directly.
    const byRun: Record<string, string[]> = { "run-1": [READ], "run-2": [] };
    const { agent, model } = twoToolAgent({
      templateToolsProvider: (input) => byRun[input.runId],
    });

    await collect(agent, userTurn());
    const first = threadAgent(agent);
    expect(first).toBeDefined();

    await collect(agent, userTurn({ runId: "run-2" }));
    expect(threadAgent(agent)).toBe(first);
    expect(offered(model)).toEqual(new Set());
  });

  it("withholds every template tool for an empty selection", async () => {
    const { agent, model } = twoToolAgent({ templateToolsProvider: () => [] });
    await collect(agent, userTurn());
    expect(offered(model)).toEqual(new Set());
  });

  it("treats null and undefined as declining to filter that request", async () => {
    const byRun: Record<string, string[] | null | undefined> = {
      "run-1": [],
      "run-2": null,
      "run-3": undefined,
    };
    const { agent, model } = twoToolAgent({
      templateToolsProvider: (input) => byRun[input.runId],
    });

    await collect(agent, userTurn());
    expect(offered(model)).toEqual(new Set());

    await collect(agent, userTurn({ runId: "run-2" }));
    expect(offered(model)).toEqual(new Set([READ, DELETE]));

    await collect(agent, userTurn({ runId: "run-3" }));
    expect(offered(model)).toEqual(new Set([READ, DELETE]));
  });

  it("may be async and may read the caller identity off the request", async () => {
    const { agent, model } = twoToolAgent({
      templateToolsProvider: async (input) =>
        (input.forwardedProps as { role?: string } | undefined)?.role ===
        "admin"
          ? null
          : [READ],
    });

    await collect(agent, userTurn({ forwardedProps: { role: "reader" } }));
    expect(offered(model)).toEqual(new Set([READ]));

    await collect(
      agent,
      userTurn({ runId: "run-2", forwardedProps: { role: "admin" } }),
    );
    expect(offered(model)).toEqual(new Set([READ, DELETE]));
  });

  it("lets two threads see different tools at the same time", async () => {
    const { agent, model } = twoToolAgent({
      templateToolsProvider: (input) =>
        input.threadId === "wide" ? null : [READ],
    });

    await collect(agent, userTurn({ threadId: "wide" }));
    expect(offered(model)).toEqual(new Set([READ, DELETE]));

    await collect(agent, userTurn({ threadId: "narrow", runId: "run-2" }));
    expect(offered(model)).toEqual(new Set([READ]));

    expect(registryNames(agent, "wide")).toEqual(new Set([READ, DELETE]));
    expect(registryNames(agent, "narrow")).toEqual(new Set([READ]));
  });
});

describe("what the thread keeps across a filter change", () => {
  it("keeps its history and the calls a filtered-out tool already made", async () => {
    // History is not rewritten: the filter answers "what may this request
    // call", not "what happened on this thread". Removing the record would
    // leave an assistant tool-use block with no result behind it.
    const del = recordingTool(DELETE);
    const read = recordingTool(READ);
    const { agent, model } = realStrandsAgent(
      [
        modelTurn.toolUse({ toolUseId: "tu-1", name: DELETE, input: {} }),
        modelTurn.text("done"),
        modelTurn.text("second turn"),
      ],
      {
        tools: [read.tool, del.tool],
        config: {
          replayHistoryIntoStrands: false,
          templateToolsProvider: (input) =>
            input.runId === "run-1" ? null : [READ],
        },
      },
    );

    await collect(agent, userTurn());
    const core = threadAgent(agent)!;
    expect(del.calls, "the first run never called the tool").toHaveLength(1);

    // Live history carries `ToolUseBlock` instances, which name the tool
    // directly; a serialized message nests the same fields under `toolUse`.
    const deleteCallIds = (messages: typeof core.messages) =>
      messages.flatMap((message) =>
        (message.content ?? [])
          .map((block) => {
            const b = block as {
              name?: string;
              toolUseId?: string;
              toolUse?: { name?: string; toolUseId?: string };
            };
            return b.toolUse ?? b;
          })
          .filter((use) => use.name === DELETE)
          .map((use) => use.toolUseId as string),
      );
    const recorded = deleteCallIds(core.messages);
    expect(recorded.length).toBeGreaterThan(0);

    await collect(agent, userTurn({ runId: "run-2" }));
    expect(threadAgent(agent)).toBe(core);
    expect(
      deleteCallIds(core.messages),
      "filtering the tool out erased the call it already made",
    ).toEqual(recorded);
    expect(offered(model)).toEqual(new Set([READ]));
  });

  it("keeps the session manager the thread was built with", async () => {
    const sessionManager = {
      initAgent: vi.fn(async () => {}),
      appendMessage: vi.fn(async () => {}),
      redactLatestMessage: vi.fn(async () => {}),
      syncAgent: vi.fn(async () => {}),
    };
    const { agent } = twoToolAgent({
      sessionManagerProvider: () => sessionManager as never,
      templateToolsProvider: (input) =>
        input.runId === "run-1" ? null : [READ],
    });

    await collect(agent, userTurn());
    const core = threadAgent(agent)!;
    const held = (core as unknown as { sessionManager?: unknown })
      .sessionManager;
    expect(held).toBe(sessionManager);

    await collect(agent, userTurn({ runId: "run-2" }));
    expect(threadAgent(agent)).toBe(core);
    expect(
      (core as unknown as { sessionManager?: unknown }).sessionManager,
    ).toBe(sessionManager);
  });
});

describe("a parked call is not orphaned", () => {
  it("keeps a tool awaiting approval registered while it is filtered out", async () => {
    // The rule `syncProxyTools` already applies, read off the checkpoint. The
    // human's answer is routed back into the tool batch the run stopped inside,
    // so a tool absent from the registry at that moment turns the answer into a
    // "tool not found" the model then re-fires.
    const del = recordingTool(DELETE);
    const read = recordingTool(READ);
    const { agent } = realStrandsAgent(
      [
        modelTurn.toolUse({ toolUseId: "tu-1", name: DELETE, input: {} }),
        modelTurn.text("done"),
      ],
      {
        tools: [read.tool, del.tool],
        config: {
          toolBehaviors: { [DELETE]: { interruptOnCall: true } },
          templateToolsProvider: (input) =>
            input.runId === "run-1" ? null : [READ],
        },
      },
    );

    const first = await collect(agent, userTurn());
    expect(errorCodes(first)).toEqual([]);
    expect(finishedOf(first).outcome?.type).toBe("interrupt");
    const interruptId = interruptsOf(first)[0].id;
    const core = threadAgent(agent)!;

    const second = await collect(
      agent,
      userTurn({
        runId: "run-2",
        resume: [
          { interruptId, status: "resolved", payload: { approved: true } },
        ] as never,
      }),
    );

    expect(threadAgent(agent)).toBe(core);
    expect(
      core.toolRegistry.get(DELETE),
      "the tool the resume routes back into was filtered out from under it",
    ).toBeDefined();
    expect(errorCodes(second)).toEqual([]);
    expect(del.calls, "the approved tool never ran").toHaveLength(1);
  });

  it("leaves a parked frontend proxy alone when the filter allows nothing", async () => {
    // A template filter has no reach over client-declared tools. Their proxies
    // are a different producer's entries and are re-synchronised from
    // `RunAgentInput.tools` every request, so the filter must not remove one.
    const clientTool = {
      name: "confirm_in_client",
      description: "Confirm in the client",
      parameters: { type: "object", properties: {} },
    };
    const { agent } = twoToolAgent({
      toolBehaviors: {
        confirm_in_client: { continueAfterFrontendCall: false },
      },
      templateToolsProvider: () => [],
    });

    await collect(agent, userTurn({ tools: [clientTool] as never }));
    const core = threadAgent(agent)!;
    expect(new Set(core.toolRegistry.list().map((t) => t.name))).toEqual(
      new Set(["confirm_in_client"]),
    );

    await collect(
      agent,
      userTurn({ runId: "run-2", tools: [clientTool] as never }),
    );
    expect(threadAgent(agent)).toBe(core);
    expect(core.toolRegistry.get("confirm_in_client")).toBeDefined();
  });
});

describe("filtering removes the capability", () => {
  it("does not run a filtered-out tool the model calls anyway", async () => {
    // The point of touching the registry rather than only the tool specs.
    // Withholding a tool from the specs and leaving it registered would make
    // the filter advice a model can ignore. A model calling the name anyway,
    // because it was primed by a stale turn or by the visible history, has to
    // be refused by the dispatcher rather than served.
    const del = recordingTool(DELETE);
    const read = recordingTool(READ);
    const { agent, model } = realStrandsAgent(
      [
        modelTurn.toolUse({ toolUseId: "tu-1", name: DELETE, input: {} }),
        modelTurn.text("done"),
      ],
      {
        tools: [read.tool, del.tool],
        config: { templateToolsProvider: () => [READ] },
      },
    );

    const events = await collect(agent, userTurn());

    expect(del.calls, "a filtered-out tool executed").toEqual([]);
    expect(model.offeredToolNames[0]).toEqual(new Set([READ]));
    expect(errorCodes(events)).toEqual([]);
  });
});

describe("a pause is out of the filter's reach on a plain turn", () => {
  it("refuses the turn before the filter runs and keeps the pause intact", async () => {
    // A turn that submits no answer is refused ahead of the tool sync, so a
    // provider that would have withheld the parked tool never runs and the
    // checkpoint is still there for the resume that follows.
    const consulted: string[] = [];
    const del = recordingTool(DELETE);
    const read = recordingTool(READ);
    const { agent } = realStrandsAgent(
      [
        modelTurn.toolUse({ toolUseId: "tu-1", name: DELETE, input: {} }),
        modelTurn.text("done"),
      ],
      {
        tools: [read.tool, del.tool],
        config: {
          toolBehaviors: { [DELETE]: { interruptOnCall: true } },
          templateToolsProvider: (input) => {
            consulted.push(input.runId);
            return input.runId === "run-1" ? null : [READ];
          },
          logger: { debug() {}, warn() {}, error() {} },
        },
      },
    );

    const first = await collect(agent, userTurn());
    const interruptId = interruptsOf(first)[0].id;
    const core = threadAgent(agent)!;

    const refused = await collect(agent, userTurn({ runId: "run-2" }));
    expect(errorCodes(refused)).toEqual(["PENDING_INTERRUPTS"]);
    expect(consulted, "the provider ran on a turn that was refused").toEqual([
      "run-1",
    ]);
    expect(core.toolRegistry.get(DELETE)).toBeDefined();

    const resumed = await collect(
      agent,
      userTurn({
        runId: "run-3",
        resume: [
          { interruptId, status: "resolved", payload: { approved: true } },
        ] as never,
      }),
    );
    expect(errorCodes(resumed)).toEqual([]);
    expect(threadAgent(agent)).toBe(core);
    expect(del.calls).toHaveLength(1);
    expect(consulted).toEqual(["run-1", "run-3"]);
  });
});

describe("the provider failure mode", () => {
  it("ends the run rather than running unfiltered", async () => {
    // Terminal, matching `threadAgentConfig`. Degrading to an unfiltered run
    // would hand the model exactly the tools the caller meant to withhold,
    // which is the one outcome this hook exists to prevent.
    const { agent, model } = twoToolAgent({
      templateToolsProvider: () => {
        throw new Error("authz lookup failed");
      },
      logger: { debug() {}, warn() {}, error() {} },
    });

    const events = await collect(agent, userTurn());
    expect(events.map((e) => e.type)).toEqual([
      EventType.RUN_STARTED,
      EventType.RUN_ERROR,
    ]);
    const error = events[1] as BaseEvent & { code?: string; message?: string };
    expect(error.code).toBe("TEMPLATE_TOOLS_PROVIDER_ERROR");
    expect(error.message).toContain("authz lookup failed");
    expect(model.calls, "the model ran despite the failure").toBe(0);
  });

  it("reports a rejected promise the same way", async () => {
    const { agent } = twoToolAgent({
      templateToolsProvider: async () => {
        throw new Error("network down");
      },
      logger: { debug() {}, warn() {}, error() {} },
    });
    expect(errorCodes(await collect(agent, userTurn()))).toEqual([
      "TEMPLATE_TOOLS_PROVIDER_ERROR",
    ]);
  });
});

describe("no provider configured", () => {
  it("leaves the registry and the offered specs untouched", async () => {
    const { agent, model } = twoToolAgent({});
    await collect(agent, userTurn());
    // Identity, not just names: an unconfigured hook must add no step at all,
    // so nothing may be removed and re-registered even to the same effect.
    const held = new Map(
      threadAgent(agent)!
        .toolRegistry.list()
        .map((t) => [t.name, t]),
    );
    expect([...held.keys()].sort()).toEqual([DELETE, READ]);

    await collect(agent, userTurn({ runId: "run-2" }));
    for (const [name, tool] of held) {
      expect(threadAgent(agent)!.toolRegistry.get(name)).toBe(tool);
    }
    expect(model.offeredToolNames).toEqual([
      new Set([READ, DELETE]),
      new Set([READ, DELETE]),
    ]);
  });
});

// ---------------------------------------------------------------------------
// The pieces, in isolation
// ---------------------------------------------------------------------------

describe("selection resolution", () => {
  const read = fakeTool(READ) as unknown as Tool;
  const del = fakeTool(DELETE) as unknown as Tool;
  const index = indexTemplateTools([read, del]);

  it("accepts names and tool objects alike", () => {
    expect(resolveTemplateToolSelection([READ], index)).toEqual(
      new Set([READ]),
    );
    expect(resolveTemplateToolSelection([del], index)).toEqual(
      new Set([DELETE]),
    );
  });

  it("distinguishes declining to filter from filtering everything out", () => {
    expect(resolveTemplateToolSelection(null, index)).toBeNull();
    expect(resolveTemplateToolSelection(undefined, index)).toBeNull();
    expect(resolveTemplateToolSelection([], index)).toEqual(new Set());
  });

  it("refuses a tool the template never contributed", () => {
    const warn = vi.fn();
    const log = { debug() {}, warn, error() {} };
    const smuggled = fakeTool(
      "smuggled",
      "not on the template",
    ) as unknown as Tool;
    expect(
      resolveTemplateToolSelection([READ, "smuggled", smuggled], index, log),
    ).toEqual(new Set([READ]));
    expect(warn).toHaveBeenCalledTimes(2);
    expect(warn.mock.calls.every(([m]) => /does not contribute/.test(m))).toBe(
      true,
    );
  });

  it("refuses an entry that names no tool", () => {
    const warn = vi.fn();
    const log = { debug() {}, warn, error() {} };
    expect(
      resolveTemplateToolSelection([null, 7, ""] as never, index, log),
    ).toEqual(new Set());
    expect(warn).toHaveBeenCalledTimes(3);
  });
});

describe("syncTemplateTools", () => {
  function registry(): { registry: StrandsToolRegistry; tools: Tool[] } {
    const core = new StrandsAgentCore({
      model: new ScriptedModel([]),
      tools: [recordingTool(READ).tool, recordingTool(DELETE).tool] as never,
      printer: false,
    });
    return { registry: core.toolRegistry, tools: core.tools.slice() };
  }

  it("restores a removed tool as the same instance", () => {
    const { registry: reg, tools } = registry();
    const original = reg.get(DELETE);
    expect(original).toBeDefined();

    syncTemplateTools(reg, tools, [READ]);
    expect(reg.get(DELETE)).toBeUndefined();

    syncTemplateTools(reg, tools, null);
    expect(reg.get(DELETE)).toBe(original);
  });

  it("leaves an entry another producer owns alone", () => {
    const { registry: reg, tools } = registry();
    reg.remove(DELETE);
    const impostor = fakeTool(
      DELETE,
      "another producer's tool",
    ) as unknown as Tool;
    reg.add(impostor);

    syncTemplateTools(reg, tools, [READ]);
    expect(reg.get(DELETE)).toBe(impostor);

    syncTemplateTools(reg, tools, null);
    expect(reg.get(DELETE)).toBe(impostor);
  });

  it("keeps an exempt name however the selection reads", () => {
    const { registry: reg, tools } = registry();
    expect(
      syncTemplateTools(reg, tools, [], { exemptNames: new Set([DELETE]) }),
    ).toEqual(new Set([DELETE]));
    expect(new Set(reg.list().map((t) => t.name))).toEqual(new Set([DELETE]));
  });

  it("returns what the registry holds", () => {
    const { registry: reg, tools } = registry();
    expect(syncTemplateTools(reg, tools, [READ])).toEqual(new Set([READ]));
    expect(syncTemplateTools(reg, tools, null)).toEqual(
      new Set([READ, DELETE]),
    );
  });
});

describe("parkedBatchToolNames", () => {
  it("parks nothing for an idle agent", () => {
    expect(parkedBatchToolNames(undefined)).toEqual(new Set());
    expect(parkedBatchToolNames({})).toEqual(new Set());
    expect(
      parkedBatchToolNames({ _interruptState: { activated: false } }),
    ).toEqual(new Set());
  });

  it("names every tool in the parked batch", () => {
    const agent = {
      _interruptState: {
        activated: true,
        pendingToolExecution: {
          assistantMessageData: {
            role: "assistant",
            content: [
              { toolUse: { toolUseId: "a", name: DELETE, input: {} } },
              { toolUse: { toolUseId: "b", name: READ, input: {} } },
              { text: "thinking" },
            ],
          },
        },
      },
    };
    expect(parkedBatchToolNames(agent)).toEqual(new Set([DELETE, READ]));
  });

  it("names nothing for a checkpoint carrying no tool batch", () => {
    expect(
      parkedBatchToolNames({ _interruptState: { activated: true } }),
    ).toEqual(new Set());
  });
});
