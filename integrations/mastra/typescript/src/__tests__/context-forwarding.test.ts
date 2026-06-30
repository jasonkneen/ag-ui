import { describe, it, expect } from "vitest";
import type { Context } from "@ag-ui/client";
import { MastraAgent } from "../mastra";
import {
  FakeLocalAgent,
  FakeRemoteAgent,
  makeInput,
  collectEvents,
} from "./helpers";

// ---------------------------------------------------------------------------
// OSS-392: input.context must be forwarded onto Mastra's RequestContext under
// the "ag-ui" key, reachable by tools via `execute(input, { requestContext })`,
// on the initial run AND after resume on BOTH the local and remote paths.
//
// These tests read the context BACK through the exact RequestContext.get()
// channel a Mastra tool uses — not merely asserting the option is present — so
// they prove reachability, not plumbing. (A real-LLM run with a tool that calls
// requestContext.get("ag-ui") covers the live Mastra runtime end-to-end.)
// ---------------------------------------------------------------------------

const CONTEXT_A: Context[] = [{ description: "tier", value: "premium" }];
const CONTEXT_B: Context[] = [{ description: "tier", value: "enterprise" }];

/** Reads the forwarded context back the same way a tool's execute would. */
function readForwardedContext(opts: any): Context[] | undefined {
  const reqCtx = opts?.requestContext;
  if (!reqCtx || typeof reqCtx.get !== "function") return undefined;
  return reqCtx.get("ag-ui")?.context;
}

function makeLocal(opts: { streamChunks?: any[]; resumeChunks?: any[] } = {}) {
  const fake = new FakeLocalAgent(opts);
  const agent = new MastraAgent({
    agentId: "test-agent",
    agent: fake as any,
    resourceId: "resource-1",
  });
  return { agent, fake };
}

function makeRemote(opts: { streamChunks?: any[]; resumeChunks?: any[] } = {}) {
  const fake = new FakeRemoteAgent(opts);
  const agent = new MastraAgent({
    agentId: "test-agent",
    agent: fake as any,
    resourceId: "resource-1",
  });
  return { agent, fake };
}

/** Legacy resume command (forwardedProps.command), as CopilotKit < 1.61.2 sends. */
function legacyResumeInput(context: Context[]) {
  return makeInput({
    context,
    forwardedProps: {
      command: {
        resume: { approved: true },
        interruptEvent: JSON.stringify({
          type: "mastra_suspend",
          toolCallId: "tc-1",
          runId: "mastra-run-xyz",
        }),
      },
    },
  });
}

/** Standard resume channel (RunAgentInput.resume), as CopilotKit >= 1.61.2 sends. */
function standardResumeInput(context: Context[]) {
  return makeInput({
    context,
    resume: [
      {
        interruptId: "mastra-run-xyz::tc-1",
        status: "resolved",
        payload: { approved: true },
      },
    ],
  } as any);
}

describe("context forwarding (OSS-392): initial run", () => {
  it("forwards input.context onto requestContext for a local agent", async () => {
    const { agent, fake } = makeLocal({
      streamChunks: [{ type: "text-delta", payload: { text: "hi" } }],
    });

    await collectEvents(agent, makeInput({ context: CONTEXT_A }));

    expect(readForwardedContext(fake.lastStreamOpts)).toEqual(CONTEXT_A);
  });

  it("forwards input.context onto requestContext for a remote agent", async () => {
    const { agent, fake } = makeRemote({
      streamChunks: [{ type: "text-delta", payload: { text: "hi" } }],
    });

    await collectEvents(agent, makeInput({ context: CONTEXT_A }));

    expect(readForwardedContext(fake.lastStreamOpts)).toEqual(CONTEXT_A);
  });

  it("forwards an empty context as []", async () => {
    const { agent, fake } = makeLocal({
      streamChunks: [{ type: "text-delta", payload: { text: "hi" } }],
    });

    await collectEvents(agent, makeInput({ context: [] }));

    expect(readForwardedContext(fake.lastStreamOpts)).toEqual([]);
  });
});

describe("context forwarding (OSS-392): resume re-sets context", () => {
  it("forwards the resume request's context on a local legacy-channel resume", async () => {
    const { agent, fake } = makeLocal({
      resumeChunks: [{ type: "text-delta", payload: { text: "approved" } }],
    });

    await collectEvents(agent, legacyResumeInput(CONTEXT_B));

    expect(readForwardedContext(fake.lastResumeOpts)).toEqual(CONTEXT_B);
  });

  it("forwards the resume request's context on a local standard-channel resume", async () => {
    const { agent, fake } = makeLocal({
      resumeChunks: [{ type: "text-delta", payload: { text: "approved" } }],
    });

    await collectEvents(agent, standardResumeInput(CONTEXT_B));

    expect(readForwardedContext(fake.lastResumeOpts)).toEqual(CONTEXT_B);
  });

  it("forwards the resume request's context on a remote resume", async () => {
    const { agent, fake } = makeRemote({
      resumeChunks: [{ type: "text-delta", payload: { text: "approved" } }],
    });

    await collectEvents(agent, legacyResumeInput(CONTEXT_B));

    expect(fake.resumeCalls).toHaveLength(1);
    expect(readForwardedContext(fake.resumeCalls[0].opts)).toEqual(CONTEXT_B);
  });
});

describe("context forwarding (OSS-392): resume does not reuse a stale context", () => {
  // The bug this guards: the resume path reused the constructor/initial-run
  // requestContext verbatim and never re-set input.context, so a resume request
  // carrying a NEW context silently saw the prior turn's value. We reuse a
  // single agent instance (shared this.requestContext) across an initial run
  // then a resume, and assert the resume sees the fresh context, not the stale.

  it("local resume overwrites the prior run's context (does not drop the new one)", async () => {
    const fake = new FakeLocalAgent({
      streamChunks: [{ type: "text-delta", payload: { text: "first" } }],
      resumeChunks: [{ type: "text-delta", payload: { text: "resumed" } }],
    });
    const agent = new MastraAgent({
      agentId: "test-agent",
      agent: fake as any,
      resourceId: "resource-1",
    });

    // Initial run sets context A on the shared requestContext.
    await collectEvents(agent, makeInput({ context: CONTEXT_A }));
    expect(readForwardedContext(fake.lastStreamOpts)).toEqual(CONTEXT_A);

    // Resume carries a DIFFERENT context B — it must be re-set, not dropped.
    await collectEvents(agent, legacyResumeInput(CONTEXT_B));
    expect(readForwardedContext(fake.lastResumeOpts)).toEqual(CONTEXT_B);
  });

  it("remote resume overwrites the prior run's context (does not drop the new one)", async () => {
    const fake = new FakeRemoteAgent({
      streamChunks: [{ type: "text-delta", payload: { text: "first" } }],
      resumeChunks: [{ type: "text-delta", payload: { text: "resumed" } }],
    });
    const agent = new MastraAgent({
      agentId: "test-agent",
      agent: fake as any,
      resourceId: "resource-1",
    });

    await collectEvents(agent, makeInput({ context: CONTEXT_A }));
    expect(readForwardedContext(fake.lastStreamOpts)).toEqual(CONTEXT_A);

    await collectEvents(agent, legacyResumeInput(CONTEXT_B));
    expect(fake.resumeCalls).toHaveLength(1);
    expect(readForwardedContext(fake.resumeCalls[0].opts)).toEqual(CONTEXT_B);
  });
});
