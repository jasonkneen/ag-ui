/**
 * Tests for the LangGraph A2UI tool's streaming subagent.
 *
 * The parity fix: the inner render_a2ui call must be surfaced as PROGRESSIVE
 * start -> many args deltas -> end, mirroring the Strands adapter — not one
 * final bulk push. `streamRenderSubagent` is the piece that produces those
 * deltas; we drive it with a fake model that streams a fixed render_a2ui call
 * as several AIMessageChunks (one arg fragment each), like a real provider.
 */

import { describe, it, expect } from "vitest";
import { AIMessageChunk } from "@langchain/core/messages";

import {
  streamRenderSubagent,
  type A2UIRenderStreamEvent,
} from "./a2ui-tool";

// A structurally-valid render_a2ui result.
const VALID_ARGS = {
  surfaceId: "s1",
  components: [
    { id: "root", component: "Column", children: ["t"] },
    { id: "t", component: "Text", text: "hi" },
  ],
  data: {},
};

/** Split JSON into `parts` non-empty fragments, the way a provider streams. */
function argChunks(args: unknown, parts = 4): string[] {
  const text = JSON.stringify(args);
  const size = Math.max(1, Math.floor(text.length / parts));
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out.length ? out : [text];
}

/** Fake bound model: streams a fixed render_a2ui call as several chunks. */
function fakeBoundModel(args: unknown, callId = "call-1") {
  return {
    async *stream(_messages: unknown[]) {
      const fragments = argChunks(args);
      for (let i = 0; i < fragments.length; i++) {
        yield new AIMessageChunk({
          content: "",
          tool_call_chunks: [
            {
              // Name + id only on the first fragment, mirroring how providers
              // stamp them once at the start of the call.
              name: i === 0 ? "render_a2ui" : undefined,
              args: fragments[i],
              id: i === 0 ? callId : undefined,
              index: 0,
              type: "tool_call_chunk",
            },
          ],
        });
      }
    },
  };
}

describe("streamRenderSubagent (progressive A2UI paint)", () => {
  it("pushes incremental args deltas, not one bulk paint", async () => {
    const pushed: A2UIRenderStreamEvent[] = [];
    const captured = await streamRenderSubagent(
      fakeBoundModel(VALID_ARGS),
      "PROMPT",
      [],
      (e) => pushed.push(e),
    );

    const kinds = pushed.map((p) => p.kind);
    // Exactly one start, one end, and MULTIPLE args deltas in between — this is
    // the whole point: incremental emission, not one bulk paint.
    expect(kinds[0]).toBe("start");
    expect(kinds[kinds.length - 1]).toBe("end");
    expect(kinds.filter((k) => k === "start")).toHaveLength(1);
    expect(kinds.filter((k) => k === "end")).toHaveLength(1);
    const argsDeltas = pushed.filter((p) => p.kind === "args");
    expect(argsDeltas.length).toBeGreaterThan(1);

    // The start carries the render tool name + a stable id reused by every
    // delta and the end.
    expect(pushed[0].toolCallName).toBe("render_a2ui");
    const callId = pushed[0].toolCallId;
    expect(pushed.every((p) => p.toolCallId === callId)).toBe(true);

    // Concatenating the streamed deltas reconstructs the full render args JSON —
    // the deltas ARE the surface, not a placeholder.
    const joined = argsDeltas.map((p) => p.delta).join("");
    expect(JSON.parse(joined)).toEqual(VALID_ARGS);

    // And the captured args (fed to the recovery loop / envelope) parse back to
    // the same surface.
    expect(captured).toEqual(VALID_ARGS);
  });

  it("returns the captured render args for the envelope", async () => {
    const captured = await streamRenderSubagent(
      fakeBoundModel(VALID_ARGS),
      "PROMPT",
      [],
      () => {},
    );
    expect(captured).toEqual(VALID_ARGS);
  });
});
