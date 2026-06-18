/**
 * A2UI subagent tool factory for LangGraph TS agents.
 *
 * Thin adapter over ``@ag-ui/a2ui-toolkit`` — the heavy lifting (op builders,
 * prompt assembly, history walkers, output envelope) lives in the toolkit so
 * each new framework adapter (ADK, Mastra, Strands, …) only owns the
 * framework-specific glue: tool decorator, runtime state access, model
 * binding + invoke.
 *
 * Streaming: the subagent's `render_a2ui` call must STREAM to the AG-UI wire —
 * the a2ui middleware's "building" skeleton and progressive paint key off the
 * inner tool-call's arg deltas, not the final result. A prior assumption that a
 * nested `model.stream()` would auto-surface via the graph's `OnChatModelStream`
 * is FALSE — those deltas do not propagate, so this adapter emits them
 * EXPLICITLY. It mirrors the Strands adapter's per-delta `push(...)`: where
 * Strands re-yields `ToolStreamEvent` payloads that its agent.ts turns into
 * inner TOOL_CALL_START/ARGS/END, this adapter dispatches granular
 * `a2ui_render_{start,args,end}` custom events (via LangGraph's
 * `dispatchCustomEvent`) that `agent.ts`'s OnCustomEvent handler turns into the
 * same inner TOOL_CALL_START/ARGS/END on the wire. That is the channel the
 * adapter ALREADY uses for manually-emitted tool calls — no new transport.
 *
 * Example usage in a chat node:
 *
 *   import { getA2UITools } from "@ag-ui/langgraph";
 *
 *   const a2ui = getA2UITools({ model: new ChatOpenAI({ model: "gpt-4o" }) });
 *
 *   const modelWithTools = chatModel.bindTools(
 *     [...state.tools, a2ui],
 *     { parallel_tool_calls: false },
 *   );
 *
 * Signature note: the factory takes a single `A2UIToolParams` object owned by
 * `@ag-ui/a2ui-toolkit`. Every framework adapter (LG, Strands, ADK, …) shares
 * that exact params shape — only the body below is framework-specific. A new
 * knob added to `A2UIToolParams` reaches this adapter with no signature change.
 */

import { tool, type ToolRuntime } from "@langchain/core/tools";
import { SystemMessage } from "@langchain/core/messages";
import { dispatchCustomEvent } from "@langchain/core/callbacks/dispatch";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  A2UI_OPERATIONS_KEY,
  BASIC_CATALOG_ID,
  GENERATE_A2UI_ARG_DESCRIPTIONS,
  RENDER_A2UI_TOOL_DEF,
  buildA2UIEnvelope,
  prepareA2UIRequest,
  resolveA2UIToolParams,
  wrapErrorEnvelope,
  runA2UIGenerationWithRecovery,
  type A2UIToolParams,
} from "@ag-ui/a2ui-toolkit";

import { CustomEventNames } from "./types";

/** Name of the render tool the A2UI middleware injects (and the subagent binds). */
const RENDER_A2UI_TOOL_NAME = RENDER_A2UI_TOOL_DEF.function.name;

// Per-process fallback-id sequence: providers that never stamp a tool-call id
// must not reuse one id across recovery attempts (two full lifecycles under one
// toolCallId mis-merge in id-keyed consumers).
let a2uiRenderSeq = 0;

/**
 * Loose type for the subagent model.
 *
 * Typed as `any` (rather than `BaseChatModel`) to tolerate `@langchain/core` version
 * skew between this package and the consumer — e.g. `ChatOpenAI` shipping its own
 * peer-pinned core. The factory only needs `bindTools` + `stream`, which is checked
 * at runtime.
 */
export type A2UISubagentModel = any;

// Re-export the toolkit constants/types for callers that previously imported
// them from this package — keeps the public surface stable.
export { A2UI_OPERATIONS_KEY, BASIC_CATALOG_ID };
export type { A2UIToolParams };

/** Tool arguments exposed to the main agent's planner. */
interface GenerateA2UIArgs {
  /**
   * `"create"` to render a new surface, `"update"` to modify a surface
   * previously rendered in this conversation. Defaults to `"create"`.
   */
  intent?: "create" | "update";
  /**
   * Required when `intent="update"`. The surface id of the prior render
   * to modify.
   */
  target_surface_id?: string;
  /** Optional natural-language description of the changes to apply on update. */
  changes?: string;
}

/** One sub-agent render_a2ui streaming step, surfaced on the AG-UI wire. */
export interface A2UIRenderStreamEvent {
  kind: "start" | "args" | "end";
  /** The subagent's tool-call id — fresh per recovery attempt. */
  toolCallId: string;
  /** Tool name (start only). */
  toolCallName?: string;
  /** Raw args-JSON fragment (args only). */
  delta?: string;
}

/**
 * Run the structured-output subagent once: stream the model, push per-event
 * render progress (start / args deltas / end) via `push`, and return the
 * captured `render_a2ui` args — or `null` if the model produced no call.
 *
 * Mirrors the Strands adapter's `invokeRenderSubagent`: `push` is the LangGraph
 * analogue of Strands' per-delta callback. Each streamed chunk's tool-call
 * `args` is the INCREMENTAL JSON fragment, re-emitted as one `"args"` delta; the
 * fragments accumulate (via chunk concat) into the final `render_a2ui` args
 * returned to the recovery loop.
 */
export async function streamRenderSubagent(
  modelWithTool: A2UISubagentModel,
  prompt: string,
  messages: unknown[],
  push: (e: A2UIRenderStreamEvent) => void,
): Promise<Record<string, unknown> | null> {
  let liveCallId: string | null = null;
  let anyArgsStreamed = false;
  let accumulated: any = null;
  // Per-invocation fallback id (mirrors the Strands per-attempt uuid).
  const fallbackCallId = `a2ui-render-${++a2uiRenderSeq}`;

  try {
    const gen = await modelWithTool.stream([
      new SystemMessage(prompt),
      ...(messages as any[]),
    ]);
    for await (const chunk of gen) {
      // Accumulate the streamed AIMessageChunks so the final parsed tool_calls
      // (the captured args) reconstruct even when each frame only carries an
      // incremental arg fragment.
      accumulated = accumulated === null ? chunk : accumulated.concat(chunk);

      const toolCallChunks: Array<{
        name?: string;
        args?: string;
        id?: string;
        index?: number;
      }> = chunk?.tool_call_chunks ?? [];
      for (const tcc of toolCallChunks) {
        // Only the render call drives the synthetic stream; ignore any foreign
        // tool fragments (the subagent is tool_choice-pinned to render_a2ui,
        // but stay defensive).
        if (tcc.name != null && tcc.name !== RENDER_A2UI_TOOL_NAME) continue;
        // `||` (not `??`): an empty-string id must take the fallback — a falsy
        // live id would disable the close/delta guards below.
        let callId: string = tcc.id || liveCallId || fallbackCallId;
        if (liveCallId === fallbackCallId && tcc.id) {
          // Provider delivered the real id only after id-less frames: same
          // logical call — keep the latched fallback id so the synthetic stream
          // stays continuous (no spurious end/start).
          callId = liveCallId;
        }
        if (callId !== liveCallId) {
          // New render call (normally the only one). Close any previous call
          // first so streamed arg deltas never mis-attribute across ids
          // (mirrors the Strands per-call reset).
          if (liveCallId !== null) {
            push({ kind: "end", toolCallId: liveCallId });
          }
          liveCallId = callId;
          push({
            kind: "start",
            toolCallId: callId,
            toolCallName: RENDER_A2UI_TOOL_NAME,
          });
        }
        if (typeof tcc.args === "string" && tcc.args.length > 0) {
          anyArgsStreamed = true;
          push({ kind: "args", toolCallId: callId, delta: tcc.args });
        }
      }
    }
  } catch (err) {
    // The provider stream died mid-call (model 429, network drop, ...): close
    // the live synthetic call before unwinding — an unclosed inner
    // TOOL_CALL_START is a wire-protocol violation, and the next recovery
    // attempt would open a fresh call on top of it.
    if (liveCallId !== null) {
      push({ kind: "end", toolCallId: liveCallId });
      liveCallId = null;
    }
    throw err;
  }

  let captured: Record<string, unknown> | null = null;
  const toolCalls: Array<{ name?: string; args?: Record<string, unknown> }> =
    accumulated?.tool_calls ?? [];
  for (const call of toolCalls) {
    if (call.name == null || call.name === RENDER_A2UI_TOOL_NAME) {
      captured = (call.args ?? {}) as Record<string, unknown>;
      break;
    }
  }

  if (liveCallId !== null) {
    // Some providers deliver parsed tool_calls without streaming arg fragments
    // (no "args" deltas pushed). Emit the captured args as a single delta so the
    // middleware still sees components before the result (no bulk paint).
    if (captured !== null && !anyArgsStreamed) {
      push({ kind: "args", toolCallId: liveCallId, delta: JSON.stringify(captured) });
    }
    push({ kind: "end", toolCallId: liveCallId });
  } else if (captured !== null) {
    // The provider returned the render_a2ui call without emitting ANY
    // tool_call_chunks: synthesize the full triplet so the middleware still
    // sees components before the result (no bulk paint).
    const syntheticId = `a2ui-render-${++a2uiRenderSeq}`;
    push({ kind: "start", toolCallId: syntheticId, toolCallName: RENDER_A2UI_TOOL_NAME });
    push({ kind: "args", toolCallId: syntheticId, delta: JSON.stringify(captured) });
    push({ kind: "end", toolCallId: syntheticId });
  }

  return captured;
}

/**
 * Build a LangGraph tool that delegates A2UI surface generation to a subagent.
 *
 * The returned tool is ready to bind into a chat model alongside any other tools.
 *
 * @param params Shared `A2UIToolParams` (model + behavior knobs). The toolkit
 *   owns the shape and fills defaults via `resolveA2UIToolParams`.
 */
export function getA2UITools<TModel = A2UISubagentModel>(
  params: A2UIToolParams<TModel>,
) {
  // Shared: normalize knobs + fill canonical defaults (toolName, catalogId, …)
  // so this adapter never re-implements default logic. A new params field +
  // its default lives entirely in the toolkit.
  const {
    model,
    guidelines,
    defaultSurfaceId,
    defaultCatalogId,
    toolName,
    toolDescription,
    catalog,
    recovery,
    onA2UIAttempt,
  } = resolveA2UIToolParams(params);
  // Loose-typed locally: the generic TModel only guarantees the shape the
  // toolkit needs; bindTools/stream are checked at runtime (see guard below).
  const chatModel = model as A2UISubagentModel;

  return tool(
    async (
      input: GenerateA2UIArgs,
      runtime: ToolRuntime<Record<string, unknown>, unknown>,
    ): Promise<string> => {
      // Defensive: a custom state schema (or a non-graph invocation) may not
      // preseed `state`/`messages` — mirror the Python adapter's graceful
      // degrade (`state.get("messages", [])`) instead of throwing mid-tool.
      const state = (runtime.state ?? {}) as Record<string, unknown>;
      const allMessages = (state.messages as Array<any>) ?? [];
      // Strip current (unbalanced) tool call from history.
      const messages = allMessages.slice(0, -1);

      // Shared: decide create/update, find prior surface, build the prompt.
      const prep = prepareA2UIRequest({
        intent: input.intent,
        targetSurfaceId: input.target_surface_id,
        changes: input.changes,
        messages,
        state,
        guidelines,
      });
      if (prep.error) return wrapErrorEnvelope(prep.error);

      // Glue: bind the structured-output tool.
      if (!chatModel.bindTools) {
        return wrapErrorEnvelope("Provided model does not support bindTools");
      }
      const modelWithTool = chatModel.bindTools([RENDER_A2UI_TOOL_DEF], {
        tool_choice: { type: "function", function: { name: "render_a2ui" } },
      });

      // The LangGraph analogue of the Strands adapter's `push`: surface each
      // render-stream step as a granular custom event on the run's config so it
      // routes through streamEvents -> OnCustomEvent -> the inner
      // TOOL_CALL_START/ARGS/END the a2ui middleware paints from. `config` is
      // threaded explicitly (mirrors the example nodes' dispatchCustomEvent
      // calls) so the events land on THIS run's stream.
      const config = (runtime as { config?: RunnableConfig }).config;
      const push = (e: A2UIRenderStreamEvent) => {
        const dispatch =
          e.kind === "start"
            ? dispatchCustomEvent(
                CustomEventNames.A2UIRenderStart,
                { id: e.toolCallId, name: e.toolCallName },
                config,
              )
            : e.kind === "args"
              ? dispatchCustomEvent(
                  CustomEventNames.A2UIRenderArgs,
                  { id: e.toolCallId, delta: e.delta },
                  config,
                )
              : dispatchCustomEvent(
                  CustomEventNames.A2UIRenderEnd,
                  { id: e.toolCallId },
                  config,
                );
        // dispatchCustomEvent rejects when there is no parent run id (tool
        // invoked outside a graph run — no streamEvents consumer to paint to).
        // The surface still generates from the captured args; there is simply
        // no live stream to surface the deltas onto, so swallow rather than
        // crashing the generation.
        void dispatch.catch(() => {});
      };

      // Shared: validate→retry loop. On each retry the prompt is re-augmented
      // with the prior attempt's structured errors; only a validated surface is
      // committed (the middleware gate suppresses any unvalidated attempt, so a
      // rejected attempt never paints). Returns a structured hard-failure
      // envelope once the attempt cap is hit.
      const { envelope } = await runA2UIGenerationWithRecovery({
        basePrompt: prep.prompt,
        catalog,
        config: recovery,
        onAttempt: onA2UIAttempt,
        invokeSubagent: (prompt) =>
          streamRenderSubagent(modelWithTool, prompt, messages, push),
        buildEnvelope: (args) =>
          buildA2UIEnvelope({
            args,
            isUpdate: prep.isUpdate,
            targetSurfaceId: input.target_surface_id,
            prior: prep.prior,
            defaultSurfaceId,
            defaultCatalogId,
          }),
      });
      return envelope;
    },
    {
      name: toolName,
      description: toolDescription,
      schema: {
        type: "object",
        properties: {
          intent: {
            type: "string",
            enum: ["create", "update"],
            description: GENERATE_A2UI_ARG_DESCRIPTIONS.intent,
          },
          target_surface_id: {
            type: "string",
            description: GENERATE_A2UI_ARG_DESCRIPTIONS.target_surface_id,
          },
          changes: {
            type: "string",
            description: GENERATE_A2UI_ARG_DESCRIPTIONS.changes,
          },
        },
      } as any,
    },
  );
}
