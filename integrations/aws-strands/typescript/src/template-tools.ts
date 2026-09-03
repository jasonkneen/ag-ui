/**
 * Per-request filtering of the tools the template agent contributed.
 *
 * The adapter builds one Strands `Agent` per thread and keeps it. That instance
 * is load-bearing: it holds the thread's `SessionManager`, its native interrupt
 * checkpoint and its conversation history. Changing which tools a request sees
 * therefore has to be done to the registry the live instance already owns, the
 * way client-declared tools are already synchronised, and never by constructing
 * a replacement.
 *
 * Scope is the template's own tools. Client-declared tools arrive on
 * `RunAgentInput.tools` every request and are synchronised by
 * {@link syncProxyTools}; a caller that wants fewer of those sends fewer.
 * Auto-injected A2UI tools are the adapter's and are refreshed per turn. What
 * no per-request channel reached until now is the set the wrapped template
 * contributed once, at construction.
 */

import type { Tool } from "@strands-agents/sdk";

import type { StrandsToolRegistry } from "./client-proxy-tool";
import { DEFAULT_LOGGER, type Logger } from "./logger";

const LOG_PREFIX = "[@ag-ui/aws-strands]";

/** One entry a `templateToolsProvider` may return: a template tool or its name. */
export type TemplateToolSelectionEntry = Tool | string;

/** Index the template's tools by the name a registry holds them under. */
export function indexTemplateTools(
  templateTools: readonly unknown[],
): Map<string, Tool> {
  const indexed = new Map<string, Tool>();
  for (const tool of templateTools) {
    const name = (tool as { name?: unknown } | null | undefined)?.name;
    if (typeof name === "string" && name.length > 0) {
      indexed.set(name, tool as Tool);
    }
  }
  return indexed;
}

/**
 * Read one provider answer as the template tool names a request may see.
 *
 * Entries are either the template's own tool objects or their names, so a
 * caller can write the filter with whichever it has to hand.
 *
 * `null`/`undefined` means the provider declined to filter this request and
 * every template tool stays available. An empty array is a real answer and
 * means none of them do.
 *
 * A name the template never contributed is dropped with a warning. This hook
 * narrows what the wrapped agent already gave the adapter; it cannot hand the
 * model a capability the template did not carry, so honouring an unknown name
 * is the one thing it must not do.
 */
export function resolveTemplateToolSelection(
  selection: Iterable<TemplateToolSelectionEntry> | null | undefined,
  templateIndex: ReadonlyMap<string, Tool>,
  log: Logger = DEFAULT_LOGGER,
): Set<string> | null {
  if (selection == null) return null;

  const allowed = new Set<string>();
  for (const entry of selection) {
    const name =
      typeof entry === "string"
        ? entry
        : (entry as { name?: unknown } | null | undefined)?.name;
    if (typeof name !== "string" || name.length === 0) {
      log.warn(
        `${LOG_PREFIX} templateToolsProvider returned an entry that names no tool`,
      );
      continue;
    }
    if (!templateIndex.has(name)) {
      log.warn(
        `${LOG_PREFIX} templateToolsProvider named "${name}", which the template ` +
          "agent does not contribute; it stays unavailable. This hook filters " +
          "the template's tools and cannot add one.",
      );
      continue;
    }
    allowed.add(name);
  }
  return allowed;
}

/**
 * Tool names in the batch a live interrupt checkpoint would resume.
 *
 * A parked run resumes into the tool batch it stopped inside: Strands
 * re-dispatches every `toolUse` in the assistant message it checkpointed,
 * answering the ones that already completed from the checkpoint and running the
 * one that is waiting. A tool absent from the registry at that moment turns the
 * human's answer into a "tool not found" the model then re-fires, so nothing in
 * that batch is filtered out while the pause is open.
 *
 * This is the same rule `syncProxyTools` applies to a proxy parked in a
 * frontend-tool interrupt, read off the checkpoint instead of off a
 * frontend-wait index because a template tool can park through the approval
 * hook, through an interrupt of its own, or not at all, and the batch answers
 * all three at once.
 */
export function parkedBatchToolNames(agent: unknown): Set<string> {
  const state = (agent as { _interruptState?: unknown } | null | undefined)
    ?._interruptState as
    | {
        activated?: unknown;
        pendingToolExecution?: { assistantMessageData?: unknown };
      }
    | undefined;
  if (!state || state.activated !== true) return new Set();
  const message = state.pendingToolExecution?.assistantMessageData as
    | { content?: unknown }
    | undefined;
  if (!message || typeof message !== "object") return new Set();
  const content = message.content;
  if (!Array.isArray(content)) return new Set();

  const names = new Set<string>();
  for (const block of content) {
    const toolUse = (block as { toolUse?: unknown } | null | undefined)
      ?.toolUse;
    const name = (toolUse as { name?: unknown } | null | undefined)?.name;
    if (typeof name === "string" && name.length > 0) names.add(name);
  }
  return names;
}

/**
 * Make `toolRegistry` hold exactly the template tools `selection` allows.
 *
 * Only the template's own tools are touched, and only by identity: an entry
 * some other producer owns under a template tool's name is left alone rather
 * than removed, so a client proxy or an auto-injected A2UI tool cannot be
 * dropped by a filter aimed at the template.
 *
 * Removal is not destructive. The template tool objects outlive the registry
 * entry, so a later request that allows a name again restores the same
 * instance, and history stays untouched throughout: a filtered-out tool's
 * earlier calls and results remain in the thread's messages, which is what lets
 * the model read what it already did with a tool it can no longer call.
 *
 * Returns the template tool names the registry holds after the call.
 */
export function syncTemplateTools(
  toolRegistry: StrandsToolRegistry,
  templateTools: readonly unknown[],
  selection: Iterable<TemplateToolSelectionEntry> | null | undefined,
  options: { exemptNames?: ReadonlySet<string>; log?: Logger } = {},
): Set<string> {
  const log = options.log ?? DEFAULT_LOGGER;
  const templateIndex = indexTemplateTools(templateTools);
  const allowed = resolveTemplateToolSelection(selection, templateIndex, log);
  const exempt = options.exemptNames ?? new Set<string>();

  const registered = new Set<string>();
  for (const [name, tool] of templateIndex) {
    const keep = allowed === null || allowed.has(name) || exempt.has(name);
    const existing = toolRegistry.get(name);
    if (keep) {
      if (existing === tool) {
        registered.add(name);
      } else if (existing === undefined) {
        toolRegistry.add(tool);
        registered.add(name);
        log.debug(`${LOG_PREFIX} Restored template tool: ${name}`);
      } else {
        // Something else answers to this name now. Overwriting it would make a
        // filter that allows a tool destroy another producer's.
        log.debug(
          `${LOG_PREFIX} Template tool ${name} is shadowed by another registered ` +
            "tool; leaving it in place",
        );
      }
      continue;
    }
    if (existing === tool) {
      toolRegistry.remove(name);
      log.debug(`${LOG_PREFIX} Filtered out template tool: ${name}`);
    }
  }
  return registered;
}
