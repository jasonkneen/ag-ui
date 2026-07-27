import type { BetaManagedAgentsCustomToolParams } from "@anthropic-ai/sdk/resources/beta/agents/agents";
import type { Tool } from "@ag-ui/client";
import { TOOL_DESCRIPTION_MAX_LENGTH, TOOL_NAME_MAX_LENGTH } from "./constants";

export type CustomToolParams = BetaManagedAgentsCustomToolParams;

const TOOL_NAME_PATTERN = new RegExp(`^[A-Za-z0-9_-]{1,${TOOL_NAME_MAX_LENGTH}}$`);

const canonicalize = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, canonicalize(item)]),
  );
};

/** Managed Agents tool names allow only [A-Za-z0-9_-], up to 128 chars. */
export const normalizeToolName = (name: string): string => {
  if (TOOL_NAME_PATTERN.test(name)) return name;
  return name.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, TOOL_NAME_MAX_LENGTH) || "tool";
};

const toInputSchema = (parameters: unknown): CustomToolParams["input_schema"] => {
  if (parameters && typeof parameters === "object") {
    const p = parameters as { properties?: Record<string, unknown>; required?: string[] };
    return { type: "object", properties: p.properties ?? {}, required: p.required ?? [] };
  }
  return { type: "object", properties: {} };
};

/** An AG-UI (frontend) or backend tool definition → managed-agent custom tool. */
export const customToolFrom = (tool: Pick<Tool, "name" | "description" | "parameters">): CustomToolParams => ({
  type: "custom",
  name: normalizeToolName(tool.name),
  description: (tool.description || `Tool ${tool.name}`).slice(0, TOOL_DESCRIPTION_MAX_LENGTH),
  input_schema: toInputSchema(tool.parameters),
});

/** Stable representation used to detect any change to custom tool definitions. */
export const customToolsFingerprint = (tools: CustomToolParams[]): string => JSON.stringify(canonicalize(tools)) ?? "[]";
