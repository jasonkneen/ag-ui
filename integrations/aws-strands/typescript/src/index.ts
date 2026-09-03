/** AWS Strands integration for AG-UI. */

export {
  StrandsAgent,
  INTERRUPT_CANCELLED,
  buildSnapshotMessages,
  buildStrandsSeed,
  convertMessagesForStrandsSeed,
} from "./agent";
export type { StrandsAgentOptions } from "./agent";

export {
  createProxyTool,
  syncProxyTools,
  isProxyTool,
} from "./client-proxy-tool";
export type { StrandsToolRegistry } from "./client-proxy-tool";

export { syncTemplateTools, parkedBatchToolNames } from "./template-tools";
export type { TemplateToolSelectionEntry } from "./template-tools";

export { CITATIONS_METADATA_KEY } from "./citations";
export type { AguiCitation, AguiCitationLocation } from "./citations";

export { convertAguiContentToStrands, flattenContentToText } from "./utils";

export {
  getA2UITools,
  planA2UIInjection,
  isAutoInjectedA2UITool,
  A2UI_STREAM_KEY,
} from "./a2ui-tool";
export type {
  A2UIToolParams,
  A2UIAttemptRecord,
  A2UIToolGlue,
  A2UIInjectConfig,
  A2UIInjectionPlan,
  A2UIRenderStreamEvent,
  PlanA2UIInjectionInput,
} from "./a2ui-tool";

// Server-side Express transport helpers (`createStrandsApp`,
// `addStrandsExpressEndpoint`, `addPing`, `addCapabilities`,
// `capabilitiesFor`, `DEFAULT_CAPABILITIES`, and associated types) live at
// `@ag-ui/aws-strands/server`. Keeping them off the main entry lets
// client-side bundlers (Next.js, Vite, etc.) trace this package without
// pulling Express / cors into the browser graph.

export type { Logger } from "./logger";

export { buildContextExtras } from "./config";
export type {
  StrandsAgentConfig,
  ToolBehavior,
  ToolCallContext,
  ToolCallContextExtras,
  ToolResultContext,
  ToolStreamEventContext,
  ToolStreamEventHandler,
  PredictStateMapping,
  SessionManagerProvider,
  TemplateToolsProvider,
  ThreadAgentConfigProvider,
  StateContextBuilder,
  StateFromArgs,
  StateFromResult,
  CustomResultHandler,
  ArgsStreamer,
  MaybePromise,
  StatePayload,
} from "./config";

// Thin HttpAgent subclass for AG-UI clients pointing at a Strands endpoint.
import { HttpAgent } from "@ag-ui/client";
export class AWSStrandsAgent extends HttpAgent {}
