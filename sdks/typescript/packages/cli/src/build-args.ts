/**
 * Version spec for the underlying CopilotKit CLI. Pinned to the v3 line (not
 * `@latest`) so a future major with different arguments cannot silently break
 * `create-ag-ui-app`; `@3` still picks up bug-fix releases within v3.
 */
export const COPILOTKIT_CLI_SPEC = "copilotkit@3";

/** Framework flags as parsed by commander, in selection-priority order. */
export interface FrameworkOptions {
  langgraphPy?: boolean;
  langgraphJs?: boolean;
  crewaiFlows?: boolean;
  mastra?: boolean;
  ag2?: boolean;
  llamaindex?: boolean;
  agno?: boolean;
  pydanticAi?: boolean;
  adk?: boolean;
}

/**
 * Builds the argv passed to `npx` to invoke the CopilotKit CLI's `create`
 * command. Pure and exported so the mapping (and the version pin) is unit
 * tested without spawning a process.
 *
 * @param options - Parsed commander framework flags.
 * @param projectName - Validated project name.
 * @returns The argv array for `spawn("npx", ...)`.
 */
export function buildCopilotKitCreateArgs(
  options: FrameworkOptions,
  projectName: string,
): string[] {
  const frameworkArgs: string[] = [];

  if (options.langgraphPy) {
    frameworkArgs.push("-f", "langgraph-py");
  } else if (options.langgraphJs) {
    frameworkArgs.push("-f", "langgraph-js");
  } else if (options.crewaiFlows) {
    frameworkArgs.push("-f", "flows");
  } else if (options.mastra) {
    frameworkArgs.push("-f", "mastra");
  } else if (options.ag2) {
    frameworkArgs.push("-f", "ag2");
  } else if (options.llamaindex) {
    frameworkArgs.push("-f", "llamaindex");
  } else if (options.agno) {
    frameworkArgs.push("-f", "agno");
  } else if (options.pydanticAi) {
    frameworkArgs.push("-f", "pydantic-ai");
  } else if (options.adk) {
    frameworkArgs.push("-f", "adk");
  }

  return [COPILOTKIT_CLI_SPEC, "create", "--no-banner", "-n", projectName, ...frameworkArgs];
}
