import { HttpAgent } from "@ag-ui/client";
import { A2UIMiddleware } from "@ag-ui/a2ui-middleware";

/**
 * Minimal Agent Spec client that speaks the AG-UI protocol over HTTP.
 *
 * Automatically enables A2UI rendering middleware for the `a2ui_chat` endpoint.
 */
export class AgentSpecAgent extends HttpAgent {
  constructor(config: ConstructorParameters<typeof HttpAgent>[0]) {
    super(config);

    const rawUrl = config.url ?? "";
    let pathToCheck = rawUrl;
    try {
      pathToCheck = new URL(rawUrl).pathname;
    } catch {
      // rawUrl might be relative; fall back to string checks
    }

    const trimmed = pathToCheck.replace(/\/+$/, "");
    if (trimmed.endsWith("a2ui_chat")) {
      // Agent Spec backends typically define the tool in the Agent Spec config, but we
      // enable injection to match upstream A2UI patterns and to be future-proof.
      this.use(new A2UIMiddleware({ injectA2UITool: true }));
    }
  }
}

