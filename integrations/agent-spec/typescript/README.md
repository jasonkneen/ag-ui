# @ag-ui/agent-spec

Minimal Agent Spec client wrapper for the AG‑UI protocol over HTTP.

This package extends `HttpAgent` and automatically enables A2UI rendering for
the `a2ui_chat` route via `@ag-ui/a2ui-middleware`.

## Installation

### Package consumers

```bash
pnpm add @ag-ui/agent-spec
# or
npm install @ag-ui/agent-spec
# or
yarn add @ag-ui/agent-spec
```

## Working in this monorepo

### Install and build

From the repo root:

```bash
pnpm install
pnpm build --projects=demo-viewer
```

> Note: this repo uses Nx. Avoid `pnpm build --filter=...` because `pnpm build` runs Nx under the hood and `--filter`
> will be forwarded to underlying build tools (e.g., `tsdown`), which can fail with “No valid configuration found”.

### Start the Agent Spec example server (Python)

From the repo root:

```bash
cd integrations/agent-spec/python/examples
uv sync --extra langgraph --extra wayflow
uv run dev
```

This starts a FastAPI server on `http://localhost:9003` by default (configurable via `PORT` in a local `.env`).

### Run Dojo

From the repo root:

```bash
pnpm nx run demo-viewer:dev
```

If your Agent Spec server is not running on `http://localhost:9003`, set `AGENT_SPEC_URL`:

```bash
AGENT_SPEC_URL=http://localhost:9003 pnpm nx run demo-viewer:dev
```

Then open `http://localhost:3000` and select:
- Integration: **Open Agent Spec (LangGraph)** or **Open Agent Spec (Wayflow)**
- Feature: **a2ui_chat**

#### Notes for testing A2UI locally

Some upstream A2UI testing flows require forcing all `@ag-ui/client` imports (including those inside CopilotKit’s
pre-bundled code) to resolve to the local workspace version via a webpack alias in `apps/dojo/next.config.ts`.

## Notes

- Agent Spec backends typically define tools (including `send_a2ui_json_to_client`) in the Agent Spec configuration.
- `injectA2UITool` is enabled client-side for `a2ui_chat` to match upstream A2UI patterns, but the Agent Spec backend
  does not currently consume `RunAgentInput.tools`.
