# @ag-ui/mcp-apps-middleware

MCP Apps middleware for AG-UI that enables UI-enabled tools from MCP (Model Context Protocol) servers.

## Installation

```bash
npm install @ag-ui/mcp-apps-middleware
# or
pnpm add @ag-ui/mcp-apps-middleware
```

## Usage

```typescript
import { MCPAppsMiddleware } from "@ag-ui/mcp-apps-middleware";

const agent = new YourAgent().use(
  new MCPAppsMiddleware({
    mcpServers: [
      {
        type: "http",
        url: "http://localhost:3001/mcp",
        serverId: "weather-server",
      },
    ],
  }),
);
```

## Features

- Discovers UI-enabled tools from MCP servers
- Injects tools into the agent's tool list
- Executes tool calls and emits activity snapshots with resource URIs
- Supports proxied MCP requests for frontend resource fetching

## Configuration

```typescript
interface MCPAppsMiddlewareConfig {
  mcpServers?: MCPClientConfig[];
}

type MCPClientConfig =
  | { type: "http"; url: string; serverId?: string }
  | {
      type: "sse";
      url: string;
      headers?: Record<string, string>;
      serverId?: string;
    };
```

### Server ID

The optional `serverId` field provides a stable identifier for the server. This is useful when:

- Server URLs may change (e.g., different environments)
- You want human-readable server identification
- Frontend code needs to reference servers by name

If `serverId` is not provided, the server is identified by an MD5 hash of its configuration.

## Activity Snapshot

The middleware emits activity snapshots with the following structure:

```typescript
{
  type: "ACTIVITY_SNAPSHOT",
  activityType: "mcp-apps",
  content: {
    result: MCPToolCallResult,     // Result from the tool execution
    resourceUri: string,           // URI of the UI resource to fetch
    serverHash: string,            // MD5 hash of server config
    serverId?: string,           // Server ID (if configured)
    toolInput: Record<string, unknown>  // Arguments passed to the tool
  },
  replace: true
}
```

The frontend should fetch the resource content via proxied MCP request using `resourceUri` and either `serverHash` or `serverId`.

## Proxied MCP Requests

The middleware supports proxied MCP requests from the frontend. Pass a `ProxiedMCPRequest` in `forwardedProps.__proxiedMCPRequest`:

```typescript
interface ProxiedMCPRequest {
  serverHash: string; // MD5 hash of server config
  serverId?: string; // Optional server ID for lookup
  method: string; // MCP method (e.g., "resources/read", "tools/call")
  params?: Record<string, unknown>;
}
```

Server lookup prefers `serverId` if provided, falling back to `serverHash`.

## Exported Utilities

```typescript
import {
  MCPAppsActivityType, // "mcp-apps" constant
  getServerHash, // Generate server hash from config
} from "@ag-ui/mcp-apps-middleware";
```

## License

MIT

## Proxy connections

The middleware accepts only `tools/call`, `resources/read`, `notifications/message`, and `ping` from an iframe proxy request.
It rejects other methods before it connects to the MCP server.
HTTP discovery, tool calls, and proxy requests delete their MCP sessions before closing the client.
If a server rejects session deletion or does not respond within three seconds, the client still closes and preserves the original operation result.

Server hashes exclude headers so browser-visible references do not contain a checksum of credentials.
This changes hashes for servers configured with headers. Recreate activity messages after upgrading;
use stable `serverId` values for references that must survive configuration changes.
If multiple configurations share a transport type and URL, each must have a distinct `serverId`.
Hash-only requests for that endpoint are rejected because they cannot identify the intended credential scope.
