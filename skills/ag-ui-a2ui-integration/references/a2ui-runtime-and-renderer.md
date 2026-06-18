# A2UI Runtime and Renderer Wiring

A2UI needs a server-side runtime path that tells the agent how to emit UI and
a client-side renderer path that turns emitted A2UI operations into a visible
surface.

## Runtime Side

Enable A2UI on the runtime. For dynamic schema flows, inject the A2UI tool so
the agent can create and update surfaces.

```ts
import { CopilotRuntime } from "@copilotkit/runtime";

const runtime = new CopilotRuntime({
  agents: {
    default: myAgUiAgent,
  },
  a2ui: { injectA2UITool: true },
});
```

For static UI generation or apps that already expose the necessary A2UI tool
contract, `a2ui: true` or `a2ui: {}` can be enough. Scope to specific agents
when needed:

```ts
const runtime = new CopilotRuntime({
  agents: {
    booking: bookingAgent,
    support: supportAgent,
  },
  a2ui: {
    injectA2UITool: true,
    agents: ["booking"],
  },
});
```

## Client Side

Enable A2UI on the app shell that owns the AG-UI conversation.

```tsx
import { CopilotKitProvider } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";

export function AppShell() {
  return (
    <CopilotKitProvider runtimeUrl="/api/copilotkit" a2ui>
      <App />
    </CopilotKitProvider>
  );
}
```

Pass a theme or catalog through the `a2ui` prop when the app needs custom
rendering:

```tsx
<CopilotKitProvider
  runtimeUrl="/api/copilotkit"
  a2ui={{ catalog: myCatalog, theme: myTheme }}
>
  <App />
</CopilotKitProvider>
```

## Component Catalog Pattern

Use a catalog when the agent needs app-specific components beyond the built-in
A2UI catalog. Define runtime schemas and matching renderers.

```tsx
import {
  createCatalog,
  type CatalogRenderers,
} from "@copilotkit/a2ui-renderer";
import { z } from "zod";

const definitions = {
  ProductCard: {
    description: "A product card with title, price, and availability.",
    props: z.object({
      title: z.string(),
      price: z.number(),
      inStock: z.boolean(),
    }),
  },
};

const renderers = {
  ProductCard: ({ props }) => (
    <article>
      <h3>{props.title}</h3>
      <p>${props.price}</p>
      <button disabled={!props.inStock}>Add to cart</button>
    </article>
  ),
} satisfies CatalogRenderers<typeof definitions>;

export const myCatalog = createCatalog(definitions, renderers, {
  includeBasicCatalog: true,
});
```

Keep the schema value available at runtime. TypeScript types alone disappear
at runtime and cannot teach the agent what it may render.

## Agent Instructions

Tell the agent when to emit A2UI and what interaction should flow back:

```text
When the user asks to compare products, create one A2UI surface named
"product-comparison". Use ProductCard components for each item. When the user
clicks Add to cart, continue the AG-UI run with the selected product id.
```

Prefer concrete UI goals and component names over broad instructions like
"render a nice interface".
