---
name: ag-ui-a2ui-integration
description: "Use when adding A2UI rendering to any AG-UI-supported framework or custom AG-UI application, scaffolding an AG-UI app that should render A2UI, adapting an AG-UI integration to emit A2UI surfaces, or wiring CopilotKit's A2UI runtime and renderer around an AG-UI agent."
version: 1.0.0
---

# AG-UI + A2UI Integration Skill

## Overview

Use this skill to add A2UI rendering to an AG-UI application. Treat AG-UI as
the transport and agent integration layer, and A2UI as the UI payload format
that the agent emits and the client renders.

This is a developer-facing skill artifact. It is meant to be loaded by coding
agents and used against a real app or repo, not published as a docs page.

## When to Use

- Adding A2UI rendering to an existing AG-UI app.
- Creating an AG-UI quickstart that should display A2UI surfaces.
- Connecting any AG-UI-supported framework or custom AG-UI agent to an
  A2UI-capable frontend.
- Adding or extending an A2UI component catalog.
- Debugging why an A2UI surface does not render or why a user action does not
  flow back to the agent.

## When NOT to Use

- For AG-UI protocol event semantics only, use the AG-UI protocol skill or
  protocol docs.
- For A2UI renderer internals outside an AG-UI app, use the A2UI renderer
  docs or renderer-specific skills.
- For generic CopilotKit frontend work without A2UI, use CopilotKit-specific
  setup and React skills.

## Workflow

1. Inspect the app shape: framework, package manager, AG-UI endpoint, frontend
   shell, and any existing A2UI renderer or catalog.
2. If starting from a blank app, inspect the current AG-UI CLI and integration
   docs. Use a CLI flag when the target framework is scaffoldable; otherwise
   start from the framework's AG-UI package or example.
3. Select the framework adapter from
   `references/framework-adapters.md`, or use that reference to find the
   closest AG-UI integration pattern. Preserve the app's existing agent
   architecture.
4. Wire the A2UI runtime and renderer using
   `references/a2ui-runtime-and-renderer.md`.
5. Add or extend the component catalog only when the built-in A2UI catalog is
   not enough for the requested UI.
6. Verify the streaming path with `references/verification.md`: agent stream,
   rendered A2UI surface, and a user interaction flowing back through AG-UI.

## AG-UI Framework Support

This skill is not limited to the framework examples below. For any target
framework, first check the AG-UI repository's `integrations/` directory, the
AG-UI docs, and the current CLI source. If AG-UI supports the framework, use
that integration's documented package, endpoint helper, or scaffold path. If
there is no framework adapter, implement the custom AG-UI agent path and keep
the A2UI runtime/client wiring the same.

## Common AG-UI CLI Flags

Use the CLI flags that exist in `sdks/typescript/packages/cli/src/index.ts`.
The table is a quick reference for known scaffold paths, not the full AG-UI
support matrix. Do not invent flags.

| Framework            | CLI flag         |
| -------------------- | ---------------- |
| ADK                  | `--adk`          |
| LangGraph Python     | `--langgraph-py` |
| LangGraph JavaScript | `--langgraph-js` |
| CrewAI Flows         | `--crewai-flows` |
| Mastra               | `--mastra`       |
| Pydantic AI          | `--pydantic-ai`  |
| LlamaIndex           | `--llamaindex`   |
| Agno                 | `--agno`         |
| AG2                  | `--ag2`          |

Strands has AG-UI integration packages and examples, but no Strands CLI flag
is present in the current AG-UI CLI source. Use the Strands integration docs
instead of guessing a scaffold command.

## Key Rules

- Keep the integration AG-UI-first for every supported framework. CopilotKit is
  the common runtime and renderer path for A2UI in web apps, but do not
  describe AG-UI as merely a CopilotKit implementation detail.
- Enable A2UI on both sides: runtime support on the server and renderer
  support on the client.
- If dynamic component schemas are needed, inject the A2UI tool on the runtime
  and pass a real runtime schema value. Type-only definitions are not enough.
- Emit `createSurface` once per `surfaceId`; use update operations for later
  changes.
- Preserve AG-UI run boundaries and error events. Do not swallow server or
  stream errors.
- Verify with a real browser or client run when possible. A static typecheck is
  not enough for streaming UI work.

## References

- `references/framework-adapters.md` - framework-specific AG-UI adapter
  patterns.
- `references/a2ui-runtime-and-renderer.md` - server/client A2UI wiring and
  catalog patterns.
- `references/verification.md` - checks to confirm the integration works.
- `sources.md` - source files and docs used by this skill.
