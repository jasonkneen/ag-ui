# OSS-248 — Re-enable A2UI generation & design guidelines

**Date:** 2026-06-08
**Issue:** [OSS-248](https://linear.app/copilotkit/issue/OSS-248/re-enable-generation-and-design-guidlines)
**Status:** Design approved

## Problem

The legacy `copilotkit.a2ui.a2ui_prompt(component_schema, generation_guidelines, design_guidelines)`
shipped two rich built-in prompt blocks (`DEFAULT_GENERATION_GUIDELINES`,
`DEFAULT_DESIGN_GUIDELINES`) and let hosts override either one. The refactor into the
framework-agnostic `a2ui-toolkit` + per-framework adapters dropped both the defaults and
the override knobs. The current subagent prompt has terse generation rules and **zero design
guidance**, so generated surfaces regressed in visual quality.

## Goals

1. Re-ship the legacy generation + design guideline defaults so subagent output is
   well-designed out of the box.
2. Let hosts override either block, per-field (legacy behavior).
3. Expose the knobs on the A2UI tool factories (`get_a2ui_tools` / `getA2UITools`).
4. Do it in a way that does **not** require editing every framework adapter each time a
   new prompt knob is added (the "100 adapters" problem).

Non-goals: middleware config prop (explicitly out of scope), adapters beyond LangGraph
TS + Python.

## Core design — one shared guidelines bag, owned by the toolkit

Adapters currently re-declare and manually forward every knob. Adding a knob means editing
every adapter signature *and* its pass-through call — O(adapters) edits per knob.

Instead, the **toolkit** owns a single guidelines object. Adapters expose it as **one**
opaque option and forward it verbatim. A future knob is added once, in the toolkit; adapter
code is untouched.

```ts
// toolkit (TS)
export interface A2UIGuidelines {
  generationGuidelines?: string; // override; defaults to DEFAULT_GENERATION_GUIDELINES
  designGuidelines?: string;     // override; defaults to DEFAULT_DESIGN_GUIDELINES
  compositionGuide?: string;     // existing knob, folded in
}
```

```py
# toolkit (Python) — snake_case mirror
class A2UIGuidelines(TypedDict, total=False):
    generation_guidelines: Optional[str]
    design_guidelines: Optional[str]
    composition_guide: Optional[str]
```

### Per-field fallback (matches legacy)

```
resolved_generation = override is None ? DEFAULT_GENERATION_GUIDELINES : override
resolved_design     = override is None ? DEFAULT_DESIGN_GUIDELINES     : override
```

`null`/`None` → built-in default. Empty string `""` → explicit "none" (escape hatch:
host can suppress a block). Only non-empty blocks are appended to the prompt.

### Prompt section order

`generation` → `## Design Guidelines\n{design}` → context (incl. `## Available Components`)
→ `composition` → edit block. Faithful to the legacy `a2ui_prompt` ordering
(generation lead, design header, components).

## Changes by layer

### 1. Toolkit (`a2ui-toolkit`, TS + Python) — the only layer that grows per knob
- Port `DEFAULT_GENERATION_GUIDELINES` + `DEFAULT_DESIGN_GUIDELINES` verbatim from the
  legacy `copilotkit/a2ui.py` as exported module constants.
- Add the `A2UIGuidelines` type.
- `buildSubagentPrompt` / `build_subagent_prompt`: replace the lone `compositionGuide`
  param with `guidelines`; resolve per-field defaults; render in the order above.
- `prepareA2UIRequest` / `prepare_a2ui_request`: replace `compositionGuide` with
  `guidelines`; forward verbatim.

### 2. Adapters (LangGraph TS + Python) — thin, touched once
- `getA2UITools` options / `get_a2ui_tools` kwargs: **remove** `compositionGuide` /
  `composition_guide`; **add** one `guidelines?: A2UIGuidelines` field.
- Forward `guidelines` straight into `prepareA2UIRequest`.

### 3. Middleware — no change (out of scope).

### 4. Example agents + tests (clean-replace fallout)
- `integrations/langgraph/python/examples/agents/a2ui_dynamic_schema/agent.py`
- `integrations/langgraph/typescript/examples/src/agents/a2ui_recovery/agent.ts`
- `integrations/langgraph/typescript/examples/src/agents/a2ui_dynamic_schema/agent.ts`
  → move `compositionGuide: X` to `guidelines: { compositionGuide: X }`.
- Update toolkit tests (`toolkit.test.ts`, `test_toolkit.py`) for the new signature.

## Behavior change

Built-in defaults apply automatically, so existing callers that pass nothing now get rich
design guidance injected into the subagent prompt. This is the intended re-enable. The
middleware's `RENDER_A2UI_TOOL_GUIDELINES` (direct-tool path) is orthogonal and untouched.

## Testing

- **Toolkit (TS + Py):** `build_subagent_prompt` — defaults applied when absent; per-field
  override respected; `""` suppresses a block; section ordering; existing null-value guard
  preserved.
- **Adapters:** `guidelines` forwarded into the subagent prompt; clean-replaced
  `compositionGuide` path still reaches the prompt via `guidelines.compositionGuide`.
