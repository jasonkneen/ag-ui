# Verification

Use this checklist before calling an AG-UI + A2UI integration complete.

## Static Checks

- Install dependencies with the app's existing package manager.
- Run the app's typecheck, lint, and unit tests when available.
- Run the AG-UI package or integration tests touched by the change.
- Confirm no invented package names, CLI flags, or import paths were added.

## Runtime Checks

- Start the AG-UI backend or runtime route.
- Start the frontend app.
- Trigger a user prompt that should produce A2UI.
- Confirm the stream begins with a valid AG-UI run and ends with
  `RUN_FINISHED` or `RUN_ERROR`.
- Confirm an A2UI surface renders, not just a text explanation.
- Confirm a user interaction in the rendered surface flows back to the agent.
- Check the browser console and backend logs for schema, hydration, stream, or
  action bridge errors.

## Common Failure Modes

| Symptom                        | Likely cause                                                 | Fix                                                                        |
| ------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------------------------- |
| No A2UI surface appears        | A2UI is enabled only on the client or only on the runtime    | Enable both sides                                                          |
| Agent describes UI in prose    | Agent lacks A2UI tool/schema instructions                    | Inject the A2UI tool and add concrete agent instructions                   |
| Custom component never renders | Catalog is missing or schema key does not match renderer key | Register the catalog and align definition and renderer names               |
| Action clicks do nothing       | The action bridge is not connected to AG-UI run input        | Verify the client action handler continues the run with the action payload |
| Surface flickers or duplicates | Agent emits `createSurface` repeatedly for the same surface  | Emit `createSurface` once and update afterward                             |

A runtime smoke test should show the agent stream in logs or devtools, a
rendered A2UI surface in the page, and one user interaction returning through
AG-UI.
