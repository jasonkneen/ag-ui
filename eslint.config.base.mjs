import js from "@eslint/js";
import tseslint from "typescript-eslint";

/**
 * Shared flat config for the TypeScript SDK packages under `sdks/typescript/packages`.
 *
 * Each package re-exports this from its own `eslint.config.mjs`, which makes the
 * `ignores` patterns below resolve relative to that package's directory.
 *
 * Deliberately uses the non-type-checked `recommended` preset: it needs no
 * `project` wiring, so `lint` stays independent of build state and runs in
 * roughly a second per package.
 */
export default tseslint.config(
  {
    ignores: ["dist/**", "coverage/**", "**/generated/**"],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
);
