/**
 * Server-side entry point for `@ag-ui/aws-strands`.
 *
 * Import from `@ag-ui/aws-strands/server` when you need the Express transport
 * helpers. The main entry point (`@ag-ui/aws-strands`) stays free of Express
 * / cors references so Next.js / Turbopack / Vite bundlers tracing the
 * client-side graph don't pull server-only modules into the browser build.
 */

import {
  addStrandsExpressEndpoint,
  addPing,
  addCapabilities,
} from "./endpoint";
import type { StrandsAgent } from "./agent";
import type { StrandsAguiCapabilitiesOverrides } from "./endpoint";

export {
  addStrandsExpressEndpoint,
  addPing,
  addCapabilities,
  capabilitiesFor,
  DEFAULT_CAPABILITIES,
} from "./endpoint";

export type {
  AddStrandsEndpointOptions,
  StrandsAguiCapabilities,
  StrandsAguiCapabilitiesOverrides,
} from "./endpoint";

export interface CreateStrandsAppOptions {
  /** Path for the agent endpoint. Default `/`. */
  path?: string;
  /** Path for the ping endpoint. Pass `null` or `""` to disable. Default `/ping`. */
  pingPath?: string | null;
  /**
   * Path for the capabilities endpoint. Pass `null` or `""` to disable.
   * Default `/capabilities`.
   */
  capabilitiesPath?: string | null;
  /** Override capabilities advertised at {@link CreateStrandsAppOptions.capabilitiesPath}. */
  capabilities?: StrandsAguiCapabilitiesOverrides;
  /**
   * Override CORS origin. Default `"*"` (wide-open, matches the Python adapter,
   * which configures Starlette `CORSMiddleware` with `allow_origins=["*"]`).
   *
   * Note: with the `cors` package, a literal `"*"` is emitted verbatim as
   * `Access-Control-Allow-Origin: *`, whereas `true` would reflect the request's
   * `Origin` header back per-request, a more permissive posture. Stick to `"*"`
   * to match the Python adapter.
   *
   * An array containing `"*"`, and an empty array, are both normalized to the
   * bare `"*"` so they mean allow-all as they do in Python. Credentials follow
   * from the resolved value: only a concrete allow-list enables them.
   */
  corsOrigin?: string | string[] | boolean;
}

/**
 * Reduce an origin option to the form the `cors` package actually honours.
 *
 * `cors` compares array entries to the request Origin by string equality, so
 * a `"*"` sitting inside an array never matches and an empty array matches
 * nothing either: both deny every origin. Starlette reads both as allow-all,
 * and `create_strands_app` maps a falsy list to `["*"]` besides, so an
 * unnormalized array is a silent cross-SDK divergence. Collapse those cases
 * to the bare string, which `cors` does treat as a wildcard.
 */
function normalizeCorsOrigin(
  origin: string | string[] | boolean,
): string | string[] | boolean {
  if (!Array.isArray(origin)) return origin;
  if (origin.length === 0 || origin.includes("*")) return "*";
  return origin;
}

/**
 * Whether credentialed cross-origin requests may be allowed for `origin`.
 *
 * Only a concrete origin qualifies, whether given as a single string or a
 * list. `Access-Control-Allow-Origin: *`
 * together with `Access-Control-Allow-Credentials: true` is a pairing browsers
 * reject outright, and `origin: true` reflects whatever Origin the caller sent,
 * which would extend credentials to every site. The Python adapter applies the
 * same rule via `allow_credentials=bool(origins) and not is_wildcard`.
 *
 * Call this on the output of {@link normalizeCorsOrigin}, so the wildcard
 * spellings have already collapsed to `"*"`.
 */
function allowsCredentials(origin: string | string[] | boolean): boolean {
  // An empty string disables CORS in the `cors` package, so there is no
  // origin to grant credentials against.
  if (typeof origin === "string") return origin !== "*" && origin !== "";
  // Normalization has already collapsed the wildcard spellings, so a
  // surviving array is always a concrete list. Kept as a guard rather than a
  // bare `true` so calling this on an unnormalized value stays safe.
  if (Array.isArray(origin)) return origin.length > 0 && !origin.includes("*");
  return false;
}

/** Create an Express app with a single Strands agent endpoint and optional ping endpoint. */
export async function createStrandsApp(
  agent: StrandsAgent,
  options: CreateStrandsAppOptions = {},
): Promise<import("express").Express> {
  const {
    path = "/",
    pingPath = "/ping",
    capabilitiesPath = "/capabilities",
    capabilities,
    corsOrigin = "*",
  } = options;

  // Lazy dynamic imports so `express` / `cors` are only required at runtime
  // when `createStrandsApp` is actually called.
  const expressModule = await import("express");
  const corsModule = await import("cors");
  const express = (expressModule.default ??
    expressModule) as typeof import("express");
  const cors = (corsModule.default ?? corsModule) as typeof import("cors");

  const app = express();
  const resolvedCorsOrigin = normalizeCorsOrigin(corsOrigin);
  app.use(
    cors({
      origin: resolvedCorsOrigin,
      credentials: allowsCredentials(resolvedCorsOrigin),
    }),
  );
  app.use(express.json({ limit: "50mb" }));

  addStrandsExpressEndpoint(app, agent, { path });

  if (pingPath) {
    addPing(app, pingPath);
  }

  if (capabilitiesPath) {
    addCapabilities(app, capabilitiesPath, { agent, overrides: capabilities });
  }

  return app;
}
