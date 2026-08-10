import { TokenUsage } from "./events";

/**
 * Accept a value only if it is a real, finite number.
 *
 * Shared by both vendor mappers so they guard identically. Providers do hand
 * over strings, `null`s and `NaN`s in their usage metadata, and a bad value must
 * not reach the wire: consumers validate every incoming event and throw on
 * failure, so one malformed count would fail an otherwise-successful run at its
 * final event — costing the user the answer, not just the token count.
 */
const num = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;

const COUNT_KEYS = [
  "inputTokens",
  "outputTokens",
  "totalTokens",
  "reasoningTokens",
  "cachedInputTokens",
] as const;

/**
 * Build a {@link TokenUsage} from already-guarded counts, or `undefined` when no
 * count survived. Returning `undefined` rather than a labels-only entry keeps
 * "the provider reported no usage" distinct from "the provider reported usage",
 * so callers omit the field instead of emitting an entry that claims nothing.
 */
function buildEntry(
  counts: Partial<Record<(typeof COUNT_KEYS)[number], number | undefined>>,
  { provider, model }: { provider?: string; model?: string },
): TokenUsage | undefined {
  if (!COUNT_KEYS.some((key) => counts[key] !== undefined)) return undefined;

  const entry: TokenUsage = {};
  if (provider != null) entry.provider = provider;
  if (model != null) entry.model = model;
  for (const key of COUNT_KEYS) {
    const value = counts[key];
    if (value !== undefined) entry[key] = value;
  }
  return entry;
}

/**
 * Map a LangChain-family `usage_metadata` object into an AG-UI {@link TokenUsage}.
 *
 * LangChain and LangGraph both attach usage as `{ input_tokens, output_tokens,
 * total_tokens, input_token_details: { cache_read }, output_token_details:
 * { reasoning } }`. This maps only those numeric counts plus optional
 * provider/model labels — never prompt/completion content. Returns `undefined`
 * when no usable count is present, so callers can omit usage rather than
 * report zeros.
 */
export function tokenUsageFromLangChainMetadata(
  usageMetadata: any,
  { provider, model }: { provider?: string; model?: string },
): TokenUsage | undefined {
  if (!usageMetadata) return undefined;

  const inputDetails = usageMetadata.input_token_details ?? {};
  const outputDetails = usageMetadata.output_token_details ?? {};

  return buildEntry(
    {
      inputTokens: num(usageMetadata.input_tokens),
      outputTokens: num(usageMetadata.output_tokens),
      totalTokens: num(usageMetadata.total_tokens),
      reasoningTokens: num(outputDetails.reasoning),
      cachedInputTokens: num(inputDetails.cache_read),
    },
    { provider, model },
  );
}

/**
 * Map an AI-SDK (v5) `LanguageModelUsage` object into an AG-UI {@link TokenUsage}.
 *
 * AI-SDK's keys already match: `inputTokens`, `outputTokens`, `totalTokens`,
 * `reasoningTokens`, `cachedInputTokens`. AI-SDK reports `NaN`/`undefined` for
 * counts a provider didn't return, so only finite numbers are copied. Returns
 * `undefined` when no finite count is present (so callers omit empty usage).
 */
export function tokenUsageFromAiSdkUsage(
  usage: any,
  { provider, model }: { provider?: string; model?: string },
): TokenUsage | undefined {
  if (!usage) return undefined;

  return buildEntry(
    {
      inputTokens: num(usage.inputTokens),
      outputTokens: num(usage.outputTokens),
      totalTokens: num(usage.totalTokens),
      reasoningTokens: num(usage.reasoningTokens),
      cachedInputTokens: num(usage.cachedInputTokens),
    },
    { provider, model },
  );
}

/**
 * Sum per-call {@link TokenUsage} entries into one entry per `(provider, model)`
 * pair. Order follows first appearance. A count field stays `undefined` when no
 * member of the group reported it, so "not reported" stays distinct from zero.
 *
 * Protocol-agnostic: works on any producer's `TokenUsage[]`, so integrations
 * share it rather than reimplementing aggregation.
 */
export function aggregateTokenUsage(entries: TokenUsage[]): TokenUsage[] {
  const grouped = new Map<string, TokenUsage>();

  for (const entry of entries) {
    const key = `${entry.provider ?? ""} ${entry.model ?? ""}`;
    let target = grouped.get(key);
    if (!target) {
      target = { provider: entry.provider, model: entry.model };
      grouped.set(key, target);
    }
    for (const field of COUNT_KEYS) {
      const value = entry[field];
      if (value == null) continue;
      target[field] = (target[field] ?? 0) + value;
    }
  }

  return [...grouped.values()];
}
