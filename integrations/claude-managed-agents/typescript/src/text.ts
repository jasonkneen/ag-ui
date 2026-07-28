/** Helpers for turning Managed Agents content blocks into display text. */

import { SEARCH_RESULT_PREVIEW_CHARS } from "./constants";

/** Any content block; only `type` is required to route it. */
type ContentBlock = { type: string } & Record<string, any>;

/** U+FFFD, substituted for any entity that does not denote a usable character. */
const REPLACEMENT_CHARACTER = "�";

/**
 * The character an entity's code point denotes, or U+FFFD.
 *
 * Surrogate code points (U+D800–U+DFFF) are rejected as well as out-of-range
 * ones: alone they make the string ill-formed UTF-16, which cannot be encoded
 * as UTF-8 and turns into U+FFFD (or an encoder error) somewhere downstream.
 * Substituting here keeps every port's output well-formed and identical.
 */
const codePointOf = (n: number): string =>
  Number.isInteger(n) && n >= 0 && n <= 0x10ffff && !(n >= 0xd800 && n <= 0xdfff)
    ? String.fromCodePoint(n)
    : REPLACEMENT_CHARACTER;

const decodeEntities = (s: string): string =>
  s
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => codePointOf(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => codePointOf(Number(dec)))
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");

/** Concatenate the text of every `text` block. */
export const textOf = (content: ReadonlyArray<ContentBlock> | null | undefined): string =>
  (content ?? [])
    .filter((b) => b.type === "text" && typeof b.text === "string")
    .map((b) => b.text as string)
    .join("");

/**
 * Tool results mix block types: text, search results, images, documents.
 * Flatten them into a readable string for the UI.
 */
export const describeToolResult = (content: ReadonlyArray<ContentBlock> | null | undefined): string =>
  (content ?? [])
    .map((block) => {
      if (block.type === "text" && typeof block.text === "string") return decodeEntities(block.text);
      if (block.type === "search_result") {
        const inner = Array.isArray(block.content) ? textOf(block.content) : "";
        const title = decodeEntities(String(block.title ?? ""));
        const source = String(block.source ?? "");
        return `[search result] ${title} — ${source}${inner ? `\n${decodeEntities(inner).slice(0, SEARCH_RESULT_PREVIEW_CHARS)}` : ""}`;
      }
      return `[${String(block.type)}]`;
    })
    .join("\n")
    .trim();
