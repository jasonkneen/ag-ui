/** Helpers for turning Managed Agents content blocks into display text. */

/** Any content block; only `type` is required to route it. */
type ContentBlock = { type: string } & Record<string, any>;

const codePointOf = (n: number): string =>
  Number.isInteger(n) && n >= 0 && n <= 0x10ffff ? String.fromCodePoint(n) : "�";

const decodeEntities = (s: string): string =>
  s
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => codePointOf(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => codePointOf(Number(dec)))
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");

/** Concatenate the text of every `text` block. */
export const textOf = (content: ReadonlyArray<ContentBlock> | undefined): string =>
  (content ?? [])
    .filter((b) => b.type === "text" && typeof b.text === "string")
    .map((b) => b.text as string)
    .join("");

/**
 * Tool results mix block types: text, search results, images, documents.
 * Flatten them into a readable string for the UI.
 */
export const describeToolResult = (content: ReadonlyArray<ContentBlock> | undefined): string =>
  (content ?? [])
    .map((block) => {
      if (block.type === "text" && typeof block.text === "string") return decodeEntities(block.text);
      if (block.type === "search_result") {
        const inner = Array.isArray(block.content) ? textOf(block.content) : "";
        const title = decodeEntities(String(block.title ?? ""));
        const source = String(block.source ?? "");
        return `[search result] ${title} — ${source}${inner ? `\n${decodeEntities(inner).slice(0, 300)}` : ""}`;
      }
      return `[${String(block.type)}]`;
    })
    .join("\n")
    .trim();
