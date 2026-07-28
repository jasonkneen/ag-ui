import { describe, expect, it } from "vitest";
import { SEARCH_RESULT_PREVIEW_CHARS, TOOL_DESCRIPTION_MAX_LENGTH, TOOL_NAME_MAX_LENGTH } from "../constants";
import { describeToolResult, textOf } from "../text";
import { customToolFrom, normalizeToolName } from "../tools";

describe("normalizeToolName", () => {
  it("keeps names that are already valid", () => {
    expect(normalizeToolName("show_chart")).toBe("show_chart");
    expect(normalizeToolName("Get-Weather_2")).toBe("Get-Weather_2");
  });

  it("replaces invalid characters, truncates, and never returns empty", () => {
    expect(normalizeToolName("search web!")).toBe("search_web_");
    expect(normalizeToolName("x".repeat(200))).toBe("x".repeat(TOOL_NAME_MAX_LENGTH));
    expect(normalizeToolName("")).toBe("tool");
  });
});

describe("customToolFrom", () => {
  it("builds an input schema and a default description", () => {
    expect(customToolFrom({ name: "ping", description: "", parameters: { properties: { a: { type: "string" } } } })).toEqual({
      type: "custom",
      name: "ping",
      description: "Tool ping",
      input_schema: { type: "object", properties: { a: { type: "string" } }, required: [] },
    });
  });

  it("handles missing parameters", () => {
    expect(customToolFrom({ name: "ping", description: "Ping", parameters: undefined as never }).input_schema).toEqual({
      type: "object",
      properties: {},
    });
  });

  it("normalizes the name and caps the description length", () => {
    const tool = customToolFrom({ name: "search web!", description: "d".repeat(TOOL_DESCRIPTION_MAX_LENGTH + 50), parameters: {} });
    expect(tool.name).toBe("search_web_");
    expect(tool.description).toHaveLength(TOOL_DESCRIPTION_MAX_LENGTH);
  });
});

describe("textOf", () => {
  it("joins only the text blocks", () => {
    expect(textOf([{ type: "text", text: "a" }, { type: "image" }, { type: "text", text: "b" }])).toBe("ab");
    expect(textOf(null)).toBe("");
  });
});

describe("describeToolResult", () => {
  it("decodes entities and summarizes each block type", () => {
    const described = describeToolResult([
      { type: "text", text: "5 &lt; 6 &amp;&amp; &#x1F600; &#65;" },
      { type: "search_result", title: "T", source: "https://x", content: [{ type: "text", text: "inner" }] },
      { type: "document" },
    ]);
    expect(described).toBe("5 < 6 && \u{1F600} A\n[search result] T — https://x\ninner\n[document]");
  });

  it("shows only a preview of a long search result body", () => {
    const described = describeToolResult([
      { type: "search_result", title: "T", source: "https://x", content: [{ type: "text", text: "b".repeat(SEARCH_RESULT_PREVIEW_CHARS + 200) }] },
    ]);
    expect(described).toBe(`[search result] T — https://x\n${"b".repeat(SEARCH_RESULT_PREVIEW_CHARS)}`);
  });

  it("substitutes U+FFFD for entities that do not denote a usable character", () => {
    // A lone surrogate makes the string ill-formed UTF-16: it cannot be encoded
    // as UTF-8, so it would arrive as U+FFFD (or crash the encoder) anyway.
    // Every port rejects it here instead, identically.
    expect(describeToolResult([{ type: "text", text: "a&#xD800;b" }])).toBe("a�b");
    expect(describeToolResult([{ type: "text", text: "a&#55296;b" }])).toBe("a�b");
    expect(describeToolResult([{ type: "text", text: "a&#xDFFF;b" }])).toBe("a�b");
    // Out of range, and the boundaries around the surrogate block, still work.
    expect(describeToolResult([{ type: "text", text: "a&#x110000;b" }])).toBe("a�b");
    expect(describeToolResult([{ type: "text", text: "a&#xD7FF;&#xE000;b" }])).toBe("a\uD7FF\uE000b");
    // A well-formed surrogate pair written as one code point is unaffected.
    expect(describeToolResult([{ type: "text", text: "&#x1F600;" }])).toBe("\u{1F600}");
  });

  it("uses a [type] placeholder for unknown blocks and handles nothing", () => {
    expect(describeToolResult([{ type: "image", source: {} }])).toBe("[image]");
    expect(describeToolResult(null)).toBe("");
  });
});
