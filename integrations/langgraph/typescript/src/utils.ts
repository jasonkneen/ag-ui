import { Message as LangGraphMessage } from "@langchain/langgraph-sdk";
import { State, SchemaKeys, LangGraphReasoning } from "./types";
import {
  Message,
  ReasoningMessage,
  ToolCall,
  TextInputContent,
  ImageInputContent,
  AudioInputContent,
  VideoInputContent,
  DocumentInputContent,
  InputContentDataSource,
  InputContentUrlSource,
  InputContent,
  UserMessage,
} from "@ag-ui/client";

export const DEFAULT_SCHEMA_KEYS = ["messages", "tools"];

export function filterObjectBySchemaKeys(obj: Record<string, any>, schemaKeys: string[]) {
  return Object.fromEntries(Object.entries(obj).filter(([key]) => schemaKeys.includes(key)));
}

export function getStreamPayloadInput({
  mode,
  state,
  schemaKeys,
}: {
  mode: "start" | "continue";
  state: State;
  schemaKeys: SchemaKeys;
}) {
  let input = mode === "start" ? state : null;
  // Do not input keys that are not part of the input schema
  if (input && schemaKeys?.input) {
    input = filterObjectBySchemaKeys(input, [...DEFAULT_SCHEMA_KEYS, ...schemaKeys.input]);
  }

  return input;
}

const MEDIA_CONTENT_TYPES = new Set(["image", "audio", "video", "document"]);

/**
 * Which LangChain standard content block an AG-UI media item becomes, or `null`
 * to keep the pre-existing `image_url` block.
 *
 * THE ALLOW-LIST IS NARROW ON PURPOSE. A standard block is only an improvement
 * where the translator downstream can actually accept it. Where it cannot, the
 * block is REJECTED INSIDE THE TRANSLATOR and the run dies — strictly worse than
 * the degraded-but-alive `image_url` payload that shipped before this change,
 * because it turns a bad request into a dead run. So this converter emits a
 * standard block only for combinations measured to convert, and leaves every
 * other combination exactly as it was: this change improves the paths it can
 * prove and regresses none.
 *
 * Measured against `@langchain/core@1.1.40` + `@langchain/openai@1.2.0` (JS) and
 * `langchain-core@1.2.13` (Python), through the real OpenAI translators:
 *
 *   AG-UI item        JS (@langchain/openai)              Python
 *   ----------------  ----------------------------------  --------------------------------
 *   audio, data       input_audio ✓                       input_audio ✓
 *   audio, url        throws ("must be formatted as a     throws ("Key base64 is required
 *                     data URL")                          for audio blocks")
 *   video, any        throws ("Unable to convert content  throws ("Block of type video is
 *                     block type 'video' ... not          not supported")
 *                     recognized")
 *   document, data    file.file_data ✓ — but ONLY with a  file.file_data ✓
 *                     filename; see {@link deriveFilename}
 *   document, url     throws (JS: needs a data URL)       throws ("does not support file
 *                                                         URLs")
 *   image, any        already worked as `image_url`, and is left alone
 *
 * Revisit a row when its translator grows support for that combination.
 */
function standardBlockTypeFor(
  mediaType: string,
  source: InputContentDataSource | InputContentUrlSource
): "audio" | "file" | null {
  // Every URL-sourced media block throws in both runtimes; only inline data
  // converts.
  if (source?.type !== "data") return null;
  if (mediaType === "audio") return "audio";
  if (mediaType === "document") return "file";
  return null;
}

/**
 * A filename for a `file` block whose AG-UI item did not carry one.
 *
 * Not cosmetic. `@langchain/openai` THROWS on a file block with no filename
 * ("a filename or name or title is needed via meta-data for OpenAI when working
 * with multimodal blocks"), so the document path this converter claims to
 * support has to carry one or it is not actually supported. Python only warns
 * and omits the key, but both runtimes emit the same block, so both substitute
 * the same derived name.
 */
function deriveFilename(mimeType: string | undefined): string {
  const subtype = (mimeType ?? "").split("/")[1]?.split(";")[0]?.split("+")[0]?.trim();
  return subtype ? `attachment.${subtype}` : "attachment";
}

/** The return leg: which AG-UI media type a standard block becomes. */
const AGUI_MEDIA_TYPES: Record<string, "audio" | "video" | "document" | "image"> = {
  audio: "audio",
  video: "video",
  file: "document",
  image: "image",
};

/**
 * The media block this adapter emits: a LangChain `source_type` data block.
 *
 * THIS IS THE ONLY REPRESENTATION THAT TRANSLATES IN BOTH RUNTIMES, and this
 * adapter needs both — it is a JavaScript package that talks to a LangGraph
 * server which is usually the Python one. Measured against the locked
 * dependencies (`@langchain/core@1.1.40` + `@langchain/openai@1.2.0`, and
 * `langchain-core` on the Python side):
 *
 *   shape                                    JS default path      Python
 *   ---------------------------------------  -------------------  ------------
 *   source_type + data + mime_type           file.file_data ✓     same ✓
 *   Python native (base64/mime_type)         VERBATIM ✗           ✓
 *   JS native (data/mimeType/metadata)       VERBATIM ✗           ValueError ✗
 *
 * "VERBATIM" is the trap. The block is neither translated nor rejected: it is
 * forwarded to the provider exactly as written, so the request carries a content
 * part the API does not accept. On the JS side that happens because the default
 * conversion path gates translation behind `isDataContentBlock`, which tests for
 * `source_type` and nothing else; a block without it is not recognised as media
 * at all.
 *
 * `@langchain/core` does mark this family `@deprecated` ("Use
 * ContentBlock.Multimodal.Data instead"), and the JS-native shape it points to
 * works — but ONLY on the v1 conversion path, which requires
 * `response_metadata.output_version === "v1"` on the message. This adapter does
 * not set that marker and cannot set it for a graph it does not own, so on the
 * path that actually runs, the non-deprecated shape is the one that silently
 * fails. Deprecated-and-translated beats current-and-forwarded-raw.
 *
 * Revisit when the JS default path translates native blocks without a marker.
 */
interface StandardMediaBlock {
  type: "audio" | "file";
  /**
   * The recognition key. `@langchain/core`'s `isDataContentBlock` gate tests for
   * `source_type` and nothing else, so a block without it is not seen as media.
   */
  source_type: "base64";
  /** Base64 payload. */
  data: string;
  mime_type?: string;
  /**
   * `filename` lives under `metadata` rather than at the top level, because that
   * is where both runtimes look — JS via `getRequiredFilenameFromMetadata`,
   * Python via `convert_to_openai_data_block`'s backward-compat branch. JS THROWS
   * when it cannot find one on a file block, which is why the document path never
   * emits without it (see {@link deriveFilename}).
   */
  metadata?: { filename?: string };
}

/** A LangChain content block as this adapter emits it. */
type LangchainContentBlock =
  | { type: "text"; text?: string }
  | { type: "image_url"; image_url: { url: string } }
  | StandardMediaBlock;

function mediaSourceToUrl(source: InputContentDataSource | InputContentUrlSource): string | null {
  if (source.type === "data") {
    return `data:${source.mimeType};base64,${source.value}`;
  } else if (source.type === "url") {
    return source.value;
  }
  return null;
}

/**
 * The attachment's original filename, if the client sent one.
 *
 * `metadata: { filename }` is the established AG-UI carrier for it — the client's
 * own `backward-compatibility-0-0-47` middleware migrates the legacy
 * `BinaryInputContent.filename` into exactly that shape. Note this reads ONE key
 * out of metadata rather than copying the object: a top-level `metadata` key on
 * the block itself is what issue #2100 was about, and `filename` is a documented
 * field of the block.
 */
function filenameFromMetadata(metadata: unknown): string | undefined {
  if (metadata && typeof metadata === "object") {
    const filename = (metadata as { filename?: unknown }).filename;
    if (typeof filename === "string" && filename) return filename;
  }
  return undefined;
}

/**
 * Build the standard media block for an inline-data source.
 *
 * Only reached for combinations {@link standardBlockTypeFor} vouched for, so it
 * takes a data source and always succeeds.
 */
function standardMediaBlock(
  type: StandardMediaBlock["type"],
  source: InputContentDataSource,
  filename?: string
): StandardMediaBlock {
  const block: StandardMediaBlock = {
    type,
    source_type: "base64",
    data: source.value,
    mime_type: source.mimeType,
  };
  const name = filename ?? (type === "file" ? deriveFilename(source.mimeType) : undefined);
  if (name) block.metadata = { filename: name };
  return block;
}

/**
 * A media block on the way BACK from LangChain, in any of the three shapes this
 * adapter can legitimately receive. See {@link readIncomingMediaBlock}.
 */
interface IncomingMediaBlock {
  type: string;
  text?: string;
  image_url?: any;
  // Native LangChain.js (`ContentBlock.Multimodal.Data`).
  data?: string;
  mimeType?: string;
  fileId?: string;
  metadata?: { filename?: string; name?: string; title?: string };
  url?: string;
  // LangChain Python, and the deprecated JS `source_type` family.
  base64?: string;
  mime_type?: string;
  filename?: string;
  source_type?: string;
}

/**
 * Normalize an incoming media block to `{ value, isUrl, mimeType, filename }`.
 *
 * DELIBERATELY ACCEPTS THREE SHAPES, which is the "intentional cross-runtime
 * representation" half of this converter:
 *
 *   1. native LangChain.js — `data` / `url` / `fileId`, `mimeType`,
 *      `metadata.filename`. What this adapter now emits.
 *   2. LangChain Python — `base64` / `url`, `mime_type`, `filename`. NOT
 *      hypothetical: this package drives a LangGraph server through
 *      `@langchain/langgraph-sdk`, and that server is usually the Python one, so
 *      a graph's own messages come back in Python's shape. Refusing them here
 *      would reproduce the exact bug this file is fixing, mirrored — the
 *      attachment would vanish from `MESSAGES_SNAPSHOT` instead of from the
 *      provider request.
 *   3. the deprecated JS `source_type` family, still accepted by
 *      `@langchain/core`'s legacy conversion path and therefore still in the
 *      wild.
 *
 * Being liberal inbound and strict outbound is the whole point: one wire shape
 * leaves this adapter, three can arrive.
 */
function readIncomingMediaBlock(item: IncomingMediaBlock): {
  value: string;
  isUrl: boolean;
  mimeType?: string;
  filename?: string;
} | null {
  const filename = item.metadata?.filename ?? item.metadata?.name ?? item.metadata?.title ?? item.filename;
  const mimeType = item.mimeType ?? item.mime_type;
  const inlineData = item.data ?? item.base64;

  if (typeof inlineData === "string" && inlineData) {
    return { value: inlineData, isUrl: false, mimeType, filename };
  }
  if (typeof item.url === "string" && item.url) {
    return { value: item.url, isUrl: true, mimeType, filename };
  }
  // `fileId`-only blocks reference provider-side storage with no bytes and no
  // URL, and AG-UI's typed content classes have nowhere to put that.
  return null;
}

/**
 * Convert LangChain's multimodal content to AG-UI format.
 *
 * `image_url` blocks are converted to `ImageInputContent` with the appropriate
 * source type (data or URL). LangChain's standard media blocks (`image` /
 * `audio` / `video` / `file`) are converted back to the matching AG-UI content
 * type, which is what keeps a non-image attachment in the thread across a
 * MESSAGES_SNAPSHOT — a block kind missing here is an attachment that vanishes
 * from a reopened thread.
 */
function convertLangchainMultimodalToAgui(content: IncomingMediaBlock[]): InputContent[] {
  const aguiContent: InputContent[] = [];

  for (const item of content) {
    if (item.type === "text" && item.text) {
      aguiContent.push({
        type: "text",
        text: item.text,
      });
    } else if (AGUI_MEDIA_TYPES[item.type]) {
      const type = AGUI_MEDIA_TYPES[item.type];
      const incoming = readIncomingMediaBlock(item);

      if (!incoming) {
        console.warn(
          `[convertLangchainMultimodalToAgui] Dropping ${item.type} block: no data, base64 or url to carry back`
        );
        continue;
      }

      const metadata = incoming.filename ? { filename: incoming.filename } : undefined;

      if (incoming.isUrl) {
        aguiContent.push({
          type,
          source: {
            type: "url",
            value: incoming.value,
            ...(incoming.mimeType ? { mimeType: incoming.mimeType } : {}),
          },
          ...(metadata ? { metadata } : {}),
        } as InputContent);
      } else {
        aguiContent.push({
          type,
          source: {
            type: "data",
            value: incoming.value,
            // A base64 block with no MIME type is malformed rather than merely
            // terse, but an AG-UI data source REQUIRES one, so fall back to the
            // least wrong thing instead of dropping the attachment.
            mimeType: incoming.mimeType || "application/octet-stream",
          },
          ...(metadata ? { metadata } : {}),
        } as InputContent);
      }
    } else if (item.type === "image_url") {
      // LangChain only uses `image_url` blocks for all media, so we always
      // produce ImageInputContent here. The true media type is not recoverable.
      const imageUrl = typeof item.image_url === "string"
        ? item.image_url
        : item.image_url?.url;

      if (!imageUrl) continue;

      // Parse data URLs to extract base64 data
      if (imageUrl.startsWith("data:")) {
        // Format: data:mime_type;base64,data
        const [header, data] = imageUrl.split(",", 2);
        const mimeType = header.includes(":")
          ? header.split(":")[1].split(";")[0]
          : "image/png";

        aguiContent.push({
          type: "image",
          source: {
            type: "data",
            value: data || "",
            mimeType,
          },
        });
      } else {
        // Regular URL
        aguiContent.push({
          type: "image",
          source: {
            type: "url",
            value: imageUrl,
          },
        });
      }
    }
  }

  return aguiContent;
}

/**
 * Convert AG-UI multimodal content to LangChain's format.
 *
 * Handles the new typed content classes (ImageInputContent, AudioInputContent,
 * VideoInputContent, DocumentInputContent) as well as legacy BinaryInputContent
 * for backwards compatibility.
 *
 * Inline audio and inline documents use the standard block for their modality
 * (`audio`, `file`), because the block KIND is what providers validate: a PDF
 * sent as `image_url` carries its real MIME type inside the data URL and is
 * still rejected —
 *
 *     BadRequestError: 400 - Invalid MIME type. Only image types are supported.
 *     (code: invalid_image_format)
 *
 * — which killed the run rather than degrading it. Routing every modality through
 * `image_url` was correct when this converter was written (#1457) and stopped
 * being correct once LangChain grew standard multimodal blocks.
 *
 * Everything else — images, video, and any URL-sourced media — keeps `image_url`,
 * because the standard block for those combinations throws inside the translator.
 * See {@link standardBlockTypeFor} for the measured table.
 */
function convertAguiMultimodalToLangchain(content: InputContent[]): LangchainContentBlock[] {
  const langchainContent: LangchainContentBlock[] = [];

  for (const item of content) {
    if (item.type === "text") {
      langchainContent.push({
        type: "text",
        text: item.text,
      });
    } else if (MEDIA_CONTENT_TYPES.has(item.type)) {
      // ImageInputContent, AudioInputContent, VideoInputContent, DocumentInputContent
      const mediaItem = item as ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent;
      const blockType = standardBlockTypeFor(item.type, mediaItem.source);

      if (blockType) {
        langchainContent.push(
          standardMediaBlock(
            blockType,
            mediaItem.source as InputContentDataSource,
            filenameFromMetadata((mediaItem as { metadata?: unknown }).metadata)
          )
        );
        continue;
      }

      const url = mediaSourceToUrl(mediaItem.source);
      if (url) {
        langchainContent.push({
          type: "image_url",
          image_url: { url },
        });
      } else {
        console.warn(`[convertAguiMultimodalToLangchain] Dropping ${item.type} content: source could not be converted to URL`);
      }
    } else if (item.type === "binary") {
      // Legacy BinaryInputContent — backwards compatibility.
      //
      // Split on the MIME type, which is the only modality signal a legacy item
      // carries (the typed classes above announce their own), and only for inline
      // data with a declared MIME type. That is the same narrow allow-list the
      // typed path uses, for the same reason: url-only, id-only, image and video
      // items keep the historical `image_url` reference form because the standard
      // block for those throws inside the translator.
      const mimeType = item.mimeType ?? "";

      if (item.data && !item.url && mimeType && !mimeType.startsWith("image/") && !mimeType.startsWith("video/")) {
        const blockType = mimeType.startsWith("audio/") ? "audio" : "file";
        langchainContent.push(
          standardMediaBlock(blockType, { type: "data", value: item.data, mimeType }, item.filename)
        );
        continue;
      }

      let url: string;

      // Prioritize url, then data, then id
      if (item.url) {
        url = item.url;
      } else if (item.data) {
        // Construct data URL from base64 data
        url = `data:${item.mimeType};base64,${item.data}`;
      } else if (item.id) {
        // Use id as a reference
        url = item.id;
      } else {
        console.warn("[convertAguiMultimodalToLangchain] Dropping BinaryInputContent: no url, data, or id provided");
        continue;
      }

      langchainContent.push({
        type: "image_url",
        image_url: { url },
      });
    }
  }

  return langchainContent;
}

// A reasoning content block as it appears on a LangChain assistant message
// (OpenAI Responses `responses/v1` shape). It is not part of the LangGraph SDK's
// typed content union, so it is declared here for narrowing.
interface ReasoningSummaryEntry {
  type?: string;
  text?: string;
}

interface ReasoningContentBlock {
  type: "reasoning";
  id?: string;
  summary?: ReasoningSummaryEntry[];
  encrypted_content?: string;
  // Flat-text shapes emitted by some non-OpenAI providers.
  reasoning?: string;
  text?: string;
}

function isReasoningBlock(block: unknown): block is ReasoningContentBlock {
  return (
    typeof block === "object" &&
    block !== null &&
    (block as { type?: unknown }).type === "reasoning"
  );
}

// Extract the human-readable reasoning text from a reasoning content block.
function reasoningBlockSummaryText(block: ReasoningContentBlock): string {
  if (Array.isArray(block.summary)) {
    const parts = block.summary
      .map((entry) => entry?.text)
      .filter((text): text is string => Boolean(text));
    // Join multi-part summaries with a newline so the parts stay legible
    // instead of being mashed together ("A\nB", not "AB").
    if (parts.length) return parts.join("\n");
  }
  return block.reasoning ?? block.text ?? "";
}

// Turn a LangChain reasoning content block into an AG-UI ReasoningMessage,
// preserving the block id (the provider's `rs_…` handle — under store=true it is
// the only round-trip key) and any encrypted content (needed for store=false).
// Returns null only for a wholly empty block (nothing to render or round-trip).
function reasoningBlockToAguiMessage(
  block: ReasoningContentBlock,
  assistantId: string,
  index = 0,
): ReasoningMessage | null {
  const text = reasoningBlockSummaryText(block);
  const encrypted = block.encrypted_content;
  if (!block.id && !text && !encrypted) return null;
  const message: ReasoningMessage = {
    // Include the block index in the fallback id so multiple id-less reasoning
    // blocks on one message don't collide on the same id.
    id: String(block.id ?? `${assistantId}-reasoning-${index}`),
    role: "reasoning",
    content: text,
  };
  if (encrypted) message.encryptedValue = encrypted;
  return message;
}

// Rebuild the LangChain reasoning content block from an AG-UI ReasoningMessage
// (inverse of reasoningBlockToAguiMessage).
function aguiReasoningMessageToBlock(message: ReasoningMessage): ReasoningContentBlock {
  const block: ReasoningContentBlock = {
    type: "reasoning",
    id: message.id,
    summary: message.content
      ? [{ type: "summary_text", text: message.content }]
      : [],
  };
  if (message.encryptedValue) block.encrypted_content = message.encryptedValue;
  return block;
}

export function langchainMessagesToAgui(messages: LangGraphMessage[]): Message[] {
  const out: Message[] = [];
  for (const message of messages) {
    switch (message.type) {
      case "human": {
        // Handle multimodal content
        let userContent: string | InputContent[];
        if (Array.isArray(message.content)) {
          userContent = convertLangchainMultimodalToAgui(message.content as any);
        } else {
          userContent = stringifyIfNeeded(resolveMessageContent(message.content));
        }

        out.push({
          id: message.id!,
          role: "user",
          content: userContent,
        });
        break;
      }
      case "ai": {
        // "generic" messages are treated the same as "ai" — LangGraph
        // emits them for non-chat models that don't set a specific type.
        // Surface reasoning content blocks as standalone ReasoningMessages
        // placed BEFORE the assistant message (matching streaming order), so a
        // client with no persistent checkpoint can round-trip them.
        if (Array.isArray(message.content)) {
          message.content.forEach((block, index) => {
            if (isReasoningBlock(block)) {
              const reasoningMsg = reasoningBlockToAguiMessage(block, message.id!, index);
              if (reasoningMsg) out.push(reasoningMsg);
            }
          });
        }
        const aiContent = resolveMessageContent(message.content);
        out.push({
          id: message.id!,
          role: "assistant",
          content: aiContent ? stringifyIfNeeded(aiContent) : '',
          toolCalls: message.tool_calls?.map((tc) => ({
            id: tc.id!,
            type: "function",
            function: {
              name: tc.name,
              // Default missing args to "{}" (parity with the Python side);
              // JSON.stringify(undefined) would emit an invalid `undefined`.
              arguments: JSON.stringify(tc.args ?? {}),
            },
          })),
        });
        break;
      }
      case "system":
        out.push({
          id: message.id!,
          role: "system",
          content: stringifyIfNeeded(resolveMessageContent(message.content)),
        });
        break;
      case "tool":
        out.push({
          id: message.id!,
          role: "tool",
          content: stringifyIfNeeded(resolveMessageContent(message.content)),
          toolCallId: message.tool_call_id,
          // A LangChain tool result signals failure only through `status`, with no
          // error text. Restore AG-UI's `error` so the failure survives the round
          // trip; the value is a fixed sentinel (#2305) because the original text is
          // not recoverable from the flag alone.
          ...(message.status === "error" ? { error: "error" } : {}),
        });
        break;
      default:
        if ((message as any).type === "generic") {
          // Re-enter the "ai" branch for generic messages
          const aiMsg = message as any;
          if (Array.isArray(aiMsg.content)) {
            aiMsg.content.forEach((block: any, index: number) => {
              if (isReasoningBlock(block)) {
                const reasoningMsg = reasoningBlockToAguiMessage(block, aiMsg.id, index);
                if (reasoningMsg) out.push(reasoningMsg);
              }
            });
          }
          const aiContent = resolveMessageContent(aiMsg.content);
          out.push({
            id: aiMsg.id,
            role: "assistant",
            content: aiContent ? stringifyIfNeeded(aiContent) : '',
            toolCalls: aiMsg.tool_calls?.map((tc: any) => ({
              id: tc.id!,
              type: "function",
              function: {
                name: tc.name,
                arguments: JSON.stringify(tc.args ?? {}),
              },
            })),
          });
          break;
        }
        throw new Error("message type returned from LangGraph is not supported.");
    }
  }
  return out;
}

export function aguiMessagesToLangChain(messages: Message[]): LangGraphMessage[] {
  const out: LangGraphMessage[] = [];
  // Reasoning is display-only at the AG-UI layer but lives as a content block ON
  // the assistant AIMessage at the LangChain layer. To round-trip reasoning
  // without loss (so a stateless client can hand the model back its own
  // chain-of-thought), buffer reasoning messages and re-attach them as content
  // blocks on the assistant that follows (matching streaming order). Developer
  // messages stay dropped — they are configured on the agent itself.
  //
  // Reasoning that is NOT immediately followed by an assistant message (trailing,
  // or followed by a user/tool/system message) is intentionally discarded: there
  // is no assistant to attach it to, and re-materializing it as a standalone
  // message causes exponential message duplication and tool-call loops under the
  // add_messages reducer. The snapshot side (langchainMessagesToAgui) only ever
  // emits reasoning immediately before its assistant, so this drop never affects
  // a real round-trip — only hand-crafted / partial inputs.
  let pendingReasoning: ReasoningContentBlock[] = [];
  for (const message of messages) {
    switch (message.role) {
      case "reasoning":
        pendingReasoning.push(aguiReasoningMessageToBlock(message));
        continue;
      case "developer":
        continue;
      case "user": {
        pendingReasoning = [];
        // Handle multimodal content
        let content: UserMessage['content'];
        if (typeof message.content === "string") {
          content = message.content;
        } else if (Array.isArray(message.content)) {
          content = convertAguiMultimodalToLangchain(message.content) as any;
        } else {
          content = String(message.content);
        }

        out.push({
          id: message.id,
          role: message.role,
          content,
          type: "human",
        } as LangGraphMessage);
        break;
      }
      case "assistant": {
        // Fold any buffered reasoning blocks onto this assistant message.
        let content: string | Array<ReasoningContentBlock | { type: "text"; text: string }>;
        if (pendingReasoning.length) {
          const blocks: Array<ReasoningContentBlock | { type: "text"; text: string }> = [
            ...pendingReasoning,
          ];
          if (message.content) blocks.push({ type: "text", text: message.content });
          content = blocks;
          pendingReasoning = [];
        } else {
          content = message.content ?? "";
        }
        out.push({
          id: message.id,
          type: "ai",
          role: message.role,
          content,
          tool_calls: (message.toolCalls ?? []).map((tc: ToolCall) => ({
            id: tc.id,
            name: tc.function.name,
            // Guard empty/absent arguments (parity with the Python side):
            // JSON.parse("") throws and would abort the whole conversion.
            args: tc.function.arguments ? JSON.parse(tc.function.arguments) : {},
            type: "tool_call",
          })),
        } as LangGraphMessage);
        break;
      }
      case "system":
        pendingReasoning = [];
        out.push({
          id: message.id,
          role: message.role,
          content: message.content,
          type: "system",
        } as LangGraphMessage);
        break;
      case "tool":
        pendingReasoning = [];
        out.push({
          content: message.content,
          role: message.role,
          type: message.role,
          tool_call_id: message.toolCallId,
          id: message.id,
          // Carry the AG-UI failure signal onto LangChain's tool-result status, so a
          // client-reported tool failure is not delivered to the model as a success.
          status: message.error ? "error" : "success",
        } as LangGraphMessage);
        break;
      default:
        console.error(`Message role ${(message as { role: string }).role} is not implemented`);
        throw new Error("message role is not supported.");
    }
  }
  return out;
}

function stringifyIfNeeded(item: any) {
  if (typeof item === "string") return item;
  return JSON.stringify(item);
}

export function resolveReasoningContent(eventData: any): LangGraphReasoning | null {
  const content = eventData.chunk?.content

  if (content && Array.isArray(content) && content.length && content[0]) {
    const block = content[0];

    // Old langchain-anthropic format: { type: "thinking", thinking: "..." }
    if (block.type === 'thinking' && block.thinking) {
      const result: LangGraphReasoning = {
        text: block.thinking,
        type: 'text',
        index: block.index ?? 0,
      }
      // Extract signature if present (Anthropic extended thinking signature)
      if (block.signature) {
        result.signature = block.signature;
      }
      return result;
    }

    // New LangChain standardized format: { type: "reasoning", reasoning: "..." }
    if (block.type === 'reasoning' && block.reasoning) {
      return {
        text: block.reasoning,
        type: 'text',
        index: block.index ?? 0,
      }
    }

    // OpenAI Responses API v1 format: { type: "reasoning", summary: [{ text: "..." }] }
    //
    // The reasoning item's canonical id (OpenAI `rs_…`) only travels on
    // text-less chunks: the `response.output_item.added` chunk ({ id,
    // summary: [] }) and — depending on the langchain-openai version — the
    // `…summary_part.added` chunk ({ id, summary: [{ text: "" }] }). The
    // `…summary_text.delta` chunks carry text but no id. Surface the id
    // carriers (instead of dropping them for having no text) so the streamed
    // reasoning message can adopt the canonical id — the id the snapshot
    // converter emits for the same block; handleReasoningEvent stashes the id
    // without opening a message, so summary-less (store=true) items still
    // render nothing. Only the first summary part takes the id: later parts
    // belong to the same item, and reusing its id would mint two messages
    // with one id.
    if (block.type === 'reasoning' && Array.isArray(block.summary)) {
      if (block.summary.length === 0 && block.id) {
        return { type: 'text', text: '', index: block.index ?? 0, id: String(block.id) };
      }
      const part = block.summary[0];
      if (part && typeof part === 'object' && (part.text || block.id)) {
        const result: LangGraphReasoning = {
          type: 'text',
          text: part.text ?? '',
          index: part.index ?? 0,
        };
        if (block.id && (part.index ?? 0) === 0) {
          result.id = String(block.id);
        }
        return result;
      }
    }

    // Bedrock Converse API format: { type: "reasoning_content", reasoning_content: { type: "text", text: "..." } }
    if (block.type === 'reasoning_content' && block.reasoning_content?.text) {
      return {
        type: 'text',
        text: block.reasoning_content.text,
        index: block.reasoning_content.index ?? 0,
      }
    }
  }

  // OpenAI legacy format via additional_kwargs
  if (eventData.chunk?.additional_kwargs?.reasoning?.summary?.[0]) {
    const data = eventData.chunk.additional_kwargs.reasoning.summary[0]
    if (!data || !data.text) return null
    return {
      type: 'text',
      text: data.text,
      index: data.index ?? 0,
    }
  }

  // DeepSeek-style format: additional_kwargs.reasoning_content (plain string)
  const reasoningContent = eventData.chunk?.additional_kwargs?.reasoning_content
  if (reasoningContent && typeof reasoningContent === 'string') {
    return {
      type: 'text',
      text: reasoningContent,
      index: 0,
    }
  }

  return null
}

/**
 * Resolves encrypted reasoning content from Anthropic responses.
 * This handles:
 * - `signature` fields on thinking blocks (cryptographic verification)
 * - `redacted_thinking` blocks with encrypted `data` (redacted chain-of-thought)
 */
export function resolveEncryptedReasoningContent(eventData: any): string | null {
  const content = eventData.chunk?.content

  if (!content || !Array.isArray(content) || !content.length || !content[0]) {
    return null;
  }

  // Anthropic redacted_thinking block: { type: "redacted_thinking", data: "..." }
  if (content[0].type === 'redacted_thinking' && content[0].data) {
    return content[0].data;
  }

  return null;
}

export function resolveMessageContent(content?: LangGraphMessage['content']): string | null {
  if (!content) return null;

  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content) && content.length) {
    const contentText = content.find(c => c.type === 'text')?.text
    return contentText ?? null;
  }

  return null
}
