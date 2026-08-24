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
 * The audio MIME types an `input_audio` part can actually carry, mapped to the
 * ONE spelling the provider accepts for each.
 *
 * `input_audio.format` is an enum of exactly two values — `"wav" | "mp3"` in the
 * OpenAI SDK's own `ChatCompletionContentPartInputAudio.InputAudio` — and both
 * runtimes derive that string from the block's `mime_type`. So the constraint is
 * not "audio converts": it is "audio converts for two subtypes, spelled the way
 * the provider spells them". Measured on `@langchain/openai@1.2.0` and
 * `langchain-core@1.2.13`, emitting through this adapter's own converter:
 *
 *   AG-UI mimeType     JS (@langchain/openai)     Python (langchain-core)
 *   -----------------  -------------------------  --------------------------
 *   audio/wav          format "wav" ✓             format "wav" ✓
 *   audio/mp3          format "mp3" ✓             format "mp3" ✓
 *   audio/mpeg         THREW                      format "mpeg" ✗ (API 400)
 *   audio/ogg          THREW                      format "ogg" ✗ (API 400)
 *   audio/aac          THREW                      format "aac" ✗ (API 400)
 *   audio/webm         THREW                      format "webm" ✗ (API 400)
 *   audio/x-wav        THREW                      format "x-wav" ✗ (API 400)
 *   AUDIO/WAV          THREW                      format "WAV" ✗ (API 400)
 *   audio/wav;codecs=1 format "wav" ✓             format "wav; codecs=1" ✗
 *
 * Two things fall out of that table, and this map exists for both.
 *
 * FIRST: `audio/mpeg` is the IANA-registered MIME type for MP3, and it is what
 * browsers, OS file pickers and `file(1)` report for a `.mp3`. It is therefore
 * the single most common audio attachment on the web, and it is NOT on the
 * provider's allow-list — `audio/mp3` is, which is the non-standard spelling.
 * Refusing `audio/mpeg` would leave the common case permanently on `image_url`
 * (a guaranteed provider 400 for a non-image part); passing it through unchanged
 * kills the JS run inside the translator and sends Python an invalid enum value.
 * Rewriting the spelling is the only outcome where an MP3 actually reaches the
 * model, so this map normalizes rather than merely narrows.
 *
 * SECOND: the two runtimes disagree about everything they do NOT accept. JS
 * parses the MIME type and throws on an unlisted subtype; Python takes
 * `mime_type.split("/")[-1]` verbatim and forwards it, so a bad type dies
 * locally in one runtime and at the API in the other — and a case difference or
 * a `;codecs=` parameter is enough to split them on a type BOTH could have
 * handled. Normalizing to a canonical spelling before emitting removes that
 * divergence at the source: after this map, the only `mime_type` either runtime
 * ever sees on an audio block is `audio/wav` or `audio/mp3`, which JS's parser
 * and Python's naive split both reduce to the same accepted enum value.
 *
 * Keys are the case-folded MIME type with any parameters stripped (MIME types
 * are case-insensitive per RFC 2045 §5.1, so `AUDIO/WAV` is a legal spelling of
 * a supported type and must not be treated as an unsupported one). The WAV
 * aliases are the registered and de-facto spellings of the same RIFF/WAVE
 * container; they name a format the provider accepts and differ only in how they
 * are written, which is the same defect as `audio/mpeg`.
 *
 * Kept in lockstep with `_OPENAI_AUDIO_MIME_TYPES` in the Python adapter. A
 * divergence here is the class of bug this converter exists to fix.
 *
 * Revisit when `input_audio.format` grows a third value.
 */
const OPENAI_AUDIO_MIME_TYPES = new Map<string, string>([
  ["audio/wav", "audio/wav"],
  ["audio/x-wav", "audio/wav"],
  ["audio/wave", "audio/wav"],
  ["audio/vnd.wave", "audio/wav"],
  ["audio/mp3", "audio/mp3"],
  ["audio/mpeg", "audio/mp3"],
]);

/**
 * The provider-accepted spelling for an audio MIME type, or `undefined` if the
 * provider cannot carry that audio format at all.
 *
 * A `Map` rather than an object literal for the same reason as
 * {@link AGUI_MEDIA_TYPES}: the key is derived from a client-supplied MIME
 * string, and an object literal would answer `"constructor"` with an inherited
 * function.
 */
function normalizedAudioMimeType(mimeType: string | undefined): string | undefined {
  // Parameters (`;codecs=…`, `;charset=…`) are part of a legal MIME type but not
  // part of its identity, and Python's translator would forward them into the
  // `format` enum verbatim.
  const base = (mimeType ?? "").split(";")[0].trim().toLowerCase();
  return OPENAI_AUDIO_MIME_TYPES.get(base);
}

/**
 * Which LangChain standard content block an AG-UI media item becomes — with the
 * MIME type to emit for it — or `null` to keep the pre-existing `image_url`
 * block.
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
 *   audio, data,      input_audio ✓                       input_audio ✓
 *   wav/mp3 spelling  (after {@link OPENAI_AUDIO_MIME_TYPES} normalizes the
 *                     spelling — the raw MIME type does NOT necessarily convert)
 *   audio, data,      throws ("must have mime type of     forwards an invalid
 *   any other type    audio/wav or audio/mp3")            `format` enum → API 400
 *                     — so these keep `image_url`; see {@link OPENAI_AUDIO_MIME_TYPES}
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
 * Note what the audio rows do NOT say: they do not say "audio, data converts".
 * That claim held only for the `audio/wav` this was first measured on. The
 * document rows are unqualified because `file.file_data` carries the MIME type
 * inside the data URL rather than through an enum, so no subtype is special.
 *
 * Revisit a row when its translator grows support for that combination.
 */
function standardBlockTypeFor(
  mediaType: string,
  source: InputContentDataSource | InputContentUrlSource
): { type: "audio" | "file"; mimeType?: string } | null {
  // Every URL-sourced media block throws in both runtimes; only inline data
  // converts.
  if (source?.type !== "data") return null;
  if (mediaType === "audio") {
    const mimeType = normalizedAudioMimeType(source.mimeType);
    return mimeType ? { type: "audio", mimeType } : null;
  }
  if (mediaType === "document") return { type: "file", mimeType: source.mimeType };
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

/**
 * The return leg: which AG-UI media type a standard block becomes.
 *
 * A `Map` rather than an object literal, matching {@link MEDIA_CONTENT_TYPES},
 * and for the same reason: the key is `item.type` off an inbound block, which is
 * whatever the LangGraph server relayed from model or tool output and is not
 * author-controlled. An object literal answers `["constructor"]` or
 * `["toString"]` with an inherited `Object.prototype` member — truthy, and a
 * function — so a block typed after a prototype key would pass the gate and be
 * emitted with a FUNCTION as its AG-UI content type, failing schema validation
 * downstream. A `Map` sees only what was put in it, and `get` returns the media
 * type in the same lookup that decides the branch.
 */
const AGUI_MEDIA_TYPES = new Map<string, "audio" | "video" | "document" | "image">([
  ["audio", "audio"],
  ["video", "video"],
  ["file", "document"],
  ["image", "image"],
]);

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
 * KNOWN LIMIT, measured and deliberately not addressed here: the Responses API.
 * Everything above is Chat Completions, which is where the reported failure was.
 * On `useResponsesApi: true` the same emitted block behaves differently, and no
 * single shape fixes both — the JS-native shape is correct on Responses/v1 and
 * forwarded raw on the Chat Completions default path, which is the bug this
 * change exists to fix. Measured on `@langchain/openai@1.2.0`, before and after:
 *
 *   case                          before                     after
 *   ----------------------------  -------------------------  ----------------------
 *   audio, Responses, v1          input_image (wrong kind)   NO PART EMITTED
 *   audio, Responses, default     input_image (wrong kind)   input_audio ✓
 *   audio, Completions, either    input_image (wrong kind)   input_audio ✓
 *   document, Responses, v1       input_image (wrong kind)   input_file ✓
 *   document, Responses, default  input_image (wrong kind)   file.file_data
 *                                                            (Chat Completions
 *                                                            part shape; the
 *                                                            Responses form is
 *                                                            input_file)
 *
 * Five of the six rows go from a wrong-kind part to a right-kind one. Row 1 goes
 * the other way, and it is the one to know about: the attachment stops being
 * emitted at all, with no throw and no warning, so the model answers without ever
 * seeing it. The cause is upstream and not reachable from here —
 * `dist/converters/responses.js` handles the v1 block kinds in one chain and
 * audio's branch is empty (`} else if (block.type === "audio") {}`), the only
 * occurrence of "audio" in that converter, sitting between `file` and `image`
 * branches that both resolve a real part.
 *
 * Note what row 1 is NOT: a working path that this change broke. Before, the
 * audio went out labelled as an image, so the request carried a part the API
 * does not accept for audio. The run failed then and produces a wrong answer now
 * — worse to diagnose, but not a lost capability.
 *
 * This converter cannot tell the two transports apart. It receives messages and
 * nothing else; the model is constructed inside the graph, on the far side of the
 * LangGraph server call. Emitting per-transport is not possible from here.
 *
 * Revisit when the JS default path translates native blocks without a marker, or
 * when the Responses v1 converter grows an audio branch.
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

/**
 * The `image_url` URL for an AG-UI media source, or `null` if there isn't one.
 *
 * `source` is typed as required but arrives off the wire, so it is read
 * OPTIONALLY. A media item with no source at all is not a type error the caller
 * can rule out — {@link standardBlockTypeFor} already reads the same field as
 * `source?.type`, and the two paths must not disagree about whether it can be
 * absent. More importantly, this runs inside the loop that converts a whole
 * message list: a throw here discards every other block and every other message,
 * where the Python adapter's `isinstance` chain simply returns `None` and lets
 * the caller warn and drop the one bad item. Returning `null` is what keeps the
 * two runtimes degrading the same way.
 */
function mediaSourceToUrl(
  source: InputContentDataSource | InputContentUrlSource | null | undefined
): string | null {
  if (source?.type === "data") {
    return `data:${source.mimeType};base64,${source.value}`;
  } else if (source?.type === "url") {
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
 * always succeeds. Takes `mimeType` separately rather than reading it off the
 * source, because for audio the type that goes on the wire is the normalized
 * spelling {@link standardBlockTypeFor} resolved, not the one the client sent.
 */
function standardMediaBlock(
  type: StandardMediaBlock["type"],
  data: string,
  mimeType: string | undefined,
  filename?: string
): StandardMediaBlock {
  const block: StandardMediaBlock = {
    type,
    source_type: "base64",
    data,
    mime_type: mimeType,
  };
  const name = filename ?? (type === "file" ? deriveFilename(mimeType) : undefined);
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
    // A content array relayed by the LangGraph server can carry a JSON `null`
    // (or a bare string) where a block is expected, and `item.type` on one of
    // those aborts the conversion of every OTHER block and every other message.
    // One unusable block is dropped like any other unusable block.
    if (!item || typeof item !== "object") {
      console.warn("[convertLangchainMultimodalToAgui] Dropping content block: not an object");
      continue;
    }

    // Resolved before the chain so the branch gate and the emitted content type
    // are one lookup rather than two. `undefined` for every block kind this
    // converter does not recognise, prototype key or not.
    const type = AGUI_MEDIA_TYPES.get(item.type);

    if (item.type === "text" && item.text) {
      aguiContent.push({
        type: "text",
        text: item.text,
      });
    } else if (type) {
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
 * Inline documents, and inline audio IN A FORMAT THE PROVIDER CAN CARRY, use the
 * standard block for their modality (`audio`, `file`), because the block KIND is
 * what providers validate: a PDF sent as `image_url` carries its real MIME type
 * inside the data URL and is still rejected —
 *
 *     BadRequestError: 400 - Invalid MIME type. Only image types are supported.
 *     (code: invalid_image_format)
 *
 * — which killed the run rather than degrading it. Routing every modality through
 * `image_url` was correct when this converter was written (#1457) and stopped
 * being correct once LangChain grew standard multimodal blocks.
 *
 * Everything else — images, video, any URL-sourced media, and audio in a format
 * outside {@link OPENAI_AUDIO_MIME_TYPES} — keeps `image_url`, because the
 * standard block for those combinations throws inside the translator (JS) or
 * forwards an invalid `format` enum to the API (Python). See
 * {@link standardBlockTypeFor} for the measured table.
 *
 * Audio MIME types are NORMALIZED, not merely filtered: `audio/mpeg` — the
 * standard type for MP3 and the commonest audio attachment there is — is emitted
 * as the `audio/mp3` spelling the provider's enum actually lists. See
 * {@link OPENAI_AUDIO_MIME_TYPES}.
 */
function convertAguiMultimodalToLangchain(content: InputContent[]): LangchainContentBlock[] {
  const langchainContent: LangchainContentBlock[] = [];

  for (const item of content) {
    // Same reason as the inbound converter: this array is client JSON, nothing
    // validates it at this boundary in TypeScript, and `item.type` on a `null`
    // entry throws from inside the loop that converts the whole message list.
    if (!item || typeof item !== "object") {
      console.warn("[convertAguiMultimodalToLangchain] Dropping content item: not an object");
      continue;
    }

    if (item.type === "text") {
      langchainContent.push({
        type: "text",
        text: item.text,
      });
    } else if (MEDIA_CONTENT_TYPES.has(item.type)) {
      // ImageInputContent, AudioInputContent, VideoInputContent, DocumentInputContent
      const mediaItem = item as ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent;
      const standard = standardBlockTypeFor(item.type, mediaItem.source);

      if (standard) {
        langchainContent.push(
          standardMediaBlock(
            standard.type,
            (mediaItem.source as InputContentDataSource).value,
            standard.mimeType,
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
      // data with a declared MIME type. The decision then goes through the SAME
      // {@link standardBlockTypeFor} the typed path uses, so an audio type the
      // provider cannot carry is refused identically on both paths — url-only,
      // id-only, image and video items, and unsupported audio types, all keep the
      // historical `image_url` reference form because the standard block for
      // those throws inside the translator.
      const mimeType = item.mimeType ?? "";
      // Modality is read off a case-folded copy: MIME types are case-insensitive
      // (RFC 2045 §5.1), so `AUDIO/WAV` names the same modality as `audio/wav`
      // and must not be routed as a document. The ORIGINAL string is what gets
      // emitted for documents, where it is carried inside a data URL rather than
      // matched against an enum.
      const modality = mimeType.split(";")[0].trim().toLowerCase();

      if (item.data && !item.url && mimeType && !modality.startsWith("image/") && !modality.startsWith("video/")) {
        const mediaType = modality.startsWith("audio/") ? "audio" : "document";
        const standard = standardBlockTypeFor(mediaType, { type: "data", value: item.data, mimeType });
        if (standard) {
          langchainContent.push(
            standardMediaBlock(standard.type, item.data, standard.mimeType, item.filename)
          );
          continue;
        }
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
          tool_calls: (message.toolCalls ?? [])
            // Same reason the `arguments` read below is guarded: a tool call
            // missing its `function` is client JSON, not a shape the type can
            // rule out, and `tc.function.name` on one throws from inside the
            // loop that converts the whole message list. Python never reaches
            // this because Pydantic rejects the payload at parse time; here the
            // one unusable call is dropped (it has no name, so there is nothing
            // to invoke) and the rest of the conversion survives.
            .filter((tc: ToolCall) => {
              if (tc?.function) return true;
              console.warn(
                "[aguiMessagesToLangChain] Dropping tool call: no function name or arguments"
              );
              return false;
            })
            .map((tc: ToolCall) => ({
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
    // `c?.type`, not `c.type`: this array is whatever the graph put on the
    // message, and a `null` entry in it would throw out of `find` — aborting
    // the conversion of the whole message list rather than skipping the entry
    // and finding the text block that follows it.
    const contentText = content.find(c => c?.type === 'text')?.text
    return contentText ?? null;
  }

  return null
}
