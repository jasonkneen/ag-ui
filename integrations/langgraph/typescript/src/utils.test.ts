/**
 * Tests for multimodal message conversion between AG-UI and LangChain formats.
 */

import { Message as LangGraphMessage } from "@langchain/langgraph-sdk";
import {
  Message,
  UserMessage,
  TextInputContent,
  BinaryInputContent,
  ImageInputContent,
  AudioInputContent,
  VideoInputContent,
  DocumentInputContent,
} from "@ag-ui/client";
// Imported at the TOP LEVEL, not dynamically inside the boundary tests below.
// `@langchain/openai` pulls a large module graph, and on a cold CI runner the
// first `await import()` of it took 7.9s — charged to the test that happened to
// run first, which then failed vitest's 5s default while the later ones passed
// on the warm cache. File-level import time is not charged to a test.
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { aguiMessagesToLangChain, langchainMessagesToAgui, resolveReasoningContent } from "./utils";

describe("Multimodal Message Conversion", () => {
  describe("aguiMessagesToLangChain", () => {
    it("should convert text-only AG-UI message to LangChain", () => {
      const aguiMessage: UserMessage = {
        id: "test-1",
        role: "user",
        content: "Hello, world!",
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      expect(lcMessages).toHaveLength(1);
      expect(lcMessages[0].type).toBe("human");
      expect(lcMessages[0].content).toBe("Hello, world!");
      expect(lcMessages[0].id).toBe("test-1");
    });

    it("should convert ImageInputContent with URL source to LangChain", () => {
      const aguiMessage: UserMessage = {
        id: "test-img-url",
        role: "user",
        content: [
          { type: "text", text: "What's in this image?" },
          {
            type: "image",
            source: {
              type: "url",
              value: "https://example.com/photo.jpg",
            },
          } as ImageInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      expect(lcMessages).toHaveLength(1);
      expect(lcMessages[0].type).toBe("human");
      expect(Array.isArray(lcMessages[0].content)).toBe(true);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toHaveLength(2);

      expect(content[0].type).toBe("text");
      expect(content[0].text).toBe("What's in this image?");

      expect(content[1].type).toBe("image_url");
      expect(content[1].image_url.url).toBe("https://example.com/photo.jpg");
    });

    it("should convert ImageInputContent with data source to LangChain", () => {
      const aguiMessage: UserMessage = {
        id: "test-img-data",
        role: "user",
        content: [
          { type: "text", text: "Analyze this" },
          {
            type: "image",
            source: {
              type: "data",
              value: "iVBORw0KGgoAAAANSUhEUgAAAAUA",
              mimeType: "image/png",
            },
          } as ImageInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      expect(lcMessages).toHaveLength(1);
      expect(Array.isArray(lcMessages[0].content)).toBe(true);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toHaveLength(2);

      const imageContent = content[1];
      expect(imageContent.type).toBe("image_url");
      expect(imageContent.image_url.url).toBe(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
      );
    });

    it("should convert AudioInputContent with inline data to an audio block", () => {
      const aguiMessage: UserMessage = {
        id: "test-audio-data",
        role: "user",
        content: [
          { type: "text", text: "Transcribe this audio" },
          {
            type: "audio",
            source: {
              type: "data",
              value: "SGVsbG8=",
              mimeType: "audio/wav",
            },
          } as AudioInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toHaveLength(2);
      // An `audio` block, NOT `image_url`: providers validate the block kind, so
      // audio announced as an image is rejected outright. Base64 audio is one of
      // the two combinations both translators accept — see the boundary test
      // "carries an emitted audio block to OpenAI as an input_audio part".
      expect(content[1]).toEqual({
        type: "audio",
        source_type: "base64",
        data: "SGVsbG8=",
        mime_type: "audio/wav",
      });
    });

    // ── Combinations deliberately LEFT on the legacy `image_url` path ────────
    //
    // These are pinned decisions, not aspirations. For each one, the standard
    // media block THROWS inside the translator, so emitting it would convert the
    // pre-existing degraded request into a dead run. They stay on `image_url`
    // until the translators accept them. Do not "finish the job" by flipping one
    // of these without re-measuring the boundary first.

    it("keeps audio by URL on the legacy image_url path", () => {
      // JS: "URL audio blocks with source_type url must be formatted as a data
      // URL for ChatOpenAI". Python: ValueError "Key base64 is required for audio
      // blocks".
      const aguiMessage: UserMessage = {
        id: "test-audio-url",
        role: "user",
        content: [
          { type: "text", text: "Transcribe this audio" },
          {
            type: "audio",
            source: {
              type: "url",
              value: "https://example.com/audio.mp3",
            },
          } as AudioInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toHaveLength(2);
      expect(content[1]).toEqual({
        type: "image_url",
        image_url: { url: "https://example.com/audio.mp3" },
      });
    });

    it("keeps video on the legacy image_url path, whatever the source", () => {
      // JS: "Unable to convert content block type 'video' to provider-specific
      // format: not recognized." Python: ValueError "Block of type video is not
      // supported." Neither source type changes that.
      const aguiMessage: UserMessage = {
        id: "test-video",
        role: "user",
        content: [
          {
            type: "video",
            source: {
              type: "data",
              value: "dmlkZW9kYXRh",
              mimeType: "video/mp4",
            },
          } as VideoInputContent,
          {
            type: "video",
            source: {
              type: "url",
              value: "https://example.com/clip.mp4",
            },
          } as VideoInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toEqual([
        { type: "image_url", image_url: { url: "data:video/mp4;base64,dmlkZW9kYXRh" } },
        { type: "image_url", image_url: { url: "https://example.com/clip.mp4" } },
      ]);
    });

    it("keeps documents by URL on the legacy image_url path", () => {
      // JS: "URL file blocks with source_type url must be formatted as a data URL
      // for ChatOpenAI". Python: ValueError "OpenAI Chat Completions does not
      // support file URLs." Fetching the bytes on the caller's behalf is not this
      // adapter's job, so the URL keeps going out as it always did.
      const aguiMessage: UserMessage = {
        id: "test-doc",
        role: "user",
        content: [
          { type: "text", text: "Summarize this document" },
          {
            type: "document",
            source: {
              type: "url",
              value: "https://example.com/doc.pdf",
            },
            metadata: { filename: "doc.pdf" },
          } as DocumentInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toHaveLength(2);
      expect(content[1]).toEqual({
        type: "image_url",
        image_url: { url: "https://example.com/doc.pdf" },
      });
    });

    it("keeps a legacy binary by URL on the legacy image_url path", () => {
      // Same rule as the typed items above: only inline data converts.
      const aguiMessage: UserMessage = {
        id: "test-binary-pdf-url",
        role: "user",
        content: [
          {
            type: "binary",
            mimeType: "application/pdf",
            url: "https://example.com/legacy.pdf",
            filename: "legacy.pdf",
          } as BinaryInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toEqual([
        { type: "image_url", image_url: { url: "https://example.com/legacy.pdf" } },
      ]);
    });

    it("should send an attached PDF as a file block, not an image", () => {
      // THE REGRESSION THIS GUARDS. A PDF handed to a provider as `image_url` —
      // with `application/pdf` sitting inside the data URL — is rejected on the
      // block kind:
      //
      //     BadRequestError: 400 - Invalid MIME type. Only image types are
      //     supported. (code: invalid_image_format)
      //
      // and the exception kills the run rather than degrading it.
      const aguiMessage: UserMessage = {
        id: "test-doc-data",
        role: "user",
        content: [
          { type: "text", text: "Summarize this invoice" },
          {
            type: "document",
            source: {
              type: "data",
              value: "JVBERi0xLjQK",
              mimeType: "application/pdf",
            },
            metadata: { filename: "invoice-q2.pdf" },
          } as DocumentInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content[1]).toEqual({
        type: "file",
        source_type: "base64",
        data: "JVBERi0xLjQK",
        mime_type: "application/pdf",
        // `source_type` is the recognition key — the JS default conversion path
        // gates translation behind `isDataContentBlock`, which tests for exactly
        // that field. `filename` sits under `metadata` because that is where
        // BOTH runtimes look (JS `getRequiredFilenameFromMetadata`, Python
        // `convert_to_openai_data_block`'s backward-compat branch); without it
        // the JS translator THROWS.
        metadata: { filename: "invoice-q2.pdf" },
      });
      // Issue #2100 restated rather than dropped. The block DOES carry a
      // `metadata` key now, because that is the native home for the filename —
      // but only `filename`, never AG-UI's metadata object wholesale, which is
      // what made strict providers 400.
      expect(Object.keys(content[1].metadata)).toEqual(["filename"]);
    });

    it("derives a filename for a document that arrived without one", () => {
      // NOT cosmetic. `@langchain/openai` throws on a file block with no filename
      // ("a filename or name or title is needed via meta-data for OpenAI when
      // working with multimodal blocks"), so a document without one would land on
      // the very failure this narrowing exists to avoid. The substitute comes
      // from the MIME subtype and is applied in both runtimes so they agree.
      const aguiMessage: UserMessage = {
        id: "test-doc-no-filename",
        role: "user",
        content: [
          {
            type: "document",
            source: {
              type: "data",
              value: "JVBERi0xLjQK",
              mimeType: "application/pdf",
            },
          } as DocumentInputContent,
        ],
      };

      const content = aguiMessagesToLangChain([aguiMessage])[0].content as Array<any>;
      expect(content[0]).toEqual({
        type: "file",
        source_type: "base64",
        data: "JVBERi0xLjQK",
        mime_type: "application/pdf",
        metadata: { filename: "attachment.pdf" },
      });
    });

    it("should keep a document a document across the LangChain round trip", () => {
      // This is the MESSAGES_SNAPSHOT path. A block kind the return leg does not
      // understand is an attachment that disappears from the thread on the next
      // snapshot: the file was sent, the model read it, and a reopened thread
      // shows a bare line of text.
      const aguiMessage: UserMessage = {
        id: "test-doc-roundtrip",
        role: "user",
        content: [
          { type: "text", text: "Summarize this invoice" },
          {
            type: "document",
            source: {
              type: "data",
              value: "JVBERi0xLjQK",
              mimeType: "application/pdf",
            },
            metadata: { filename: "invoice-q2.pdf" },
          } as DocumentInputContent,
        ],
      };

      const roundTripped = langchainMessagesToAgui(
        aguiMessagesToLangChain([aguiMessage])
      );

      const content = (roundTripped[0] as UserMessage).content as Array<any>;
      expect(content[1]).toEqual({
        type: "document",
        source: {
          type: "data",
          value: "JVBERi0xLjQK",
          mimeType: "application/pdf",
        },
        metadata: { filename: "invoice-q2.pdf" },
      });
    });

    it("should send a legacy binary PDF as a file block", () => {
      // `BinaryInputContent` is deprecated but still accepted, and a deprecated
      // path that 400s is not meaningfully more supported than one that raises.
      const aguiMessage: UserMessage = {
        id: "test-binary-pdf",
        role: "user",
        content: [
          { type: "text", text: "Summarize this" },
          {
            type: "binary",
            mimeType: "application/pdf",
            data: "JVBERi0xLjQK",
            filename: "legacy-invoice.pdf",
          } as BinaryInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content[1]).toEqual({
        type: "file",
        source_type: "base64",
        data: "JVBERi0xLjQK",
        mime_type: "application/pdf",
        metadata: { filename: "legacy-invoice.pdf" },
      });
    });

    it("should handle BinaryInputContent for backwards compatibility", () => {
      const aguiMessage: UserMessage = {
        id: "test-binary-compat",
        role: "user",
        content: [
          { type: "text", text: "What's in this image?" },
          {
            type: "binary",
            mimeType: "image/jpeg",
            url: "https://example.com/photo.jpg",
          } as BinaryInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toHaveLength(2);

      expect(content[1].type).toBe("image_url");
      expect(content[1].image_url.url).toBe("https://example.com/photo.jpg");
    });

    it("should handle BinaryInputContent with base64 data for backwards compat", () => {
      const aguiMessage: UserMessage = {
        id: "test-binary-data",
        role: "user",
        content: [
          {
            type: "binary",
            mimeType: "image/png",
            data: "iVBORw0KGgoAAAANSUhEUgAAAAUA",
          } as BinaryInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      const content = lcMessages[0].content as Array<any>;
      expect(content).toHaveLength(1);
      expect(content[0].type).toBe("image_url");
      expect(content[0].image_url.url).toBe(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
      );
    });
  });

  // ── Modality survives the `image_url` round trip ──────────────────────────
  //
  // `image_url` is the fallback block for every modality the outbound leg cannot
  // send as a standard block — video always, audio outside the provider's format
  // enum, and every URL-sourced item — so reading the block kind literally on the
  // way back rewrote the thread: the user attached a video and MESSAGES_SNAPSHOT
  // came back holding an image, permanently, for every later read.
  //
  // The MIME type inside the data URL is the recovery signal, and these tests
  // pin BOTH halves: the type that comes back, and the fact that the block going
  // out is byte-for-byte what it was (the outbound shape is provider-measured and
  // must not move).
  describe("modality survives the image_url round trip", () => {
    const roundTrip = (item: any) => {
      const lc = aguiMessagesToLangChain([
        { id: "rt", role: "user", content: [item] } as UserMessage,
      ]);
      const agui = langchainMessagesToAgui(lc);
      return {
        wire: (lc[0].content as Array<any>)[0],
        content: ((agui[0] as UserMessage).content as Array<any>)[0],
      };
    };

    it("keeps a video a video across the round trip", () => {
      const { wire, content } = roundTrip({
        type: "video",
        source: { type: "data", value: "SGVsbG8=", mimeType: "video/mp4" },
        metadata: { filename: "clip.mp4" },
      } as VideoInputContent);

      // Unchanged on the wire: video still has no standard block that any
      // translator accepts, so it stays on `image_url` deliberately.
      expect(wire).toEqual({
        type: "image_url",
        image_url: { url: "data:video/mp4;base64,SGVsbG8=" },
      });
      expect(content).toEqual({
        type: "video",
        source: { type: "data", value: "SGVsbG8=", mimeType: "video/mp4" },
      });
    });

    it("keeps audio the provider cannot carry an audio across the round trip", () => {
      // `audio/ogg` is outside `input_audio.format`, so it rides `image_url` too.
      const { wire, content } = roundTrip({
        type: "audio",
        source: { type: "data", value: "SGVsbG8=", mimeType: "audio/ogg" },
      } as AudioInputContent);

      expect(wire).toEqual({
        type: "image_url",
        image_url: { url: "data:audio/ogg;base64,SGVsbG8=" },
      });
      expect(content).toEqual({
        type: "audio",
        source: { type: "data", value: "SGVsbG8=", mimeType: "audio/ogg" },
      });
    });

    it("keeps a legacy binary video a video across the round trip", () => {
      const { wire, content } = roundTrip({
        type: "binary",
        mimeType: "video/mp4",
        data: "SGVsbG8=",
      } as BinaryInputContent);

      expect(wire).toEqual({
        type: "image_url",
        image_url: { url: "data:video/mp4;base64,SGVsbG8=" },
      });
      expect(content).toEqual({
        type: "video",
        source: { type: "data", value: "SGVsbG8=", mimeType: "video/mp4" },
      });
    });

    it("still brings a genuine image back as an image", () => {
      const { content } = roundTrip({
        type: "image",
        source: { type: "data", value: "SGVsbG8=", mimeType: "image/png" },
      } as ImageInputContent);

      expect(content).toEqual({
        type: "image",
        source: { type: "data", value: "SGVsbG8=", mimeType: "image/png" },
      });
    });

    it("reads a non-media MIME type on the image_url path as a document", () => {
      // Nothing in this adapter emits a document as an `image_url` data URL, but a
      // graph relaying its own content can, and `document` is what the legacy
      // binary OUTBOUND leg calls the same MIME type. Symmetry, not guesswork.
      const agui = langchainMessagesToAgui([
        {
          id: "doc-data-url",
          type: "human",
          content: [
            { type: "image_url", image_url: { url: "data:application/pdf;base64,JVBERi0=" } },
          ],
        } as unknown as LangGraphMessage,
      ]);

      expect(((agui[0] as UserMessage).content as Array<any>)[0]).toEqual({
        type: "document",
        source: { type: "data", value: "JVBERi0=", mimeType: "application/pdf" },
      });
    });

    it("leaves a data URL with no MIME type an image", () => {
      // Nothing to read, so the pre-existing default stands rather than a guess.
      const agui = langchainMessagesToAgui([
        {
          id: "no-mime",
          type: "human",
          content: [{ type: "image_url", image_url: { url: "data:;base64,SGVsbG8=" } }],
        } as unknown as LangGraphMessage,
      ]);

      expect(((agui[0] as UserMessage).content as Array<any>)[0].type).toBe("image");
    });

    it("KNOWN LIMIT: a URL-sourced video comes back as an image", () => {
      // Not an oversight — an `image_url` block carries `{ url }` and nothing
      // else, so an https-hosted video arrives with no MIME type and no other
      // modality signal. Adding a key to the block is what issue #2100 was about
      // (providers 400 on unexpected keys inside a content block), and a file
      // extension is not a signal on signed or extensionless CDN URLs. This test
      // exists so the limit is visible and a future fix has to change it
      // deliberately.
      const { content } = roundTrip({
        type: "video",
        source: { type: "url", value: "https://example.com/clip.mp4", mimeType: "video/mp4" },
      } as VideoInputContent);

      expect(content).toEqual({
        type: "image",
        source: { type: "url", value: "https://example.com/clip.mp4" },
      });
    });
  });

  describe("langchainMessagesToAgui", () => {
    it("should convert text-only LangChain message to AG-UI", () => {
      const lcMessage: LangGraphMessage = {
        id: "test-4",
        type: "human",
        content: "Hello from LangChain",
      };

      const aguiMessages = langchainMessagesToAgui([lcMessage]);

      expect(aguiMessages).toHaveLength(1);
      expect(aguiMessages[0].role).toBe("user");
      expect(aguiMessages[0].content).toBe("Hello from LangChain");
    });

    it("should convert LangChain image_url to ImageInputContent with URL source", () => {
      const lcMessage: LangGraphMessage = {
        id: "test-lc-url",
        type: "human",
        content: [
          { type: "text", text: "What do you see?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/image.jpg" },
          },
        ] as any,
      };

      const aguiMessages = langchainMessagesToAgui([lcMessage]);

      expect(aguiMessages).toHaveLength(1);
      expect(aguiMessages[0].role).toBe("user");
      expect(Array.isArray(aguiMessages[0].content)).toBe(true);

      const content = aguiMessages[0].content as Array<TextInputContent | ImageInputContent>;
      expect(content).toHaveLength(2);

      // Check text content
      expect(content[0].type).toBe("text");
      expect((content[0] as TextInputContent).text).toBe("What do you see?");

      // Check image content - should now be ImageInputContent with URL source
      const imageContent = content[1] as ImageInputContent;
      expect(imageContent.type).toBe("image");
      expect(imageContent.source.type).toBe("url");
      expect((imageContent.source as { type: "url"; value: string }).value).toBe(
        "https://example.com/image.jpg"
      );
    });

    it("should convert LangChain data URL to ImageInputContent with data source", () => {
      const lcMessage: LangGraphMessage = {
        id: "test-lc-data",
        type: "human",
        content: [
          { type: "text", text: "Check this out" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,iVBORw0KGgo" },
          },
        ] as any,
      };

      const aguiMessages = langchainMessagesToAgui([lcMessage]);

      expect(aguiMessages).toHaveLength(1);
      expect(Array.isArray(aguiMessages[0].content)).toBe(true);

      const content = aguiMessages[0].content as Array<TextInputContent | ImageInputContent>;
      expect(content).toHaveLength(2);

      // Check that data URL was parsed correctly into ImageInputContent
      const imageContent = content[1] as ImageInputContent;
      expect(imageContent.type).toBe("image");
      expect(imageContent.source.type).toBe("data");

      const dataSource = imageContent.source as { type: "data"; value: string; mimeType: string };
      expect(dataSource.value).toBe("iVBORw0KGgo");
      expect(dataSource.mimeType).toBe("image/png");
    });
  });

  describe("Edge cases", () => {
    it("should handle empty content arrays", () => {
      const aguiMessage: UserMessage = {
        id: "test-7",
        role: "user",
        content: [],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      expect(lcMessages).toHaveLength(1);
      expect(Array.isArray(lcMessages[0].content)).toBe(true);
      expect((lcMessages[0].content as Array<any>)).toHaveLength(0);
    });

    it("should handle BinaryInputContent with only id for backwards compat", () => {
      const aguiMessage: UserMessage = {
        id: "test-8",
        role: "user",
        content: [
          {
            type: "binary",
            mimeType: "image/jpeg",
            id: "img-123",
          } as BinaryInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      expect(lcMessages).toHaveLength(1);
      const content = lcMessages[0].content as Array<any>;
      expect(content).toHaveLength(1);
      expect(content[0].type).toBe("image_url");
      expect(content[0].image_url.url).toBe("img-123");
    });

    it("should skip media content with unknown source type", () => {
      const aguiMessage: UserMessage = {
        id: "test-unknown-source",
        role: "user",
        content: [
          { type: "text", text: "Hello" },
          {
            type: "image",
            source: { type: "unknown" as any, value: "foo" },
          } as any,
        ],
      };
      const lcMessages = aguiMessagesToLangChain([aguiMessage]);
      const content = lcMessages[0].content as Array<any>;
      // Only text should remain, image with unknown source should be dropped
      expect(content).toHaveLength(1);
      expect(content[0].type).toBe("text");
    });

    it("should skip binary content without any source", () => {
      const aguiMessage: UserMessage = {
        id: "test-9",
        role: "user",
        content: [
          { type: "text", text: "Hello" },
          {
            type: "binary",
            mimeType: "image/jpeg",
            // No url, data, or id
          } as BinaryInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      expect(lcMessages).toHaveLength(1);
      const content = lcMessages[0].content as Array<any>;
      // Binary content should be skipped, only text remains
      expect(content).toHaveLength(1);
      expect(content[0].type).toBe("text");
    });
  });

  // ── Malformed input from the wire ────────────────────────────────────────
  //
  // Every value these converters read arrives over a wire: AG-UI content from a
  // client's JSON, LangChain content from whatever the LangGraph server relayed
  // out of model or tool output. Neither side is validated at this boundary in
  // TypeScript (Python's Pydantic models reject most of it before it reaches the
  // converter), so a missing key here is a `TypeError` thrown from INSIDE the
  // loop that converts a whole message list — which discards every other message
  // and every other block along with the bad one.
  //
  // The rule these tests pin: a malformed ITEM is dropped with a warning; it
  // never takes down its neighbours, its message, or the conversion.
  describe("malformed input is dropped, not thrown on", () => {
    it("drops a media item with no source instead of throwing", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const aguiMessage: UserMessage = {
        id: "test-no-source",
        role: "user",
        content: [
          { type: "text", text: "Hello" },
          // `source` absent entirely — the shape `standardBlockTypeFor` already
          // guards with `source?.type` but `mediaSourceToUrl` did not.
          { type: "image" } as unknown as ImageInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      expect(lcMessages).toHaveLength(1);
      const content = lcMessages[0].content as Array<any>;
      expect(content).toEqual([{ type: "text", text: "Hello" }]);
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining("Dropping image content: source could not be converted to URL")
      );
      warn.mockRestore();
    });

    it("keeps the surrounding content when one media item has no source", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const aguiMessage: UserMessage = {
        id: "test-bad-item-neighbours",
        role: "user",
        content: [
          { type: "text", text: "look at these" },
          { type: "audio", source: null } as unknown as AudioInputContent,
          {
            type: "image",
            source: { type: "url", value: "https://example.com/photo.jpg" },
          } as ImageInputContent,
        ],
      };

      const lcMessages = aguiMessagesToLangChain([aguiMessage]);

      expect(lcMessages).toHaveLength(1);
      const content = lcMessages[0].content as Array<any>;
      // The text and the good image both survive; only the bad item is gone.
      expect(content).toEqual([
        { type: "text", text: "look at these" },
        { type: "image_url", image_url: { url: "https://example.com/photo.jpg" } },
      ]);
      warn.mockRestore();
    });

    it("does not let one bad media item abort a whole multi-message conversion", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const lcMessages = aguiMessagesToLangChain([
        { id: "m1", role: "user", content: "first" } as UserMessage,
        {
          id: "m2",
          role: "user",
          content: [{ type: "document" } as unknown as DocumentInputContent],
        } as UserMessage,
        { id: "m3", role: "user", content: "third" } as UserMessage,
      ]);

      // The messages either side of the bad one are still there.
      expect(lcMessages.map((m) => m.id)).toEqual(["m1", "m2", "m3"]);
      expect(lcMessages[2].content).toBe("third");
      warn.mockRestore();
    });

    it("drops a null entry in an AG-UI content array", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const aguiMessage: UserMessage = {
        id: "test-null-agui-item",
        role: "user",
        content: [
          { type: "text", text: "before" },
          null as unknown as TextInputContent,
          { type: "text", text: "after" },
        ],
      };

      const content = aguiMessagesToLangChain([aguiMessage])[0].content as Array<any>;
      expect(content).toEqual([
        { type: "text", text: "before" },
        { type: "text", text: "after" },
      ]);
      warn.mockRestore();
    });

    it("drops a null entry in an inbound LangChain content array", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const agui = langchainMessagesToAgui([
        {
          id: "null-inbound-block",
          type: "human",
          content: [
            { type: "text", text: "before" },
            null,
            { type: "text", text: "after" },
          ],
        } as unknown as LangGraphMessage,
      ]);

      const content = (agui[0] as UserMessage).content as Array<any>;
      expect(content).toEqual([
        { type: "text", text: "before" },
        { type: "text", text: "after" },
      ]);
      warn.mockRestore();
    });

    it("tolerates a null block when resolving a message's text content", () => {
      // `resolveMessageContent` runs `content.find(c => c.type === "text")` over
      // an array the graph handed back; a null entry there killed the whole
      // conversion before the text block after it was ever looked at.
      const agui = langchainMessagesToAgui([
        {
          id: "null-block-system",
          type: "system",
          content: [null, { type: "text", text: "you are helpful" }],
        } as unknown as LangGraphMessage,
      ]);

      expect(agui).toHaveLength(1);
      expect(agui[0].content).toBe("you are helpful");
    });

    it("drops a tool call with no function instead of aborting the conversion", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const lcMessages = aguiMessagesToLangChain([
        {
          id: "assistant-bad-tc",
          role: "assistant",
          content: "calling",
          toolCalls: [
            { id: "tc-broken", type: "function" },
            {
              id: "tc-good",
              type: "function",
              function: { name: "get_weather", arguments: '{"city":"Paris"}' },
            },
          ],
        } as unknown as Message,
        { id: "after", role: "user", content: "and then" } as UserMessage,
      ]);

      // The conversion completed, and the message after the bad tool call is
      // still present.
      expect(lcMessages.map((m) => m.id)).toEqual(["assistant-bad-tc", "after"]);
      const toolCalls = (lcMessages[0] as any).tool_calls;
      expect(toolCalls).toEqual([
        { id: "tc-good", name: "get_weather", args: { city: "Paris" }, type: "tool_call" },
      ]);
      expect(warn).toHaveBeenCalledWith(expect.stringContaining("Dropping tool call"));
      warn.mockRestore();
    });
  });

  // ── Provider boundary ────────────────────────────────────────────────────
  //
  // The tests above assert the SHAPE this adapter emits. On their own that is
  // exactly the trap that let the wrong shape ship: a converter and its tests
  // agreeing on an invented schema look identical to a correct one. These tests
  // hand the emitted blocks to the real `@langchain/openai` translator and read
  // what lands on the wire, with a stub `fetch` so nothing leaves the process.
  describe("provider boundary (@langchain/openai)", () => {
    /** Convert through ChatOpenAI and return the outgoing content parts. */
    async function partsOnTheWire(blocks: unknown[]): Promise<any[]> {
      let body: any;
      const model = new ChatOpenAI({
        apiKey: "test-not-used",
        model: "gpt-4o-mini",
        configuration: {
          fetch: async (_url: any, init: any) => {
            body = JSON.parse(init.body);
            return new Response(
              JSON.stringify({
                id: "chatcmpl-test",
                object: "chat.completion",
                created: 0,
                model: "gpt-4o-mini",
                choices: [
                  {
                    index: 0,
                    message: { role: "assistant", content: "ok" },
                    finish_reason: "stop",
                  },
                ],
                usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
              }),
              { status: 200, headers: { "content-type": "application/json" } },
            );
          },
        },
      });

      // NO `response_metadata` marker. Setting `output_version: "v1"` here would
      // select a conversion path the adapter's real output never reaches — and
      // that is exactly how the previous round's shape passed its own tests
      // while being forwarded to the provider raw.
      await model.invoke([
        new HumanMessage({ content: [{ type: "text", text: "hi" }, ...blocks] as any }),
      ]);

      return body.messages[0].content;
    }

    it("carries an emitted PDF block to OpenAI as a file part", async () => {
      const aguiMessage: UserMessage = {
        id: "boundary-pdf",
        role: "user",
        content: [
          {
            type: "document",
            source: { type: "data", value: "JVBERi0xLjQK", mimeType: "application/pdf" },
            metadata: { filename: "invoice-q2.pdf" },
          } as DocumentInputContent,
        ],
      };

      const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
        (b) => b.type !== "text",
      );
      const parts = await partsOnTheWire(emitted);

      expect(parts).toEqual([
        { type: "text", text: "hi" },
        {
          type: "file",
          file: {
            file_data: "data:application/pdf;base64,JVBERi0xLjQK",
            // The real filename, NOT the `LC_AUTOGENERATED` placeholder the
            // translator substitutes when it cannot find one.
            filename: "invoice-q2.pdf",
          },
        },
      ]);
    });

    it("carries an emitted audio block to OpenAI as an input_audio part", async () => {
      const aguiMessage: UserMessage = {
        id: "boundary-audio",
        role: "user",
        content: [
          {
            type: "audio",
            source: { type: "data", value: "SGVsbG8=", mimeType: "audio/wav" },
          } as AudioInputContent,
        ],
      };

      const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
        (b) => b.type !== "text",
      );
      const parts = await partsOnTheWire(emitted);

      expect(parts[1]).toEqual({
        type: "input_audio",
        input_audio: { data: "SGVsbG8=", format: "wav" },
      });
    });

    it("carries an emitted filename-less PDF to OpenAI with the derived name", async () => {
      // The document path claims to work. It only does because the converter
      // substitutes a filename: without one this exact call throws inside
      // `@langchain/openai` before a request is ever built.
      const aguiMessage: UserMessage = {
        id: "boundary-pdf-no-filename",
        role: "user",
        content: [
          {
            type: "document",
            source: { type: "data", value: "JVBERi0xLjQK", mimeType: "application/pdf" },
          } as DocumentInputContent,
        ],
      };

      const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
        (b) => b.type !== "text",
      );
      const parts = await partsOnTheWire(emitted);

      expect(parts[1]).toEqual({
        type: "file",
        file: {
          file_data: "data:application/pdf;base64,JVBERi0xLjQK",
          filename: "attachment.pdf",
        },
      });
    });

    it("carries an emitted legacy-binary PDF to OpenAI as a file part", async () => {
      const aguiMessage: UserMessage = {
        id: "boundary-legacy-pdf",
        role: "user",
        content: [
          {
            type: "binary",
            mimeType: "application/pdf",
            data: "JVBERi0xLjQK",
            filename: "legacy-invoice.pdf",
          } as BinaryInputContent,
        ],
      };

      const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
        (b) => b.type !== "text",
      );
      const parts = await partsOnTheWire(emitted);

      expect(parts[1]).toEqual({
        type: "file",
        file: {
          file_data: "data:application/pdf;base64,JVBERi0xLjQK",
          filename: "legacy-invoice.pdf",
        },
      });
    });

    it("carries an emitted legacy-binary audio clip to OpenAI as an input_audio part", async () => {
      const aguiMessage: UserMessage = {
        id: "boundary-legacy-audio",
        role: "user",
        content: [
          {
            type: "binary",
            mimeType: "audio/wav",
            data: "SGVsbG8=",
          } as BinaryInputContent,
        ],
      };

      const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
        (b) => b.type !== "text",
      );
      const parts = await partsOnTheWire(emitted);

      expect(parts[1]).toEqual({
        type: "input_audio",
        input_audio: { data: "SGVsbG8=", format: "wav" },
      });
    });

    // ── Audio MIME types ────────────────────────────────────────────────
    //
    // `input_audio.format` is an enum of exactly two values, and both runtimes
    // derive it from the block's `mime_type`. So "audio, data converts" was never
    // true as stated — it was true of the `audio/wav` it was measured on. These
    // cases pin the real constraint at the provider boundary, per spelling.

    // Everything the converter ADMITS, with the exact part that lands on the
    // wire. `audio/mpeg` is the load-bearing row: it is the IANA type for MP3 and
    // what a browser reports for a `.mp3`, and it is NOT what the provider's enum
    // lists — so it only works because the converter rewrites the spelling.
    it.each([
      ["audio/wav", "wav"],
      ["audio/mp3", "mp3"],
      ["audio/mpeg", "mp3"],
      ["audio/x-wav", "wav"],
      ["audio/wave", "wav"],
      ["audio/vnd.wave", "wav"],
      // MIME types are case-insensitive (RFC 2045 §5.1) and may carry parameters;
      // neither makes a supported format unsupported.
      ["AUDIO/MPEG", "mp3"],
      ["audio/WAV", "wav"],
      ["audio/mpeg; charset=binary", "mp3"],
      ["audio/wav; codecs=1", "wav"],
    ])("carries %s to OpenAI as input_audio format %s", async (mimeType, format) => {
      const aguiMessage: UserMessage = {
        id: "boundary-audio-mime",
        role: "user",
        content: [
          {
            type: "audio",
            source: { type: "data", value: "SGVsbG8=", mimeType },
          } as AudioInputContent,
        ],
      };

      const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
        (b) => b.type !== "text",
      );
      const parts = await partsOnTheWire(emitted);

      expect(parts[1]).toEqual({
        type: "input_audio",
        input_audio: { data: "SGVsbG8=", format },
      });
    });

    // The legacy `binary` path routes through the SAME gate, so it admits and
    // normalizes identically. A divergence between the two paths would put the
    // same clip on the wire two different ways depending on which client sent it.
    it.each([
      ["audio/mpeg", "mp3"],
      ["audio/x-wav", "wav"],
      ["AUDIO/MPEG", "mp3"],
    ])("carries legacy-binary %s to OpenAI as input_audio format %s", async (mimeType, format) => {
      const aguiMessage: UserMessage = {
        id: "boundary-legacy-audio-mime",
        role: "user",
        content: [{ type: "binary", mimeType, data: "SGVsbG8=" } as BinaryInputContent],
      };

      const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
        (b) => b.type !== "text",
      );
      const parts = await partsOnTheWire(emitted);

      expect(parts[1]).toEqual({
        type: "input_audio",
        input_audio: { data: "SGVsbG8=", format },
      });
    });

    // Why the rewrite is not cosmetic: hand the provider the type the client
    // ACTUALLY sent for an MP3 and the run dies inside the translator, before a
    // request is built. This is the throw the normalization step exists to avoid,
    // pinned so it cannot be quietly reintroduced by widening the gate instead.
    it("would throw at the provider for a raw audio/mpeg block — hence the rewrite", async () => {
      await expect(
        partsOnTheWire([
          { type: "audio", source_type: "base64", data: "SGVsbG8=", mime_type: "audio/mpeg" },
        ]),
      ).rejects.toThrow(/must have mime type of audio\/wav or audio\/mp3/);
    });

    // Everything the converter REFUSES. Two claims per row, and both matter: the
    // block stays on the pre-existing `image_url` path (so this change regresses
    // nothing), and that path reaches the wire WITHOUT throwing — degraded but
    // alive, which is the whole premise of the narrow gate.
    it.each(["audio/ogg", "audio/aac", "audio/webm", "audio/flac", "audio/mp4"])(
      "keeps %s on the image_url path, which does not throw",
      async (mimeType) => {
        const aguiMessage: UserMessage = {
          id: "boundary-audio-unsupported",
          role: "user",
          content: [
            {
              type: "audio",
              source: { type: "data", value: "SGVsbG8=", mimeType },
            } as AudioInputContent,
          ],
        };

        const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
          (b) => b.type !== "text",
        );
        expect(emitted).toEqual([
          { type: "image_url", image_url: { url: `data:${mimeType};base64,SGVsbG8=` } },
        ]);

        const parts = await partsOnTheWire(emitted);
        expect(parts[1]).toEqual({
          type: "image_url",
          image_url: { url: `data:${mimeType};base64,SGVsbG8=` },
        });
      },
    );

    it.each(["audio/ogg", "audio/webm"])(
      "keeps legacy-binary %s on the image_url path",
      async (mimeType) => {
        const aguiMessage: UserMessage = {
          id: "boundary-legacy-audio-unsupported",
          role: "user",
          content: [{ type: "binary", mimeType, data: "SGVsbG8=" } as BinaryInputContent],
        };

        const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
          (b) => b.type !== "text",
        );
        expect(emitted).toEqual([
          { type: "image_url", image_url: { url: `data:${mimeType};base64,SGVsbG8=` } },
        ]);
      },
    );

    // The normalization is audio-only. Documents carry their MIME type inside a
    // `file_data` data URL where no enum constrains it, so rewriting one there
    // would corrupt a working path.
    it("does not rewrite a document MIME type", async () => {
      const aguiMessage: UserMessage = {
        id: "boundary-doc-mime-untouched",
        role: "user",
        content: [
          {
            type: "document",
            source: { type: "data", value: "JVBERi0xLjQK", mimeType: "application/vnd.ms-excel" },
          } as DocumentInputContent,
        ],
      };

      const emitted = (aguiMessagesToLangChain([aguiMessage])[0].content as any[]).filter(
        (b) => b.type !== "text",
      );
      const parts = await partsOnTheWire(emitted);

      expect(parts[1]).toEqual({
        type: "file",
        file: {
          file_data: "data:application/vnd.ms-excel;base64,JVBERi0xLjQK",
          filename: "attachment.vnd.ms-excel",
        },
      });
    });

    // The other half of the decision: the combinations this converter REFUSES to
    // announce as standard blocks, and the throw that is the reason why. If one
    // of these ever stops throwing, the corresponding row in
    // `standardBlockTypeFor` can be revisited — but not before.
    it.each([
      [
        "audio by url",
        { type: "audio", source_type: "url", url: "https://example.com/a.wav", mime_type: "audio/wav" },
        /must be formatted as a data URL/,
      ],
      [
        "video by base64",
        { type: "video", source_type: "base64", data: "AAA=", mime_type: "video/mp4" },
        /'video'.*not recognized/,
      ],
      [
        "video by url",
        { type: "video", source_type: "url", url: "https://example.com/v.mp4", mime_type: "video/mp4" },
        /'video'.*not recognized/,
      ],
      [
        "file by url",
        {
          type: "file",
          source_type: "url",
          url: "https://example.com/d.pdf",
          mime_type: "application/pdf",
          metadata: { filename: "d.pdf" },
        },
        /must be formatted as a data URL/,
      ],
      [
        "file by base64 with no filename",
        { type: "file", source_type: "base64", data: "JVBERi0xLjQK", mime_type: "application/pdf" },
        /a filename or name or title is needed/,
      ],
    ])("would throw at the provider for %s — hence the image_url fallback", async (_name, block, message) => {
      await expect(partsOnTheWire([block])).rejects.toThrow(message);
    });

    it("forwards a block with no source_type RAW — the regression this guards", async () => {
      // The failure mode, pinned. Without `source_type` the default path's
      // `isDataContentBlock` gate does not recognise the block as media, so it is
      // neither translated NOR rejected: it goes to the provider verbatim, and the
      // request carries a content part the API will not accept.
      //
      // This is what the previous round of this PR emitted. It passed its own
      // tests because those tests set `response_metadata.output_version = "v1"`,
      // selecting a conversion path the adapter's real output never reaches.
      const nativeJsShapeWithNoSourceType = {
        type: "file",
        data: "JVBERi0xLjQK",
        mimeType: "application/pdf",
        metadata: { filename: "invoice-q2.pdf" },
      };

      const parts = await partsOnTheWire([nativeJsShapeWithNoSourceType]);

      expect(parts[1]).toEqual(nativeJsShapeWithNoSourceType);
      expect(parts[1]).not.toHaveProperty("file");
    });
  });

  // ── Native LangChain.js blocks on the way back ───────────────────────────
  describe("langchainMessagesToAgui with native LangChain.js blocks", () => {
    it("accepts a native JS file block", () => {
      const agui = langchainMessagesToAgui([
        {
          id: "native-file",
          type: "human",
          content: [
            { type: "text", text: "Summarize this" },
            {
              type: "file",
              data: "JVBERi0xLjQK",
              mimeType: "application/pdf",
              metadata: { filename: "invoice-q2.pdf" },
            },
          ],
        } as unknown as LangGraphMessage,
      ]);

      const content = (agui[0] as UserMessage).content as Array<any>;
      expect(content[1]).toEqual({
        type: "document",
        source: { type: "data", value: "JVBERi0xLjQK", mimeType: "application/pdf" },
        metadata: { filename: "invoice-q2.pdf" },
      });
    });

    it("keeps MIME type and filename on a native JS file block by url", () => {
      const agui = langchainMessagesToAgui([
        {
          id: "native-url",
          type: "human",
          content: [
            {
              type: "file",
              url: "https://example.com/doc.pdf",
              mimeType: "application/pdf",
              metadata: { filename: "doc.pdf" },
            },
          ],
        } as unknown as LangGraphMessage,
      ]);

      const content = (agui[0] as UserMessage).content as Array<any>;
      expect(content[0]).toEqual({
        type: "document",
        source: { type: "url", value: "https://example.com/doc.pdf", mimeType: "application/pdf" },
        metadata: { filename: "doc.pdf" },
      });
    });

    it("still accepts a Python-shaped block, because the graph may be Python", () => {
      // This package drives a LangGraph server through
      // `@langchain/langgraph-sdk`, and that server is usually the Python one,
      // so a graph's own messages come back in Python's field names. Refusing
      // them would drop the attachment from MESSAGES_SNAPSHOT — the same bug,
      // mirrored onto the return leg.
      const agui = langchainMessagesToAgui([
        {
          id: "python-shaped",
          type: "human",
          content: [
            {
              type: "file",
              base64: "JVBERi0xLjQK",
              mime_type: "application/pdf",
              filename: "invoice-q2.pdf",
            },
          ],
        } as unknown as LangGraphMessage,
      ]);

      const content = (agui[0] as UserMessage).content as Array<any>;
      expect(content[0]).toEqual({
        type: "document",
        source: { type: "data", value: "JVBERi0xLjQK", mimeType: "application/pdf" },
        metadata: { filename: "invoice-q2.pdf" },
      });
    });

    it("drops a fileId-only block loudly rather than inventing a source", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const agui = langchainMessagesToAgui([
        {
          id: "file-id-only",
          type: "human",
          content: [{ type: "file", fileId: "file-abc123", mimeType: "application/pdf" }],
        } as unknown as LangGraphMessage,
      ]);

      const content = (agui[0] as UserMessage).content as Array<any>;
      expect(content).toHaveLength(0);
      expect(warn).toHaveBeenCalledWith(expect.stringContaining("Dropping file block"));
      warn.mockRestore();
    });

    it("ignores a block whose type is an Object.prototype key", () => {
      // `item.type` is not author-controlled: it rides in on content blocks the
      // LangGraph server relays from model and tool output. A bare bracket
      // lookup into an object literal answers "constructor" / "toString" with an
      // INHERITED FUNCTION, which is truthy, so the media branch would be
      // entered and that function written out as the AG-UI content type — a
      // malformed item that fails schema validation downstream. A prototype key
      // has to be exactly as unrecognized as any other unknown block type:
      // skipped, leaving only the blocks this converter actually understands.
      const agui = langchainMessagesToAgui([
        {
          id: "prototype-typed",
          type: "human",
          content: [
            { type: "text", text: "before" },
            { type: "constructor", data: "JVBERi0xLjQK", mimeType: "application/pdf" },
            { type: "toString", url: "https://example.com/doc.pdf" },
            { type: "valueOf", base64: "JVBERi0xLjQK", mime_type: "audio/mpeg" },
            { type: "hasOwnProperty", data: "JVBERi0xLjQK" },
            { type: "isPrototypeOf", data: "JVBERi0xLjQK" },
          ],
        } as unknown as LangGraphMessage,
      ]);

      const content = (agui[0] as UserMessage).content as Array<any>;
      expect(content).toEqual([{ type: "text", text: "before" }]);
    });
  });
});

describe("resolveReasoningContent - DeepSeek-style reasoning_content", () => {
  it("should return LangGraphReasoning when reasoning_content is a non-empty string", () => {
    const eventData = {
      chunk: {
        content: null,
        additional_kwargs: { reasoning_content: "thinking step by step" },
      },
    };

    const result = resolveReasoningContent(eventData);

    expect(result).not.toBeNull();
    expect(result!.type).toBe("text");
    expect(result!.text).toBe("thinking step by step");
    expect(result!.index).toBe(0);
  });

  it("should return null when reasoning_content is empty string", () => {
    const eventData = {
      chunk: {
        content: null,
        additional_kwargs: { reasoning_content: "" },
      },
    };

    expect(resolveReasoningContent(eventData)).toBeNull();
  });

  it("should return null when reasoning_content is not present", () => {
    const eventData = {
      chunk: {
        content: null,
        additional_kwargs: { some_other_key: "value" },
      },
    };

    expect(resolveReasoningContent(eventData)).toBeNull();
  });

  it("should prioritize content block formats over additional_kwargs.reasoning_content", () => {
    const eventData = {
      chunk: {
        content: [{ type: "thinking", thinking: "from content block" }],
        additional_kwargs: { reasoning_content: "from additional_kwargs" },
      },
    };

    const result = resolveReasoningContent(eventData);

    expect(result).not.toBeNull();
    expect(result!.text).toBe("from content block");
  });
});
