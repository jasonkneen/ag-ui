"""
Tests for multimodal message conversion between AG-UI and LangChain formats.
"""

import unittest
import warnings
from ag_ui.core import (
    UserMessage,
    TextInputContent,
    BinaryInputContent,
    ImageInputContent,
    AudioInputContent,
    VideoInputContent,
    DocumentInputContent,
    InputContentDataSource,
    InputContentUrlSource,
)
from langchain_core.messages import (
    HumanMessage,
    convert_to_openai_messages,
    is_data_content_block,
)

from ag_ui_langgraph.utils import (
    agui_messages_to_langchain,
    langchain_messages_to_agui,
    convert_agui_multimodal_to_langchain,
    convert_langchain_multimodal_to_agui,
    flatten_user_content,
)


class TestMultimodalConversion(unittest.TestCase):
    """Test multimodal message conversion between AG-UI and LangChain."""

    def test_agui_text_only_to_langchain(self):
        """Test converting a text-only AG-UI message to LangChain."""
        agui_message = UserMessage(
            id="test-1",
            role="user",
            content="Hello, world!"
        )

        lc_messages = agui_messages_to_langchain([agui_message])

        self.assertEqual(len(lc_messages), 1)
        self.assertIsInstance(lc_messages[0], HumanMessage)
        self.assertEqual(lc_messages[0].content, "Hello, world!")
        self.assertEqual(lc_messages[0].id, "test-1")

    # ── BinaryInputContent backwards compatibility ──────────────────────

    def test_agui_binary_url_to_langchain(self):
        """Test converting BinaryInputContent with URL to LangChain (backwards compat)."""
        agui_message = UserMessage(
            id="test-2",
            role="user",
            content=[
                TextInputContent(type="text", text="What's in this image?"),
                BinaryInputContent(
                    type="binary",
                    mime_type="image/jpeg",
                    url="https://example.com/photo.jpg"
                ),
            ]
        )

        lc_messages = agui_messages_to_langchain([agui_message])

        self.assertEqual(len(lc_messages), 1)
        self.assertIsInstance(lc_messages[0], HumanMessage)
        self.assertIsInstance(lc_messages[0].content, list)
        self.assertEqual(len(lc_messages[0].content), 2)

        # Check text content
        self.assertEqual(lc_messages[0].content[0]["type"], "text")
        self.assertEqual(lc_messages[0].content[0]["text"], "What's in this image?")

        # Check image content
        self.assertEqual(lc_messages[0].content[1]["type"], "image_url")
        self.assertEqual(
            lc_messages[0].content[1]["image_url"]["url"],
            "https://example.com/photo.jpg"
        )

    def test_agui_binary_data_to_langchain(self):
        """Test converting BinaryInputContent with base64 data to LangChain (backwards compat)."""
        agui_message = UserMessage(
            id="test-3",
            role="user",
            content=[
                TextInputContent(type="text", text="Analyze this"),
                BinaryInputContent(
                    type="binary",
                    mime_type="image/png",
                    data="iVBORw0KGgoAAAANSUhEUgAAAAUA",
                    filename="test.png"
                ),
            ]
        )

        lc_messages = agui_messages_to_langchain([agui_message])

        self.assertEqual(len(lc_messages), 1)
        self.assertIsInstance(lc_messages[0].content, list)
        self.assertEqual(len(lc_messages[0].content), 2)

        # Check that data URL is properly formatted
        image_content = lc_messages[0].content[1]
        self.assertEqual(image_content["type"], "image_url")
        self.assertTrue(
            image_content["image_url"]["url"].startswith("data:image/png;base64,")
        )

    # ── ImageInputContent ───────────────────────────────────────────────

    def test_agui_image_url_source_to_langchain(self):
        """Test converting ImageInputContent with URL source to LangChain."""
        agui_message = UserMessage(
            id="test-img-url",
            role="user",
            content=[
                TextInputContent(type="text", text="Describe this image"),
                ImageInputContent(
                    type="image",
                    source=InputContentUrlSource(
                        type="url",
                        value="https://example.com/photo.jpg",
                    ),
                ),
            ]
        )

        lc_messages = agui_messages_to_langchain([agui_message])

        self.assertEqual(len(lc_messages), 1)
        content = lc_messages[0].content
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "https://example.com/photo.jpg")

    def test_agui_image_data_source_to_langchain(self):
        """Test converting ImageInputContent with data source to LangChain."""
        agui_message = UserMessage(
            id="test-img-data",
            role="user",
            content=[
                ImageInputContent(
                    type="image",
                    source=InputContentDataSource(
                        type="data",
                        value="iVBORw0KGgoAAAANSUhEUgAAAAUA",
                        mime_type="image/png",
                    ),
                ),
            ]
        )

        lc_messages = agui_messages_to_langchain([agui_message])

        content = lc_messages[0].content
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(
            content[0]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
        )

    def test_agui_input_metadata_not_leaked_to_langchain_blocks(self):
        """AG-UI InputContent metadata must NOT be attached to the LangChain
        content blocks handed to the model.

        Attaching a top-level ``metadata`` key produces a non-standard content
        block (e.g. ``{"type": "image_url", "image_url": {...}, "metadata": {...}}``)
        that strict OpenAI-compatible providers reject with a 400
        ("Unexpected keys in a message content image dict"), aborting the run.
        The block must carry only spec-compliant keys. See issue #2100.
        """
        content_list = [
            TextInputContent(
                type="text",
                text="Describe this image",
                metadata={"source": "prompt"},
            ),
            ImageInputContent(
                type="image",
                source=InputContentUrlSource(
                    type="url",
                    value="https://example.com/photo.jpg",
                ),
                metadata={"provider_hint": "vision"},
            ),
            BinaryInputContent(
                type="binary",
                mime_type="image/png",
                url="https://example.com/legacy.png",
                metadata={"legacy": True},
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        # No block leaks a top-level "metadata" key into the model payload.
        for block in lc_content:
            self.assertNotIn("metadata", block)

        # Blocks remain spec-compliant and otherwise unchanged.
        self.assertEqual(lc_content[0], {"type": "text", "text": "Describe this image"})
        self.assertEqual(
            lc_content[1],
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        )
        self.assertEqual(
            lc_content[2],
            {"type": "image_url", "image_url": {"url": "https://example.com/legacy.png"}},
        )

    # ── AudioInputContent ───────────────────────────────────────────────

    def test_agui_audio_data_source_to_langchain(self):
        """Test converting AudioInputContent with data source to LangChain."""
        content_list = [
            AudioInputContent(
                type="audio",
                source=InputContentDataSource(
                    type="data",
                    value="SGVsbG8gV29ybGQ=",
                    mime_type="audio/mp3",
                ),
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        # An `audio` block, NOT `image_url`: providers validate the block kind,
        # so audio announced as an image is rejected outright. Base64 audio is
        # one of the two combinations the translator accepts — see
        # `test_emitted_audio_block_translates_for_openai`.
        self.assertEqual(len(lc_content), 1)
        self.assertEqual(
            lc_content[0],
            {
                "type": "audio",
                "base64": "SGVsbG8gV29ybGQ=",
                "mime_type": "audio/mp3",
            },
        )

    # ── Combinations deliberately LEFT on the legacy `image_url` path ────
    #
    # These are pinned decisions, not aspirations. For each one the standard
    # media block RAISES inside `convert_to_openai_data_block` (and throws in the
    # mirrored TypeScript adapter), so emitting it would turn the pre-existing
    # degraded request into a dead run. They stay on `image_url` until the
    # translators accept them — see
    # `test_refused_combinations_raise_in_the_translator`, which pins the throws.
    # Do not "finish the job" by flipping one of these without re-measuring.

    def test_agui_audio_url_source_stays_on_image_url(self):
        """Audio by URL keeps the legacy `image_url` block."""
        content_list = [
            AudioInputContent(
                type="audio",
                source=InputContentUrlSource(
                    type="url",
                    value="https://example.com/audio.mp3",
                ),
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(len(lc_content), 1)
        self.assertEqual(
            lc_content[0],
            {"type": "image_url", "image_url": {"url": "https://example.com/audio.mp3"}},
        )

    # ── VideoInputContent ───────────────────────────────────────────────

    def test_agui_video_url_source_stays_on_image_url(self):
        """Video by URL keeps the legacy `image_url` block."""
        content_list = [
            VideoInputContent(
                type="video",
                source=InputContentUrlSource(
                    type="url",
                    value="https://example.com/video.mp4",
                ),
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(len(lc_content), 1)
        self.assertEqual(
            lc_content[0],
            {"type": "image_url", "image_url": {"url": "https://example.com/video.mp4"}},
        )

    def test_agui_video_data_source_stays_on_image_url(self):
        """Video by inline data keeps the legacy `image_url` block."""
        content_list = [
            VideoInputContent(
                type="video",
                source=InputContentDataSource(
                    type="data",
                    value="AAAA",
                    mime_type="video/mp4",
                ),
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(len(lc_content), 1)
        self.assertEqual(
            lc_content[0],
            {"type": "image_url", "image_url": {"url": "data:video/mp4;base64,AAAA"}},
        )

    # ── DocumentInputContent ────────────────────────────────────────────

    def test_agui_document_url_source_stays_on_image_url(self):
        """Documents by URL keep the legacy `image_url` block.

        Fetching the bytes on the caller's behalf is not this adapter's job, so
        the URL keeps going out exactly as it always did.
        """
        content_list = [
            DocumentInputContent(
                type="document",
                source=InputContentUrlSource(
                    type="url",
                    value="https://example.com/doc.pdf",
                ),
                metadata={"filename": "doc.pdf"},
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(len(lc_content), 1)
        self.assertEqual(
            lc_content[0],
            {"type": "image_url", "image_url": {"url": "https://example.com/doc.pdf"}},
        )

    def test_agui_legacy_binary_url_stays_on_image_url(self):
        """A legacy binary by URL keeps the legacy `image_url` block too."""
        content_list = [
            BinaryInputContent(
                type="binary",
                mime_type="application/pdf",
                url="https://example.com/legacy.pdf",
                filename="legacy.pdf",
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(
            lc_content,
            [{"type": "image_url", "image_url": {"url": "https://example.com/legacy.pdf"}}],
        )

    def test_agui_document_data_source_to_langchain(self):
        """Test converting DocumentInputContent with data source to LangChain."""
        content_list = [
            DocumentInputContent(
                type="document",
                source=InputContentDataSource(
                    type="data",
                    value="JVBERi0xLjQK",
                    mime_type="application/pdf",
                ),
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        # THE REGRESSION THIS FILE EXISTS TO CATCH. A PDF handed to a provider as
        # `image_url` — with `application/pdf` sitting inside the data URL — is
        # rejected on the block kind:
        #
        #     openai.BadRequestError: 400 - Invalid MIME type. Only image types
        #     are supported. (code: invalid_image_format)
        #
        # and the exception kills the run rather than degrading it.
        #
        # The filename is DERIVED from the MIME subtype because the item carried
        # none: langchain-core only warns about that, but the mirrored TypeScript
        # adapter's translator throws, so both runtimes substitute the same name.
        self.assertEqual(len(lc_content), 1)
        self.assertEqual(
            lc_content[0],
            {
                "type": "file",
                "base64": "JVBERi0xLjQK",
                "mime_type": "application/pdf",
                "filename": "attachment.pdf",
            },
        )

    def test_agui_document_filename_reaches_the_file_block(self):
        """A document's `metadata.filename` lands on the file block.

        A file block without a filename is degraded in both runtimes —
        langchain-core warns and drops the key, `@langchain/openai` throws — and
        `metadata: {filename}` is where AG-UI puts it: the client's own
        `backward-compatibility-0-0-47` middleware migrates the legacy
        `BinaryInputContent.filename` into exactly that shape.
        """
        content_list = [
            DocumentInputContent(
                type="document",
                source=InputContentDataSource(
                    type="data",
                    value="JVBERi0xLjQK",
                    mime_type="application/pdf",
                ),
                metadata={"filename": "invoice-q2.pdf"},
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(lc_content[0]["filename"], "invoice-q2.pdf")
        # ENUMERATED, not negated. `assertNotIn("metadata", ...)` reads like a
        # guard but cannot fail: nothing in `_standard_media_block` writes a
        # `metadata` key, so it would pass for every input including a block
        # that lost its filename. Pinning the whole key set constrains the
        # output in both directions — a key gained (issue #2100's top-level
        # `metadata` object, which makes strict providers 400) and a key lost.
        # The TypeScript counterpart does the same with
        # `expect(Object.keys(content[1].metadata)).toEqual(["filename"])`.
        self.assertEqual(
            sorted(lc_content[0]), ["base64", "filename", "mime_type", "type"]
        )

    def test_document_survives_the_langchain_round_trip(self):
        """AG-UI -> LangChain -> AG-UI keeps the document a document.

        This is the MESSAGES_SNAPSHOT path. A block kind the return leg does not
        understand is an attachment that disappears from the thread on the next
        snapshot: the file was sent, the model read it, and a reopened thread
        shows a bare line of text.
        """
        original = [
            TextInputContent(type="text", text="Summarize this"),
            DocumentInputContent(
                type="document",
                source=InputContentDataSource(
                    type="data",
                    value="JVBERi0xLjQK",
                    mime_type="application/pdf",
                ),
                metadata={"filename": "invoice-q2.pdf"},
            ),
        ]

        round_tripped = convert_langchain_multimodal_to_agui(
            convert_agui_multimodal_to_langchain(original)
        )

        self.assertEqual(len(round_tripped), 2)
        self.assertIsInstance(round_tripped[1], DocumentInputContent)
        self.assertEqual(round_tripped[1].source.value, "JVBERi0xLjQK")
        self.assertEqual(round_tripped[1].source.mime_type, "application/pdf")
        self.assertEqual(round_tripped[1].metadata, {"filename": "invoice-q2.pdf"})

    def test_audio_survives_the_langchain_round_trip(self):
        """Same guard as the document round trip, for inline audio.

        Video is absent on purpose: it is not emitted as a standard block, so it
        round-trips through `image_url` and comes back as an image. That is the
        pre-existing behaviour, and it is a live run rather than a dead one.
        """
        original = [
            AudioInputContent(
                type="audio",
                source=InputContentDataSource(
                    type="data", value="SGVsbG8=", mime_type="audio/mp3"
                ),
            ),
        ]

        round_tripped = convert_langchain_multimodal_to_agui(
            convert_agui_multimodal_to_langchain(original)
        )

        self.assertIsInstance(round_tripped[0], AudioInputContent)
        self.assertEqual(round_tripped[0].source.mime_type, "audio/mp3")
        self.assertEqual(round_tripped[0].source.value, "SGVsbG8=")

    # ── LangChain to AG-UI (new types) ─────────────────────────────────

    def test_langchain_image_url_to_agui_produces_image_input_content(self):
        """Test converting LangChain image_url with regular URL to AG-UI produces ImageInputContent."""
        lc_content = [
            {"type": "text", "text": "What do you see?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            },
        ]

        agui_content = convert_langchain_multimodal_to_agui(lc_content)

        self.assertEqual(len(agui_content), 2)
        self.assertIsInstance(agui_content[0], TextInputContent)
        self.assertEqual(agui_content[0].text, "What do you see?")

        self.assertIsInstance(agui_content[1], ImageInputContent)
        self.assertIsInstance(agui_content[1].source, InputContentUrlSource)
        self.assertEqual(agui_content[1].source.value, "https://example.com/image.jpg")

    def test_langchain_data_url_to_agui_produces_image_input_content(self):
        """Test converting LangChain data URL to AG-UI produces ImageInputContent with data source."""
        lc_content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"}
            },
        ]

        agui_content = convert_langchain_multimodal_to_agui(lc_content)

        self.assertEqual(len(agui_content), 1)
        self.assertIsInstance(agui_content[0], ImageInputContent)
        self.assertIsInstance(agui_content[0].source, InputContentDataSource)
        self.assertEqual(agui_content[0].source.mime_type, "image/png")
        self.assertEqual(agui_content[0].source.value, "iVBORw0KGgo")

    def test_langchain_jpeg_data_url_to_agui(self):
        """Test converting LangChain JPEG data URL to AG-UI."""
        lc_content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}
            },
        ]

        agui_content = convert_langchain_multimodal_to_agui(lc_content)

        self.assertEqual(len(agui_content), 1)
        self.assertIsInstance(agui_content[0], ImageInputContent)
        self.assertIsInstance(agui_content[0].source, InputContentDataSource)
        self.assertEqual(agui_content[0].source.mime_type, "image/jpeg")
        self.assertEqual(agui_content[0].source.value, "/9j/4AAQ")

    # ── Round-trip tests ────────────────────────────────────────────────

    def test_round_trip_langchain_url_to_agui_and_back(self):
        """Test round-trip: LangChain image_url -> AG-UI ImageInputContent -> LangChain image_url."""
        original_lc = [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}},
        ]

        agui_content = convert_langchain_multimodal_to_agui(original_lc)
        result_lc = convert_agui_multimodal_to_langchain(agui_content)

        self.assertEqual(len(result_lc), 2)
        self.assertEqual(result_lc[0]["type"], "text")
        self.assertEqual(result_lc[0]["text"], "Look at this")
        self.assertEqual(result_lc[1]["type"], "image_url")
        self.assertEqual(result_lc[1]["image_url"]["url"], "https://example.com/pic.png")

    def test_round_trip_langchain_data_url_to_agui_and_back(self):
        """Test round-trip: LangChain data URL -> AG-UI ImageInputContent -> LangChain data URL."""
        original_lc = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]

        agui_content = convert_langchain_multimodal_to_agui(original_lc)
        result_lc = convert_agui_multimodal_to_langchain(agui_content)

        self.assertEqual(len(result_lc), 1)
        self.assertEqual(result_lc[0]["type"], "image_url")
        self.assertEqual(
            result_lc[0]["image_url"]["url"],
            "data:image/png;base64,abc123"
        )

    # ── Mixed content types ─────────────────────────────────────────────

    def test_mixed_content_types_to_langchain(self):
        """Test converting a mix of new typed content and legacy BinaryInputContent."""
        content_list = [
            TextInputContent(type="text", text="Multi-media message"),
            ImageInputContent(
                type="image",
                source=InputContentUrlSource(type="url", value="https://example.com/img.jpg"),
            ),
            AudioInputContent(
                type="audio",
                source=InputContentDataSource(type="data", value="audiodata", mime_type="audio/wav"),
            ),
            BinaryInputContent(
                type="binary",
                mime_type="image/gif",
                url="https://example.com/old.gif",
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(len(lc_content), 4)
        self.assertEqual(lc_content[0]["type"], "text")
        self.assertEqual(lc_content[0]["text"], "Multi-media message")

        self.assertEqual(lc_content[1]["type"], "image_url")
        self.assertEqual(lc_content[1]["image_url"]["url"], "https://example.com/img.jpg")

        self.assertEqual(
            lc_content[2],
            {"type": "audio", "base64": "audiodata", "mime_type": "audio/wav"},
        )

        # Legacy, and an IMAGE — so it keeps the historical `image_url` shape.
        self.assertEqual(lc_content[3]["type"], "image_url")
        self.assertEqual(lc_content[3]["image_url"]["url"], "https://example.com/old.gif")

    def test_legacy_binary_pdf_becomes_a_file_block(self):
        """A legacy binary item splits on its MIME type, like the typed ones.

        `BinaryInputContent` is deprecated but still accepted, and a deprecated
        path that 400s is not meaningfully more supported than one that raises.
        Its typed `filename` goes straight onto the block.
        """
        content_list = [
            BinaryInputContent(
                type="binary",
                mime_type="application/pdf",
                data="JVBERi0xLjQK",
                filename="legacy-invoice.pdf",
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(
            lc_content[0],
            {
                "type": "file",
                "base64": "JVBERi0xLjQK",
                "mime_type": "application/pdf",
                "filename": "legacy-invoice.pdf",
            },
        )

    def test_legacy_binary_id_only_keeps_the_reference_form(self):
        """An `id`-only legacy item carries no bytes and no URL.

        There is nothing to classify and nothing to attach, so it keeps the
        historical reference form rather than inventing a block shape.
        """
        content_list = [
            BinaryInputContent(
                type="binary",
                mime_type="application/pdf",
                id="file-abc123",
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        self.assertEqual(
            lc_content[0],
            {"type": "image_url", "image_url": {"url": "file-abc123"}},
        )

    # ── flatten_user_content ────────────────────────────────────────────

    def test_flatten_multimodal_content(self):
        """Test flattening multimodal content to plain text."""
        content = [
            TextInputContent(type="text", text="Hello"),
            BinaryInputContent(
                type="binary",
                mime_type="image/jpeg",
                url="https://example.com/image.jpg"
            ),
            TextInputContent(type="text", text="World"),
        ]

        flattened = flatten_user_content(content)

        self.assertIn("Hello", flattened)
        self.assertIn("World", flattened)
        self.assertIn("[Binary content: https://example.com/image.jpg]", flattened)

    def test_flatten_with_filename(self):
        """Test flattening binary content with filename."""
        content = [
            TextInputContent(type="text", text="Check this file"),
            BinaryInputContent(
                type="binary",
                mime_type="application/pdf",
                url="https://example.com/doc.pdf",
                filename="report.pdf"
            ),
        ]

        flattened = flatten_user_content(content)

        self.assertIn("Check this file", flattened)
        self.assertIn("[Binary content: report.pdf]", flattened)

    def test_flatten_image_input_content(self):
        """Test flattening ImageInputContent to plain text."""
        content = [
            TextInputContent(type="text", text="Here is an image"),
            ImageInputContent(
                type="image",
                source=InputContentUrlSource(type="url", value="https://example.com/img.jpg"),
            ),
        ]

        flattened = flatten_user_content(content)

        self.assertIn("Here is an image", flattened)
        self.assertIn("[Image: https://example.com/img.jpg]", flattened)

    def test_flatten_image_data_source(self):
        """Test flattening ImageInputContent with data source."""
        content = [
            ImageInputContent(
                type="image",
                source=InputContentDataSource(type="data", value="abc", mime_type="image/png"),
            ),
        ]

        flattened = flatten_user_content(content)
        self.assertIn("[Image: image/png]", flattened)

    def test_flatten_audio_input_content(self):
        """Test flattening AudioInputContent to plain text."""
        content = [
            AudioInputContent(
                type="audio",
                source=InputContentUrlSource(type="url", value="https://example.com/a.mp3"),
            ),
        ]

        flattened = flatten_user_content(content)
        self.assertIn("[Audio: https://example.com/a.mp3]", flattened)

    def test_flatten_video_input_content(self):
        """Test flattening VideoInputContent to plain text."""
        content = [
            VideoInputContent(
                type="video",
                source=InputContentUrlSource(type="url", value="https://example.com/v.mp4"),
            ),
        ]

        flattened = flatten_user_content(content)
        self.assertIn("[Video: https://example.com/v.mp4]", flattened)

    def test_flatten_document_input_content(self):
        """Test flattening DocumentInputContent to plain text."""
        content = [
            DocumentInputContent(
                type="document",
                source=InputContentUrlSource(type="url", value="https://example.com/doc.pdf"),
            ),
        ]

        flattened = flatten_user_content(content)
        self.assertIn("[Document: https://example.com/doc.pdf]", flattened)

    def test_flatten_document_data_source(self):
        """Test flattening DocumentInputContent with data source."""
        content = [
            DocumentInputContent(
                type="document",
                source=InputContentDataSource(type="data", value="pdf-data", mime_type="application/pdf"),
            ),
        ]

        flattened = flatten_user_content(content)
        self.assertIn("[Document: application/pdf]", flattened)

    def test_flatten_string_content(self):
        """Test flattening plain string content."""
        self.assertEqual(flatten_user_content("Hello"), "Hello")

    def test_flatten_none_content(self):
        """Test flattening None content."""
        self.assertEqual(flatten_user_content(None), "")

    # ── BinaryInputContent guard ─────────────────────────────────────────

    def test_binary_content_malformed_is_dropped(self):
        """Test that a BinaryInputContent with no url, data, or id is dropped.

        Uses model_construct to bypass Pydantic validation, simulating a
        malformed object that reaches the conversion function.
        """
        binary_item = BinaryInputContent.model_construct(
            type="binary",
            mime_type="image/png",
            url=None,
            data=None,
            id=None,
        )

        content_list = [
            TextInputContent(type="text", text="Keep me"),
            binary_item,
        ]

        lc_content = convert_agui_multimodal_to_langchain(content_list)

        # Only the text item should remain; the malformed binary is dropped
        self.assertEqual(len(lc_content), 1)
        self.assertEqual(lc_content[0]["type"], "text")
        self.assertEqual(lc_content[0]["text"], "Keep me")

    # ── convert helpers direct tests ────────────────────────────────────

    def test_convert_agui_multimodal_to_langchain_helper(self):
        """Test the convert_agui_multimodal_to_langchain helper with BinaryInputContent."""
        agui_content = [
            TextInputContent(type="text", text="Test text"),
            BinaryInputContent(
                type="binary",
                mime_type="image/png",
                url="https://example.com/test.png"
            ),
        ]

        lc_content = convert_agui_multimodal_to_langchain(agui_content)

        self.assertEqual(len(lc_content), 2)
        self.assertEqual(lc_content[0]["type"], "text")
        self.assertEqual(lc_content[0]["text"], "Test text")
        self.assertEqual(lc_content[1]["type"], "image_url")
        self.assertEqual(lc_content[1]["image_url"]["url"], "https://example.com/test.png")

    def test_convert_langchain_multimodal_to_agui_helper(self):
        """Test the convert_langchain_multimodal_to_agui helper function."""
        lc_content = [
            {"type": "text", "text": "Test text"},
            {"type": "image_url", "image_url": {"url": "https://example.com/test.png"}},
        ]

        agui_content = convert_langchain_multimodal_to_agui(lc_content)

        self.assertEqual(len(agui_content), 2)
        self.assertIsInstance(agui_content[0], TextInputContent)
        self.assertEqual(agui_content[0].text, "Test text")
        self.assertIsInstance(agui_content[1], ImageInputContent)
        self.assertIsInstance(agui_content[1].source, InputContentUrlSource)
        self.assertEqual(agui_content[1].source.value, "https://example.com/test.png")


class TestProviderBoundary(unittest.TestCase):
    """The EMITTED block, run down the real path to the provider payload.

    The tests above assert the SHAPE this converter emits. On their own that is
    the trap that lets a wrong shape ship: a converter and its tests agreeing on
    an invented schema look identical to a correct one. These take the emitted
    block and read what a provider would actually receive — no network, the
    conversion is a pure function.

    They go through `convert_to_openai_messages`, not
    `convert_to_openai_data_block`, ON PURPOSE. The real path gates translation
    behind `is_data_content_block` and only calls the translator for blocks that
    pass; a block that fails the gate is FORWARDED VERBATIM instead. Calling the
    translator directly skips the gate, so a shape the real path would never
    translate can still look translated in a test — see
    `test_js_native_spelling_is_forwarded_unrecognized_not_rejected` for the
    measurement that makes that concrete.
    """

    @staticmethod
    def _emit(item):
        """The block this converter actually puts on the wire for `item`."""
        blocks = convert_agui_multimodal_to_langchain([item])
        return blocks[0]

    @staticmethod
    def _provider_payload(block):
        """What an OpenAI-compatible provider actually receives for `block`.

        A block the gate rejects comes back out of here unchanged, so an
        equality assertion against the translated form catches that too.
        """
        [message] = convert_to_openai_messages([HumanMessage(content=[dict(block)])])
        return message["content"][0]

    def test_emitted_document_block_translates_for_openai(self):
        block = self._emit(
            DocumentInputContent(
                type="document",
                source=InputContentDataSource(
                    type="data", value="JVBERi0xLjQK", mime_type="application/pdf"
                ),
                metadata={"filename": "invoice-q2.pdf"},
            )
        )

        self.assertTrue(is_data_content_block(block))
        self.assertEqual(
            self._provider_payload(block),
            {
                "type": "file",
                "file": {
                    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                    "filename": "invoice-q2.pdf",
                },
            },
        )

    def test_emitted_filename_less_document_carries_the_derived_name(self):
        """The derived filename is why this path can claim to work.

        langchain-core only warns when a file block has no filename, but the
        mirrored TypeScript adapter's translator THROWS. Both runtimes emit the
        same block, so both substitute the same name.
        """
        block = self._emit(
            DocumentInputContent(
                type="document",
                source=InputContentDataSource(
                    type="data", value="JVBERi0xLjQK", mime_type="application/pdf"
                ),
            )
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            translated = self._provider_payload(block)

        self.assertEqual(
            translated,
            {
                "type": "file",
                "file": {
                    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                    "filename": "attachment.pdf",
                },
            },
        )
        # And langchain-core stays silent, because the converter supplied the
        # name. Measured with langchain-core 1.2.13: a file block with NO
        # filename warns ("OpenAI may require a filename for file uploads") and
        # substitutes the literal `LC_AUTOGENERATED` — so this list is empty
        # only while the derivation keeps working.
        self.assertEqual([str(w.message) for w in caught], [])

    def test_emitted_audio_block_translates_for_openai(self):
        block = self._emit(
            AudioInputContent(
                type="audio",
                source=InputContentDataSource(
                    type="data", value="SGVsbG8=", mime_type="audio/wav"
                ),
            )
        )

        self.assertTrue(is_data_content_block(block))
        self.assertEqual(
            self._provider_payload(block),
            {"type": "input_audio", "input_audio": {"data": "SGVsbG8=", "format": "wav"}},
        )

    def test_emitted_legacy_binary_document_translates_for_openai(self):
        block = self._emit(
            BinaryInputContent(
                type="binary",
                mime_type="application/pdf",
                data="JVBERi0xLjQK",
                filename="legacy-invoice.pdf",
            )
        )

        self.assertTrue(is_data_content_block(block))
        self.assertEqual(
            self._provider_payload(block),
            {
                "type": "file",
                "file": {
                    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                    "filename": "legacy-invoice.pdf",
                },
            },
        )

    def test_emitted_legacy_binary_audio_translates_for_openai(self):
        block = self._emit(
            BinaryInputContent(
                type="binary", mime_type="audio/wav", data="SGVsbG8="
            )
        )

        self.assertTrue(is_data_content_block(block))
        self.assertEqual(
            self._provider_payload(block),
            {"type": "input_audio", "input_audio": {"data": "SGVsbG8=", "format": "wav"}},
        )

    def test_refused_combinations_stay_off_the_standard_block_path(self):
        """The other half of the decision, pinned to BOTH its halves.

        Each row is a combination this converter refuses to announce as a
        standard block, paired with the exception that is the reason why. Two
        assertions, because either one alone is weak:

        1. the converter really does keep that combination on `image_url` — this
           is what a well-meaning "finish the job" edit to `_STANDARD_BLOCK_TYPES`
           breaks, and pinning only the library behaviour would not notice;
        2. the standard block for it really does die, measured on the REAL path
           (`convert_to_openai_messages`) rather than by calling the translator
           past the `is_data_content_block` gate. All four of these DO pass that
           gate, so the gate is not what saves them — the raise is.

        If one of these ever stops raising, the corresponding row in
        `_STANDARD_BLOCK_TYPES` can be revisited. Not before.
        """
        refused = {
            "audio by url": (
                AudioInputContent(
                    type="audio",
                    source=InputContentUrlSource(
                        type="url", value="https://example.com/a.wav"
                    ),
                ),
                {"type": "audio", "url": "https://example.com/a.wav", "mime_type": "audio/wav"},
                "Key base64 is required for audio blocks",
            ),
            "video by base64": (
                VideoInputContent(
                    type="video",
                    source=InputContentDataSource(
                        type="data", value="AAA=", mime_type="video/mp4"
                    ),
                ),
                {"type": "video", "base64": "AAA=", "mime_type": "video/mp4"},
                "Block of type video is not supported",
            ),
            "video by url": (
                VideoInputContent(
                    type="video",
                    source=InputContentUrlSource(
                        type="url", value="https://example.com/v.mp4"
                    ),
                ),
                {"type": "video", "url": "https://example.com/v.mp4", "mime_type": "video/mp4"},
                "Block of type video is not supported",
            ),
            "file by url": (
                DocumentInputContent(
                    type="document",
                    source=InputContentUrlSource(
                        type="url", value="https://example.com/d.pdf"
                    ),
                    metadata={"filename": "d.pdf"},
                ),
                {
                    "type": "file",
                    "url": "https://example.com/d.pdf",
                    "mime_type": "application/pdf",
                    "filename": "d.pdf",
                },
                "does not support file URLs",
            ),
        }

        for name, (agui_item, standard_block, message) in refused.items():
            with self.subTest(name):
                self.assertEqual(self._emit(agui_item)["type"], "image_url")

                self.assertTrue(is_data_content_block(standard_block))
                with self.assertRaisesRegex(ValueError, message):
                    self._provider_payload(standard_block)


def _as_typescript_wire_block(block):
    """Re-spell a block THIS package emits into the one the TypeScript adapter
    puts on the wire for the same AG-UI item.

    Both adapters emit the same block; they differ only in field NAMES. Python
    uses langchain-core's spelling (`base64`, top-level `filename`); the
    TypeScript half uses the `source_type` family (`source_type` + `data`,
    `metadata.filename`) — see `standardMediaBlock` in
    `integrations/langgraph/typescript/src/utils.ts`.

    Only the names are rewritten here. Every VALUE comes from this package's own
    converter, so the cross-runtime tests below are driven by what Python
    actually emits today rather than by a literal someone typed once and that
    silently stops matching. The remaining hand-maintained part is this mapping
    itself, which is four lines and sits next to the file it mirrors.
    """
    wire = {
        "type": block["type"],
        "source_type": "base64",
        "data": block["base64"],
        "mime_type": block["mime_type"],
    }
    if "filename" in block:
        wire["metadata"] = {"filename": block["filename"]}
    return wire


class TestCrossRuntimeWireShape(unittest.TestCase):
    """One AG-UI item, out through PYTHON and back in through the TYPESCRIPT
    wire shape, using THIS package's converters on both legs.

    These two adapters implement one protocol and can front the same LangGraph
    server, so the block one emits has to be both translatable for the provider
    AND readable by the other one's return leg. Nothing else in either test
    suite covers that seam, and it is where this PR's first two rounds went
    wrong: each half was verified against its own runtime only.

    Asserting a hand-copied TypeScript literal against `langchain_core` alone
    would not cover it — that exercises the library, not this package. The round
    trips below call `convert_agui_multimodal_to_langchain` on the way out and
    `convert_langchain_multimodal_to_agui` on the way back, with
    `_as_typescript_wire_block` standing in for the sibling runtime in between.
    """

    def test_document_survives_the_cross_runtime_round_trip(self):
        """AG-UI -> Python outbound -> TS wire shape -> Python return leg.

        The inbound half is the shape the sibling runtime actually produces
        (`source_type` + `data` + `metadata.filename`), which Python's return leg
        used to drop on the floor — an attachment that vanishes from a reopened
        thread.
        """
        original = DocumentInputContent(
            type="document",
            source=InputContentDataSource(
                type="data", value="JVBERi0xLjQK", mime_type="application/pdf"
            ),
            metadata={"filename": "invoice-q2.pdf"},
        )

        [emitted] = convert_agui_multimodal_to_langchain([original])
        wire = _as_typescript_wire_block(emitted)

        # The sibling runtime's spelling still reaches the provider correctly …
        self.assertTrue(is_data_content_block(wire))
        [message] = convert_to_openai_messages([HumanMessage(content=[dict(wire)])])
        self.assertEqual(
            message["content"],
            [
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                        "filename": "invoice-q2.pdf",
                    },
                }
            ],
        )

        # … and this package's return leg rebuilds the item it started as.
        [returned] = convert_langchain_multimodal_to_agui([wire])
        self.assertIsInstance(returned, DocumentInputContent)
        self.assertIsInstance(returned.source, InputContentDataSource)
        self.assertEqual(returned.source.value, original.source.value)
        self.assertEqual(returned.source.mime_type, original.source.mime_type)
        self.assertEqual(returned.metadata, {"filename": "invoice-q2.pdf"})

    def test_audio_survives_the_cross_runtime_round_trip(self):
        """Same seam, for inline audio — which translates to `input_audio`
        rather than `file`, so it exercises a different translator branch."""
        original = AudioInputContent(
            type="audio",
            source=InputContentDataSource(
                type="data", value="SGVsbG8=", mime_type="audio/wav"
            ),
            metadata={"filename": "clip.wav"},
        )

        [emitted] = convert_agui_multimodal_to_langchain([original])
        wire = _as_typescript_wire_block(emitted)

        self.assertTrue(is_data_content_block(wire))
        [message] = convert_to_openai_messages([HumanMessage(content=[dict(wire)])])
        self.assertEqual(
            message["content"],
            [{"type": "input_audio", "input_audio": {"data": "SGVsbG8=", "format": "wav"}}],
        )

        [returned] = convert_langchain_multimodal_to_agui([wire])
        self.assertIsInstance(returned, AudioInputContent)
        self.assertIsInstance(returned.source, InputContentDataSource)
        self.assertEqual(returned.source.value, original.source.value)
        self.assertEqual(returned.source.mime_type, original.source.mime_type)
        self.assertEqual(returned.metadata, {"filename": "clip.wav"})

    # ── The return leg reads all three vocabularies ────────────────────
    #
    # `convert_langchain_multimodal_to_agui` builds the user message inside
    # MESSAGES_SNAPSHOT. A vocabulary it cannot read is an attachment that
    # vanishes from a reopened thread — the file was sent, the model read it,
    # and the thread shows a bare line of text. It used to read only shape 2
    # (LangChain Python), so every base64 media block the TypeScript adapter
    # sends was dropped outright.

    def test_js_native_base64_block_is_read(self):
        """Shape 1, base64: native LangChain.js — `data` + `mimeType`."""
        agui_content = convert_langchain_multimodal_to_agui([
            {
                "type": "file",
                "data": "JVBERi0xLjQK",
                "mimeType": "application/pdf",
                "metadata": {"filename": "invoice-q2.pdf"},
            },
        ])

        self.assertEqual(len(agui_content), 1)
        self.assertIsInstance(agui_content[0], DocumentInputContent)
        self.assertIsInstance(agui_content[0].source, InputContentDataSource)
        self.assertEqual(agui_content[0].source.value, "JVBERi0xLjQK")
        self.assertEqual(agui_content[0].source.mime_type, "application/pdf")
        self.assertEqual(agui_content[0].metadata, {"filename": "invoice-q2.pdf"})

    def test_js_native_url_block_is_read(self):
        """Shape 1, url: native LangChain.js — `url` + `mimeType`."""
        agui_content = convert_langchain_multimodal_to_agui([
            {
                "type": "audio",
                "url": "https://example.com/clip.wav",
                "mimeType": "audio/wav",
                "metadata": {"filename": "clip.wav"},
            },
        ])

        self.assertEqual(len(agui_content), 1)
        self.assertIsInstance(agui_content[0], AudioInputContent)
        self.assertIsInstance(agui_content[0].source, InputContentUrlSource)
        self.assertEqual(agui_content[0].source.value, "https://example.com/clip.wav")
        self.assertEqual(agui_content[0].source.mime_type, "audio/wav")
        self.assertEqual(agui_content[0].metadata, {"filename": "clip.wav"})

    def test_python_native_base64_block_is_read(self):
        """Shape 2, base64: LangChain Python — `base64` + top-level `filename`."""
        agui_content = convert_langchain_multimodal_to_agui([
            {
                "type": "file",
                "base64": "JVBERi0xLjQK",
                "mime_type": "application/pdf",
                "filename": "invoice-q2.pdf",
            },
        ])

        self.assertEqual(len(agui_content), 1)
        self.assertIsInstance(agui_content[0], DocumentInputContent)
        self.assertIsInstance(agui_content[0].source, InputContentDataSource)
        self.assertEqual(agui_content[0].source.value, "JVBERi0xLjQK")
        self.assertEqual(agui_content[0].source.mime_type, "application/pdf")
        self.assertEqual(agui_content[0].metadata, {"filename": "invoice-q2.pdf"})

    def test_python_native_url_block_is_read(self):
        """Shape 2, url: LangChain Python — `url` + `mime_type`."""
        agui_content = convert_langchain_multimodal_to_agui([
            {
                "type": "video",
                "url": "https://example.com/demo.mp4",
                "mime_type": "video/mp4",
                "filename": "demo.mp4",
            },
        ])

        self.assertEqual(len(agui_content), 1)
        self.assertIsInstance(agui_content[0], VideoInputContent)
        self.assertIsInstance(agui_content[0].source, InputContentUrlSource)
        self.assertEqual(agui_content[0].source.value, "https://example.com/demo.mp4")
        self.assertEqual(agui_content[0].source.mime_type, "video/mp4")
        self.assertEqual(agui_content[0].metadata, {"filename": "demo.mp4"})

    # Shape 3, base64 (the `source_type` family the TypeScript adapter emits) is
    # covered by `test_document_survives_the_cross_runtime_round_trip` above,
    # which builds it from this package's own outbound leg instead of a literal.

    def test_ts_emitted_url_block_is_read_back_into_agui(self):
        """Shape 3, url: the `source_type` family, URL variant."""
        agui_content = convert_langchain_multimodal_to_agui([
            {
                "type": "file",
                "source_type": "url",
                "url": "https://example.com/invoice-q2.pdf",
                "mime_type": "application/pdf",
                "metadata": {"filename": "invoice-q2.pdf"},
            },
        ])

        self.assertEqual(len(agui_content), 1)
        self.assertIsInstance(agui_content[0], DocumentInputContent)
        self.assertIsInstance(agui_content[0].source, InputContentUrlSource)
        self.assertEqual(
            agui_content[0].source.value, "https://example.com/invoice-q2.pdf"
        )
        self.assertEqual(agui_content[0].source.mime_type, "application/pdf")
        self.assertEqual(agui_content[0].metadata, {"filename": "invoice-q2.pdf"})

    def test_filename_falls_back_to_metadata_name_and_title(self):
        """`metadata.name` / `metadata.title` are the other spellings the
        provider translators read, so the return leg must not lose them."""
        by_name, by_title = convert_langchain_multimodal_to_agui([
            {
                "type": "file",
                "source_type": "base64",
                "data": "JVBERi0xLjQK",
                "mime_type": "application/pdf",
                "metadata": {"name": "named.pdf"},
            },
            {
                "type": "file",
                "source_type": "base64",
                "data": "JVBERi0xLjQK",
                "mime_type": "application/pdf",
                "metadata": {"title": "titled.pdf"},
            },
        ])

        self.assertEqual(by_name.metadata, {"filename": "named.pdf"})
        self.assertEqual(by_title.metadata, {"filename": "titled.pdf"})

    def test_base64_block_without_mime_type_is_not_dropped(self):
        """AG-UI's data source REQUIRES a MIME type; a malformed block degrades
        to the least wrong one rather than losing the attachment."""
        agui_content = convert_langchain_multimodal_to_agui([
            {"type": "file", "source_type": "base64", "data": "JVBERi0xLjQK"},
        ])

        self.assertEqual(len(agui_content), 1)
        self.assertEqual(agui_content[0].source.mime_type, "application/octet-stream")
        self.assertIsNone(agui_content[0].metadata)

    def test_reference_only_block_is_still_dropped(self):
        """A block that names provider-side storage carries no bytes and no URL,
        and AG-UI's typed classes have nowhere to put that."""
        agui_content = convert_langchain_multimodal_to_agui([
            {"type": "file", "source_type": "id", "id": "file-abc123"},
            {"type": "file", "fileId": "file-abc123"},
        ])

        self.assertEqual(agui_content, [])

    def test_js_native_spelling_is_forwarded_unrecognized_not_rejected(self):
        """Why neither adapter emits LangChain.js's native field names.

        An earlier version of this test called `convert_to_openai_data_block`
        directly, saw a `ValueError`, and concluded Python REJECTS that shape.
        That verdict came from calling PAST the gate the real path uses, and it
        is wrong for three of the four modalities. Measured with langchain-core
        1.2.13 through `convert_to_openai_messages`, on
        `{type, data, mimeType, metadata.filename}`:

            file   is_data_content_block -> False, forwarded VERBATIM
            audio  is_data_content_block -> False, forwarded VERBATIM
            video  is_data_content_block -> False, forwarded VERBATIM
            image  is_data_content_block -> False, raises ("Unrecognized
                   content block … does not have a 'source' or 'image' key")

        Silently forwarded is not better than rejected, it is worse: the run
        does not die here with a stack trace naming the block, it dies at the
        provider on a content block nobody in this repo emitted deliberately.
        Which is the actual reason the spelling matters — not that Python
        refuses it, but that Python does not RECOGNIZE it.
        """
        [emitted] = convert_agui_multimodal_to_langchain([
            DocumentInputContent(
                type="document",
                source=InputContentDataSource(
                    type="data", value="JVBERi0xLjQK", mime_type="application/pdf"
                ),
                metadata={"filename": "invoice-q2.pdf"},
            )
        ])

        # The spelling this package emits is recognized, and translated.
        self.assertTrue(is_data_content_block(emitted))
        [message] = convert_to_openai_messages([HumanMessage(content=[dict(emitted)])])
        self.assertEqual(
            message["content"],
            [
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                        "filename": "invoice-q2.pdf",
                    },
                }
            ],
        )

        # The SAME payload under LangChain.js's native names is not recognized,
        # and goes to the provider untouched. Built from `emitted` so it tracks
        # whatever this package emits rather than a frozen literal.
        js_native = {
            "type": emitted["type"],
            "data": emitted["base64"],
            "mimeType": emitted["mime_type"],
            "metadata": {"filename": emitted["filename"]},
        }
        self.assertFalse(is_data_content_block(js_native))
        [message] = convert_to_openai_messages([HumanMessage(content=[dict(js_native)])])
        self.assertEqual(message["content"], [js_native])


if __name__ == "__main__":
    unittest.main()
