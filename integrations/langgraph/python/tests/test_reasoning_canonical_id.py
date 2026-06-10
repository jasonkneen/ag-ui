"""The streamed reasoning message must adopt the provider's canonical
reasoning id when the stream carries one.

Since 2111267 the snapshot converter (``_reasoning_block_to_agui_message``)
emits checkpointed reasoning under the provider's canonical block id (OpenAI
``rs_…``). If the streaming path mints a fresh ``uuid4`` instead, the client
can never reconcile the streamed copy with the snapshot copy and renders the
same reasoning twice (the langgraph-python dojo e2e strict-mode failure).

With ``use_responses_api=True``, langchain-openai surfaces the canonical id on
the ``response.reasoning_summary_part.added`` chunk (empty text, ``id`` set);
the subsequent ``response.reasoning_summary_text.delta`` chunks carry text but
no id. These tests pin that:

  * ``resolve_reasoning_content`` surfaces the part-added chunk (instead of
    dropping it for having empty text) and extracts the block id,
  * ``handle_reasoning_event`` opens the reasoning message under that id and
    does not emit an empty content delta for the id-bearing chunk,
  * everything else (store=true empty-summary items, id-less providers,
    non-first summary parts) behaves exactly as before.
"""

import unittest

from ag_ui.core import EventType

from ag_ui_langgraph.utils import resolve_reasoning_content
from tests._helpers import make_agent, _record_dispatch


class FakeChunk:
    def __init__(self, content=None, additional_kwargs=None):
        self.content = content or []
        self.additional_kwargs = additional_kwargs or {}


class TestResolveReasoningContentCanonicalId(unittest.TestCase):
    def test_summary_part_added_chunk_carries_id(self):
        """`response.reasoning_summary_part.added` shape: empty text, id set.

        Must be surfaced (not dropped) so the id can seed REASONING_START.
        """
        chunk = FakeChunk(content=[{
            "type": "reasoning",
            "id": "rs-canonical",
            "summary": [{"index": 0, "type": "summary_text", "text": ""}],
            "index": 0,
        }])
        result = resolve_reasoning_content(chunk)
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["id"], "rs-canonical")
        self.assertEqual(result["index"], 0)

    def test_summary_text_delta_chunk_has_no_id(self):
        """`response.reasoning_summary_text.delta` shape: text, no id —
        unchanged behavior, and no id key invented."""
        chunk = FakeChunk(content=[{
            "type": "reasoning",
            "summary": [{"index": 0, "type": "summary_text", "text": "Because X"}],
            "index": 0,
        }])
        result = resolve_reasoning_content(chunk)
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "Because X")
        self.assertIsNone(result.get("id"))

    def test_id_attached_when_text_and_id_both_present(self):
        chunk = FakeChunk(content=[{
            "type": "reasoning",
            "id": "rs-canonical",
            "summary": [{"index": 0, "type": "summary_text", "text": "Hi"}],
            "index": 0,
        }])
        result = resolve_reasoning_content(chunk)
        self.assertEqual(result["text"], "Hi")
        self.assertEqual(result["id"], "rs-canonical")

    def test_store_true_empty_summary_item_still_dropped(self):
        """`response.output_item.added` for a store=true reasoning item has an
        id but an empty summary list — must stay dropped (no ghost reasoning
        bubble for summary-less reasoning)."""
        chunk = FakeChunk(content=[{
            "type": "reasoning",
            "id": "rs-canonical",
            "summary": [],
            "index": 0,
        }])
        self.assertIsNone(resolve_reasoning_content(chunk))

    def test_non_first_summary_part_does_not_reuse_id(self):
        """A second summary part (summary index 1) belongs to the same
        reasoning item; reusing the canonical id there would mint two AG-UI
        messages with the same id. It must fall back to the uuid path."""
        chunk = FakeChunk(content=[{
            "type": "reasoning",
            "id": "rs-canonical",
            "summary": [{"index": 1, "type": "summary_text", "text": ""}],
            "index": 0,
        }])
        result = resolve_reasoning_content(chunk)
        self.assertIsNotNone(result)
        self.assertEqual(result["index"], 1)
        self.assertIsNone(result.get("id"))


class TestHandleReasoningEventCanonicalId(unittest.TestCase):
    def setUp(self):
        self.agent = _record_dispatch(make_agent())
        self.agent.active_run = {}

    def _events(self, reasoning_data):
        return list(self.agent.handle_reasoning_event(reasoning_data))

    def test_reasoning_start_uses_canonical_id(self):
        self._events({"type": "text", "text": "", "index": 0, "id": "rs-canonical"})
        start_events = [
            e for e in self.agent.dispatched if e.type == EventType.REASONING_START
        ]
        self.assertEqual(len(start_events), 1)
        self.assertEqual(start_events[0].message_id, "rs-canonical")

    def test_empty_text_chunk_emits_no_content_delta(self):
        self._events({"type": "text", "text": "", "index": 0, "id": "rs-canonical"})
        content_events = [
            e
            for e in self.agent.dispatched
            if e.type == EventType.REASONING_MESSAGE_CONTENT
        ]
        self.assertEqual(content_events, [])

    def test_subsequent_deltas_join_the_canonical_message(self):
        self._events({"type": "text", "text": "", "index": 0, "id": "rs-canonical"})
        self._events({"type": "text", "text": "Because X", "index": 0})
        start_events = [
            e for e in self.agent.dispatched if e.type == EventType.REASONING_START
        ]
        content_events = [
            e
            for e in self.agent.dispatched
            if e.type == EventType.REASONING_MESSAGE_CONTENT
        ]
        self.assertEqual(len(start_events), 1)
        self.assertEqual(len(content_events), 1)
        self.assertEqual(content_events[0].message_id, "rs-canonical")
        self.assertEqual(content_events[0].delta, "Because X")

    def test_uuid_fallback_when_stream_has_no_id(self):
        self._events({"type": "text", "text": "thinking…", "index": 0})
        start_events = [
            e for e in self.agent.dispatched if e.type == EventType.REASONING_START
        ]
        self.assertEqual(len(start_events), 1)
        self.assertTrue(start_events[0].message_id)
        self.assertNotEqual(start_events[0].message_id, "rs-canonical")


if __name__ == "__main__":
    unittest.main()
