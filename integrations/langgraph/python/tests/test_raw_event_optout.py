"""Tests for the ``emit_raw_events`` opt-out (OSS-607).

The LangGraph integration piggy-backs the full underlying LangGraph event onto
almost every emitted AG-UI event via ``raw_event``. On graphs with large state
this inflates payloads (Function Health reported ~1.5 MB events). Constructing
the agent with ``emit_raw_events=False`` strips that piggy-backed ``raw_event``
at the single ``_dispatch_event`` choke point, while leaving the explicit
``EventType.RAW`` passthrough channel untouched.
"""
import unittest
from unittest.mock import MagicMock

from ag_ui.core import EventType, RawEvent, TextMessageEndEvent

from ag_ui_langgraph import LangGraphAgent


def _make_agent(**kwargs):
    graph = MagicMock()
    graph.nodes = {}
    return LangGraphAgent(name="test", graph=graph, **kwargs)


class TestEmitRawEventsOptOut(unittest.TestCase):
    def test_default_is_on_and_preserves_raw_event(self):
        agent = _make_agent()
        self.assertTrue(agent.emit_raw_events)
        ev = TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END, message_id="m1", raw_event={"big": "payload"}
        )
        out = agent._dispatch_event(ev)
        self.assertEqual(out.raw_event, {"big": "payload"})

    def test_opt_out_strips_piggybacked_raw_event(self):
        agent = _make_agent(emit_raw_events=False)
        self.assertFalse(agent.emit_raw_events)
        ev = TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END, message_id="m1", raw_event={"big": "payload"}
        )
        out = agent._dispatch_event(ev)
        self.assertIsNone(out.raw_event)

    def test_opt_out_leaves_explicit_raw_event_type_untouched(self):
        # EventType.RAW is an explicit, opt-in passthrough channel — not the
        # piggy-backed raw_event — so the opt-out must not empty its payload.
        agent = _make_agent(emit_raw_events=False)
        ev = RawEvent(type=EventType.RAW, event={"explicit": "raw"})
        out = agent._dispatch_event(ev)
        self.assertEqual(out.event, {"explicit": "raw"})

    def test_clone_preserves_emit_raw_events(self):
        agent = _make_agent(emit_raw_events=False)
        cloned = agent.clone()
        self.assertFalse(cloned.emit_raw_events)


if __name__ == "__main__":
    unittest.main()
