"""Tests for LangGraphAgent.clone() subclass preservation."""

import unittest
from unittest.mock import MagicMock

from ag_ui_langgraph import LangGraphAgent


class SubclassAgent(LangGraphAgent):
    """Test subclass that adds custom behavior."""

    def __init__(self, *, name, graph, description=None, config=None, custom_flag=False):
        super().__init__(name=name, graph=graph, description=description, config=config)
        self.custom_flag = custom_flag

    def custom_method(self):
        return "subclass behavior"


class TestClone(unittest.TestCase):
    """Test that clone() preserves subclass identity and behavior."""

    def _make_graph(self):
        """Create a mock compiled graph for testing."""
        graph = MagicMock()
        graph.config_specs = []
        return graph

    def test_clone_returns_same_class(self):
        """clone() should return an instance of the same class, not the base."""
        agent = SubclassAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertIsInstance(cloned, SubclassAgent)

    def test_clone_base_class(self):
        """clone() on the base class should still return LangGraphAgent."""
        agent = LangGraphAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertIsInstance(cloned, LangGraphAgent)

    def test_clone_copies_fields(self):
        """clone() should copy name, graph, description, and config."""
        graph = self._make_graph()
        config = {"recursion_limit": 50}
        agent = LangGraphAgent(
            name="my-agent",
            graph=graph,
            description="A test agent",
            config=config,
        )
        cloned = agent.clone()
        self.assertEqual(cloned.name, "my-agent")
        self.assertIs(cloned.graph, graph)
        self.assertEqual(cloned.description, "A test agent")
        self.assertEqual(cloned.config, config)

    def test_clone_subclass_has_overridden_methods(self):
        """clone() of a subclass should have the subclass's methods."""
        agent = SubclassAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertEqual(cloned.custom_method(), "subclass behavior")

    def test_clone_isolates_mutable_state(self):
        """clone() should produce a separate instance (not the same object)."""
        agent = LangGraphAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertIsNot(agent, cloned)
        self.assertIsNot(agent.messages_in_process, cloned.messages_in_process)


if __name__ == "__main__":
    unittest.main()
