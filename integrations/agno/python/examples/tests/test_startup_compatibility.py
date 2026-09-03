from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from agno.agent import Agent
from agno.os import AgentOS
from agno.os.interfaces.agui import AGUI
from agno.team import Team
from starlette.routing import Mount

from server import api as api_package
from server import app as dojo_app
from server.api.agentic_chat_reasoning import agent as reasoning_agent
from server.api.agentic_generative_ui import agent as agentic_generative_ui_agent
from server.api.predictive_state_updates import agent as predictive_state_updates_agent
from server.api.shared_state import agent as shared_state_agent

API_ROOT = Path(api_package.__file__).resolve().parent

# Demos that deliberately run without a db. A new stateless demo must be added
# here on purpose; every other demo must persist runs so Agno can resume them.
STATELESS_DEMOS = frozenset({"backend_tool_rendering"})

EXPECTED_DOJO_MOUNTS = {
    "/agentic_chat",
    "/agentic_chat_reasoning",
    "/agentic_chat_multimodal",
    "/agentic_generative_ui",
    "/backend_tool_rendering",
    "/human_in_the_loop",
    "/predictive_state_updates",
    "/shared_state",
    "/tool_based_generative_ui",
}


def demo_names_on_disk() -> set[str]:
    return {path.stem for path in API_ROOT.glob("*.py") if path.name != "__init__.py"}


def demo_names_exported_by_package() -> set[str]:
    return {name.removesuffix("_app") for name in api_package.__all__}


def demo_agents() -> dict[str, Agent]:
    return {
        name: importlib.import_module(f"server.api.{name}").agent
        for name in sorted(demo_names_on_disk())
    }


class StartupCompatibilityTests(unittest.TestCase):
    def test_dojo_imports_and_mounts_every_existing_demo(self) -> None:
        mounts = {route.path for route in dojo_app.routes if isinstance(route, Mount)}

        self.assertEqual(mounts, EXPECTED_DOJO_MOUNTS)

    def test_expected_mounts_match_the_api_package(self) -> None:
        self.assertTrue(demo_names_on_disk(), f"no demo modules found under {API_ROOT}")

        self.assertEqual(
            EXPECTED_DOJO_MOUNTS, {f"/{name}" for name in demo_names_on_disk()}
        )
        self.assertEqual(
            EXPECTED_DOJO_MOUNTS,
            {f"/{name}" for name in demo_names_exported_by_package()},
        )

    def test_public_agent_and_team_interfaces_build_an_agent_os_app(self) -> None:
        member = Agent(id="startup-agent")
        team = Team(id="startup-team", members=[member])
        agent_os = AgentOS(
            agents=[member],
            teams=[team],
            interfaces=[
                AGUI(agent=member),
                AGUI(team=team, prefix="/team"),
            ],
        )

        paths = agent_os.get_app().openapi()["paths"]

        self.assertIn("post", paths["/agui"])
        self.assertIn("get", paths["/status"])
        self.assertIn("post", paths["/team/agui"])
        self.assertIn("get", paths["/team/status"])

    def test_reasoning_demo_uses_agno_reasoning_event_pipeline(self) -> None:
        self.assertIsNotNone(reasoning_agent.reasoning_model)

    def test_every_demo_persists_runs_for_agno_resume_unless_declared_stateless(
        self,
    ) -> None:
        agents = demo_agents()
        self.assertTrue(
            STATELESS_DEMOS <= agents.keys(), STATELESS_DEMOS - agents.keys()
        )

        for name, agent in agents.items():
            if name in STATELESS_DEMOS:
                continue
            with self.subTest(agent=name):
                self.assertIsNotNone(agent.db)

    def test_stateless_demos_is_exactly_the_set_of_demos_without_a_db(self) -> None:
        without_db = {name for name, agent in demo_agents().items() if agent.db is None}

        self.assertEqual(without_db, STATELESS_DEMOS)

    def test_stateful_demos_persist_session_state_in_a_db(self) -> None:
        stateful_agents = (
            ("agentic_generative_ui", agentic_generative_ui_agent),
            ("predictive_state_updates", predictive_state_updates_agent),
            ("shared_state", shared_state_agent),
        )

        for name, agent in stateful_agents:
            with self.subTest(agent=name):
                self.assertTrue(agent.enable_agentic_state)
                self.assertIsNotNone(agent.db)

    def test_agentic_generative_ui_streams_steps_from_agno_session_state(self) -> None:
        self.assertEqual(agentic_generative_ui_agent.session_state, {"steps": []})
        self.assertTrue(agentic_generative_ui_agent.add_session_state_to_context)
        self.assertTrue(agentic_generative_ui_agent.enable_agentic_state)


if __name__ == "__main__":
    unittest.main()
