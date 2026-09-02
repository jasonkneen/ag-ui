from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[3]
DOJO_REASONING_PAGE = (
    REPO_ROOT
    / "apps"
    / "dojo"
    / "src"
    / "app"
    / "[integrationId]"
    / "feature"
    / "(v2)"
    / "agentic_chat_reasoning"
    / "page.tsx"
)
DOJO_FILES = REPO_ROOT / "apps" / "dojo" / "src" / "files.json"
AGNO_REASONING_AGENT = PROJECT_ROOT / "server" / "api" / "agentic_chat_reasoning.py"
AGNO_PYPROJECT = PROJECT_ROOT / "pyproject.toml"
AGNO_REASONING_KEY = "agno::agentic_chat_reasoning"


def _source_page() -> str:
    return DOJO_REASONING_PAGE.read_text()


def _generated_page() -> str:
    files = json.loads(DOJO_FILES.read_text())
    entries = files[AGNO_REASONING_KEY]
    for entry in entries:
        if entry["name"] == "page.tsx":
            return entry["content"]
    raise AssertionError(f"{AGNO_REASONING_KEY} is missing page.tsx")


def _uses_o4_mini_reasoning_model(agent_source: str) -> bool:
    for node in ast.walk(ast.parse(agent_source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Agent"
        ):
            continue

        for keyword in node.keywords:
            if keyword.arg != "reasoning_model" or not isinstance(
                keyword.value, ast.Call
            ):
                continue

            reasoning_model = keyword.value
            if not (
                isinstance(reasoning_model.func, ast.Name)
                and reasoning_model.func.id == "OpenAIResponses"
            ):
                return False

            return any(
                model_keyword.arg == "id"
                and isinstance(model_keyword.value, ast.Constant)
                and model_keyword.value.value == "o4-mini"
                for model_keyword in reasoning_model.keywords
            )

    return False


class AgnoReasoningProviderContractTests(unittest.TestCase):
    def test_agno_reasoning_page_does_not_offer_unsupported_provider_choices(
        self,
    ) -> None:
        expected_gate = (
            'const supportsReasoningModelSelection = integrationId !== "agno";'
        )
        expected_fixed_label = "OpenAI o4-mini"

        for label, content in {
            "source": _source_page(),
            "generated": _generated_page(),
        }.items():
            with self.subTest(label=label):
                self.assertIn(expected_gate, content)
                self.assertIn(expected_fixed_label, content)
                self.assertIn("{supportsReasoningModelSelection ? (", content)
                self.assertIn('handleModelChange("Anthropic")', content)
                self.assertIn('handleModelChange("Gemini")', content)

        self.assertEqual(_generated_page(), _source_page())

    def test_agno_reasoning_backend_stays_on_openai_responses(self) -> None:
        agent_source = AGNO_REASONING_AGENT.read_text()
        pyproject = AGNO_PYPROJECT.read_text()

        self.assertIn("from agno.db.in_memory import InMemoryDb", agent_source)
        self.assertIn("from agno.models.openai import OpenAIResponses", agent_source)
        self.assertIn("db=InMemoryDb()", agent_source)
        self.assertIn('model=OpenAIResponses(id="o4-mini")', agent_source)
        self.assertTrue(_uses_o4_mini_reasoning_model(agent_source))
        self.assertIn('reasoning_summary="auto"', agent_source)
        self.assertNotIn("agno.models.anthropic", agent_source)
        self.assertNotIn("agno.models.anthropic", pyproject)

    def test_o4_mini_must_be_configured_on_the_reasoning_model(self) -> None:
        regressed_source = """
agent = Agent(
    model=OpenAIResponses(id="o4-mini"),
    reasoning_model=OpenAIResponses(
        id="gpt-4.1",
        reasoning_summary="auto",
    ),
)
"""

        self.assertFalse(_uses_o4_mini_reasoning_model(regressed_source))


if __name__ == "__main__":
    unittest.main()
