from __future__ import annotations

import json
import unittest
from pathlib import Path

from server.api.shared_state import agent as shared_state_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[3]
DOJO_SHARED_STATE_PAGE = (
    REPO_ROOT
    / "apps"
    / "dojo"
    / "src"
    / "app"
    / "[integrationId]"
    / "feature"
    / "(v2)"
    / "shared_state"
    / "page.tsx"
)
DOJO_FILES = REPO_ROOT / "apps" / "dojo" / "src" / "files.json"
AGNO_SHARED_STATE_KEY = "agno::shared_state"


def _source_page() -> str:
    return DOJO_SHARED_STATE_PAGE.read_text()


def _generated_page() -> str:
    files = json.loads(DOJO_FILES.read_text())
    entries = files[AGNO_SHARED_STATE_KEY]
    for entry in entries:
        if entry["name"] == "page.tsx":
            return entry["content"]
    raise AssertionError(f"{AGNO_SHARED_STATE_KEY} is missing page.tsx")


class DojoSharedStateContractTests(unittest.TestCase):
    def test_agno_shared_state_does_not_send_recipe_defaults_on_hydration(
        self,
    ) -> None:
        self.assertEqual(shared_state_agent.session_state, {})
        self.assertTrue(shared_state_agent.add_session_state_to_context)
        self.assertTrue(shared_state_agent.enable_agentic_state)

    def test_agno_shared_state_requires_one_complete_nested_recipe_update(
        self,
    ) -> None:
        instructions = shared_state_agent.instructions

        self.assertIsInstance(instructions, str)
        self.assertIn(
            'session_state_updates must have exactly one top-level key: "recipe"',
            instructions,
        )
        self.assertIn("Never send recipe fields at the top level", instructions)
        self.assertIn("Never send a partial recipe object", instructions)
        for field in (
            "title",
            "skill_level",
            "cooking_time",
            "special_preferences",
            "ingredients",
            "instructions",
        ):
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', instructions)

    def test_recipe_card_form_prevents_native_submit(self) -> None:
        expected_form_tag = (
            "<form\n"
            '      data-testid="recipe-card"\n'
            "      onSubmit={(event) => event.preventDefault()}"
        )

        for label, content in {
            "source": _source_page(),
            "generated": _generated_page(),
        }.items():
            with self.subTest(label=label):
                self.assertIn(expected_form_tag, content)

        self.assertEqual(_generated_page(), _source_page())

    def test_agno_generated_shared_state_preserves_zero_cooking_time_value(
        self,
    ) -> None:
        expected_fragments = (
            "value={",
            "cookingTimeValues.find((t) => t.label === recipe.cooking_time)",
            "?.value ?? 3",
        )
        legacy_expression = "?.value || 3"

        for label, content in {
            "source": _source_page(),
            "generated": _generated_page(),
        }.items():
            with self.subTest(label=label):
                for fragment in expected_fragments:
                    self.assertIn(fragment, content)
                self.assertNotIn(legacy_expression, content)

        self.assertEqual(_generated_page(), _source_page())


if __name__ == "__main__":
    unittest.main()
