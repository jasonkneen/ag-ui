from __future__ import annotations

import json
import unittest
from pathlib import Path

from ag_ui.core import Tool
from agno.os.interfaces.agui.input import parse_client_tools

from server.api import human_in_the_loop, tool_based_generative_ui

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[3]
DOJO_FILES = REPO_ROOT / "apps" / "dojo" / "src" / "files.json"
HITL_PAGE = (
    REPO_ROOT
    / "apps"
    / "dojo"
    / "src"
    / "app"
    / "[integrationId]"
    / "feature"
    / "(v2)"
    / "human_in_the_loop"
    / "page.tsx"
)


def _generated_hitl_page() -> str:
    catalog = json.loads(DOJO_FILES.read_text())
    return next(
        entry["content"]
        for entry in catalog["agno::human_in_the_loop"]
        if entry["name"] == "page.tsx"
    )


def _hitl_tool() -> Tool:
    return Tool(
        name="generate_task_steps",
        description="Generates a list of steps for the user to perform",
        parameters={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["enabled", "disabled", "executing"],
                            },
                        },
                        "required": ["description", "status"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        },
    )


def _haiku_tool() -> Tool:
    return Tool(
        name="generate_haiku",
        description="Generate a haiku",
        parameters={
            "type": "object",
            "properties": {
                "english": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "japanese": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "image_name": {"type": "string", "enum": ["supported.jpg"]},
                "gradient": {
                    "type": "string",
                    "pattern": (
                        r"^(?!.*url\s*\()(?:linear|radial|conic)-gradient\(.+\)$"
                    ),
                },
            },
            "required": ["english", "japanese", "image_name", "gradient"],
            "additionalProperties": False,
        },
    )


class ClientToolDedupContractTests(unittest.TestCase):
    def test_hitl_uses_one_authoritative_silent_client_tool(self) -> None:
        self.assertFalse(hasattr(human_in_the_loop, "generate_task_steps"))
        self.assertNotIn(
            "generate_task_steps",
            [tool.name for tool in human_in_the_loop.agent.tools or []],
        )

        parsed = parse_client_tools([_hitl_tool()])

        self.assertEqual([tool.name for tool in parsed], ["generate_task_steps"])
        self.assertTrue(parsed[0].external_execution)
        self.assertTrue(parsed[0].external_execution_silent)
        step_schema = parsed[0].parameters["properties"]["steps"]["items"]
        self.assertEqual(step_schema["required"], ["description", "status"])
        self.assertIsNotNone(human_in_the_loop.agent.db)

        source_page = HITL_PAGE.read_text()
        generated_page = _generated_hitl_page()
        self.assertEqual(generated_page, source_page)
        for label, page in {"source": source_page, "generated": generated_page}.items():
            with self.subTest(page=label):
                start = page.index('name: "generate_task_steps"')
                end = page.index("render:", start)
                client_tool_block = page[start:end]
                self.assertIn(
                    'status: z.enum(["enabled", "disabled", "executing"])',
                    client_tool_block,
                )
                self.assertNotIn(
                    'status: z.enum(["enabled", "disabled", "executing"]).optional()',
                    client_tool_block,
                )

    def test_haiku_uses_one_authoritative_silent_client_tool(self) -> None:
        self.assertFalse(hasattr(tool_based_generative_ui, "generate_haiku"))
        self.assertNotIn(
            "generate_haiku",
            [tool.name for tool in tool_based_generative_ui.agent.tools or []],
        )

        client_tool = _haiku_tool()
        parsed = parse_client_tools([client_tool])

        self.assertEqual([tool.name for tool in parsed], ["generate_haiku"])
        self.assertTrue(parsed[0].external_execution)
        self.assertTrue(parsed[0].external_execution_silent)
        self.assertEqual(parsed[0].parameters, client_tool.parameters)
        self.assertIsNotNone(tool_based_generative_ui.agent.db)


if __name__ == "__main__":
    unittest.main()
