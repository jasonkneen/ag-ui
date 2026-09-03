from __future__ import annotations

import importlib
import json
import re
import unittest
from pathlib import Path

from ag_ui.core import Tool
from agno.agent import Agent
from agno.os.interfaces.agui.input import parse_client_tools
from agno.tools.function import Function
from agno.tools.toolkit import Toolkit

from server.api import human_in_the_loop, tool_based_generative_ui

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[3]
DOJO_FILES = REPO_ROOT / "apps" / "dojo" / "src" / "files.json"
DEMO_MODULES_DIR = PROJECT_ROOT / "server" / "api"
DOJO_FEATURES_DIR = (
    REPO_ROOT / "apps" / "dojo" / "src" / "app" / "[integrationId]" / "feature" / "(v2)"
)
HITL_PAGE = DOJO_FEATURES_DIR / "human_in_the_loop" / "page.tsx"

# A client tool hook call, optionally with a generic argument, up to its "(".
CLIENT_TOOL_HOOK_CALL = re.compile(
    r"\buse(?:FrontendTool|HumanInTheLoop)\b(?:<[^()]*>)?\s*\("
)
TOOL_NAME_FIELD = re.compile(r"\bname:\s*[\"']([A-Za-z0-9_.-]+)[\"']")


def _client_tool_names(page: str) -> set[str]:
    """Tool names the Dojo page registers through useFrontendTool/useHumanInTheLoop."""
    names: set[str] = set()
    for call in CLIENT_TOOL_HOOK_CALL.finditer(page):
        match = TOOL_NAME_FIELD.search(page, call.end())
        if match is not None:
            names.add(match.group(1))
    return names


def _native_tool_names(agent: Agent) -> set[str]:
    """Tool names the agent registers itself, before any client tools are appended."""
    names: set[str] = set()
    for entry in agent.tools or []:
        if isinstance(entry, Toolkit):
            names.update(entry.functions)
        elif isinstance(entry, Function):
            names.add(entry.name)
        else:
            names.add(getattr(entry, "__name__", repr(entry)))
    return names


def _demos_with_dojo_page() -> list[tuple[str, Path]]:
    demos = []
    for module_path in sorted(DEMO_MODULES_DIR.glob("*.py")):
        if module_path.name.startswith("_"):
            continue
        page = DOJO_FEATURES_DIR / module_path.stem / "page.tsx"
        if page.exists():
            demos.append((module_path.stem, page))
    return demos


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
    def test_no_demo_registers_a_native_tool_named_like_a_client_tool(self) -> None:
        # Agno appends AG-UI client tools after the agent's own tools and keeps
        # the first duplicate name, so a same-name native stub shadows the
        # silent client tool and the run emits a spurious "needs external
        # execution" assistant message.
        demos = _demos_with_dojo_page()
        self.assertIn("agentic_chat", [name for name, _ in demos])

        for demo, page in demos:
            with self.subTest(demo=demo):
                module = importlib.import_module(f"server.api.{demo}")
                client_names = _client_tool_names(page.read_text())
                native_names = _native_tool_names(module.agent)
                self.assertEqual(
                    native_names & client_names,
                    set(),
                    f"{demo}: native tools shadow client tools {client_names}",
                )

    def test_client_tool_name_helper_reads_both_hooks(self) -> None:
        page = """
        import { useFrontendTool, useHumanInTheLoop } from "@copilotkit/react-core";
        useFrontendTool({ name: "change_background", parameters: z.object({}) });
        useHumanInTheLoop<{ steps: Step[] }>({
          name: "generate_task_steps",
        });
        useHumanInTheLoop(
          {
            name: 'confirm_changes',
          },
          [],
        );
        const other = { name: "not_a_tool" };
        """
        self.assertEqual(
            _client_tool_names(page),
            {"change_background", "generate_task_steps", "confirm_changes"},
        )

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
