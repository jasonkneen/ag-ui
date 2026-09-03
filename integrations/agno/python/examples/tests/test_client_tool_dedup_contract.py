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

from server.api import agentic_chat, human_in_the_loop, tool_based_generative_ui

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[3]
DOJO_FILES = REPO_ROOT / "apps" / "dojo" / "src" / "files.json"
DEMO_MODULES_DIR = PROJECT_ROOT / "server" / "api"
DOJO_FEATURES_DIR = (
    REPO_ROOT / "apps" / "dojo" / "src" / "app" / "[integrationId]" / "feature" / "(v2)"
)
HITL_PAGE = DOJO_FEATURES_DIR / "human_in_the_loop" / "page.tsx"
HAIKU_PAGE = DOJO_FEATURES_DIR / "tool_based_generative_ui" / "page.tsx"

# Client tools each page is known to register, so the guard cannot pass on an
# empty set when the hook regex silently stops matching.
EXPECTED_CLIENT_TOOL_NAMES = {
    "agentic_chat": {"change_background"},
    "human_in_the_loop": {"generate_task_steps"},
    "tool_based_generative_ui": {"generate_haiku"},
    "predictive_state_updates": {"confirm_changes", "write_document"},
}

# A client tool hook call, optionally with a generic argument, up to its "(".
CLIENT_TOOL_HOOK_CALL = re.compile(
    r"\buse(?:FrontendTool|HumanInTheLoop)\b(?:<[^()]*>)?\s*\("
)
TOOL_NAME_FIELD = re.compile(r"\bname:\s*[\"']([A-Za-z0-9_.-]+)[\"']")


def _skip_string_literal(page: str, start: int) -> int:
    """Return the index after the literal opened at start, or start + 1 when the
    quote is not a string (an apostrophe in JSX text never closes on its line)."""
    quote = page[start]
    index = start + 1
    while index < len(page):
        char = page[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        if char == "\n" and quote != "`":
            return start + 1
        index += 1
    return start + 1 if quote != "`" else index


# Depth of a key written directly on the hook's argument object: the call's
# own parenthesis plus the object's brace.
HOOK_OBJECT_KEY_DEPTH = 2


def _hook_argument_span(
    page: str, label: str, call_start: int, open_paren: int
) -> tuple[int, dict[int, int]]:
    """Locate the ")" balancing the hook call's "(" and record, for each code
    character in between, its bracket depth counted from the call. Characters
    inside string literals, template literals and comments are not recorded."""
    depth = 0
    depths: dict[int, int] = {}
    index = open_paren
    while index < len(page):
        pair = page[index : index + 2]
        if pair == "//":
            newline = page.find("\n", index)
            index = len(page) if newline == -1 else newline
            continue
        if pair == "/*":
            close = page.find("*/", index + 2)
            index = len(page) if close == -1 else close + 2
            continue
        char = page[index]
        if char in "'\"`":
            index = _skip_string_literal(page, index)
            continue
        depths[index] = depth
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index, depths
        index += 1
    raise AssertionError(
        f"{label}: client tool hook call at offset {call_start} never closes its parentheses"
    )


def _client_tool_names(page: str, label: str) -> set[str]:
    """Tool names the Dojo page registers through useFrontendTool/useHumanInTheLoop.

    Only a name field written directly on the hook's argument object counts;
    nested objects, strings and comments inside the call are ignored, and a
    hook without exactly one such field fails loud instead of being skipped.
    """
    names: set[str] = set()
    for call in CLIENT_TOOL_HOOK_CALL.finditer(page):
        open_paren = call.end() - 1
        close_paren, depths = _hook_argument_span(page, label, call.start(), open_paren)
        found = [
            match.group(1)
            for match in TOOL_NAME_FIELD.finditer(page, open_paren + 1, close_paren)
            if depths.get(match.start()) == HOOK_OBJECT_KEY_DEPTH
        ]
        if len(found) != 1:
            raise AssertionError(
                f"{label}: client tool hook call at offset {call.start()} declares "
                f"{len(found)} name fields, expected exactly one: {found}"
            )
        names.add(found[0])
    return names


def _native_tool_names(agent: Agent) -> set[str]:
    """Tool names the agent registers itself, before any client tools are appended."""
    names: set[str] = set()
    for entry in agent.tools or []:
        if isinstance(entry, Toolkit):
            names.update(entry.functions)
        elif isinstance(entry, Function):
            names.add(entry.name)
        elif callable(entry) and hasattr(entry, "__name__"):
            names.add(entry.__name__)
        else:
            raise AssertionError(
                f"unrecognised tool shape {type(entry).__name__} in agent.tools: {entry!r}"
            )
    return names


def _assert_no_client_tool_shadowed(
    demo: str, page: str, agent: Agent, label: str
) -> None:
    """The guard for one demo: its page registers the pinned client tools and
    none of the agent's native tools reuses one of their names."""
    client_names = _client_tool_names(page, label)
    expected = EXPECTED_CLIENT_TOOL_NAMES.get(demo)
    if expected is not None and client_names != expected:
        raise AssertionError(
            f"{label}: expected client tools {sorted(expected)}, found {sorted(client_names)}"
        )
    shadowed = _native_tool_names(agent) & client_names
    if shadowed:
        raise AssertionError(
            f"{demo}: native tools shadow client tools {sorted(shadowed)}"
        )


def _declared_image_names(page: str) -> list[str]:
    """The VALID_IMAGE_NAMES the haiku page declares as const."""
    match = re.search(r"const VALID_IMAGE_NAMES = \[(.*?)\] as const;", page, re.DOTALL)
    names = re.findall(r'"([^"]+)"', match.group(1)) if match else []
    if not names:
        raise AssertionError(
            "haiku page no longer declares VALID_IMAGE_NAMES as a const array"
        )
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


def _haiku_tool(image_names: list[str]) -> Tool:
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
                "image_name": {"type": "string", "enum": image_names},
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
        self.assertLessEqual(
            set(EXPECTED_CLIENT_TOOL_NAMES), {name for name, _ in demos}
        )

        for demo, page in demos:
            with self.subTest(demo=demo):
                module = importlib.import_module(f"server.api.{demo}")
                _assert_no_client_tool_shadowed(
                    demo, page.read_text(), module.agent, str(page)
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
            _client_tool_names(page, "synthetic.tsx"),
            {"change_background", "generate_task_steps", "confirm_changes"},
        )

    def test_client_tool_name_helper_fails_loud_on_a_hook_without_a_name(self) -> None:
        # The old search ran to the end of the page and adopted the next
        # unrelated name literal, so a nameless hook produced a phantom tool.
        page = """
        useFrontendTool({ description: "registered without a name" });
        const vegetable = { name: "Carrots" };
        """
        with self.assertRaisesRegex(AssertionError, r"synthetic\.tsx.*offset \d+"):
            _client_tool_names(page, "synthetic.tsx")

    def test_client_tool_name_helper_reads_the_hook_level_name_only(self) -> None:
        page = """
        useFrontendTool({
          render: ({ args }) => <Card item={{ name: "decoy" }} />,
          parameters: z.object({ name: z.string().describe("name: 'field'") }),
          name: "real_tool",
        });
        """
        self.assertEqual(_client_tool_names(page, "synthetic.tsx"), {"real_tool"})

    def test_client_tool_name_helper_skips_strings_comments_and_templates(self) -> None:
        page = """
        useHumanInTheLoop(
          {
            // name: "commented_out" ) (
            description: "unbalanced ) paren and name: 'quoted' inside a string",
            instructions: `) template name: "templated" (`,
            /* name: "block_comment" ) */
            name: "real_tool",
            render: () => <p>It's rendered {"("}</p>,
          },
          [],
        );
        useFrontendTool({ name: "second_tool" });
        """
        self.assertEqual(
            _client_tool_names(page, "synthetic.tsx"), {"real_tool", "second_tool"}
        )

    def test_client_tool_name_helper_fails_loud_on_an_unbalanced_hook_call(
        self,
    ) -> None:
        page = 'useFrontendTool({ name: "dangling"'
        with self.assertRaisesRegex(AssertionError, r"synthetic\.tsx.*offset 0"):
            _client_tool_names(page, "synthetic.tsx")

    def test_dojo_pages_register_the_pinned_client_tools(self) -> None:
        for demo, expected in EXPECTED_CLIENT_TOOL_NAMES.items():
            with self.subTest(demo=demo):
                page = DOJO_FEATURES_DIR / demo / "page.tsx"
                self.assertEqual(
                    _client_tool_names(page.read_text(), str(page)), expected
                )

    def test_guard_fails_when_a_pinned_page_registers_no_client_tool(self) -> None:
        # A renamed hook must not let the guard pass on an empty intersection.
        page = 'useRenderTool({ name: "change_background" });'
        with self.assertRaisesRegex(AssertionError, r"expected client tools"):
            _assert_no_client_tool_shadowed(
                "agentic_chat", page, agentic_chat.agent, "synthetic.tsx"
            )

    def test_guard_fails_when_a_native_tool_shadows_a_client_tool(self) -> None:
        def change_background(background: str) -> str:
            return background

        agent = Agent(tools=[change_background])
        page = 'useFrontendTool({ name: "change_background" });'
        with self.assertRaisesRegex(AssertionError, r"shadow.*change_background"):
            _assert_no_client_tool_shadowed(
                "agentic_chat", page, agent, "synthetic.tsx"
            )

    def test_native_tool_name_helper_reads_toolkits_functions_and_callables(
        self,
    ) -> None:
        def plain_callable(value: int) -> int:
            return value

        def toolkit_member(value: int) -> int:
            return value

        agent = Agent(
            tools=[
                Toolkit(name="kit", tools=[toolkit_member]),
                Function(name="declared_function"),
                plain_callable,
            ]
        )
        self.assertEqual(
            _native_tool_names(agent),
            {"toolkit_member", "declared_function", "plain_callable"},
        )

    def test_native_tool_name_helper_fails_loud_on_an_unrecognised_tool_shape(
        self,
    ) -> None:
        with self.assertRaisesRegex(AssertionError, r"unrecognised tool shape.*int"):
            _native_tool_names(Agent(tools=[42]))

    def test_declared_image_names_fails_loud_when_the_page_drops_the_array(
        self,
    ) -> None:
        with self.assertRaisesRegex(AssertionError, r"VALID_IMAGE_NAMES"):
            _declared_image_names("const SOMETHING_ELSE = [] as const;")

    def test_hitl_uses_one_authoritative_silent_client_tool(self) -> None:
        self.assertFalse(hasattr(human_in_the_loop, "generate_task_steps"))
        self.assertNotIn(
            "generate_task_steps", _native_tool_names(human_in_the_loop.agent)
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
            "generate_haiku", _native_tool_names(tool_based_generative_ui.agent)
        )

        image_names = _declared_image_names(HAIKU_PAGE.read_text())
        client_tool = _haiku_tool(image_names)
        parsed = parse_client_tools([client_tool])

        # parse_client_tools must keep the name and schema and mark the tool
        # as silent external execution.
        self.assertEqual([tool.name for tool in parsed], ["generate_haiku"])
        self.assertTrue(parsed[0].external_execution)
        self.assertTrue(parsed[0].external_execution_silent)
        self.assertEqual(parsed[0].parameters, client_tool.parameters)
        self.assertEqual(
            parsed[0].parameters["properties"]["image_name"]["enum"], image_names
        )
        self.assertIsNotNone(tool_based_generative_ui.agent.db)


if __name__ == "__main__":
    unittest.main()
