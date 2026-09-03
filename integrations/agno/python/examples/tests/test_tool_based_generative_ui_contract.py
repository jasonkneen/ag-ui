from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[3]
DOJO_PAGE = (
    REPO_ROOT
    / "apps"
    / "dojo"
    / "src"
    / "app"
    / "[integrationId]"
    / "feature"
    / "(v2)"
    / "tool_based_generative_ui"
    / "page.tsx"
)
AGENTIC_DOJO_PAGE = (
    REPO_ROOT
    / "apps"
    / "dojo"
    / "src"
    / "app"
    / "[integrationId]"
    / "feature"
    / "(v2)"
    / "agentic_generative_ui"
    / "page.tsx"
)
DOJO_FILES = REPO_ROOT / "apps" / "dojo" / "src" / "files.json"
DOJO_DIR = REPO_ROOT / "apps" / "dojo"
EXPECTED_IMAGE_NAMES = (
    "Osaka_Castle_Turret_Stone_Wall_Pine_Trees_Daytime.jpg",
    "Tokyo_Skyline_Night_Tokyo_Tower_Mount_Fuji_View.jpg",
    "Itsukushima_Shrine_Miyajima_Floating_Torii_Gate_Sunset_Long_Exposure.jpg",
    "Takachiho_Gorge_Waterfall_River_Lush_Greenery_Japan.jpg",
    "Bonsai_Tree_Potted_Japanese_Art_Green_Foliage.jpeg",
    "Shirakawa-go_Gassho-zukuri_Thatched_Roof_Village_Aerial_View.jpg",
    "Ginkaku-ji_Silver_Pavilion_Kyoto_Japanese_Garden_Pond_Reflection.jpg",
    "Senso-ji_Temple_Asakusa_Cherry_Blossoms_Kimono_Umbrella.jpg",
    "Cherry_Blossoms_Sakura_Night_View_City_Lights_Japan.jpg",
    "Mount_Fuji_Lake_Reflection_Cherry_Blossoms_Sakura_Spring.jpg",
)


def _frontend_tool_block() -> str:
    page = DOJO_PAGE.read_text()
    start = page.index('name: "generate_haiku"')
    end = page.index("render:", start)
    return page[start:end]


def _run_javascript(source: str, expression: str, cwd: Path | None = None) -> object:
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", f"{source}\n{expression}"],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return json.loads(completed.stdout)


def _run_typescript(source: str, expression: str) -> object:
    if not (DOJO_DIR / "node_modules" / "zod").exists():
        raise AssertionError(
            "zod is not installed under apps/dojo/node_modules; run pnpm install"
        )
    completed = subprocess.run(
        [
            "node",
            "--input-type=module-typescript",
            "--eval",
            f'import {{ z }} from "zod";\n{source}\n{expression}',
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=DOJO_DIR,
    )
    return json.loads(completed.stdout)


def _dojo_node_modules_root() -> Path | None:
    for candidate in (REPO_ROOT / "apps" / "dojo", REPO_ROOT):
        modules = candidate / "node_modules"
        if (modules / "zod").is_dir() and (modules / "zod-to-json-schema").is_dir():
            return candidate
    return None


def _haiku_tool_json_schema(cwd: Path) -> dict[str, object]:
    page = DOJO_PAGE.read_text()
    start = page.index("const VALID_IMAGE_NAMES = [")
    end = page.index(".strict();", start) + len(".strict();")
    schema = page[start:end]
    schema = schema.replace("] as const;", "];")
    schema = schema.replace("value: string", "value")
    schema = schema.replace("): boolean", ")")
    source = (
        'import { z } from "zod";\n'
        'import { zodToJsonSchema } from "zod-to-json-schema";\n'
        f"{schema}"
    )
    expression = "console.log(JSON.stringify(zodToJsonSchema(HAIKU_SCHEMA, {})))"
    result = _run_javascript(source, expression, cwd=cwd)
    if not isinstance(result, dict):
        raise TypeError("zodToJsonSchema did not return an object schema")
    return result


def _agent_step_results(values: list[object]) -> list[bool]:
    page = AGENTIC_DOJO_PAGE.read_text()
    start = page.index("function isAgentStep")
    end = page.index("\n\nfunction normalizeAgentSteps", start)
    validator = page[start:end].replace(
        "(value: unknown): value is AgentStep", "(value)"
    )
    expression = f"console.log(JSON.stringify({json.dumps(values)}.map(isAgentStep)))"
    result = _run_javascript(validator, expression)
    if not isinstance(result, list) or not all(
        isinstance(item, bool) for item in result
    ):
        raise AssertionError("isAgentStep did not return a boolean list")
    return result


def _generated_agentic_page() -> str:
    catalog = json.loads(DOJO_FILES.read_text())
    return next(
        entry["content"]
        for entry in catalog["agno::agentic_generative_ui"]
        if entry["name"] == "page.tsx"
    )


def _generated_haiku_page() -> str:
    catalog = json.loads(DOJO_FILES.read_text())
    return next(
        entry["content"]
        for entry in catalog["agno::tool_based_generative_ui"]
        if entry["name"] == "page.tsx"
    )


def _task_progress_block(page: str) -> str:
    start = page.index("function TaskProgress")
    end = page.index("\n\n// Enhanced Icons", start)
    return page[start:end]


def _active_step_indices(step_lists: list[list[dict[str, str]]]) -> list[int]:
    page = AGENTIC_DOJO_PAGE.read_text()
    if "function resolveActiveStepIndex" not in page:
        raise AssertionError("resolveActiveStepIndex is not defined")
    start = page.index("function resolveActiveStepIndex")
    end = page.index("\n\nconst Chat", start)
    resolver = page[start:end].replace("(steps: AgentStep[]): number", "(steps)")
    expression = (
        "console.log(JSON.stringify("
        f"{json.dumps(step_lists)}.map(resolveActiveStepIndex)))"
    )
    result = _run_javascript(resolver, expression)
    if not isinstance(result, list) or not all(type(item) is int for item in result):
        raise AssertionError("resolveActiveStepIndex did not return an int list")
    return result


def _step(description: str, status: str) -> dict[str, str]:
    return {"description": description, "status": status}


def _haiku_validator_results(function_name: str, values: list[str]) -> list[bool]:
    page = DOJO_PAGE.read_text()
    if "const SAFE_GRADIENT_INNER_FUNCTIONS" not in page:
        raise AssertionError("haiku validators are not defined")
    start = page.index("const SAFE_GRADIENT_INNER_FUNCTIONS")
    end = page.index("\n\nconst HAIKU_LINE", start)
    validators = page[start:end]
    validators = validators.replace("value: string", "value")
    validators = validators.replace("): boolean", ")")
    expression = (
        f"console.log(JSON.stringify({json.dumps(values)}.map({function_name})))"
    )
    result = _run_javascript(validators, expression)
    if not isinstance(result, list) or not all(
        isinstance(item, bool) for item in result
    ):
        raise AssertionError(f"{function_name} did not return a boolean list")
    return result


def _haiku_parse_results(payloads: list[object]) -> list[dict[str, object]]:
    page = DOJO_PAGE.read_text()
    if "function parseHaikuToolArgs" not in page:
        raise AssertionError("parseHaikuToolArgs is not defined")
    start = page.index("const VALID_IMAGE_NAMES = [")
    end = page.index("\n\nfunction HaikuDisplay", start)
    helpers = page[start:end].replace("interface Haiku", "type Haiku =")
    expression = (
        f"console.log(JSON.stringify({json.dumps(payloads)}.map(parseHaikuToolArgs)))"
    )
    result = _run_typescript(helpers, expression)
    if not isinstance(result, list) or not all(
        isinstance(item, dict) for item in result
    ):
        raise AssertionError("parseHaikuToolArgs did not return an object list")
    return result


def _handler_block() -> str:
    block = _frontend_tool_block()
    start = block.index("handler:")
    return block[start:]


def _valid_haiku_payload() -> dict[str, object]:
    return {
        "japanese": ["古池や", "蛙飛び込む", "水の音"],
        "english": ["An old silent pond", "A frog jumps into the pond", "Splash"],
        "image_name": EXPECTED_IMAGE_NAMES[0],
        "gradient": "linear-gradient(135deg, #0f172a 0%, #2563eb 100%)",
    }


def _declared_image_names() -> tuple[str, ...]:
    page = DOJO_PAGE.read_text()
    start = page.index("const VALID_IMAGE_NAMES = [")
    end = page.index("] as const;", start)
    block = page[start:end]
    return tuple(re.findall(r'^\s+"([^"]+)",$', block, flags=re.MULTILINE))


class ToolBasedGenerativeUIContractTests(unittest.TestCase):
    def test_agentic_step_guard_preserves_supported_stream_statuses(self) -> None:
        steps = [
            {"description": "Queued", "status": "pending"},
            {"description": "Working", "status": "in_progress"},
            {"description": "Done", "status": "completed"},
            {"description": "Unknown", "status": "cancelled"},
        ]

        self.assertEqual(_agent_step_results(steps), [True, True, True, False])

    def test_active_task_row_is_the_streamed_in_progress_step(self) -> None:
        step_lists = [
            [
                _step("Plan", "pending"),
                _step("Build", "in_progress"),
                _step("Launch", "pending"),
            ],
            [
                _step("Plan", "completed"),
                _step("Build", "in_progress"),
                _step("Launch", "pending"),
            ],
        ]

        self.assertEqual(_active_step_indices(step_lists), [1, 1])

    def test_active_task_row_falls_back_to_the_first_pending_step(self) -> None:
        step_lists = [
            [
                _step("Plan", "completed"),
                _step("Build", "pending"),
                _step("Launch", "pending"),
            ],
            [_step("Plan", "pending"), _step("Build", "pending")],
            [_step("Plan", "completed"), _step("Build", "completed")],
            [],
        ]

        self.assertEqual(_active_step_indices(step_lists), [1, 0, -1, -1])

    def test_task_progress_rows_follow_the_resolved_active_index(self) -> None:
        pages = {
            "source": AGENTIC_DOJO_PAGE.read_text(),
            "generated": _generated_agentic_page(),
        }

        for label, page in pages.items():
            with self.subTest(page=label):
                block = _task_progress_block(page)
                self.assertIn(
                    "const activeStepIndex = resolveActiveStepIndex(steps);", block
                )
                self.assertIn("const isActive = index === activeStepIndex;", block)
                self.assertNotIn("isCurrentPending", block)
                self.assertNotIn("isFuturePending", block)

    def test_generated_haiku_page_matches_source(self) -> None:
        source_page = DOJO_PAGE.read_text()
        generated_page = _generated_haiku_page()

        self.assertEqual(generated_page, source_page)
        for label, page in {"source": source_page, "generated": generated_page}.items():
            with self.subTest(page=label):
                self.assertIn("gradient: SAFE_GRADIENT", page)
                self.assertIn(".enum(VALID_IMAGE_NAMES)", page)

    def test_frontend_schema_is_the_strict_haiku_contract(self) -> None:
        block = _frontend_tool_block()

        self.assertIn("parameters: HAIKU_SCHEMA", block)

        page = DOJO_PAGE.read_text()
        schema_start = page.index("const HAIKU_SCHEMA = z")
        schema_end = page.index("\n\n", schema_start)
        schema = page[schema_start:schema_end]
        fields = re.findall(r"^\s+(\w+): ", schema, flags=re.MULTILINE)

        compact = re.sub(r"\s+", "", schema)

        self.assertEqual(fields, ["japanese", "english", "image_name", "gradient"])
        self.assertIn("japanese:z.array(HAIKU_LINE).length(3)", compact)
        self.assertIn("english:z.array(HAIKU_LINE).length(3)", compact)
        self.assertIn("image_name:z.enum(VALID_IMAGE_NAMES)", compact)
        self.assertIn("gradient:SAFE_GRADIENT", compact)
        self.assertRegex(schema, r"\}\)\s*\.strict\(\);$")

    def test_haiku_lines_reject_empty_or_whitespace_only_text(self) -> None:
        values = ["古池や", "  an old pond  ", "", "   \t"]

        self.assertEqual(
            _haiku_validator_results("isNonBlankHaikuLine", values),
            [True, True, False, False],
        )

    def test_image_schema_allowlist_is_exact(self) -> None:
        self.assertEqual(_declared_image_names(), EXPECTED_IMAGE_NAMES)

    def test_gradient_schema_accepts_intended_gradient_functions(self) -> None:
        gradients = [
            "linear-gradient(135deg, #0f172a 0%, #2563eb 100%)",
            "radial-gradient(circle at center, rgba(255, 255, 255, 0.8), #000)",
            "conic-gradient(from 90deg, red, blue, red)",
        ]

        self.assertEqual(
            _haiku_validator_results("isSafeGradient", gradients),
            [True, True, True],
        )

    def test_gradient_schema_rejects_url_capable_css_image_functions(self) -> None:
        gradients = [
            'linear-gradient(red, url("https://example.test/pixel.png"))',
            'linear-gradient(red, blue), image-set("https://example.test/x" 1x)',
            'linear-gradient(red, src("https://example.test/x"))',
            'linear-gradient(red, image("https://example.test/x"))',
            "linear-gradient(red, cross-fade(red, blue, 50%))",
            "linear-gradient(red, element(#preview))",
            "linear-gradient(red, paint(worklet))",
            'linear-gradient(red, -webkit-image-set("https://example.test/x" 1x))',
            r"linear-gradient(red, u\72l(https://example.test/x))",
        ]

        self.assertEqual(
            _haiku_validator_results("isSafeGradient", gradients),
            [False] * len(gradients),
        )

    def test_gradient_schema_rejects_non_gradient_and_extra_layers(self) -> None:
        gradients = [
            "red",
            "linear-gradient(red, blue), red",
            "linear-gradient(red, blue",
            "linear-gradient(red, blue))",
        ]

        self.assertEqual(
            _haiku_validator_results("isSafeGradient", gradients),
            [False, False, False, False],
        )

    def test_streaming_preview_validates_complete_args_before_render(self) -> None:
        page = DOJO_PAGE.read_text()
        render_start = page.index("render:", page.index('name: "generate_haiku"'))
        render_end = page.index("\n      },", render_start)
        render = page[render_start:render_end]

        self.assertIn("HAIKU_SCHEMA.safeParse(args)", render)
        self.assertIn("if (!parsed.success)", render)
        self.assertIn("haiku={parsed.data}", render)
        self.assertNotIn("args as Haiku", render)

    def test_handler_parser_rejects_invalid_arguments_and_names_the_fields(
        self,
    ) -> None:
        valid = _valid_haiku_payload()
        missing_english = {k: v for k, v in valid.items() if k != "english"}
        results = _haiku_parse_results(
            [
                missing_english,
                {**valid, "japanese": "not a list"},
                {**valid, "image_name": "Not_In_The_Allowlist.jpg"},
                {**valid, "gradient": 'url("https://example.test/pixel.png")'},
            ]
        )

        for result, field in zip(
            results, ["english", "japanese", "image_name", "gradient"]
        ):
            with self.subTest(field=field):
                self.assertEqual(result["ok"], False)
                self.assertNotIn("haiku", result)
                self.assertIsInstance(result["message"], str)
                self.assertIn(field, result["message"])

    def test_handler_parser_accepts_the_valid_haiku_payload(self) -> None:
        valid = _valid_haiku_payload()

        self.assertEqual(_haiku_parse_results([valid]), [{"ok": True, "haiku": valid}])

    def test_handler_stores_only_parsed_haiku_arguments(self) -> None:
        handler = _handler_block()

        self.assertRegex(handler, r"parseHaikuToolArgs\(\s*args\s*\)")
        self.assertRegex(
            handler, r"if\s*\(\s*!parsed\.ok\s*\)\s*return parsed\.message;"
        )
        self.assertIn("parsed.haiku", handler)
        self.assertNotRegex(handler, r"handler:\s*async\s*\(\s*\{")

    def test_client_tool_keeps_follow_up_disabled(self) -> None:
        self.assertIn("followUp: false", _frontend_tool_block())

    def test_client_tool_registration_describes_the_tool(self) -> None:
        match = re.search(r'description:\s*"([^"]*)"', _frontend_tool_block())

        self.assertIsNotNone(match, "generate_haiku has no description")
        assert match is not None
        self.assertIn("haiku", match.group(1).lower())

    def test_schema_source_describes_every_haiku_field(self) -> None:
        page = DOJO_PAGE.read_text()
        schema_start = page.index("const HAIKU_SCHEMA = z")
        schema_end = page.index(".strict();", schema_start)
        schema = page[schema_start:schema_end]
        parts = re.split(r"^ {4}(\w+): ", schema, flags=re.MULTILINE)
        descriptions: dict[str, str] = {}
        for name, body in zip(parts[1::2], parts[2::2]):
            match = re.search(r'\.describe\(\s*"([^"]+)"', body)
            self.assertIsNotNone(match, f"{name} has no .describe()")
            assert match is not None
            descriptions[name] = match.group(1)

        self.assertEqual(
            sorted(descriptions), ["english", "gradient", "image_name", "japanese"]
        )
        self.assertIn("three", descriptions["japanese"].lower())
        self.assertIn("three", descriptions["english"].lower())
        self.assertIn("gradient", descriptions["gradient"].lower())
        self.assertIn("url", descriptions["gradient"].lower())

    def test_tool_json_schema_carries_field_descriptions_to_the_model(self) -> None:
        cwd = _dojo_node_modules_root()
        if cwd is None:
            self.skipTest("dojo node_modules (zod, zod-to-json-schema) not installed")

        schema = _haiku_tool_json_schema(cwd)
        properties = schema["properties"]
        assert isinstance(properties, dict)

        self.assertEqual(
            sorted(properties), ["english", "gradient", "image_name", "japanese"]
        )
        for name, field in properties.items():
            with self.subTest(field=name):
                self.assertTrue(field.get("description"), f"{name} lacks description")
        self.assertEqual(properties["image_name"]["enum"], list(EXPECTED_IMAGE_NAMES))
        self.assertEqual(properties["japanese"]["minItems"], 3)
        self.assertEqual(properties["japanese"]["maxItems"], 3)
        gradient = properties["gradient"]["description"].lower()
        self.assertIn("linear-gradient", gradient)
        self.assertIn("radial-gradient", gradient)
        self.assertIn("conic-gradient", gradient)
        self.assertIn("url", gradient)
        self.assertIs(schema["additionalProperties"], False)


if __name__ == "__main__":
    unittest.main()
