from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = PROJECT_ROOT / "server" / "api"
DOJO_FILES = PROJECT_ROOT.parents[3] / "apps" / "dojo" / "src" / "files.json"
PRIVATE_AGENT_MODULE = "agno.agent.agent"


def _imported_symbols(tree: ast.AST) -> set[tuple[str, str]]:
    return {
        (node.module or "", alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def _imported_modules(tree: ast.AST) -> set[str]:
    imported_modules: set[str] = set()
    string_bindings: dict[str, str] = {}
    importlib_aliases = {"importlib"}
    import_module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        string_bindings[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                string_bindings[node.target.id] = node.value.value
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
            if node.module == "agno.os.interfaces.agui":
                imported_modules.update(
                    f"{node.module}.{alias.name}" for alias in node.names
                )
            if node.module == "importlib":
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
            importlib_aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "importlib"
            )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_import_call = (
            isinstance(node.func, ast.Name)
            and node.func.id in {"__import__", *import_module_aliases}
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_aliases
        )
        if not is_import_call:
            continue

        module_expression: ast.expr | None = node.args[0] if node.args else None
        if module_expression is None:
            module_expression = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "name"),
                None,
            )
        if isinstance(module_expression, ast.Constant) and isinstance(
            module_expression.value, str
        ):
            imported_modules.add(module_expression.value)
        elif isinstance(module_expression, ast.Name):
            module_name = string_bindings.get(module_expression.id)
            if module_name is not None:
                imported_modules.add(module_name)

    return imported_modules


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _is_private_agent_module(module: str) -> bool:
    return module == PRIVATE_AGENT_MODULE or module.startswith(
        f"{PRIVATE_AGENT_MODULE}."
    )


def _private_agent_imports(tree: ast.AST) -> set[str]:
    imports = {
        module for module in _imported_modules(tree) if _is_private_agent_module(module)
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "agno.agent":
                imports.update(
                    PRIVATE_AGENT_MODULE
                    for alias in node.names
                    if alias.name == "agent"
                )

    return imports


def _assert_no_private_agent_imports(
    test_case: unittest.TestCase, tree: ast.AST
) -> None:
    test_case.assertFalse(
        _private_agent_imports(tree),
        f"found private Agent module import: {sorted(_private_agent_imports(tree))}",
    )


def _assert_imports_public_agent(test_case: unittest.TestCase, tree: ast.AST) -> None:
    _assert_no_private_agent_imports(test_case, tree)
    test_case.assertIn(("agno.agent", "Agent"), _imported_symbols(tree))


def _assert_uses_stock_public_agui(test_case: unittest.TestCase, tree: ast.AST) -> None:
    imported_modules = _imported_modules(tree)
    called_names = _called_names(tree)

    test_case.assertIn(("agno.os.interfaces.agui", "AGUI"), _imported_symbols(tree))
    test_case.assertFalse(
        {
            module
            for module in imported_modules
            if module == "server.local_agui" or module.startswith("server.local_agui.")
        },
        "found local AG-UI compatibility adapter import",
    )
    test_case.assertIn("AGUI", called_names)
    test_case.assertNotIn("ResumeAwareAGUI", called_names)


def _generated_agno_python_sources(catalog: object) -> list[tuple[str, str]]:
    if not isinstance(catalog, dict):
        return []

    return [
        (entry["name"], entry["content"])
        for key, entries in catalog.items()
        if isinstance(key, str) and key.startswith("agno::")
        if isinstance(entries, list)
        for entry in entries
        if isinstance(entry, dict) and entry.get("language") == "python"
    ]


def _assert_unique_generated_source_names(
    test_case: unittest.TestCase, generated_sources: list[tuple[str, str]]
) -> None:
    names = [name for name, _ in generated_sources]
    test_case.assertEqual(
        len(names),
        len(set(names)),
        "generated Agno Python entries must have unique names",
    )


class PublicAguiAdoptionTests(unittest.TestCase):
    def test_imported_modules_reports_import_from_and_direct_imports(self) -> None:
        tree = ast.parse(
            """
import agno.os.interfaces.agui.media
from agno.os.interfaces.agui.utils import coerce_event
"""
        )

        self.assertEqual(
            _imported_modules(tree),
            {
                "agno.os.interfaces.agui.media",
                "agno.os.interfaces.agui.utils",
            },
        )

    def test_imported_modules_reports_agui_media_imported_from_package(self) -> None:
        tree = ast.parse("from agno.os.interfaces.agui import media, utils")

        self.assertIn("agno.os.interfaces.agui.media", _imported_modules(tree))
        self.assertIn("agno.os.interfaces.agui.utils", _imported_modules(tree))

    def test_imported_modules_reports_dynamic_import_aliases_and_constants(
        self,
    ) -> None:
        tree = ast.parse(
            """
import importlib as loader
from importlib import import_module as load_module

legacy_media = "agno.os.interfaces.agui.media"
loader.import_module(name=legacy_media)
load_module("agno.os.interfaces.agui.utils")
__import__(name="agno.os.interfaces.agui.input")
"""
        )

        self.assertTrue(
            {
                "agno.os.interfaces.agui.media",
                "agno.os.interfaces.agui.utils",
                "agno.os.interfaces.agui.input",
            }.issubset(_imported_modules(tree))
        )

    def test_private_agent_module_direct_import_fails_public_agent_guard(self) -> None:
        tree = ast.parse(
            """
import agno.agent.agent as private_agent
from agno.agent import Agent
"""
        )

        with self.assertRaisesRegex(
            AssertionError,
            "found private Agent module import",
        ):
            _assert_imports_public_agent(self, tree)

    def test_private_agent_package_import_fails_public_agent_guard(self) -> None:
        tree = ast.parse(
            """
from agno.agent import agent as private_agent
from agno.agent import Agent
"""
        )

        with self.assertRaisesRegex(AssertionError, "private Agent"):
            _assert_imports_public_agent(self, tree)

    def test_dynamic_private_agent_import_fails_public_agent_guard(self) -> None:
        tree = ast.parse(
            """
import importlib
from agno.agent import Agent
private_agent = importlib.import_module("agno.agent.agent")
"""
        )

        with self.assertRaisesRegex(AssertionError, "private Agent"):
            _assert_imports_public_agent(self, tree)

    def test_indirect_keyword_private_agent_import_fails_public_agent_guard(
        self,
    ) -> None:
        tree = ast.parse(
            """
import importlib as loader
from agno.agent import Agent

private_module = "agno.agent.agent"
private_agent = loader.import_module(name=private_module)
"""
        )

        with self.assertRaisesRegex(AssertionError, "private Agent"):
            _assert_imports_public_agent(self, tree)

    def test_dynamic_local_adapter_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
import importlib
from agno.os.interfaces.agui import AGUI

adapter_module = "server.local_agui"
importlib.import_module(adapter_module)
AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "local AG-UI"):
            _assert_uses_stock_public_agui(self, tree)

    def test_generated_catalog_collection_preserves_duplicate_names(self) -> None:
        catalog = {
            "agno::example": [
                {
                    "name": "example.py",
                    "language": "python",
                    "content": "first",
                },
                {
                    "name": "example.py",
                    "language": "python",
                    "content": "second",
                },
            ]
        }

        self.assertEqual(
            _generated_agno_python_sources(catalog),
            [("example.py", "first"), ("example.py", "second")],
        )
        with self.assertRaisesRegex(AssertionError, "unique names"):
            _assert_unique_generated_source_names(
                self, _generated_agno_python_sources(catalog)
            )

    def test_dunder_private_agent_import_fails_public_agent_guard(self) -> None:
        tree = ast.parse(
            """
from agno.agent import Agent
private_agent = __import__("agno.agent.agent", fromlist=["Agent"])
"""
        )

        with self.assertRaisesRegex(AssertionError, "private Agent"):
            _assert_imports_public_agent(self, tree)

    def test_multiline_private_agent_import_fails_public_agent_guard(self) -> None:
        tree = ast.parse(
            """
from agno.agent import Agent
from agno.agent.agent import (
    Agent as PrivateAgent,
)
"""
        )

        with self.assertRaisesRegex(AssertionError, "private Agent"):
            _assert_imports_public_agent(self, tree)

    def test_backend_tool_example_describes_its_weather_tool_inventory(self) -> None:
        source = (API_ROOT / "backend_tool_rendering.py").read_text()
        module_docstring = ast.get_docstring(ast.parse(source)) or ""

        self.assertIn("weather", module_docstring.lower())
        self.assertNotIn("finance", module_docstring.lower())
        self.assertNotIn("YFinanceTools", module_docstring)

    def test_demos_import_agent_from_public_module(self) -> None:
        demo_paths = sorted(
            path for path in API_ROOT.glob("*.py") if path.name != "__init__.py"
        )

        for path in demo_paths:
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(), filename=str(path))
                _assert_imports_public_agent(self, tree)

    def test_demos_use_the_stock_public_agui_interface(self) -> None:
        demo_paths = sorted(
            path for path in API_ROOT.glob("*.py") if path.name != "__init__.py"
        )

        for path in demo_paths:
            with self.subTest(path=path.name):
                source = path.read_text()
                tree = ast.parse(source, filename=str(path))
                _assert_uses_stock_public_agui(self, tree)

    def test_project_has_no_local_agui_compatibility_adapter(self) -> None:
        self.assertFalse((PROJECT_ROOT / "server" / "local_agui.py").exists())

    def test_project_has_no_pre_split_agui_imports(self) -> None:
        legacy_modules = {
            "agno.os.interfaces.agui.media",
            "agno.os.interfaces.agui.utils",
        }

        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            if ".venv" in path.parts:
                continue
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                imported_modules = _imported_modules(
                    ast.parse(path.read_text(), filename=str(path))
                )
                legacy_imports = legacy_modules & imported_modules
                self.assertFalse(
                    legacy_imports,
                    f"found legacy AG-UI imports: {sorted(legacy_imports)}",
                )

    def test_project_has_no_private_agent_imports(self) -> None:
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            if ".venv" in path.parts:
                continue
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                tree = ast.parse(path.read_text(), filename=str(path))
                _assert_no_private_agent_imports(self, tree)

    def test_generated_agno_python_sources_use_stock_public_interfaces(self) -> None:
        catalog = json.loads(DOJO_FILES.read_text())
        generated_python = _generated_agno_python_sources(catalog)
        generated_names = [name for name, _ in generated_python]
        demo_names = {
            path.name for path in API_ROOT.glob("*.py") if path.name != "__init__.py"
        }

        _assert_unique_generated_source_names(self, generated_python)
        self.assertEqual(set(generated_names), demo_names)
        for name, source in sorted(generated_python):
            with self.subTest(path=name):
                tree = ast.parse(source, filename=name)
                _assert_imports_public_agent(self, tree)
                _assert_uses_stock_public_agui(self, tree)


if __name__ == "__main__":
    unittest.main()
