from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = PROJECT_ROOT / "server" / "api"
DOJO_ROOT = PROJECT_ROOT.parents[3] / "apps" / "dojo" / "src"
DOJO_FILES = DOJO_ROOT / "files.json"
DOJO_FEATURES = DOJO_ROOT / "app" / "[integrationId]" / "feature"
PRIVATE_AGENT_MODULE = "agno.agent.agent"
SERVER_PACKAGE = "server"
API_PACKAGE = "server.api"
AGUI_PACKAGE = "agno.os.interfaces.agui"
AGUI_SYMBOL = "AGUI"
STOCK_AGUI_IMPORT = f"from {AGUI_PACKAGE} import {AGUI_SYMBOL}"


def _module_package(path: Path) -> str:
    return ".".join(path.relative_to(PROJECT_ROOT).parts[:-1])


def _resolve_import_from(node: ast.ImportFrom, package: str) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = package.split(".") if package else []
    if node.level > len(package_parts):
        raise ValueError(
            f"relative import (level {node.level}) reaches beyond package {package!r}"
        )
    base = package_parts[: len(package_parts) - node.level + 1]
    return ".".join([*base, *node.module.split(".")] if node.module else base)


def _is_module_under(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def _imported_symbols(
    tree: ast.AST, package: str = API_PACKAGE
) -> set[tuple[str, str]]:
    return {
        (_resolve_import_from(node, package), alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def _imported_modules(tree: ast.AST, package: str = API_PACKAGE) -> set[str]:
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
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(node, package)
            imported_modules.add(module)
            if module == "agno.os.interfaces.agui" or node.module is None:
                imported_modules.update(
                    f"{module}.{alias.name}" for alias in node.names
                )
            if module == "importlib":
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


def _private_agent_imports(tree: ast.AST, package: str = API_PACKAGE) -> set[str]:
    imports = {
        module
        for module in _imported_modules(tree, package)
        if _is_private_agent_module(module)
    }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and _resolve_import_from(node, package) == "agno.agent"
        ):
            imports.update(
                PRIVATE_AGENT_MODULE for alias in node.names if alias.name == "agent"
            )

    return imports


def _assert_no_private_agent_imports(
    test_case: unittest.TestCase, tree: ast.AST, package: str = API_PACKAGE
) -> None:
    private_imports = _private_agent_imports(tree, package)
    test_case.assertFalse(
        private_imports,
        f"found private Agent module import: {sorted(private_imports)}",
    )


def _assert_imports_public_agent(
    test_case: unittest.TestCase, tree: ast.AST, package: str = API_PACKAGE
) -> None:
    _assert_no_private_agent_imports(test_case, tree, package)
    test_case.assertIn(("agno.agent", "Agent"), _imported_symbols(tree, package))


def _agui_internals_imports(tree: ast.AST, package: str = API_PACKAGE) -> set[str]:
    """Modules under agno.os.interfaces.agui other than the public AGUI symbol.

    Every submodule of the package (input, handlers, resume, router, state,
    stream, utils, ...) is an AG-UI internal; only the AGUI class is public.
    """
    return {
        module
        for module in _imported_modules(tree, package)
        if module.startswith(f"{AGUI_PACKAGE}.")
        and module != f"{AGUI_PACKAGE}.{AGUI_SYMBOL}"
    }


def _binds_name(target: ast.expr, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_binds_name(element, name) for element in target.elts)
    if isinstance(target, ast.Starred):
        return _binds_name(target.value, name)
    return False


def _agui_bindings(tree: ast.AST, package: str = API_PACKAGE) -> set[str]:
    """Every statement that binds the module-level name AGUI, as source text."""
    bindings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(node, package)
            for alias in node.names:
                if (alias.asname or alias.name) == AGUI_SYMBOL:
                    suffix = f" as {alias.asname}" if alias.asname else ""
                    bindings.add(f"from {module} import {alias.name}{suffix}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name) == AGUI_SYMBOL:
                    suffix = f" as {alias.asname}" if alias.asname else ""
                    bindings.add(f"import {alias.name}{suffix}")
        elif isinstance(node, ast.Assign):
            if any(_binds_name(target, AGUI_SYMBOL) for target in node.targets):
                bindings.add(f"{AGUI_SYMBOL} = ...")
        elif isinstance(node, ast.AnnAssign):
            if _binds_name(node.target, AGUI_SYMBOL):
                bindings.add(f"{AGUI_SYMBOL}: ... = ...")
        elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):
            if _binds_name(node.target, AGUI_SYMBOL):
                bindings.add(f"{AGUI_SYMBOL} = ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == AGUI_SYMBOL:
                bindings.add(f"def {AGUI_SYMBOL}")
    return bindings


def _assert_uses_stock_public_agui(
    test_case: unittest.TestCase, tree: ast.AST, package: str = API_PACKAGE
) -> None:
    imported_modules = _imported_modules(tree, package)
    called_names = _called_names(tree)

    internal_imports = _agui_internals_imports(tree, package)
    test_case.assertFalse(
        internal_imports,
        f"found AG-UI internals imports: {sorted(internal_imports)}; "
        f"examples may import only {STOCK_AGUI_IMPORT!r}",
    )
    server_imports = {
        module
        for module in imported_modules
        if _is_module_under(module, SERVER_PACKAGE)
        and not _is_module_under(module, API_PACKAGE)
    }
    test_case.assertFalse(
        server_imports,
        f"found imports of server modules outside server.api: {sorted(server_imports)}",
    )
    bindings = _agui_bindings(tree, package)
    test_case.assertEqual(
        bindings,
        {STOCK_AGUI_IMPORT},
        f"{AGUI_SYMBOL} must be bound only by {STOCK_AGUI_IMPORT!r}; "
        f"found {sorted(bindings)}",
    )
    test_case.assertIn("AGUI", called_names)
    test_case.assertNotIn("ResumeAwareAGUI", called_names)


def _assert_server_contains_only_api_package(
    test_case: unittest.TestCase, server_root: Path
) -> None:
    entries = {
        path.name for path in server_root.iterdir() if path.name != "__pycache__"
    }
    test_case.assertEqual(
        entries,
        {"__init__.py", "api"},
        "server/ may only contain __init__.py and the api package",
    )


def _mounted_demo_modules(tree: ast.AST) -> set[str]:
    """Demo module file names that server/__init__.py imports from .api and mounts."""
    imported_apps: set[str] = set()
    mounted_apps: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "api"
        ):
            imported_apps.update(alias.asname or alias.name for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mount"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
        ):
            mounted_apps.add(node.args[1].id)
    return {
        f"{app_name.removesuffix('_app')}.py"
        for app_name in imported_apps & mounted_apps
        if app_name.endswith("_app")
    }


def _assert_api_contains_only_mounted_demos(
    test_case: unittest.TestCase, api_root: Path, mounted_modules: set[str]
) -> None:
    entries = {path.name for path in api_root.iterdir() if path.name != "__pycache__"}
    test_case.assertEqual(
        entries,
        {"__init__.py", *mounted_modules},
        "server/api/ may only contain __init__.py and the mounted demo modules",
    )


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


def _generated_agno_entries(catalog: object) -> list[tuple[str, str, str]]:
    if not isinstance(catalog, dict):
        return []

    return [
        (key.split("::", 1)[1], entry["name"], entry["content"])
        for key, entries in catalog.items()
        if isinstance(key, str) and key.startswith("agno::")
        if isinstance(entries, list)
        for entry in entries
        if isinstance(entry, dict)
        and (entry.get("name") == "page.tsx" or entry.get("language") == "python")
    ]


def _generated_entry_source_path(demo: str, name: str) -> Path:
    if name.endswith(".py"):
        return API_ROOT / name
    for version in ("(v2)", "(v1)"):
        candidate = DOJO_FEATURES / version / demo / name
        if candidate.exists():
            return candidate
    return DOJO_FEATURES / "(v2)" / demo / name


def _assert_generated_entry_matches_source(
    test_case: unittest.TestCase, demo: str, name: str, content: str
) -> None:
    source_path = _generated_entry_source_path(demo, name)
    test_case.assertTrue(source_path.exists(), f"missing source {source_path}")
    test_case.assertEqual(
        content,
        source_path.read_text(),
        f"generated agno::{demo} {name} differs from {source_path}; "
        "regenerate apps/dojo/src/files.json",
    )


def _assert_unique_generated_source_names(
    test_case: unittest.TestCase, generated_sources: list[tuple[str, str]]
) -> None:
    names = [name for name, _ in generated_sources]
    test_case.assertEqual(
        len(names),
        len(set(names)),
        "generated Agno Python entries must have unique names",
    )


def _write_server_layout(root: Path, extra_files: list[str]) -> Path:
    server_root = root / "server"
    for relative in ["__init__.py", "api/__init__.py", *extra_files]:
        path = server_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return server_root


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

    def test_relative_agent_import_does_not_satisfy_public_agent_guard(self) -> None:
        tree = ast.parse("from .agno.agent import Agent")

        self.assertEqual(
            _imported_symbols(tree, package="server.api"),
            {("server.api.agno.agent", "Agent")},
        )
        with self.assertRaises(AssertionError):
            _assert_imports_public_agent(self, tree)

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

    def test_imported_modules_resolves_relative_imports_against_the_package(
        self,
    ) -> None:
        tree = ast.parse(
            """
from ..local_agui import ResumeAwareAGUI as AGUIShim
from .agent.agent import Agent
from . import agui_compat
"""
        )

        self.assertEqual(
            _imported_modules(tree, package="server.api"),
            {
                "server.local_agui",
                "server.api.agent.agent",
                "server.api",
                "server.api.agui_compat",
            },
        )

    def test_imported_modules_rejects_relative_imports_beyond_the_package(
        self,
    ) -> None:
        tree = ast.parse("from ... import local_agui")

        with self.assertRaisesRegex(ValueError, "beyond"):
            _imported_modules(tree, package="server.api")

    def test_module_package_is_the_dotted_directory_under_the_project_root(
        self,
    ) -> None:
        self.assertEqual(_module_package(API_ROOT / "agentic_chat.py"), "server.api")
        self.assertEqual(
            _module_package(PROJECT_ROOT / "server" / "__init__.py"), "server"
        )
        self.assertEqual(_module_package(PROJECT_ROOT / "migrate_v3.py"), "")

    def test_relative_local_adapter_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from ..local_agui import ResumeAwareAGUI as AGUI

AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "outside server.api"):
            _assert_uses_stock_public_agui(self, tree)

    def test_renamed_local_adapter_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from server.agui_compat import Adapter as AGUI

AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "outside server.api"):
            _assert_uses_stock_public_agui(self, tree)

    def test_aliased_local_adapter_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from server import agui_compat as compat

AGUI = compat.Adapter
AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "outside server.api"):
            _assert_uses_stock_public_agui(self, tree)

    def test_stray_server_module_fails_layout_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_root = _write_server_layout(Path(tmp), ["agui_compat.py"])

            with self.assertRaisesRegex(AssertionError, "api package"):
                _assert_server_contains_only_api_package(self, server_root)

    def test_stray_server_package_fails_layout_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_root = _write_server_layout(Path(tmp), ["compat/__init__.py"])

            with self.assertRaisesRegex(AssertionError, "api package"):
                _assert_server_contains_only_api_package(self, server_root)

    def test_layout_guard_ignores_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_root = _write_server_layout(
                Path(tmp), ["__pycache__/__init__.cpython-312.pyc"]
            )

            _assert_server_contains_only_api_package(self, server_root)

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

        with self.assertRaisesRegex(AssertionError, "outside server.api"):
            _assert_uses_stock_public_agui(self, tree)

    def test_api_local_adapter_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from server.api.local_agui import ResumeAwareAGUI as AGUI

AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AGUI"):
            _assert_uses_stock_public_agui(self, tree)

    def test_relative_api_local_adapter_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from .local_agui import ResumeAwareAGUI as AGUI

AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AGUI"):
            _assert_uses_stock_public_agui(self, tree)

    def test_api_local_adapter_named_agui_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from server.api.local_agui import AGUI

AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AGUI"):
            _assert_uses_stock_public_agui(self, tree)

    def test_relative_api_local_adapter_named_agui_fails_stock_agui_guard(
        self,
    ) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from .local_agui import AGUI

AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AGUI"):
            _assert_uses_stock_public_agui(self, tree)

    def test_reassigned_agui_name_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from agno.os.interfaces.agui import AGUI as StockAGUI

AGUI = StockAGUI
AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AGUI"):
            _assert_uses_stock_public_agui(self, tree)

    def test_annotated_agui_rebinding_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from server.api import agui_compat

AGUI: type = agui_compat.Adapter
AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AGUI"):
            _assert_uses_stock_public_agui(self, tree)

    def test_aliased_stock_agui_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI as AGUI

AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AGUI"):
            _assert_uses_stock_public_agui(self, tree)

    def test_agui_bindings_reports_every_binding_of_the_name(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from .local_agui import ResumeAwareAGUI as AGUI
import server.api.agui_compat as AGUI

AGUI = object()
AGUI: type = object()
"""
        )

        self.assertEqual(
            _agui_bindings(tree, package="server.api"),
            {
                "from agno.os.interfaces.agui import AGUI",
                "from server.api.local_agui import ResumeAwareAGUI as AGUI",
                "import server.api.agui_compat as AGUI",
                "AGUI = ...",
                "AGUI: ... = ...",
            },
        )

    def test_agui_internals_submodule_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
from agno.os.interfaces.agui import AGUI
from agno.os.interfaces.agui.resume import ResumeAwareAGUI as Adapter

Adapter(agent=object())
AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AG-UI internals"):
            _assert_uses_stock_public_agui(self, tree)

    def test_agui_internals_direct_import_fails_stock_agui_guard(self) -> None:
        tree = ast.parse(
            """
import agno.os.interfaces.agui.state
from agno.os.interfaces.agui import AGUI

AGUI(agent=object())
"""
        )

        with self.assertRaisesRegex(AssertionError, "AG-UI internals"):
            _assert_uses_stock_public_agui(self, tree)

    def test_agui_internals_imports_reports_every_submodule_but_not_agui(
        self,
    ) -> None:
        tree = ast.parse(
            """
import agno.os.interfaces.agui
import agno.os.interfaces.agui.input
from agno.os.interfaces.agui import AGUI, utils
from agno.os.interfaces.agui.handlers import handle_run
from agno.os.interfaces.agui.router import build_router
"""
        )

        self.assertEqual(
            _agui_internals_imports(tree),
            {
                "agno.os.interfaces.agui.input",
                "agno.os.interfaces.agui.utils",
                "agno.os.interfaces.agui.handlers",
                "agno.os.interfaces.agui.router",
            },
        )

    def test_mounted_demo_modules_are_derived_from_the_api_import(self) -> None:
        tree = ast.parse(
            """
from .api import (
    agentic_chat_app,
    shared_state_app,
)

app.mount("/agentic_chat", agentic_chat_app, "Agentic Chat")
app.mount("/shared_state", shared_state_app, "Shared State")
"""
        )

        self.assertEqual(
            _mounted_demo_modules(tree),
            {"agentic_chat.py", "shared_state.py"},
        )

    def test_mounted_demo_modules_match_the_real_api_package(self) -> None:
        tree = ast.parse((PROJECT_ROOT / "server" / "__init__.py").read_text())
        demo_names = {
            path.name for path in API_ROOT.glob("*.py") if path.name != "__init__.py"
        }

        self.assertEqual(_mounted_demo_modules(tree), demo_names)

    def test_stray_api_module_fails_layout_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_root = _write_server_layout(
                Path(tmp), ["api/agentic_chat.py", "api/agui_compat.py"]
            )

            with self.assertRaisesRegex(AssertionError, "mounted demo"):
                _assert_api_contains_only_mounted_demos(
                    self, server_root / "api", {"agentic_chat.py"}
                )

    def test_stray_api_package_fails_layout_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_root = _write_server_layout(
                Path(tmp), ["api/agentic_chat.py", "api/compat/__init__.py"]
            )

            with self.assertRaisesRegex(AssertionError, "mounted demo"):
                _assert_api_contains_only_mounted_demos(
                    self, server_root / "api", {"agentic_chat.py"}
                )

    def test_api_layout_guard_accepts_mounted_demos_and_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_root = _write_server_layout(
                Path(tmp),
                ["api/agentic_chat.py", "api/__pycache__/agentic_chat.cpython-312.pyc"],
            )

            _assert_api_contains_only_mounted_demos(
                self, server_root / "api", {"agentic_chat.py"}
            )

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

    def test_mutated_generated_entry_fails_source_parity(self) -> None:
        catalog = json.loads(DOJO_FILES.read_text())
        demo, name, content = next(
            (demo, name, content)
            for demo, name, content in _generated_agno_entries(catalog)
            if name.endswith(".py")
        )

        with self.assertRaisesRegex(AssertionError, "differs from"):
            _assert_generated_entry_matches_source(
                self, demo, name, content + "\n# drifted\n"
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
                _assert_imports_public_agent(self, tree, _module_package(path))

    def test_demos_use_the_stock_public_agui_interface(self) -> None:
        demo_paths = sorted(
            path for path in API_ROOT.glob("*.py") if path.name != "__init__.py"
        )

        for path in demo_paths:
            with self.subTest(path=path.name):
                source = path.read_text()
                tree = ast.parse(source, filename=str(path))
                _assert_uses_stock_public_agui(self, tree, _module_package(path))

    def test_server_package_contains_only_the_api_package(self) -> None:
        _assert_server_contains_only_api_package(self, PROJECT_ROOT / "server")

    def test_api_package_contains_only_the_mounted_demos(self) -> None:
        tree = ast.parse((PROJECT_ROOT / "server" / "__init__.py").read_text())

        _assert_api_contains_only_mounted_demos(
            self, API_ROOT, _mounted_demo_modules(tree)
        )

    def test_project_imports_nothing_from_agui_internals(self) -> None:
        # The contract tests pin upstream internals on purpose; the shipped
        # example code must not.
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            if ".venv" in path.parts or "tests" in path.parts:
                continue
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                internal_imports = _agui_internals_imports(
                    ast.parse(path.read_text(), filename=str(path)),
                    _module_package(path),
                )
                self.assertFalse(
                    internal_imports,
                    "found AG-UI internals imports (media, utils, or any other "
                    f"agno.os.interfaces.agui submodule): {sorted(internal_imports)}",
                )

    def test_project_has_no_private_agent_imports(self) -> None:
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            if ".venv" in path.parts:
                continue
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                tree = ast.parse(path.read_text(), filename=str(path))
                _assert_no_private_agent_imports(self, tree, _module_package(path))

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

    def test_generated_agno_entries_match_their_sources(self) -> None:
        catalog = json.loads(DOJO_FILES.read_text())
        entries = _generated_agno_entries(catalog)

        self.assertTrue(entries, "no agno:: entries found in the Dojo catalog")
        for demo, name, content in entries:
            with self.subTest(demo=demo, name=name):
                _assert_generated_entry_matches_source(self, demo, name, content)


if __name__ == "__main__":
    unittest.main()
