from __future__ import annotations

import ast
import json
import re
import tomllib
import unittest
from pathlib import Path

from packaging.requirements import Requirement

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

# The only agno extras this example may pull in. Adding a provider extra
# (anthropic, aws, ...) must be a deliberate change to this set.
EXPECTED_AGNO_EXTRAS = frozenset({"agui", "google", "os"})
EXPECTED_PROVIDER_MODULES = frozenset({"agno.models.openai"})

SELECTION_MARKER = re.compile(r"\{\s*supportsReasoningModelSelection\s*\?\s*\(")
FALLBACK_MARKER = re.compile(r"\s*:\s*\(")


def _source_page() -> str:
    return DOJO_REASONING_PAGE.read_text()


def _generated_page() -> str:
    files = json.loads(DOJO_FILES.read_text())
    entries = files[AGNO_REASONING_KEY]
    for entry in entries:
        if entry["name"] == "page.tsx":
            return entry["content"]
    raise AssertionError(f"{AGNO_REASONING_KEY} is missing page.tsx")


def _selection_branch(page: str) -> str:
    """Return the JSX rendered only when supportsReasoningModelSelection is true.

    Slices from the conditional's opening paren to its balanced closing paren
    and requires that paren to be followed by the `) : (` fallback branch.
    """
    match = SELECTION_MARKER.search(page)
    if match is None:
        raise AssertionError("page has no supportsReasoningModelSelection conditional")

    depth = 1
    for index in range(match.end(), len(page)):
        char = page[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                if FALLBACK_MARKER.match(page, index + 1) is None:
                    raise AssertionError(
                        "supportsReasoningModelSelection conditional has no ) : ( branch"
                    )
                return page[match.end() : index]

    raise AssertionError("supportsReasoningModelSelection conditional is unbalanced")


def _call_name(call: ast.Call | None) -> str | None:
    if call is None:
        return None
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _module_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
    return assignments


def _agent_call(tree: ast.Module) -> ast.Call:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "Agent"
    ]
    if len(calls) != 1:
        raise AssertionError(
            f"expected exactly one Agent(...) call, found {len(calls)}"
        )
    return calls[0]


def _agent_keyword_call(agent_source: str, keyword_name: str) -> ast.Call | None:
    """Resolve an Agent(...) keyword to the call that builds it.

    Follows a bare name to its module-level assignment so a hoisted
    `reasoning = OpenAIResponses(...)` is inspected the same as an inline call.
    """
    tree = ast.parse(agent_source)
    assignments = _module_assignments(tree)
    for keyword in _agent_call(tree).keywords:
        if keyword.arg != keyword_name:
            continue
        value = keyword.value
        if isinstance(value, ast.Name):
            value = assignments.get(value.id)
        return value if isinstance(value, ast.Call) else None
    return None


def _constant_keyword(call: ast.Call | None, name: str) -> object:
    if call is None:
        return None
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _provider_modules(agent_source: str) -> set[str]:
    """Collect every agno.models.<provider> module the source imports."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(agent_source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "agno.models":
                modules.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif node.module.startswith("agno.models."):
                modules.add(".".join(node.module.split(".")[:3]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agno.models."):
                    modules.add(".".join(alias.name.split(".")[:3]))
    return modules


def _agno_extras(pyproject_text: str) -> set[str]:
    dependencies = tomllib.loads(pyproject_text)["project"]["dependencies"]
    for dependency in dependencies:
        requirement = Requirement(dependency)
        if requirement.name == "agno":
            return set(requirement.extras)
    raise AssertionError("pyproject.toml does not depend on agno")


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
                selection_branch = _selection_branch(content)
                self.assertIn('handleModelChange("OpenAI")', selection_branch)
                self.assertIn('handleModelChange("Anthropic")', selection_branch)
                self.assertIn('handleModelChange("Gemini")', selection_branch)
                self.assertEqual(content.count("handleModelChange("), 3)

        self.assertEqual(_generated_page(), _source_page())

    def test_agno_reasoning_backend_stays_on_openai_responses(self) -> None:
        agent_source = AGNO_REASONING_AGENT.read_text()

        self.assertEqual(_provider_modules(agent_source), EXPECTED_PROVIDER_MODULES)
        self.assertEqual(
            _call_name(_agent_keyword_call(agent_source, "db")), "InMemoryDb"
        )

        model = _agent_keyword_call(agent_source, "model")
        self.assertEqual(_call_name(model), "OpenAIResponses")
        self.assertEqual(_constant_keyword(model, "id"), "o4-mini")

        reasoning_model = _agent_keyword_call(agent_source, "reasoning_model")
        self.assertEqual(_call_name(reasoning_model), "OpenAIResponses")
        self.assertEqual(_constant_keyword(reasoning_model, "id"), "o4-mini")
        self.assertEqual(_constant_keyword(reasoning_model, "reasoning_effort"), "high")
        self.assertEqual(
            _constant_keyword(reasoning_model, "reasoning_summary"), "auto"
        )

    def test_agno_dependency_extras_pull_no_other_providers(self) -> None:
        self.assertEqual(_agno_extras(AGNO_PYPROJECT.read_text()), EXPECTED_AGNO_EXTRAS)

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

        reasoning_model = _agent_keyword_call(regressed_source, "reasoning_model")
        self.assertNotEqual(_constant_keyword(reasoning_model, "id"), "o4-mini")

    def test_reasoning_summary_must_sit_on_the_reasoning_model(self) -> None:
        regressed_source = """
agent = Agent(
    model=OpenAIResponses(id="o4-mini", reasoning_summary="auto"),
    reasoning_model=OpenAIResponses(id="o4-mini"),
)
"""

        reasoning_model = _agent_keyword_call(regressed_source, "reasoning_model")
        self.assertIsNone(_constant_keyword(reasoning_model, "reasoning_summary"))

    def test_hoisted_reasoning_model_is_resolved(self) -> None:
        hoisted_source = """
reasoning = OpenAIResponses(id="o4-mini", reasoning_summary="auto")
agent = Agent(model=OpenAIResponses(id="o4-mini"), reasoning_model=reasoning)
"""

        reasoning_model = _agent_keyword_call(hoisted_source, "reasoning_model")
        self.assertEqual(_call_name(reasoning_model), "OpenAIResponses")
        self.assertEqual(_constant_keyword(reasoning_model, "id"), "o4-mini")
        self.assertEqual(
            _constant_keyword(reasoning_model, "reasoning_summary"), "auto"
        )

    def test_provider_extra_on_agno_dependency_is_detected(self) -> None:
        regressed_pyproject = """
[project]
dependencies = ["agno[agui, anthropic, google, os] >=3.0.4,<4", "openai>=1.99.1"]
"""

        self.assertNotEqual(_agno_extras(regressed_pyproject), EXPECTED_AGNO_EXTRAS)

    def test_other_provider_import_is_detected(self) -> None:
        regressed_source = """
from agno.models.openai import OpenAIResponses
from agno.models.anthropic import Claude
"""

        self.assertNotEqual(
            _provider_modules(regressed_source), EXPECTED_PROVIDER_MODULES
        )

    def test_dropdown_items_outside_the_selection_branch_are_not_counted(self) -> None:
        regressed_page = """
{supportsReasoningModelSelection ? (
  <Button onClick={() => setOpen(true)}>{selectedModel}</Button>
) : (
  <DropdownMenuItem onClick={() => handleModelChange("Anthropic")} />
)}
<DropdownMenuItem onClick={() => handleModelChange("Gemini")} />
"""

        selection_branch = _selection_branch(regressed_page)
        self.assertNotIn('handleModelChange("Anthropic")', selection_branch)
        self.assertNotIn('handleModelChange("Gemini")', selection_branch)


if __name__ == "__main__":
    unittest.main()
