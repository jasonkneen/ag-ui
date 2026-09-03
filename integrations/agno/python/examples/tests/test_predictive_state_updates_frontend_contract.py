from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[3]
DOJO_PREDICTIVE_STATE_PAGE = (
    REPO_ROOT
    / "apps"
    / "dojo"
    / "src"
    / "app"
    / "[integrationId]"
    / "feature"
    / "(v2)"
    / "predictive_state_updates"
    / "page.tsx"
)
DOJO_FILES = REPO_ROOT / "apps" / "dojo" / "src" / "files.json"
AGNO_PREDICTIVE_STATE_KEY = "agno::predictive_state_updates"


def _source_page() -> str:
    return DOJO_PREDICTIVE_STATE_PAGE.read_text()


def _generated_page() -> str:
    files = json.loads(DOJO_FILES.read_text())
    entries = files[AGNO_PREDICTIVE_STATE_KEY]
    for entry in entries:
        if entry["name"] == "page.tsx":
            return entry["content"]
    raise AssertionError(f"{AGNO_PREDICTIVE_STATE_KEY} is missing page.tsx")


def _write_document_hitl_block(content: str) -> str:
    start = content.index('name: "write_document"')
    end = content.index("return <></>;", start)
    return content[start:end]


def _confirm_changes_block(content: str) -> str:
    start = content.index('name: "confirm_changes"')
    end = content.index("// Action to write the document.", start)
    return content[start:end]


def _on_confirm_body(block: str) -> str:
    """Return the body of the onConfirm arrow, found by brace matching."""
    marker = "onConfirm={() => {"
    start = block.index(marker) + len(marker)
    depth = 1
    for index in range(start, len(block)):
        char = block[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return block[start:index]
    raise AssertionError("unterminated onConfirm body")


def _normalize(code: str) -> str:
    """Strip whitespace and the trailing commas Prettier adds when wrapping."""
    return re.sub(r",(?=[)\]}])", "", re.sub(r"\s+", "", code))


def _assert_contains_code(test: unittest.TestCase, haystack: str, needle: str) -> None:
    test.assertIn(_normalize(needle), _normalize(haystack), needle)


class PredictiveStateUpdatesFrontendContractTests(unittest.TestCase):
    def test_write_document_confirm_commits_reviewed_tool_document(self) -> None:
        expected_commit = """
            const acceptedDocument = args.document || "";
            editor?.commands.setContent(fromMarkdown(acceptedDocument));
            setCurrentDocument(acceptedDocument);
            setAgentState({ document: acceptedDocument });
        """

        for label, content in {
            "source": _source_page(),
            "generated": _generated_page(),
        }.items():
            with self.subTest(label=label):
                on_confirm = _on_confirm_body(_write_document_hitl_block(content))
                _assert_contains_code(self, on_confirm, expected_commit)
                # Confirm must commit the reviewed tool document, never
                # whatever agent state happens to hold at that moment.
                self.assertNotIn("agentState", on_confirm)

        self.assertEqual(_generated_page(), _source_page())

    def test_legacy_confirm_changes_stays_state_based(self) -> None:
        state_based_fragments = (
            'fromMarkdown(agentState?.document || "")',
            'setCurrentDocument(agentState?.document || "")',
            'setAgentState({ document: agentState?.document || "" })',
        )

        for label, content in {
            "source": _source_page(),
            "generated": _generated_page(),
        }.items():
            with self.subTest(label=label):
                on_confirm = _on_confirm_body(_confirm_changes_block(content))
                for fragment in state_based_fragments:
                    _assert_contains_code(self, on_confirm, fragment)


if __name__ == "__main__":
    unittest.main()
