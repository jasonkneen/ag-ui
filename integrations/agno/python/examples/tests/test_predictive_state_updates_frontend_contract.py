from __future__ import annotations

import json
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


class PredictiveStateUpdatesFrontendContractTests(unittest.TestCase):
    def test_write_document_confirm_commits_reviewed_tool_document(self) -> None:
        expected_commit = (
            'const acceptedDocument = args.document || "";\n'
            "                editor?.commands.setContent(fromMarkdown(acceptedDocument));\n"
            "                setCurrentDocument(acceptedDocument);\n"
            "                setAgentState({ document: acceptedDocument });"
        )
        stale_commit = (
            'editor?.commands.setContent(fromMarkdown(agentState?.document || ""));\n'
            '                setCurrentDocument(agentState?.document || "");\n'
            '                setAgentState({ document: agentState?.document || "" });'
        )

        for label, content in {
            "source": _source_page(),
            "generated": _generated_page(),
        }.items():
            with self.subTest(label=label):
                write_document_block = _write_document_hitl_block(content)
                self.assertIn(expected_commit, write_document_block)
                self.assertNotIn(stale_commit, write_document_block)

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
                start = content.index('name: "confirm_changes"')
                end = content.index("// Action to write the document.", start)
                confirm_changes_block = content[start:end]
                for fragment in state_based_fragments:
                    self.assertIn(fragment, confirm_changes_block)


if __name__ == "__main__":
    unittest.main()
