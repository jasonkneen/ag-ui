from __future__ import annotations

import unittest

from server.api.agentic_generative_ui import agent as agentic_generative_ui_agent

STEP_STATUSES = ("pending", "in_progress", "completed")


def _prompt() -> str:
    instructions = agentic_generative_ui_agent.instructions
    if isinstance(instructions, str):
        return instructions
    return "\n".join(instructions)


class DojoAgenticGenerativeUiContractTests(unittest.TestCase):
    def test_agno_agentic_generative_ui_streams_steps_through_agentic_state(
        self,
    ) -> None:
        self.assertEqual(agentic_generative_ui_agent.session_state, {"steps": []})
        self.assertTrue(agentic_generative_ui_agent.add_session_state_to_context)
        self.assertTrue(agentic_generative_ui_agent.enable_agentic_state)
        self.assertTrue(agentic_generative_ui_agent.markdown)

    def test_agno_agentic_generative_ui_requires_the_complete_steps_list(
        self,
    ) -> None:
        prompt = _prompt()

        self.assertIn(
            'session_state_updates must have exactly one top-level key: "steps"',
            prompt,
        )
        self.assertIn("complete list of step objects", prompt)
        self.assertIn("Never send a partial steps list", prompt)
        self.assertIn("Copy every unchanged step", prompt)

    def test_agno_agentic_generative_ui_names_every_dojo_step_status(self) -> None:
        prompt = _prompt()

        self.assertIn('"description"', prompt)
        self.assertIn('"status"', prompt)
        for status in STEP_STATUSES:
            with self.subTest(status=status):
                self.assertIn(f'"{status}"', prompt)

    def test_agno_agentic_generative_ui_marks_the_active_step_in_progress(
        self,
    ) -> None:
        prompt = _prompt()

        self.assertIn('First write every step as "pending"', prompt)
        self.assertIn(
            'write the full list with the current step set to "in_progress"', prompt
        )
        self.assertIn(
            'set that step to "completed" and the following step to "in_progress"',
            prompt,
        )
        self.assertIn('the last step set to "completed"', prompt)
        self.assertNotIn(
            'write the full list again with that step set to "completed"', prompt
        )
        self.assertIn("exactly 10 steps", prompt)
        self.assertIn("gerund form", prompt)
        self.assertIn("Do NOT repeat the steps", prompt)


if __name__ == "__main__":
    unittest.main()
