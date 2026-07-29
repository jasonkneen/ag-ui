"""An example demonstrating async human-in-the-loop via flow interrupts.

A ``@human_feedback`` method whose provider is the bridge's
:data:`agui_feedback_provider` PAUSES the flow: the run terminates with an AG-UI
interrupt (``RUN_FINISHED.outcome`` / legacy ``on_interrupt``), and the next
request carrying ``RunAgentInput.resume[]`` resumes it via ``Flow.from_pending``
+ ``resume_async``. This is distinct from the tool-based HITL demo
(``human_in_the_loop.py``), which never pauses the flow.
"""

from typing import List

from crewai.flow.flow import Flow, start, listen
from crewai.flow import human_feedback
from pydantic import BaseModel

from ..sdk import CopilotKitState
from .._hitl import agui_feedback_provider


class PlanStep(BaseModel):
    description: str


class AgentState(CopilotKitState):
    """Flow state: the proposed plan plus the applied result."""

    plan: List[PlanStep] = []
    result: str = ""


class InterruptFlow(Flow[AgentState]):
    """A flow that proposes a plan, pauses for human approval, then applies it."""

    @start()
    @human_feedback(
        message="Review the proposed plan and approve or request changes.",
        provider=agui_feedback_provider,
    )
    def propose_plan(self):
        """Propose a plan from the latest user message, then pause for review.

        The method's return value is what the human reviews. Raising
        ``HumanFeedbackPending`` (done by the provider) pauses the run here; the
        framework persists the pending state so a later resume continues at
        ``apply_plan``.
        """
        last_user = next(
            (
                m
                for m in reversed(self.state.messages)
                if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
                == "user"
            ),
            None,
        )
        topic = ""
        if last_user is not None:
            topic = (
                last_user.get("content")
                if isinstance(last_user, dict)
                else getattr(last_user, "content", "")
            ) or ""
        steps = [
            PlanStep(description=f"Understand the request: {topic}".strip()),
            PlanStep(description="Draft an approach"),
            PlanStep(description="Execute and summarize"),
        ]
        self.state.plan = steps
        return {"plan": [s.model_dump() for s in steps]}

    @listen(propose_plan)
    def apply_plan(self, feedback):
        """Apply the plan using the human's feedback (delivered on resume)."""
        answer = getattr(feedback, "feedback", feedback)
        self.state.result = f"Applied {len(self.state.plan)} step plan with feedback: {answer}"
        return {"result": self.state.result}
