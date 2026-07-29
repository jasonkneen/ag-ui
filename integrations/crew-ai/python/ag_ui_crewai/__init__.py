from .endpoint import (
  add_crewai_flow_fastapi_endpoint,
  add_crewai_crew_fastapi_endpoint,
  crewai_prepare_inputs,
)
from .crews import ChatWithCrewFlow
from ._hitl import AGUIFeedbackProvider, agui_feedback_provider
from .sdk import (
  CopilotKitState,
  StateItem,
  copilotkit_predict_state,
  copilotkit_emit_state,
  copilotkit_stream,
  copilotkit_exit,
)
# from .enterprise import CrewEnterpriseEventListener

# CREW_ENTERPRISE_EVENT_LISTENER = CrewEnterpriseEventListener()

# The Crew chat path was undiscoverable — the four symbols below (the
# crew-endpoint factory, its input-prep helper, the flow, and the exit
# signal) were absent from the package top level. Export them alongside
# the pre-existing flow-path surface.
__all__ = [
  "add_crewai_flow_fastapi_endpoint",
  "add_crewai_crew_fastapi_endpoint",
  "crewai_prepare_inputs",
  "ChatWithCrewFlow",
  "AGUIFeedbackProvider",
  "agui_feedback_provider",
  "CopilotKitState",
  "StateItem",
  "copilotkit_predict_state",
  "copilotkit_emit_state",
  "copilotkit_stream",
  "copilotkit_exit",
]
