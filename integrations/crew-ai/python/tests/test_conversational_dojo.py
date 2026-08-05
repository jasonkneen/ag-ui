"""CrewAI Conversational Flow dojo matrix parity."""

import importlib

import pytest


EXPECTED_CONVERSATIONAL_FEATURES = {
    "agentic_chat",
    "agentic_chat_reasoning",
    "agentic_chat_multimodal",
    "backend_tool_rendering",
    "interrupt",
    "human_in_the_loop",
    "agentic_generative_ui",
    "predictive_state_updates",
    "shared_state",
    "tool_based_generative_ui",
    "subgraphs",
    "a2ui_dynamic_schema",
    "a2ui_recovery",
    "a2ui_fixed_schema",
}


def _conversational_examples():
    try:
        return importlib.import_module("ag_ui_crewai.examples.conversational")
    except ModuleNotFoundError:
        pytest.fail("conversational dojo examples are not implemented")


def test_conversational_example_matrix_matches_regular_flows():
    examples = _conversational_examples()

    assert set(examples.CONVERSATIONAL_FLOW_TYPES) == EXPECTED_CONVERSATIONAL_FEATURES
    assert "crew_chat" not in examples.CONVERSATIONAL_FLOW_TYPES
    for flow_type in examples.CONVERSATIONAL_FLOW_TYPES.values():
        assert flow_type.conversational is True
        assert flow_type.conversational_config.defer_trace_finalization is False


def test_dojo_registers_a_conversational_route_for_every_feature():
    dojo = importlib.import_module("ag_ui_crewai.dojo")
    paths = {route.path for route in dojo.app.routes}

    assert {
        f"/conversational_flows/{feature}"
        for feature in EXPECTED_CONVERSATIONAL_FEATURES
    }.issubset(paths)


def test_conversational_examples_keep_litellm_compatible_message_dicts():
    examples = _conversational_examples()
    flow = examples.CONVERSATIONAL_FLOW_TYPES["agentic_chat"]()

    flow.receive_user_message("hello")

    assert flow.state.current_user_message == "hello"
    assert flow.state.messages[-1] == {"role": "user", "content": "hello"}
