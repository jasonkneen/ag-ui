"""CrewAI Conversational Flow dojo matrix parity."""

import importlib
from collections.abc import Mapping

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


def test_conversational_examples_preserve_every_regular_flow_method():
    examples = _conversational_examples()

    for feature, flow_type in examples.CONVERSATIONAL_FLOW_TYPES.items():
        regular_flow_type = flow_type.__mro__[2]
        regular_methods = set(regular_flow_type.flow_definition().methods)
        conversational_methods = set(flow_type.flow_definition().methods)

        assert regular_methods <= conversational_methods, feature


def test_regular_end_methods_do_not_trigger_builtin_conversation_termination():
    examples = _conversational_examples()

    for feature, flow_type in examples.CONVERSATIONAL_FLOW_TYPES.items():
        end_definition = flow_type.flow_definition().methods["end_conversation"]

        assert end_definition.listen != "end", feature


@pytest.mark.parametrize(
    "feature",
    ["a2ui_dynamic_schema", "a2ui_recovery", "a2ui_fixed_schema"],
)
def test_untyped_mapping_flows_keep_mapping_compatible_state(feature):
    examples = _conversational_examples()
    state = examples.CONVERSATIONAL_FLOW_TYPES[feature]().state

    assert state.get("copilotkit") == {"actions": []}
    assert state["messages"] == []


def test_untyped_mapping_flows_preserve_a2ui_runtime_input():
    from ag_ui_crewai._conversation import (
        ConversationalTurn,
        hydrate_conversational_flow,
    )

    examples = _conversational_examples()
    flow = examples.CONVERSATIONAL_FLOW_TYPES["a2ui_recovery"]()
    hydrate_conversational_flow(
        flow,
        {"ag-ui": {"inject_a2ui_tool": True}},
        ConversationalTurn(message="compare", history=[], current_media=[]),
    )

    assert isinstance(flow.state, Mapping)
    assert flow.state.get("ag-ui") == {"inject_a2ui_tool": True}


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
