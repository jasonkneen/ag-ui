"""Unit tests for the tool-definition and text helpers."""

from ag_ui_claude_managed_agents import (
    BackendTool,
    custom_tool_from,
    normalize_tool_name,
)
from ag_ui_claude_managed_agents.text import describe_tool_result, text_of


def test_normalize_tool_name_keeps_valid_names():
    assert normalize_tool_name("show_chart") == "show_chart"
    assert normalize_tool_name("Get-Weather_2") == "Get-Weather_2"


def test_normalize_tool_name_replaces_invalid_characters_and_truncates():
    assert normalize_tool_name("search web!") == "search_web_"
    assert normalize_tool_name("x" * 200) == "x" * 128
    assert normalize_tool_name("") == "tool"


def test_custom_tool_from_builds_input_schema_and_default_description():
    tool = BackendTool(
        name="ping",
        description="",
        parameters={"properties": {"a": {"type": "string"}}},
        handler=lambda _i: "pong",
    )
    assert custom_tool_from(tool) == {
        "type": "custom",
        "name": "ping",
        "description": "Tool ping",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": [],
        },
    }


def test_custom_tool_from_handles_missing_parameters():
    tool = BackendTool(
        name="ping", description="Ping", parameters=None, handler=lambda _i: "pong"
    )  # type: ignore[arg-type]
    assert custom_tool_from(tool)["input_schema"] == {
        "type": "object",
        "properties": {},
    }


def test_text_of_joins_text_blocks_only():
    assert (
        text_of(
            [
                {"type": "text", "text": "a"},
                {"type": "image"},
                {"type": "text", "text": "b"},
            ]
        )
        == "ab"
    )
    assert text_of(None) == ""


def test_describe_tool_result_decodes_entities_and_summarizes_blocks():
    described = describe_tool_result(
        [
            {"type": "text", "text": "5 &lt; 6 &amp;&amp; &#x1F600; &#65;"},
            {
                "type": "search_result",
                "title": "T",
                "source": "https://x",
                "content": [{"type": "text", "text": "inner"}],
            },
            {"type": "document"},
        ]
    )
    assert (
        described
        == "5 < 6 && \U0001f600 A\n[search result] T — https://x\ninner\n[document]"
    )
