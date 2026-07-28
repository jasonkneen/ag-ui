"""Unit tests for the tool-definition and text helpers."""

from types import SimpleNamespace

from ag_ui_claude_managed_agents import (
    BackendTool,
    custom_tool_from,
    normalize_tool_name,
)
from ag_ui_claude_managed_agents.constants import (
    SEARCH_RESULT_PREVIEW_CHARS,
    TOOL_DESCRIPTION_MAX_LENGTH,
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


def test_custom_tool_from_normalizes_name_and_caps_description() -> None:
    """The API caps descriptions; a long one must be truncated, not rejected."""
    tool = custom_tool_from(
        SimpleNamespace(
            name="show chart!", description="d" * (TOOL_DESCRIPTION_MAX_LENGTH + 50), parameters=None
        )
    )
    assert tool["name"] == "show_chart_"
    assert len(tool["description"]) == TOOL_DESCRIPTION_MAX_LENGTH


def test_describe_tool_result_previews_a_long_search_result_body() -> None:
    body = "x" * (SEARCH_RESULT_PREVIEW_CHARS + 200)
    out = describe_tool_result(
        [
            {
                "type": "search_result",
                "title": "Docs",
                "source": "https://example.com",
                "content": [{"type": "text", "text": body}],
            }
        ]
    )
    header, preview = out.split("\n", 1)
    assert "[search result] Docs" in header
    assert "https://example.com" in header
    # Only a readable prefix of the body is shown.
    assert preview == "x" * SEARCH_RESULT_PREVIEW_CHARS


def test_describe_tool_result_placeholders_unknown_blocks_and_handles_nothing() -> None:
    assert describe_tool_result([{"type": "image"}]) == "[image]"
    assert describe_tool_result([]) == ""
    assert describe_tool_result(None) == ""
