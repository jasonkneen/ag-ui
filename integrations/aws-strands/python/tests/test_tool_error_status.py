from ag_ui.core import ToolMessage

from ag_ui_strands.agent import _build_strands_history, _build_snapshot_messages


def _tool_message(**overrides):
    fields = dict(
        id="t1",
        role="tool",
        content="Tool failed: invalid id",
        tool_call_id="tc1",
    )
    fields.update(overrides)
    return ToolMessage(**fields)


class TestBedrockToolResultStatus:
    def test_error_maps_onto_bedrock_status(self):
        # A client-reported tool failure must reach the model as an error, not a
        # silent success -- AG-UI's ToolMessage.error sets Bedrock's toolResult status.
        history = _build_strands_history([_tool_message(error="invalid id")])
        tool_result = history[0]["content"][0]["toolResult"]
        assert tool_result["status"] == "error"

    def test_defaults_to_success_without_error(self):
        history = _build_strands_history([_tool_message(content="42")])
        tool_result = history[0]["content"][0]["toolResult"]
        assert tool_result["status"] == "success"

    def test_blank_failed_result_uses_error_diagnostic_as_content(self):
        history = _build_strands_history(
            [_tool_message(content="", error="boom")]
        )

        assert history[0]["content"][0]["toolResult"] == {
            "toolUseId": "tc1",
            "content": [{"text": "boom"}],
            "status": "error",
        }

    def test_explicit_failed_result_content_wins_over_error_diagnostic(self):
        history = _build_strands_history(
            [_tool_message(content="client failure details", error="boom")]
        )

        assert history[0]["content"][0]["toolResult"] == {
            "toolUseId": "tc1",
            "content": [{"text": "client failure details"}],
            "status": "error",
        }

    def test_empty_error_value_still_maps_to_failure(self):
        history = _build_strands_history(
            [_tool_message(content="", error="")]
        )

        assert history[0]["content"][0]["toolResult"] == {
            "toolUseId": "tc1",
            "content": [{"text": ""}],
            "status": "error",
        }

    def test_invalid_utf8_failure_diagnostic_is_provider_safe(self):
        history = _build_strands_history(
            [_tool_message(content="", error="boom\udcff")]
        )

        assert history[0]["content"][0]["toolResult"] == {
            "toolUseId": "tc1",
            "content": [{"text": "boom\\udcff"}],
            "status": "error",
        }

    def test_successful_empty_result_remains_empty(self):
        history = _build_strands_history([_tool_message(content="")])

        assert history[0]["content"][0]["toolResult"] == {
            "toolUseId": "tc1",
            "content": [{"text": ""}],
            "status": "success",
        }


class TestSnapshotPreservesClientFields:
    def test_preserves_error_and_encrypted_value(self):
        # _build_snapshot_messages rebuilds the client's own message; it must not
        # drop the client's error / encrypted_value on the snapshot echo.
        snapshot = _build_snapshot_messages(
            [_tool_message(error="invalid id", encrypted_value="enc-abc")]
        )
        assert snapshot[0].error == "invalid id"
        assert snapshot[0].encrypted_value == "enc-abc"

    def test_leaves_error_unset_when_absent(self):
        snapshot = _build_snapshot_messages([_tool_message(content="42")])
        assert snapshot[0].error is None
