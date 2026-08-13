"""Tests for reconciling frontend-tool results into a Strands SessionManager.

Frontend (proxy) tools return a ``"Forwarded to client"`` placeholder result
server-side; the real result only arrives on the next run inside
``RunAgentInput.messages``. These tests exercise the helper that overwrites the
persisted placeholder ``toolResult`` with the real client result so the session
store (and the agent's in-memory history) hold the true value.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from strands.session.file_session_manager import FileSessionManager
from strands.types.session import SessionAgent, SessionMessage

from ag_ui_strands import session_reconcile
from ag_ui_strands.session_reconcile import (
    _FrontendToolResult,
    has_placeholder_results,
    reconcile_frontend_tool_results,
    resolve_native_ids,
)

PLACEHOLDER = "Forwarded to client"


def _client_result(content, status="success"):
    return _FrontendToolResult(content=content, status=status)


def _make_session(tmp_path, session_id="s1", agent_id="default"):
    sm = FileSessionManager(session_id=session_id, storage_dir=str(tmp_path))
    sm.session_repository.create_agent(
        session_id,
        SessionAgent(agent_id=agent_id, state={}, conversation_manager_state={}),
    )
    return sm


def _seed(sm, agent_id, index, message):
    sm.session_repository.create_message(
        sm.session_id, agent_id, SessionMessage(message=message, message_id=index)
    )


def _tool_result_block(tool_use_id, text):
    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": text}],
        }
    }


def test_has_active_proxy_placeholder_requires_active_state_and_exact_reserved_result():
    exact_result = {
        "toolUseId": "native-frontend",
        "status": "success",
        "content": [{"text": PLACEHOLDER}],
    }
    frontend_meta = {"native-frontend": {"is_frontend": True}}

    assert not session_reconcile.has_active_proxy_placeholder(
        SimpleNamespace(
            _interrupt_state=SimpleNamespace(
                activated=False,
                context={"tool_results": [exact_result]},
            )
        ),
        frontend_meta,
    )
    assert not session_reconcile.has_active_proxy_placeholder(
        SimpleNamespace(), frontend_meta
    )
    assert not session_reconcile.has_active_proxy_placeholder(
        SimpleNamespace(
            _interrupt_state=SimpleNamespace(activated=True, context={})
        ),
        frontend_meta,
    )
    assert not session_reconcile.has_active_proxy_placeholder(
        SimpleNamespace(
            _interrupt_state=SimpleNamespace(
                activated=True,
                context={
                    "tool_results": [
                        {
                            **exact_result,
                            "content": [{"text": f"{PLACEHOLDER}."}],
                        }
                    ]
                },
            )
        ),
        frontend_meta,
    )
    structurally_non_exact_results = [
        {**exact_result, "status": "error"},
        {key: value for key, value in exact_result.items() if key != "toolUseId"},
        {key: value for key, value in exact_result.items() if key != "status"},
        {**exact_result, "toolUseId": ""},
        {
            **exact_result,
            "content": [
                {"text": PLACEHOLDER},
                {"text": "additional content"},
            ],
        },
        {
            **exact_result,
            "content": [{"text": PLACEHOLDER, "unexpected": True}],
        },
        {**exact_result, "unexpected": True},
    ]
    for parked_result in structurally_non_exact_results:
        assert not session_reconcile.has_active_proxy_placeholder(
            SimpleNamespace(
                _interrupt_state=SimpleNamespace(
                    activated=True,
                    context={"tool_results": [parked_result]},
                )
            ),
            frontend_meta,
        )
    non_frontend_meta = [
        {"native-frontend": {"is_frontend": False}},
        {"native-frontend": {}},
        {},
        {"native-frontend": {"is_frontend": "true"}},
        {"native-frontend": None},
    ]
    for tool_call_meta in non_frontend_meta:
        assert not session_reconcile.has_active_proxy_placeholder(
            SimpleNamespace(
                _interrupt_state=SimpleNamespace(
                    activated=True,
                    context={"tool_results": [exact_result]},
                )
            ),
            tool_call_meta,
        )
    assert session_reconcile.has_active_proxy_placeholder(
        SimpleNamespace(
            _interrupt_state=SimpleNamespace(
                activated=True,
                context={"tool_results": [exact_result]},
            )
        ),
        frontend_meta,
    )


def test_reconcile_overwrites_persisted_placeholder_in_store(tmp_path):
    sm = _make_session(tmp_path)
    agent_id = "default"
    _seed(sm, agent_id, 0, {"role": "user", "content": [{"text": "set it"}]})
    _seed(
        sm,
        agent_id,
        1,
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "tu-1", "name": "approve", "input": {}}}],
        },
    )
    _seed(
        sm,
        agent_id,
        2,
        {"role": "user", "content": [_tool_result_block("tu-1", PLACEHOLDER)]},
    )

    agent = SimpleNamespace(agent_id=agent_id, messages=[])
    corrected = reconcile_frontend_tool_results(
        sm, agent, {"tu-1": _client_result('{"approved": false}')}
    )

    assert corrected == {"tu-1"}
    persisted = sm.session_repository.list_messages(sm.session_id, agent_id)
    result_block = persisted[2].message["content"][0]["toolResult"]
    assert result_block["content"] == [{"text": '{"approved": false}'}]


def test_reconcile_returns_set_of_corrected_tool_use_ids(tmp_path):
    sm = _make_session(tmp_path)
    agent_id = "default"
    _seed(
        sm,
        agent_id,
        0,
        {"role": "user", "content": [_tool_result_block("tu-1", PLACEHOLDER)]},
    )
    agent = SimpleNamespace(
        agent_id=agent_id,
        messages=[{"role": "user", "content": [_tool_result_block("tu-1", PLACEHOLDER)]}],
    )

    corrected = reconcile_frontend_tool_results(
        sm,
        agent,
        {"tu-1": _client_result("R"), "tu-absent": _client_result("X")},
    )

    assert corrected == {"tu-1"}


def test_reconcile_corrects_in_memory_agent_messages(tmp_path):
    sm = _make_session(tmp_path)
    agent_id = "default"
    _seed(sm, agent_id, 0, {"role": "user", "content": [{"text": "set it"}]})
    _seed(
        sm,
        agent_id,
        1,
        {"role": "user", "content": [_tool_result_block("tu-1", PLACEHOLDER)]},
    )

    # The cached agent still holds the placeholder in its live message list.
    agent = SimpleNamespace(
        agent_id=agent_id,
        messages=[
            {"role": "user", "content": [{"text": "set it"}]},
            {"role": "user", "content": [_tool_result_block("tu-1", PLACEHOLDER)]},
        ],
    )

    reconcile_frontend_tool_results(
        sm, agent, {"tu-1": _client_result('{"approved": true}')}
    )

    in_memory = agent.messages[1]["content"][0]["toolResult"]
    assert in_memory["content"] == [{"text": '{"approved": true}'}]


@pytest.mark.parametrize(
    ("status", "content"),
    [
        pytest.param("success", "real client result", id="success"),
        pytest.param("error", "boom", id="failure-diagnostic"),
    ],
)
def test_reconcile_updates_status_and_content_in_every_active_copy(
    tmp_path, status, content
):
    sm = _make_session(tmp_path)
    agent_id = "default"
    _seed(
        sm,
        agent_id,
        0,
        {"role": "user", "content": [_tool_result_block("tu-1", PLACEHOLDER)]},
    )
    live_message = {
        "role": "user",
        "content": [_tool_result_block("tu-1", PLACEHOLDER)],
    }
    parked_result = _tool_result_block("tu-1", PLACEHOLDER)["toolResult"]
    agent = SimpleNamespace(
        agent_id=agent_id,
        messages=[live_message],
        _interrupt_state=SimpleNamespace(
            activated=True,
            context={"tool_results": [parked_result]},
        ),
    )
    result = _client_result(content, status=status)

    corrected = reconcile_frontend_tool_results(sm, agent, {"tu-1": result})

    assert corrected == {"tu-1"}
    expected = {
        "toolUseId": "tu-1",
        "status": status,
        "content": [{"text": content}],
    }
    persisted = sm.session_repository.list_messages(sm.session_id, agent_id)
    assert persisted[0].message["content"][0]["toolResult"] == expected
    assert live_message["content"][0]["toolResult"] == expected
    assert parked_result == expected


def test_active_interrupt_context_reconcile_raises_typed_error(tmp_path):
    sm = _make_session(tmp_path)
    agent_id = "default"
    _seed(
        sm,
        agent_id,
        0,
        {"role": "user", "content": [_tool_result_block("tu-1", PLACEHOLDER)]},
    )

    class ExplodingToolResults(list):
        def __iter__(self):
            raise RuntimeError("boom")

    parked_results = ExplodingToolResults(
        [
            {
                "toolUseId": "tu-1",
                "status": "success",
                "content": [{"text": PLACEHOLDER}],
            }
        ]
    )
    interrupt_state = SimpleNamespace(
        activated=True,
        context={"tool_results": parked_results},
    )
    agent = SimpleNamespace(
        agent_id=agent_id,
        messages=[],
        _interrupt_state=interrupt_state,
    )

    with pytest.raises(RuntimeError) as exc_info:
        reconcile_frontend_tool_results(
            sm,
            agent,
            {"tu-1": _client_result("real"), "tu-2": _client_result("other")},
        )

    assert isinstance(
        exc_info.value, session_reconcile.ActiveInterruptReconciliationError
    )
    assert exc_info.value.affected_native_ids == frozenset({"tu-1", "tu-2"})
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"
    persisted = sm.session_repository.list_messages(sm.session_id, agent_id)
    assert persisted[0].message["content"][0]["toolResult"]["content"] == [
        {"text": "real"}
    ]
    assert interrupt_state.activated
    assert interrupt_state.context["tool_results"] is parked_results


def test_reconcile_handles_parallel_tool_calls_in_one_message(tmp_path):
    sm = _make_session(tmp_path)
    agent_id = "default"
    _seed(
        sm,
        agent_id,
        0,
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "tu-1", "name": "a", "input": {}}},
                {"toolUse": {"toolUseId": "tu-2", "name": "b", "input": {}}},
            ],
        },
    )
    _seed(
        sm,
        agent_id,
        1,
        {
            "role": "user",
            "content": [
                _tool_result_block("tu-1", PLACEHOLDER),
                _tool_result_block("tu-2", PLACEHOLDER),
            ],
        },
    )

    agent = SimpleNamespace(agent_id=agent_id, messages=[])
    corrected = reconcile_frontend_tool_results(
        sm,
        agent,
        {"tu-1": _client_result("R1"), "tu-2": _client_result("R2")},
    )

    assert corrected == {"tu-1", "tu-2"}
    blocks = sm.session_repository.list_messages(sm.session_id, agent_id)[1].message[
        "content"
    ]
    assert blocks[0]["toolResult"]["content"] == [{"text": "R1"}]
    assert blocks[1]["toolResult"]["content"] == [{"text": "R2"}]


def test_resolve_maps_wire_id_to_native_id():
    # Client speaks the wire id; the store holds the native id. The durable
    # wire->native map (from session state) bridges them.
    resolved = resolve_native_ids(
        wire_to_native={"wire-1": "native-1", "wire-2": "native-2"},
        frontend_results={
            "wire-1": _client_result("R1"),
            "wire-2": _client_result("R2", status="error"),
        },
    )
    assert resolved == {
        "native-1": _client_result("R1"),
        "native-2": _client_result("R2", status="error"),
    }


def test_resolve_skips_results_absent_from_map():
    # A wire id not in the map (fresh session with no recorded mapping, or a
    # pruned entry) is dropped — the caller then degrades to the legacy path.
    resolved = resolve_native_ids(
        wire_to_native={"wire-1": "native-1"},
        frontend_results={
            "wire-1": _client_result("R1"),
            "wire-unknown": _client_result("R2"),
        },
    )
    assert resolved == {"native-1": _client_result("R1")}


def test_has_placeholder_results_detects_remaining_stub():
    assert has_placeholder_results(
        [{"role": "user", "content": [_tool_result_block("tu-1", PLACEHOLDER)]}]
    )
    assert not has_placeholder_results(
        [{"role": "user", "content": [_tool_result_block("tu-1", "real result")]}]
    )
    assert not has_placeholder_results([])


def test_has_placeholder_results_scopes_to_only_ids():
    messages = [
        {"role": "user", "content": [_tool_result_block("tu-old", PLACEHOLDER)]},
        {"role": "user", "content": [_tool_result_block("tu-new", "real")]},
    ]
    # A stale placeholder for tu-old must not count when scoped to tu-new.
    assert not has_placeholder_results(messages, only_ids={"tu-new"})
    assert has_placeholder_results(messages, only_ids={"tu-old"})


def test_reconcile_leaves_non_placeholder_results_untouched(tmp_path):
    sm = _make_session(tmp_path)
    agent_id = "default"
    _seed(
        sm,
        agent_id,
        0,
        {"role": "user", "content": [_tool_result_block("tu-1", "already the real result")]},
    )

    agent = SimpleNamespace(agent_id=agent_id, messages=[])
    corrected = reconcile_frontend_tool_results(
        sm, agent, {"tu-1": _client_result("SHOULD NOT APPLY")}
    )

    assert corrected == set()
    block = sm.session_repository.list_messages(sm.session_id, agent_id)[0].message[
        "content"
    ][0]["toolResult"]
    assert block["content"] == [{"text": "already the real result"}]
