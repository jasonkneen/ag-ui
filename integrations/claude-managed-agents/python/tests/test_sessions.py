"""The default store must not alias the records it holds."""

from ag_ui_claude_managed_agents.sessions import InMemorySessionStore
from ag_ui_claude_managed_agents.types import SessionRecord


def _record() -> SessionRecord:
    return SessionRecord(
        session_id="sesn_1",
        tool_names=["a"],
        pending_client_tool_use_ids=["ctu_1"],
    )


def test_does_not_alias_the_record_it_was_given() -> None:
    store = InMemorySessionStore()
    original = _record()
    store.set("t", original)

    original.pending_client_tool_use_ids.append("ctu_2")
    original.session_id = "sesn_mutated"

    read = store.get("t")
    assert read is not None
    assert read.session_id == "sesn_1"
    assert read.pending_client_tool_use_ids == ["ctu_1"]


def test_does_not_alias_the_record_it_hands_out() -> None:
    store = InMemorySessionStore()
    store.set("t", _record())

    # The agent mutates records in place between persists; those mutations
    # must not be visible until they are actually written back.
    first = store.get("t")
    assert first is not None
    first.pending_client_tool_use_ids.append("ctu_2")
    first.last_user_message_id = "m_unpersisted"

    second = store.get("t")
    assert second is not None
    assert second.pending_client_tool_use_ids == ["ctu_1"]
    assert second.last_user_message_id is None


def test_unknown_and_deleted_threads_read_as_none() -> None:
    store = InMemorySessionStore()
    assert store.get("nope") is None
    store.set("t", _record())
    store.delete("t")
    assert store.get("t") is None
