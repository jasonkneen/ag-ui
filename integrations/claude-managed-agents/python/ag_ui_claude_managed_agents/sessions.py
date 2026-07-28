"""Thread-to-session stores."""

from copy import deepcopy

from .types import SessionRecord


class InMemorySessionStore:
    """In-memory thread-to-session store. Mappings are lost on restart."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}

    def get(self, thread_id: str) -> SessionRecord | None:
        record = self._records.get(thread_id)
        # Hand out a copy: the agent mutates records in place between
        # persists, so an aliased record would make an unpersisted mutation
        # indistinguishable from a persisted one — and a dropped write would
        # only surface against a real out-of-process store.
        return None if record is None else deepcopy(record)

    def set(self, thread_id: str, record: SessionRecord) -> None:
        self._records[thread_id] = deepcopy(record)

    def delete(self, thread_id: str) -> None:
        self._records.pop(thread_id, None)
