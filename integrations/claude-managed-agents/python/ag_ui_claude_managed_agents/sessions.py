"""Thread-to-session stores."""

from copy import deepcopy

from .types import SessionRecord


class InMemorySessionStore:
    """In-memory thread-to-session store. Mappings are lost on restart."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}

    def get(self, key: str) -> SessionRecord | None:
        record = self._records.get(key)
        # Hand out a copy: the agent mutates records in place between
        # persists, so an aliased record would make an unpersisted mutation
        # indistinguishable from a persisted one — and a dropped write would
        # only surface against a real out-of-process store.
        return None if record is None else deepcopy(record)

    def set(self, key: str, record: SessionRecord) -> None:
        self._records[key] = deepcopy(record)

    def delete(self, key: str) -> None:
        self._records.pop(key, None)
