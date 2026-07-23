"""Thread-to-session stores."""

from .types import SessionRecord


class InMemorySessionStore:
    """In-memory thread-to-session store. Mappings are lost on restart."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}

    def get(self, thread_id: str) -> SessionRecord | None:
        return self._records.get(thread_id)

    def set(self, thread_id: str, record: SessionRecord) -> None:
        self._records[thread_id] = record

    def delete(self, thread_id: str) -> None:
        self._records.pop(thread_id, None)
