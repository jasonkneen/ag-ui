import type { SessionRecord, SessionStore } from "./types";

/**
 * A defensive copy of a record.
 *
 * The store must not hand out a reference to the record it holds: the agent
 * mutates records in place between persists, so an aliased record would make
 * an unpersisted mutation indistinguishable from a persisted one — and a
 * dropped write would only surface against a real out-of-process store.
 */
const copy = (record: SessionRecord): SessionRecord => structuredClone(record);

/** In-memory thread↔session store. Mappings are lost on restart. */
export class InMemorySessionStore implements SessionStore {
  private records = new Map<string, SessionRecord>();

  get(key: string): SessionRecord | undefined {
    const record = this.records.get(key);
    return record === undefined ? undefined : copy(record);
  }

  set(key: string, record: SessionRecord): void {
    this.records.set(key, copy(record));
  }

  delete(key: string): void {
    this.records.delete(key);
  }
}
