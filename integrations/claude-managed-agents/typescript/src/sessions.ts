import type { SessionRecord, SessionStore } from "./types";

/** In-memory thread↔session store. Mappings are lost on restart. */
export class InMemorySessionStore implements SessionStore {
  private records = new Map<string, SessionRecord>();

  get(threadId: string): SessionRecord | undefined {
    return this.records.get(threadId);
  }

  set(threadId: string, record: SessionRecord): void {
    this.records.set(threadId, record);
  }

  delete(threadId: string): void {
    this.records.delete(threadId);
  }
}
