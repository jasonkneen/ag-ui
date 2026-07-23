import { vi } from "vitest";

/** A scripted stand-in for the Anthropic client's managed-agents surface. */
export interface FakeClientOptions {
  /** Events yielded by each successive `events.stream` call. */
  streams?: unknown[][];
  agentTools?: unknown[];
  sessionId?: string;
}

export function createFakeClient(options: FakeClientOptions = {}) {
  const streams = [...(options.streams ?? [])];
  const sent: { sessionId: string; events: unknown[] }[] = [];

  const stream = vi.fn(async (_sessionId: string) => {
    const events = streams.shift() ?? [];
    const controller = new AbortController();
    return {
      controller,
      async *[Symbol.asyncIterator]() {
        for (const event of events) {
          if (controller.signal.aborted) return;
          yield event;
        }
      },
    };
  });

  const send = vi.fn(async (sessionId: string, params: { events: unknown[] }) => {
    sent.push({ sessionId, events: params.events });
    return { data: params.events.map((event, i) => ({ ...(event as object), id: `sent_${sent.length}_${i}` })) };
  });

  const create = vi.fn(async () => ({ id: options.sessionId ?? "sesn_1" }));
  const update = vi.fn(async () => ({}));
  const retrieve = vi.fn(async () => ({ tools: options.agentTools ?? [{ type: "agent_toolset_20260401", configs: [], default_config: {} }] }));

  const client = {
    beta: {
      agents: { retrieve },
      sessions: {
        create,
        update,
        events: { stream, send },
      },
    },
  };

  return { client: client as any, sent, spies: { stream, send, create, update, retrieve } };
}
