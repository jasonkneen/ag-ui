import {
  BaseEvent,
  AGUIEvent,
  EventSchemas,
  EventType,
  Message,
  RunFinishedOutcome,
} from "@ag-ui/core";
import * as protoEvents from "./generated/events";
import * as protoPatch from "./generated/patch";

/**
 * These converters run against values that have crossed a wire boundary, so
 * they accept `unknown` and narrow once rather than trusting a static type.
 */
type LooseRecord = Record<string, unknown>;

const asRecord = (value: unknown): LooseRecord | undefined =>
  value && typeof value === "object" ? (value as LooseRecord) : undefined;

const toProtoSource = (source: unknown): unknown => {
  const rec = asRecord(source);
  if (!rec) {
    return undefined;
  }

  if (rec.type === "data") {
    return {
      data: {
        value: rec.value,
        mimeType: rec.mimeType,
      },
    };
  }

  if (rec.type === "url") {
    return {
      url: {
        value: rec.value,
        mimeType: rec.mimeType,
      },
    };
  }

  return undefined;
};

const toProtoContentPart = (part: unknown): unknown => {
  const rec = asRecord(part);
  if (!rec) {
    return undefined;
  }

  switch (rec.type) {
    case "text":
      return {
        text: {
          text: rec.text,
        },
      };
    case "image":
      return {
        image: {
          source: toProtoSource(rec.source),
          metadata: rec.metadata,
        },
      };
    case "audio":
      return {
        audio: {
          source: toProtoSource(rec.source),
          metadata: rec.metadata,
        },
      };
    case "video":
      return {
        video: {
          source: toProtoSource(rec.source),
          metadata: rec.metadata,
        },
      };
    case "document":
      return {
        document: {
          source: toProtoSource(rec.source),
          metadata: rec.metadata,
        },
      };
    case "binary": {
      const source = rec.data
        ? { data: { value: rec.data, mimeType: rec.mimeType } }
        : rec.url
          ? { url: { value: rec.url, mimeType: rec.mimeType } }
          : rec.id
            ? { url: { value: rec.id, mimeType: rec.mimeType } }
            : undefined;

      if (!source) {
        return undefined;
      }

      return {
        document: {
          source,
          metadata: {
            legacyBinary: true,
            filename: rec.filename,
            id: rec.id,
          },
        },
      };
    }
    default:
      return undefined;
  }
};

const fromProtoSource = (source: unknown): unknown => {
  const rec = asRecord(source);
  if (!rec) {
    return undefined;
  }

  const data = asRecord(rec.data);
  if (data) {
    return {
      type: "data",
      value: data.value,
      mimeType: data.mimeType,
    };
  }

  const url = asRecord(rec.url);
  if (url) {
    return {
      type: "url",
      value: url.value,
      mimeType: url.mimeType,
    };
  }

  return undefined;
};

const fromProtoContentPart = (part: unknown): unknown => {
  const rec = asRecord(part);
  if (!rec) {
    return undefined;
  }

  const text = asRecord(rec.text);
  if (text) {
    return {
      type: "text",
      text: text.text,
    };
  }

  const image = asRecord(rec.image);
  if (image) {
    return {
      type: "image",
      source: fromProtoSource(image.source),
      metadata: image.metadata,
    };
  }

  const audio = asRecord(rec.audio);
  if (audio) {
    return {
      type: "audio",
      source: fromProtoSource(audio.source),
      metadata: audio.metadata,
    };
  }

  const video = asRecord(rec.video);
  if (video) {
    return {
      type: "video",
      source: fromProtoSource(video.source),
      metadata: video.metadata,
    };
  }

  const document = asRecord(rec.document);
  if (document) {
    return {
      type: "document",
      source: fromProtoSource(document.source),
      metadata: document.metadata,
    };
  }

  return undefined;
};

function toCamelCase(str: string): string {
  return str.toLowerCase().replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

/**
 * Encodes an event message to a protocol buffer binary format.
 */
export function encode(event: BaseEvent): Uint8Array {
  /**
   * In previous versions of AG-UI, we didn't really validate the events
   * against a schema. With stronger types for events and Zod schemas, we
   * can now validate.
   *
   * However, I don't want to break compatibility with existing clients
   * even if they are encoding invalid events. This surfaces a warning
   * to them in those situations.
   *
   * @author mikeryandev
   */
  let validatedEvent: AGUIEvent | BaseEvent;
  try {
    validatedEvent = EventSchemas.parse(event) as AGUIEvent;
  } catch (err) {
    console.warn(
      "[ag-ui][proto.encode] Malformed devent detected, falling back to unvalidated event",
      err,
      event,
    );
    validatedEvent = event;
  }
  const oneofField = toCamelCase(validatedEvent.type);
  const { type, timestamp, rawEvent, ...rest } = validatedEvent as AGUIEvent as LooseRecord;

  // since protobuf does not support optional arrays, we need to ensure that the toolCalls array is always present
  if (type === EventType.MESSAGES_SNAPSHOT && Array.isArray(rest.messages)) {
    rest.messages = (rest.messages as Message[]).map((message) => {
      const untypedMessage = message as LooseRecord;
      const normalizedMessage: LooseRecord = { ...untypedMessage, contentParts: [] };

      if (Array.isArray(untypedMessage.content)) {
        const contentParts = untypedMessage.content
          .map((part: unknown) => toProtoContentPart(part))
          .filter((part: unknown) => part !== undefined);

        normalizedMessage.contentParts = contentParts;
        normalizedMessage.content = undefined;
      }

      if (untypedMessage.toolCalls === undefined) {
        normalizedMessage.toolCalls = [];
      }

      return normalizedMessage;
    });
  }

  // RunFinishedEvent: flatten the nested `outcome` discriminated union into the
  // proto's `outcome` (string) and `interrupts` (repeated) fields. The wire
  // shape stays stable; the TS layer just exposes a richer object.
  if (type === EventType.RUN_FINISHED) {
    const outcome = rest.outcome as RunFinishedOutcome | undefined;
    if (outcome === undefined) {
      rest.outcome = "";
      rest.interrupts = [];
    } else if (outcome.type === "interrupt") {
      rest.outcome = "interrupt";
      rest.interrupts = outcome.interrupts;
    } else {
      rest.outcome = "success";
      rest.interrupts = [];
    }
  }

  // custom mapping for json patch operations
  if (type === EventType.STATE_DELTA && Array.isArray(rest.delta)) {
    rest.delta = (rest.delta as LooseRecord[]).map((operation) => ({
      ...operation,
      op: protoPatch.JsonPatchOperationType[
        String(operation.op).toUpperCase() as keyof typeof protoPatch.JsonPatchOperationType
      ],
    }));
  }

  const eventMessage = {
    [oneofField]: {
      baseEvent: {
        type: protoEvents.EventType[event.type as keyof typeof protoEvents.EventType],
        timestamp,
        rawEvent,
      },
      ...rest,
    },
  };
  return protoEvents.Event.encode(eventMessage).finish();
}

/**
 * Decodes a protocol buffer binary format to an event message.
 * The format includes a 4-byte length prefix followed by the message.
 */
export function decode(data: Uint8Array): BaseEvent {
  const event = protoEvents.Event.decode(data);
  const decoded = Object.values(event).find((value) => value !== undefined);
  if (!decoded) {
    throw new Error("Invalid event");
  }
  decoded.type = protoEvents.EventType[decoded.baseEvent.type];
  decoded.timestamp = decoded.baseEvent.timestamp;
  decoded.rawEvent = decoded.baseEvent.rawEvent;
  delete decoded.baseEvent;

  // we want tool calls to be optional, so we need to remove them if they are empty
  if (decoded.type === EventType.MESSAGES_SNAPSHOT) {
    for (const message of (decoded as LooseRecord).messages as Message[]) {
      const untypedMessage = message as LooseRecord;

      if (untypedMessage.role === "user" && Array.isArray(untypedMessage.contentParts)) {
        const contentParts = untypedMessage.contentParts
          .map((part: unknown) => fromProtoContentPart(part))
          .filter((part: unknown) => part !== undefined);

        if (contentParts.length > 0) {
          untypedMessage.content = contentParts;
        }
      }

      if (Array.isArray(untypedMessage.contentParts) && untypedMessage.contentParts.length === 0) {
        untypedMessage.contentParts = undefined;
      }

      if (Array.isArray(untypedMessage.toolCalls) && untypedMessage.toolCalls.length === 0) {
        untypedMessage.toolCalls = undefined;
      }
    }
  }

  // RunFinishedEvent: rebuild the nested `outcome` discriminated union from the
  // flat proto fields. Empty/missing `outcome` decodes to `undefined` (legacy
  // event); "success" decodes to `{ type: "success" }`; "interrupt" decodes to
  // `{ type: "interrupt", interrupts }`.
  if (decoded.type === EventType.RUN_FINISHED) {
    const runFinished = decoded as LooseRecord;
    const wireOutcome: string | undefined =
      typeof runFinished.outcome === "string" && runFinished.outcome !== ""
        ? runFinished.outcome
        : undefined;
    const wireInterrupts: unknown[] = Array.isArray(runFinished.interrupts)
      ? runFinished.interrupts
      : [];

    delete runFinished.interrupts;

    if (wireOutcome === "interrupt") {
      runFinished.outcome = { type: "interrupt", interrupts: wireInterrupts };
    } else if (wireOutcome === "success") {
      runFinished.outcome = { type: "success" };
    } else {
      delete runFinished.outcome;
    }
  }

  // custom mapping for json patch operations
  if (decoded.type === EventType.STATE_DELTA) {
    for (const operation of (decoded as LooseRecord).delta as LooseRecord[]) {
      operation.op = protoPatch.JsonPatchOperationType[
        operation.op as protoPatch.JsonPatchOperationType
      ].toLowerCase();
      Object.keys(operation).forEach((key) => {
        if (operation[key] === undefined) {
          delete operation[key];
        }
      });
    }
  }

  Object.keys(decoded).forEach((key) => {
    if (decoded[key] === undefined) {
      delete decoded[key];
    }
  });

  return EventSchemas.parse(decoded);
}
