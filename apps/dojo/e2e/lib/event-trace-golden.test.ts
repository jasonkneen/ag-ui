import assert from "node:assert/strict";
import test from "node:test";
import {
  defineEventTrace,
  getEventTraceDestination,
} from "./event-trace-golden";

test("associates each plain journey array with its generated destination", () => {
  const golden = defineEventTrace("file:///agenticChatPage.event-trace.ts", {
    sendsAndReceivesMessage: [{ type: "RUN_STARTED" }],
    retainsMemory: [{ type: "RUN_FINISHED" }],
  });

  assert.deepEqual(getEventTraceDestination(golden.sendsAndReceivesMessage), {
    sourceUrl: "file:///agenticChatPage.event-trace.ts",
    journeyKey: "sendsAndReceivesMessage",
  });
  assert.deepEqual(getEventTraceDestination(golden.retainsMemory), {
    sourceUrl: "file:///agenticChatPage.event-trace.ts",
    journeyKey: "retainsMemory",
  });
});
