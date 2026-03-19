import {
  BaseEvent,
  Message,
  RunAgentInput,
  RunErrorEvent,
  RunFinishedEvent,
  RunStartedEvent,
  State,
  StateDeltaEvent,
  StateSnapshotEvent,
  StepFinishedEvent,
  StepStartedEvent,
  TextMessageContentEvent,
  TextMessageEndEvent,
  TextMessageStartEvent,
  ToolCallArgsEvent,
  ToolCallEndEvent,
  ToolCallResultEvent,
  ToolCallStartEvent,
  MessagesSnapshotEvent,
  RawEvent,
  CustomEvent,
  ToolCall,
  ActivitySnapshotEvent,
  ActivityDeltaEvent,
  ActivityMessage,
  ReasoningStartEvent,
  ReasoningMessageStartEvent,
  ReasoningMessageContentEvent,
  ReasoningMessageEndEvent,
  ReasoningEndEvent,
  ReasoningEncryptedValueEvent,
} from "@ag-ui/core";
import { AbstractAgent } from "./agent";
import { structuredClone_ } from "@/utils";

export interface AgentStateMutation {
  messages?: Message[];
  state?: State;
  stopPropagation?: boolean;
}

export interface AgentSubscriberParams {
  messages: Message[];
  state: State;
  agent: AbstractAgent;
  input: RunAgentInput;
}

// Utility type to allow callbacks to be implemented either synchronously or asynchronously.
export type MaybePromise<T> = T | Promise<T>;

export interface AgentSubscriber {
  // Request lifecycle
  onRunInitialized?(
    params: AgentSubscriberParams,
  ): MaybePromise<Omit<AgentStateMutation, "stopPropagation"> | void>;
  onRunFailed?(
    params: { error: Error } & AgentSubscriberParams,
  ): MaybePromise<Omit<AgentStateMutation, "stopPropagation"> | void>;
  onRunFinalized?(
    params: AgentSubscriberParams,
  ): MaybePromise<Omit<AgentStateMutation, "stopPropagation"> | void>;

  // Events
  onEvent?(
    params: { event: BaseEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onRunStartedEvent?(
    params: { event: RunStartedEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;
  onRunFinishedEvent?(
    params: { event: RunFinishedEvent; result?: any } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;
  onRunErrorEvent?(
    params: { event: RunErrorEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onStepStartedEvent?(
    params: { event: StepStartedEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;
  onStepFinishedEvent?(
    params: { event: StepFinishedEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onTextMessageStartEvent?(
    params: { event: TextMessageStartEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;
  onTextMessageContentEvent?(
    params: {
      event: TextMessageContentEvent;
      textMessageBuffer: string;
    } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;
  onTextMessageEndEvent?(
    params: { event: TextMessageEndEvent; textMessageBuffer: string } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onToolCallStartEvent?(
    params: { event: ToolCallStartEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;
  onToolCallArgsEvent?(
    params: {
      event: ToolCallArgsEvent;
      toolCallBuffer: string;
      toolCallName: string;
      partialToolCallArgs: Record<string, any>;
    } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;
  onToolCallEndEvent?(
    params: {
      event: ToolCallEndEvent;
      toolCallName: string;
      toolCallArgs: Record<string, any>;
    } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onToolCallResultEvent?(
    params: { event: ToolCallResultEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onStateSnapshotEvent?(
    params: { event: StateSnapshotEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onStateDeltaEvent?(
    params: { event: StateDeltaEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onMessagesSnapshotEvent?(
    params: { event: MessagesSnapshotEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onActivitySnapshotEvent?(
    params: {
      event: ActivitySnapshotEvent;
      activityMessage?: ActivityMessage;
      existingMessage?: Message;
    } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onActivityDeltaEvent?(
    params: {
      event: ActivityDeltaEvent;
      activityMessage?: ActivityMessage;
    } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onRawEvent?(
    params: { event: RawEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onCustomEvent?(
    params: { event: CustomEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  // Reasoning events
  onReasoningStartEvent?(
    params: { event: ReasoningStartEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onReasoningMessageStartEvent?(
    params: { event: ReasoningMessageStartEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onReasoningMessageContentEvent?(
    params: {
      event: ReasoningMessageContentEvent;
      reasoningMessageBuffer: string;
    } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onReasoningMessageEndEvent?(
    params: {
      event: ReasoningMessageEndEvent;
      reasoningMessageBuffer: string;
    } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onReasoningEndEvent?(
    params: { event: ReasoningEndEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  onReasoningEncryptedValueEvent?(
    params: { event: ReasoningEncryptedValueEvent } & AgentSubscriberParams,
  ): MaybePromise<AgentStateMutation | void>;

  // State changes
  onMessagesChanged?(
    params: Omit<AgentSubscriberParams, "input"> & { input?: RunAgentInput },
  ): MaybePromise<void>;
  onStateChanged?(
    params: Omit<AgentSubscriberParams, "input"> & { input?: RunAgentInput },
  ): MaybePromise<void>;
  onNewMessage?(
    params: { message: Message } & Omit<AgentSubscriberParams, "input"> & {
        input?: RunAgentInput;
      },
  ): MaybePromise<void>;
  onNewToolCall?(
    params: { toolCall: ToolCall } & Omit<AgentSubscriberParams, "input"> & {
        input?: RunAgentInput;
      },
  ): MaybePromise<void>;
}

export async function runSubscribersWithMutation(
  subscribers: AgentSubscriber[],
  initialMessages: Message[],
  initialState: State,
  executor: (
    subscriber: AgentSubscriber,
    messages: Message[],
    state: State,
  ) => MaybePromise<AgentStateMutation | void>,
): Promise<AgentStateMutation> {
  const baselineMessages = structuredClone_(initialMessages);
  const baselineState = structuredClone_(initialState);
  let messages: Message[] = baselineMessages;
  let state: State = baselineState;

  let stopPropagation: boolean | undefined = undefined;

  for (const subscriber of subscribers) {
    try {
      // Subscribers receive shared references and must not mutate them in-place.
      // Mutations should only be communicated via the return value.
      const mutation = await executor(subscriber, messages, state);

      if (mutation === undefined) {
        // Nothing returned – keep going
        continue;
      }

      // Replace with a defensive copy of the subscriber's mutation,
      // but skip if the subscriber returned the same reference (no-op).
      if (mutation.messages !== undefined && mutation.messages !== messages) {
        messages = structuredClone_(mutation.messages);
      }

      if (mutation.state !== undefined && mutation.state !== state) {
        state = structuredClone_(mutation.state);
      }

      stopPropagation = mutation.stopPropagation;

      if (stopPropagation === true) {
        break;
      }
    } catch (error) {
      // Log subscriber errors but continue processing (silence during tests)
      const isTestEnvironment =
        process.env.NODE_ENV === "test" || process.env.VITEST_WORKER_ID !== undefined;

      if (!isTestEnvironment) {
        console.error("Subscriber error:", error);
      }
      // Skip this subscriber's mutation and continue
      continue;
    }
  }

  return {
    ...(messages !== baselineMessages ? { messages } : {}),
    ...(state !== baselineState ? { state } : {}),
    ...(stopPropagation !== undefined ? { stopPropagation } : {}),
  };
}
