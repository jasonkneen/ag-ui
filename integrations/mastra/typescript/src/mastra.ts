import type {
  AgentConfig,
  BaseEvent,
  CustomEvent,
  RunAgentInput,
  RunFinishedEvent,
  RunStartedEvent,
  StateSnapshotEvent,
  TextMessageChunkEvent,
  ToolCallArgsEvent,
  ToolCallEndEvent,
  ToolCallResultEvent,
  ToolCallStartEvent,
} from "@ag-ui/client";
import { AbstractAgent, EventType } from "@ag-ui/client";
import type { StorageThreadType } from "@mastra/core/memory";
import type { Agent as LocalMastraAgent } from "@mastra/core/agent";
import { RequestContext } from "@mastra/core/request-context";
import { randomUUID } from "@ag-ui/client";
import { Observable } from "rxjs";
import type { MastraClient } from "@mastra/client-js";
import {
  convertAGUIMessagesToMastra,
  GetLocalAgentsOptions,
  getLocalAgents,
  getRemoteAgents,
  GetRemoteAgentsOptions,
  GetLocalAgentOptions,
  getLocalAgent,
  GetNetworkOptions,
  getNetwork,
} from "./utils";

type RemoteMastraAgent = ReturnType<MastraClient["getAgent"]>;

export interface MastraAgentConfig extends AgentConfig {
  agent: LocalMastraAgent | RemoteMastraAgent;
  resourceId: string;
  requestContext?: RequestContext;
}

interface MastraAgentStreamOptions {
  onTextPart?: (text: string) => void;
  onFinishMessagePart?: () => void;
  onToolCallPart?: (streamPart: {
    toolCallId: string;
    toolName: string;
    args: any;
  }) => void;
  onToolResultPart?: (streamPart: { toolCallId: string; result: any }) => void;
  onError?: (error: Error) => void;
  onRunFinished?: () => Promise<void>;
  onToolSuspended?: (payload: {
    toolCallId: string;
    toolName: string;
    suspendPayload: any;
    args: Record<string, any>;
    resumeSchema: string;
  }) => void;
}

export class MastraAgent extends AbstractAgent {
  agent: LocalMastraAgent | RemoteMastraAgent;
  resourceId: string;
  requestContext?: RequestContext;

  constructor(private config: MastraAgentConfig) {
    const { agent, resourceId, requestContext, ...rest } = config;
    super(rest);
    this.agent = agent;
    this.resourceId = resourceId;
    this.requestContext = requestContext ?? new RequestContext();
  }

  public clone() {
    return new MastraAgent(this.config);
  }

  run(input: RunAgentInput): Observable<BaseEvent> {
    let messageId = randomUUID();

    return new Observable<BaseEvent>((subscriber) => {
      const run = async () => {
        const runStartedEvent: RunStartedEvent = {
          type: EventType.RUN_STARTED,
          threadId: input.threadId,
          runId: input.runId,
        };

        subscriber.next(runStartedEvent);

        // Check for resume from interrupt
        // Note: input.forwardedProps is typed as ZodAny, so no cast needed
        const forwardedCommand = input.forwardedProps?.command;
        if (forwardedCommand?.resume != null && forwardedCommand?.interruptEvent) {
          const interruptEvent =
            typeof forwardedCommand.interruptEvent === "string"
              ? JSON.parse(forwardedCommand.interruptEvent)
              : forwardedCommand.interruptEvent;

          if (!this.isLocalMastraAgent(this.agent)) {
            // TODO: Remote agent resume not yet implemented
            console.warn(
              "Resume from interrupt is not yet supported for remote Mastra agents",
            );
            subscriber.next({
              type: EventType.RUN_FINISHED,
              threadId: input.threadId,
              runId: input.runId,
            } as RunFinishedEvent);
            subscriber.complete();
            return;
          }

          try {
            const response = await this.agent.resumeStream(
              forwardedCommand.resume,
              {
                toolCallId: interruptEvent.toolCallId,
                runId: interruptEvent.runId,
                memory: {
                  thread: input.threadId,
                  resource: this.resourceId ?? input.threadId,
                },
                requestContext: this.requestContext,
              },
            );

            if (response && typeof response === "object") {
              const callbacks = this.makeStreamCallbacks(
                subscriber,
                () => messageId,
                (id) => { messageId = id; },
                input.runId,
              );
              await this.processFullStream(response.fullStream, callbacks);
            }

            subscriber.next({
              type: EventType.RUN_FINISHED,
              threadId: input.threadId,
              runId: input.runId,
            } as RunFinishedEvent);
            subscriber.complete();
          } catch (error) {
            console.error("Resume stream error:", error);
            subscriber.error(error);
          }
          return;
        }

        // Handle local agent memory management (from Mastra implementation)
        if (this.isLocalMastraAgent(this.agent)) {
          const memory = await this.agent.getMemory({
            requestContext: this.requestContext,
          });

          if (
            memory &&
            input.state &&
            Object.keys(input.state).length > 0
          ) {
            let thread: StorageThreadType | null = await memory.getThreadById({
              threadId: input.threadId,
            });

            if (!thread) {
              thread = {
                id: input.threadId,
                title: "",
                metadata: {},
                resourceId: this.resourceId ?? input.threadId,
                createdAt: new Date(),
                updatedAt: new Date(),
              };
            }

            let existingMemory: Record<string, any> = {};
            try {
              existingMemory = JSON.parse(
                (thread.metadata?.workingMemory as string) ?? "{}",
              );
            } catch {
              // Working memory metadata is not valid JSON - start fresh
            }
            const { messages, ...rest } = input.state;
            const workingMemory = JSON.stringify({
              ...existingMemory,
              ...rest,
            });

            // Update thread metadata with new working memory
            await memory.saveThread({
              thread: {
                ...thread,
                metadata: {
                  ...thread.metadata,
                  workingMemory,
                },
              },
            });
          }
        }

        try {
          await this.streamMastraAgent(input, {
            onTextPart: (text) => {
              const event: TextMessageChunkEvent = {
                type: EventType.TEXT_MESSAGE_CHUNK,
                role: "assistant",
                messageId,
                delta: text,
              };
              subscriber.next(event);
            },
            onToolCallPart: (streamPart) => {
              const startEvent: ToolCallStartEvent = {
                type: EventType.TOOL_CALL_START,
                parentMessageId: messageId,
                toolCallId: streamPart.toolCallId,
                toolCallName: streamPart.toolName,
              };
              subscriber.next(startEvent);

              const argsEvent: ToolCallArgsEvent = {
                type: EventType.TOOL_CALL_ARGS,
                toolCallId: streamPart.toolCallId,
                delta: JSON.stringify(streamPart.args),
              };
              subscriber.next(argsEvent);

              const endEvent: ToolCallEndEvent = {
                type: EventType.TOOL_CALL_END,
                toolCallId: streamPart.toolCallId,
              };
              subscriber.next(endEvent);
            },
            onToolResultPart(streamPart) {
              const toolCallResultEvent: ToolCallResultEvent = {
                type: EventType.TOOL_CALL_RESULT,
                toolCallId: streamPart.toolCallId,
                content: JSON.stringify(streamPart.result),
                messageId: randomUUID(),
                role: "tool",
              };

              subscriber.next(toolCallResultEvent);
            },
            onToolSuspended: (payload) => {
              const event: CustomEvent = {
                type: EventType.CUSTOM,
                name: "on_interrupt",
                value: JSON.stringify({
                  type: "mastra_suspend",
                  toolCallId: payload.toolCallId,
                  toolName: payload.toolName,
                  suspendPayload: payload.suspendPayload,
                  args: payload.args,
                  resumeSchema: payload.resumeSchema,
                  runId: input.runId,
                }),
              };
              subscriber.next(event);
            },
            onFinishMessagePart: async () => {
              messageId = randomUUID();
            },
            onError: (error) => {
              console.error("error", error);
              // Handle error
              subscriber.error(error);
            },
            onRunFinished: async () => {
              if (this.isLocalMastraAgent(this.agent)) {
                try {
                  const memory = await this.agent.getMemory({
                    requestContext: this.requestContext,
                  });
                  if (memory) {
                    const workingMemory = await memory.getWorkingMemory({
                      resourceId: this.resourceId,
                      threadId: input.threadId,
                      memoryConfig: {
                        workingMemory: {
                          enabled: true,
                        },
                      },
                    });

                    if (typeof workingMemory === "string") {
                      let snapshot: Record<string, any> | null = null;
                      try {
                        snapshot = JSON.parse(workingMemory);
                      } catch {
                        // Working memory is not valid JSON (e.g. markdown template)
                        // Wrap it so the client still receives the state
                        snapshot = { workingMemory };
                      }

                      if (snapshot && !("$schema" in snapshot)) {
                        const stateSnapshotEvent: StateSnapshotEvent = {
                          type: EventType.STATE_SNAPSHOT,
                          snapshot,
                        };

                        subscriber.next(stateSnapshotEvent);
                      }
                    }
                  }
                } catch (error) {
                  console.error("Error sending state snapshot", error);
                }
              }

              // Emit run finished event
              subscriber.next({
                type: EventType.RUN_FINISHED,
                threadId: input.threadId,
                runId: input.runId,
              } as RunFinishedEvent);

              // Complete the observable
              subscriber.complete();
            },
          });
        } catch (error) {
          console.error("Stream error:", error);
          subscriber.error(error);
        }
      };

      run();

      return () => {};
    });
  }

  isLocalMastraAgent(
    agent: LocalMastraAgent | RemoteMastraAgent,
  ): agent is LocalMastraAgent {
    return "getMemory" in agent;
  }

  /**
   * Creates the callback set used by processFullStream to emit AG-UI events.
   * messageId is accessed/mutated via getter/setter to share state with run().
   */
  private makeStreamCallbacks(
    subscriber: { next: (event: BaseEvent) => void },
    getMessageId: () => string,
    setMessageId: (id: string) => void,
    runId: string,
  ): MastraAgentStreamOptions {
    return {
      onTextPart: (text) => {
        subscriber.next({
          type: EventType.TEXT_MESSAGE_CHUNK,
          role: "assistant",
          messageId: getMessageId(),
          delta: text,
        } as TextMessageChunkEvent);
      },
      onToolCallPart: (streamPart) => {
        subscriber.next({
          type: EventType.TOOL_CALL_START,
          parentMessageId: getMessageId(),
          toolCallId: streamPart.toolCallId,
          toolCallName: streamPart.toolName,
        } as ToolCallStartEvent);
        subscriber.next({
          type: EventType.TOOL_CALL_ARGS,
          toolCallId: streamPart.toolCallId,
          delta: JSON.stringify(streamPart.args),
        } as ToolCallArgsEvent);
        subscriber.next({
          type: EventType.TOOL_CALL_END,
          toolCallId: streamPart.toolCallId,
        } as ToolCallEndEvent);
      },
      onToolResultPart: (streamPart) => {
        subscriber.next({
          type: EventType.TOOL_CALL_RESULT,
          toolCallId: streamPart.toolCallId,
          content: JSON.stringify(streamPart.result),
          messageId: randomUUID(),
          role: "tool",
        } as ToolCallResultEvent);
      },
      onToolSuspended: (payload) => {
        subscriber.next({
          type: EventType.CUSTOM,
          name: "on_interrupt",
          value: JSON.stringify({
            type: "mastra_suspend",
            toolCallId: payload.toolCallId,
            toolName: payload.toolName,
            suspendPayload: payload.suspendPayload,
            args: payload.args,
            resumeSchema: payload.resumeSchema,
            runId,
          }),
        } as CustomEvent);
      },
      onFinishMessagePart: async () => {
        setMessageId(randomUUID());
      },
      onError: (error) => {
        throw error;
      },
    };
  }

  /**
   * Processes a Mastra fullStream, mapping chunks to AG-UI events via callbacks.
   * Buffers tool-call chunks: if followed by tool-call-suspended, the TOOL_CALL_*
   * events are suppressed (the tool hasn't executed yet — emitting them confuses
   * CopilotKit's orchestration which expects a TOOL_CALL_RESULT to follow).
   */
  private async processFullStream(
    stream: AsyncIterable<any>,
    callbacks: MastraAgentStreamOptions,
  ): Promise<void> {
    let pendingToolCall: {
      toolCallId: string;
      toolName: string;
      args: any;
    } | null = null;

    const flushPendingToolCall = () => {
      if (pendingToolCall) {
        callbacks.onToolCallPart?.(pendingToolCall);
        pendingToolCall = null;
      }
    };

    for await (const chunk of stream) {
      switch (chunk.type) {
        case "text-delta": {
          flushPendingToolCall();
          callbacks.onTextPart?.(chunk.payload.text);
          break;
        }
        case "tool-call": {
          // Buffer — don't emit yet, wait to see if suspend follows
          flushPendingToolCall(); // flush any previous buffered call
          pendingToolCall = {
            toolCallId: chunk.payload.toolCallId,
            toolName: chunk.payload.toolName,
            args: chunk.payload.args,
          };
          break;
        }
        case "tool-result": {
          flushPendingToolCall(); // tool executed normally, emit the call
          callbacks.onToolResultPart?.({
            toolCallId: chunk.payload.toolCallId,
            result: chunk.payload.result,
          });
          break;
        }
        case "error": {
          flushPendingToolCall();
          callbacks.onError?.(new Error(chunk.payload.error as string));
          break;
        }
        case "tool-call-suspended": {
          // Suppress the buffered tool-call — tool didn't execute
          pendingToolCall = null;
          callbacks.onToolSuspended?.({
            toolCallId: chunk.payload.toolCallId,
            toolName: chunk.payload.toolName,
            suspendPayload: chunk.payload.suspendPayload,
            args: chunk.payload.args,
            resumeSchema: chunk.payload.resumeSchema,
          });
          break;
        }
        case "finish": {
          flushPendingToolCall();
          callbacks.onFinishMessagePart?.();
          break;
        }
      }
    }
    // Flush any remaining buffered tool call at end of stream
    flushPendingToolCall();
  }

  /**
   * Streams in process or remote mastra agent.
   * @param input - The input for the mastra agent.
   * @param options - The options for the mastra agent.
   * @returns The stream of the mastra agent.
   */
  private async streamMastraAgent(
    { threadId, runId, messages, tools, context: inputContext }: RunAgentInput,
    {
      onTextPart,
      onFinishMessagePart,
      onToolCallPart,
      onToolResultPart,
      onToolSuspended,
      onError,
      onRunFinished,
    }: MastraAgentStreamOptions,
  ): Promise<void> {
    const clientTools = tools.reduce(
      (acc, tool) => {
        acc[tool.name as string] = {
          id: tool.name,
          description: tool.description,
          inputSchema: tool.parameters,
        };
        return acc;
      },
      {} as Record<string, any>,
    );
    const resourceId = this.resourceId ?? threadId;

    const convertedMessages = convertAGUIMessagesToMastra(messages);
    this.requestContext?.set("ag-ui", { context: inputContext });
    const requestContext = this.requestContext;

    if (this.isLocalMastraAgent(this.agent)) {
      // Local agent - use the agent's stream method directly
      try {
        const response = await this.agent.stream(convertedMessages, {
          memory: {
            thread: threadId,
            resource: resourceId,
          },
          runId,
          clientTools,
          requestContext,
        });

        // For local agents, the response should already be a stream
        // Process it using the agent's built-in streaming mechanism
        if (response && typeof response === "object") {
          await this.processFullStream(response.fullStream, {
            onTextPart,
            onFinishMessagePart,
            onToolCallPart,
            onToolResultPart,
            onToolSuspended,
            onError,
          });

          await onRunFinished?.();
        } else {
          throw new Error("Invalid response from local agent");
        }
      } catch (error) {
        onError?.(error as Error);
      }
    } else {
      // Remote agent - use the remote agent's stream method
      try {
        const response = await this.agent.stream(convertedMessages, {
          memory: {
            thread: threadId,
            resource: resourceId,
          },
          runId,
          clientTools,
          requestContext,
        });

        // Remote agents use processDataStream (callback-based), not an
        // async iterable — can't use processFullStream directly. Buffer
        // tool-call chunks with the same logic as processFullStream.
        if (response && typeof response.processDataStream === "function") {
          let pendingToolCall: {
            toolCallId: string;
            toolName: string;
            args: any;
          } | null = null;
          const flushPendingToolCall = () => {
            if (pendingToolCall) {
              onToolCallPart?.(pendingToolCall);
              pendingToolCall = null;
            }
          };

          await response.processDataStream({
            onChunk: async (chunk: any) => {
              switch (chunk.type) {
                case "text-delta": {
                  flushPendingToolCall();
                  onTextPart?.(chunk.payload.text);
                  break;
                }
                case "tool-call": {
                  flushPendingToolCall();
                  pendingToolCall = {
                    toolCallId: chunk.payload.toolCallId,
                    toolName: chunk.payload.toolName,
                    args: chunk.payload.args,
                  };
                  break;
                }
                case "tool-result": {
                  flushPendingToolCall();
                  onToolResultPart?.({
                    toolCallId: chunk.payload.toolCallId,
                    result: chunk.payload.result,
                  });
                  break;
                }
                case "tool-call-suspended": {
                  pendingToolCall = null;
                  onToolSuspended?.({
                    toolCallId: chunk.payload.toolCallId,
                    toolName: chunk.payload.toolName,
                    suspendPayload: chunk.payload.suspendPayload,
                    args: chunk.payload.args,
                    resumeSchema: chunk.payload.resumeSchema,
                  });
                  break;
                }
                case "finish": {
                  flushPendingToolCall();
                  onFinishMessagePart?.();
                  break;
                }
              }
            },
          });
          flushPendingToolCall();
          await onRunFinished?.();
        } else {
          throw new Error("Invalid response from remote agent");
        }
      } catch (error) {
        onError?.(error as Error);
      }
    }
  }

  static async getRemoteAgents(
    options: GetRemoteAgentsOptions,
  ): Promise<Record<string, AbstractAgent>> {
    return getRemoteAgents(options);
  }

  static getLocalAgents(
    options: GetLocalAgentsOptions,
  ): Record<string, AbstractAgent> {
    return getLocalAgents(options);
  }

  static getLocalAgent(options: GetLocalAgentOptions) {
    return getLocalAgent(options);
  }

  static getNetwork(options: GetNetworkOptions) {
    return getNetwork(options);
  }
}
