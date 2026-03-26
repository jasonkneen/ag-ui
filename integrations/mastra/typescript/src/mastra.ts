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
          // #2: Safely parse interruptEvent — client-supplied data
          let interruptEvent: any;
          try {
            interruptEvent =
              typeof forwardedCommand.interruptEvent === "string"
                ? JSON.parse(forwardedCommand.interruptEvent)
                : forwardedCommand.interruptEvent;
          } catch (err) {
            subscriber.error(
              new Error("Invalid interruptEvent: malformed JSON", {
                cause: err,
              }),
            );
            return;
          }

          // #6: Validate required fields for resume
          if (!interruptEvent?.toolCallId || !interruptEvent?.runId) {
            subscriber.error(
              new Error(
                "Invalid interruptEvent: missing toolCallId or runId",
              ),
            );
            return;
          }

          // #3: Remote agent resume is not yet supported — error, don't fake success
          if (!this.isLocalMastraAgent(this.agent)) {
            subscriber.error(
              new Error(
                "Resume from interrupt is not yet supported for remote Mastra agents",
              ),
            );
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

            // #5: Null/invalid response from resumeStream is an error
            if (!response || typeof response !== "object" || !response.fullStream) {
              subscriber.error(
                new Error("resumeStream returned no valid response (missing fullStream)"),
              );
              return;
            }

            const callbacks = this.makeStreamCallbacks(
              subscriber,
              () => messageId,
              (id) => { messageId = id; },
              input.runId,
            );
            const hadError = await this.processFullStream(response.fullStream, {
              ...callbacks,
              onError: (error) => {
                subscriber.error(error);
              },
            });

            if (!hadError) {
              await this.emitWorkingMemorySnapshot(subscriber, input.threadId);
              subscriber.next({
                type: EventType.RUN_FINISHED,
                threadId: input.threadId,
                runId: input.runId,
              } as RunFinishedEvent);
              subscriber.complete();
            }
          } catch (error) {
            // #9: Let subscriber.error carry the event — no console.error
            subscriber.error(error);
          }
          return;
        }

        // Handle local agent memory management (from Mastra implementation)
        if (this.isLocalMastraAgent(this.agent)) {
          try {
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
          } catch (error) {
            subscriber.error(error);
            return;
          }
        }

        try {
          const streamCallbacks = this.makeStreamCallbacks(
            subscriber,
            () => messageId,
            (id) => { messageId = id; },
            input.runId,
          );

          await this.streamMastraAgent(input, {
            ...streamCallbacks,
            onError: (error) => {
              subscriber.error(error);
            },
            onRunFinished: async () => {
              await this.emitWorkingMemorySnapshot(subscriber, input.threadId);
              subscriber.next({
                type: EventType.RUN_FINISHED,
                threadId: input.threadId,
                runId: input.runId,
              } as RunFinishedEvent);
              subscriber.complete();
            },
          });
        } catch (error) {
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
   * Fetches working memory from a local agent and emits a STATE_SNAPSHOT event
   * if valid working memory is available. Errors are caught and logged to avoid
   * disrupting the run lifecycle.
   */
  private async emitWorkingMemorySnapshot(
    subscriber: { next: (event: BaseEvent) => void },
    threadId: string,
  ): Promise<void> {
    if (!this.isLocalMastraAgent(this.agent)) return;
    try {
      const memory = await this.agent.getMemory({
        requestContext: this.requestContext,
      });
      if (memory) {
        const workingMemory = await memory.getWorkingMemory({
          resourceId: this.resourceId,
          threadId,
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
            subscriber.next({
              type: EventType.STATE_SNAPSHOT,
              snapshot,
            } as StateSnapshotEvent);
          }
        }
      }
    } catch (error) {
      console.error("Error sending state snapshot", error);
    }
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
   * Creates a stateful chunk processor that maps Mastra stream chunks to
   * AG-UI events via callbacks. Buffers tool-call chunks: if followed by
   * tool-call-suspended, the TOOL_CALL_* events are suppressed (the tool
   * hasn't executed yet — emitting them confuses CopilotKit's orchestration
   * which expects a TOOL_CALL_RESULT to follow).
   *
   * Used by both the local agent path (async iterable) and the remote agent
   * path (processDataStream callback) — single source of truth for chunk
   * handling and buffering logic.
   *
   * @returns handleChunk (returns true if processing should stop, e.g. on error)
   *          and flush (emits any buffered tool-call at end of stream).
   */
  private createChunkProcessor(callbacks: MastraAgentStreamOptions) {
    let pendingToolCall: {
      toolCallId: string;
      toolName: string;
      args: any;
    } | null = null;

    const flush = () => {
      if (pendingToolCall) {
        callbacks.onToolCallPart?.(pendingToolCall);
        pendingToolCall = null;
      }
    };

    /** @returns true if processing should stop (error chunk received). */
    const handleChunk = (chunk: any): boolean => {
      switch (chunk.type) {
        case "text-delta": {
          flush();
          callbacks.onTextPart?.(chunk.payload.text);
          break;
        }
        case "tool-call": {
          flush();
          pendingToolCall = {
            toolCallId: chunk.payload.toolCallId,
            toolName: chunk.payload.toolName,
            args: chunk.payload.args,
          };
          break;
        }
        case "tool-result": {
          flush();
          callbacks.onToolResultPart?.({
            toolCallId: chunk.payload.toolCallId,
            result: chunk.payload.result,
          });
          break;
        }
        case "error": {
          const error = new Error(chunk.payload.error as string);
          if (callbacks.onError) {
            callbacks.onError(error);
          }
          return true;
        }
        case "tool-call-suspended": {
          if (pendingToolCall?.toolCallId === chunk.payload.toolCallId) {
            pendingToolCall = null;
          } else {
            flush();
          }
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
          flush();
          callbacks.onFinishMessagePart?.();
          break;
        }
      }
      return false;
    };

    return { handleChunk, flush };
  }

  /**
   * Processes a Mastra fullStream (async iterable) using createChunkProcessor.
   * @returns true if processing stopped due to an error chunk.
   */
  private async processFullStream(
    stream: AsyncIterable<any>,
    callbacks: MastraAgentStreamOptions,
  ): Promise<boolean> {
    const { handleChunk, flush } = this.createChunkProcessor(callbacks);
    for await (const chunk of stream) {
      if (handleChunk(chunk)) return true;
    }
    flush();
    return false;
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
          const hadError = await this.processFullStream(response.fullStream, {
            onTextPart,
            onFinishMessagePart,
            onToolCallPart,
            onToolResultPart,
            onToolSuspended,
            onError,
          });

          if (!hadError) await onRunFinished?.();
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

        // Remote agents use processDataStream (callback-based) — share
        // chunk handling logic via createChunkProcessor.
        if (response && typeof response.processDataStream === "function") {
          const { handleChunk, flush } = this.createChunkProcessor({
            onTextPart,
            onFinishMessagePart,
            onToolCallPart,
            onToolResultPart,
            onToolSuspended,
            onError,
          });
          let stopped = false;

          await response.processDataStream({
            onChunk: async (chunk: any) => {
              if (stopped) return;
              if (handleChunk(chunk)) stopped = true;
            },
          });
          if (!stopped) flush();
          if (!stopped) await onRunFinished?.();
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
