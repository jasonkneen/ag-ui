using System.Collections.Generic;
using AGUI.Server;
using Microsoft.Extensions.AI;
using OpenAI.Chat;

namespace AGUIDojoServer.A2UI;

/// <summary>
/// Extracts OpenAI's per-chunk streamed tool-call argument fragments off a
/// <see cref="ChatResponseUpdate"/> so A2UI surfaces paint progressively. Registered on the
/// endpoint's <see cref="AGUIStreamOptions"/> via
/// <see cref="AGUIStreamOptions.MapStreamingToolCallArguments"/>.
/// </summary>
/// <remarks>
/// This is the deliberate, isolated home for the OpenAI-SDK coupling: the provider-neutral
/// <c>AGUI.Server</c> conversion never references any provider SDK, and a different provider
/// would register its own extractor here instead.
/// </remarks>
internal static class OpenAIStreamingToolArguments
{
    public static IEnumerable<AGUIToolCallArgumentFragment>? Extract(ChatResponseUpdate update)
    {
        // Agent/middleware pipelines wrap the provider update once.
        object? raw = update.RawRepresentation;
        if (raw is ChatResponseUpdate inner)
        {
            raw = inner.RawRepresentation;
        }

        return raw is StreamingChatCompletionUpdate streaming ? Fragments(streaming) : null;
    }

    private static IEnumerable<AGUIToolCallArgumentFragment> Fragments(StreamingChatCompletionUpdate streaming)
    {
        foreach (StreamingChatToolCallUpdate toolCall in streaming.ToolCallUpdates ?? [])
        {
            yield return new AGUIToolCallArgumentFragment
            {
                Index = toolCall.Index,
                ToolCallId = toolCall.ToolCallId,
                FunctionName = toolCall.FunctionName,
                ArgumentsDelta = toolCall.FunctionArgumentsUpdate?.ToString() ?? string.Empty,
            };
        }
    }
}
