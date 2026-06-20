using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text.Json;
using AGUI.Abstractions;
using AGUI.Server;
using Microsoft.Extensions.AI;

namespace AGUIDojoServer.SharedState;

[SuppressMessage("Performance", "CA1812:Avoid uninstantiated internal classes", Justification = "Instantiated by ChatClientAgentFactory.CreateSharedState")]
internal sealed class SharedStateAgent : DelegatingChatClient
{
    private readonly JsonSerializerOptions _jsonSerializerOptions;

    public SharedStateAgent(IChatClient innerClient, JsonSerializerOptions jsonSerializerOptions)
        : base(innerClient)
    {
        _jsonSerializerOptions = jsonSerializerOptions;
    }

    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Extract AG-UI input to check for state
        var agentInput = options?.AdditionalProperties?[AGUIConstants.RunAgentInputKey] as RunAgentInput;
        if (agentInput?.State is not { ValueKind: not JsonValueKind.Undefined } state)
        {
            await foreach (var update in base.GetStreamingResponseAsync(chatMessages, options, cancellationToken).ConfigureAwait(false))
            {
                yield return update;
            }
            yield break;
        }

        // First pass: get structured JSON state output
        var firstPassOptions = options!.Clone();
        firstPassOptions.ResponseFormat = ChatResponseFormat.ForJsonSchema<RecipeResponse>(
            schemaName: "RecipeResponse",
            schemaDescription: "A response containing a recipe with title, skill level, cooking time, preferences, ingredients, and instructions");

        ChatMessage stateUpdateMessage = new(
            ChatRole.System,
            [
                new TextContent("Here is the current state in JSON format:"),
                new TextContent(state.GetRawText()),
                new TextContent("The new state is:")
            ]);

        var firstPassMessages = new List<ChatMessage>(chatMessages) { stateUpdateMessage };

        // Collect all updates from the first pass
        var allUpdates = new List<ChatResponseUpdate>();

        await foreach (var update in base.GetStreamingResponseAsync(firstPassMessages, firstPassOptions, cancellationToken).ConfigureAwait(false))
        {
            allUpdates.Add(update);

            // Yield non-text updates (tool calls, etc.)
            bool hasNonTextContent = false;
            foreach (var content in update.Contents)
            {
                if (content is not TextContent)
                {
                    hasNonTextContent = true;
                    break;
                }
            }

            if (hasNonTextContent)
            {
                yield return update;
            }
        }

        // Use ToChatResponse to get the aggregated response with messages
        var response = allUpdates.ToChatResponse();
        var responseText = response.Text;

        if (string.IsNullOrWhiteSpace(responseText))
        {
            yield break;
        }

        // Try to parse the accumulated text as JSON for state snapshot
        JsonElement stateSnapshot;
        try
        {
            stateSnapshot = JsonSerializer.Deserialize<JsonElement>(responseText, _jsonSerializerOptions);
        }
        catch (JsonException)
        {
            yield break;
        }

        yield return new ChatResponseUpdate
        {
            Contents = [],
            RawRepresentation = new StateSnapshotEvent { Snapshot = stateSnapshot }
        };

        // Second pass: stream a concise summary
        var secondPassMessages = new List<ChatMessage>(chatMessages);
        // Add the first pass response messages
        secondPassMessages.AddRange(response.Messages);
        secondPassMessages.Add(new ChatMessage(
            ChatRole.System,
            [new TextContent("Please provide a concise summary of the state changes in at most two sentences.")]));

        await foreach (var update in base.GetStreamingResponseAsync(secondPassMessages, options, cancellationToken).ConfigureAwait(false))
        {
            yield return update;
        }
    }
}
