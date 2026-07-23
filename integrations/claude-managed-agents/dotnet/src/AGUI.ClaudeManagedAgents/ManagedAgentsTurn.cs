using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using AGUI.Abstractions;
using Anthropic.Models.Beta.Sessions;
using Anthropic.Models.Beta.Sessions.Events;

namespace AGUI.ClaudeManagedAgents;

/// <summary>
/// Drives one turn of a managed session: opens the event stream, posts the outbound events,
/// and translates the session's events into AG-UI events until the session goes idle.
/// </summary>
/// <remarks>
/// Invariant: no <c>TEXT_MESSAGE</c> or <c>REASONING</c> block is left open when a turn
/// finishes or errors. Every exit path closes them, and <see cref="CloseOpenBlocks"/> lets the
/// caller close whatever is still open when the turn throws.
/// </remarks>
internal sealed class ManagedAgentsTurn
{
    private const int MaxToolResultLength = 4000;

    private readonly IManagedAgentsClient _client;
    private readonly string _sessionId;
    private readonly IReadOnlyList<JsonElement> _outbound;
    private readonly IReadOnlyDictionary<string, string> _clientTools;
    private readonly IReadOnlyDictionary<string, ManagedAgentsBackendTool> _backendTools;
    private readonly string? _toolConfirmation;
    private readonly bool _streamDeltas;
    private readonly Func<CancellationToken, Task>? _onSent;

    private readonly Queue<BaseEvent> _pending = new();
    private readonly Dictionary<string, StringBuilder> _previews = new(StringComparer.Ordinal);
    private readonly HashSet<string> _closedMessages = new(StringComparer.Ordinal);
    private readonly HashSet<string> _openReasoning = new(StringComparer.Ordinal);
    private readonly List<string> _openReasoningOrder = new();
    private readonly HashSet<string> _ackedToolUses = new(StringComparer.Ordinal);
    private readonly HashSet<string> _clientParks = new(StringComparer.Ordinal);
    private readonly HashSet<string> _askedConfirmations = new(StringComparer.Ordinal);
    private bool _done;

    /// <summary>
    /// Initializes a new instance of the <see cref="ManagedAgentsTurn"/> class.
    /// </summary>
    /// <param name="client">The Managed Agents client.</param>
    /// <param name="sessionId">The session to drive.</param>
    /// <param name="outbound">Events posted into the session once the stream is open.</param>
    /// <param name="clientTools">Frontend tools, keyed by managed-agent (normalized) name to the original AG-UI name. Calls to these park the session.</param>
    /// <param name="backendTools">Custom tools executed on this server, keyed by managed-agent (normalized) name.</param>
    /// <param name="toolConfirmation">How to answer built-in tools gated on confirmation, or <see langword="null"/> to fail the run.</param>
    /// <param name="streamDeltas">Whether to request text and thinking previews.</param>
    /// <param name="onSent">Called once the outbound events have been posted into the session.</param>
    internal ManagedAgentsTurn(
        IManagedAgentsClient client,
        string sessionId,
        IReadOnlyList<JsonElement> outbound,
        IReadOnlyDictionary<string, string> clientTools,
        IReadOnlyDictionary<string, ManagedAgentsBackendTool> backendTools,
        string? toolConfirmation,
        bool streamDeltas,
        Func<CancellationToken, Task>? onSent = null)
    {
        _client = client;
        _sessionId = sessionId;
        _outbound = outbound;
        _clientTools = clientTools;
        _backendTools = backendTools;
        _toolConfirmation = toolConfirmation;
        _streamDeltas = streamDeltas;
        _onSent = onSent;
    }

    /// <summary>
    /// The outcome of the turn. Valid once <see cref="RunAsync"/> completes; a turn that
    /// throws leaves the default errored outcome.
    /// </summary>
    internal ManagedAgentsTurnOutcome Outcome { get; } = new();

    /// <summary>
    /// Runs the turn, yielding the translated AG-UI events.
    /// </summary>
    internal async IAsyncEnumerable<BaseEvent> RunAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Open the stream before sending so no early events are missed.
        var stream = await _client.OpenEventStreamAsync(_sessionId, _streamDeltas, cancellationToken).ConfigureAwait(false);
        await using (((IAsyncDisposable)stream).ConfigureAwait(false))
        {
            await SendOutboundAsync(cancellationToken).ConfigureAwait(false);
            if (_onSent is not null)
            {
                await _onSent(cancellationToken).ConfigureAwait(false);
            }

            await foreach (var streamEvent in stream.WithCancellation(cancellationToken).ConfigureAwait(false))
            {
                await HandleAsync(streamEvent, cancellationToken).ConfigureAwait(false);
                foreach (var evt in Drain())
                {
                    yield return evt;
                }

                if (_done)
                {
                    yield break;
                }
            }
        }

        // The stream ended without a terminal event.
        CloseAll();
        foreach (var evt in Drain())
        {
            yield return evt;
        }

        cancellationToken.ThrowIfCancellationRequested();
        Fail("The session event stream ended before the reply completed.", "stream_ended");
        foreach (var evt in Drain())
        {
            yield return evt;
        }
    }

    /// <summary>
    /// Posts the outbound events. A parked session accepts only tool results, so those go
    /// first (which resumes it) and any user messages follow in a second call: the API
    /// validates a whole batch against the session's current state, so mixing them fails.
    /// </summary>
    private async Task SendOutboundAsync(CancellationToken cancellationToken)
    {
        var followUps = _outbound.Where(IsFollowUp).ToList();
        var results = _outbound.Where(static evt => !IsFollowUp(evt)).ToList();

        if (results.Count > 0)
        {
            await _client.SendEventsAsync(_sessionId, results, cancellationToken).ConfigureAwait(false);
        }

        if (followUps.Count > 0)
        {
            await SendFollowUpsAsync(followUps, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// The session un-parks asynchronously after a tool result is posted, so a follow-up user
    /// message can race ahead of that transition and be rejected as sent-while-parked. Retries
    /// briefly on that specific error only.
    /// </summary>
    private async Task SendFollowUpsAsync(IReadOnlyList<JsonElement> followUps, CancellationToken cancellationToken)
    {
        for (var attempt = 0; ; attempt++)
        {
            try
            {
                await _client.SendEventsAsync(_sessionId, followUps, cancellationToken).ConfigureAwait(false);
                return;
            }
            catch (Exception ex) when (attempt < SentWhileParkedRetryDelays.Length && IsSentWhileParked(ex))
            {
                await Task.Delay(SentWhileParkedRetryDelays[attempt], cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static readonly TimeSpan[] SentWhileParkedRetryDelays =
    [
        TimeSpan.FromMilliseconds(150),
        TimeSpan.FromMilliseconds(300),
        TimeSpan.FromMilliseconds(600),
        TimeSpan.FromMilliseconds(1000),
        TimeSpan.FromMilliseconds(1500),
        TimeSpan.FromMilliseconds(2000),
    ];

    private static bool IsFollowUp(JsonElement evt)
    {
        if (evt.ValueKind != JsonValueKind.Object
            || !evt.TryGetProperty("type", out var type)
            || type.ValueKind != JsonValueKind.String)
        {
            return false;
        }

        return type.GetString() is "user.message" or "system.message";
    }

    /// <summary>
    /// Whether the API rejected an event because the session is still parked on tool calls
    /// (HTTP 400 whose message contains <c>waiting on responses</c>).
    /// </summary>
    private static bool IsSentWhileParked(Exception ex)
    {
        return ex is Anthropic.Exceptions.AnthropicApiException api
            ? api.StatusCode == System.Net.HttpStatusCode.BadRequest
                && api.Message.Contains("waiting on responses", StringComparison.Ordinal)
            : ex is ManagedAgentsSendException { StatusCode: (int)System.Net.HttpStatusCode.BadRequest } sent
                && sent.Message.Contains("waiting on responses", StringComparison.Ordinal);
    }

    /// <summary>
    /// Closes every open text and reasoning block and returns the pending events, including
    /// the closing ones. Used when the turn throws so the caller can honor the closing invariant.
    /// </summary>
    internal IReadOnlyList<BaseEvent> CloseOpenBlocks()
    {
        CloseAll();
        return Drain().ToList();
    }

    private IEnumerable<BaseEvent> Drain()
    {
        while (_pending.TryDequeue(out var evt))
        {
            yield return evt;
        }
    }

    private void Emit(BaseEvent evt) => _pending.Enqueue(evt);

    private async Task HandleAsync(BetaManagedAgentsStreamSessionEvents streamEvent, CancellationToken cancellationToken)
    {
        if (streamEvent.TryPickStartEvent(out var start))
        {
            HandleEventStart(start);
            return;
        }

        if (streamEvent.TryPickDeltaEvent(out var delta))
        {
            HandleEventDelta(delta, streamEvent.Json);
            return;
        }

        if (streamEvent.TryPickAgentThinkingEvent(out var thinking))
        {
            // The thinking stretch finished. Its text is not exposed by the API today, so this
            // is a progress signal: close the reasoning block we opened.
            if (_openReasoning.Contains(thinking.ID))
            {
                CloseReasoning(thinking.ID);
            }
            else
            {
                Emit(new ReasoningStartEvent { MessageId = thinking.ID });
                Emit(new ReasoningEndEvent { MessageId = thinking.ID });
            }

            return;
        }

        if (streamEvent.TryPickAgentMessageEvent(out var message))
        {
            HandleAgentMessage(message.ID, streamEvent.Json);
            return;
        }

        if (streamEvent.TryPickAgentCustomToolUseEvent(out var customToolUse))
        {
            await HandleCustomToolUseAsync(customToolUse.ID, customToolUse.Name, InputJsonOf(streamEvent.Json), cancellationToken).ConfigureAwait(false);
            return;
        }

        if (streamEvent.TryPickAgentToolUseEvent(out var toolUse))
        {
            EmitToolCall(toolUse.ID, toolUse.Name, InputJsonOf(streamEvent.Json));
            if (string.Equals(toolUse.EvaluatedPermission?.Raw(), "ask", StringComparison.Ordinal))
            {
                _askedConfirmations.Add(toolUse.ID);
            }

            return;
        }

        if (streamEvent.TryPickAgentMcpToolUseEvent(out var mcpToolUse))
        {
            EmitToolCall(mcpToolUse.ID, $"{mcpToolUse.McpServerName}: {mcpToolUse.Name}", InputJsonOf(streamEvent.Json));
            if (string.Equals(mcpToolUse.EvaluatedPermission?.Raw(), "ask", StringComparison.Ordinal))
            {
                _askedConfirmations.Add(mcpToolUse.ID);
            }

            return;
        }

        if (streamEvent.TryPickAgentToolResultEvent(out var toolResult))
        {
            EmitToolResult(toolResult.ToolUseID, DescribeContent(streamEvent.Json));
            return;
        }

        if (streamEvent.TryPickAgentMcpToolResultEvent(out var mcpToolResult))
        {
            EmitToolResult(mcpToolResult.McpToolUseID, DescribeContent(streamEvent.Json));
            return;
        }

        if (streamEvent.TryPickSpanModelRequestEndEvent(out _))
        {
            // Closes any preview whose buffered agent.message never arrived
            // (e.g. an interrupted or errored model request).
            CloseAll();
            return;
        }

        if (streamEvent.TryPickSessionErrorEvent(out var sessionError))
        {
            HandleSessionError(sessionError.Error.Json);
            return;
        }

        if (streamEvent.TryPickSessionStatusIdleEvent(out var idle))
        {
            await HandleIdleAsync(idle.StopReason, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (streamEvent.TryPickSessionStatusTerminatedEvent(out _) || streamEvent.TryPickSessionDeletedEvent(out _))
        {
            CloseAll();
            Emit(new RunErrorEvent
            {
                Message = "The managed session ended on the server. Send another message to start a fresh one.",
                Code = "session_ended",
            });
            Finish(ManagedAgentsTurnStatus.Errored, sessionEnded: true);
            return;
        }

        // status_running, rescheduled, spans, thread events, echoed user events: ignored.
    }

    private void HandleEventStart(BetaManagedAgentsStartEvent start)
    {
        if (start.Event.TryPickAgentMessage(out var preview))
        {
            Emit(new TextMessageStartEvent { MessageId = preview.ID, Role = AGUIRoles.Assistant });
            _previews[preview.ID] = new StringBuilder();
        }
        else if (start.Event.TryPickAgentThinking(out var thinkingPreview))
        {
            OpenReasoning(thinkingPreview.ID);
            Emit(new ReasoningStartEvent { MessageId = thinkingPreview.ID });
            Emit(new ReasoningMessageStartEvent { MessageId = thinkingPreview.ID, Role = AGUIRoles.Reasoning });
        }
    }

    private void HandleEventDelta(BetaManagedAgentsDeltaEvent delta, JsonElement rawEvent)
    {
        if (!_previews.TryGetValue(delta.EventID, out var preview))
        {
            return; // best-effort; the buffered agent.message is canonical
        }

        var text = TextDeltaOf(rawEvent);
        if (text is null)
        {
            return;
        }

        preview.Append(text);
        Emit(new TextMessageContentEvent { MessageId = delta.EventID, Delta = text });
    }

    private void HandleAgentMessage(string messageId, JsonElement rawEvent)
    {
        if (_closedMessages.Contains(messageId))
        {
            return;
        }

        var finalText = ManagedAgentsText.TextOf(ContentBlocksOf(rawEvent));
        if (!_previews.TryGetValue(messageId, out var preview))
        {
            Emit(new TextMessageStartEvent { MessageId = messageId, Role = AGUIRoles.Assistant });
            if (finalText.Length > 0)
            {
                Emit(new TextMessageContentEvent { MessageId = messageId, Delta = finalText });
            }
        }
        else
        {
            var previewed = preview.ToString();
            if (finalText.StartsWith(previewed, StringComparison.Ordinal))
            {
                if (finalText.Length > previewed.Length)
                {
                    Emit(new TextMessageContentEvent { MessageId = messageId, Delta = finalText.Substring(previewed.Length) });
                }
            }
            else
            {
                // The preview diverged from the final text: close it and re-emit the corrected whole.
                CloseMessage(messageId);
                if (finalText.Length > 0)
                {
                    var corrected = $"corrected_{messageId}";
                    Emit(new TextMessageStartEvent { MessageId = corrected, Role = AGUIRoles.Assistant });
                    Emit(new TextMessageContentEvent { MessageId = corrected, Delta = finalText });
                    Emit(new TextMessageEndEvent { MessageId = corrected });
                }

                return;
            }
        }

        CloseMessage(messageId);
    }

    private async Task HandleCustomToolUseAsync(string toolUseId, string name, string inputJson, CancellationToken cancellationToken)
    {
        // Report the frontend's original tool name, which may differ from the normalized name
        // registered on the managed agent.
        _clientTools.TryGetValue(name, out var originalName);
        EmitToolCall(toolUseId, originalName ?? name, inputJson);
        if (originalName is not null)
        {
            // The frontend executes this tool. Leave it unanswered; the session will park on
            // it and the next run supplies the result.
            _clientParks.Add(toolUseId);
            return;
        }

        if (_backendTools.TryGetValue(name, out var backendTool))
        {
            await RunBackendToolAsync(toolUseId, backendTool, inputJson, cancellationToken).ConfigureAwait(false);
            return;
        }

        // Nothing can execute this tool. Answer with an error so the agent recovers.
        var text = $"No handler is registered for tool \"{name}\".";
        EmitToolResult(toolUseId, text);
        await _client
            .SendEventsAsync(_sessionId, [ManagedAgentsSessionEvents.CustomToolResult(toolUseId, text, isError: true)], cancellationToken)
            .ConfigureAwait(false);
        _ackedToolUses.Add(toolUseId);
    }

    private async Task RunBackendToolAsync(string toolUseId, ManagedAgentsBackendTool tool, string inputJson, CancellationToken cancellationToken)
    {
        string text;
        var isError = false;
        try
        {
            using var input = JsonDocument.Parse(inputJson);
            text = await tool.Handler(input.RootElement.Clone()).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            isError = true;
            text = ex.Message;
        }

        EmitToolResult(toolUseId, text);
        await _client
            .SendEventsAsync(_sessionId, [ManagedAgentsSessionEvents.CustomToolResult(toolUseId, text, isError)], cancellationToken)
            .ConfigureAwait(false);
        _ackedToolUses.Add(toolUseId);
    }

    private void HandleSessionError(JsonElement error)
    {
        var retryType = error.TryGetProperty("retry_status", out var retryStatus) && retryStatus.ValueKind == JsonValueKind.Object
            ? StringPropertyOf(retryStatus, "type")
            : null;
        if (string.Equals(retryType, "retrying", StringComparison.Ordinal))
        {
            return; // transient; the session recovers on its own
        }

        var message = StringPropertyOf(error, "message") ?? "The session reported an error.";
        Fail(message, StringPropertyOf(error, "type"));
    }

    private async Task HandleIdleAsync(StopReason stopReason, CancellationToken cancellationToken)
    {
        if (stopReason.TryPickBetaManagedAgentsSessionEndTurn(out _))
        {
            CloseAll();
            Finish(ManagedAgentsTurnStatus.Finished);
            return;
        }

        if (stopReason.TryPickBetaManagedAgentsSessionRetriesExhausted(out _))
        {
            Fail("The session gave up after exhausting its retries.", "retries_exhausted");
            return;
        }

        if (!stopReason.TryPickBetaManagedAgentsSessionRequiresAction(out var requiresAction))
        {
            return;
        }

        // requires_action: work out what the session is blocked on.
        var blockedOn = requiresAction.EventIds.Where(id => !_ackedToolUses.Contains(id)).ToList();
        if (blockedOn.Count == 0)
        {
            return; // everything is already answered; wait for it to resume
        }

        var confirmations = blockedOn.Where(id => _askedConfirmations.Contains(id)).ToList();
        if (confirmations.Count > 0)
        {
            if (_toolConfirmation is null)
            {
                await InterruptAsync().ConfigureAwait(false);
                Fail(
                    "A tool requires confirmation but no confirmation policy is configured. " +
                    "Set `ToolConfirmation` to \"allow\" or \"deny\", or use a permission policy that does not ask.",
                    "tool_confirmation_required");
                return;
            }

            var events = confirmations
                .Select(id => ManagedAgentsSessionEvents.ToolConfirmation(id, _toolConfirmation))
                .ToList();
            await _client.SendEventsAsync(_sessionId, events, cancellationToken).ConfigureAwait(false);
            foreach (var id in confirmations)
            {
                _ackedToolUses.Add(id);
            }

            if (confirmations.Count == blockedOn.Count)
            {
                return;
            }
        }

        var clientToolUseIds = blockedOn.Where(id => _clientParks.Contains(id)).ToList();
        var unknown = blockedOn.Where(id => !_askedConfirmations.Contains(id) && !_clientParks.Contains(id)).ToList();
        if (unknown.Count > 0)
        {
            await InterruptAsync().ConfigureAwait(false);
            Fail("The agent is waiting on an action this integration cannot answer.", "unsupported_action");
            return;
        }

        if (clientToolUseIds.Count > 0)
        {
            // Hand control back to the frontend to execute its tools.
            CloseAll();
            Outcome.ClientToolUseIds = clientToolUseIds;
            Finish(ManagedAgentsTurnStatus.Parked);
        }
    }

    private async Task InterruptAsync()
    {
        try
        {
            await _client
                .SendEventsAsync(_sessionId, [ManagedAgentsSessionEvents.Interrupt()], CancellationToken.None)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Best effort: the run is already failing.
        }
    }

    private void EmitToolCall(string toolCallId, string toolCallName, string inputJson)
    {
        Emit(new ToolCallStartEvent { ToolCallId = toolCallId, ToolCallName = toolCallName });
        Emit(new ToolCallArgsEvent { ToolCallId = toolCallId, Delta = inputJson });
        Emit(new ToolCallEndEvent { ToolCallId = toolCallId });
    }

    private void EmitToolResult(string toolUseId, string content)
    {
        Emit(new ToolCallResultEvent
        {
            MessageId = $"result_{toolUseId}",
            ToolCallId = toolUseId,
            Content = content,
            Role = AGUIRoles.Tool,
        });
    }

    private void Fail(string message, string? code = null)
    {
        CloseAll();
        Emit(new RunErrorEvent { Message = message, Code = code });
        Finish(ManagedAgentsTurnStatus.Errored);
    }

    private void Finish(ManagedAgentsTurnStatus status, bool sessionEnded = false)
    {
        Outcome.Status = status;
        Outcome.SessionEnded = sessionEnded;
        _done = true;
    }

    private void CloseMessage(string messageId)
    {
        Emit(new TextMessageEndEvent { MessageId = messageId });
        _previews.Remove(messageId);
        _closedMessages.Add(messageId);
    }

    private void OpenReasoning(string messageId)
    {
        if (_openReasoning.Add(messageId))
        {
            _openReasoningOrder.Add(messageId);
        }
    }

    private void CloseReasoning(string messageId)
    {
        Emit(new ReasoningMessageEndEvent { MessageId = messageId });
        Emit(new ReasoningEndEvent { MessageId = messageId });
        _openReasoning.Remove(messageId);
        _openReasoningOrder.Remove(messageId);
    }

    private void CloseAll()
    {
        foreach (var messageId in _previews.Keys.ToList())
        {
            CloseMessage(messageId);
        }

        foreach (var reasoningId in _openReasoningOrder.ToList())
        {
            CloseReasoning(reasoningId);
        }
    }

    private static string InputJsonOf(JsonElement rawEvent)
    {
        if (rawEvent.ValueKind == JsonValueKind.Object
            && rawEvent.TryGetProperty("input", out var input)
            && input.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null))
        {
            return JsonSerializer.Serialize(input);
        }

        return "{}";
    }

    private static IEnumerable<JsonElement>? ContentBlocksOf(JsonElement rawEvent)
    {
        if (rawEvent.ValueKind == JsonValueKind.Object
            && rawEvent.TryGetProperty("content", out var content)
            && content.ValueKind == JsonValueKind.Array)
        {
            return content.EnumerateArray().Select(static block => block.Clone());
        }

        return null;
    }

    private static string DescribeContent(JsonElement rawEvent)
    {
        return ManagedAgentsText.Truncate(ManagedAgentsText.DescribeToolResult(ContentBlocksOf(rawEvent)), MaxToolResultLength);
    }

    private static string? TextDeltaOf(JsonElement rawEvent)
    {
        if (rawEvent.ValueKind != JsonValueKind.Object
            || !rawEvent.TryGetProperty("delta", out var delta)
            || delta.ValueKind != JsonValueKind.Object
            || !string.Equals(StringPropertyOf(delta, "type"), "content_delta", StringComparison.Ordinal)
            || !delta.TryGetProperty("content", out var content)
            || content.ValueKind != JsonValueKind.Object
            || !string.Equals(StringPropertyOf(content, "type"), "text", StringComparison.Ordinal))
        {
            return null;
        }

        return StringPropertyOf(content, "text");
    }

    private static string? StringPropertyOf(JsonElement element, string name)
    {
        return element.ValueKind == JsonValueKind.Object
            && element.TryGetProperty(name, out var property)
            && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;
    }
}
