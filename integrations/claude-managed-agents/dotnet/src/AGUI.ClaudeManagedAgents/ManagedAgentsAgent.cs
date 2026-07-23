using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text.Json;
using AGUI.Abstractions;

namespace AGUI.ClaudeManagedAgents;

/// <summary>
/// An AG-UI agent backed by Claude Managed Agents. Each AG-UI thread maps to one managed
/// session; each run drives one turn of that session and streams its events as AG-UI events.
/// </summary>
public sealed class ManagedAgentsAgent
{
    /// <summary>
    /// The <c>CUSTOM</c> event name emitted with the new session ID when a session is created.
    /// </summary>
    public const string SessionCustomEventName = "managed_agents.session";

    private readonly ManagedAgentsAgentOptions _options;
    private readonly IManagedAgentsClient _client;
    private readonly ISessionStore _store;
    private readonly IReadOnlyDictionary<string, ManagedAgentsBackendTool> _backendTools;

    // A thread runs one turn at a time. Keys are scoped to this managed agent so distinct
    // agents never collide.
    private readonly ConcurrentDictionary<string, byte> _busyThreads = new(StringComparer.Ordinal);

    /// <summary>
    /// Initializes a new instance of the <see cref="ManagedAgentsAgent"/> class.
    /// </summary>
    /// <param name="options">The agent configuration.</param>
    public ManagedAgentsAgent(ManagedAgentsAgentOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (string.IsNullOrEmpty(options.AgentId))
        {
            throw new ArgumentException("AgentId is required.", nameof(options));
        }

        if (string.IsNullOrEmpty(options.EnvironmentId))
        {
            throw new ArgumentException("EnvironmentId is required.", nameof(options));
        }

        _options = options;
        _client = options.Client ?? new AnthropicManagedAgentsClient(options.AnthropicClient ?? new Anthropic.AnthropicClient());
        _store = options.SessionStore ?? new InMemorySessionStore();
        _backendTools = options.BackendTools.ToDictionary(
            static tool => ManagedAgentsCustomTools.NormalizeToolName(tool.Name),
            static tool => tool,
            StringComparer.Ordinal);
    }

    /// <summary>
    /// Runs one turn for the input's thread and streams the resulting AG-UI events.
    /// </summary>
    /// <param name="input">The AG-UI run input.</param>
    /// <param name="cancellationToken">A token that aborts the run, for example when the client disconnects.</param>
    /// <returns>The AG-UI event stream for the run.</returns>
    public IAsyncEnumerable<BaseEvent> RunAsync(RunAgentInput input, CancellationToken cancellationToken = default)
    {
        return RunAsync(input, ownerId: null, cancellationToken);
    }

    /// <summary>
    /// Runs one turn for the input's thread on behalf of <paramref name="ownerId"/> and streams
    /// the resulting AG-UI events.
    /// </summary>
    /// <param name="input">The AG-UI run input.</param>
    /// <param name="ownerId">
    /// The authenticated caller that owns the thread, taken from the host's authenticated principal
    /// (never from a client-supplied value). When set, the thread↔session mapping is scoped to this
    /// owner so one caller cannot resume, mutate, or evict another caller's session by guessing a
    /// thread ID. Leave <see langword="null"/> only for single-user or already-scoped deployments.
    /// </param>
    /// <param name="cancellationToken">A token that aborts the run, for example when the client disconnects.</param>
    /// <returns>The AG-UI event stream for the run.</returns>
    public async IAsyncEnumerable<BaseEvent> RunAsync(
        RunAgentInput input,
        string? ownerId,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);

        var threadId = input.ThreadId;
        var runId = input.RunId;
        yield return new RunStartedEvent { ThreadId = threadId, RunId = runId };
        if (input.State is JsonElement state && state.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null))
        {
            yield return new StateSnapshotEvent { Snapshot = state };
        }

        var threadKey = ThreadKey(threadId, ownerId);
        var busyKey = $"{_options.AgentId}:{threadKey}";
        if (!_busyThreads.TryAdd(busyKey, 0))
        {
            yield return new RunErrorEvent { Message = "A run is already in progress on this thread." };
            yield break;
        }

        // Check for something sendable before touching the API, so a malformed run does not
        // create an orphan session.
        if (!HasSendableContent(input.Messages))
        {
            _busyThreads.TryRemove(busyKey, out _);
            yield return new RunErrorEvent { Message = "There is nothing to send: this run has no user message or tool result." };
            yield break;
        }

        var run = new RunContext();
        using var timeout = new CancellationTokenSource(_options.TurnTimeout);
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeout.Token);
        try
        {
            var events = RunCoreAsync(input, threadKey, run, linked.Token).GetAsyncEnumerator(CancellationToken.None);
            await using var eventsScope = events.ConfigureAwait(false);
            BaseEvent? errorEvent = null;
            while (true)
            {
                BaseEvent current;
                try
                {
                    if (!await events.MoveNextAsync().ConfigureAwait(false))
                    {
                        break;
                    }

                    current = events.Current;
                }
                catch (OperationCanceledException) when (linked.IsCancellationRequested)
                {
                    // Interrupt the session so it does not keep working on a turn nobody hears.
                    await InterruptAsync(run.SessionId).ConfigureAwait(false);
                    if (cancellationToken.IsCancellationRequested)
                    {
                        // The client went away; there is nobody left to tell.
                        yield break;
                    }

                    errorEvent = new RunErrorEvent
                    {
                        Message = $"The turn exceeded the {(int)Math.Round(_options.TurnTimeout.TotalSeconds)}s limit and was interrupted.",
                    };
                    break;
                }
                catch (Exception ex)
                {
                    errorEvent = new RunErrorEvent { Message = string.IsNullOrEmpty(ex.Message) ? "The run failed." : ex.Message };
                    break;
                }

                yield return current;
            }

            if (errorEvent is not null)
            {
                // Close any block the failed turn left open before reporting the error.
                if (run.Turn is not null)
                {
                    foreach (var evt in run.Turn.CloseOpenBlocks())
                    {
                        yield return evt;
                    }
                }

                yield return errorEvent;
            }
        }
        finally
        {
            _busyThreads.TryRemove(busyKey, out _);
        }
    }

    private async IAsyncEnumerable<BaseEvent> RunCoreAsync(
        RunAgentInput input,
        string threadKey,
        RunContext run,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var record = await _store.GetAsync(threadKey, cancellationToken).ConfigureAwait(false);
        if (record is null)
        {
            // A tool result only answers a pending call, so it cannot start a thread: creating
            // a session for it would orphan the session.
            if (!HasUserText(input.Messages))
            {
                yield return new RunErrorEvent { Message = "There is nothing to send: a tool result arrived for a thread with no session." };
                yield break;
            }

            // The busy-thread gate serializes runs per thread, so creation cannot race.
            record = await CreateSessionAsync(input, cancellationToken).ConfigureAwait(false);
            await _store.SetAsync(threadKey, record, cancellationToken).ConfigureAwait(false);
            yield return new CustomEvent
            {
                Name = SessionCustomEventName,
                Value = JsonSerializer.SerializeToElement(new SessionEventValue(record.SessionId, input.ThreadId)),
            };
        }

        run.SessionId = record.SessionId;
        await SyncClientToolsAsync(record, input.Tools, cancellationToken).ConfigureAwait(false);

        var outbound = OutboundEvents(record, input.Messages);
        if (outbound.Events.Count == 0)
        {
            yield return new RunErrorEvent { Message = "There is nothing new to send: no user message or tool result in this run." };
            yield break;
        }

        // Some parked tool calls are still unanswered: post what we have and stay parked
        // instead of waiting on a session that will not resume.
        if (outbound.StillParked.Count > 0)
        {
            await _client.SendEventsAsync(record.SessionId, outbound.Events, cancellationToken).ConfigureAwait(false);
            record.PendingClientToolUseIds = outbound.StillParked;
            record.LastUserMessageId = outbound.LastUserMessageId ?? record.LastUserMessageId;
            await _store.SetAsync(threadKey, record, cancellationToken).ConfigureAwait(false);
            yield return new RunFinishedEvent { ThreadId = input.ThreadId, RunId = input.RunId };
            yield break;
        }

        // Normalized names may collide (e.g. "search.web" and "search_web");
        // last write wins rather than throwing on a duplicate key.
        var clientTools = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var tool in input.Tools ?? [])
        {
            clientTools[ManagedAgentsCustomTools.NormalizeToolName(tool.Name)] = tool.Name;
        }

        var turn = new ManagedAgentsTurn(
            _client,
            record.SessionId,
            outbound.Events,
            clientTools,
            _backendTools,
            _options.ToolConfirmation,
            _options.StreamDeltas,
            // Persist delivery as soon as the events land, so a timeout or disconnect later in
            // the turn does not re-post them next run.
            onSent: async ct =>
            {
                record.PendingClientToolUseIds = [];
                if (outbound.LastUserMessageId is not null)
                {
                    record.LastUserMessageId = outbound.LastUserMessageId;
                }

                await _store.SetAsync(threadKey, record, ct).ConfigureAwait(false);
            });
        run.Turn = turn;

        await foreach (var evt in turn.RunAsync(cancellationToken).ConfigureAwait(false))
        {
            yield return evt;
        }

        var outcome = turn.Outcome;
        await RecordOutcomeAsync(threadKey, record, outcome, cancellationToken).ConfigureAwait(false);
        if (outcome.Status != ManagedAgentsTurnStatus.Errored)
        {
            yield return new RunFinishedEvent { ThreadId = input.ThreadId, RunId = input.RunId };
        }
    }

    private async Task RecordOutcomeAsync(
        string threadKey,
        ManagedAgentsSessionRecord record,
        ManagedAgentsTurnOutcome outcome,
        CancellationToken cancellationToken)
    {
        if (outcome.Status == ManagedAgentsTurnStatus.Errored && outcome.SessionEnded)
        {
            await _store.DeleteAsync(threadKey, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (outcome.Status != ManagedAgentsTurnStatus.Parked)
        {
            return;
        }

        record.PendingClientToolUseIds = outcome.ClientToolUseIds;
        await _store.SetAsync(threadKey, record, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Works out what to post into the session for this run: results for any tool calls the
    /// frontend was asked to run, plus every user message not yet delivered (in order).
    /// </summary>
    private static OutboundEventSet OutboundEvents(ManagedAgentsSessionRecord record, IList<AGUIMessage> messages)
    {
        var events = new List<JsonElement>();
        var pending = new HashSet<string>(record.PendingClientToolUseIds, StringComparer.Ordinal);
        var pendingOrder = new List<string>(record.PendingClientToolUseIds);

        foreach (var message in messages)
        {
            if (message is not AGUIToolMessage toolMessage || !pending.Contains(toolMessage.ToolCallId))
            {
                continue;
            }

            events.Add(ManagedAgentsSessionEvents.CustomToolResult(
                toolMessage.ToolCallId,
                toolMessage.Content,
                isError: !string.IsNullOrEmpty(toolMessage.Error)));
            pending.Remove(toolMessage.ToolCallId);
            pendingOrder.Remove(toolMessage.ToolCallId);
        }

        // User messages after the last delivered one; on first contact, just the newest.
        string? lastUserMessageId = null;
        var userMessages = messages.OfType<AGUIUserMessage>().ToList();
        var deliveredIndex = record.LastUserMessageId is null
            ? -1
            : userMessages.FindIndex(message => string.Equals(message.Id, record.LastUserMessageId, StringComparison.Ordinal));
        var undelivered = deliveredIndex >= 0
            ? userMessages.Skip(deliveredIndex + 1)
            : userMessages.Skip(Math.Max(0, userMessages.Count - 1));
        foreach (var message in undelivered)
        {
            var text = UserTextOf(message).Trim();
            if (text.Length == 0)
            {
                continue;
            }

            events.Add(ManagedAgentsSessionEvents.UserMessage(text));
            lastUserMessageId = message.Id;
        }

        // The user moved on without answering the tools the frontend was asked to run: fail
        // those calls so the agent can respond to the new message.
        if (lastUserMessageId is not null && pending.Count > 0)
        {
            // Prepend one at a time, so the last pending call ends up first (matching the
            // reference implementation's unshift order).
            events.InsertRange(0, pendingOrder.AsEnumerable().Reverse().Select(static toolUseId => ManagedAgentsSessionEvents.CustomToolResult(
                toolUseId,
                "The user did not provide a result for this tool call.",
                isError: true)));
            pending.Clear();
            pendingOrder.Clear();
        }

        return new OutboundEventSet(events, pendingOrder, lastUserMessageId);
    }

    private async Task<ManagedAgentsSessionRecord> CreateSessionAsync(RunAgentInput input, CancellationToken cancellationToken)
    {
        var customTools = CustomToolsFor(input.Tools);
        var request = new ManagedAgentSessionRequest
        {
            AgentId = _options.AgentId,
            AgentVersion = _options.AgentVersion,
            EnvironmentId = _options.EnvironmentId,
            Title = _options.SessionTitle?.Invoke(input.ThreadId) ?? $"AG-UI thread {input.ThreadId}",
        };

        if (customTools.Count > 0)
        {
            // Overrides replace the tool list, so keep the agent's own tools.
            var baseTools = await BaseToolsAsync(cancellationToken).ConfigureAwait(false);
            request.OverrideTools = baseTools.Concat(customTools.Values).ToList();
        }

        var sessionId = await _client.CreateSessionAsync(request, cancellationToken).ConfigureAwait(false);
        return new ManagedAgentsSessionRecord
        {
            SessionId = sessionId,
            ToolNames = customTools.Keys.ToList(),
            PendingClientToolUseIds = [],
        };
    }

    /// <summary>
    /// Frontend tools plus configured backend tools, as custom tool definitions keyed by
    /// normalized name in registration order. On a collision the frontend tool wins, matching
    /// dispatch order in the turn loop.
    /// </summary>
    private OrderedToolMap CustomToolsFor(IList<AGUITool>? clientTools)
    {
        var tools = new OrderedToolMap();
        foreach (var tool in _options.BackendTools)
        {
            tools.Set(ManagedAgentsCustomTools.NormalizeToolName(tool.Name), ManagedAgentsCustomTools.CustomToolFrom(tool.Name, tool.Description, tool.Parameters));
        }

        foreach (var tool in clientTools ?? [])
        {
            var custom = ManagedAgentsCustomTools.CustomToolFrom(tool.Name, tool.Description, tool.Parameters);
            tools.Set(ManagedAgentsCustomTools.NameOf(custom) ?? ManagedAgentsCustomTools.NormalizeToolName(tool.Name), custom);
        }

        return tools;
    }

    /// <summary>
    /// Registers any client tools the session's agent does not yet have. The tool list is a
    /// full replacement, so it is merged with what the agent has.
    /// </summary>
    private async Task SyncClientToolsAsync(ManagedAgentsSessionRecord record, IList<AGUITool>? clientTools, CancellationToken cancellationToken)
    {
        var desired = CustomToolsFor(clientTools);
        var known = new HashSet<string>(record.ToolNames, StringComparer.Ordinal);
        if (desired.Keys.All(known.Contains))
        {
            return;
        }

        var baseTools = await BaseToolsAsync(cancellationToken).ConfigureAwait(false);
        var tools = baseTools.Concat(desired.Values).ToList();
        await _client.UpdateSessionToolsAsync(record.SessionId, tools, cancellationToken).ConfigureAwait(false);
        record.ToolNames = desired.Keys.ToList();
    }

    /// <summary>
    /// The tools defined on the managed agent itself, fetched fresh so console edits apply.
    /// </summary>
    private Task<IReadOnlyList<JsonElement>> BaseToolsAsync(CancellationToken cancellationToken)
    {
        return _client.GetAgentToolsAsync(_options.AgentId, _options.AgentVersion, cancellationToken);
    }

    private async Task InterruptAsync(string? sessionId)
    {
        if (sessionId is null)
        {
            return;
        }

        try
        {
            await _client
                .SendEventsAsync(sessionId, [ManagedAgentsSessionEvents.Interrupt()], CancellationToken.None)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Best effort: the run is already ending.
        }
    }

    /// <summary>
    /// Whether the run carries a user message with text or a tool result.
    /// </summary>
    private static bool HasSendableContent(IList<AGUIMessage> messages)
    {
        return messages.Any(static message => message is AGUIToolMessage) || HasUserText(messages);
    }

    /// <summary>
    /// Whether the run carries a user message with text, the only thing that can start a thread.
    /// </summary>
    private static bool HasUserText(IList<AGUIMessage> messages)
    {
        return messages.Any(static message => message is AGUIUserMessage user && UserTextOf(user).Trim().Length > 0);
    }

    private static string UserTextOf(AGUIUserMessage message)
    {
        return string.Concat(message.Content.OfType<AGUITextInputContent>().Select(static part => part.Text));
    }

    private static string ThreadKey(string threadId, string? ownerId)
    {
        return ownerId is null ? threadId : $"{ownerId}:{threadId}";
    }

    private sealed class RunContext
    {
        internal string? SessionId { get; set; }

        internal ManagedAgentsTurn? Turn { get; set; }
    }

    private sealed class OutboundEventSet
    {
        internal OutboundEventSet(List<JsonElement> events, List<string> stillParked, string? lastUserMessageId)
        {
            Events = events;
            StillParked = stillParked;
            LastUserMessageId = lastUserMessageId;
        }

        internal List<JsonElement> Events { get; }

        internal List<string> StillParked { get; }

        internal string? LastUserMessageId { get; }
    }

    private sealed class SessionEventValue
    {
        internal SessionEventValue(string sessionId, string threadId)
        {
            SessionId = sessionId;
            ThreadId = threadId;
        }

        [System.Text.Json.Serialization.JsonPropertyName("sessionId")]
        public string SessionId { get; }

        [System.Text.Json.Serialization.JsonPropertyName("threadId")]
        public string ThreadId { get; }
    }

    /// <summary>
    /// A name → tool map that preserves first-registration order while letting a later
    /// registration replace the definition.
    /// </summary>
    private sealed class OrderedToolMap
    {
        private readonly List<string> _order = [];
        private readonly Dictionary<string, JsonElement> _tools = new(StringComparer.Ordinal);

        internal void Set(string name, JsonElement tool)
        {
            if (!_tools.ContainsKey(name))
            {
                _order.Add(name);
            }

            _tools[name] = tool;
        }

        internal int Count => _order.Count;

        internal IEnumerable<string> Keys => _order;

        internal IEnumerable<JsonElement> Values => _order.Select(name => _tools[name]);
    }
}
