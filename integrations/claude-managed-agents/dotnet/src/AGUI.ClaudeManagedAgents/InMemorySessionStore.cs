using System.Collections.Concurrent;

namespace AGUI.ClaudeManagedAgents;

/// <summary>
/// In-memory thread↔session store. Mappings are lost on restart.
/// </summary>
public sealed class InMemorySessionStore : ISessionStore
{
    private readonly ConcurrentDictionary<string, ManagedAgentsSessionRecord> _records = new(StringComparer.Ordinal);

    /// <summary>
    /// A defensive copy of a record. The store must not hand out a reference to the record it
    /// holds: the agent mutates records in place between persists, so an aliased record would
    /// make an unpersisted mutation indistinguishable from a persisted one — and a dropped write
    /// would only surface against a real out-of-process store.
    /// </summary>
    private static ManagedAgentsSessionRecord Copy(ManagedAgentsSessionRecord record) => new()
    {
        SessionId = record.SessionId,
        ToolNames = [.. record.ToolNames],
        ToolDefinitionsFingerprint = record.ToolDefinitionsFingerprint,
        LastUserMessageId = record.LastUserMessageId,
        PendingClientToolUseIds = [.. record.PendingClientToolUseIds],
    };

    /// <inheritdoc />
    public ValueTask<ManagedAgentsSessionRecord?> GetAsync(string threadKey, CancellationToken cancellationToken)
    {
        _records.TryGetValue(threadKey, out var record);
        return new ValueTask<ManagedAgentsSessionRecord?>(record is null ? null : Copy(record));
    }

    /// <inheritdoc />
    public ValueTask SetAsync(string threadKey, ManagedAgentsSessionRecord record, CancellationToken cancellationToken)
    {
        _records[threadKey] = Copy(record);
        return default;
    }

    /// <inheritdoc />
    public ValueTask DeleteAsync(string threadKey, CancellationToken cancellationToken)
    {
        _records.TryRemove(threadKey, out _);
        return default;
    }
}
