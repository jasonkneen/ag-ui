using System.Collections.Concurrent;

namespace AGUI.ClaudeManagedAgents;

/// <summary>
/// In-memory thread↔session store. Mappings are lost on restart.
/// </summary>
public sealed class InMemorySessionStore : ISessionStore
{
    private readonly ConcurrentDictionary<string, ManagedAgentsSessionRecord> _records = new(StringComparer.Ordinal);

    /// <inheritdoc />
    public ValueTask<ManagedAgentsSessionRecord?> GetAsync(string threadKey, CancellationToken cancellationToken)
    {
        _records.TryGetValue(threadKey, out var record);
        return new ValueTask<ManagedAgentsSessionRecord?>(record);
    }

    /// <inheritdoc />
    public ValueTask SetAsync(string threadKey, ManagedAgentsSessionRecord record, CancellationToken cancellationToken)
    {
        _records[threadKey] = record;
        return default;
    }

    /// <inheritdoc />
    public ValueTask DeleteAsync(string threadKey, CancellationToken cancellationToken)
    {
        _records.TryRemove(threadKey, out _);
        return default;
    }
}
