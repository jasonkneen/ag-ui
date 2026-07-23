namespace AGUI.ClaudeManagedAgents;

/// <summary>
/// Where thread↔session mappings live. The default is in-memory (lost on restart, in which case
/// a fresh session is created). Provide your own to survive restarts or run multiple replicas.
/// </summary>
/// <remarks>
/// Records are keyed by the value the agent derives from the run: the thread ID, scoped by the
/// owner ID when one is supplied to <see cref="ManagedAgentsAgent.RunAsync(AGUI.Abstractions.RunAgentInput, string?, CancellationToken)"/>.
/// </remarks>
public interface ISessionStore
{
    /// <summary>
    /// Gets the record stored under <paramref name="threadKey"/>, or <see langword="null"/> if none exists.
    /// </summary>
    ValueTask<ManagedAgentsSessionRecord?> GetAsync(string threadKey, CancellationToken cancellationToken);

    /// <summary>
    /// Stores <paramref name="record"/> under <paramref name="threadKey"/>, replacing any existing record.
    /// </summary>
    ValueTask SetAsync(string threadKey, ManagedAgentsSessionRecord record, CancellationToken cancellationToken);

    /// <summary>
    /// Removes the record stored under <paramref name="threadKey"/>, if any.
    /// </summary>
    ValueTask DeleteAsync(string threadKey, CancellationToken cancellationToken);
}
