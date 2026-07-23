namespace AGUI.ClaudeManagedAgents;

/// <summary>
/// How a turn ended.
/// </summary>
internal enum ManagedAgentsTurnStatus
{
    /// <summary>The session went idle after the reply completed.</summary>
    Finished,

    /// <summary>The session is parked on custom tool calls the frontend must answer.</summary>
    Parked,

    /// <summary>A <c>RUN_ERROR</c> was already emitted.</summary>
    Errored,
}

/// <summary>
/// The result of driving one turn of a managed session.
/// </summary>
internal sealed class ManagedAgentsTurnOutcome
{
    internal ManagedAgentsTurnStatus Status { get; set; } = ManagedAgentsTurnStatus.Errored;

    /// <summary>Custom tool calls the frontend must answer when <see cref="Status"/> is parked.</summary>
    internal IList<string> ClientToolUseIds { get; set; } = [];

    /// <summary>Whether the session itself ended on the server (terminated or deleted).</summary>
    internal bool SessionEnded { get; set; }
}
