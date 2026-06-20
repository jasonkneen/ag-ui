namespace AGUI.Server;

/// <summary>
/// Well-known string constants used by the AG-UI ASP.NET Core hosting layer.
/// </summary>
public static class AGUIConstants
{
    /// <summary>
    /// The key used by <see cref="RunAgentInputExtensions.ToChatRequestContext"/> to stash the
    /// originating <see cref="AGUI.Abstractions.RunAgentInput"/> inside
    /// <see cref="Microsoft.Extensions.AI.ChatOptions.AdditionalProperties"/>. Delegating
    /// <see cref="Microsoft.Extensions.AI.IChatClient"/> implementations can use this key to
    /// recover the AG-UI input without taking a hard dependency on the hosting layer's
    /// <see cref="ChatRequestContext"/>.
    /// </summary>
    public const string RunAgentInputKey = "agui_input";
}
