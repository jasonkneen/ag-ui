using System.Text.Json.Serialization;

namespace AGUI.Abstractions;

// Keep in sync with sdks/typescript/packages/core/src/events.ts
public sealed class RunErrorEvent : BaseEvent
{
    [JsonPropertyName("type")]
    public override string Type => AGUIEventTypes.RunError;

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("code")]
    public string? Code { get; set; }
}
