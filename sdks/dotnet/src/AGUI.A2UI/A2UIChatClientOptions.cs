using AGUI.Server;
using Microsoft.Extensions.AI;

namespace AGUI.A2UI;

/// <summary>
/// Options for <see cref="A2UIChatClient"/>: whether to inject the <c>generate_a2ui</c>
/// tool, plus the shared toolkit knobs (tool name/description, guidelines, default surface
/// and catalog ids, validation catalog, recovery config, and the per-attempt callback).
/// </summary>
public sealed class A2UIChatClientOptions
{
    /// <summary>
    /// Gets whether to inject the <c>generate_a2ui</c> tool. <see langword="null"/> (the
    /// default) means "auto": the per-run forwarded <c>injectA2UITool</c> flag wins when
    /// present, otherwise injection is on because wrapping with <see cref="A2UIChatClient"/>
    /// is itself the opt-in. An explicit value here is the backend override and is used only
    /// when the run forwards no flag, so a client-sent <see langword="false"/> still wins.
    /// </summary>
    public bool? InjectA2UITool { get; init; }

    /// <summary>
    /// Gets the shared toolkit parameters (tool name/description, guidelines, default surface
    /// and catalog ids, validation catalog, recovery config, per-attempt callback). Defaults
    /// are filled per the shared toolkit rules when unset.
    /// </summary>
    public A2UIToolParams? ToolParams { get; init; }

    /// <summary>
    /// Gets an optional extractor that reads provider-native streamed tool-call argument
    /// fragments off a render sub-agent update (the same kind registered on the server's
    /// <see cref="AGUIStreamOptions.MapStreamingToolCallArguments"/>). It lets the adapter learn
    /// the streamed <c>render_a2ui</c> call id before the coalesced tool call arrives, so a
    /// generation attempt that streams fragments and then fails mid-stream can still emit a
    /// balancing tool result and leave the persisted conversation valid. When unset, balancing
    /// falls back to the coalesced call id (which is unavailable on a mid-stream failure).
    /// </summary>
    public Func<ChatResponseUpdate, IEnumerable<AGUIToolCallArgumentFragment>?>? StreamingToolCallArgumentExtractor { get; init; }
}
