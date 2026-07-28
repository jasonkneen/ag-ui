using AGUI.A2UI;

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
}
