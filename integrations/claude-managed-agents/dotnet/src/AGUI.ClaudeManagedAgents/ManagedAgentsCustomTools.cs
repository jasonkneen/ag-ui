using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace AGUI.ClaudeManagedAgents;

/// <summary>
/// Builds managed-agent custom tool definitions from AG-UI (frontend) and backend tools.
/// </summary>
internal static partial class ManagedAgentsCustomTools
{
    /// <summary>
    /// Managed Agents tool names allow only <c>[A-Za-z0-9_-]</c>, at most 128 characters.
    /// </summary>
    internal static string NormalizeToolName(string name)
    {
        if (ValidNamePattern().IsMatch(name))
        {
            return name;
        }

        var normalized = ManagedAgentsText.Truncate(InvalidNameCharacterPattern().Replace(name, "_"), ManagedAgentsLimits.ToolNameMaxLength);
        return normalized.Length == 0 ? "tool" : normalized;
    }

    /// <summary>
    /// Converts a tool definition into a managed-agent <c>custom</c> tool JSON object.
    /// </summary>
    internal static JsonElement CustomToolFrom(string name, string? description, JsonElement parameters)
    {
        var displayDescription = string.IsNullOrEmpty(description) ? $"Tool {name}" : description;
        var tool = new JsonObject
        {
            ["type"] = "custom",
            ["name"] = NormalizeToolName(name),
            ["description"] = ManagedAgentsText.Truncate(displayDescription, ManagedAgentsLimits.ToolDescriptionMaxLength),
            ["input_schema"] = InputSchemaFrom(parameters),
        };
        return JsonSerializer.SerializeToElement(tool);
    }

    /// <summary>
    /// Reads the <c>name</c> of a tool definition JSON object, or <see langword="null"/> if it has none.
    /// </summary>
    internal static string? NameOf(JsonElement tool)
    {
        if (tool.ValueKind == JsonValueKind.Object
            && tool.TryGetProperty("name", out var name)
            && name.ValueKind == JsonValueKind.String)
        {
            return name.GetString();
        }

        return null;
    }

    private static JsonObject InputSchemaFrom(JsonElement parameters)
    {
        var schema = new JsonObject { ["type"] = "object" };
        if (parameters.ValueKind != JsonValueKind.Object)
        {
            schema["properties"] = new JsonObject();
            return schema;
        }

        schema["properties"] = parameters.TryGetProperty("properties", out var properties) && properties.ValueKind == JsonValueKind.Object
            ? JsonNode.Parse(properties.GetRawText())
            : new JsonObject();
        schema["required"] = parameters.TryGetProperty("required", out var required) && required.ValueKind == JsonValueKind.Array
            ? JsonNode.Parse(required.GetRawText())
            : new JsonArray();
        return schema;
    }

    [GeneratedRegex("^[A-Za-z0-9_-]{1,128}$")]
    private static partial Regex ValidNamePattern();

    [GeneratedRegex("[^A-Za-z0-9_-]")]
    private static partial Regex InvalidNameCharacterPattern();
}
