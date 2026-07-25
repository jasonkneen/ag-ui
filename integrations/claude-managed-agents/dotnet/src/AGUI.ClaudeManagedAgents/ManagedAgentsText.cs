using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AGUI.ClaudeManagedAgents;

/// <summary>
/// Helpers for turning Managed Agents content blocks into display text.
/// </summary>
internal static partial class ManagedAgentsText
{
    private const string ReplacementCharacter = "�";

    /// <summary>
    /// Concatenates the text of every <c>text</c> block.
    /// </summary>
    internal static string TextOf(IEnumerable<JsonElement>? blocks)
    {
        if (blocks is null)
        {
            return string.Empty;
        }

        var builder = new StringBuilder();
        foreach (var block in blocks)
        {
            if (IsType(block, "text") && TryGetString(block, "text", out var text))
            {
                builder.Append(text);
            }
        }

        return builder.ToString();
    }

    /// <summary>
    /// Flattens the mixed block types of a tool result (text, search results, images,
    /// documents) into a readable string for the UI.
    /// </summary>
    internal static string DescribeToolResult(IEnumerable<JsonElement>? blocks)
    {
        if (blocks is null)
        {
            return string.Empty;
        }

        var parts = new List<string>();
        foreach (var block in blocks)
        {
            if (IsType(block, "text") && TryGetString(block, "text", out var text))
            {
                parts.Add(DecodeEntities(text));
                continue;
            }

            if (IsType(block, "search_result"))
            {
                var inner = block.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.Array
                    ? TextOf(content.EnumerateArray())
                    : string.Empty;
                var title = DecodeEntities(TryGetString(block, "title", out var t) ? t : string.Empty);
                var source = TryGetString(block, "source", out var s) ? s : string.Empty;
                var summary = inner.Length == 0
                    ? string.Empty
                    : "\n" + Truncate(DecodeEntities(inner), ManagedAgentsLimits.SearchResultPreviewChars);
                parts.Add($"[search result] {title} — {source}{summary}");
                continue;
            }

            var type = TryGetString(block, "type", out var blockType) ? blockType : "unknown";
            parts.Add($"[{type}]");
        }

        return string.Join("\n", parts).Trim();
    }

    /// <summary>
    /// Truncates <paramref name="value"/> to at most <paramref name="length"/> UTF-16 units.
    /// </summary>
    internal static string Truncate(string value, int length)
    {
        return value.Length <= length ? value : value.Substring(0, length);
    }

    private static bool IsType(JsonElement block, string type)
    {
        return block.ValueKind == JsonValueKind.Object
            && TryGetString(block, "type", out var actual)
            && string.Equals(actual, type, StringComparison.Ordinal);
    }

    private static bool TryGetString(JsonElement element, string name, out string value)
    {
        if (element.ValueKind == JsonValueKind.Object
            && element.TryGetProperty(name, out var property)
            && property.ValueKind == JsonValueKind.String)
        {
            value = property.GetString() ?? string.Empty;
            return true;
        }

        value = string.Empty;
        return false;
    }

    private static string DecodeEntities(string value)
    {
        var decoded = HexEntity().Replace(value, match => CodePointOf(match.Groups[1].Value, NumberStyles.HexNumber));
        decoded = DecimalEntity().Replace(decoded, match => CodePointOf(match.Groups[1].Value, NumberStyles.Integer));
        return decoded
            .Replace("&quot;", "\"", StringComparison.Ordinal)
            .Replace("&lt;", "<", StringComparison.Ordinal)
            .Replace("&gt;", ">", StringComparison.Ordinal)
            .Replace("&amp;", "&", StringComparison.Ordinal);
    }

    private static string CodePointOf(string digits, NumberStyles style)
    {
        if (!long.TryParse(digits, style, CultureInfo.InvariantCulture, out var codePoint))
        {
            return ReplacementCharacter;
        }

        if (codePoint < 0 || codePoint > 0x10FFFF || (codePoint >= 0xD800 && codePoint <= 0xDFFF))
        {
            return ReplacementCharacter;
        }

        return char.ConvertFromUtf32((int)codePoint);
    }

    [GeneratedRegex("&#x([0-9a-fA-F]+);")]
    private static partial Regex HexEntity();

    [GeneratedRegex("&#(\\d+);")]
    private static partial Regex DecimalEntity();
}
