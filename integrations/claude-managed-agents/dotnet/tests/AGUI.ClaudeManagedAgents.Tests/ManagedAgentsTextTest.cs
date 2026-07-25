using System.Text.Json;
using Xunit;

namespace AGUI.ClaudeManagedAgents.Tests;

public class ManagedAgentsTextTest
{
    private static IEnumerable<JsonElement> Blocks(string json)
    {
        return FakeManagedAgentsClient.Json(json).EnumerateArray().Select(static block => block.Clone()).ToList();
    }

    [Fact]
    public void TextOfJoinsOnlyTextBlocks()
    {
        Assert.Equal(
            "ab",
            ManagedAgentsText.TextOf(Blocks("""[{"type":"text","text":"a"},{"type":"image"},{"type":"text","text":"b"}]""")));
        Assert.Equal(string.Empty, ManagedAgentsText.TextOf(null));
    }

    [Fact]
    public void DescribeToolResultDecodesEntitiesAndSummarizesBlocks()
    {
        var described = ManagedAgentsText.DescribeToolResult(Blocks(
            """
            [
              {"type": "text", "text": "5 &lt; 6 &amp;&amp; &#x1F600; &#65;"},
              {"type": "search_result", "title": "T", "source": "https://x", "content": [{"type": "text", "text": "inner"}]},
              {"type": "document"}
            ]
            """));

        Assert.Equal("5 < 6 && \U0001F600 A\n[search result] T — https://x\ninner\n[document]", described);
    }

    [Fact]
    public void ReplacesLoneSurrogateEntitiesRatherThanEmittingInvalidText()
    {
        var described = ManagedAgentsText.DescribeToolResult(Blocks("""[{"type":"text","text":"a&#xD800;b&#55296;c"}]"""));

        Assert.Equal("a�b�c", described);
    }

    [Fact]
    public void TruncatesTheSearchResultBodyToItsPreviewLength()
    {
        var body = new string('x', ManagedAgentsLimits.SearchResultPreviewChars + 50);
        var described = ManagedAgentsText.DescribeToolResult(Blocks(
            $$"""[{"type":"search_result","title":"T","source":"s","content":[{"type":"text","text":"{{body}}"}]}]"""));

        Assert.Equal($"[search result] T — s\n{new string('x', ManagedAgentsLimits.SearchResultPreviewChars)}", described);
    }

    [Fact]
    public void UsesAnUnknownPlaceholderForABlockWithNoType()
    {
        Assert.Equal("[unknown]", ManagedAgentsText.DescribeToolResult(Blocks("""[{"data":"raw"}]""")));
    }

    [Fact]
    public void TruncatesLongText()
    {
        var text = new string('y', ManagedAgentsLimits.ToolResultMaxChars + 1);

        Assert.Equal(ManagedAgentsLimits.ToolResultMaxChars, ManagedAgentsText.Truncate(text, ManagedAgentsLimits.ToolResultMaxChars).Length);
        Assert.Equal("short", ManagedAgentsText.Truncate("short", ManagedAgentsLimits.ToolResultMaxChars));
    }
}
