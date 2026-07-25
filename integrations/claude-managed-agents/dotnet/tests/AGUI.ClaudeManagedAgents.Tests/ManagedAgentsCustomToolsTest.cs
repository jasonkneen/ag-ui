using Xunit;

namespace AGUI.ClaudeManagedAgents.Tests;

public class ManagedAgentsCustomToolsTest
{
    [Fact]
    public void KeepsValidToolNames()
    {
        Assert.Equal("show_chart", ManagedAgentsCustomTools.NormalizeToolName("show_chart"));
        Assert.Equal("Get-Weather_2", ManagedAgentsCustomTools.NormalizeToolName("Get-Weather_2"));
    }

    [Fact]
    public void ReplacesInvalidCharactersAndTruncatesLongNames()
    {
        Assert.Equal("search_web_", ManagedAgentsCustomTools.NormalizeToolName("search web!"));
        Assert.Equal(new string('x', ManagedAgentsLimits.ToolNameMaxLength), ManagedAgentsCustomTools.NormalizeToolName(new string('x', 200)));
        Assert.Equal("tool", ManagedAgentsCustomTools.NormalizeToolName(string.Empty));
    }

    [Fact]
    public void BuildsAnInputSchemaAndADefaultDescription()
    {
        var tool = ManagedAgentsCustomTools.CustomToolFrom(
            "ping",
            description: string.Empty,
            FakeManagedAgentsClient.Json("""{"properties":{"a":{"type":"string"}}}"""));

        var expected = FakeManagedAgentsClient.Json(
            """
            {
              "type": "custom",
              "name": "ping",
              "description": "Tool ping",
              "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
            }
            """);
        Assert.True(System.Text.Json.JsonElement.DeepEquals(expected, tool), tool.GetRawText());
    }

    [Fact]
    public void HandlesMissingParametersAndNormalizesTheName()
    {
        var tool = ManagedAgentsCustomTools.CustomToolFrom("lookup docs", "Lookup", parameters: default);

        Assert.Equal("lookup_docs", ManagedAgentsCustomTools.NameOf(tool));
        Assert.Equal("""{"type":"object","properties":{}}""", tool.GetProperty("input_schema").GetRawText());
    }

    [Fact]
    public void TruncatesLongDescriptions()
    {
        var description = new string('d', ManagedAgentsLimits.ToolDescriptionMaxLength + 10);
        var tool = ManagedAgentsCustomTools.CustomToolFrom("ping", description, parameters: default);

        Assert.Equal(ManagedAgentsLimits.ToolDescriptionMaxLength, tool.GetProperty("description").GetString()!.Length);
    }
}
