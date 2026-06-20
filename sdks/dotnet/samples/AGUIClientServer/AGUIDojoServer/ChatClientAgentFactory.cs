using System.ComponentModel;
using System.Text.Json;
using AGUI.Abstractions;
using AGUI.Server;
using AGUIDojoServer.AgenticUI;
using AGUIDojoServer.BackendToolRendering;
using AGUIDojoServer.PredictiveStateUpdates;
using AGUIDojoServer.SharedState;
using System.ClientModel;
using Azure.AI.OpenAI;
using Azure.Identity;
using OpenAI;
using Microsoft.Extensions.AI;
using ChatClient = OpenAI.Chat.ChatClient;

namespace AGUIDojoServer;

internal static class ChatClientAgentFactory
{
    private static ChatClient? s_chatClient;

    public static void Initialize(IConfiguration configuration)
    {
        // Resolve a ChatClient from configuration. Three modes are supported, all
        // producing the same OpenAI.Chat.ChatClient so the rest of the factory is
        // provider-agnostic:
        //
        // 1. OPENAI_BASE_URL set — talk to any OpenAI-compatible endpoint with an
        //    API key. This covers a local OpenAI mock (e.g. aimock used by the
        //    dojo/e2e), the real OpenAI API, and Azure OpenAI via its
        //    OpenAI-compatible surface (https://{resource}.openai.azure.com/openai/v1/).
        // 2. AZURE_OPENAI_ENDPOINT set — talk to Azure OpenAI using Entra ID
        //    (DefaultAzureCredential), no API key required.
        // 3. Otherwise — talk to the public OpenAI API with OPENAI_API_KEY.
        var modelName = configuration["OPENAI_CHAT_MODEL_ID"]
            ?? configuration["AZURE_OPENAI_DEPLOYMENT_NAME"]
            ?? "gpt-4o";

        string? baseUrl = configuration["OPENAI_BASE_URL"];
        string? azureEndpoint = configuration["AZURE_OPENAI_ENDPOINT"];
        string? apiKey = configuration["OPENAI_API_KEY"];

        if (string.IsNullOrEmpty(baseUrl) && !string.IsNullOrEmpty(azureEndpoint))
        {
            var azureClient = new AzureOpenAIClient(
                new Uri(azureEndpoint),
                new DefaultAzureCredential());
            s_chatClient = azureClient.GetChatClient(modelName);
            return;
        }

        var options = new OpenAIClientOptions();
        if (!string.IsNullOrEmpty(baseUrl))
        {
            options.Endpoint = new Uri(baseUrl);
        }

        var openAIClient = new OpenAIClient(
            new ApiKeyCredential(apiKey ?? string.Empty),
            options);
        s_chatClient = openAIClient.GetChatClient(modelName);
    }

    private static IChatClient CreateBaseChatClient()
    {
        return s_chatClient!.AsIChatClient()
            .AsBuilder()
            .UseFunctionInvocation()
            .Build();
    }

    public static IChatClient CreateAgenticChat()
    {
        return CreateBaseChatClient();
    }

    public static IChatClient CreateBackendToolRendering()
    {
        return CreateBaseChatClient();
    }

    public static IList<AITool> CreateBackendToolRenderingTools(JsonSerializerOptions options)
    {
        return [AIFunctionFactory.Create(
            GetWeather,
            name: "get_weather",
            description: "Get the weather for a given location.",
            options)];
    }

    public static IChatClient CreateHumanInTheLoop()
    {
        return CreateBaseChatClient();
    }

    public static IChatClient CreateToolBasedGenerativeUI()
    {
        return CreateBaseChatClient();
    }

    public static IChatClient CreateAgenticUI()
    {
        return CreateBaseChatClient();
    }

    public const string AgenticUISystemPrompt = """
        When planning use tools only, without any other messages.
        IMPORTANT:
        - Use the `create_plan` tool to set the initial state of the steps
        - Use the `update_plan_step` tool to update the status of each step
        - Do NOT repeat the plan or summarise it in a message
        - Do NOT confirm the creation or updates in a message
        - Do NOT ask the user for additional information or next steps
        - Do NOT leave a plan hanging, always complete the plan via `update_plan_step` if one is ongoing.
        - Continue calling update_plan_step until all steps are marked as completed.

        Only one plan can be active at a time, so do not call the `create_plan` tool
        again until all the steps in current plan are completed.
        """;

    public static IList<AITool> CreateAgenticUITools(JsonSerializerOptions options)
    {
        return
        [
            AIFunctionFactory.Create(
                AgenticPlanningTools.CreatePlan,
                name: "create_plan",
                description: "Create a plan with multiple steps.",
                AGUIDojoServerSerializerContext.Default.Options),
            AIFunctionFactory.Create(
                AgenticPlanningTools.UpdatePlanStepAsync,
                name: "update_plan_step",
                description: "Update a step in the plan with new description or status.",
                AGUIDojoServerSerializerContext.Default.Options)
        ];
    }

    public static AGUIStreamOptions CreateAgenticUIStreamOptions()
    {
        var options = new AGUIStreamOptions();
        options.MapResultAsStateSnapshot("create_plan");
        options.MapResultAsStateDelta("update_plan_step");
        return options;
    }

    public static IChatClient CreateSharedState(JsonSerializerOptions options)
    {
        var innerClient = s_chatClient!.AsIChatClient()
            .AsBuilder()
            .UseFunctionInvocation()
            .Build();

        return new SharedStateAgent(innerClient, options);
    }

    public static IChatClient CreatePredictiveStateUpdates()
    {
        return CreateBaseChatClient();
    }

    public const string PredictiveStateUpdatesSystemPrompt = """
        You are a document editor assistant. When asked to write or edit content:

        IMPORTANT:
        - Use the `write_document` tool with the full document text in Markdown format
        - Format the document extensively so it's easy to read
        - You can use all kinds of markdown (headings, lists, bold, etc.)
        - However, do NOT use italic or strike-through formatting
        - You MUST write the full document, even when changing only a few words
        - When making edits to the document, try to make them minimal - do not change every word
        - Keep stories SHORT!
        - After you are done writing the document you MUST call a confirm_changes tool after you call write_document

        After the user confirms the changes, provide a brief summary of what you wrote.
        """;

    public static IList<AITool> CreatePredictiveStateUpdatesTools(JsonSerializerOptions options)
    {
        return
        [
            AIFunctionFactory.Create(
                WriteDocument,
                name: "write_document",
                description: "Write a document. Use markdown formatting to format the document.",
                AGUIDojoServerSerializerContext.Default.Options)
        ];
    }

    public static AGUIStreamOptions CreatePredictiveStateUpdatesStreamOptions(JsonSerializerOptions jsonSerializerOptions)
    {
        string? lastEmittedDocument = null;
        var options = new AGUIStreamOptions();
        options.MapCall("write_document", fcc =>
        {
            var documentContent = fcc.Arguments?.TryGetValue("document", out var documentValue) == true
                ? documentValue?.ToString()
                : null;

            if (documentContent is null || documentContent == lastEmittedDocument)
            {
                return [];
            }

            var events = new List<BaseEvent>();
            int startIndex = 0;
            if (lastEmittedDocument is not null && documentContent.StartsWith(lastEmittedDocument, StringComparison.Ordinal))
            {
                startIndex = lastEmittedDocument.Length;
            }

            const int chunkSize = 10;
            for (int i = startIndex; i < documentContent.Length; i += chunkSize)
            {
                int length = Math.Min(chunkSize, documentContent.Length - i);
                string chunk = documentContent.Substring(0, i + length);

                var stateUpdate = new DocumentState { Document = chunk };
                var stateJson = JsonSerializer.SerializeToElement(stateUpdate, jsonSerializerOptions);

                events.Add(new StateSnapshotEvent { Snapshot = stateJson });
            }

            lastEmittedDocument = documentContent;
            return events;
        });
        return options;
    }

    [Description("Get the weather for a given location.")]
    private static WeatherInfo GetWeather([Description("The location to get the weather for.")] string location) => new()
    {
        Temperature = 20,
        Conditions = "sunny",
        Humidity = 50,
        WindSpeed = 10,
        FeelsLike = 25
    };

    [Description("Write a document in markdown format.")]
    private static string WriteDocument([Description("The document content to write.")] string document)
    {
        // Simply return success - the document is tracked via state updates
        return "Document written successfully";
    }
}
