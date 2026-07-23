using System.Text.Json;
using AGUI.Abstractions;
using Xunit;

namespace AGUI.ClaudeManagedAgents.Tests;

public class ManagedAgentsAgentTest
{
    private const string IdleEndTurn =
        """{"type":"session.status_idle","id":"idle_1","stop_reason":{"type":"end_turn"}}""";

    private static ManagedAgentsAgent NewAgent(FakeManagedAgentsClient fake, ISessionStore? store = null)
    {
        return new ManagedAgentsAgent(new ManagedAgentsAgentOptions
        {
            AgentId = "agent_1",
            EnvironmentId = "env_1",
            Client = fake,
            SessionStore = store,
        });
    }

    private static RunAgentInput BaseInput(Action<RunAgentInput>? configure = null)
    {
        var input = new RunAgentInput
        {
            ThreadId = "thread_1",
            RunId = "run_1",
            State = JsonSerializer.SerializeToElement(new { }),
            Messages = [new AGUIUserMessage { Id = "u1", Content = "Hello" }],
            Tools = [],
        };
        configure?.Invoke(input);
        return input;
    }

    private static async Task<List<BaseEvent>> CollectAsync(ManagedAgentsAgent agent, RunAgentInput input)
    {
        var events = new List<BaseEvent>();
        await foreach (var evt in agent.RunAsync(input))
        {
            events.Add(evt);
        }

        return events;
    }

    private static IEnumerable<string> Types(IEnumerable<BaseEvent> events) => events.Select(e => e.Type);

    private static AGUITool Tool(string name, string description, string parametersJson)
    {
        return new AGUITool
        {
            Name = name,
            Description = description,
            Parameters = FakeManagedAgentsClient.Json(parametersJson),
        };
    }

    private static void AssertJson(string expected, JsonElement actual)
    {
        var expectedElement = FakeManagedAgentsClient.Json(expected);
        Assert.True(
            JsonElement.DeepEquals(expectedElement, actual),
            $"Expected {expectedElement.GetRawText()}\nActual   {actual.GetRawText()}");
    }

    [Fact]
    public async Task CreatesASessionForANewThreadAndStreamsAReply()
    {
        var fake = new FakeManagedAgentsClient([
            """{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"Hi!"}]}""",
            IdleEndTurn,
        ]);

        var events = await CollectAsync(NewAgent(fake), BaseInput());

        var created = Assert.Single(fake.CreatedSessions);
        Assert.Equal(("agent_1", "env_1", "AG-UI thread thread_1"), (created.AgentId, created.EnvironmentId, created.Title));
        Assert.Null(created.OverrideTools);

        Assert.Equal(
            [
                AGUIEventTypes.RunStarted,
                AGUIEventTypes.StateSnapshot,
                AGUIEventTypes.Custom,
                AGUIEventTypes.TextMessageStart,
                AGUIEventTypes.TextMessageContent,
                AGUIEventTypes.TextMessageEnd,
                AGUIEventTypes.RunFinished,
            ],
            Types(events));

        var custom = Assert.IsType<CustomEvent>(events[2]);
        Assert.Equal(ManagedAgentsAgent.SessionCustomEventName, custom.Name);
        AssertJson("""{"sessionId":"sesn_1","threadId":"thread_1"}""", custom.Value!.Value);
        AssertJson("""{"type":"user.message","content":[{"type":"text","text":"Hello"}]}""", fake.Sent[0].Single());
    }

    [Fact]
    public async Task ReusesTheSessionOnTheThreadsNextRunAndSendsOnlyTheNewMessage()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"one"}]}""", IdleEndTurn],
            ["""{"type":"agent.message","id":"msg_2","content":[{"type":"text","text":"two"}]}""", IdleEndTurn]);
        var store = new InMemorySessionStore();

        await CollectAsync(NewAgent(fake, store), BaseInput());
        await CollectAsync(NewAgent(fake, store), BaseInput(input =>
        {
            input.RunId = "run_2";
            input.Messages =
            [
                new AGUIUserMessage { Id = "u1", Content = "Hello" },
                new AGUIAssistantMessage { Id = "a1", Content = "one" },
                new AGUIUserMessage { Id = "u2", Content = "Follow-up" },
            ];
        }));

        Assert.Single(fake.CreatedSessions);
        AssertJson("""{"type":"user.message","content":[{"type":"text","text":"Follow-up"}]}""", fake.Sent[1].Single());
    }

    [Fact]
    public async Task RegistersFrontendToolsAsCustomToolsWhenCreatingTheSession()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);

        await CollectAsync(NewAgent(fake), BaseInput(input =>
            input.Tools = [Tool("show_chart", "Render a chart", """{"type":"object","properties":{"title":{"type":"string"}}}""")]));

        var created = Assert.Single(fake.CreatedSessions);
        var tools = Assert.IsAssignableFrom<IList<JsonElement>>(created.OverrideTools);
        Assert.Equal(2, tools.Count);
        AssertJson("""{"type":"agent_toolset_20260401","configs":[],"default_config":{}}""", tools[0]);
        AssertJson(
            """
            {
              "type": "custom",
              "name": "show_chart",
              "description": "Render a chart",
              "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}, "required": []}
            }
            """,
            tools[1]);
    }

    [Fact]
    public async Task RoundTripsAFrontendToolByParkingThenResumingWithTheClientsResult()
    {
        var fake = new FakeManagedAgentsClient(
            [
                """{"type":"agent.custom_tool_use","id":"ctu_1","name":"show_chart","input":{"title":"Sales"}}""",
                """{"type":"session.status_idle","id":"idle_1","stop_reason":{"type":"requires_action","event_ids":["ctu_1"]}}""",
            ],
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"Chart shown."}]}""", IdleEndTurn]);
        var store = new InMemorySessionStore();
        var tools = new List<AGUITool> { Tool("show_chart", "Render a chart", """{"type":"object"}""") };

        var first = await CollectAsync(NewAgent(fake, store), BaseInput(input => input.Tools = tools));
        Assert.Equal(
            [
                AGUIEventTypes.RunStarted,
                AGUIEventTypes.StateSnapshot,
                AGUIEventTypes.Custom,
                AGUIEventTypes.ToolCallStart,
                AGUIEventTypes.ToolCallArgs,
                AGUIEventTypes.ToolCallEnd,
                AGUIEventTypes.RunFinished,
            ],
            Types(first));
        Assert.Equal(["ctu_1"], (await store.GetAsync("thread_1", default))!.PendingClientToolUseIds);

        var second = await CollectAsync(NewAgent(fake, store), BaseInput(input =>
        {
            input.RunId = "run_2";
            input.Tools = tools;
            input.Messages =
            [
                new AGUIUserMessage { Id = "u1", Content = "Hello" },
                new AGUIToolMessage { Id = "t1", ToolCallId = "ctu_1", Content = "rendered" },
            ];
        }));

        AssertJson(
            """{"type":"user.custom_tool_result","custom_tool_use_id":"ctu_1","content":[{"type":"text","text":"rendered"}],"is_error":false}""",
            fake.Sent[1].Single());
        Assert.Contains(AGUIEventTypes.TextMessageContent, Types(second));
        Assert.Equal(AGUIEventTypes.RunFinished, second[^1].Type);
        Assert.Empty((await store.GetAsync("thread_1", default))!.PendingClientToolUseIds);
    }

    [Fact]
    public async Task SharesTheDefaultSessionStoreAcrossRunsSoAResumedRunFindsItsParkedSession()
    {
        var fake = new FakeManagedAgentsClient(
            [
                """{"type":"agent.custom_tool_use","id":"ctu_1","name":"show_chart","input":{}}""",
                """{"type":"session.status_idle","id":"idle_1","stop_reason":{"type":"requires_action","event_ids":["ctu_1"]}}""",
            ],
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"Done."}]}""", IdleEndTurn]);
        // No session store passed: the default in-memory store persists across runs.
        var agent = new ManagedAgentsAgent(new ManagedAgentsAgentOptions { AgentId = "agent_1", EnvironmentId = "env_1", Client = fake });
        var tools = new List<AGUITool> { Tool("show_chart", "Render a chart", """{"type":"object"}""") };

        await CollectAsync(agent, BaseInput(input => input.Tools = tools));
        await CollectAsync(agent, BaseInput(input =>
        {
            input.RunId = "run_2";
            input.Tools = tools;
            input.Messages =
            [
                new AGUIUserMessage { Id = "u1", Content = "Hello" },
                new AGUIToolMessage { Id = "t1", ToolCallId = "ctu_1", Content = "rendered" },
            ];
        }));

        Assert.Single(fake.CreatedSessions);
        AssertJson(
            """{"type":"user.custom_tool_result","custom_tool_use_id":"ctu_1","content":[{"type":"text","text":"rendered"}],"is_error":false}""",
            fake.Sent[1].Single());
    }

    [Fact]
    public async Task ForwardsEveryUndeliveredUserMessageInOrder()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);
        var store = new InMemorySessionStore();
        await store.SetAsync(
            "thread_1",
            new ManagedAgentsSessionRecord { SessionId = "sesn_1", ToolNames = [], PendingClientToolUseIds = [], LastUserMessageId = "u1" },
            default);

        await CollectAsync(NewAgent(fake, store), BaseInput(input => input.Messages =
        [
            new AGUIUserMessage { Id = "u1", Content = "delivered" },
            new AGUIUserMessage { Id = "u2", Content = "second" },
            new AGUIUserMessage { Id = "u3", Content = "third" },
        ]));

        var sent = fake.Sent[0];
        Assert.Equal(2, sent.Count);
        AssertJson("""{"type":"user.message","content":[{"type":"text","text":"second"}]}""", sent[0]);
        AssertJson("""{"type":"user.message","content":[{"type":"text","text":"third"}]}""", sent[1]);
        Assert.Equal("u3", (await store.GetAsync("thread_1", default))!.LastUserMessageId);
    }

    [Fact]
    public async Task AbandonsParkedToolCallsWhenTheUserSendsANewMessageInstead()
    {
        var fake = new FakeManagedAgentsClient([
            """{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"Moving on."}]}""",
            IdleEndTurn,
        ]);
        var store = new InMemorySessionStore();
        await store.SetAsync(
            "thread_1",
            new ManagedAgentsSessionRecord { SessionId = "sesn_1", ToolNames = [], PendingClientToolUseIds = ["ctu_1"], LastUserMessageId = "u1" },
            default);

        var events = await CollectAsync(NewAgent(fake, store), BaseInput(input => input.Messages =
        [
            new AGUIUserMessage { Id = "u1", Content = "old" },
            new AGUIUserMessage { Id = "u2", Content = "never mind" },
        ]));

        // The abandoned result is posted first (resuming the parked session), then the user
        // message follows in a second call: the API rejects the two mixed in one batch.
        Assert.Equal(2, fake.Sent.Count);
        AssertJson(
            """
            {
              "type": "user.custom_tool_result",
              "custom_tool_use_id": "ctu_1",
              "content": [{"type": "text", "text": "The user did not provide a result for this tool call."}],
              "is_error": true
            }
            """,
            fake.Sent[0].Single());
        AssertJson("""{"type":"user.message","content":[{"type":"text","text":"never mind"}]}""", fake.Sent[1].Single());
        Assert.Equal(AGUIEventTypes.RunFinished, events[^1].Type);

        var record = (await store.GetAsync("thread_1", default))!;
        Assert.Empty(record.PendingClientToolUseIds);
        Assert.Equal("u2", record.LastUserMessageId);
    }

    [Fact]
    public async Task ErrorsWhenARunHasNothingNewToSend()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);
        var store = new InMemorySessionStore();
        await store.SetAsync(
            "thread_1",
            new ManagedAgentsSessionRecord { SessionId = "sesn_1", ToolNames = [], PendingClientToolUseIds = [], LastUserMessageId = "u1" },
            default);

        var events = await CollectAsync(NewAgent(fake, store), BaseInput());

        Assert.IsType<RunErrorEvent>(events[^1]);
        Assert.Empty(fake.Sent);
    }

    [Fact]
    public async Task ErrorsWithoutTouchingTheApiWhenARunHasNoUserMessageOrToolResult()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);

        var events = await CollectAsync(NewAgent(fake), BaseInput(input => input.Messages = []));

        Assert.Equal([AGUIEventTypes.RunStarted, AGUIEventTypes.StateSnapshot, AGUIEventTypes.RunError], Types(events));
        Assert.Empty(fake.CreatedSessions);
    }

    [Fact]
    public async Task RetriesTheFollowUpUserMessageWhileTheSessionIsStillParked()
    {
        var fake = new FakeManagedAgentsClient([
            """{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"Moving on."}]}""",
            IdleEndTurn,
        ]);
        var store = new InMemorySessionStore();
        await store.SetAsync(
            "thread_1",
            new ManagedAgentsSessionRecord { SessionId = "sesn_1", ToolNames = [], PendingClientToolUseIds = ["ctu_1"], LastUserMessageId = "u1" },
            default);

        // The result posts fine; the follow-up user message races the un-park and is rejected
        // twice as sent-while-parked before the session accepts it.
        var parked = new ManagedAgentsSendException(400, "session is waiting on responses to events [ctu_1]");
        var rejections = 0;
        fake.SendGuard = batch =>
        {
            var isFollowUp = batch.Any(e => e.GetProperty("type").GetString() == "user.message");
            return isFollowUp && rejections++ < 2 ? parked : null;
        };

        var events = await CollectAsync(NewAgent(fake, store), BaseInput(input => input.Messages =
        [
            new AGUIUserMessage { Id = "u1", Content = "old" },
            new AGUIUserMessage { Id = "u2", Content = "never mind" },
        ]));

        // One results batch, then the user message: two rejected attempts plus one success.
        Assert.Equal(4, fake.SendAttempts.Count);
        Assert.Equal(2, fake.Sent.Count);
        AssertJson("""{"type":"user.message","content":[{"type":"text","text":"never mind"}]}""", fake.Sent[1].Single());
        Assert.Equal(AGUIEventTypes.RunFinished, events[^1].Type);
    }

    [Fact]
    public async Task ErrorsWithoutCreatingASessionWhenAFreshThreadCarriesOnlyAToolResult()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);

        var events = await CollectAsync(NewAgent(fake), BaseInput(input =>
            input.Messages = [new AGUIToolMessage { Id = "t1", ToolCallId = "ctu_1", Content = "rendered" }]));

        Assert.Equal([AGUIEventTypes.RunStarted, AGUIEventTypes.StateSnapshot, AGUIEventTypes.RunError], Types(events));
        Assert.Equal(
            "There is nothing to send: a tool result arrived for a thread with no session.",
            Assert.IsType<RunErrorEvent>(events[^1]).Message);
        Assert.Empty(fake.CreatedSessions);
        Assert.Empty(fake.Sent);
    }

    [Fact]
    public async Task UpdatesTheSessionsToolsWhenTheFrontendAddsANewOne()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn], [IdleEndTurn]);
        var store = new InMemorySessionStore();

        await CollectAsync(NewAgent(fake, store), BaseInput());
        Assert.Empty(fake.Updates);

        await CollectAsync(NewAgent(fake, store), BaseInput(input =>
        {
            input.RunId = "run_2";
            input.Messages =
            [
                new AGUIUserMessage { Id = "u1", Content = "Hello" },
                new AGUIUserMessage { Id = "u2", Content = "Show me a chart" },
            ];
            input.Tools = [Tool("show_chart", "Render a chart", """{"type":"object"}""")];
        }));

        var update = Assert.Single(fake.Updates);
        Assert.Equal(2, update.Count);
        AssertJson("""{"type":"agent_toolset_20260401","configs":[],"default_config":{}}""", update[0]);
        AssertJson(
            """{"type":"custom","name":"show_chart","description":"Render a chart","input_schema":{"type":"object","properties":{},"required":[]}}""",
            update[1]);
    }

    [Fact]
    public async Task RejectsASecondRunWhileTheThreadIsBusy()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"slow"}]}""", IdleEndTurn]);
        fake.Gate = new TaskCompletionSource();
        var store = new InMemorySessionStore();
        var agent = NewAgent(fake, store);

        // Start a run and drive it into the turn, which blocks on the gate mid-stream.
        await using var slow = agent.RunAsync(BaseInput()).GetAsyncEnumerator();
        Assert.True(await slow.MoveNextAsync());
        Assert.Equal(AGUIEventTypes.RunStarted, slow.Current.Type);
        Assert.True(await slow.MoveNextAsync());
        Assert.Equal(AGUIEventTypes.StateSnapshot, slow.Current.Type);
        var pending = slow.MoveNextAsync(); // creates the session, then blocks in the turn stream

        // Give the first run time to enter the busy section.
        await Task.Delay(100);
        var second = await CollectAsync(agent, BaseInput(input => input.RunId = "run_2"));
        var error = Assert.IsType<RunErrorEvent>(second[^1]);
        Assert.Equal("A run is already in progress on this thread.", error.Message);

        fake.Gate.SetResult();
        await pending;
        while (await slow.MoveNextAsync())
        {
        }
    }

    [Fact]
    public async Task ScopesSessionsByOwnerSoThreadIdsDoNotCollideAcrossCallers()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"a"}]}""", IdleEndTurn],
            ["""{"type":"agent.message","id":"msg_2","content":[{"type":"text","text":"b"}]}""", IdleEndTurn]);
        var store = new InMemorySessionStore();
        var agent = NewAgent(fake, store);

        await foreach (var _ in agent.RunAsync(BaseInput(), "alice"))
        {
        }

        await foreach (var _ in agent.RunAsync(BaseInput(input => input.RunId = "run_2"), "bob"))
        {
        }

        // Each owner gets their own session even for the same thread ID.
        Assert.Equal(2, fake.CreatedSessions.Count);
        Assert.NotNull(await store.GetAsync("alice:thread_1", default));
        Assert.NotNull(await store.GetAsync("bob:thread_1", default));
    }
}
