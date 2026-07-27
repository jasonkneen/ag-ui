using System.Text.Json;
using AGUI.Abstractions;
using Xunit;

namespace AGUI.ClaudeManagedAgents.Tests;

public class ManagedAgentsAgentTest
{
    private const string IdleEndTurn =
        """{"type":"session.status_idle","id":"idle_1","stop_reason":{"type":"end_turn"}}""";

    private static ManagedAgentsAgent NewAgent(FakeManagedAgentsClient fake, ISessionStore? store = null, Action<ManagedAgentsAgentOptions>? configure = null)
    {
        var options = new ManagedAgentsAgentOptions
        {
            ManagedAgentId = "agent_1",
            EnvironmentId = "env_1",
            Client = fake,
            SessionStore = store,
        };
        configure?.Invoke(options);
        return new ManagedAgentsAgent(options);
    }

    private static ManagedAgentsSessionRecord Record(IList<string> pending, string? lastUserMessageId = "u1")
    {
        return new ManagedAgentsSessionRecord { SessionId = "sesn_1", ToolNames = [], PendingClientToolUseIds = pending, LastUserMessageId = lastUserMessageId };
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
        Assert.Equal(("agent_1", "env_1", "AG-UI thread thread_1"), (created.ManagedAgentId, created.EnvironmentId, created.Title));
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
    public async Task DedupesRegisteredToolsAgainstTheAgentsOwnCustomTools()
    {
        // A custom tool already defined on the managed agent must not be sent
        // twice when a frontend tool of the same name is registered: the
        // frontend definition wins and the agent's copy is dropped.
        var fake = new FakeManagedAgentsClient([IdleEndTurn])
        {
            AgentTools =
            [
                FakeManagedAgentsClient.Json("""{"type":"agent_toolset_20260401","configs":[],"default_config":{}}"""),
                FakeManagedAgentsClient.Json("""{"type":"custom","name":"show_chart","description":"Agent's own copy","input_schema":{"type":"object","properties":{}}}"""),
            ],
        };

        await CollectAsync(NewAgent(fake), BaseInput(input =>
            input.Tools = [Tool("show_chart", "Render a chart", """{"type":"object","properties":{}}""")]));

        var created = Assert.Single(fake.CreatedSessions);
        var tools = Assert.IsAssignableFrom<IList<JsonElement>>(created.OverrideTools);
        Assert.Equal(2, tools.Count);
        AssertJson("""{"type":"agent_toolset_20260401","configs":[],"default_config":{}}""", tools[0]);
        Assert.Equal("Render a chart", tools[1].GetProperty("description").GetString());
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
        var agent = new ManagedAgentsAgent(new ManagedAgentsAgentOptions { ManagedAgentId = "agent_1", EnvironmentId = "env_1", Client = fake });
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

        Assert.Equal("nothing_to_send", Assert.IsType<RunErrorEvent>(events[^1]).Code);
        Assert.Empty(fake.Sent);
    }

    [Fact]
    public async Task ErrorsWithoutTouchingTheApiWhenARunHasNoUserMessageOrToolResult()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);

        var events = await CollectAsync(NewAgent(fake), BaseInput(input => input.Messages = []));

        Assert.Equal([AGUIEventTypes.RunStarted, AGUIEventTypes.StateSnapshot, AGUIEventTypes.RunError], Types(events));
        Assert.Equal("empty_run", Assert.IsType<RunErrorEvent>(events[^1]).Code);
        Assert.Empty(fake.CreatedSessions);
    }

    [Fact]
    public async Task ReleasesTheThreadGateWhenABadInputFailsTheRun()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"ok"}]}""", IdleEndTurn]);
        var agent = NewAgent(fake);

        // A null message list must not leak the busy gate before the run fails.
        var bad = await CollectAsync(agent, BaseInput(input => input.Messages = null!));
        Assert.Equal("empty_run", Assert.IsType<RunErrorEvent>(bad[^1]).Code);

        var good = await CollectAsync(agent, BaseInput());
        Assert.Equal(AGUIEventTypes.RunFinished, good[^1].Type);
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
        var error = Assert.IsType<RunErrorEvent>(events[^1]);
        Assert.Equal(
            ("There is nothing to send: a tool result arrived for a thread with no session.", "tool_result_without_session"),
            (error.Message, error.Code));
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
        Assert.Equal(("A run is already in progress on this thread.", "run_in_progress"), (error.Message, error.Code));

        fake.Gate.SetResult();
        await pending;
        while (await slow.MoveNextAsync())
        {
        }
    }

    [Fact]
    public async Task RejectsASecondRunOnTheSameThreadFromAnotherAgentInstance()
    {
        // The busy gate is shared across instances, so per-request construction
        // (e.g. one agent per HTTP request) still serializes runs on a thread.
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"slow"}]}""", IdleEndTurn]);
        fake.Gate = new TaskCompletionSource();
        var store = new InMemorySessionStore();
        var first = NewAgent(fake, store);
        var second = NewAgent(new FakeManagedAgentsClient([IdleEndTurn]), store);

        await using var slow = first.RunAsync(BaseInput()).GetAsyncEnumerator();
        Assert.True(await slow.MoveNextAsync());
        Assert.True(await slow.MoveNextAsync());
        var pending = slow.MoveNextAsync();
        await Task.Delay(100);

        var events = await CollectAsync(second, BaseInput(input => input.RunId = "run_2"));
        var error = Assert.IsType<RunErrorEvent>(events[^1]);
        Assert.Equal("run_in_progress", error.Code);

        fake.Gate.SetResult();
        await pending;
        while (await slow.MoveNextAsync())
        {
        }
    }

    [Fact]
    public async Task DoesNotRejectRunsOnTheSameThreadIdAcrossDifferentStores()
    {
        // The busy gate is keyed by store identity: distinct stores are distinct
        // tenants, so one caller's slow run cannot block another's thread of the
        // same (client-supplied) id.
        var slowFake = new FakeManagedAgentsClient(
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"slow"}]}""", IdleEndTurn]);
        slowFake.Gate = new TaskCompletionSource();
        var first = NewAgent(slowFake, new InMemorySessionStore());
        var second = NewAgent(new FakeManagedAgentsClient([IdleEndTurn]), new InMemorySessionStore());

        await using var slow = first.RunAsync(BaseInput()).GetAsyncEnumerator();
        Assert.True(await slow.MoveNextAsync());
        Assert.True(await slow.MoveNextAsync());
        var pending = slow.MoveNextAsync();
        await Task.Delay(100);

        var events = await CollectAsync(second, BaseInput(input => input.RunId = "run_2"));
        Assert.IsType<RunFinishedEvent>(events[^1]);

        slowFake.Gate.SetResult();
        await pending;
        while (await slow.MoveNextAsync())
        {
        }
    }

    [Fact]
    public async Task KeysTheSessionStoreByThreadId()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"a"}]}""", IdleEndTurn]);
        var store = new InMemorySessionStore();
        var agent = NewAgent(fake, store);

        await foreach (var _ in agent.RunAsync(BaseInput()))
        {
        }

        Assert.Single(fake.CreatedSessions);
        Assert.NotNull(await store.GetAsync("thread_1", default));
    }

    [Fact]
    public async Task PinsTheAgentVersionAndUsesTheConfiguredTitle()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);

        await CollectAsync(
            NewAgent(fake, configure: options =>
            {
                options.AgentVersion = 3;
                options.SessionTitle = threadId => $"Chat {threadId}";
            }),
            BaseInput());

        var created = Assert.Single(fake.CreatedSessions);
        Assert.Equal((3, "Chat thread_1"), (created.AgentVersion, created.Title));
    }

    [Fact]
    public async Task RegistersBackendAndFrontendToolsWithTheFrontendWinningANameCollision()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);
        fake.AgentTools = [];
        var agent = NewAgent(fake, configure: options => options.BackendTools.Add(new ManagedAgentsBackendTool
        {
            Name = "lookup docs",
            Description = "Backend lookup",
            Handler = static _ => Task.FromResult("x"),
        }));

        await CollectAsync(agent, BaseInput(input =>
            input.Tools = [Tool("lookup docs", "Frontend lookup", """{"type":"object"}""")]));

        // Both normalize to "lookup_docs"; the frontend definition wins.
        var tools = Assert.Single(fake.CreatedSessions).OverrideTools!;
        AssertJson(
            """{"type":"custom","name":"lookup_docs","description":"Frontend lookup","input_schema":{"type":"object","properties":{},"required":[]}}""",
            Assert.Single(tools));
    }

    [Fact]
    public async Task LetsALaterBackendToolReplaceAnEarlierOneWithTheSameNormalizedName()
    {
        var fake = new FakeManagedAgentsClient(
            [
                """{"type":"agent.custom_tool_use","id":"ctu_1","name":"search_web","input":{}}""",
                """{"type":"session.status_idle","id":"idle_1","stop_reason":{"type":"requires_action","event_ids":["ctu_1"]}}""",
                IdleEndTurn,
            ]);
        fake.AgentTools = [];

        // "search.web" and "search_web" both normalize to "search_web": constructing the
        // agent must not throw, and the last registration wins.
        var agent = NewAgent(fake, configure: options =>
        {
            options.BackendTools.Add(new ManagedAgentsBackendTool { Name = "search.web", Handler = static _ => Task.FromResult("first") });
            options.BackendTools.Add(new ManagedAgentsBackendTool { Name = "search_web", Handler = static _ => Task.FromResult("second") });
        });

        var events = await CollectAsync(agent, BaseInput());

        Assert.Single(Assert.Single(fake.CreatedSessions).OverrideTools!);
        var result = Assert.Single(events.OfType<ToolCallResultEvent>());
        Assert.Equal("second", result.Content);
    }

    [Fact]
    public async Task ForwardsAToolMessagesErrorFlagAsAnErrorResult()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);
        var store = new InMemorySessionStore();
        await store.SetAsync("thread_1", Record(["ctu_1"]), default);

        await CollectAsync(NewAgent(fake, store), BaseInput(input => input.Messages =
        [
            new AGUIUserMessage { Id = "u1", Content = "Hello" },
            new AGUIToolMessage { Id = "t1", ToolCallId = "ctu_1", Content = "boom", Error = "failed" },
        ]));

        AssertJson(
            """{"type":"user.custom_tool_result","custom_tool_use_id":"ctu_1","content":[{"type":"text","text":"boom\nfailed"}],"is_error":true}""",
            fake.Sent[0].Single());
    }

    [Fact]
    public async Task SendsOnlyTheErrorTextWhenAToolMessageHasNoContent()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);
        var store = new InMemorySessionStore();
        await store.SetAsync("thread_1", Record(["ctu_1"]), default);

        await CollectAsync(NewAgent(fake, store), BaseInput(input => input.Messages =
        [
            new AGUIUserMessage { Id = "u1", Content = "Hello" },
            new AGUIToolMessage { Id = "t1", ToolCallId = "ctu_1", Content = "", Error = "failed" },
        ]));

        AssertJson(
            """{"type":"user.custom_tool_result","custom_tool_use_id":"ctu_1","content":[{"type":"text","text":"failed"}],"is_error":true}""",
            fake.Sent[0].Single());
    }

    [Fact]
    public async Task StaysParkedWhenOnlySomePendingToolCallsAreAnswered()
    {
        var fake = new FakeManagedAgentsClient();
        var store = new InMemorySessionStore();
        await store.SetAsync("thread_1", Record(["ctu_1", "ctu_2"]), default);

        var events = await CollectAsync(NewAgent(fake, store), BaseInput(input => input.Messages =
        [
            new AGUIUserMessage { Id = "u1", Content = "Hello" },
            new AGUIToolMessage { Id = "t1", ToolCallId = "ctu_1", Content = "done" },
        ]));

        // The answered call is posted, the run finishes without streaming, and the unanswered
        // call stays pending.
        AssertJson(
            """{"type":"user.custom_tool_result","custom_tool_use_id":"ctu_1","content":[{"type":"text","text":"done"}],"is_error":false}""",
            Assert.Single(fake.Sent).Single());
        Assert.Empty(fake.StreamRequests);
        Assert.Equal(AGUIEventTypes.RunFinished, events[^1].Type);
        Assert.Equal(["ctu_2"], (await store.GetAsync("thread_1", default))!.PendingClientToolUseIds);
    }

    [Fact]
    public async Task DeletesTheThreadRecordWhenTheSessionEndsSoTheNextRunStartsFresh()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"session.status_terminated","id":"term_1"}"""],
            [IdleEndTurn]);
        var store = new InMemorySessionStore();

        var events = await CollectAsync(NewAgent(fake, store), BaseInput());
        Assert.Equal("session_ended", Assert.IsType<RunErrorEvent>(events[^1]).Code);
        Assert.Null(await store.GetAsync("thread_1", default));

        // The next run creates a fresh session.
        await CollectAsync(NewAgent(fake, store), BaseInput(input => input.RunId = "run_2"));
        Assert.Equal(2, fake.CreatedSessions.Count);
    }

    [Fact]
    public async Task InterruptsTheSessionAndStopsWhenTheClientDisconnects()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.message","id":"msg_1","content":[{"type":"text","text":"unheard"}]}""", IdleEndTurn]);
        fake.Gate = new TaskCompletionSource();
        using var client = new CancellationTokenSource();
        var events = new List<BaseEvent>();

        await using var run = NewAgent(fake).RunAsync(BaseInput(), client.Token).GetAsyncEnumerator();

        // RUN_STARTED, STATE_SNAPSHOT, and the session CUSTOM event arrive before the turn opens.
        for (var i = 0; i < 3; i++)
        {
            Assert.True(await run.MoveNextAsync());
            events.Add(run.Current);
        }

        // The next event drives the turn: it posts the user message, then waits on the stream.
        var pending = run.MoveNextAsync();
        while (fake.Sent.Count == 0)
        {
            await Task.Delay(5);
        }

        // Disconnect while the turn waits: the run ends with no further events.
        client.Cancel();
        Assert.False(await pending);

        // No error or further events reach the departed client, but the session is interrupted.
        Assert.Equal([AGUIEventTypes.RunStarted, AGUIEventTypes.StateSnapshot, AGUIEventTypes.Custom], Types(events));
        Assert.Contains("user.interrupt", fake.SentTypes);
    }

    [Fact]
    public async Task InterruptsTheSessionAndErrorsWhenTheTurnTimesOut()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);
        fake.Gate = new TaskCompletionSource(); // never released: the turn stalls

        var events = await CollectAsync(
            NewAgent(fake, configure: options => options.TurnTimeout = TimeSpan.FromMilliseconds(50)),
            BaseInput());

        var error = Assert.IsType<RunErrorEvent>(events[^1]);
        Assert.Equal("The turn exceeded the 0.05s limit and was interrupted.", error.Message);
        Assert.Equal("turn_timeout", error.Code);
        Assert.Contains("user.interrupt", fake.SentTypes);
    }

    [Fact]
    public async Task PostsAnInterruptedResultWhenABackendToolIsStillRunningAtTheTimeout()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.custom_tool_use","id":"ctu_1","name":"slow_tool","input":{}}"""]);
        fake.AgentTools = [];
        var released = new TaskCompletionSource<string>();
        var agent = NewAgent(fake, configure: options =>
        {
            options.TurnTimeout = TimeSpan.FromMilliseconds(50);
            options.BackendTools.Add(new ManagedAgentsBackendTool { Name = "slow_tool", Handler = _ => released.Task });
        });

        var events = await CollectAsync(agent, BaseInput());
        released.SetResult("too late");

        // The tool ran past the timeout: its call is answered anyway so the session is not left
        // parked on it, then the turn is interrupted and errors.
        Assert.IsType<RunErrorEvent>(events[^1]);
        Assert.Contains(events.OfType<ToolCallStartEvent>(), start => start.ToolCallId == "ctu_1");
        AssertJson(
            """{"type":"user.custom_tool_result","custom_tool_use_id":"ctu_1","content":[{"type":"text","text":"Tool execution was interrupted."}],"is_error":true}""",
            Assert.Single(fake.SentEvents, evt => evt.GetProperty("type").GetString() == "user.custom_tool_result"));
        Assert.Contains("user.interrupt", fake.SentTypes);
    }

    [Fact]
    public async Task PostsABackendToolResultEvenWhenTheClientLeavesRightAfterTheHandlerCompletes()
    {
        var fake = new FakeManagedAgentsClient(
            ["""{"type":"agent.custom_tool_use","id":"ctu_1","name":"get_time","input":{}}""", IdleEndTurn]);
        fake.AgentTools = [];
        using var client = new CancellationTokenSource();
        var agent = NewAgent(fake, configure: options =>
            options.BackendTools.Add(new ManagedAgentsBackendTool
            {
                Name = "get_time",
                Handler = _ =>
                {
                    client.Cancel(); // the caller disconnects while the tool finishes
                    return Task.FromResult("noon");
                },
            }));

        await foreach (var _ in agent.RunAsync(BaseInput(), client.Token))
        {
        }

        // The tool already ran, so its result reaches the session despite the disconnect.
        AssertJson(
            """{"type":"user.custom_tool_result","custom_tool_use_id":"ctu_1","content":[{"type":"text","text":"noon"}],"is_error":false}""",
            Assert.Single(fake.SentEvents, evt => evt.GetProperty("type").GetString() == "user.custom_tool_result"));
    }

    [Fact]
    public async Task DoesNotRequestPreviewsWhenStreamDeltasIsDisabled()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);

        await CollectAsync(NewAgent(fake, configure: options => options.StreamDeltas = false), BaseInput());

        Assert.Equal([("sesn_1", false)], fake.StreamRequests);
    }

    [Fact]
    public async Task ExtractsTheTextFromMultimodalUserContent()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);

        await CollectAsync(NewAgent(fake), BaseInput(input => input.Messages =
        [
            new AGUIUserMessage
            {
                Id = "u1",
                Content =
                [
                    new AGUITextInputContent { Text = "Look here" },
                    new AGUIImageInputContent { Source = new AGUIInputContentUrlSource { Value = "https://x/y.png" } },
                ],
            },
        ]));

        AssertJson("""{"type":"user.message","content":[{"type":"text","text":"Look here"}]}""", fake.Sent[0].Single());
    }

    [Fact]
    public async Task AbandonsMultiplePendingToolCallsInTheirOriginalOrder()
    {
        var fake = new FakeManagedAgentsClient([IdleEndTurn]);
        var store = new InMemorySessionStore();
        await store.SetAsync("thread_1", Record(["ctu_1", "ctu_2"]), default);

        await CollectAsync(NewAgent(fake, store), BaseInput(input => input.Messages =
        [
            new AGUIUserMessage { Id = "u1", Content = "old" },
            new AGUIUserMessage { Id = "u2", Content = "never mind" },
        ]));

        // Both abandoned results are batched before the user message, in the order they were parked.
        var abandoned = fake.Sent[0];
        Assert.Equal(
            ["ctu_1", "ctu_2"],
            abandoned.Select(evt => evt.GetProperty("custom_tool_use_id").GetString()));
        Assert.All(abandoned, evt => Assert.True(evt.GetProperty("is_error").GetBoolean()));
        AssertJson("""{"type":"user.message","content":[{"type":"text","text":"never mind"}]}""", fake.Sent[1].Single());
    }
}
