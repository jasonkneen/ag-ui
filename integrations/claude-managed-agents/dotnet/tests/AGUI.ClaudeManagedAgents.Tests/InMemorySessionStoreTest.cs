using Xunit;

namespace AGUI.ClaudeManagedAgents.Tests;

public class InMemorySessionStoreTest
{
    private static ManagedAgentsSessionRecord Record() => new()
    {
        SessionId = "sesn_1",
        ToolNames = ["a"],
        PendingClientToolUseIds = ["ctu_1"],
    };

    [Fact]
    public async Task DoesNotAliasTheRecordItWasGiven()
    {
        var store = new InMemorySessionStore();
        var original = Record();
        await store.SetAsync("t", original, CancellationToken.None);

        original.PendingClientToolUseIds.Add("ctu_2");
        original.SessionId = "sesn_mutated";

        var read = await store.GetAsync("t", CancellationToken.None);
        Assert.NotNull(read);
        Assert.Equal("sesn_1", read!.SessionId);
        Assert.Equal(["ctu_1"], read.PendingClientToolUseIds);
    }

    [Fact]
    public async Task DoesNotAliasTheRecordItHandsOut()
    {
        var store = new InMemorySessionStore();
        await store.SetAsync("t", Record(), CancellationToken.None);

        // The agent mutates records in place between persists; those mutations
        // must not be visible until they are actually written back.
        var first = await store.GetAsync("t", CancellationToken.None);
        first!.PendingClientToolUseIds.Add("ctu_2");
        first.LastUserMessageId = "m_unpersisted";

        var second = await store.GetAsync("t", CancellationToken.None);
        Assert.NotNull(second);
        Assert.Equal(["ctu_1"], second!.PendingClientToolUseIds);
        Assert.Null(second.LastUserMessageId);
    }

    [Fact]
    public async Task UnknownAndDeletedThreadsReadAsNull()
    {
        var store = new InMemorySessionStore();
        Assert.Null(await store.GetAsync("nope", CancellationToken.None));
        await store.SetAsync("t", Record(), CancellationToken.None);
        await store.DeleteAsync("t", CancellationToken.None);
        Assert.Null(await store.GetAsync("t", CancellationToken.None));
    }
}
