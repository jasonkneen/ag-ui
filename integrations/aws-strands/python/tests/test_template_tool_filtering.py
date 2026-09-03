"""Per-request filtering of the tools the template agent contributed.

The adapter keeps one Strands ``Agent`` per thread and that instance is
load-bearing: it holds the thread's ``SessionManager``, its native interrupt
checkpoint and its conversation history. So the property these tests exist to
pin is not only that a filter takes effect, but that it takes effect on the
registry the live instance already owns. A filter applied by rebuilding the
thread's agent would pass a "the model saw fewer tools" assertion while
silently discarding a conversation and any approval waiting inside it, which is
why the identity of the cached agent is asserted alongside the tool specs.
"""

from __future__ import annotations

import logging

import pytest
from ag_ui.core import EventType, RunAgentInput, Tool, UserMessage
from strands import Agent as StrandsAgentCore
from strands import tool
from strands.interrupt import Interrupt as StrandsInterrupt
from strands.models.model import Model as StrandsModel
from strands.tools.registry import ToolRegistry

from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig, ToolBehavior
from ag_ui_strands.template_tools import (
    index_template_tools,
    parked_batch_tool_names,
    resolve_template_tool_selection,
    sync_template_tools,
)
from tests.interrupt_state_stub import InterruptStateStub


THREAD_ID = "template-filter-thread"


@tool
def read_docs(topic: str) -> str:
    """Read the documentation for a topic."""
    return f"docs about {topic}"


@tool
def delete_record(record_id: str) -> str:
    """Delete a record."""
    return f"deleted {record_id}"


class RecordingModel(StrandsModel):
    """Answers with text and records the tool specs it was offered."""

    def __init__(self) -> None:
        self.offered_tool_names: list[set[str]] = []

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(
        self, output_model, prompt=None, system_prompt=None, **kwargs
    ):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.offered_tool_names.append(
            {spec["name"] for spec in (tool_specs or [])}
        )
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": "ok"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


class DeleteThenAnswerModel(StrandsModel):
    """Calls ``delete_record`` once, then answers with text."""

    def __init__(self) -> None:
        self.offered_tool_names: list[set[str]] = []
        self.called_delete = False

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(
        self, output_model, prompt=None, system_prompt=None, **kwargs
    ):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.offered_tool_names.append(
            {spec["name"] for spec in (tool_specs or [])}
        )
        yield {"messageStart": {"role": "assistant"}}
        if not self.called_delete:
            self.called_delete = True
            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "delete-1",
                            "name": "delete_record",
                        }
                    }
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": '{"record_id": "r-1"}'}}
                }
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return
        yield {"contentBlockDelta": {"delta": {"text": "done"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def make_agent(
    model: StrandsModel,
    *,
    config: StrandsAgentConfig | None = None,
    tools=None,
) -> StrandsAgent:
    core = StrandsAgentCore(
        model=model,
        tools=[read_docs, delete_record] if tools is None else tools,
        system_prompt="Help the user.",
    )
    return StrandsAgent(
        core,
        name="template-filter-test",
        config=config or StrandsAgentConfig(),
    )


def run_input(
    run_id: str,
    *,
    thread_id: str = THREAD_ID,
    tools: list[Tool] | None = None,
    forwarded_props: dict | None = None,
    resume=None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        state={},
        messages=[
            UserMessage(id=f"user-{run_id}", role="user", content="Do the thing.")
        ],
        tools=tools or [],
        context=[],
        forwarded_props=forwarded_props or {},
        **({"resume": resume} if resume is not None else {}),
    )


async def drain(agent: StrandsAgent, request: RunAgentInput) -> list:
    return [event async for event in agent.run(request)]


def registry_names(agent: StrandsAgent, thread_id: str = THREAD_ID) -> set[str]:
    return set(agent._agents_by_thread[thread_id].tool_registry.registry)


# ---------------------------------------------------------------------------
# The rescope: filtering happens on the live agent, not by replacing it
# ---------------------------------------------------------------------------


class TestFilteringWithoutRebuildingTheThreadAgent:
    async def test_the_filtered_set_varies_between_two_requests_on_one_thread(self):
        allowed_by_run = {"r1": ["read_docs"], "r2": ["read_docs", "delete_record"]}
        model = RecordingModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                template_tools_provider=lambda data: allowed_by_run[data.run_id]
            ),
        )

        await drain(agent, run_input("r1"))
        assert model.offered_tool_names[-1] == {"read_docs"}
        assert registry_names(agent) == {"read_docs"}

        await drain(agent, run_input("r2"))
        assert model.offered_tool_names[-1] == {"read_docs", "delete_record"}
        assert registry_names(agent) == {"read_docs", "delete_record"}

    async def test_the_cached_thread_agent_is_the_same_instance_across_the_change(
        self,
    ):
        """The whole point of applying the filter to the registry.

        Recreating the per-thread agent whenever the resolved tool set changed
        would satisfy the assertion above and still be wrong, so identity is
        asserted directly.
        """
        allowed_by_run = {"r1": ["read_docs"], "r2": []}
        model = RecordingModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                template_tools_provider=lambda data: allowed_by_run[data.run_id]
            ),
        )

        await drain(agent, run_input("r1"))
        first = agent._agents_by_thread[THREAD_ID]

        await drain(agent, run_input("r2"))
        assert agent._agents_by_thread[THREAD_ID] is first
        assert model.offered_tool_names[-1] == set()

    async def test_an_empty_selection_withholds_every_template_tool(self):
        model = RecordingModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(template_tools_provider=lambda data: []),
        )
        await drain(agent, run_input("r1"))
        assert model.offered_tool_names[-1] == set()

    async def test_returning_none_declines_to_filter_that_request(self):
        selection_by_run = {"r1": [], "r2": None}
        model = RecordingModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                template_tools_provider=lambda data: selection_by_run[data.run_id]
            ),
        )

        await drain(agent, run_input("r1"))
        assert model.offered_tool_names[-1] == set()

        await drain(agent, run_input("r2"))
        assert model.offered_tool_names[-1] == {"read_docs", "delete_record"}

    async def test_the_provider_may_be_async_and_may_read_the_caller_identity(self):
        async def provider(data: RunAgentInput):
            if (data.forwarded_props or {}).get("role") == "admin":
                return None
            return [read_docs]

        model = RecordingModel()
        agent = make_agent(
            model, config=StrandsAgentConfig(template_tools_provider=provider)
        )

        await drain(agent, run_input("r1", forwarded_props={"role": "reader"}))
        assert model.offered_tool_names[-1] == {"read_docs"}

        await drain(agent, run_input("r2", forwarded_props={"role": "admin"}))
        assert model.offered_tool_names[-1] == {"read_docs", "delete_record"}

    async def test_two_threads_can_see_different_tools_at_the_same_time(self):
        model = RecordingModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                template_tools_provider=lambda data: (
                    None if data.thread_id == "wide" else ["read_docs"]
                )
            ),
        )
        await drain(agent, run_input("r1", thread_id="wide"))
        assert model.offered_tool_names[-1] == {"read_docs", "delete_record"}
        await drain(agent, run_input("r2", thread_id="narrow"))
        assert model.offered_tool_names[-1] == {"read_docs"}
        assert registry_names(agent, "wide") == {"read_docs", "delete_record"}
        assert registry_names(agent, "narrow") == {"read_docs"}


class TestWhatTheThreadKeepsAcrossAFilterChange:
    async def test_the_session_manager_and_the_persisted_history_both_survive(
        self, tmp_path
    ):
        from strands.session.file_session_manager import FileSessionManager

        session_manager = FileSessionManager(
            session_id="template-filter-session", storage_dir=str(tmp_path)
        )
        allowed_by_run = {"r1": None, "r2": ["read_docs"]}
        model = RecordingModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                session_manager_provider=lambda data: session_manager,
                template_tools_provider=lambda data: allowed_by_run[data.run_id],
            ),
        )

        await drain(agent, run_input("r1"))
        core = agent._agents_by_thread[THREAD_ID]
        turns_after_first_run = len(core.messages)
        assert turns_after_first_run > 0

        await drain(agent, run_input("r2"))
        assert agent._agents_by_thread[THREAD_ID] is core
        assert core._session_manager is session_manager
        assert len(core.messages) > turns_after_first_run
        assert model.offered_tool_names[-1] == {"read_docs"}

    async def test_a_filtered_out_tool_keeps_the_calls_it_already_made(
        self, tmp_path
    ):
        """History is not rewritten, so the model still reads what it did.

        The filter answers "what may this request call", not "what happened on
        this thread". Removing the record would leave an assistant tool-use
        block with no result behind it.

        Backed by a real session manager on purpose: with none configured the
        thread's history is reconciled against ``RunAgentInput.messages`` every
        turn, so what a filter did to it could not be told apart from what the
        replay did.
        """
        from strands.session.file_session_manager import FileSessionManager

        session_manager = FileSessionManager(
            session_id="template-filter-history", storage_dir=str(tmp_path)
        )
        allowed_by_run = {"r1": None, "r2": ["read_docs"]}
        model = DeleteThenAnswerModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                session_manager_provider=lambda data: session_manager,
                template_tools_provider=lambda data: allowed_by_run[data.run_id],
            ),
        )

        def delete_calls(messages):
            found = []
            for message in messages:
                for block in message.get("content") or []:
                    tool_use = block.get("toolUse") if isinstance(block, dict) else None
                    if isinstance(tool_use, dict) and tool_use.get("name") == "delete_record":
                        found.append(tool_use["toolUseId"])
            return found

        await drain(agent, run_input("r1"))
        core = agent._agents_by_thread[THREAD_ID]
        recorded = delete_calls(core.messages)
        assert recorded, "the first run never called delete_record"

        await drain(agent, run_input("r2"))
        assert agent._agents_by_thread[THREAD_ID] is core
        assert delete_calls(core.messages) == recorded, (
            "filtering delete_record out erased the call it already made"
        )
        assert model.offered_tool_names[-1] == {"read_docs"}


class TestAParkedCallIsNotOrphaned:
    async def test_a_tool_awaiting_approval_stays_registered_while_filtered_out(self):
        """The rule ``sync_proxy_tools`` already applies, read off the checkpoint.

        The human's answer is routed back into the tool batch the run stopped
        inside. A tool absent from the registry at that moment turns the answer
        into a "tool not found" the model then re-fires, so the batch is exempt
        until the pause closes.
        """
        model = DeleteThenAnswerModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                tool_behaviors={"delete_record": ToolBehavior(interrupt_on_call=True)},
                template_tools_provider=lambda data: (
                    None if data.run_id == "r1" else ["read_docs"]
                ),
            ),
        )

        await drain(agent, run_input("r1"))
        core = agent._agents_by_thread[THREAD_ID]
        assert core._interrupt_state.activated, "the first run did not park"
        parked = list(agent._pending_interrupts_by_thread[THREAD_ID])
        assert parked, "no AG-UI interrupt was recorded for the pause"

        from ag_ui_strands import ResumeEntry

        events = await drain(
            agent,
            run_input(
                "r2",
                resume=[
                    ResumeEntry(
                        interrupt_id=parked[0],
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        assert agent._agents_by_thread[THREAD_ID] is core
        assert "delete_record" in core.tool_registry.registry, (
            "the tool the resume routes back into was filtered out from under it"
        )
        assert [e for e in events if e.type == EventType.RUN_ERROR] == []
        assert [e for e in events if e.type == EventType.RUN_FINISHED]

    async def test_a_plain_turn_against_a_pause_is_refused_before_the_filter_runs(
        self,
    ):
        """The pause is not reachable by a filter at all on a non-resume turn.

        A turn that submits no answer is refused ahead of the tool sync, so a
        provider that would have withheld the parked tool never runs and the
        checkpoint and its AG-UI bookkeeping are still there for the resume
        that follows.
        """
        from ag_ui_strands import ResumeEntry
        from tests.error_code_table import assert_contract_error

        consulted: list[str] = []

        def provider(data: RunAgentInput):
            consulted.append(data.run_id)
            return None if data.run_id == "r1" else ["read_docs"]

        model = DeleteThenAnswerModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                tool_behaviors={"delete_record": ToolBehavior(interrupt_on_call=True)},
                template_tools_provider=provider,
            ),
        )

        await drain(agent, run_input("r1"))
        core = agent._agents_by_thread[THREAD_ID]
        parked = dict(agent._pending_interrupts_by_thread[THREAD_ID])
        assert parked

        refused = await drain(agent, run_input("r2"))
        assert_contract_error(
            next(e for e in refused if e.type == EventType.RUN_ERROR),
            "PENDING_INTERRUPTS",
        )
        assert consulted == ["r1"], "the provider ran on a turn that was refused"
        assert core._interrupt_state.activated
        assert agent._pending_interrupts_by_thread[THREAD_ID] == parked
        assert "delete_record" in core.tool_registry.registry

        events = await drain(
            agent,
            run_input(
                "r3",
                resume=[
                    ResumeEntry(
                        interrupt_id=next(iter(parked)),
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )
        assert [e for e in events if e.type == EventType.RUN_ERROR] == []
        assert agent._agents_by_thread[THREAD_ID] is core
        assert consulted == ["r1", "r3"]

    async def test_a_parked_frontend_call_survives_a_filter_that_allows_nothing(self):
        """A template filter has no reach over client-declared tools.

        Client tools are re-synchronised from ``RunAgentInput.tools`` every
        request and their proxies are a different producer's entries, so the
        filter must not be able to remove one, parked or not.
        """
        client_tool = Tool(
            name="confirm_in_client",
            description="Confirm in the client",
            parameters={"type": "object", "properties": {}},
        )
        model = RecordingModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                tool_behaviors={
                    "confirm_in_client": ToolBehavior(
                        continue_after_frontend_call=False
                    )
                },
                template_tools_provider=lambda data: [],
            ),
        )

        await drain(agent, run_input("r1", tools=[client_tool]))
        core = agent._agents_by_thread[THREAD_ID]
        assert "confirm_in_client" in core.tool_registry.registry
        assert set(core.tool_registry.registry) == {"confirm_in_client"}

        await drain(agent, run_input("r2", tools=[client_tool]))
        assert agent._agents_by_thread[THREAD_ID] is core
        assert "confirm_in_client" in core.tool_registry.registry


class TestFilteringRemovesTheCapability:
    async def test_a_filtered_out_tool_the_model_calls_anyway_does_not_run(self):
        """The point of touching the registry rather than only the tool specs.

        Withholding a tool from the specs and leaving it registered would make
        the filter advice a model can ignore. A model calling the name anyway,
        because it was primed by a stale turn or by the visible history, has to
        be refused by the dispatcher rather than served.
        """
        executions: list[str] = []

        @tool(name="delete_record")
        def audited_delete(record_id: str) -> str:
            """Delete a record, recording that it ran."""
            executions.append(record_id)
            return f"deleted {record_id}"

        model = DeleteThenAnswerModel()
        agent = make_agent(
            model,
            config=StrandsAgentConfig(
                template_tools_provider=lambda data: ["read_docs"]
            ),
            tools=[read_docs, audited_delete],
        )

        events = await drain(agent, run_input("r1"))

        assert executions == [], "a filtered-out tool executed"
        assert model.offered_tool_names[0] == {"read_docs"}
        assert [e for e in events if e.type == EventType.RUN_ERROR] == []


class TestTheProviderFailureMode:
    async def test_a_raising_provider_ends_the_run_rather_than_running_unfiltered(
        self, caplog
    ):
        """Terminal, matching ``thread_agent_kwargs``.

        Degrading to an unfiltered run would hand the model exactly the tools
        the caller meant to withhold, which is the one outcome this hook exists
        to prevent.
        """
        from tests.error_code_table import assert_contract_error

        def provider(data: RunAgentInput):
            raise RuntimeError("authz lookup failed")

        model = RecordingModel()
        agent = make_agent(
            model, config=StrandsAgentConfig(template_tools_provider=provider)
        )

        with caplog.at_level(logging.ERROR, logger="ag_ui_strands.agent"):
            events = await drain(agent, run_input("r1"))

        assert [e.type for e in events] == [
            EventType.RUN_STARTED,
            EventType.RUN_ERROR,
        ]
        assert_contract_error(events[1], "TEMPLATE_TOOLS_PROVIDER_ERROR")
        assert "authz lookup failed" in events[1].message
        assert model.offered_tool_names == [], "the model ran despite the failure"
        assert "template_tools_provider failed" in caplog.text

    async def test_the_run_error_is_bracketed_by_a_run_started(self):
        """A client that brackets on the lifecycle events needs the opener."""
        agent = make_agent(
            RecordingModel(),
            config=StrandsAgentConfig(
                template_tools_provider=lambda data: (_ for _ in ()).throw(
                    ValueError("boom")
                )
            ),
        )
        events = await drain(agent, run_input("r1"))
        assert events[0].type == EventType.RUN_STARTED
        assert events[0].run_id == "r1"


class TestNoProviderConfigured:
    async def test_the_registry_is_untouched_when_no_provider_is_configured(self):
        model = RecordingModel()
        agent = make_agent(model)
        await drain(agent, run_input("r1"))
        await drain(agent, run_input("r2"))
        assert registry_names(agent) == {"read_docs", "delete_record"}
        assert model.offered_tool_names == [
            {"read_docs", "delete_record"},
            {"read_docs", "delete_record"},
        ]

    async def test_no_provider_never_reaches_the_sync(self, monkeypatch):
        """Byte-identical behaviour, asserted as "the code did not run".

        A sync that happens to be a no-op today could stop being one; the
        contract is that an unconfigured hook adds no step at all.
        """
        import ag_ui_strands.agent as agent_module

        def explode(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("sync_template_tools ran with no provider configured")

        monkeypatch.setattr(agent_module, "sync_template_tools", explode)
        agent = make_agent(RecordingModel())
        events = await drain(agent, run_input("r1"))
        assert [e for e in events if e.type == EventType.RUN_ERROR] == []


# ---------------------------------------------------------------------------
# The pieces, in isolation
# ---------------------------------------------------------------------------


class TestSelectionResolution:
    def test_names_and_tool_objects_are_both_accepted(self):
        index = index_template_tools([read_docs, delete_record])
        assert resolve_template_tool_selection(["read_docs"], index) == {"read_docs"}
        assert resolve_template_tool_selection([delete_record], index) == {
            "delete_record"
        }

    def test_none_and_empty_are_different_answers(self):
        index = index_template_tools([read_docs])
        assert resolve_template_tool_selection(None, index) is None
        assert resolve_template_tool_selection([], index) == set()

    def test_a_tool_the_template_never_contributed_is_refused(self, caplog):
        @tool
        def smuggled() -> str:
            """Not on the template."""
            return "no"

        index = index_template_tools([read_docs])
        with caplog.at_level(logging.WARNING, logger="ag_ui_strands.template_tools"):
            assert resolve_template_tool_selection(
                ["read_docs", "smuggled", smuggled], index
            ) == {"read_docs"}
        assert caplog.text.count("does not contribute") == 2

    def test_an_entry_that_names_no_tool_is_refused(self, caplog):
        index = index_template_tools([read_docs])
        with caplog.at_level(logging.WARNING, logger="ag_ui_strands.template_tools"):
            assert resolve_template_tool_selection([None, 7, ""], index) == set()
        assert caplog.text.count("names no tool") == 3


class TestSyncTemplateTools:
    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.process_tools([read_docs, delete_record])
        return registry

    def test_a_removed_tool_is_restored_as_the_same_instance(self):
        registry = self._registry()
        original = registry.registry["delete_record"]

        sync_template_tools(registry, [read_docs, delete_record], ["read_docs"])
        assert set(registry.registry) == {"read_docs"}

        sync_template_tools(registry, [read_docs, delete_record], None)
        assert registry.registry["delete_record"] is original

    def test_an_entry_another_producer_owns_is_left_alone(self):
        registry = self._registry()
        registry.registry["delete_record"] = "not ours"

        sync_template_tools(registry, [read_docs, delete_record], ["read_docs"])
        assert registry.registry["delete_record"] == "not ours"

        sync_template_tools(registry, [read_docs, delete_record], None)
        assert registry.registry["delete_record"] == "not ours"

    def test_exempt_names_are_kept_however_the_selection_reads(self):
        registry = self._registry()
        kept = sync_template_tools(
            registry,
            [read_docs, delete_record],
            [],
            exempt_names={"delete_record"},
        )
        assert kept == {"delete_record"}
        assert set(registry.registry) == {"delete_record"}

    def test_the_returned_set_is_what_the_registry_holds(self):
        registry = self._registry()
        assert sync_template_tools(
            registry, [read_docs, delete_record], ["read_docs"]
        ) == {"read_docs"}
        assert sync_template_tools(registry, [read_docs, delete_record], None) == {
            "read_docs",
            "delete_record",
        }


class TestParkedBatchToolNames:
    def _agent(self, state):
        class _Agent:
            _interrupt_state = state

        return _Agent()

    def test_an_idle_agent_parks_nothing(self):
        assert parked_batch_tool_names(self._agent(InterruptStateStub())) == set()
        assert parked_batch_tool_names(object()) == set()

    def test_every_tool_in_the_parked_batch_is_named(self):
        state = InterruptStateStub(
            interrupts={"i1": StrandsInterrupt("i1", "ag_ui:tool_call:delete_record")},
        )
        state.activate(
            {
                "tool_use_message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "a", "name": "delete_record"}},
                        {"toolUse": {"toolUseId": "b", "name": "read_docs"}},
                        {"text": "thinking"},
                    ],
                }
            }
        )
        assert parked_batch_tool_names(self._agent(state)) == {
            "delete_record",
            "read_docs",
        }

    def test_a_checkpoint_without_a_tool_batch_names_nothing(self):
        state = InterruptStateStub()
        state.activate({"tool_results": []})
        assert parked_batch_tool_names(self._agent(state)) == set()
