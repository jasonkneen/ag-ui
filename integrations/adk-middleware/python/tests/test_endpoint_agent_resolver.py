#!/usr/bin/env python
"""Black-box endpoint tests for minimal async agent resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunStartedEvent,
    ToolMessage,
    UserMessage,
)
from ag_ui_adk.adk_agent import ADKAgent
from ag_ui_adk.endpoint import add_adk_fastapi_endpoint, create_adk_app
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _run_input(
    *,
    thread_id: str = "thread-1",
    run_id: str = "run-1",
    messages=None,
    state=None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        messages=messages
        if messages is not None
        else [UserMessage(id="user-1", role="user", content="hello")],
        tools=[],
        context=[],
        state={} if state is None else state,
        forwarded_props={},
    )


def _agent(name: str, *, capabilities=None):
    agent = MagicMock(spec=ADKAgent)
    agent.name = name
    agent.get_capabilities.return_value = capabilities

    async def run(input_data):
        yield RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id,
        )

    agent.run = MagicMock(side_effect=run)
    return agent


def _state_agent(name: str, state: dict):
    agent = _agent(name)
    adk_agent = MagicMock()
    adk_agent.name = name
    agent._adk_agent = adk_agent
    agent._static_app_name = f"{name}_app"
    agent._static_user_id = f"{name}_user"
    agent._session_lookup_cache = {}
    agent._get_session_metadata = MagicMock(
        return_value=(f"{name}_session", f"{name}_app", f"{name}_user")
    )
    agent._session_manager = MagicMock()
    agent._session_manager.get_session_state = AsyncMock(return_value=state)
    agent._session_manager._session_service = MagicMock()
    session = MagicMock()
    session.events = []
    agent._session_manager._session_service.get_session = AsyncMock(
        return_value=session
    )
    return agent


def test_resolver_runs_after_extractor_and_can_fallback_to_default_agent():
    default_agent = _agent("default")
    selected_agent = _agent("selected")
    resolver_inputs = []

    async def extractor(request, input_data):
        return {"tenant": request.headers["x-tenant"], "from_extractor": True}

    async def resolver(request, input_data):
        resolver_inputs.append(input_data)
        if input_data.state["tenant"] == "selected":
            return selected_agent
        return None

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app,
        default_agent,
        path="/agent",
        extract_state_from_request=extractor,
        agent_resolver=resolver,
    )
    client = TestClient(app)

    selected_response = client.post(
        "/agent",
        json=_run_input(state={"client_state": "preserved"}).model_dump(),
        headers={"x-tenant": "selected"},
    )
    fallback_response = client.post(
        "/agent",
        json=_run_input(run_id="run-2").model_dump(),
        headers={"x-tenant": "unknown"},
    )

    assert selected_response.status_code == 200
    assert fallback_response.status_code == 200
    assert selected_agent.run.call_count == 1
    assert default_agent.run.call_count == 1
    assert resolver_inputs[0].state == {
        "client_state": "preserved",
        "tenant": "selected",
        "from_extractor": True,
    }


def test_resolver_can_route_by_request_headers_and_query_params():
    default_agent = _agent("default")
    selected_agent = _agent("selected")

    async def resolver(request, input_data):
        if (
            request.headers.get("x-route-agent") == "selected"
            and request.query_params.get("region") == "west"
        ):
            return selected_agent
        return None

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app, default_agent, path="/agent", agent_resolver=resolver
    )
    client = TestClient(app)

    response = client.post(
        "/agent?region=west",
        json=_run_input().model_dump(),
        headers={"x-route-agent": "selected"},
    )

    assert response.status_code == 200
    selected_agent.run.assert_called_once()
    default_agent.run.assert_not_called()


def test_create_adk_app_forwards_agent_resolver_functionally():
    default_agent = _agent("default")
    selected_agent = _agent("selected")

    async def resolver(request, input_data):
        return selected_agent if input_data.state.get("agent") == "selected" else None

    app = create_adk_app(default_agent, path="/agent", agent_resolver=resolver)
    client = TestClient(app)

    response = client.post(
        "/agent", json=_run_input(state={"agent": "selected"}).model_dump()
    )

    assert response.status_code == 200
    selected_agent.run.assert_called_once()
    default_agent.run.assert_not_called()


def test_capabilities_uses_resolver_after_extractor_and_defaults_on_none():
    default_agent = _agent("default", capabilities={"identity": {"name": "default"}})
    selected_agent = _agent(
        "selected", capabilities={"identity": {"name": "selected"}}
    )
    resolver_inputs = []

    async def extractor(request, input_data):
        if "x-capability-agent" in request.headers:
            return {"capability_agent": request.headers["x-capability-agent"]}
        return {}

    async def resolver(request, input_data):
        resolver_inputs.append(input_data)
        if input_data.state.get("capability_agent") == "selected":
            return selected_agent
        return None

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app,
        default_agent,
        path="/agent",
        extract_state_from_request=extractor,
        agent_resolver=resolver,
    )
    client = TestClient(app)

    selected_response = client.get(
        "/agent/capabilities", headers={"x-capability-agent": "selected"}
    )
    fallback_response = client.get("/agent/capabilities")

    assert selected_response.status_code == 200
    assert selected_response.json()["identity"]["name"] == "selected"
    assert fallback_response.status_code == 200
    assert fallback_response.json()["identity"]["name"] == "default"
    assert resolver_inputs[0].state == {"capability_agent": "selected"}
    assert resolver_inputs[0].messages == []


def test_agents_state_uses_resolved_agent_after_extractor_merge():
    default_agent = _state_agent("default", {"source": "default"})
    selected_agent = _state_agent("selected", {"source": "selected"})
    resolver_inputs = []

    async def extractor(request, input_data):
        return {"state_agent": request.headers["x-state-agent"]}

    async def resolver(request, input_data):
        resolver_inputs.append(input_data)
        if input_data.state["state_agent"] == "selected":
            return selected_agent
        return None

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app,
        default_agent,
        path="/",
        extract_state_from_request=extractor,
        agent_resolver=resolver,
    )
    client = TestClient(app)

    response = client.post(
        "/agents/state",
        json={"threadId": "thread-state"},
        headers={"x-state-agent": "selected"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == {"source": "selected"}
    assert resolver_inputs[0].thread_id == "thread-state"
    assert resolver_inputs[0].state == {"state_agent": "selected"}
    selected_agent._session_manager.get_session_state.assert_awaited_once()
    default_agent._session_manager.get_session_state.assert_not_awaited()


def test_tool_result_routing_remains_resolver_responsibility():
    default_agent = _agent("default")
    selected_agent = _agent("selected")
    resolver = AsyncMock(return_value=selected_agent)

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app, default_agent, path="/agent", agent_resolver=resolver
    )
    client = TestClient(app)

    response = client.post(
        "/agent",
        json=_run_input(
            messages=[
                ToolMessage(
                    id="tool-message-1",
                    role="tool",
                    tool_call_id="tool-call-1",
                    content='{"ok": true}',
                )
            ]
        ).model_dump(),
    )

    assert response.status_code == 200
    resolver.assert_awaited_once()
    selected_agent.run.assert_called_once()
    default_agent.run.assert_not_called()
