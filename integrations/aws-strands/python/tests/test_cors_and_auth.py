"""Tests for CORS defaults, content-type enforcement, and authentication."""

from __future__ import annotations

import inspect
import json
import warnings
from types import SimpleNamespace

import pytest
from fastapi import Header, HTTPException
from fastapi.testclient import TestClient

from ag_ui_strands.utils import create_strands_app


def _implicit_cors_warnings(caught):
    return [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning)
        and "Implicit wildcard CORS" in str(warning.message)
    ]


@pytest.fixture
def agent():
    return SimpleNamespace(name="test-agent")


# ---------------------------------------------------------------------------
# CORS defaults
# ---------------------------------------------------------------------------


class TestCorsDefaults:
    def test_factory_exposes_explicit_cors_switch(self):
        parameters = inspect.signature(create_strands_app).parameters

        assert "cors_enabled" in parameters
        assert parameters["cors_enabled"].default is None

    def test_implicit_default_preserves_wildcard_origin(self, agent):
        with pytest.warns(FutureWarning, match="Implicit wildcard CORS"):
            client = TestClient(create_strands_app(agent))

        resp = client.get("/ping", headers={"Origin": "https://evil.example"})

        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_empty_origins_preserve_legacy_wildcard_origin(self, agent):
        with pytest.warns(FutureWarning, match="Implicit wildcard CORS"):
            client = TestClient(create_strands_app(agent, origins=[]))

        resp = client.get("/ping", headers={"Origin": "https://evil.example"})

        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_explicit_origin_is_allowed(self, agent):
        client = TestClient(create_strands_app(agent, origins=["https://app.example"]))

        resp = client.get("/ping", headers={"Origin": "https://app.example"})

        assert resp.headers.get("access-control-allow-origin") == "https://app.example"

    def test_other_origin_still_rejected_when_allowlist_set(self, agent):
        client = TestClient(create_strands_app(agent, origins=["https://app.example"]))

        resp = client.get("/ping", headers={"Origin": "https://evil.example"})

        assert resp.headers.get("access-control-allow-origin") is None

    def test_wildcard_remains_available_as_explicit_opt_in(self, agent):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client = TestClient(create_strands_app(agent, origins=["*"]))

        assert _implicit_cors_warnings(caught) == []

        resp = client.get("/ping", headers={"Origin": "https://anything.example"})

        assert resp.headers.get("access-control-allow-origin") == "*"
        # A wildcard origin must never be paired with credentials.
        assert resp.headers.get("access-control-allow-credentials") is None

    def test_explicit_cors_disable_adds_no_middleware_or_warning(self, agent):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client = TestClient(create_strands_app(agent, cors_enabled=False))

        assert _implicit_cors_warnings(caught) == []

        resp = client.get("/ping", headers={"Origin": "https://evil.example"})

        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") is None

    def test_explicit_cors_enable_acknowledges_wildcard(self, agent):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client = TestClient(create_strands_app(agent, cors_enabled=True))

        assert _implicit_cors_warnings(caught) == []

        resp = client.get("/ping", headers={"Origin": "https://anything.example"})

        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_existing_origin_configuration_keeps_wildcard_methods(self, agent):
        client = TestClient(create_strands_app(agent, origins=["https://app.example"]))

        resp = client.options(
            "/",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert resp.status_code == 200
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "DELETE" in allowed

    def test_existing_origin_configuration_keeps_wildcard_headers(self, agent):
        client = TestClient(create_strands_app(agent, origins=["https://app.example"]))

        resp = client.options(
            "/",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-not-allowed",
            },
        )

        assert resp.status_code == 200
        allowed = resp.headers.get("access-control-allow-headers", "").lower()
        assert "x-not-allowed" in allowed

    def test_explicit_method_allowlist_rejects_unlisted_method(self, agent):
        client = TestClient(
            create_strands_app(
                agent,
                origins=["https://app.example"],
                allow_methods=["POST"],
            )
        )

        resp = client.options(
            "/",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert resp.status_code == 400
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "DELETE" not in allowed
        assert "POST" in allowed

    def test_explicit_header_allowlist_rejects_unlisted_header(self, agent):
        client = TestClient(
            create_strands_app(
                agent,
                origins=["https://app.example"],
                allow_headers=["Content-Type"],
            )
        )

        resp = client.options(
            "/",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-not-allowed",
            },
        )

        assert resp.status_code == 400
        allowed = resp.headers.get("access-control-allow-headers", "").lower()
        assert "x-not-allowed" not in allowed


# ---------------------------------------------------------------------------
# Content type
# ---------------------------------------------------------------------------


class _RecordingAgent:
    name = "test-agent"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, input_data):
        self.calls += 1
        if False:
            yield input_data


_VALID_BODY = {
    "threadId": "thread-1",
    "runId": "run-1",
    "state": {},
    "messages": [
        {
            "id": "message-1",
            "role": "user",
            "content": "hello",
        }
    ],
    "tools": [],
    "context": [],
    "forwardedProps": {},
}


class TestContentType:
    def test_missing_content_type_rejects_valid_body_before_agent_run(self):
        from fastapi import FastAPI

        from ag_ui_strands.endpoint import add_strands_fastapi_endpoint

        agent = _RecordingAgent()
        app = FastAPI()
        add_strands_fastapi_endpoint(app, agent, "/agent")
        client = TestClient(app)

        resp = client.post("/agent", content=json.dumps(_VALID_BODY))

        assert agent.calls == 0
        assert resp.status_code == 415

    def test_non_json_content_type_rejects_valid_body_before_agent_run(self):
        from fastapi import FastAPI

        from ag_ui_strands.endpoint import add_strands_fastapi_endpoint

        agent = _RecordingAgent()
        app = FastAPI()
        add_strands_fastapi_endpoint(app, agent, "/agent")
        client = TestClient(app)

        resp = client.post(
            "/agent",
            content=json.dumps(_VALID_BODY),
            headers={"Content-Type": "text/plain"},
        )

        assert agent.calls == 0
        assert resp.status_code == 415

    @pytest.mark.parametrize(
        "content_type",
        ["application/json", "application/vnd.ag-ui+json; charset=utf-8"],
    )
    def test_json_compatible_content_type_reaches_agent(self, content_type):
        from fastapi import FastAPI

        from ag_ui_strands.endpoint import add_strands_fastapi_endpoint

        agent = _RecordingAgent()
        app = FastAPI()
        add_strands_fastapi_endpoint(app, agent, "/agent")
        client = TestClient(app)

        resp = client.post(
            "/agent",
            content=json.dumps(_VALID_BODY),
            headers={"Content-Type": content_type},
        )

        assert resp.status_code == 200
        assert agent.calls == 1


# ---------------------------------------------------------------------------
# Authentication hook
# ---------------------------------------------------------------------------


def _require_token(authorization: str | None = Header(default=None)) -> None:
    if authorization != "Bearer secret":
        raise HTTPException(status_code=401, detail="Unauthorized")


class TestAuthHook:
    def test_agent_endpoint_rejects_unauthenticated_request(self, agent):
        client = TestClient(
            create_strands_app(agent, auth=_require_token, cors_enabled=False)
        )

        resp = client.post("/", json={})

        assert resp.status_code == 401

    def test_agent_endpoint_accepts_authenticated_request(self, agent):
        client = TestClient(
            create_strands_app(agent, auth=_require_token, cors_enabled=False)
        )

        resp = client.post(
            "/", json={}, headers={"Authorization": "Bearer secret"}
        )

        # Not 401: the auth hook passed and the request reached body validation.
        assert resp.status_code != 401

    def test_ping_stays_unauthenticated(self, agent):
        """The health probe must keep working for load balancers / AgentCore."""
        client = TestClient(
            create_strands_app(agent, auth=_require_token, cors_enabled=False)
        )

        assert client.get("/ping").status_code == 200

    def test_endpoint_helper_accepts_auth_dependency(self, agent):
        from fastapi import FastAPI

        from ag_ui_strands.endpoint import add_strands_fastapi_endpoint

        app = FastAPI()
        add_strands_fastapi_endpoint(app, agent, "/agent", auth=_require_token)
        client = TestClient(app)

        assert client.post("/agent", json={}).status_code == 401

    def test_no_auth_by_default_is_unchanged(self, agent):
        client = TestClient(create_strands_app(agent, cors_enabled=False))

        assert client.post("/", json={}).status_code != 401
