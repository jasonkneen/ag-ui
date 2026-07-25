"""FastAPI endpoint helper."""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder

from .agent import ManagedAgentsAgent


def add_managed_agents_fastapi_endpoint(
    app: FastAPI, agent: ManagedAgentsAgent, path: str = "/"
) -> None:
    """Add a Managed Agents endpoint to the FastAPI app.

    POST `path` runs one turn and streams the encoded AG-UI events. Closing
    the response (a client disconnect) cancels the run, which interrupts the
    managed session. GET `{path}/health` reports the managed agent id.
    """

    @app.post(path)
    async def managed_agents_endpoint(input_data: RunAgentInput, request: Request):
        accept_header = request.headers.get("accept")
        encoder = EventEncoder(accept=accept_header)

        async def event_generator():
            async for event in agent.run(input_data):
                yield encoder.encode(event)

        return StreamingResponse(
            event_generator(), media_type=encoder.get_content_type()
        )

    @app.get(f"{path.rstrip('/')}/health")
    def health():
        """Health check."""
        return {"status": "ok", "agent": {"managedAgentId": agent.managed_agent_id}}
