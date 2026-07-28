import uuid
import copy
import json
import weakref
from typing import Any, Optional, Protocol, cast, runtime_checkable
from crewai import Crew, Flow
from crewai.flow import start
from crewai.cli.crew_chat import (
  initialize_chat_llm as crew_chat_initialize_chat_llm,
  generate_crew_chat_inputs as crew_chat_generate_crew_chat_inputs,
  generate_crew_tool_schema as crew_chat_generate_crew_tool_schema,
  build_system_message as crew_chat_build_system_message,
  create_tool_function as crew_chat_create_tool_function
)
from litellm import acompletion
from ._env import _parse_env_float
from .sdk import (
  copilotkit_stream,
  copilotkit_exit,
  copilotkit_emit_state,
)

# Cache of generated chat-input schemas, keyed by the ``@CrewBase``
# instance passed to ``add_crewai_crew_fastapi_endpoint`` (CPK-7717
# review round 2, finding 3).
#
# The primary store is a ``WeakKeyDictionary`` keyed on the crew OBJECT
# itself — NOT ``id(crew)``. Keying on ``id`` was unsafe: CPython reuses
# ``id`` values once an object is garbage-collected, and now that
# ``ChatWithCrewFlow`` is exported for direct construction the wrapper is
# not necessarily retained for the process lifetime. A freshly allocated
# crew wrapper could therefore inherit a collected wrapper's ``id`` and
# silently be served the collected crew's schema. A ``WeakKeyDictionary``
# keys on true object identity and auto-evicts an entry the moment its
# crew is collected, so no stale schema can outlive its crew.
#
# ``_CREW_INPUTS_FALLBACK`` handles the unlikely crew object that is not
# weak-referenceable (or not hashable): we key on ``id`` but store a
# STRONG reference to the crew alongside the schema, which keeps the
# object alive so its ``id`` cannot be reused while the entry is live,
# and we verify the stored object *is* the crew before serving (dropping
# the entry on mismatch). Either path preserves the caching win for a
# repeated construction of the same crew (which otherwise re-issues the
# LLM call in ``generate_crew_chat_inputs``).
_CREW_INPUTS_CACHE: "weakref.WeakKeyDictionary[Any, Any]" = (
    weakref.WeakKeyDictionary()
)
_CREW_INPUTS_FALLBACK: dict = {}


def _crew_inputs_cache_get(crew: Any) -> Any:
    """Return the cached chat-input schema for ``crew`` or ``None`` on miss.

    ``None`` is an unambiguous miss sentinel: ``generate_crew_chat_inputs``
    always returns a populated ``ChatInputs`` object, never ``None``.
    """
    try:
        cached = _CREW_INPUTS_CACHE.get(crew)
    except TypeError:
        # ``crew`` is unhashable — cannot live in the WeakKeyDictionary.
        cached = None
    if cached is not None:
        return cached
    entry = _CREW_INPUTS_FALLBACK.get(id(crew))
    if entry is not None:
        stored_crew, schema = entry
        if stored_crew is crew:
            return schema
        # ``id`` was reused by a DIFFERENT object; drop the stale entry so
        # we regenerate rather than serve another crew's schema.
        _CREW_INPUTS_FALLBACK.pop(id(crew), None)
    return None


def _crew_inputs_cache_set(crew: Any, schema: Any) -> None:
    """Cache ``schema`` for ``crew`` in the identity-safe store."""
    try:
        _CREW_INPUTS_CACHE[crew] = schema
        return
    except TypeError:
        # Not weak-referenceable (or unhashable): fall back to an id key
        # with a STRONG crew reference so the id cannot be reused while
        # live, plus an identity check on read.
        _CREW_INPUTS_FALLBACK[id(crew)] = (crew, schema)


@runtime_checkable
class CrewBaseInstance(Protocol):
    """Structural type for a ``@CrewBase``-decorated crew instance.

    ``add_crewai_crew_fastapi_endpoint`` and ``ChatWithCrewFlow`` require a
    class decorated with crewai's ``@CrewBase`` — NOT a bare
    :class:`crewai.Crew`. The flow calls ``crew.crew()`` to build the
    underlying ``Crew`` and reads the crew's name; a plain ``Crew`` has
    neither a ``.crew()`` factory nor a name accessor, so annotating the
    parameter as ``Crew`` was actively misleading (CPK-7717 defect 5).

    Name accessor (CPK-7717 review round 2, finding 2): crewai's
    ``@CrewBase`` decorator does NOT expose a public ``.name`` attribute —
    it sets ``_crew_name`` on the wrapped class (to the decorated class'
    ``__name__``). We therefore express the structural requirement as
    ``crew()`` + ``_crew_name``. The runtime name reader
    (:func:`_read_crew_name`) additionally accepts a hand-rolled ``.name``
    for flexibility, but ``_crew_name`` is the canonical @CrewBase
    accessor and the one the protocol pins.

    ``@runtime_checkable`` lets callers/tests assert conformance via
    ``isinstance`` (structural: method/attribute presence only).
    """

    _crew_name: str

    def crew(self) -> Crew:  # pragma: no cover - structural protocol stub
        ...


def _read_crew_name(crew: Any) -> str:
    """Return a non-empty crew name from a ``@CrewBase`` instance.

    Accepts a hand-rolled ``.name`` OR crewai's canonical ``_crew_name``
    (set by the ``@CrewBase`` decorator to the class name). crewai's
    ``ChatInputs.crew_name`` is a required non-empty ``str`` and is used
    as the crew tool's function name, so a missing/empty/non-string name
    would raise a Pydantic validation error deep inside
    ``generate_crew_chat_inputs``. Fail loudly here with an actionable
    message instead (CPK-7717 review round 2, finding 2).
    """
    for attr in ("name", "_crew_name"):
        value = getattr(crew, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(
        "Could not determine a crew name: the crew exposes neither a "
        "non-empty ``name`` nor ``_crew_name`` attribute. Pass a "
        "``@CrewBase``-decorated crew (crewai sets ``_crew_name`` to the "
        "decorated class name) or set a non-empty ``name`` — crewai's "
        "ChatInputs.crew_name requires a non-empty string."
    )

# Per-read idle guard (seconds) for LiteLLM streaming requests. LiteLLM
# forwards this to the underlying HTTP client, where it acts as a
# *per-read* / socket-recv timeout — NOT a session-level ceiling. That means
# a trickle-feeding server can still keep the coroutine alive indefinitely
# by sending a single byte before each timeout expires; the session-level
# cap for that scenario is enforced by ``AGUI_CREWAI_FLOW_TIMEOUT_SECONDS``
# in ``endpoint.py``. Override this per-read guard with the
# ``AGUI_CREWAI_LLM_TIMEOUT_SECONDS`` environment variable; set to a
# non-positive value to disable it (the outer flow ceiling still applies).
_DEFAULT_LLM_TIMEOUT_SECONDS = 120.0


def _llm_timeout_seconds() -> float | None:
    """Return the configured LLM read timeout, or ``None`` to disable it.

    A non-positive value (``0`` / negative) disables the read timeout. NaN
    and any other non-finite float is treated as unparseable and falls back
    to the default — ``float('nan') > 0`` is False, which would otherwise
    silently disable the guard. Mirrors the NaN handling in
    ``endpoint._flow_timeout_seconds`` (R5 HIGH #1).

    CR7 LOW: delegates to ``_env._parse_env_float`` so the three
    env-parsed float helpers (flow ceiling / cancel-join ceiling / LLM
    read timeout) share a single parse + policy path rather than
    triplicating the scaffolding. CR8 MEDIUM: the helper lives on a
    neutral ``_env`` module (rather than ``endpoint``) so we can
    import it at module load time without a circular dependency
    (``endpoint`` imports ``ChatWithCrewFlow`` from ``crews``).
    """
    return _parse_env_float(
        "AGUI_CREWAI_LLM_TIMEOUT_SECONDS",
        _DEFAULT_LLM_TIMEOUT_SECONDS,
        allow_disable=True,
    )


CREW_EXIT_TOOL = {
    "type": "function",
    "function": {
        "name": "crew_exit",
        "description": "Call this when the user has indicated that they are done with the crew",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


class ChatWithCrewFlow(Flow):
    """Chat with crew"""

    def __init__(
            self, *,
            crew: "CrewBaseInstance"
        ):
        super().__init__()


        self.crew = copy.deepcopy(cast(Any, crew).crew())

        if self.crew.chat_llm is None:
            raise ValueError("Crew chat LLM is not set")

        # Read the crew name from the real ``@CrewBase`` accessor
        # (``_crew_name``) or a hand-rolled ``.name`` — never ``crew.name``
        # unconditionally, which AttributeErrors on a real @CrewBase
        # instance (CPK-7717 review round 2, finding 2). Fails loudly with
        # an actionable message if neither yields a usable non-empty name,
        # rather than passing ``None`` into ``ChatInputs`` (validation
        # error deep in ``generate_crew_chat_inputs``).
        self.crew_name = _read_crew_name(crew)
        self.chat_llm = crew_chat_initialize_chat_llm(self.crew)

        # Identity-safe cache keyed on the crew object itself (finding 3),
        # not ``id(crew)`` which is reused after GC.
        cached = _crew_inputs_cache_get(crew)
        if cached is None:
            self.crew_chat_inputs = crew_chat_generate_crew_chat_inputs(
                self.crew,
                self.crew_name,
                self.chat_llm
            )
            _crew_inputs_cache_set(crew, self.crew_chat_inputs)
        else:
            self.crew_chat_inputs = cached

        self.crew_tool_schema = crew_chat_generate_crew_tool_schema(self.crew_chat_inputs)
        self.system_message = crew_chat_build_system_message(self.crew_chat_inputs)

    def _completion_llm_kwargs(self) -> dict:
        """Return the connection kwargs for a litellm ``acompletion`` call.

        Defect 1 (CPK-7717): the completion call sites previously passed
        the raw ``self.crew.chat_llm`` model STRING to ``acompletion``,
        which drops the credentials/endpoint that ``initialize_chat_llm``
        resolved. litellm then falls back to environment/default
        credentials, breaking local and self-hosted models
        (CopilotKit#2742).

        Round-2 review finding 1: forwarding only ``model`` / ``api_key``
        / ``base_url`` still dropped ``api_base`` / ``api_version`` and the
        provider-specific ``additional_params`` — so Azure and other
        custom-endpoint users still hit the wrong endpoint. crewai's own
        ``LLM._prepare_completion_params`` forwards ``api_base``,
        ``base_url``, ``api_version``, ``api_key`` AND spreads
        ``additional_params`` (see ``crewai/llm.py``); we mirror that here
        so both completion call sites authenticate exactly the way
        ``generate_crew_chat_inputs`` does (it drives the same LLM object).

        Defensive: ``getattr`` with a fall back to the crew's model string
        keeps the helper working if ``chat_llm`` was not resolved (e.g. in
        unit tests that construct the flow via ``__new__``). Every field is
        forwarded only when present/non-None so we never override litellm's
        own resolution with ``None``. ``additional_params`` is spread first
        so the explicit connection fields win on any key collision.
        """
        llm = getattr(self, "chat_llm", None)
        model = getattr(llm, "model", None) or self.crew.chat_llm
        kwargs: dict = {"model": model}

        additional = getattr(llm, "additional_params", None)
        if isinstance(additional, dict):
            for key, value in additional.items():
                if value is not None:
                    kwargs[key] = value

        for attr in ("api_key", "base_url", "api_base", "api_version"):
            value = getattr(llm, attr, None)
            if value is not None:
                kwargs[attr] = value
        return kwargs

    @start()
    async def chat(self):
        """Chat with the crew"""

        system_message = self.system_message
        if self.state.get("inputs"):
            system_message += "\n\nCurrent inputs: " + json.dumps(self.state["inputs"])

        messages = [
            {
                "role": "system",
                "content": system_message,
                "id": str(uuid.uuid4()) + "-system"
            },
            *self.state["messages"]
        ]

        tools = [action for action in self.state["copilotkit"]["actions"]
                 if action["function"]["name"] != self.crew_name]

        tools += [self.crew_tool_schema, CREW_EXIT_TOOL]

        response = await copilotkit_stream(
            await acompletion(
                **self._completion_llm_kwargs(),
                messages=messages,
                tools=tools,
                parallel_tool_calls=False,
                stream=True,
                timeout=_llm_timeout_seconds(),
            )
        )

        message = cast(Any, response).choices[0]["message"]
        self.state["messages"].append(message)

        if message.get("tool_calls"):
            if message["tool_calls"][0]["function"]["name"] == self.crew_name:
                # run the crew
                crew_function = crew_chat_create_tool_function(self.crew, messages)
                args = json.loads(message["tool_calls"][0]["function"]["arguments"])
                result = crew_function(**args)

                if isinstance(result, str):
                    self.state["outputs"] = result
                elif hasattr(result, "json_dict"):
                    self.state["outputs"] = result.json_dict
                elif hasattr(result, "raw"):
                    self.state["outputs"] = result.raw
                else:
                    raise ValueError("Unexpected result type", type(result))

                self.state["messages"].append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": message["tool_calls"][0]["id"]
                })

                # Defect 3 (CPK-7717): surface the state mutation from the
                # crew run so a StateSnapshotEvent reaches the bridge as
                # soon as the crew output is applied, rather than only at
                # method-finish. NOTE on granularity: this emits ONE
                # snapshot after the crew result lands. Per-tool /
                # intermediate mutations *inside* the crew run are not yet
                # observable here — surfacing those requires the CrewAI
                # StreamFrame integration tracked in CPK-7719. Until then
                # this is the finest granularity reachable without that
                # work.
                await copilotkit_emit_state(self.state)

                # Defect 2 (CPK-7717): a backend tool result on its own
                # leaves the assistant silent — the single-pass @start()
                # chat had no follow-up completion after the crew tool ran
                # (the crew_exit branch below always had one). Issue a
                # streamed follow-up so the assistant produces text about
                # the crew result. ``tool_choice="none"`` forces a text
                # answer (no further tool calls), mirroring the crew_exit
                # branch.
                response = await copilotkit_stream(
                    await acompletion( # pylint: disable=too-many-arguments
                        **self._completion_llm_kwargs(),
                        messages=[
                            {
                                "role": "system",
                                "content": "The crew has finished running. "
                                           "Use the tool result to answer the "
                                           "user's request.",
                                "id": str(uuid.uuid4()) + "-system"
                            },
                            *self.state["messages"]
                        ],
                        tools=tools,
                        parallel_tool_calls=False,
                        stream=True,
                        tool_choice="none",
                        timeout=_llm_timeout_seconds(),
                    )
                )
                message = cast(Any, response).choices[0]["message"]
                self.state["messages"].append(message)
            elif message["tool_calls"][0]["function"]["name"] == CREW_EXIT_TOOL["function"]["name"]:
                await copilotkit_exit()
                self.state["messages"].append({
                    "role": "tool",
                    "content": "Crew exited",  # E2E: aimock-setup.ts matches this exact string
                    "tool_call_id": message["tool_calls"][0]["id"]
                })

                response = await copilotkit_stream(
                    await acompletion( # pylint: disable=too-many-arguments
                        **self._completion_llm_kwargs(),
                        messages = [
                            {
                                "role": "system",
                                "content": "Indicate to the user that the crew has exited",
                                "id": str(uuid.uuid4()) + "-system"
                            },
                            *self.state["messages"]
                        ],
                        tools=tools,
                        parallel_tool_calls=False,
                        stream=True,
                        tool_choice="none",
                        timeout=_llm_timeout_seconds(),
                    )
                )
                message = cast(Any, response).choices[0]["message"]
                self.state["messages"].append(message)
