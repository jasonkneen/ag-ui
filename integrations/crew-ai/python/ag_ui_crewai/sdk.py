"""
This is a placeholder for the copilotkit_stream function.
"""

import copy
import uuid
from dataclasses import dataclass
from typing import (
  List,
  Any,
  Optional,
  Mapping,
  Dict,
  Literal,
  Sequence,
  TypedDict,
  Union,
)
from litellm.types.utils import (
  ModelResponse,
  Choices,
  Message as LiteLLMMessage,
  ChatCompletionMessageToolCall,
  Function as LiteLLMFunction
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from crewai.flow.flow import FlowState
from crewai.utilities.events import crewai_event_bus
from pydantic import BaseModel, Field, TypeAdapter
from ag_ui.core import EventType, Message
from .context import flow_context
from .events import (
  BridgedTextMessageChunkEvent,
  BridgedToolCallChunkEvent,
  BridgedCustomEvent,
  BridgedStateSnapshotEvent
)
from .utils import yield_control

class CopilotKitProperties(BaseModel):
    """CopilotKit properties"""
    actions: List[Any] = Field(default_factory=list)

class CopilotKitState(FlowState):
    """CopilotKit state"""
    messages: List[Any] = Field(default_factory=list)
    copilotkit: CopilotKitProperties = Field(default_factory=CopilotKitProperties)

class PredictStateConfig(TypedDict):
    """
    Predict State Config
    """
    tool_name: str
    tool_argument: Optional[str]


@dataclass(frozen=True)
class StateItem:
    """A single predicted-state binding.

    Mirrors the LangGraph ``StateItem`` (``ag_ui_langgraph`` middleware) so
    the two integrations describe shared-state streaming with the same
    vocabulary: as ``tool``'s ``tool_argument`` streams in, the client
    renders it under ``state_key`` before the tool result lands. When
    ``tool_argument`` is ``None`` the whole tool-call argument object is
    emitted under ``state_key``.
    """
    state_key: str
    tool: str
    tool_argument: Optional[str] = None


# Attributes stashed on the running ``Flow`` object to coordinate node-exit
# STATE_SNAPSHOT suppression between these SDK hooks and the endpoint's
# ``MethodExecutionFinished`` listener. LangGraph performs the equivalent
# suppression inside its own event loop (``ag_ui_langgraph.agent``: the
# ``model_made_tool_call`` / ``manually_emitted_state`` flags). CrewAI has
# no such loop, so the state travels on the flow instance instead. The same
# object is visible via ``flow_context`` here and as ``source`` in the
# listener. Names are underscore-prefixed to avoid colliding with any real
# CrewAI ``Flow`` field.
_PREDICT_STATE_TOOLS_ATTR = "_ag_ui_predict_state_tools"
_PREDICTED_TOOL_STREAMED_ATTR = "_ag_ui_predicted_tool_streamed"
_MANUAL_STATE_EMITTED_ATTR = "_ag_ui_manual_state_emitted"


def _record_predicted_tools(flow: Any, tools: "set[str]") -> None:
    """Remember which tool names predict state for the current node.

    Unions with any tools already recorded for the node so that two
    ``copilotkit_predict_state`` calls in the same node both take effect (the
    second must not drop the first's bindings). The set is reset at node entry
    and consumed at node exit, so it only ever accumulates within one node.
    """
    if flow is not None:
        existing = getattr(flow, _PREDICT_STATE_TOOLS_ATTR, None) or set()
        setattr(flow, _PREDICT_STATE_TOOLS_ATTR, existing | set(tools))


def _mark_predicted_tool_streamed(flow: Any, tool_name: Optional[str]) -> None:
    """Flag that a predicted tool call actually started streaming.

    This is the CrewAI analogue of LangGraph's ``model_made_tool_call``: the
    node-exit snapshot must only be suppressed when the predicted tool is
    genuinely invoked, otherwise a node that declared ``predict_state`` but
    took a different branch would silently drop a legitimate state update.
    """
    if flow is None or not tool_name:
        return
    predicted = getattr(flow, _PREDICT_STATE_TOOLS_ATTR, None)
    if predicted and tool_name in predicted:
        setattr(flow, _PREDICTED_TOOL_STREAMED_ATTR, True)


def _mark_manual_state_emitted(flow: Any) -> None:
    """Flag that ``copilotkit_emit_state`` published an authoritative snapshot."""
    if flow is not None:
        setattr(flow, _MANUAL_STATE_EMITTED_ATTR, True)


def reset_node_snapshot_suppression(flow: Any) -> None:
    """Clear all per-node suppression coordination state.

    Called at node entry (``MethodExecutionStarted``) so a node always starts
    from a clean slate. Without this, a node that calls
    ``copilotkit_predict_state`` and then raises before
    ``MethodExecutionFinished`` fires would leave its predicted-tool set on
    the flow, and the next node could spuriously suppress a legitimate
    snapshot. This is the entry-side counterpart to the exit-side reset in
    ``consume_node_exit_snapshot_suppression``.

    The suppression flags are stored on the single flow instance, which is
    correct for a sequential node chain (router chains, the shipped
    shared-state demos). CrewAI does run listeners triggered by the same
    method concurrently (``asyncio.gather``), so if two parallel branches
    each stream state, these per-flow flags are best-effort and a branch's
    entry-reset can race another's in-flight flag. That only affects the
    mid-run smoothing: the terminal ``STATE_SNAPSHOT`` emitted on
    ``FlowFinished`` always delivers the authoritative final ``flow.state``,
    so a race can at worst cause a transient flicker, never a wrong final
    state. Per-branch coordination is left as future work.
    """
    if flow is None:
        return
    setattr(flow, _PREDICTED_TOOL_STREAMED_ATTR, False)
    setattr(flow, _MANUAL_STATE_EMITTED_ATTR, False)
    setattr(flow, _PREDICT_STATE_TOOLS_ATTR, set())


def consume_node_exit_snapshot_suppression(flow: Any) -> bool:
    """Return whether this node's auto STATE_SNAPSHOT should be suppressed.

    The node-exit snapshot rebuilds state from ``flow.state`` and would wipe
    a prediction the client is already rendering (predicted tool arguments)
    or a manual snapshot whose shape differs from ``flow.state`` (keys that
    live only in the emitted payload). Suppressing it here mirrors
    LangGraph's node-exit suppression. The authoritative full state still
    reaches the client: either a later node emits its own snapshot, or, when
    the last node of the run was the one that suppressed, the terminal
    STATE_SNAPSHOT that ``FlowFinished`` emits in exactly that case (see
    ``endpoint.py``) delivers the true ``flow.state``. That guarantees the
    client ends every run consistent even when a node emitted only a partial
    payload or the predicted node was the last one.

    The per-node flags are cleared on read so the next node starts clean.
    """
    if flow is None:
        return False
    predicted = getattr(flow, _PREDICTED_TOOL_STREAMED_ATTR, False)
    manual = getattr(flow, _MANUAL_STATE_EMITTED_ATTR, False)
    # Reset per-node coordination state so the next node is not affected.
    setattr(flow, _PREDICTED_TOOL_STREAMED_ATTR, False)
    setattr(flow, _MANUAL_STATE_EMITTED_ATTR, False)
    setattr(flow, _PREDICT_STATE_TOOLS_ATTR, set())
    return bool(predicted or manual)


def _normalize_predict_state(
        config: Union[Mapping[str, PredictStateConfig], Sequence[StateItem]]
    ) -> List[Dict[str, Any]]:
    """Normalize either supported ``predict_state`` shape to the wire form.

    Accepts the historical mapping form (``{state_key: {tool_name, tool_argument}}``)
    and a sequence of :class:`StateItem`. ``tool_argument`` is optional in
    both shapes.
    """
    if isinstance(config, Mapping):
        return [
            {
                "state_key": k,
                "tool": v["tool_name"],
                "tool_argument": v.get("tool_argument"),
            }
            for k, v in config.items()
        ]
    return [
        {
            "state_key": item.state_key,
            "tool": item.tool,
            "tool_argument": item.tool_argument,
        }
        for item in config
    ]


async def copilotkit_predict_state(
        config: Union[Mapping[str, PredictStateConfig], Sequence[StateItem]]
    ) -> Literal[True]:
    """
    Stream tool calls as state to CopilotKit.

    To emit a tool call as streaming CrewAI state, pass the destination key in state,
    the tool name and optionally the tool argument. (If you don't pass the argument name,
    all arguments are emitted under the state key.)

    ```python
    from ag_ui_crewai import copilotkit_predict_state

    await copilotkit_predict_state(
        {
            "steps": {
                "tool_name": "SearchTool",
                "tool_argument": "steps",
            },
        }
    )
    ```

    ``copilotkit_predict_state`` must be called inside the same flow node
    that streams the predicted tool call; the prediction binding is scoped to
    that node.

    A sequence of :class:`StateItem` is also accepted, matching the LangGraph
    shared-state vocabulary:

    ```python
    await copilotkit_predict_state([
        StateItem(state_key="steps", tool="SearchTool", tool_argument="steps"),
    ])
    ```

    Parameters
    ----------
    config : Mapping[str, PredictStateConfig] | Sequence[StateItem]
        The configuration to predict the state.

    Returns
    -------
    Awaitable[bool]
        Always return True.
    """
    flow = flow_context.get(None)

    value = _normalize_predict_state(config)

    # Record which tools predict state for this node so the streaming layer
    # can tell when a predicted tool call actually fires (see
    # ``_mark_predicted_tool_streamed``).
    _record_predicted_tools(flow, {item["tool"] for item in value})

    crewai_event_bus.emit(
        flow,
        BridgedCustomEvent(
            type=EventType.CUSTOM,
            name="PredictState",
            value=value
        )
    )

    await yield_control()

    return True

async def copilotkit_emit_state(state: Any) -> Literal[True]:
    """
    Emits intermediate state to CopilotKit.
    Useful if you have a longer running node and you want to update the user with the current state of the node.

    ### Examples

    ```python
    from ag_ui_crewai import copilotkit_emit_state

    for i in range(10):
        await some_long_running_operation(i)
        await copilotkit_emit_state({"progress": i})
    ```

    Each call streams a STATE_SNAPSHOT to the client immediately, and the
    node-exit snapshot rebuilt from ``flow.state`` is suppressed so the emitted
    payload is not clobbered mid-run. Note the end-of-run terminal snapshot is
    rebuilt from ``flow.state``: keys present only in an emitted payload (for
    example a ``{"progress": i}`` indicator not stored on ``flow.state``) are
    shown during the run but replaced by the authoritative ``flow.state`` when
    the run finishes. Write anything that must persist onto ``flow.state``.

    Parameters
    ----------
    state : Any
        The state to emit (Must be JSON serializable).

    Returns
    -------
    Awaitable[bool]
        Always return True.

    """
    flow = flow_context.get(None)

    # This manual snapshot is authoritative for the current node; flag it so
    # the endpoint suppresses the node-exit snapshot rebuilt from flow.state,
    # which would otherwise wipe keys that live only in ``state``.
    _mark_manual_state_emitted(flow)

    # Deep-copy so the snapshot is a point-in-time capture. Callers commonly
    # pass the live flow state and keep mutating it (e.g. a progress loop that
    # emits after each step); without the copy a later mutation could corrupt
    # this already-queued snapshot before it is encoded. Matches the copy
    # discipline in endpoint._flow_state_snapshot. State is JSON-serializable
    # (it is about to be SSE-encoded), so deep-copy is safe.
    crewai_event_bus.emit(
        flow,
        BridgedStateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=copy.deepcopy(state)
        )
    )

    await yield_control()

    return True

async def copilotkit_stream(response):
    """
    Stream litellm responses token by token to CopilotKit.

    ```python
    response = await copilotkit_stream(
        await acompletion(
            model="openai/gpt-4o",
            messages=messages,
            tools=tools,
            stream=True # this must be set to True for streaming
        )
    )
    ```
    """
    if isinstance(response, ModelResponse):
        return _copilotkit_stream_response(response)
    if isinstance(response, CustomStreamWrapper):
        return await _copilotkit_stream_custom_stream_wrapper(response)
    raise ValueError("Invalid response type")


async def _copilotkit_stream_custom_stream_wrapper(response: CustomStreamWrapper):
    flow = flow_context.get(None)

    message_id: Optional[str] = None
    tool_call_id: str = ""
    content = ""
    created = 0
    model = ""
    system_fingerprint = ""
    finish_reason=None
    all_tool_calls = []

    async for chunk in response:
        if message_id is None:
            message_id = chunk["id"]

        text_content = chunk["choices"][0]["delta"]["content"] or None

        # Stream text messages
        if text_content is not None:
            # add to the current text message
            content += text_content
            crewai_event_bus.emit(
                flow,
                BridgedTextMessageChunkEvent(
                    type=EventType.TEXT_MESSAGE_CHUNK,
                    message_id=message_id,
                    role="assistant",
                    delta=text_content,
                )
            )
            # yield control to the event loop
            await yield_control()

        # Stream tool calls
        tool_calls = chunk["choices"][0]["delta"]["tool_calls"] or None
        tool_call_id = tool_calls[0].id if tool_calls is not None else None
        tool_call_arguments = tool_calls[0].function["arguments"] if tool_calls is not None else None
        tool_call_name = tool_calls[0].function["name"] if tool_calls is not None else None

        if tool_call_id is not None:
            all_tool_calls.append(
                {
                    "id": tool_call_id,
                    "name": tool_call_name,
                    "arguments": "",
                }
            )

        # If this tool was registered via copilotkit_predict_state, the client
        # is already rendering its streamed arguments as state. Flag it so the
        # node-exit snapshot is suppressed (parity with LangGraph's
        # model_made_tool_call). The name is checked on whichever chunk carries
        # it, not only the id-bearing chunk, since some providers stream the
        # tool id and function name in separate deltas.
        if tool_call_name is not None:
            _mark_predicted_tool_streamed(flow, tool_call_name)

        if tool_call_arguments is not None:
            # add to the current tool call
            all_tool_calls[-1]["arguments"] += tool_call_arguments
            crewai_event_bus.emit(
                flow,
                BridgedToolCallChunkEvent(
                    type=EventType.TOOL_CALL_CHUNK,
                    tool_call_id=tool_call_id,
                    tool_call_name=tool_call_name,
                    delta=tool_call_arguments,
                )
            )
            # yield control to the event loop
            await yield_control()

        # Stream finish reason
        finish_reason = chunk["choices"][0]["finish_reason"]
        created = chunk["created"]
        model = chunk["model"]
        system_fingerprint = chunk["system_fingerprint"]

        if finish_reason is not None:
            break

    tool_calls = [
        ChatCompletionMessageToolCall(
            function=LiteLLMFunction(
                arguments=tool_call["arguments"],
                name=tool_call["name"]
            ),
            id=tool_call["id"],
            type="function"
        )
        for tool_call in all_tool_calls
    ]
    return ModelResponse(
        id=message_id,
        created=created,
        model=model,
        object='chat.completion',
        system_fingerprint=system_fingerprint,
        choices=[
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=LiteLLMMessage(
                    content=content,
                    role='assistant',
                    tool_calls=tool_calls if len(tool_calls) > 0 else None,
                    function_call=None
                )
            )
        ]
    )

def _copilotkit_stream_response(response: ModelResponse):
    return response


message_adapter = TypeAdapter(Message)

def litellm_messages_to_ag_ui_messages(messages: List[LiteLLMMessage]) -> List[Message]:
    """
    Converts a list of LiteLLM messages to a list of ag_ui messages.
    """
    ag_ui_messages: List[Message] = []
    for message in messages:
        message_dict = message.model_dump() if not isinstance(message, Mapping) else message

        # whitelist the fields we want to keep
        whitelist = ["content", "role", "tool_calls", "id", "name", "tool_call_id"]
        message_dict = {k: v for k, v in message_dict.items() if k in whitelist}
        if not "id" in message_dict:
            message_dict["id"] = str(uuid.uuid4())
        # remove all None values
        message_dict = {k: v for k, v in message_dict.items() if v is not None}

        if "tool_calls" in message_dict:
            for tool_call in message_dict["tool_calls"]:
                if "type" not in tool_call:
                    tool_call["type"] = "function"

        ag_ui_message = message_adapter.validate_python(message_dict)
        ag_ui_messages.append(ag_ui_message)

    return ag_ui_messages


async def copilotkit_exit() -> Literal[True]:
    """
    Exits the current agent after the run completes. Calling copilotkit_exit() will
    not immediately stop the agent. Instead, it signals to CopilotKit to stop the agent after
    the run completes.

    ### Examples

    ```python
    from copilotkit.crewai import copilotkit_exit

    def my_function():
        await copilotkit_exit()
        return state
    ```

    Returns
    -------
    Awaitable[bool]
        Always return True.
    """

    flow = flow_context.get(None)

    crewai_event_bus.emit(
        flow,
        BridgedCustomEvent(
            type=EventType.CUSTOM,
            name="Exit",
            value=""
        )
    )

    await yield_control()

    return True