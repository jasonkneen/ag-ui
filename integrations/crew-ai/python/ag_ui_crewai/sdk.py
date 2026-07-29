"""
Streaming and state helpers (copilotkit_stream and related utilities) for the CrewAI AG-UI bridge.
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
# The event bus moved from ``crewai.utilities.events`` (0.x) to
# ``crewai.events`` (1.x); ``_capabilities`` resolves whichever exists.
from ._capabilities import crewai_event_bus
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
    """A predicted-state binding: stream ``tool``'s ``tool_argument`` into ``state_key``.

    ``tool_argument=None`` streams the whole tool-call argument object. Mirrors
    the LangGraph ``StateItem`` vocabulary.
    """
    state_key: str
    tool: str
    tool_argument: Optional[str] = None


# Per-node suppression flags stashed on the running Flow (visible via
# flow_context in the hooks and as ``source`` in the endpoint listener).
# Underscore-prefixed to avoid clashing with real Flow fields.
_PREDICT_STATE_TOOLS_ATTR = "_ag_ui_predict_state_tools"
_PREDICTED_TOOL_STREAMED_ATTR = "_ag_ui_predicted_tool_streamed"
_MANUAL_STATE_EMITTED_ATTR = "_ag_ui_manual_state_emitted"


def _record_predicted_tools(flow: Any, tools: "set[str]") -> None:
    """Record the tools that predict state for this node.

    Unions so two predict_state calls in one node both take effect.
    """
    if flow is not None:
        existing = getattr(flow, _PREDICT_STATE_TOOLS_ATTR, None) or set()
        setattr(flow, _PREDICT_STATE_TOOLS_ATTR, existing | set(tools))


def _mark_predicted_tool_streamed(flow: Any, tool_name: Optional[str]) -> None:
    """Flag that a predicted tool actually streamed.

    Suppression only fires when the tool is genuinely invoked, so a node that
    declared predict_state but took another branch still emits its snapshot.
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
    """Clear the per-node suppression flags at node entry.

    Guards against a node that declared predict_state then raised (leaving a
    stale tool set) suppressing the next node's snapshot. Flags live on the
    single flow, so under parallel fan-out branches they are best-effort; the
    terminal FlowFinished snapshot still guarantees a correct final state.
    """
    if flow is None:
        return
    setattr(flow, _PREDICTED_TOOL_STREAMED_ATTR, False)
    setattr(flow, _MANUAL_STATE_EMITTED_ATTR, False)
    setattr(flow, _PREDICT_STATE_TOOLS_ATTR, set())


def consume_node_exit_snapshot_suppression(flow: Any) -> bool:
    """Whether this node's auto STATE_SNAPSHOT should be suppressed; resets the flags.

    Suppressed when a predicted tool streamed or a manual snapshot was emitted,
    so the node-exit rebuild from flow.state doesn't wipe what the client is
    already showing. A later node's snapshot, or the terminal FlowFinished
    snapshot, still delivers the authoritative flow.state.
    """
    if flow is None:
        return False
    predicted = getattr(flow, _PREDICTED_TOOL_STREAMED_ATTR, False)
    manual = getattr(flow, _MANUAL_STATE_EMITTED_ATTR, False)
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

    # So the streaming layer can tell when a predicted tool actually fires.
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

    The emitted payload streams to the client immediately and the node-exit
    snapshot is suppressed so it is not clobbered mid-run. At run end the state
    is rebuilt from ``flow.state``, so anything that must persist beyond the run
    should be written there, not only emitted.

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

    # Suppress the node-exit snapshot so this payload is not clobbered mid-run.
    _mark_manual_state_emitted(flow)

    # Deep-copy: callers often emit the live flow state and keep mutating it, so
    # snapshot a point-in-time copy before it is queued.
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

        # Checked on whichever chunk carries the name (some providers stream the
        # tool id and name in separate deltas), not only the id-bearing chunk.
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
        if "id" not in message_dict:
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
    from ag_ui_crewai.sdk import copilotkit_exit

    async def my_function():
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