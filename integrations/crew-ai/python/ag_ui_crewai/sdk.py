"""
Streaming and state helpers (copilotkit_stream and related utilities) for the CrewAI AG-UI bridge.
"""

import copy
import uuid
from typing import List, Any, Optional, Mapping, Dict, Literal, TypedDict
from litellm.types.utils import (
  ModelResponse,
  Choices,
  Message as LiteLLMMessage,
  ChatCompletionMessageToolCall,
  Function as LiteLLMFunction
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from crewai.flow.flow import FlowState
# CPK-7718: the event bus moved from ``crewai.utilities.events`` (0.x) to
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

async def copilotkit_predict_state(
        config: Dict[str, PredictStateConfig]
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

    Parameters
    ----------
    config : Dict[str, CopilotKitPredictStateConfig]
        The configuration to predict the state.

    Returns
    -------
    Awaitable[bool]
        Always return True.
    """
    flow = flow_context.get(None)

    value = [
        {
            "state_key": k,
            "tool": v["tool_name"],
            "tool_argument": v["tool_argument"]
        } for k, v in config.items()
    ]
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

    To install the CopilotKit SDK, run:

    ```bash
    pip install copilotkit[crewai]
    ```

    ### Examples

    ```python
    from ag_ui_crewai import copilotkit_emit_state

    for i in range(10):
        await some_long_running_operation(i)
        await copilotkit_emit_state({"progress": i})
    ```

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
    crewai_event_bus.emit(
        flow,
        BridgedStateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=state
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
    content = ""
    created = 0
    model = ""
    system_fingerprint = ""
    finish_reason=None
    # Route tool-call deltas by their OpenAI ``.index`` so parallel calls stay
    # separate; keyed in arrival order so the final reassembly preserves it.
    # A provider that omits ``.index`` falls back to last-call routing below.
    tool_calls_by_index: Dict[Any, Dict[str, Any]] = {}
    last_tool_key: Any = None
    auto_key = 0

    async for chunk in response:
        if message_id is None:
            message_id = chunk["id"]

        # Some providers send trailing/usage-only chunks with an empty
        # ``choices`` list; skip them rather than IndexError on ``choices[0]``.
        choices = chunk["choices"] or None
        if choices is None:
            continue
        choice = choices[0]
        delta = choice["delta"]

        text_content = delta["content"] or None

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

        # Stream tool calls (index-routed, one bridged chunk per arg delta)
        tool_calls = delta["tool_calls"] or None
        if tool_calls is not None:
            for tool_call in tool_calls:
                delta_id = getattr(tool_call, "id", None)
                delta_name = tool_call.function["name"]
                delta_arguments = tool_call.function["arguments"]

                # Resolve which accumulating call this delta belongs to.
                index = getattr(tool_call, "index", None)
                last_entry = tool_calls_by_index.get(last_tool_key)
                if index is not None:
                    key = index
                elif delta_id is not None and last_entry is not None \
                        and last_entry.get("id") == delta_id:
                    # No index, but this delta re-echoes the current call's id:
                    # a continuation, not a new call.
                    key = last_tool_key
                elif delta_id is not None and (
                    last_entry is None or last_entry.get("id") is not None
                ):
                    # No index, a new id: a genuinely new call.
                    key = ("auto", auto_key)
                    auto_key += 1
                else:
                    # No index, no new id: continue the current call. Covers
                    # argument-only deltas and the id-bearing delta of a call
                    # whose args streamed first (empty accumulator, no IndexError).
                    key = last_tool_key
                last_tool_key = key

                entry = tool_calls_by_index.get(key)
                if entry is None:
                    entry = {"id": delta_id, "name": delta_name, "arguments": ""}
                    tool_calls_by_index[key] = entry
                else:
                    # id/name can arrive on a later delta than the first.
                    if delta_id is not None:
                        entry["id"] = delta_id
                    if delta_name is not None:
                        entry["name"] = delta_name

                if delta_arguments is not None:
                    entry["arguments"] += delta_arguments
                    crewai_event_bus.emit(
                        flow,
                        BridgedToolCallChunkEvent(
                            type=EventType.TOOL_CALL_CHUNK,
                            tool_call_id=entry["id"],
                            tool_call_name=entry["name"],
                            delta=delta_arguments,
                        )
                    )
                    # yield control to the event loop
                    await yield_control()

        # Stream finish reason
        finish_reason = choice["finish_reason"]
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
        for tool_call in tool_calls_by_index.values()
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
            # The whitelist comprehension is a shallow copy, so this list and
            # its dicts are still the caller's (e.g. the flow-state) objects.
            # Deep-copy before stamping ``type`` so we don't mutate them in place.
            message_dict["tool_calls"] = copy.deepcopy(message_dict["tool_calls"])
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