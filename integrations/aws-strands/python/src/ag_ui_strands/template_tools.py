"""Per-request filtering of the tools the template agent contributed.

The adapter builds one Strands ``Agent`` per thread and keeps it. That instance
is load-bearing: it holds the thread's ``SessionManager``, its native interrupt
checkpoint and its conversation history. Changing which tools a request sees
therefore has to be done to the registry the live instance already owns, the
way client-declared tools are already synchronised, and never by constructing a
replacement.

Scope is the template's own tools. Client-declared tools arrive on
``RunAgentInput.tools`` every request and are synchronised by
:func:`~ag_ui_strands.client_proxy_tool.sync_proxy_tools`; a caller that wants
fewer of those sends fewer. Auto-injected A2UI tools are the adapter's and are
refreshed per turn. What no per-request channel reached until now is the set the
wrapped template contributed once, at construction.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional, Sequence, Set

from strands.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def index_template_tools(template_tools: Sequence[Any]) -> dict[str, Any]:
    """Index the template's tools by the name a registry holds them under."""
    indexed: dict[str, Any] = {}
    for tool in template_tools:
        name = getattr(tool, "tool_name", None)
        if isinstance(name, str) and name:
            indexed[name] = tool
    return indexed


def resolve_template_tool_selection(
    selection: Optional[Iterable[Any]],
    template_index: Mapping[str, Any],
) -> Optional[Set[str]]:
    """Read one provider answer as the template tool names a request may see.

    Entries are either the template's own tool objects or their names, so a
    caller can write the filter with whichever it has to hand.

    ``None`` means the provider declined to filter this request and every
    template tool stays available. An empty iterable is a real answer and means
    none of them do.

    A name the template never contributed is dropped with a warning. This hook
    narrows what the wrapped agent already gave the adapter; it cannot hand the
    model a capability the template did not carry, so honouring an unknown name
    is the one thing it must not do.
    """
    if selection is None:
        return None

    allowed: Set[str] = set()
    for entry in selection:
        name = entry if isinstance(entry, str) else getattr(entry, "tool_name", None)
        if not isinstance(name, str) or not name:
            logger.warning(
                "template_tools_provider returned an entry that names no tool: %r",
                entry,
            )
            continue
        if name not in template_index:
            logger.warning(
                "template_tools_provider named %r, which the template agent does "
                "not contribute; it stays unavailable. This hook filters the "
                "template's tools and cannot add one.",
                name,
            )
            continue
        allowed.add(name)
    return allowed


def parked_batch_tool_names(agent: Any) -> Set[str]:
    """Tool names in the batch a live interrupt checkpoint would resume.

    A parked run resumes into the tool batch it stopped inside: Strands
    re-dispatches every ``toolUse`` in the assistant message it checkpointed,
    answering the ones that already completed from the checkpoint and running
    the one that is waiting. A tool absent from the registry at that moment
    turns the human's answer into a "tool not found" the model then re-fires,
    so nothing in that batch is filtered out while the pause is open.

    This is the same rule ``sync_proxy_tools`` applies through ``exempt_names``
    to a proxy parked in a frontend-tool interrupt, read off the checkpoint
    instead of off the frontend-wait index because a template tool can park
    through the approval hook, through an interrupt of its own, or not at all,
    and the batch answers all three at once.
    """
    state = getattr(agent, "_interrupt_state", None)
    if state is None or getattr(state, "activated", False) is not True:
        return set()
    context = getattr(state, "context", None)
    if not isinstance(context, Mapping):
        return set()
    message = context.get("tool_use_message")
    if not isinstance(message, Mapping):
        return set()

    names: Set[str] = set()
    for block in message.get("content") or []:
        tool_use = block.get("toolUse") if isinstance(block, Mapping) else None
        if not isinstance(tool_use, Mapping):
            continue
        name = tool_use.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def sync_template_tools(
    tool_registry: ToolRegistry,
    template_tools: Sequence[Any],
    selection: Optional[Iterable[Any]],
    *,
    exempt_names: Set[str] | None = None,
) -> Set[str]:
    """Make *tool_registry* hold exactly the template tools *selection* allows.

    Only the template's own tools are touched, and only by identity: an entry
    some other producer owns under a template tool's name is left alone rather
    than removed, so a client proxy or an auto-injected A2UI tool cannot be
    dropped by a filter aimed at the template.

    Removal is not destructive. The template tool objects outlive the registry
    entry, so a later request that allows a name again restores the same
    instance, and history stays untouched throughout: a filtered-out tool's
    earlier calls and results remain in the thread's messages, which is what
    lets the model read what it already did with a tool it can no longer call.

    Returns the template tool names the registry holds after the call.
    """
    template_index = index_template_tools(template_tools)
    allowed = resolve_template_tool_selection(selection, template_index)
    exempt = exempt_names or set()

    registered: Set[str] = set()
    for name, tool in template_index.items():
        keep = allowed is None or name in allowed or name in exempt
        existing = tool_registry.registry.get(name)
        if keep:
            if existing is tool:
                registered.add(name)
            elif existing is None:
                # Restored by assignment rather than through
                # ``register_tool``, which raises over a name that normalizes
                # onto an existing one. Nothing new is being registered here:
                # the entry is the one the template already put in this
                # registry, so validating it again could only fail a run over a
                # collision the construction it came from had accepted.
                tool_registry.registry[name] = tool
                if getattr(tool, "is_dynamic", False):
                    tool_registry.dynamic_tools[name] = tool
                registered.add(name)
                logger.debug("Restored template tool: %s", name)
            else:
                # Something else answers to this name now. Overwriting it would
                # make a filter that allows a tool destroy another producer's.
                logger.debug(
                    "Template tool %s is shadowed by another registered tool; "
                    "leaving it in place",
                    name,
                )
            continue
        if existing is tool:
            del tool_registry.registry[name]
            tool_registry.dynamic_tools.pop(name, None)
            logger.debug("Filtered out template tool: %s", name)
    return registered
