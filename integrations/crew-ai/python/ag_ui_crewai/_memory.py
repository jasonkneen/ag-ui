"""Per-thread isolation of CrewAI crew memory.

The bug. A crew served with ``Crew(memory=True)`` leaks remembered
facts between chats. The bridge isolates requests by copying the flow, but that
copy does not touch memory: crewai 1.x builds ONE ``Memory`` over ONE on-disk
store, namespaced by a ``root_scope`` string that ``Crew.create_crew_memory``
derives from the CREW NAME. Every request to a given endpoint therefore reads
and writes the same namespace. Passing ``inputs["id"] = thread_id`` does not
help; that scopes crewai's flow-state persistence, a different subsystem.

The fix. Before the run starts, give THIS request's flow copy a crew whose
memory is a ``MemoryScope`` view rooted at a path derived from the AG-UI
``threadId``. Reads and writes through that view are confined to the thread's
namespace and below, so thread A and thread B are mutually invisible while each
still sees its own history across sequential runs. One physical store, no
directory sprawl, and ``Crew._memory`` is typed
``Memory | MemoryScope | MemorySlice`` upstream, so a view is a supported thing
to hand a crew rather than a hack.

Two constraints shape the implementation:

* **The template crew is shared across concurrent requests.** ``ChatWithCrewFlow``
  is built once per endpoint and cached, and ``_copyutil.safe_deepcopy`` pins the
  uncopyable crew BY REFERENCE, so every in-flight request holds the same live
  ``Crew`` and the same live ``Memory``. Re-pointing that shared object's memory
  per request would be a data race. So nothing shared is ever mutated: we build a
  per-request shallow crew VIEW, point only the view's ``_memory`` at the thread
  scope, and swap the view onto the per-request flow copy.
* **There are two run paths.** The legacy ``kickoff_async`` driver and the
  ``astream`` StreamFrame driver both read the crew off the flow copy, so scoping
  the flow copy before either driver is selected covers both.

Capability-detected, never version-gated: a crewai build without the unified
``Memory.scope`` view API degrades to "isolation not active" plus one warning
rather than crashing.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import re
from typing import Any

# ``_Crew`` / ``sanitize_scope_name`` are resolved in ``_capabilities`` with every
# other crewai symbol the bridge depends on, rather than re-derived here with a
# parallel ``getattr`` chain.
from ._capabilities import (
    CAPABILITIES,
    _Crew,
    sanitize_scope_name as _crewai_sanitize_scope_name,
)
from ._config import resolve_thread_scoped_memory

_LOGGER = logging.getLogger(__name__)

# Namespace the per-thread scopes live under, relative to whatever root scope the
# crew already carries. A crew named "support" therefore stores thread A's
# memories under ``/crew/support/thread/<segment>`` and keeps crewai's own
# hierarchy intact above it.
_THREAD_SCOPE_ROOT = "/thread"

# Upper bound on the readable part of a scope segment. Thread ids are usually
# UUIDs, but nothing stops a client sending a very long one, and the segment ends
# up in a stored scope path.
_MAX_READABLE_SEGMENT = 64

# Length of the digest suffix, in hex characters. 16 hex chars is 64 bits, which
# makes an accidental collision between two live threads unreachable in practice.
_DIGEST_CHARS = 16

# Emitted at most once per process: an operator whose crewai lacks the view API
# needs to hear that isolation is off, but not once per request.
_DEGRADE_WARNED = False


def _fallback_sanitize_scope_name(name: str) -> str:
    """Stand-in for ``crewai.memory.utils.sanitize_scope_name``.

    Same normalisation (lowercase, non ``[a-z0-9_-]`` to a hyphen, collapse and
    strip hyphens) so a build that does not expose crewai's helper still produces
    a segment shaped like the crew-name segment above it.
    """
    if not name:
        return "unknown"
    name = re.sub(r"[^a-z0-9_-]", "-", name.lower().strip())
    name = re.sub(r"-+", "-", name).strip("-")
    return name or "unknown"


def thread_scope_path(thread_id: str) -> str:
    """Return the scope path that isolates ``thread_id``.

    Shape: ``/thread/<readable>-<digest>``. The readable half is the sanitized
    thread id, so a stored scope tree stays diagnosable by eye. The digest half
    is what actually guarantees isolation: sanitisation is lossy (``a/b`` and
    ``a-b`` both normalise to ``a-b``, and a long id is truncated), so two
    distinct threads could otherwise collapse onto one namespace and leak into
    each other, which is the exact bug this module exists to prevent. The digest
    is taken over the RAW id and is therefore injective in practice.
    """
    raw = str(thread_id)
    sanitize = _crewai_sanitize_scope_name or _fallback_sanitize_scope_name
    try:
        readable = sanitize(raw)
    except Exception:  # noqa: BLE001 - a surprising crewai helper must not fail the run
        readable = _fallback_sanitize_scope_name(raw)
    readable = readable[:_MAX_READABLE_SEGMENT].strip("-") or "unknown"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return f"{_THREAD_SCOPE_ROOT}/{readable}-{digest}"


def _warn_isolation_unavailable(memory: Any) -> None:
    """Say once, loudly, that this deployment is running without isolation.

    Two causes worth distinguishing: the installed crewai has no unified memory
    view API at all, or it does but this particular crew was handed a memory
    object that is not one of crewai's view types. The remedy differs.
    """
    global _DEGRADE_WARNED  # pylint: disable=global-statement
    if _DEGRADE_WARNED:
        return
    _DEGRADE_WARNED = True
    if not CAPABILITIES.memory_scope_available:
        cause = (
            f"crewai {CAPABILITIES.crewai_version} does not expose the unified "
            "memory view API (crewai.memory.unified_memory.Memory.scope)"
        )
        remedy = "Upgrade crewai, or disable crew memory"
    else:
        cause = (
            f"this crew's memory is a {type(memory).__name__}, which has no "
            "scope() view factory"
        )
        remedy = (
            "Pass the crew a crewai Memory (or an already-scoped view), or "
            "disable crew memory"
        )
    _LOGGER.warning(
        "ag-ui-crewai: crew memory is enabled but %s, so PER-THREAD MEMORY "
        "ISOLATION IS NOT ACTIVE: every AG-UI threadId served by this endpoint "
        "shares one memory namespace and can read another chat's remembered "
        "facts. %s. Further occurrences are silenced.",
        cause,
        remedy,
    )


def _attribute_containers(obj: Any) -> list[dict]:
    """Return the mutable attribute stores of ``obj`` worth scanning.

    Pydantic keeps declared fields in ``__dict__`` and private attributes in
    ``__pydantic_private__``; a crewai ``Flow`` holds an assigned ``self.crew`` in
    the former. Writing back through these dicts (rather than ``setattr``) is the
    same approach ``_copyutil.rebind_bound_methods`` takes, and for the same
    reason: ``BaseModel.__setattr__`` refuses or reroutes assignments that are not
    declared fields.
    """
    containers = []
    for candidate in (
        getattr(obj, "__dict__", None),
        getattr(obj, "__pydantic_private__", None),
    ):
        if isinstance(candidate, dict):
            containers.append(candidate)
    return containers


def _thread_scoped_crew(crew: Any, scope_path: str) -> Any | None:
    """Return a shallow view of ``crew`` whose memory is scoped to ``scope_path``.

    ``None`` means "leave this crew alone": it has no active memory, or this
    crewai build cannot produce a view. The passed crew is SHARED with every
    concurrent request and is never mutated -- ``copy.copy`` of a Pydantic model
    builds fresh ``__dict__`` / ``__pydantic_private__`` mappings, so pointing the
    view's ``_memory`` elsewhere leaves the original's alone. Everything else
    (agents, tasks, the underlying store, the save pool) stays shared, exactly as
    it already was before this module existed.
    """
    memory = getattr(crew, "_memory", None)
    if memory is None:
        return None

    scope_factory = getattr(memory, "scope", None)
    if not callable(scope_factory):
        _warn_isolation_unavailable(memory)
        return None

    try:
        scoped_memory = scope_factory(scope_path)
    except Exception as exc:  # noqa: BLE001 - never fail a run over memory scoping
        _LOGGER.warning(
            "ag-ui-crewai: could not scope crew memory to %s (%s: %s); this run "
            "shares the crew-wide memory namespace with other threads.",
            scope_path,
            type(exc).__name__,
            exc,
        )
        return None

    crew_view = copy.copy(crew)
    private = getattr(crew_view, "__pydantic_private__", None)
    if isinstance(private, dict) and "_memory" in private:
        private["_memory"] = scoped_memory
    else:
        setattr(crew_view, "_memory", scoped_memory)

    if getattr(crew, "_memory", None) is not memory:
        # The shared template crew was re-pointed, which would hand every other
        # in-flight request THIS thread's namespace. Refuse rather than serve a
        # cross-thread leak under the name of a fix.
        raise RuntimeError(
            "ag-ui-crewai: building a per-thread memory view mutated the SHARED "
            "crew instead of the per-request copy. Refusing to continue; "
            "concurrent requests would read one another's memory."
        )
    return crew_view


def apply_thread_memory_scope(flow_copy: Any, thread_id: str | None) -> None:
    """Scope every crew reachable on ``flow_copy`` to ``thread_id``'s namespace.

    Called once per request, on the PER-REQUEST flow copy, before a run driver is
    selected -- so both the legacy ``kickoff_async`` path and the ``astream``
    StreamFrame path are covered.

    Scans the flow copy's own attributes for ``Crew`` instances. That reaches the
    crew-serving endpoint's ``ChatWithCrewFlow.crew`` and any crew a user's
    ``Flow`` holds as an attribute. A crew CONSTRUCTED INSIDE a flow method is not
    reachable from here and is not scoped; see the README for that limitation.

    Never raises for an environment reason: an unscopable crew degrades to the
    previous shared-namespace behaviour with a warning, because failing a chat
    outright is worse than the leak it would prevent.
    """
    if not resolve_thread_scoped_memory():
        return
    if not thread_id:
        # Nothing to derive a namespace from. Scoping every such request onto one
        # shared "unknown" namespace would isolate nothing while still hiding the
        # crew's own history, so leave the crew as-is.
        return
    if _Crew is None:  # pragma: no cover - crewai is a hard dependency
        return

    scope_path = thread_scope_path(thread_id)
    for container in _attribute_containers(flow_copy):
        for name, value in list(container.items()):
            if not isinstance(value, _Crew):
                continue
            crew_view = _thread_scoped_crew(value, scope_path)
            if crew_view is not None:
                container[name] = crew_view
