"""Shared pytest fixtures for the ag_ui_crewai tests.

Primary concern: isolate the module-level ``QUEUES`` mapping (and the
global crewai event-bus listener singleton) from test-to-test leakage. A
ghost queue from one test is harmless in isolation, but in a long test
suite it can obscure the provenance of flaky teardown races.

Intentionally we do NOT swallow the import error (finding #27). If
``ag_ui_crewai.endpoint`` cannot be imported, every downstream test will
fail with the same traceback — a clearer diagnostic than a confused test
suite running against a half-initialised module.
"""

import pytest

from ag_ui_crewai import endpoint as ep

try:
    # The crewai global event bus — used below to clear handlers
    # registered by our listener singleton so they don't accumulate
    # across tests (R5 MEDIUM #10).
    from crewai.utilities.events import crewai_event_bus as _crewai_event_bus
except Exception:  # pragma: no cover - import-time failure is a real bug
    _crewai_event_bus = None


@pytest.fixture(autouse=True)
def _clear_endpoint_queues():
    """Ensure the module-level QUEUES dict and listener singleton are
    isolated between tests.

    The crewai global event bus retains registered listeners for the
    lifetime of the process; the endpoint module caches its listener in
    ``GLOBAL_EVENT_LISTENER`` to avoid double-registration. Between
    tests we clear the QUEUES dict, clear the event-bus handlers
    registered by the listener (R5 MEDIUM #10 — listeners accumulate
    otherwise, and previously the only isolation was nulling the
    reference which let older handlers keep firing), and reset the
    listener reference so a test that patches or probes
    ``GLOBAL_EVENT_LISTENER`` starts from a known-clean baseline
    (finding #22).

    R5 MEDIUM #10 details: the crewai ``CrewAIEventsBus`` exposes a
    private ``_handlers`` dict keyed by event type. Nulling
    ``GLOBAL_EVENT_LISTENER`` only drops our Python-side reference —
    the handlers it registered on the bus persist for the process
    lifetime. Over a long suite this accumulates duplicate listeners
    that all enqueue onto ``QUEUES`` (now empty, so the ``None`` guard
    saves us), but the CPU cost and the signal confusion grow with
    suite length. Reaching into ``_handlers`` directly is a pragmatic
    workaround — crewai does not expose a public teardown API.
    """

    def _reset_bus_handlers():
        if _crewai_event_bus is None:
            return
        handlers = getattr(_crewai_event_bus, "_handlers", None)
        if handlers is None:
            return
        try:
            handlers.clear()
        except Exception:  # pragma: no cover - defensive
            # Unexpected handler-store shape; skip rather than crash.
            pass

    ep.QUEUES.clear()
    # Reset singleton; the next test that calls ``add_crewai_*`` will
    # create a fresh FastAPICrewFlowEventListener. Also clear
    # accumulated handlers on the crewai event bus so stale listeners
    # from prior tests don't keep firing and skewing queue counts
    # (R5 MEDIUM #10).
    ep.GLOBAL_EVENT_LISTENER = None
    _reset_bus_handlers()
    try:
        yield
    finally:
        ep.QUEUES.clear()
        ep.GLOBAL_EVENT_LISTENER = None
        _reset_bus_handlers()
