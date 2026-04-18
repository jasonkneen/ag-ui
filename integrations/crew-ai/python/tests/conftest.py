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


@pytest.fixture(autouse=True)
def _clear_endpoint_queues():
    """Ensure the module-level QUEUES dict and listener singleton are
    isolated between tests.

    The crewai global event bus retains registered listeners for the
    lifetime of the process; the endpoint module caches its listener in
    ``GLOBAL_EVENT_LISTENER`` to avoid double-registration. Between
    tests we clear the QUEUES dict and reset the listener reference so
    a test that patches or probes ``GLOBAL_EVENT_LISTENER`` starts from
    a known-clean baseline (finding #22).
    """

    ep.QUEUES.clear()
    # Reset singleton; the next test that calls ``add_crewai_*`` will
    # create a fresh FastAPICrewFlowEventListener. This does not
    # unregister listeners on the crewai event bus — crewai does not
    # expose a public API for that — but it does prevent cross-test
    # observation of a stale singleton reference.
    ep.GLOBAL_EVENT_LISTENER = None
    try:
        yield
    finally:
        ep.QUEUES.clear()
        ep.GLOBAL_EVENT_LISTENER = None
