"""Shared pytest fixtures for the ag_ui_crewai tests.

Primary concern: isolate the module-level ``QUEUES`` mapping (and the global
crewai event-bus listener singleton) from test-to-test leakage. A ghost
queue from one test is harmless in isolation, but in a long test suite it
can obscure the provenance of flaky teardown races.
"""

import pytest


@pytest.fixture(autouse=True)
def _clear_endpoint_queues():
    """Ensure the module-level QUEUES dict is empty between tests.

    We import lazily — some tests may patch the module before this fixture
    runs, and we do not want to force an import at collection time.
    """
    try:
        from ag_ui_crewai import endpoint as ep
    except Exception:  # pragma: no cover - import-time failures are tested elsewhere
        yield
        return

    ep.QUEUES.clear()
    try:
        yield
    finally:
        ep.QUEUES.clear()
