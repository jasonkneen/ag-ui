"""Launcher for the development-only dojo server: ``python -m ag_ui_crewai``.

Hands uvicorn the ``"ag_ui_crewai.dojo:app"`` import string and does not import the
dojo here, so the app is built once, in the reloader's worker that serves traffic.
``tests/test_launcher.py`` pins both halves of that and the ``PORT`` read below.
Why the ``-m`` target is the package, and why none of this ships: pyproject.toml.
"""

import os
import sys

import uvicorn


def main() -> int:
    """Serve the dojo, returning 0; ``tests/test_launcher.py`` pins that.

    Not every outcome comes back here: a startup failure such as a bound port exits
    non-zero from inside ``uvicorn.run``, and the reloader does not check on its
    worker, so a worker that dies at import leaves this process up with the port
    bound and requests hanging.
    """
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "ag_ui_crewai.dojo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
