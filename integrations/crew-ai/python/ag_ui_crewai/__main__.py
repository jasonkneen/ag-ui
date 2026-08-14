"""Launcher for the development-only dojo server: ``python -m ag_ui_crewai``.

Owns the ``uvicorn.run`` call and deliberately never imports
``ag_ui_crewai.dojo``: with ``reload=True`` this process is only the file-watching
supervisor, and uvicorn hands its worker the ``"ag_ui_crewai.dojo:app"`` import
string, so the FastAPI app and every demo flow are built exactly once, in the
process that serves traffic. Importing the app here would build a second copy
that nothing uses.

The ``-m`` target has to be the package, never ``ag_ui_crewai.dojo``: the
reloader spawns its worker through multiprocessing, which re-executes a module
``-m`` target as ``__mp_main__`` but skips that re-execution when the target is a
package ``__main__``.

Excluded from the published wheel and sdist alongside ``dojo.py``.
"""

import os
import sys

import uvicorn


def main() -> int:
    """Serve the dojo, returning the process exit code.

    Under ``reload=True`` uvicorn never exits the process itself: it swallows
    Ctrl-C and returns once the reload supervisor stops. So a normal return is a
    clean shutdown, and 0 is the only status there is to report.
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
