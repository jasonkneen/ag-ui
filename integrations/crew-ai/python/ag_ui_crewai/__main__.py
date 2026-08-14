"""Launcher for the dojo example server: ``python -m ag_ui_crewai``.

Deliberately trivial. Pointing ``-m`` straight at ``dojo`` made the process that
actually serves traffic build the FastAPI app twice, because uvicorn's reloader
starts its worker with multiprocessing spawn: the worker re-executes the ``-m``
target as ``__mp_main__`` and then imports ``ag_ui_crewai.dojo:app`` separately.
multiprocessing skips that re-execution when the ``-m`` target is a package
``__main__``, so routing through this module leaves exactly one copy of the app
in the worker, matching what the old ``dev`` console script did.

Excluded from the published wheel and sdist alongside ``dojo.py``, which it
imports.
"""

from .dojo import main

if __name__ == "__main__":
    main()
