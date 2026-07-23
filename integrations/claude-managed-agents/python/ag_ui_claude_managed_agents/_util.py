"""Small internal helpers."""

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    """Await `value` if it is awaitable (session-store methods may be sync or async)."""
    if inspect.isawaitable(value):
        return await value
    return value


def get(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` from a Managed Agents event, which may be a model or a dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
