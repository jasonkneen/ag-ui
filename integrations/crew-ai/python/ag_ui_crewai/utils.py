import asyncio
import re


def camel_to_snake(name: str) -> str:
    """Convert a camelCase key to snake_case.

    Frontend callers send ``forwardedProps`` keys in camelCase; downstream
    CrewAI flow / tool code reads snake_case. This mirrors the LangGraph
    adapter's ``camel_to_snake`` so both bridges normalize identically.
    """
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


async def yield_control():
    """
    Yield control to the event loop.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    loop.call_soon(future.set_result, None)
    await future
