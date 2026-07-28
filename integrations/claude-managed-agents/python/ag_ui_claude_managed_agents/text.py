"""Helpers for turning Managed Agents content blocks into display text."""

import re
from collections.abc import Sequence
from typing import Any

from ._util import get
from .constants import SEARCH_RESULT_PREVIEW_CHARS

_HEX_ENTITY = re.compile(r"&#x([0-9a-fA-F]+);")
_DEC_ENTITY = re.compile(r"&#(\d+);")


REPLACEMENT_CHARACTER = "�"
"""Substituted for any entity that does not denote a usable character."""


def _code_point(n: int) -> str:
    """The character an entity's code point denotes, or U+FFFD.

    Surrogate code points (U+D800-U+DFFF) are rejected as well as out-of-range
    ones: `chr` happily produces a lone surrogate, but that string cannot be
    encoded as UTF-8, so it would raise inside SSE encoding rather than reach
    the UI. Substituting here keeps every port's output well-formed and
    identical.
    """
    if n < 0 or n > 0x10FFFF or 0xD800 <= n <= 0xDFFF:
        return REPLACEMENT_CHARACTER
    return chr(n)


def decode_entities(s: str) -> str:
    """Decode numeric and the common named HTML entities."""
    s = _HEX_ENTITY.sub(lambda m: _code_point(int(m.group(1), 16)), s)
    s = _DEC_ENTITY.sub(lambda m: _code_point(int(m.group(1))), s)
    return (
        s.replace("&quot;", '"')
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )


def text_of(content: Sequence[Any] | None) -> str:
    """Concatenate the text of every `text` block."""
    parts: list[str] = []
    for block in content or []:
        text = get(block, "text")
        if get(block, "type") == "text" and isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def describe_tool_result(content: Sequence[Any] | None) -> str:
    """Flatten a tool result's blocks (text, search results, images, documents) into a string."""
    lines: list[str] = []
    for block in content or []:
        block_type = get(block, "type")
        text = get(block, "text")
        if block_type == "text" and isinstance(text, str):
            lines.append(decode_entities(text))
            continue
        if block_type == "search_result":
            inner_content = get(block, "content")
            inner = text_of(inner_content) if isinstance(inner_content, list) else ""
            title = decode_entities(str(get(block, "title") or ""))
            source = str(get(block, "source") or "")
            line = f"[search result] {title} — {source}"
            if inner:
                line += f"\n{decode_entities(inner)[:SEARCH_RESULT_PREVIEW_CHARS]}"
            lines.append(line)
            continue
        lines.append(f"[{block_type}]")
    return "\n".join(lines).strip()
