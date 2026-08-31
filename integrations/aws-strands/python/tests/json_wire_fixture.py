"""The canonical value the wire-serialization tests round through the bridge.

One fixture carries every axis on which Python's ``json.dumps`` and the
TypeScript adapter's ``JSON.stringify`` can diverge: nesting, a non-ASCII
letter, an astral character, a forward slash and a control character.
"""

from __future__ import annotations

import json

PARITY_VALUE = {
    "note": "café 😀 a/b \u0007",
    "nested": {"items": [1, "é"]},
}

# What JSON.stringify(PARITY_VALUE) produces, character for character.
PARITY_JSON = '{"note":"café 😀 a/b \\u0007","nested":{"items":[1,"é"]}}'

# The padded, ASCII-escaped form a bare json.dumps produces. Used as input
# wherever a test needs a string the site under test has to re-serialize.
PARITY_JSON_PYTHON_DEFAULT = json.dumps(PARITY_VALUE)
