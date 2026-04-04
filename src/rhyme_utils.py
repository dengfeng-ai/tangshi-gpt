"""Rhyme checking utilities for Tang poetry evaluation.

Uses the Pingshui rhyme table (平水韵) to check rhyme consistency.
Falls back to pypinyin (modern Mandarin) if the rhyme table is not available.
"""

import json
import os

# Load Pingshui rhyme table: char -> rhyme group name
_PINGSHUI_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "pingshui_rhyme.json"
)

_CHAR_TO_RHYME: dict[str, str] = {}
if os.path.exists(_PINGSHUI_PATH):
    with open(_PINGSHUI_PATH, encoding="utf-8") as f:
        _CHAR_TO_RHYME = json.load(f)

HAS_PINGSHUI = bool(_CHAR_TO_RHYME)


def get_rhyme_group(char: str) -> str | None:
    """Get the Pingshui rhyme group for a single Chinese character."""
    return _CHAR_TO_RHYME.get(char)


def check_rhyme_consistency(rhyme_chars: list[str]) -> dict:
    """Check whether a list of rhyme characters share the same Pingshui rhyme group.

    Returns a dict with keys:
        available: bool - whether the rhyme table is available
        consistent: bool - whether all rhyme chars share the same rhyme group
        rhyme_groups: list[str|None] - rhyme group for each character
        rhyme_chars: list[str] - the input characters
    """
    if not HAS_PINGSHUI:
        return {"available": False}

    if len(rhyme_chars) < 2:
        return {
            "available": True,
            "consistent": True,
            "rhyme_groups": [get_rhyme_group(c) for c in rhyme_chars],
            "rhyme_chars": rhyme_chars,
        }

    groups = [get_rhyme_group(c) for c in rhyme_chars]
    valid_groups = [g for g in groups if g is not None]

    if not valid_groups:
        return {
            "available": True,
            "consistent": False,
            "rhyme_groups": groups,
            "rhyme_chars": rhyme_chars,
        }

    consistent = len(set(valid_groups)) == 1
    return {
        "available": True,
        "consistent": consistent,
        "rhyme_groups": groups,
        "rhyme_chars": rhyme_chars,
    }
