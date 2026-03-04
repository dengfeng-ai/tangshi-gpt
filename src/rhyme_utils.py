"""Rhyme checking utilities for Tang poetry evaluation.

Uses pypinyin to extract finals (韵母) and check rhyme consistency.
Gracefully degrades if pypinyin is not installed.
"""

try:
    from pypinyin import pinyin, Style

    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

# Equivalence groups: map specific finals to a canonical form so that
# characters with traditionally compatible rhymes are grouped together.
_FINAL_EQUIVALENCES = {
    # -an group
    "ian": "an",
    "uan": "an",
    "üan": "an",
    # -ang group
    "iang": "ang",
    "uang": "ang",
    # -en group
    "in": "en",
    "un": "en",
    "ün": "en",
    # -eng group
    "ing": "eng",
    "ong": "eng",
    "iong": "eng",
    # -ao group
    "iao": "ao",
    # -ou group
    "iu": "ou",
    "iou": "ou",
    # -e group
    "uo": "e",
    "üe": "e",
    "ue": "e",
    # -ai group
    "uai": "ai",
    # -ei group
    "ui": "ei",
    "uei": "ei",
}


def get_normalized_final(char: str) -> str | None:
    """Get the normalized rhyme final for a single Chinese character."""
    if not HAS_PYPINYIN:
        return None

    result = pinyin(char, style=Style.FINALS, heteronym=False)
    if not result or not result[0] or not result[0][0]:
        return None

    final = result[0][0]
    return _FINAL_EQUIVALENCES.get(final, final)


def check_rhyme_consistency(rhyme_chars: list[str]) -> dict:
    """Check whether a list of rhyme characters share the same final.

    Returns a dict with keys:
        available: bool - whether pypinyin is available
        consistent: bool - whether all rhyme chars share the same final
        finals: list[str|None] - normalized final for each character
        rhyme_chars: list[str] - the input characters
    """
    if not HAS_PYPINYIN:
        return {"available": False}

    if len(rhyme_chars) < 2:
        return {
            "available": True,
            "consistent": True,
            "finals": [get_normalized_final(c) for c in rhyme_chars],
            "rhyme_chars": rhyme_chars,
        }

    finals = [get_normalized_final(c) for c in rhyme_chars]
    valid_finals = [f for f in finals if f is not None]

    if not valid_finals:
        return {
            "available": True,
            "consistent": False,
            "finals": finals,
            "rhyme_chars": rhyme_chars,
        }

    consistent = len(set(valid_finals)) == 1
    return {
        "available": True,
        "consistent": consistent,
        "finals": finals,
        "rhyme_chars": rhyme_chars,
    }
