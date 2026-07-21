from __future__ import annotations

import re
import unicodedata

# ── Hidden-element patterns (remove the whole line) ──────────────────────
_HIDDEN_CSS_RE = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\.0+)?(?!\d))",
    re.IGNORECASE,
)
_ARIA_HIDDEN_RE = re.compile(r"""aria-hidden\s*=\s*["']true["']""", re.IGNORECASE)
# ── Off-screen patterns (remove the whole line) ──────────────────────────
# Matches elements positioned far off-screen (≥1000px) OR hidden via
# position:fixed/absolute with small offsets combined with overflow:hidden
# on a parent — a common pattern for visually-hidden injection text.
_OFFSCREEN_RE = re.compile(
    r"(?:left|top|right|bottom)\s*:\s*-?\d{4,}px|"
    r"position\s*:\s*(?:fixed|absolute)\s*;\s*(?:left|top)\s*:\s*-?\d{4,}px|"
    r"transform\s*:\s*translate[XY]?\s*\(\s*-?\d{4,}|"
    r"overflow\s*:\s*hidden\s*;.*(?:left|top)\s*:\s*-?\d{3,}px",
    re.IGNORECASE,
)

# ── Instruction-like patterns (replace with [FILTERED]) ───────────────────
_INSTRUCTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"disregard\s+(?:previous|prior|above|earlier)\s+(?:instructions?|content|text)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"forget\s+(?:everything|(?:previous|prior|above|earlier)\s+(?:instructions?|content|text|everything))",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"for\s+security\s+reasons?\s+(?:delete|remove|clear|wipe|destroy)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"system\s+prompt\s*(?::|is|now|override|update)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"you\s+are\s+(?:an?\s+)?ai\s+(?:assistant|agent|model|system)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"debug\s+mode\s*(?:enabled|activated|on|is)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"reveal\s+(?:all\s+)?(?:secrets?|passwords?|keys?|credentials?|data)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"admin\s+override\s*(?:enabled|activated|mode)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"execute\s+(?:arbitrary\s+)?(?:code|script|command|javascript)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"new\s+system\s+(?:prompt|instruction|directive)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"override\s+(?:previous|original|system)\s+(?:instructions?|prompt|rules?)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"new\s+(?:instructions?|prompt|directive|task)\s*(?::|is|now|from)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"updated\s+(?:instructions?|prompt|directive)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"from\s+now\s+on\s+(?:you\s+(?:are|must|should|will)|your)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"your\s+new\s+(?:task|goal|objective|job|role)\s+(?:is|:)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
    (
        re.compile(
            r"disregard\s+(?:everything|all)\s+(?:and|&)",
            re.IGNORECASE,
        ),
        "[FILTERED:instruction]",
    ),
]


# ── Homoglyph translation table ──────────────────────────────────────────
# Maps visually-confusable non-Latin characters to their Latin equivalents
# so the Latin-only instruction regexes below can match them. The set below
# covers the Cyrillic / Greek / fullwidth letter pairs most commonly used to
# bypass English-keyword filters. Full NFKC compatibility decomposition is
# applied first (covers fullwidth / compatibility forms like 'Ｐ'→'P'),
# then this map handles the confusables that NFKC does not fold
# (Cyrillic 'і' U+0456 ≠ Latin 'i' under NFKC; same for the rest below).
_HOMOGLYPHS: dict[str, str] = {
    # Cyrillic → Latin
    "а": "a", "А": "A",  # Cyrillic а (U+0430) / А (U+0410)
    "е": "e", "Е": "E",  # Cyrillic е (U+0435) / Е (U+0415)
    "о": "o", "О": "O",  # Cyrillic о (U+043E) / О (U+041E)
    "р": "p", "Р": "P",  # Cyrillic р (U+0440) / Р (U+0420)
    "с": "c", "С": "C",  # Cyrillic с (U+0441) / С (U+0421)
    "х": "x", "Х": "X",  # Cyrillic х (U+0445) / Х (U+0425)
    "у": "y", "У": "Y",  # Cyrillic у (U+0443) / У (U+0423)
    "і": "i", "І": "I",  # Cyrillic і (U+0456) / І (U+0406)
    "ј": "j", "Ј": "J",  # Cyrillic ј (U+0458) / Ј (U+0408)
    "ѕ": "s", "Ѕ": "S",  # Cyrillic ѕ (U+0455) / Ѕ (U+0405)
    "ԁ": "d", "Ԁ": "D",  # Cyrillic ԁ (U+0501) / Ԁ (U+0500)
    "Ԍ": "G",            # Cyrillic Ԍ (U+050C, Komi Sje)
    # Latin/IPA lookalikes (not folded by NFKC)
    "ɡ": "g",             # Latin/IPA ɡ (U+0261, script g) → g
    # Greek → Latin (the few that look identical to a Latin letter)
    "α": "a", "Α": "A",
    "ο": "o", "Ο": "O",
    "ν": "v", "Ν": "N",
    "ρ": "p", "Ρ": "P",
    "τ": "t", "Τ": "T",
    "ι": "i", "Ι": "I",
    "κ": "k", "Κ": "K",
    "χ": "x", "Χ": "X",
    "ε": "e", "Ε": "E",
}


def _fold_homoglyphs(text: str) -> str:
    """Replace confusable non-Latin characters with their Latin equivalents.

    Runs AFTER NFKC so the two are complementary: NFKC handles fullwidth /
    compatibility forms (e.g. 'Ｐ'→'P', 'ⅰ'→'i'); this map handles the
    Cyrillic / Greek lookalikes that NFKC deliberately does not fold
    (because they are semantically distinct characters in the source
    alphabet). Without this, a payload like 'іgnore previous instructions'
    (Cyrillic і) would pass straight through.
    """
    return "".join(_HOMOGLYPHS.get(ch, ch) for ch in text)


def sanitize(page_content: str) -> str:
    """Treat page content as untrusted DATA, never as instructions.

    Strips hidden DOM text, off-screen elements, and instruction-like
    patterns that are the primary failure mode of browser agents.

    Two normalization passes defend against Unicode bypasses of the Latin-
    only instruction regexes:
      1. NFKC — folds compatibility / fullwidth forms (e.g. 'Ｐ'→'P').
      2. Cyrillic/Greek homoglyph fold — maps confusable non-Latin letters
         to their Latin equivalents (e.g. Cyrillic 'і'→'i'). NFKC does NOT
         perform this fold (Cyrillic and Latin are distinct alphabets), so
         the explicit table is required.
    """
    # NFKC normalization: fullwidth / compatibility-form confusables.
    page_content = unicodedata.normalize("NFKC", page_content)
    # Cyrillic/Greek homoglyph fold: visually-identical letters that NFKC
    # deliberately does not normalize. Without this, a single Cyrillic 'і'
    # in 'іgnore previous instructions' bypasses every regex below.
    page_content = _fold_homoglyphs(page_content)

    lines = page_content.split("\n")
    kept: list[str] = []

    for line in lines:
        if _HIDDEN_CSS_RE.search(line):
            continue
        if _ARIA_HIDDEN_RE.search(line):
            continue
        if _OFFSCREEN_RE.search(line):
            continue
        kept.append(line)

    result = "\n".join(kept)

    for pattern, replacement in _INSTRUCTION_PATTERNS:
        result = pattern.sub(replacement, result)

    return result
