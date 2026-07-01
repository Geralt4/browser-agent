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


def sanitize(page_content: str) -> str:
    """Treat page content as untrusted DATA, never as instructions.

    Strips hidden DOM text, off-screen elements, and instruction-like
    patterns that are the primary failure mode of browser agents.

    Applies Unicode NFKC normalization first to defeat homoglyph attacks
    (e.g. Cyrillic 'і' → Latin 'i').
    """
    # NFKC normalization converts homoglyphs to their canonical forms,
    # defeating Unicode-based bypasses of the instruction patterns below.
    page_content = unicodedata.normalize("NFKC", page_content)

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
