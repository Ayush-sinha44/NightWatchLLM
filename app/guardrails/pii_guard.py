"""
Phase 3 — PII Detection Guard

Regex-only scanner that detects personally identifiable information (PII)
in user input and blocks the request before it ever reaches the LLM.

Detected PII categories:
  • Email addresses
  • Phone numbers  — Indian (+91) and US formats
  • Credit card numbers  — 13-19 digits with optional spaces/dashes
  • Aadhaar numbers  — 12 consecutive digits (Indian national ID)

Usage:
    guard = PIIGuard()
    result = await guard.check(user_text)
    if result.blocked:
        ...
"""

import re
import logging

from app.guardrails.output_guard import GuardResult

logger = logging.getLogger("gateway.pii_guard")


# ---------------------------------------------------------------------------
# Pre-compiled PII regex patterns
# ---------------------------------------------------------------------------
# Each entry is (compiled_pattern, human-readable label).

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email addresses — standard RFC-5322-ish match.
    (
        re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        ),
        "email address",
    ),
    # Indian phone numbers: +91 followed by 10 digits, with optional
    # separators (spaces, dashes, dots).
    (
        re.compile(
            r"(?<!\d)\+91[\s\-.]?\d{5}[\s\-.]?\d{5}(?!\d)",
        ),
        "phone number (Indian +91)",
    ),
    # US phone numbers: optional +1, then 3-3-4 digit groups with
    # parentheses, spaces, dashes, or dots as separators.
    (
        re.compile(
            r"(?<!\d)(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)",
        ),
        "phone number (US)",
    ),
    # Credit card numbers: 13-19 digits, optionally separated by spaces
    # or dashes in groups of 4.
    (
        re.compile(
            r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,7}(?!\d)",
        ),
        "credit card number",
    ),
    # Aadhaar numbers: exactly 12 consecutive digits (Indian national ID).
    (
        re.compile(
            r"(?<!\d)\d{12}(?!\d)",
        ),
        "Aadhaar number",
    ),
]


# ---------------------------------------------------------------------------
# PIIGuard
# ---------------------------------------------------------------------------
class PIIGuard:
    """Regex-based PII detector.  No API calls — instant and free."""

    async def check(self, text: str) -> GuardResult:
        """Scan *text* for PII patterns and return a verdict.

        Returns GuardResult(blocked=True) on the first match, with a
        human-readable reason identifying the PII category found.
        """
        for pattern, label in _PII_PATTERNS:
            if pattern.search(text):
                logger.warning("PII detected: %s", label)
                return GuardResult(
                    blocked=True,
                    reason=f"PII detected: {label}",
                )

        return GuardResult(blocked=False)
