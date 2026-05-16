"""
Phase 2 — Output Guardrail

Two-stage filter applied to every LLM response before it reaches the user:

1. **Rule-based check** (fast, zero API calls)
   Scans for known toxic keywords/phrases using pre-compiled regex patterns.

2. **LLM-based check** (async Groq call)
   Sends the text to a strict "judge" prompt that returns a JSON verdict.

Usage:
    guard = OutputGuard()
    result = await guard.check(response_text)
    if result.blocked:
        ...
"""

import json
import re
import logging

from pydantic import BaseModel
from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger("gateway.output_guard")


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------
class GuardResult(BaseModel):
    """Outcome of an output guardrail check."""

    blocked: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Toxic keyword / phrase list  (rule-based layer)
# ---------------------------------------------------------------------------
# These are deliberately broad catch-all patterns.  Expand as needed.
# Phase 3 hook: load these from a config file or database instead of hardcoding.
_TOXIC_KEYWORDS: list[str] = [
    # --- slurs & hate speech (abbreviated / masked for safety) ---
    r"\bn[i1]gg[ae3]r?\b",
    r"\bf[a@]gg?[o0]t\b",
    r"\bk[i1]ke\b",
    r"\bsp[i1]c\b",
    r"\bch[i1]nk\b",
    r"\btr[a@]nn(?:y|ie)\b",
    r"\bwetback\b",
    r"\bcoon\b",

    # --- self-harm / suicide ---
    r"\bkill\s+(your|my|him|her|them)self\b",
    r"\bcommit\s+suicide\b",
    r"\bhow\s+to\s+(hang|cut|poison|overdose)\b",
    r"\bslit\s+(your|my)\s+wrists?\b",
    r"\bend\s+(your|my)\s+life\b",

    # --- extreme violence ---
    r"\bmass\s+(shooting|murder|killing)\b",
    r"\bbuild\s+a\s+bomb\b",
    r"\bmake\s+a?\s*(pipe\s+)?bomb\b",
    r"\bkill\s+everyone\b",
    r"\bgenocide\s+is\s+good\b",
    r"\bethnic\s+cleansing\b",

    # --- CSAM / exploitation ---
    r"\bchild\s+(porn|exploitation|abuse)\b",

    # Phase 3 hook: add patterns for PII leakage, prompt-injection echoing, etc.
]

# Pre-compile all patterns once at import time for speed.
_TOXIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE) for pattern in _TOXIC_KEYWORDS
]


# ---------------------------------------------------------------------------
# Judge prompt (LLM-based layer)
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM_PROMPT = """\
You are a content-safety classifier. You will receive a text snippet produced by \
an AI assistant. Your ONLY job is to decide whether the text is SAFE or UNSAFE.

A text is UNSAFE if it:
- Contains hate speech, slurs, or dehumanising language
- Encourages or instructs self-harm or suicide
- Provides instructions for violence, weapons, or explosives
- Contains sexual content involving minors
- Leaks clearly private personal information (SSN, credit-card numbers, etc.)

Respond with ONLY a JSON object — no markdown fences, no explanation, no extra text.

If safe:   {"verdict": "SAFE", "reason": ""}
If unsafe: {"verdict": "UNSAFE", "reason": "<brief explanation>"}
"""

# Phase 3 hook: make this prompt configurable or load from a template file.


# ---------------------------------------------------------------------------
# OutputGuard
# ---------------------------------------------------------------------------
class OutputGuard:
    """Two-stage output guardrail: fast rule-based check → LLM judge."""

    def __init__(self) -> None:
        # Reuse the same Groq credentials and model from the gateway config.
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.model

    # -- public API ----------------------------------------------------------
    async def check(self, text: str) -> GuardResult:
        """Run all output guardrail stages and return a verdict.

        Phase 3 hook: add more stages here (PII redaction, custom policy
        engine, etc.) by inserting checks between the rule-based and
        LLM-based stages.
        """
        # Stage 1: rule-based (instant, zero cost)
        rule_result = self._rule_based_check(text)
        if rule_result.blocked:
            logger.warning("Rule-based guard blocked response: %s", rule_result.reason)
            return rule_result

        # Stage 2: LLM-based judge (async Groq call)
        llm_result = await self._llm_judge_check(text)
        if llm_result.blocked:
            logger.warning("LLM judge blocked response: %s", llm_result.reason)
        return llm_result

    # -- private stages ------------------------------------------------------
    @staticmethod
    def _rule_based_check(text: str) -> GuardResult:
        """Scan *text* against the pre-compiled toxic-keyword regex list.

        This is intentionally a static method — it touches no external state
        and makes zero API calls, so it stays fast even under load.
        """
        for pattern in _TOXIC_PATTERNS:
            match = pattern.search(text)
            if match:
                return GuardResult(
                    blocked=True,
                    reason=f"Rule-based filter matched: {match.group()!r}",
                )
        return GuardResult(blocked=False)

    async def _llm_judge_check(self, text: str) -> GuardResult:
        """Ask a second LLM call to judge the safety of *text*."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=128,  # judge replies are tiny
                temperature=0,  # deterministic
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()
            verdict = json.loads(raw)

            if verdict.get("verdict") == "UNSAFE":
                return GuardResult(
                    blocked=True,
                    reason=verdict.get("reason", "Flagged by LLM judge"),
                )
            return GuardResult(blocked=False)

        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            # If the judge returns garbage, fail-open so the user isn't
            # blocked by a glitchy safety layer.  Log loudly so we fix it.
            logger.error("LLM judge returned unparseable response: %s", exc)
            # Phase 3 hook: make fail-open vs fail-closed configurable.
            return GuardResult(blocked=False, reason="")

        except Exception as exc:
            # Network / Groq errors — fail-open with a warning.
            logger.error("LLM judge call failed: %s", exc)
            return GuardResult(blocked=False, reason="")
