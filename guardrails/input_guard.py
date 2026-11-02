"""Input-Side Guardrails for Nykaa Support Agent (Task 10).

Features:
1. Fixed-format PII Masking:
   - Indian phone numbers -> [PHONE_MASKED]
   - Credit / Debit Card numbers / Last-4 -> XXXX-XXXX-XXXX-$1 or [CARD_MASKED]
2. Prompt Injection Detection:
   - Flags attempts to hijack instructions, override system rules, or jailbreak.
"""

import re
from typing import Dict, Any, Tuple

# Regex patterns for fixed-format PII
PHONE_REGEX = re.compile(
    r"(?:\+?91[\-\s]?)?[6789]\d{9}\b",
    re.IGNORECASE
)

# 16-digit card patterns (with spaces or dashes)
CARD_16_REGEX = re.compile(
    r"\b(?:\d{4}[\-\s]?){3}(\d{4})\b"
)

# Explicit card references (e.g., "card ending in 4321")
CARD_LAST4_EXPLICIT_REGEX = re.compile(
    r"\bcard\s*(?:ending\s*in|number|no\.?)?\s*[:\-]?\s*(\d{4})\b",
    re.IGNORECASE
)

# Prompt injection heuristic patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"disregard\s+(?:all\s+)?prior\s+guidelines",
    r"you\s+are\s+now\s+in\s+(?:dan|developer|god|unrestricted)\s+mode",
    r"bypass\s+(?:all\s+)?guardrails",
    r"system\s*:\s*override",
    r"act\s+as\s+an\s+unfiltered\s+ai",
    r"reveal\s+(?:the\s+)?system\s+prompt"
]

INJECTION_REGEX = re.compile(
    "|".join(PROMPT_INJECTION_PATTERNS),
    re.IGNORECASE
)


def mask_pii(text: str) -> Tuple[str, bool]:
    """Masks fixed-format PII (phone number, card numbers) from customer input.

    Returns:
        (sanitized_text, pii_detected_bool)
    """
    pii_found = False
    sanitized = text

    # Mask 16-digit cards
    if CARD_16_REGEX.search(sanitized):
        pii_found = True
        sanitized = CARD_16_REGEX.sub(r"XXXX-XXXX-XXXX-\1", sanitized)

    # Mask explicit card references
    if CARD_LAST4_EXPLICIT_REGEX.search(sanitized):
        pii_found = True
        sanitized = CARD_LAST4_EXPLICIT_REGEX.sub(r"card ending in [CARD_MASKED]", sanitized)

    # Mask phone numbers
    if PHONE_REGEX.search(sanitized):
        pii_found = True
        sanitized = PHONE_REGEX.sub("[PHONE_MASKED]", sanitized)

    return sanitized, pii_found


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    """Detects adversarial prompt injection attempts.

    Returns:
        (is_injection_bool, matched_reason)
    """
    match = INJECTION_REGEX.search(text)
    if match:
        return True, f"Adversarial instruction override pattern detected: '{match.group(0)}'"
    return False, ""


def process_input_guardrail(raw_query: str) -> Dict[str, Any]:
    """Applies all input-side guardrails to an incoming user query."""
    is_injection, injection_reason = detect_prompt_injection(raw_query)
    sanitized_text, pii_masked = mask_pii(raw_query)

    return {
        "raw_query": raw_query,
        "sanitized_query": sanitized_text,
        "pii_detected": pii_masked,
        "is_injection": is_injection,
        "rejection_reason": injection_reason if is_injection else None,
        "allowed": not is_injection
    }

