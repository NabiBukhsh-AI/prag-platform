"""Pattern detectors shared by every guardrail phase.

One pattern set, used at input, at retrieval and at output. A second copy for the retrieval scan
would drift from the first, and the drift would show up as an injection the input check catches
arriving unremarked through a document.

These are the cheap tier. A fine-tuned classifier sits above them when one exists; the patterns
stay, because they are deterministic, explainable in an incident review, and they catch the
known families at a cost of microseconds.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "INJECTION_PATTERNS",
    "OVERRIDE_PATTERNS",
    "find_first",
    "find_pii",
    "find_secret",
    "redact_pii",
]

_I = re.IGNORECASE

#: Known jailbreak and injection families: override-the-instructions, extract-the-prompt, the
#: named persona jailbreaks, and chat-template delimiter smuggling.
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
        r"\b(?:previous|prior|above|earlier|all|any|system|your)\b[^.\n]{0,20}?"
        r"\b(?:instructions?|prompts?|rules|directions|guidelines|context)\b",
        _I,
    ),
    re.compile(
        r"\b(?:reveal|print|show|repeat|output|leak|dump)\b[^.\n]{0,30}?"
        r"\b(?:system prompt|hidden prompt|initial instructions|your instructions|your prompt)\b",
        _I,
    ),
    re.compile(r"\b(?:developer mode|jailbreak(?:ed)?|do anything now)\b", _I),
    re.compile(r"<\|im_(?:start|end)\|>|\[/?INST\]|<\s*/?\s*(?:system|assistant)\s*>", _I),
)

#: Attempts to redefine the system role rather than to escape it.
OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\byou are (?:now|no longer)\b", _I),
    re.compile(r"\bfrom now on,? you\b", _I),
    re.compile(r"\bnew (?:system )?(?:instructions?|rules|role)\s*:", _I),
    re.compile(r"^\s*system\s*:", _I | re.MULTILINE),
    re.compile(r"\bact as (?:an? )?(?:unrestricted|unfiltered|uncensored)\b", _I),
)

_PII: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
)

_SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("api_key", re.compile(r"\b(?:sk|rk)[-_](?:live[-_]|test[-_]|proj[-_])?[A-Za-z0-9]{20,}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*")),
    # A long base64-looking value in a URL query string is the shape of exfiltration through a
    # rendered link or image: the model is induced to encode data into a URL the client fetches.
    (
        "encoded_payload_url",
        re.compile(r"https?://\S+?[?&][^=\s&]+=[A-Za-z0-9+/_-]{40,}={0,2}"),
    ),
)


def find_first(patterns: Iterable[re.Pattern[str]], text: str) -> re.Match[str] | None:
    """The first match of any pattern, so a verdict can quote the span that triggered it."""
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


def _luhn_valid(digits: str) -> bool:
    """Card numbers carry a check digit. Without this, every long number is a "card"."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _pii_matches(text: str) -> list[tuple[str, re.Match[str]]]:
    found = []
    for label, pattern in _PII:
        for match in pattern.finditer(text):
            if label == "CARD" and not _luhn_valid(re.sub(r"\D", "", match.group())):
                continue
            found.append((label, match))
    return found


def find_pii(text: str) -> tuple[str, ...]:
    """The PII kinds present, as labels. Labels only, so the result is safe to log."""
    return tuple(sorted({label for label, _ in _pii_matches(text)}))


def redact_pii(text: str) -> tuple[str, int]:
    """Replace each PII span with a typed placeholder, returning the text and the count."""
    matches = _pii_matches(text)
    # Right to left, so earlier offsets stay valid as later spans are replaced; overlapping
    # spans (a phone number inside a card-shaped run) keep only the first one reached.
    redacted, last_start = text, len(text) + 1
    count = 0
    for label, match in sorted(matches, key=lambda m: m[1].start(), reverse=True):
        if match.end() > last_start:
            continue
        redacted = redacted[: match.start()] + f"[REDACTED:{label}]" + redacted[match.end() :]
        last_start = match.start()
        count += 1
    return redacted, count


def find_secret(text: str) -> str | None:
    """The kind of the first secret-shaped span, or ``None``. Never the span itself."""
    for label, pattern in _SECRETS:
        if pattern.search(text):
            return label
    return None
