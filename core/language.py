"""Language detection, explicit language switching, and Persian text normalization."""
from __future__ import annotations

import re
from typing import Literal, Optional, Tuple

PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")

PERSIAN_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
ARABIC_TO_PERSIAN_CHAR_MAP = str.maketrans({
    "ي": "ی",
    "ك": "ک",
    "ة": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "\u200c": " ",  # Normalize ZWNJ to space in search/normalized_content
})

SWITCH_TO_EN_PATTERNS = [
    re.compile(r"\b(speak|reply|answer|talk|respond|switch)\s+(in\s+)?english\b", re.I),
    re.compile(r"(انگلیسی\s+(جواب\s+بده|پاسخ\s+بده|صحبت\s+کن|حرف\s+بزن)|به\s+زبان\s+انگلیسی)", re.I),
    re.compile(r"^/lang\s+en\b", re.I),
]

SWITCH_TO_FA_PATTERNS = [
    re.compile(r"\b(speak|reply|answer|talk|respond|switch)\s+(in\s+)?(persian|farsi)\b", re.I),
    re.compile(r"(فارسی\s+(جواب\s+بده|پاسخ\s+بده|صحبت\s+کن|حرف\s+بزن)|به\s+زبان\s+فارسی)", re.I),
    re.compile(r"^/lang\s+fa\b", re.I),
]


def to_ascii_digits(text: str) -> str:
    """Convert Persian and Arabic-Indic numerals to ASCII digits."""
    return text.translate(PERSIAN_DIGIT_MAP)


def normalize_persian(text: str, keep_zwnj: bool = False) -> str:
    """Normalize Arabic Yeh/Kaf, digits, and excess whitespace for reliable matching & deduplication."""
    if not text:
        return ""
    out = to_ascii_digits(text)
    if keep_zwnj:
        out = out.replace("ي", "ی").replace("ك", "ک")
    else:
        out = out.translate(ARABIC_TO_PERSIAN_CHAR_MAP)
    out = re.sub(r"\s+", " ", out).strip().lower()
    return out


def detect_explicit_language_switch(text: str) -> Optional[Literal["fa", "en"]]:
    """Return 'fa' or 'en' if the user explicitly asked to switch conversation language."""
    cleaned = text.strip()
    for pat in SWITCH_TO_EN_PATTERNS:
        if pat.search(cleaned):
            return "en"
    for pat in SWITCH_TO_FA_PATTERNS:
        if pat.search(cleaned):
            return "fa"
    return None


def detect_language(
    text: str,
    preferred: Literal["auto", "fa", "en"] = "auto",
) -> Tuple[Literal["fa", "en"], Optional[Literal["fa", "en"]]]:
    """Detect active reply language and whether an explicit language switch was requested.

    Returns:
        (reply_language, explicit_switch_or_None)
    """
    explicit = detect_explicit_language_switch(text)
    if explicit:
        return explicit, explicit

    if preferred in ("fa", "en"):
        # Still allow clear single-turn queries in English if preferred == 'auto',
        # otherwise honor explicit user preference.
        return preferred, None

    fa_count = len(PERSIAN_CHAR_RE.findall(text))
    en_count = len(LATIN_CHAR_RE.findall(text))

    # Strip slash command prefix like /calc or /google before counting Latin chars
    if text.strip().startswith("/"):
        parts = text.strip().split(maxsplit=1)
        body = parts[1] if len(parts) > 1 else ""
        fa_count = len(PERSIAN_CHAR_RE.findall(body))
        en_count = len(LATIN_CHAR_RE.findall(body))

    if fa_count == 0 and en_count >= 2:
        return "en", None
    if en_count > fa_count * 2.5 and fa_count < 3:
        return "en", None
    return "fa", None
