"""Output Quality, Degeneration, JSON Repair, and Anti-Hallucination Validator.

Specifically engineered to guard small local models (`qwen3:0.6b`) against:
1. Repetition loops (`میز میز میز...`) and n-gram collapse
2. Verbatim user-prompt echoing
3. Garbled/nonsense Persian token sequences
4. Unverified tool execution claims (claiming a file was created/deleted/searched when no tool succeeded)
5. Malformed JSON router outputs
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from core.language import normalize_persian, PERSIAN_CHAR_RE

# Phrases that claim an external/file/system side-effect occurred
ACTION_CLAIM_PATTERNS_FA = [
    re.compile(r"(فایل|سند|اکسل|ورد|یادداشت)\s+.*?(ذخیره\s+شد|ساخته\s+شد|ویرایش\s+شد|حذف\s+شد|پاک\s+شد|ایجاد\s+شد)"),
    re.compile(r"(در\s+حافظه\s+بلندمدت\s+ذخیره\s+کردم|با\s+موفقیت\s+حذف\s+کردم|نوت‌پد\s+را\s+باز\s+کردم)"),
]
ACTION_CLAIM_PATTERNS_EN = [
    re.compile(r"\bI\s+(have\s+)?(saved|created|deleted|edited|modified|opened)\s+(the\s+)?(file|document|spreadsheet|notepad)\b", re.I),
]

KNOWN_GARBLED_PERSIAN_TOKENS = {
    "نشانگ", "همچند", "خداست", "بسنجی", "سرچشته", "چنانچند",
}
FOREIGN_SCRIPT_LEAK_RE = re.compile(r"[\u0400-\u04FF\u4e00-\u9fff\u0900-\u097F\u3040-\u30ff]")
MIXED_SCRIPT_GLITCH_RE = re.compile(r"[\u0600-\u06FF]+[A-Za-z\u0400-\u04FF]+|[A-Za-z\u0400-\u04FF]+[\u0600-\u06FF]+")


@dataclass
class ValidationReport:
    valid: bool
    cleaned_text: str
    quality_score: float  # 0.0 .. 1.0
    issues: List[str] = field(default_factory=list)
    hallucinated_action: bool = False


class ResponseValidator:
    """Inspects and sanitizes LLM output before displaying to the user or executing tools."""

    @classmethod
    def validate_chat_response(
        cls,
        response_text: str,
        user_message: str,
        language: Literal["fa", "en"] = "fa",
        executed_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidationReport:
        executed_tools = executed_tools or []
        issues: List[str] = []
        score = 1.0

        text = (response_text or "").strip()
        if not text:
            return ValidationReport(
                valid=False,
                cleaned_text="",
                quality_score=0.0,
                issues=["empty_response"],
            )

        # 1. Strip verbatim echo prefix if model repeats user prompt on line 1
        norm_user = normalize_persian(user_message)
        norm_resp = normalize_persian(text)
        if norm_user and norm_resp == norm_user:
            return ValidationReport(
                valid=False,
                cleaned_text="",
                quality_score=0.05,
                issues=["verbatim_prompt_echo"],
            )

        lines = text.splitlines()
        if len(lines) >= 2 and normalize_persian(lines[0]) == norm_user:
            text = "\n".join(lines[1:]).strip()
            score -= 0.15
            issues.append("stripped_prompt_echo_prefix")

        # 2. Detect word / trigram repetition collapse
        words = text.split()
        if len(words) >= 6:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.35:
                score -= 0.65
                issues.append("severe_word_repetition")
            # Collapse consecutive identical words (>2)
            dedup_words: List[str] = []
            for w in words:
                if len(dedup_words) >= 2 and dedup_words[-1] == w and dedup_words[-2] == w:
                    continue
                dedup_words.append(w)
            if len(dedup_words) < len(words):
                text = " ".join(dedup_words)
                score -= 0.35
                issues.append("collapsed_repeated_tokens")

        # 3. Check language fidelity & known 0.6b Persian BPE artifacts
        if language == "fa":
            fa_chars = len(PERSIAN_CHAR_RE.findall(text))
            if len(text) > 12 and fa_chars < max(3, int(len(text) * 0.18)):
                score -= 0.55
                issues.append("insufficient_persian_script")
            lower_tokens = set(re.findall(r"[\u0600-\u06FF]+", text))
            bad_hits = lower_tokens.intersection(KNOWN_GARBLED_PERSIAN_TOKENS)
            if bad_hits:
                score -= 0.65
                issues.append(f"garbled_bpe_tokens:{','.join(sorted(bad_hits))}")
            # Check unexpected foreign script leakage (Cyrillic, CJK, Devanagari, etc.) from 0.6B weights
            if FOREIGN_SCRIPT_LEAK_RE.search(text):
                text = FOREIGN_SCRIPT_LEAK_RE.sub("", text).strip()
                score -= 0.50
                issues.append("stripped_foreign_script_leakage")
            if MIXED_SCRIPT_GLITCH_RE.search(text):
                score -= 0.35
                issues.append("mixed_script_word_glitch")
            # Check mid-word / mid-sentence truncation on longer responses
            if len(text) > 65 and not re.search(r"[.!؟?:)»\"'`0-9۰-۹]$", text.strip()):
                score -= 0.48
                issues.append("truncated_sentence_ending")

        # 4. Anti-hallucination check: did the model claim an action without a successful tool run?
        any_mutating_success = any(t.get("ok") for t in executed_tools)
        hallucinated = False
        if not any_mutating_success:
            patterns = ACTION_CLAIM_PATTERNS_FA if language == "fa" else ACTION_CLAIM_PATTERNS_EN
            for pat in patterns:
                if pat.search(text):
                    hallucinated = True
                    score -= 0.7
                    issues.append("hallucinated_unexecuted_action")
                    break

        # 5. If any tool FAILED, ensure the model doesn't claim success
        any_failed = any(not t.get("ok") for t in executed_tools)
        if any_failed and re.search(r"(با\s+موفقیت\s+انجام\s+شد|ذخیره\s+شد|successfully\s+completed)", text, re.I):
            hallucinated = True
            score -= 0.8
            issues.append("claimed_success_after_failed_tool")

        score = max(0.0, min(1.0, round(score, 3)))
        is_valid = score >= 0.55 and not hallucinated and bool(text.strip())
        return ValidationReport(
            valid=is_valid,
            cleaned_text=text,
            quality_score=score,
            issues=issues,
            hallucinated_action=hallucinated,
        )

    @staticmethod
    def build_safe_fallback(
        language: Literal["fa", "en"],
        reason: str = "low_confidence",
        tool_summaries: Optional[List[str]] = None,
    ) -> str:
        """Honest, deterministic fallback when the 0.6B model produces malformed/garbled output."""
        if tool_summaries:
            joined = "\n".join(f"- {s}" for s in tool_summaries)
            if language == "fa":
                return f"نتیجه بررسی و اجرای ابزار:\n{joined}"
            return f"Verified tool result:\n{joined}"

        if reason == "hallucinated_action":
            if language == "fa":
                return (
                    "هیچ عملیات فایلی یا سیستمی بدون اجرای واقعی ابزار انجام نشده است. "
                    "لطفاً نام دقیق فایل یا دستور مورد نظر را مشخص کنید تا با ابزار مجاز اجرا شود."
                )
            return (
                "No file or system action was performed because no verified tool was executed. "
                "Please specify the exact file or operation you want me to run."
            )

        if language == "fa":
            return (
                "مدل محلی کوچک (`qwen3:0.6b`) نتوانست پاسخ فارسی کاملاً مطمئن و روانی برای این پیام تولید کند. "
                "می‌توانید سؤال را کوتاه‌تر و دقیق‌تر مطرح کنید، از ابزارهای دقیق (`/calc`، `/google`) کمک بگیرید، "
                "یا حالت «معلم آنلاین رایگان» را برای تقویت پاسخ فعال نگه دارید."
            )
        return (
            "The compact local model (`qwen3:0.6b`) could not generate a reliable response for this request. "
            "Please rephrase more specifically or use a dedicated tool (`/calc`, `/google`)."
        )

    @staticmethod
    def parse_and_repair_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Deterministic RFC-8259 JSON extraction & minimal safe repair (never uses `eval`)."""
        if not raw or not raw.strip():
            return None, "empty_json_output"
        cleaned = raw.strip()
        # Remove markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, "no_json_object_found"
        candidate = cleaned[start : end + 1]
        try:
            obj = json.loads(candidate)
            return (obj, None) if isinstance(obj, dict) else (None, "json_root_not_object")
        except json.JSONDecodeError:
            # Safe trailing-comma repair only
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                obj = json.loads(repaired)
                return (obj, None) if isinstance(obj, dict) else (None, "json_root_not_object")
            except Exception as exc:
                return None, f"malformed_json: {exc}"
