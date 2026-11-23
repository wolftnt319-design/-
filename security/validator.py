"""Security Validator: Schema Validation, Secret Redaction, and Prompt Injection Defense."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
import jsonschema

# Patterns for sensitive credentials, passwords, API keys, JWTs, bank cards
SECRET_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("openai_or_api_key", re.compile(r"\b(?:sk|pk|rk|ghp|gho|glpat|hf|xoxb|xoxp)-[A-Za-z0-9_\-]{12,}\b")),
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*\b")),
    ("bank_card", re.compile(r"\b(?:\d[ -]*?){16}\b")),
    ("password_assignment", re.compile(
        r"(?i)(?:password|passwd|pwd|secret|api[_\s-]?key|token|رمز\s*عبور|پسورد|کارت\s*بانکی|شماره\s*کارت|توکن)\s*[:=]\s*\S+"
    )),
    ("persian_password_phrase", re.compile(
        r"(?:رمز(?:\s+عبور|\s+من|\s+حساب|\s+ایمیل|\s+کارت)?|پسورد(?:\s+من)?|توکن(?:\s+من)?)\s+(?:من\s+)?[\w@#$%^&*!.\-]{4,}"
    )),
]

# Prompt injection patterns inside untrusted web / document content
PROMPT_INJECTION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions_en", re.compile(r"(?i)\b(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|rules?)\b")),
    ("system_override_en", re.compile(r"(?i)(\[system\]|<\|im_start\|>\s*system|new\s+system\s+prompt\s*:|you\s+are\s+now\s+in\s+developer\s+mode)")),
    ("exfiltrate_prompt_en", re.compile(r"(?i)\b(reveal|print|output|leak|show)\s+(your\s+)?(system\s+prompt|hidden\s+instructions|api\s+key)\b")),
    ("dangerous_exec_en", re.compile(r"(?i)\b(run|execute)\s+(shell|cmd|powershell|bash|rm\s+-rf|del\s+/f|format\s+c:)\b")),
    ("ignore_instructions_fa", re.compile(r"(دستورات?\s+(قبلی|سیستم|بالا)\s+را\s+(نادیده\s+بگیر|فراموش\s+کن)|پرامپت\s+سیستم\s+را\s+(نشان\s+بده|افشا\s+کن))")),
    ("tool_hijack", re.compile(r'(?i)"intent"\s*:\s*"(notepad|file|word|excel|browser)"\s*,\s*"action"')),
]


class SecretRedactor:
    """Detects and redacts sensitive secrets before logging, memory storage, or online transmission."""

    @classmethod
    def contains_secret(cls, text: str) -> Tuple[bool, List[str]]:
        if not text:
            return False, []
        normalized_digits = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        matched: List[str] = []
        for name, pat in SECRET_PATTERNS:
            if pat.search(normalized_digits):
                matched.append(name)
        return bool(matched), matched

    @classmethod
    def redact(cls, text: str) -> str:
        if not text:
            return text
        out = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        for name, pat in SECRET_PATTERNS:
            out = pat.sub(f"[REDACTED:{name.upper()}]", out)
        return out

    @classmethod
    def redact_dict(cls, data: Any) -> Any:
        if isinstance(data, str):
            return cls.redact(data)
        if isinstance(data, dict):
            sensitive_keys = {"password", "passwd", "secret", "token", "api_key", "apikey", "card_number", "authorization"}
            cleaned = {}
            for k, v in data.items():
                if str(k).lower() in sensitive_keys:
                    cleaned[k] = "[REDACTED:KEY]"
                else:
                    cleaned[k] = cls.redact_dict(v)
            return cleaned
        if isinstance(data, list):
            return [cls.redact_dict(item) for item in data]
        return data


class PromptInjectionGuard:
    """Isolates and sanitizes untrusted web/external text so it can never hijack LUNA."""

    @classmethod
    def inspect_text(cls, text: str) -> Dict[str, Any]:
        if not text:
            return {"suspicious": False, "findings": []}
        findings: List[str] = []
        for label, pat in PROMPT_INJECTION_PATTERNS:
            if pat.search(text):
                findings.append(label)
        return {"suspicious": bool(findings), "findings": findings}

    @classmethod
    def sanitize_untrusted_content(cls, text: str, source_url: str = "external", max_chars: int = 5000) -> Tuple[str, List[str]]:
        """Strip script/control injections and wrap inside a strict non-executable data boundary."""
        if not text:
            return "", []
        # Strip HTML script/style remnants & special role tokens
        cleaned = re.sub(r"(?is)<(script|style|iframe|object|embed).*?>.*?</\1>", " ", text)
        cleaned = cleaned.replace("<|im_start|>", "").replace("<|im_end|>", "")
        inspection = cls.inspect_text(cleaned)
        findings = inspection["findings"]

        for label, pat in PROMPT_INJECTION_PATTERNS:
            cleaned = pat.sub(f"[BLOCKED_UNTRUSTED_DIRECTIVE:{label}]", cleaned)

        cleaned = SecretRedactor.redact(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()[:max_chars]

        wrapped = (
            f"<untrusted_external_data source=\"{source_url}\" executable=\"false\">\n"
            f"{cleaned}\n"
            f"</untrusted_external_data>"
        )
        return wrapped, findings


class ArgumentSchemaValidator:
    """Validates structured tool call arguments against strict JSON Schemas."""

    @staticmethod
    def validate(arguments: Any, schema: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(arguments, dict):
            return False, "Arguments must be a JSON object (dict)."
        try:
            jsonschema.validate(instance=arguments, schema=schema)
            return True, "ok"
        except jsonschema.ValidationError as exc:
            return False, f"Schema validation failed at '{'.'.join(str(p) for p in exc.path)}': {exc.message}"
        except Exception as exc:
            return False, f"Invalid schema or arguments: {exc}"
