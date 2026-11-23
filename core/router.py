"""Stage 5: 4-Stage Cascade Hybrid Router for LUNA AI.

Combines:
- Stage 0: Explicit Slash Commands (`/calc`, `/google`, `/browse`, `/files`, `/memory`, `/notepad`) & Shell-Injection Gate
- Stage 1: Deterministic Regex / Structural Argument Extractors (< 1ms CPU, 100% argument fidelity)
- Stage 2: Character/Word TF-IDF Intent Prototype Classifier (< 2ms CPU)
- Stage 3: Ollama Constrained-JSON Router (`qwen3:0.6b` with JSON Schema grammar) for ambiguous phrasing
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.language import normalize_persian, to_ascii_digits
from core.model_manager import model_manager
from core.response_validator import ResponseValidator
from memory.long_memory import LongTermMemory
from security.permissions import audit_logger
from tools.registry import ToolRegistry

IntentType = Literal[
    "calculator", "search", "browser", "word", "excel", "notepad", "file", "memory", "chat", "unknown"
]

UNSAFE_SHELL_RE = re.compile(
    r"(?i)\b(rm\s+-rf|del\s+/[fqs]|format\s+[a-z]:|powershell\s+-enc|cmd\.exe\s+/c|shutdown\s+/[sr]|reg\s+delete)\b"
)

URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)

# Detects arithmetic expressions or explicit calculation requests
MATH_SIGNAL_RE = re.compile(
    r"(?:^/calc\b|محاسبه\s+کن|حساب\s+کن|ریشه\s+دوم|ریشه\s+سوم|جذر|درصد\s+|ضربدر|تقسیم\s+بر|به\s+علاوه|منهای|به\s+توان|\bsqrt\s*\(|\bfactorial\s*\(|\bsin\s*\(|\bcos\s*\(|\blog\s*\()",
    re.I,
)
PURE_ARITHMETIC_RE = re.compile(
    r"^[\s0-9۰-۹.+\-*/%^()×÷−=؟?]+$"
)

SEARCH_PREFIX_RE = re.compile(
    r"(?i)^(?:/google|/search|در\s+اینترنت\s+جستجو\s+کن|جستجو\s+کن|در\s+گوگل\s+(?:سرچ\s+کن|بگرد\s+دنبال|جستجو\s+کن)|سرچ\s+کن|search\s+(?:the\s+web\s+for|google\s+for|for))\s*[:：]?\s*(.+)$"
)

MEMORY_QUERY_RE = re.compile(
    r"(?:در\s+حافظه(?:‌ات)?\s+.*?(?:چه|چی)\s+(?:داری|هست|می‌?دانی)|(?:اسم|نام)\s+(?:این\s+)?پروژه\s+(?:چیست|چیه)|درباره\s+من\s+چه\s+می‌?دانی|what\s+do\s+you\s+remember|what\s+is\s+(?:the|this)\s+project(?:'s)?\s+name)",
    re.I,
)

ROUTER_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["calculator", "search", "browser", "word", "excel", "notepad", "file", "memory", "chat", "unknown"],
        },
        "action": {"type": "string"},
        "arguments": {"type": "object"},
        "requires_confirmation": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["intent", "action", "arguments", "requires_confirmation", "reason"],
}

INTENT_PROTOTYPES: List[tuple[str, str]] = [
    ("calculator", "محاسبه کن حساب کن جمع ضرب تقسیم درصد جذر ریشه توان calculate math sum multiply percent sqrt"),
    ("search", "جستجو در اینترنت سرچ گوگل اخبار جدید قیمت امروز منبع وب search web google latest news"),
    ("browser", "باز کردن لینک سایت استخراج متن صفحه وب open url website extract page"),
    ("word", "فایل ورد داکس سند ورد بساز بخوان ویرایش کن word docx document paragraph"),
    ("excel", "فایل اکسل جدول شیت سلول فرمول مرتب سازی ردیف ستون excel xlsx spreadsheet cell sort"),
    ("notepad", "نوت پد یادداشت باز کن داستان بنویس ذخیره در نوت پد notepad txt open note"),
    ("file", "لیست فایل ها پوشه خواندن فایل حذف فایل جستجوی فایل list files read delete folder"),
    ("memory", "در حافظه ذخیره کن یادت بمونه فراموش کن حافظه من remember memory save forget"),
    ("chat", "سلام خوبی معرفی کن توضیح بده تفاوت چیست راهنمایی کن چطور کار میکند hello explain what is how to"),
]


@dataclass
class RouteDecision:
    intent: IntentType
    action: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    reason: str = ""
    router_stage: str = "deterministic"
    secondary_steps: List["RouteDecision"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "action": self.action,
            "arguments": self.arguments,
            "requires_confirmation": self.requires_confirmation,
            "reason": self.reason,
            "router_stage": self.router_stage,
        }


class HybridRouter:
    """Fast, safe 4-stage router built specifically to keep `qwen3:0.6b` accurate and low-latency."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self._proto_labels = [p[0] for p in INTENT_PROTOTYPES]
        self._proto_matrix = self._vec.fit_transform([normalize_persian(p[1]) for p in INTENT_PROTOTYPES])

    async def route(self, user_message: str, session_id: str = "default") -> RouteDecision:
        text = (user_message or "").strip()
        if not text:
            return RouteDecision(intent="unknown", action="none", reason="empty_message")

        # --- Stage 0: Security Gate & Explicit Slash Commands ---
        if UNSAFE_SHELL_RE.search(text):
            audit_logger.log(
                event_type="unsafe_shell_blocked",
                actor="router",
                status="blocked",
                details={"message": text[:200]},
                session_id=session_id,
            )
            return RouteDecision(
                intent="unknown",
                action="blocked_unsafe_shell",
                reason="فرمان مستقیم سیستم/شِل مسدود شد.",
                router_stage="guardrail",
            )

        slash_decision = self._route_slash(text)
        if slash_decision is not None:
            return slash_decision

        # --- Stage 1: Deterministic High-Precision Pattern Extractors ---
        det_decision = self._route_deterministic(text)
        if det_decision is not None:
            return det_decision

        # --- Stage 2: TF-IDF Prototype Classifier ---
        norm_text = normalize_persian(text)
        q_vec = self._vec.transform([norm_text])
        sims = cosine_similarity(q_vec, self._proto_matrix)[0]
        best_idx = int(sims.argmax())
        best_intent = self._proto_labels[best_idx]
        best_sim = float(sims[best_idx])

        # Clear general chat or low tool similarity -> fast direct chat (saves a 4s LLM router call!)
        if best_intent == "chat" or best_sim < 0.16:
            return RouteDecision(
                intent="chat",
                action="chat_reply",
                reason=f"گفتگوی عمومی (similarity={best_sim:.2f})",
                router_stage="tfidf",
            )

        # --- Stage 3: Constrained JSON LLM Router (`qwen3:0.6b`) for mid-band tool ambiguity ---
        llm_decision = await self._route_with_llm(text, session_id=session_id)
        if llm_decision is not None:
            return llm_decision

        return RouteDecision(
            intent="chat",
            action="chat_reply",
            reason="fallback_to_safe_chat",
            router_stage="fallback",
        )

    def _route_slash(self, text: str) -> Optional[RouteDecision]:
        if not text.startswith("/"):
            return None
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/calc" and rest:
            return RouteDecision(
                intent="calculator",
                action="calc_evaluate",
                arguments={"expression": rest},
                reason="دستور صریح /calc",
                router_stage="slash",
            )
        if cmd in ("/google", "/search") and rest:
            return RouteDecision(
                intent="search",
                action="web_search",
                arguments={"query": rest, "max_results": 5, "fetch_top_page": True},
                reason=f"دستور صریح {cmd}",
                router_stage="slash",
            )
        if cmd == "/browse" and rest:
            return RouteDecision(
                intent="browser",
                action="browser_extract",
                arguments={"url": rest},
                reason="دستور صریح /browse",
                router_stage="slash",
            )
        if cmd == "/files":
            return RouteDecision(
                intent="file",
                action="file_list",
                arguments={"subdir": rest},
                reason="دستور صریح /files",
                router_stage="slash",
            )
        if cmd == "/memory" and rest:
            return RouteDecision(
                intent="memory",
                action="memory_save",
                arguments={"content": rest, "category": "fact", "importance": 4},
                requires_confirmation=True,
                reason="دستور صریح /memory",
                router_stage="slash",
            )
        if cmd == "/notepad" and rest:
            if ":" in rest:
                fname, body = rest.split(":", 1)
                return RouteDecision(
                    intent="notepad",
                    action="notepad_write",
                    arguments={"filename": fname.strip(), "content": body.strip(), "open_window": True},
                    reason="دستور صریح /notepad",
                    router_stage="slash",
                )
            return RouteDecision(
                intent="notepad",
                action="notepad_story_flow",
                arguments={"filename": "luna_note.txt", "topic": rest},
                reason="جریان تولید متن و باز کردن Notepad",
                router_stage="slash",
            )
        return None

    def _route_deterministic(self, text: str) -> Optional[RouteDecision]:
        ascii_text = to_ascii_digits(text)

        # 1. Interactive Memory Learning ("اسم این پروژه سایه است.")
        mem_cand = LongTermMemory.detect_interactive_candidate(text)
        if mem_cand is not None:
            return RouteDecision(
                intent="memory",
                action="memory_save",
                arguments={
                    "content": mem_cand["content"],
                    "title": mem_cand["title"],
                    "category": mem_cand["category"],
                    "importance": mem_cand["importance"],
                },
                requires_confirmation=True,
                reason="تشخیص آموزش تعاملی حافظه بلندمدت",
                router_stage="deterministic",
            )

        # 2. Explicit Memory Recall Check
        if MEMORY_QUERY_RE.search(text):
            return RouteDecision(
                intent="memory",
                action="memory_search",
                arguments={"query": text},
                reason="پرسش مبتنی بر حافظه کاربر",
                router_stage="deterministic",
            )

        # 3. Direct Browser URL Extraction
        url_match = URL_RE.search(text)
        if url_match:
            url = url_match.group(0).rstrip(".,)")
            want_dl = bool(re.search(r"(دانلود|download)", text, re.I))
            want_form = bool(re.search(r"(فرم|ارسال\s+فرم|submit\s+form)", text, re.I))
            return RouteDecision(
                intent="browser",
                action="browser_extract",
                arguments={"url": url, "allow_download": want_dl, "submit_form": want_form},
                requires_confirmation=(want_dl or want_form),
                reason="درخواست استخراج لینک وب",
                router_stage="deterministic",
            )

        # 4. Explicit Web Search
        s_match = SEARCH_PREFIX_RE.match(text)
        if s_match:
            return RouteDecision(
                intent="search",
                action="web_search",
                arguments={"query": s_match.group(1).strip(), "max_results": 5, "fetch_top_page": False},
                reason="درخواست جستجوی اینترنتی",
                router_stage="deterministic",
            )

        # 5. Word (.docx) Operations
        docx_match = re.search(r"([\w\u0600-\u06FF.\-/\\]+\.docx)", text, re.I)
        if docx_match:
            fname = docx_match.group(1)
            if re.search(r"(بخوان|خواندن|محتوا|باز\s*کن|read|show)", text, re.I) and not re.search(r"(بساز|ایجاد|اضافه|ویرایش|جایگزین|create|append|replace)", text, re.I):
                return RouteDecision(
                    intent="word",
                    action="word_read",
                    arguments={"path": fname},
                    reason="خواندن فایل Word",
                    router_stage="deterministic",
                )
            if re.search(r"(بساز|ایجاد|جدید|create)", text, re.I):
                body = text.split(":", 1)[1].strip() if ":" in text else "ایجاد شده توسط LUNA AI"
                return RouteDecision(
                    intent="word",
                    action="word_create",
                    arguments={"path": fname, "title": fname.replace(".docx", ""), "paragraphs": [body]},
                    reason="ایجاد فایل Word",
                    router_stage="deterministic",
                )
            if re.search(r"(اضافه|ویرایش|پاراگراف|append|edit)", text, re.I):
                body = text.split(":", 1)[1].strip() if ":" in text else text
                return RouteDecision(
                    intent="word",
                    action="word_edit",
                    arguments={"path": fname, "operation": "append_paragraph", "text": body},
                    reason="ویرایش فایل Word",
                    router_stage="deterministic",
                )

        # 6. Excel (.xlsx) Operations
        xlsx_match = re.search(r"([\w\u0600-\u06FF.\-/\\]+\.xlsx)", text, re.I)
        if xlsx_match:
            fname = xlsx_match.group(1)
            if re.search(r"(مرتب|سورت|sort)", text, re.I):
                return RouteDecision(
                    intent="excel",
                    action="excel_edit",
                    arguments={"path": fname, "operation": "sort_sheet", "sort_by_col": 1, "descending": False},
                    reason="مرتب‌سازی فایل Excel",
                    router_stage="deterministic",
                )
            cell_m = re.search(r"(?:سلول|cell)\s+([A-Za-z]{1,2}[1-9][0-9]{0,4})\s*(?:را|=|به|to)?\s*([^\s،,]+)", ascii_text, re.I)
            if cell_m:
                c_addr, c_val = cell_m.group(1).upper(), cell_m.group(2)
                if c_val.startswith("="):
                    return RouteDecision(
                        intent="excel",
                        action="excel_edit",
                        arguments={"path": fname, "operation": "set_formula", "cell": c_addr, "value": c_val},
                        reason="ثبت فرمول در Excel",
                        router_stage="deterministic",
                    )
                num_val: Any = float(c_val) if re.match(r"^-?\d+\.\d+$", c_val) else (int(c_val) if re.match(r"^-?\d+$", c_val) else c_val)
                return RouteDecision(
                    intent="excel",
                    action="excel_edit",
                    arguments={"path": fname, "operation": "write_cell", "cell": c_addr, "value": num_val},
                    reason="ویرایش سلول Excel",
                    router_stage="deterministic",
                )
            if re.search(r"(بساز|ایجاد|جدید|create)", text, re.I):
                return RouteDecision(
                    intent="excel",
                    action="excel_create",
                    arguments={"path": fname, "headers": ["عنوان", "مقدار"], "rows": []},
                    reason="ایجاد فایل Excel",
                    router_stage="deterministic",
                )
            return RouteDecision(
                intent="excel",
                action="excel_read",
                arguments={"path": fname},
                reason="خواندن فایل Excel",
                router_stage="deterministic",
            )

        # 7. Notepad & Story Writing Workflow
        if re.search(r"(نوت‌?پد|notepad)", text, re.I):
            txt_match = re.search(r"([\w\u0600-\u06FF\-]+\.(?:txt|md))", text, re.I)
            fname = txt_match.group(1) if txt_match else "luna_note.txt"
            if re.search(r"(داستان|قصه|مقاله|متن.*بنویس|story|write\s+a)", text, re.I) and ":" not in text:
                return RouteDecision(
                    intent="notepad",
                    action="notepad_story_flow",
                    arguments={"filename": fname, "topic": text},
                    reason="جریان چندمرحله‌ای نوشتن متن و باز کردن Notepad",
                    router_stage="deterministic",
                )
            content = text.split(":", 1)[1].strip() if ":" in text else text
            return RouteDecision(
                intent="notepad",
                action="notepad_write",
                arguments={"filename": fname, "content": content, "open_window": True},
                reason="ذخیره و باز کردن در Notepad",
                router_stage="deterministic",
            )

        # 8. File Delete / List / Read
        if re.search(r"(لیست\s+فایل|فهرست\s+فایل|چند\s+تا\s+فایل|list\s+files)", text, re.I):
            return RouteDecision(
                intent="file",
                action="file_list",
                arguments={"subdir": ""},
                reason="مشاهده فهرست فایل‌های فضای کاری",
                router_stage="deterministic",
            )
        file_m = re.search(r"([\w\u0600-\u06FF.\-/\\]+\.(?:txt|md|csv|json|log))", text, re.I)
        if file_m:
            fpath = file_m.group(1)
            if re.search(r"(حذف|پاک\s+کن|delete|remove)", text, re.I):
                return RouteDecision(
                    intent="file",
                    action="file_delete",
                    arguments={"path": fpath},
                    requires_confirmation=True,
                    reason="حذف فایل (نیازمند تأیید کاربر)",
                    router_stage="deterministic",
                )
            if re.search(r"(بخوان|خواندن|نمایش|read|open|show)", text, re.I):
                return RouteDecision(
                    intent="file",
                    action="file_read",
                    arguments={"path": fpath},
                    reason="خواندن فایل متنی",
                    router_stage="deterministic",
                )

        # 9. Mathematical Expressions (Pure or Natural Language Math)
        has_digits = bool(re.search(r"[0-9]", ascii_text))
        has_op = bool(re.search(r"[+\-*/%^×÷−]", ascii_text))
        if (has_digits and PURE_ARITHMETIC_RE.match(text) and has_op) or (has_digits and MATH_SIGNAL_RE.search(text)):
            # Check if user ALSO wants to save the result in a file (multi-step!)
            secondary: List[RouteDecision] = []
            save_m = re.search(r"(?:در\s+فایل|save\s+(?:in|to))\s+([\w\-]+\.txt)", text, re.I)
            expr_part = text
            if save_m:
                expr_part = text[: save_m.start()].strip()
                secondary.append(
                    RouteDecision(
                        intent="file",
                        action="file_write",
                        arguments={"path": save_m.group(1), "content": "{{LAST_TOOL_SUMMARY}}"},
                        reason="ذخیره نتیجه محاسبه در فایل",
                    )
                )
            return RouteDecision(
                intent="calculator",
                action="calc_evaluate",
                arguments={"expression": expr_part},
                reason="تشخیص عبارت محاسباتی دقیق",
                router_stage="deterministic",
                secondary_steps=secondary,
            )

        return None

    async def _route_with_llm(self, text: str, session_id: str) -> Optional[RouteDecision]:
        """Stage 3: Constrained JSON decoding via Ollama (`qwen3:0.6b`) with strict allowlist & schema validation."""
        tool_catalog = "\n".join(
            f"- {s['name']} (intent={s['intent']}): {s['description']} | required={s['schema'].get('required', [])}"
            for s in self.registry.list_specs()
        )
        router_prompt = (
            "You are LUNA's JSON Router. Map the user message to ONE valid action.\n"
            "If no tool is strictly needed, output intent='chat' and action='chat_reply' with arguments={}.\n"
            f"Allowed tools:\n{tool_catalog}"
        )
        llm_res = await model_manager.chat(
            messages=[
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": text},
            ],
            language="en",
            json_schema=ROUTER_JSON_SCHEMA,
            think=False,
            option_overrides={"temperature": 0.05, "num_predict": 160, "num_ctx": 1536},
            timeout=20.0,
        )
        if not llm_res.ok:
            return None

        parsed, err = ResponseValidator.parse_and_repair_json(llm_res.content)
        if parsed is None:
            audit_logger.log(
                event_type="router_malformed_json",
                actor="router_llm",
                status="halted",
                details={"error": err, "raw": llm_res.content[:200]},
                session_id=session_id,
            )
            return None

        action = str(parsed.get("action", "chat_reply")).strip()
        if action in ("chat_reply", "none", "chat"):
            return RouteDecision(
                intent="chat",
                action="chat_reply",
                reason=str(parsed.get("reason", "llm_chat")),
                router_stage="llm_constrained",
            )

        if not self.registry.is_allowed(action):
            audit_logger.log(
                event_type="router_unallowlisted_tool",
                actor="router_llm",
                status="blocked",
                details={"proposed_action": action},
                session_id=session_id,
            )
            return None

        spec = self.registry.get_spec(action)
        args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
        return RouteDecision(
            intent=spec.intent,  # type: ignore[arg-type]
            action=action,
            arguments=args,
            requires_confirmation=bool(parsed.get("requires_confirmation")) or spec.requires_confirmation,
            reason=str(parsed.get("reason", "llm_routed")),
            router_stage="llm_constrained",
        )
