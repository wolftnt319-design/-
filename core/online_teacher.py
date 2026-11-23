"""Optional Free Online Teacher & Knowledge Distillation Engine.

Provides zero-cost (no paid API keys) assistance to help `qwen3:0.6b` deliver high-quality
Persian & English answers AND learn over time via local SQLite exemplar distillation:
1. `pollinations_free`: Free keyless OpenAI-compatible HTTP endpoint (with strict PII/secret scrubbing).
2. `custom_free_openai`: Optional user-supplied free OpenAI-compatible URL (e.g., local LAN server or free tier).
3. `lmarena_bridge`: Controlled Playwright helper + interactive UI distillation import for LMArena / web chats
   (strictly respects CAPTCHA/Turnstile — halts and notifies the user instead of bypassing).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import httpx

from config import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_FA, config_manager
from core.model_manager import strip_thinking_blocks
from security.validator import PromptInjectionGuard, SecretRedactor


@dataclass
class TeacherResult:
    ok: bool
    content: str
    provider: str
    latency_ms: int = 0
    error: Optional[str] = None
    captcha_detected: bool = False


class OnlineTeacher:
    """Privacy-preserving free online mentor & exemplar synthesizer for LUNA AI."""

    async def ask_teacher(
        self,
        user_message: str,
        language: Literal["fa", "en"] = "fa",
        context_snippets: Optional[List[str]] = None,
        local_draft: Optional[str] = None,
    ) -> TeacherResult:
        cfg = config_manager.settings.teacher
        if not cfg.enabled or config_manager.settings.operating_mode == "local_strict":
            return TeacherResult(
                ok=False,
                content="",
                provider="disabled",
                error="حالت معلم آنلاین غیرفعال است (Local Strict Mode).",
            )

        # Never transmit secrets, passwords, bank cards, or tokens externally!
        has_secret, secret_types = SecretRedactor.contains_secret(user_message)
        if has_secret:
            return TeacherResult(
                ok=False,
                content="",
                provider="privacy_guard",
                error=f"ارسال به معلم آنلاین به دلیل وجود داده حساس ({', '.join(secret_types)}) مسدود شد.",
            )

        safe_user_msg = SecretRedactor.redact(user_message)
        sys_prompt = SYSTEM_PROMPT_FA if language == "fa" else SYSTEM_PROMPT_EN

        context_block = ""
        if context_snippets:
            joined = "\n".join(f"- {SecretRedactor.redact(c)}" for c in context_snippets[:4])
            if language == "fa":
                context_block = f"\n\nاطلاعات تاییدشده محلی/ابزار (فقط به این داده‌ها استناد کن):\n{joined}"
            else:
                context_block = f"\n\nVerified local/tool context (rely strictly on these facts):\n{joined}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": sys_prompt + context_block},
            {"role": "user", "content": safe_user_msg},
        ]

        if cfg.provider == "custom_free_openai" and cfg.custom_base_url.strip():
            return await self._call_custom_openai(messages)
        if cfg.provider == "lmarena_bridge":
            return await self.probe_lmarena_bridge()
        return await self._call_pollinations(messages)

    async def _call_pollinations(self, messages: List[Dict[str, str]]) -> TeacherResult:
        cfg = config_manager.settings.teacher
        t0 = time.perf_counter()
        payload = {
            "model": "openai",
            "messages": messages,
            "temperature": 0.45,
        }
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout_seconds, follow_redirects=True) as client:
                resp = await client.post(cfg.pollinations_url, json=payload)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if resp.status_code != 200:
                return TeacherResult(
                    ok=False,
                    content="",
                    provider="pollinations_free",
                    latency_ms=latency_ms,
                    error=f"HTTP {resp.status_code}",
                )
            # Pollinations root POST returns plain text or OpenAI JSON depending on headers/route
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                data = resp.json()
                text = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if isinstance(data, dict)
                    else str(data)
                )
            else:
                text = resp.text
            text = strip_thinking_blocks(text).strip()
            # Guard against untrusted injection in external teacher response
            inj = PromptInjectionGuard.inspect_text(text)
            if inj["suspicious"] or not text:
                return TeacherResult(
                    ok=False,
                    content="",
                    provider="pollinations_free",
                    latency_ms=latency_ms,
                    error="پاسخ خارجی رد اعتبارسنجی امنیتی شد.",
                )
            return TeacherResult(
                ok=True,
                content=text,
                provider="pollinations_free",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            return TeacherResult(
                ok=False,
                content="",
                provider="pollinations_free",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=str(exc),
            )

    async def _call_custom_openai(self, messages: List[Dict[str, str]]) -> TeacherResult:
        cfg = config_manager.settings.teacher
        url = cfg.custom_base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
                resp = await client.post(
                    url,
                    json={"model": cfg.custom_model, "messages": messages, "temperature": 0.45},
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if resp.status_code != 200:
                return TeacherResult(ok=False, content="", provider="custom_free_openai", latency_ms=latency_ms, error=f"HTTP {resp.status_code}")
            data = resp.json()
            text = strip_thinking_blocks(data["choices"][0]["message"]["content"]).strip()
            return TeacherResult(ok=bool(text), content=text, provider="custom_free_openai", latency_ms=latency_ms)
        except Exception as exc:
            return TeacherResult(
                ok=False,
                content="",
                provider="custom_free_openai",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=str(exc),
            )

    async def probe_lmarena_bridge(self) -> TeacherResult:
        """Check LMArena reachability via Playwright; immediately halt and inform user on CAPTCHA/Turnstile."""
        t0 = time.perf_counter()
        url = config_manager.settings.teacher.lmarena_url
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                context = await browser.new_context(viewport={"width": 1280, "height": 800})
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=12000)
                title = (await page.title()) or ""
                html_lower = (await page.content()).lower()
                await browser.close()

            latency_ms = int((time.perf_counter() - t0) * 1000)
            if any(k in html_lower or k in title.lower() for k in ("just a moment", "cloudflare", "turnstile", "captcha", "verify you are human")):
                return TeacherResult(
                    ok=False,
                    content="",
                    provider="lmarena_bridge",
                    latency_ms=latency_ms,
                    captcha_detected=True,
                    error=(
                        "صفحه LMArena دارای محافظ CAPTCHA / Cloudflare Turnstile است. "
                        "طبق اصول امنیتی، لونا تلاش برای دور زدن کپچا انجام نمی‌دهد و متوقف شد. "
                        "می‌توانید پاسخ نمونه را در پنل «تقطیر دانش / LMArena» وارد کنید یا از معلم رایگان Pollinations استفاده کنید."
                    ),
                )
            return TeacherResult(
                ok=True,
                content=f"صفحه {url} در دسترس است (عنوان: {title}). برای مقایسه تعاملی یا وارد کردن پاسخ تاییدشده به حافظه آموزشی از پنل تقطیر دانش استفاده کنید.",
                provider="lmarena_bridge",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            return TeacherResult(
                ok=False,
                content="",
                provider="lmarena_bridge",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"عدم دسترسی به مرورگر LMArena: {exc}",
            )


online_teacher = OnlineTeacher()
