"""Controlled Playwright Browser Extractor (`tools/browser.py`).

Enforces:
- Strict URL scheme (`http`/`https`) and SSRF/private-network blocking (`localhost`, `file://`, private IPs)
- Timeout and character caps
- CAPTCHA / Cloudflare Turnstile detection -> immediate halt and honest notification
- Download & form submission blocking without explicit confirmation
- Untrusted boundary wrapping & Prompt Injection neutralization
"""
from __future__ import annotations

import ipaddress
import socket
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from config import config_manager
from security.validator import PromptInjectionGuard

CAPTCHA_INDICATORS = (
    "captcha",
    "g-recaptcha",
    "cf-turnstile",
    "verify you are human",
    "unusual traffic from your computer network",
    "/sorry/index",
    "attention required! | cloudflare",
)

TRUSTED_HIGH_DOMAINS = (
    "wikipedia.org",
    "python.org",
    "github.com",
    "docs.microsoft.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "arxiv.org",
    "stackoverflow.com",
    "pypi.org",
    "fastapi.tiangolo.com",
)


def score_domain_credibility(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.endswith((".edu", ".gov", ".ac.ir")) or any(host == d or host.endswith("." + d) for d in TRUSTED_HIGH_DOMAINS):
        return {"tier": "high", "score": 0.9}
    if parsed.scheme == "https" and "." in host:
        return {"tier": "medium", "score": 0.65}
    return {"tier": "low", "score": 0.35}


def validate_external_url(url: str, resolve_dns: bool = True) -> str:
    raw = (url or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise PermissionError(f"پروتکل URL غیرمجاز است ({parsed.scheme or 'none'})؛ فقط http/https مجاز است.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise PermissionError("دامنه URL معتبر نیست.")
    if host in ("localhost", "0.0.0.0") or host.endswith(".local"):
        raise PermissionError("دسترسی مرورگر به آدرس‌های داخلی (localhost) مسدود است.")

    # Check literal IP first
    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
            raise PermissionError("دسترسی مرورگر به شبکه خصوصی/داخلی مسدود است (SSRF Guard).")
    except ValueError:
        if resolve_dns:
            try:
                infos = socket.getaddrinfo(host, None)
                for info in infos:
                    ip_str = info[4][0]
                    ip_obj = ipaddress.ip_address(ip_str)
                    if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                        raise PermissionError("دامنه به آدرس IP داخلی اشاره می‌کند و مسدود شد.")
            except socket.gaierror as ge:
                raise ConnectionError(f"عدم دسترسی به اینترنت یا خطای DNS برای دامنه {host}: {ge}") from ge

    return raw


class BrowserTool:
    """Safe Playwright browser reader with CAPTCHA halt and prompt-injection isolation."""

    async def extract_page(
        self,
        url: str,
        timeout_ms: int = 12000,
        allow_download: bool = False,
        submit_form: bool = False,
    ) -> Dict[str, Any]:
        if allow_download or submit_form:
            return {
                "ok": False,
                "requires_confirmation": True,
                "error": "ارسال فرم یا دانلود فایل ناشناس نیازمند تأیید صریح کاربر است.",
            }

        try:
            safe_url = validate_external_url(url, resolve_dns=True)
        except PermissionError as pe:
            return {"ok": False, "blocked": True, "error": str(pe)}
        except ConnectionError as ce:
            return {"ok": False, "offline": True, "error": str(ce)}

        accessed_at = datetime.now(timezone.utc).isoformat()
        max_chars = config_manager.settings.security.max_web_chars

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                context = await browser.new_context(
                    accept_downloads=False,
                    java_script_enabled=True,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) LUNA-SafeReader/1.0",
                )
                page = await context.new_page()
                await page.goto(safe_url, wait_until="domcontentloaded", timeout=timeout_ms)
                final_url = page.url
                # Re-validate final redirected URL against SSRF
                validate_external_url(final_url, resolve_dns=False)
                title = (await page.title()) or ""
                html = await page.content()
                await browser.close()

            lower_probe = (title + "\n" + html[:6000]).lower()
            if any(ind in lower_probe for ind in CAPTCHA_INDICATORS):
                return {
                    "ok": False,
                    "captcha_detected": True,
                    "url": final_url,
                    "accessed_at": accessed_at,
                    "error": "صفحه مقصد کپچا (CAPTCHA) نمایش داد؛ طبق قوانین امنیتی لونا متوقف شد و تلاش برای دور زدن انجام نداد.",
                }

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "iframe", "svg", "nav", "footer", "header"]):
                tag.decompose()

            links: List[Dict[str, str]] = []
            for a in soup.find_all("a", href=True)[:15]:
                href = a["href"].strip()
                txt = a.get_text(" ", strip=True)[:80]
                if href.startswith(("http://", "https://")) and txt:
                    links.append({"title": txt, "url": href})

            clean_text = soup.get_text(separator=" ", strip=True)
            wrapped_text, injection_findings = PromptInjectionGuard.sanitize_untrusted_content(
                clean_text, source_url=final_url, max_chars=max_chars
            )
            credibility = score_domain_credibility(final_url)

            return {
                "ok": True,
                "url": final_url,
                "title": title[:160],
                "accessed_at": accessed_at,
                "credibility": credibility,
                "injection_blocked": injection_findings,
                "untrusted_content": wrapped_text,
                "links": links[:10],
                "summary": f"صفحه «{title[:70] or final_url}» (اعتبار منبع: {credibility['tier']}) در تاریخ {accessed_at[:19]}Z استخراج شد.",
            }
        except Exception as exc:
            return {
                "ok": False,
                "url": url,
                "accessed_at": accessed_at,
                "error": f"خطا در مرورگر Playwright: {exc}",
            }


browser_tool = BrowserTool()
