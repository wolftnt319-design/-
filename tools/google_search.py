"""Controlled Web Search & Self-Learning Harvester (`tools/google_search.py`).

Uses zero paid APIs:
1. Primary lightweight HTML search via DuckDuckGo HTML endpoint
2. Fallback Playwright search via Google/DuckDuckGo (detecting CAPTCHA cleanly and halting if triggered)
3. Filters unsafe links, scores domain credibility, extracts sanitized untrusted snippets,
   attaches UTC access timestamps, and prepares optional user-confirmed memory notes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from config import config_manager
from security.validator import PromptInjectionGuard
from tools.browser import CAPTCHA_INDICATORS, browser_tool, score_domain_credibility, validate_external_url


def _unwrap_ddg_url(raw_href: str) -> str:
    if "uddg=" in raw_href:
        parsed = urlparse(raw_href)
        qs = parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return unquote(qs["uddg"][0])
    if raw_href.startswith("//"):
        return "https:" + raw_href
    return raw_href


class WebSearchTool:
    """Controlled, low-footprint web search with untrusted data isolation."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        fetch_top_page: bool = False,
    ) -> Dict[str, Any]:
        q = (query or "").strip()
        if q.lower().startswith("/google"):
            q = q[7:].strip()
        if not q:
            return {"ok": False, "error": "عبارت جستجو خالی است."}

        accessed_at = datetime.now(timezone.utc).isoformat()
        max_results = max(1, min(8, int(max_results)))

        # Step 1: Lightweight DuckDuckGo HTML search
        results: List[Dict[str, Any]] = []
        captcha_hit = False
        net_error = None

        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            ) as client:
                resp = await client.post("https://html.duckduckgo.com/html/", data={"q": q})
            html = resp.text
            if any(ind in html.lower() for ind in CAPTCHA_INDICATORS) and "result__a" not in html:
                captcha_hit = True
            else:
                soup = BeautifulSoup(html, "html.parser")
                for block in soup.select(".result"):
                    a_tag = block.select_one(".result__a")
                    snip_tag = block.select_one(".result__snippet")
                    if not a_tag:
                        continue
                    url = _unwrap_ddg_url(a_tag.get("href", "").strip())
                    try:
                        validate_external_url(url, resolve_dns=False)
                    except PermissionError:
                        continue
                    title = a_tag.get_text(" ", strip=True)
                    raw_snippet = snip_tag.get_text(" ", strip=True) if snip_tag else ""
                    wrapped_snip, inj = PromptInjectionGuard.sanitize_untrusted_content(
                        raw_snippet, source_url=url, max_chars=450
                    )
                    cred = score_domain_credibility(url)
                    results.append({
                        "title": title[:140],
                        "url": url,
                        "snippet": raw_snippet[:350],
                        "untrusted_wrapped": wrapped_snip,
                        "credibility": cred,
                        "injection_blocked": inj,
                        "accessed_at": accessed_at,
                    })
                    if len(results) >= max_results:
                        break
        except httpx.RequestError as rexc:
            net_error = f"عدم دسترسی به اینترنت در جستجوی وب: {rexc}"
        except Exception as exc:
            net_error = str(exc)

        # Step 2: Playwright Google fallback if DDG returned nothing and internet wasn't completely down
        if not results and net_error is None and not captcha_hit:
            pw_res = await self._playwright_google_fallback(q, max_results=max_results, accessed_at=accessed_at)
            if pw_res.get("captcha_detected"):
                return pw_res
            results = pw_res.get("results", [])

        if captcha_hit and not results:
            return {
                "ok": False,
                "captcha_detected": True,
                "query": q,
                "accessed_at": accessed_at,
                "error": "موتور جستجو درخواست CAPTCHA نمایش داد؛ عملیات طبق اصول امنیتی متوقف شد.",
            }

        if not results:
            return {
                "ok": False,
                "offline": net_error is not None,
                "query": q,
                "accessed_at": accessed_at,
                "error": net_error or "نتیجه معتبری برای این جستجو یافت نشد.",
            }

        # Sort high credibility sources slightly higher while preserving relevance
        results.sort(key=lambda r: r["credibility"]["score"], reverse=True)

        extracted_page = None
        if fetch_top_page and results:
            top_url = results[0]["url"]
            page_res = await browser_tool.extract_page(top_url, timeout_ms=10000)
            if page_res.get("ok"):
                extracted_page = {
                    "url": page_res["url"],
                    "title": page_res["title"],
                    "credibility": page_res["credibility"],
                    "untrusted_content": page_res["untrusted_content"][: config_manager.settings.security.max_web_chars],
                }

        top = results[0]
        summary_lines = [
            f"{idx + 1}. {r['title']} ({r['credibility']['tier']}) — {r['snippet']} [لینک: {r['url']}]"
            for idx, r in enumerate(results[:4])
        ]
        return {
            "ok": True,
            "query": q,
            "accessed_at": accessed_at,
            "count": len(results),
            "results": results,
            "top_page": extracted_page,
            "suggested_memory_note": {
                "category": "web_note",
                "title": f"نکته وب: {q[:40]}",
                "content": f"{top['title']}: {top['snippet'][:220]} (منبع: {top['url']} | تاریخ دسترسی: {accessed_at[:10]})",
            },
            "summary": f"نتایج جستجوی وب برای «{q}» (تاریخ دسترسی: {accessed_at[:19]}Z):\n" + "\n".join(summary_lines),
        }

    async def _playwright_google_fallback(
        self, query: str, max_results: int, accessed_at: str
    ) -> Dict[str, Any]:
        try:
            from urllib.parse import quote_plus
            from playwright.async_api import async_playwright

            search_url = f"https://www.google.com/search?q={quote_plus(query)}&hl=fa"
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(search_url, wait_until="domcontentloaded", timeout=10000)
                cur_url = page.url
                html = await page.content()
                await browser.close()

            if "/sorry/" in cur_url or any(ind in html.lower() for ind in CAPTCHA_INDICATORS):
                return {
                    "ok": False,
                    "captcha_detected": True,
                    "query": query,
                    "accessed_at": accessed_at,
                    "error": "گوگل صفحه CAPTCHA نمایش داد؛ عملیات جستجوی مرورگر متوقف شد.",
                }

            soup = BeautifulSoup(html, "html.parser")
            out: List[Dict[str, Any]] = []
            for h3 in soup.find_all("h3"):
                a = h3.find_parent("a", href=True)
                if not a:
                    continue
                href = a["href"]
                if not href.startswith(("http://", "https://")):
                    continue
                try:
                    validate_external_url(href, resolve_dns=False)
                except PermissionError:
                    continue
                title = h3.get_text(" ", strip=True)
                wrapped, inj = PromptInjectionGuard.sanitize_untrusted_content(title, source_url=href, max_chars=350)
                out.append({
                    "title": title[:140],
                    "url": href,
                    "snippet": title[:200],
                    "untrusted_wrapped": wrapped,
                    "credibility": score_domain_credibility(href),
                    "injection_blocked": inj,
                    "accessed_at": accessed_at,
                })
                if len(out) >= max_results:
                    break
            return {"ok": True, "results": out}
        except Exception as exc:
            return {"ok": False, "results": [], "error": str(exc)}


web_search_tool = WebSearchTool()
