"""Stage 2: Lightweight Lazy-Loaded Ollama Manager for `qwen3:0.6b`.

Features:
- Lazy initialization (zero model load at startup)
- Health check & installed local model discovery
- Separate Persian / English generation parameter profiles
- Qwen3 `think` / `no_think` control + automatic `<think>` stripping
- Synchronous/Async chat, Async token streaming, and JSON-Schema constrained decoding
- Strict timeouts and structured error reporting
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

import httpx

from config import GenerationProfile, config_manager
from core.hardware_optimizer import (
    build_hardware_options,
    inspect_ollama_processor_split,
    tame_ollama_priority,
)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking_blocks(text: str) -> str:
    """Remove `<think>...</think>` blocks so history and UI stay clean."""
    if not text:
        return ""
    cleaned = THINK_BLOCK_RE.sub("", text)
    # Also handle unclosed <think> if max_tokens truncated inside thought
    if "<think>" in cleaned.lower():
        idx = cleaned.lower().find("<think>")
        cleaned = cleaned[:idx]
    return cleaned.strip()


@dataclass
class LLMResponse:
    ok: bool
    content: str
    thinking: str = ""
    model: str = ""
    latency_ms: int = 0
    eval_count: int = 0
    prompt_eval_count: int = 0
    error: Optional[str] = None
    raw_meta: Dict[str, Any] = field(default_factory=dict)


class OllamaModelManager:
    """Manages local Ollama inference with lazy loading and strict single-model discipline."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self._base_url_override = base_url
        self._model_warmed = False
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_cache_ts: float = 0.0
        self._priority_tamed: bool = False

    @property
    def base_url(self) -> str:
        return (self._base_url_override or config_manager.settings.ollama_base_url).rstrip("/")

    @property
    def active_model(self) -> str:
        return config_manager.settings.active_model

    def _select_profile(self, language: Literal["fa", "en"]) -> GenerationProfile:
        s = config_manager.settings
        return s.fa_profile if language == "fa" else s.en_profile

    def _build_options(
        self,
        profile: GenerationProfile,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hw_opts = build_hardware_options(config_manager.settings.hardware)
        opts: Dict[str, Any] = {
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "top_k": profile.top_k,
            "repeat_penalty": profile.repeat_penalty,
            "presence_penalty": profile.presence_penalty,
            "num_predict": profile.max_tokens,
            "num_ctx": profile.num_ctx,
            **hw_opts,
        }
        if profile.stop:
            opts["stop"] = profile.stop
        if profile.seed is not None:
            opts["seed"] = profile.seed
        if overrides:
            opts.update(overrides)
        return opts

    async def health_check(self, force: bool = False) -> Dict[str, Any]:
        """Cached lightweight health check of Ollama service & processor split (avoids socket churn)."""
        now = time.monotonic()
        if not force and self._health_cache is not None and (now - self._health_cache_ts) < 30.0:
            return self._health_cache

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=3.5) as client:
                v_resp = await client.get(f"{self.base_url}/api/version")
                t_resp = await client.get(f"{self.base_url}/api/tags")
            proc_split = await inspect_ollama_processor_split(self.base_url)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if v_resp.status_code != 200:
                res = {
                    "online": False,
                    "base_url": self.base_url,
                    "active_model": self.active_model,
                    "model_available": False,
                    "installed_models": [],
                    "processor": proc_split,
                    "latency_ms": latency_ms,
                    "error": f"HTTP {v_resp.status_code}",
                }
                self._health_cache, self._health_cache_ts = res, now
                return res
            models_raw = t_resp.json().get("models", []) if t_resp.status_code == 200 else []
            installed = [m.get("name", "") for m in models_raw if m.get("name")]
            available = self.active_model in installed
            res = {
                "online": True,
                "version": v_resp.json().get("version"),
                "base_url": self.base_url,
                "active_model": self.active_model,
                "model_available": available,
                "model_warmed": self._model_warmed,
                "installed_models": installed,
                "processor": proc_split,
                "latency_ms": latency_ms,
                "error": None if available else f"مدل {self.active_model} یافت نشد.",
            }
            self._health_cache, self._health_cache_ts = res, now
            return res
        except Exception as exc:
            return {
                "online": False,
                "base_url": self.base_url,
                "active_model": self.active_model,
                "model_available": False,
                "model_warmed": False,
                "installed_models": [],
                "processor": {"loaded": False, "processor_summary": "آفلاین"},
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": str(exc),
            }

    async def chat(
        self,
        messages: List[Dict[str, str]],
        language: Literal["fa", "en"] = "fa",
        json_schema: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
        option_overrides: Optional[Dict[str, Any]] = None,
        timeout: float = 45.0,
        model_override: Optional[str] = None,
    ) -> LLMResponse:
        """Execute a single chat completion against local Ollama with full error handling."""
        profile = self._select_profile(language)
        model_name = model_override or self.active_model
        use_think = profile.think if think is None else think
        if json_schema is not None:
            # Never enable chain-of-thought during grammar-constrained JSON routing
            use_think = False

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "think": use_think,
            "keep_alive": "5m",
            "options": self._build_options(profile, option_overrides),
        }
        if json_schema is not None:
            payload["format"] = json_schema

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            # Fallback if an older Ollama version rejects `"think"` field
            if resp.status_code == 400 and "think" in resp.text.lower():
                payload.pop("think", None)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                latency_ms = int((time.perf_counter() - t0) * 1000)

            if resp.status_code != 200:
                return LLMResponse(
                    ok=False,
                    content="",
                    model=model_name,
                    latency_ms=latency_ms,
                    error=f"Ollama HTTP {resp.status_code}: {resp.text[:300]}",
                )

            data = resp.json()
            self._model_warmed = True
            if config_manager.settings.hardware.low_priority_ollama and not self._priority_tamed:
                tame_ollama_priority()
                self._priority_tamed = True
            msg = data.get("message", {}) or {}
            raw_content = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
            cleaned_content = strip_thinking_blocks(raw_content) if json_schema is None else raw_content.strip()

            return LLMResponse(
                ok=True,
                content=cleaned_content,
                thinking=thinking,
                model=model_name,
                latency_ms=latency_ms,
                eval_count=int(data.get("eval_count", 0) or 0),
                prompt_eval_count=int(data.get("prompt_eval_count", 0) or 0),
                raw_meta={"done_reason": data.get("done_reason")},
            )
        except httpx.TimeoutException:
            return LLMResponse(
                ok=False,
                content="",
                model=model_name,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"مهلت پاسخگویی مدل محلی ({timeout} ثانیه) به پایان رسید.",
            )
        except Exception as exc:
            return LLMResponse(
                ok=False,
                content="",
                model=model_name,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"خطای ارتباط با Ollama: {exc}",
            )

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        language: Literal["fa", "en"] = "fa",
        think: Optional[bool] = None,
        option_overrides: Optional[Dict[str, Any]] = None,
        timeout: float = 50.0,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream tokens from Ollama `/api/chat` with live degeneration guard."""
        profile = self._select_profile(language)
        use_think = profile.think if think is None else think
        payload: Dict[str, Any] = {
            "model": self.active_model,
            "messages": messages,
            "stream": True,
            "think": use_think,
            "keep_alive": "5m",
            "options": self._build_options(profile, option_overrides),
        }
        t0 = time.perf_counter()
        accumulated = ""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield {
                            "type": "error",
                            "error": f"Ollama HTTP {resp.status_code}: {body.decode('utf-8', 'replace')[:250]}",
                        }
                        return
                    self._model_warmed = True
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        msg = chunk.get("message", {}) or {}
                        token = msg.get("content", "") or ""
                        if token:
                            accumulated += token
                            # Live stream degeneration circuit-breaker: stop if last 6 words are identical
                            words = accumulated.split()
                            if len(words) >= 8 and len(set(words[-6:])) == 1:
                                yield {"type": "done", "content": strip_thinking_blocks(" ".join(words[:-5])), "aborted_loop": True}
                                return
                            yield {"type": "token", "token": token}
                        if chunk.get("done"):
                            yield {
                                "type": "done",
                                "content": strip_thinking_blocks(accumulated),
                                "latency_ms": int((time.perf_counter() - t0) * 1000),
                            }
                            return
        except Exception as exc:
            yield {"type": "error", "error": str(exc)}


model_manager = OllamaModelManager()
