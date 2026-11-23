"""Stage 1 & 2 Tests: Environment Inspection, Real Ollama Health, and FA/EN Inference."""
from __future__ import annotations

import pytest

from core.env_checker import inspect_environment
from core.language import detect_language
from core.model_manager import OllamaModelManager
from core.response_validator import ResponseValidator


def test_environment_inspector_runs_truthfully():
    report = inspect_environment()
    assert "os" in report
    assert "python" in report
    assert "resources" in report
    assert report["sqlite_fts5"]["supported"] is True
    assert report["ollama"]["service_running"] is True
    assert report["ollama"]["target_model_ready"] is True


def test_language_detection_and_explicit_switch():
    lang_fa, sw_fa = detect_language("سلام چطوری؟ امروز هوا خوب است.")
    assert lang_fa == "fa" and sw_fa is None

    lang_en, sw_en = detect_language("Hello, how does SQLite WAL mode work?")
    assert lang_en == "en" and sw_en is None

    lang_sw, explicit = detect_language("لطفاً از این به بعد به زبان انگلیسی پاسخ بده")
    assert lang_sw == "en" and explicit == "en"


@pytest.mark.asyncio
async def test_ollama_lazy_health_and_english_chat():
    mgr = OllamaModelManager()
    assert mgr._model_warmed is False
    health = await mgr.health_check()
    assert health["online"] is True
    assert "qwen3:0.6b" in health["installed_models"]

    en_res = await mgr.chat(
        messages=[
            {"role": "system", "content": "You are LUNA. Reply in one short English sentence."},
            {"role": "user", "content": "What is 2 + 2 and what is your name?"},
        ],
        language="en",
        think=False,
        timeout=35.0,
    )
    assert en_res.ok is True
    assert len(en_res.content) > 5
    assert "<think>" not in en_res.content.lower()


@pytest.mark.asyncio
async def test_ollama_persian_chat_and_validator():
    mgr = OllamaModelManager()
    fa_res = await mgr.chat(
        messages=[
            {"role": "system", "content": "You are LUNA. Reply in short Persian using the provided fact."},
            {"role": "user", "content": "حقیقت: پایتخت ایران تهران است.\nسؤال: پایتخت ایران کجاست؟"},
        ],
        language="fa",
        think=False,
        timeout=35.0,
    )
    assert fa_res.ok is True
    assert "تهران" in fa_res.content
    val = ResponseValidator.validate_chat_response(fa_res.content, "پایتخت ایران کجاست؟", language="fa")
    assert val.hallucinated_action is False


@pytest.mark.asyncio
async def test_hardware_optimizer_hybrid_cpu_gpu_options():
    from config import HardwareConfig
    from core.hardware_optimizer import build_hardware_options, compute_safe_cpu_threads, inspect_ollama_processor_split

    safe_threads = compute_safe_cpu_threads()
    assert safe_threads >= 1

    hw = HardwareConfig(accelerator_mode="hybrid_cpu_gpu", cpu_threads=0, gpu_layers=14, num_batch=128)
    opts = build_hardware_options(hw)
    assert opts["num_gpu"] == 14
    assert opts["num_batch"] == 128
    assert opts["num_thread"] == safe_threads

    mgr = OllamaModelManager()
    proc_info = await inspect_ollama_processor_split(mgr.base_url)
    assert "processor_summary" in proc_info
