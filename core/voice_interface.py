"""Stage 12: Future Voice Architecture (Pluggable STT / TTS Interfaces & Adapters).

Per architectural specification:
- Does NOT load heavy STT/TTS models into RAM in v1 (preserving 8GB RAM headroom).
- Defines clean async contracts (`STTAdapter`, `TTSAdapter`, `VoiceRuntime`) so future
  lightweight CPU engines (Whisper INT8 ONNX, Vosk Persian, Piper TTS, or Browser WebSpeech)
  plug in without changing Agent Core or API routes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TranscriptionResult:
    ok: bool
    text: str
    language: str = "fa"
    confidence: float = 0.0
    provider: str = "disabled"
    error: str = ""


@dataclass
class SynthesisResult:
    ok: bool
    audio_bytes: bytes = b""
    mime_type: str = "audio/wav"
    provider: str = "disabled"
    error: str = ""


class STTAdapter(ABC):
    """Abstract Speech-to-Text interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str = "fa") -> TranscriptionResult:
        raise NotImplementedError


class TTSAdapter(ABC):
    """Abstract Text-to-Speech interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def synthesize(self, text: str, language: str = "fa") -> SynthesisResult:
        raise NotImplementedError


class DisabledSTTAdapter(STTAdapter):
    @property
    def name(self) -> str:
        return "disabled_v1_placeholder"

    async def transcribe(self, audio_bytes: bytes, language: str = "fa") -> TranscriptionResult:
        return TranscriptionResult(
            ok=False,
            text="",
            language=language,
            provider=self.name,
            error="ماژول STT محلی (مانند Whisper-ONNX یا Vosk) در نسخه ۱ غیرفعال است تا مصرف رم پایین بماند.",
        )


class DisabledTTSAdapter(TTSAdapter):
    @property
    def name(self) -> str:
        return "disabled_v1_placeholder"

    async def synthesize(self, text: str, language: str = "fa") -> SynthesisResult:
        return SynthesisResult(
            ok=False,
            provider=self.name,
            error="ماژول TTS محلی (مانند Piper TTS فارسی) در نسخه ۱ غیرفعال است؛ رابط آداپتور آماده اتصال است.",
        )


class VoiceRuntime:
    """Registry & coordinator for future voice adapters."""

    def __init__(self, stt: STTAdapter | None = None, tts: TTSAdapter | None = None) -> None:
        self.stt: STTAdapter = stt or DisabledSTTAdapter()
        self.tts: TTSAdapter = tts or DisabledTTSAdapter()

    def status(self) -> Dict[str, Any]:
        return {
            "stt_adapter": self.stt.name,
            "tts_adapter": self.tts.name,
            "ready_for_plugin": True,
            "supported_future_backends": ["whisper_int8_onnx", "vosk_fa", "piper_tts_fa", "browser_webspeech"],
        }


voice_runtime = VoiceRuntime()
