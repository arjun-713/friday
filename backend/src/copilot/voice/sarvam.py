"""Configuration for Sarvam's realtime Saaras speech-to-text WebSocket."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SarvamRealtimeSettings:
    """Environment-driven Saaras settings with no credential logging or persistence."""

    enabled: bool = False
    api_key: str | None = field(default=None, repr=False)
    model: str = "saaras:v3-realtime"
    language: str = "en-IN"
    mode: str = "transcribe"
    stream_type: str = "fast"
    endpointing: str = "vad"
    encoding: str = "linear16"
    sample_rate: int = 16_000
    vad_threshold: float = 0.3
    silence_ms: int = 500
    min_speech_ms: int = 250
    endpoint: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

    @classmethod
    def from_env(cls) -> SarvamRealtimeSettings:
        return cls(
            enabled=_env_bool("SARVAM_STT_ENABLED", default=False),
            api_key=os.getenv("SARVAM_API_KEY") or None,
            model=os.getenv("SARVAM_STT_MODEL", "saaras:v3-realtime"),
            language=os.getenv("SARVAM_STT_LANGUAGE", "en-IN"),
            mode=os.getenv("SARVAM_STT_MODE", "transcribe"),
            stream_type=os.getenv("SARVAM_STT_STREAM_TYPE", "fast"),
            endpointing=os.getenv("SARVAM_STT_ENDPOINTING", "vad"),
            encoding=os.getenv("SARVAM_STT_ENCODING", "linear16"),
            sample_rate=int(os.getenv("SARVAM_STT_SAMPLE_RATE", "16000")),
            vad_threshold=float(os.getenv("SARVAM_STT_VAD_THRESHOLD", "0.3")),
            silence_ms=int(os.getenv("SARVAM_STT_SILENCE_MS", "500")),
            min_speech_ms=int(os.getenv("SARVAM_STT_MIN_SPEECH_MS", "250")),
        )

    def require_credentials(self) -> None:
        if self.enabled and not self.api_key:
            raise ValueError("SARVAM_API_KEY is required when SARVAM_STT_ENABLED=true")


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SarvamTTSSettings:
    """Environment-driven Bulbul v3 WebSocket settings."""

    enabled: bool = False
    api_key: str | None = field(default=None, repr=False)
    model: str = "bulbul:v3"
    language: str = "en-IN"
    speaker: str = "manan"
    pace: float = 1.0
    sample_rate: int = 24_000
    codec: str = "linear16"
    send_completion_event: bool = True
    endpoint: str = "wss://api.sarvam.ai/text-to-speech/ws"

    @classmethod
    def from_env(cls) -> SarvamTTSSettings:
        return cls(
            enabled=_env_bool("SARVAM_TTS_ENABLED", default=False),
            api_key=os.getenv("SARVAM_API_KEY") or None,
            model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
            language=os.getenv("SARVAM_TTS_LANGUAGE", "en-IN"),
            speaker=os.getenv("SARVAM_TTS_SPEAKER", "manan"),
            pace=float(os.getenv("SARVAM_TTS_PACE", "1.0")),
            sample_rate=int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "24000")),
            codec=os.getenv("SARVAM_TTS_CODEC", "linear16"),
            send_completion_event=_env_bool("SARVAM_TTS_SEND_COMPLETION_EVENT", default=True),
        )

    def require_credentials(self) -> None:
        if self.enabled and not self.api_key:
            raise ValueError("SARVAM_API_KEY is required when SARVAM_TTS_ENABLED=true")
