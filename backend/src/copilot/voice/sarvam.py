"""Configuration for Sarvam's realtime Saaras speech-to-text WebSocket."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..config import config_section, load_runtime_config


@dataclass(frozen=True)
class SarvamRealtimeSettings:
    """YAML-driven Saaras settings with no credential logging or persistence."""

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
        values = config_section(load_runtime_config(), "voice.stt")
        return cls(
            enabled=bool(values.get("enabled", False)),
            api_key=os.getenv("SARVAM_API_KEY") or None,
            model=str(values.get("model", "saaras:v3-realtime")),
            language=str(values.get("language", "en-IN")),
            mode=str(values.get("mode", "transcribe")),
            stream_type=str(values.get("stream_type", "fast")),
            endpointing=str(values.get("endpointing", "vad")),
            encoding=str(values.get("encoding", "linear16")),
            sample_rate=int(values.get("sample_rate", 16000)),
            vad_threshold=float(values.get("vad_threshold", 0.3)),
            silence_ms=int(values.get("silence_ms", 500)),
            min_speech_ms=int(values.get("min_speech_ms", 250)),
            endpoint=str(values.get("endpoint", "wss://api.sarvam.ai/speech-to-text-realtime/ws")),
        )

    def require_credentials(self) -> None:
        if self.enabled and not self.api_key:
            raise ValueError("SARVAM_API_KEY is required when voice.stt.enabled is true")


@dataclass(frozen=True)
class SarvamTTSSettings:
    """YAML-driven Bulbul v3 WebSocket settings."""

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
        values = config_section(load_runtime_config(), "voice.tts")
        return cls(
            enabled=bool(values.get("enabled", False)),
            api_key=os.getenv("SARVAM_API_KEY") or None,
            model=str(values.get("model", "bulbul:v3")),
            language=str(values.get("language", "en-IN")),
            speaker=str(values.get("speaker", "manan")),
            pace=float(values.get("pace", 1.0)),
            sample_rate=int(values.get("sample_rate", 24000)),
            codec=str(values.get("codec", "linear16")),
            send_completion_event=bool(values.get("send_completion_event", True)),
            endpoint=str(values.get("endpoint", "wss://api.sarvam.ai/text-to-speech/ws")),
        )

    def require_credentials(self) -> None:
        if self.enabled and not self.api_key:
            raise ValueError("SARVAM_API_KEY is required when voice.tts.enabled is true")
