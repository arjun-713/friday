import pytest

from copilot.answering.litellm import LiteLLMSettings
from copilot.voice.sarvam import SarvamRealtimeSettings, SarvamTTSSettings


def test_sarvam_realtime_defaults() -> None:
    settings = SarvamRealtimeSettings()

    assert settings.model == "saaras:v3-realtime"
    assert settings.stream_type == "fast"
    assert settings.endpointing == "vad"
    assert settings.encoding == "linear16"
    assert settings.sample_rate == 16_000
    assert settings.vad_threshold == 0.3
    assert settings.silence_ms == 500
    assert settings.min_speech_ms == 250


def test_sarvam_realtime_requires_key_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    settings = SarvamRealtimeSettings(enabled=True)

    with pytest.raises(ValueError, match="SARVAM_API_KEY"):
        settings.require_credentials()


def test_sarvam_tts_defaults() -> None:
    settings = SarvamTTSSettings()

    assert settings.model == "bulbul:v3"
    assert settings.language == "en-IN"
    assert settings.speaker == "manan"
    assert settings.pace == 1.0
    assert settings.sample_rate == 24_000
    assert settings.codec == "linear16"
    assert settings.send_completion_event is True


def test_sarvam_tts_requires_key_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    settings = SarvamTTSSettings(enabled=True)

    with pytest.raises(ValueError, match="SARVAM_API_KEY"):
        settings.require_credentials()


def test_runtime_yaml_contains_non_secret_provider_settings() -> None:
    llm = LiteLLMSettings.from_env()
    stt = SarvamRealtimeSettings.from_env()
    tts = SarvamTTSSettings.from_env()

    assert llm.model == "openai/sarvam-105b-conversations"
    assert llm.api_base == "https://api.sarvam.ai/v1"
    assert llm.api_key_env == "SARVAM_API_KEY"
    assert stt.model == "saaras:v3-realtime"
    assert tts.model == "bulbul:v3"
