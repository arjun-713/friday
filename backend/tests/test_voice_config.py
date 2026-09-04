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
    assert settings.vad_threshold == 0.55
    assert settings.silence_ms == 1000
    assert settings.min_speech_ms == 400


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

    assert llm.model == "openai/gpt-5.6-luna"
    assert llm.api_base is None
    assert llm.api_key_env == "OPENAI_API_KEY"
    assert llm.reasoning_effort == "none"
    assert llm.response_format == "json_object"
    assert stt.model == "saaras:v3-realtime"
    assert tts.model == "bulbul:v3"
