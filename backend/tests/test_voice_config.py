import pytest

from copilot.voice.sarvam import SarvamRealtimeSettings


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
    monkeypatch.setenv("SARVAM_STT_ENABLED", "true")
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    settings = SarvamRealtimeSettings.from_env()

    with pytest.raises(ValueError, match="SARVAM_API_KEY"):
        settings.require_credentials()
