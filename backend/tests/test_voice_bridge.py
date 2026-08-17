from urllib.parse import parse_qs, urlparse

import pytest

from copilot.voice.bridge import _audio, _event_name, _stt_url, _transcript, _tts_url, _voice_context
from copilot.voice.sarvam import SarvamRealtimeSettings, SarvamTTSSettings


def test_saaras_realtime_url_uses_fast_vad_pcm_configuration() -> None:
    url = _stt_url(SarvamRealtimeSettings(enabled=True, api_key="test-key"))
    parsed = parse_qs(urlparse(url).query)

    assert parsed["model"] == ["saaras:v3-realtime"]
    assert parsed["stream_type"] == ["fast"]
    assert parsed["endpointing"] == ["vad"]
    assert parsed["encoding"] == ["linear16"]
    assert parsed["sample_rate"] == ["16000"]
    assert parsed["silence_duration_ms"] == ["500"]


def test_bulbul_stream_url_enables_completion_events() -> None:
    url = _tts_url(SarvamTTSSettings(enabled=True, api_key="test-key"))
    parsed = parse_qs(urlparse(url).query)

    assert parsed["model"] == ["bulbul:v3"]
    assert parsed["send_completion_event"] == ["true"]


def test_voice_event_helpers_accept_sarvam_payload_shapes() -> None:
    assert _event_name({"event": "transcript.final"}) == "transcript.final"
    assert _transcript({"data": {"transcript": "The WAN light is blinking"}}) == "The WAN light is blinking"
    assert _audio({"data": {"audio": "pcm-base64"}}) == "pcm-base64"


def test_voice_context_requires_session_id() -> None:
    context = _voice_context({"type": "session.start", "session_id": "voice-1", "manufacturer": "HP", "model": "M404"})

    assert context.session_id == "voice-1"
    assert context.manufacturer == "HP"
    assert context.model == "M404"
    with pytest.raises(ValueError, match="session_id"):
        _voice_context({"type": "session.start"})
