import asyncio
import json
from collections.abc import AsyncIterator
from urllib.parse import parse_qs, urlparse

import pytest

from copilot.voice.bridge import (
    SarvamVoiceBridge,
    VoiceTurnContext,
    _audio,
    _event_name,
    _stt_url,
    _transcript,
    _tts_url,
    _voice_context,
)
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


class _VoiceClient:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def send_json(self, event: dict[str, object]) -> None:
        self.events.append(event)


class _SttEvents:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events

    def __aiter__(self) -> AsyncIterator[str]:
        return self._events()

    async def _events(self) -> AsyncIterator[str]:
        for event in self.events:
            yield json.dumps(event)


class _VoiceService:
    async def stream_answer(self, request) -> AsyncIterator[dict[str, object]]:
        assert request.query == "The WAN light is blinking"
        yield {"type": "retrieval", "retrieval": {"abstained": False, "timings_ms": {}}}
        yield {
            "type": "complete",
            "response": {
                "status": "ready",
                "step": {"instruction": "Check the WAN cable.", "question": "Is it seated?"},
            },
        }


def test_voice_bridge_forwards_final_transcript_then_answers_and_speaks() -> None:
    async def run() -> list[dict[str, object]]:
        bridge = SarvamVoiceBridge(_VoiceService())
        client = _VoiceClient()
        spoken: list[dict[str, object]] = []

        async def speak(_client, response: dict[str, object]) -> None:
            spoken.append(response)

        bridge._speak_step = speak  # type: ignore[method-assign]
        await bridge._forward_stt(
            _SttEvents(
                [
                    {"event": "vad.speech_start"},
                    {"event": "transcript.partial", "transcript": "The WAN light"},
                    {"event": "transcript.final", "data": {"transcript": "The WAN light is blinking"}},
                ]
            ),
            client,  # type: ignore[arg-type]
            lambda: VoiceTurnContext(session_id="voice-1", manufacturer="TP-Link", model="Archer C6"),
        )
        assert bridge._turn_task is not None
        await bridge._turn_task
        assert spoken and spoken[0]["status"] == "ready"
        return client.events

    events = asyncio.run(run())

    assert [event["type"] for event in events] == [
        "assistant.cancelled",
        "speech.start",
        "transcript.partial",
        "transcript.final",
        "retrieval",
        "assistant.complete",
    ]


def test_voice_bridge_cancellation_stops_an_active_answer_before_notifying_browser() -> None:
    async def run() -> tuple[bool, list[dict[str, object]]]:
        bridge = SarvamVoiceBridge(_VoiceService())
        client = _VoiceClient()
        cancelled = asyncio.Event()

        async def wait_forever() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        bridge._turn_task = asyncio.create_task(wait_forever())
        await asyncio.sleep(0)
        await bridge._cancel_active_turn(client)  # type: ignore[arg-type]
        return cancelled.is_set(), client.events

    cancelled, events = asyncio.run(run())

    assert cancelled
    assert events == [{"type": "assistant.cancelled"}]
