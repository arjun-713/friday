"""Backend-only Sarvam voice bridge for a full-duplex troubleshooting turn."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlencode

from fastapi import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import ClientConnection, connect

from ..answering.models import TroubleshootingRequest
from ..answering.service import TroubleshootingService
from .sarvam import SarvamRealtimeSettings, SarvamTTSSettings

JsonSender = Callable[[dict[str, object]], Awaitable[None]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceTurnContext:
    """Confirmed device scope for a browser voice session."""

    session_id: str
    manufacturer: str | None = None
    model: str | None = None


class SarvamVoiceBridge:
    """Proxy microphone, transcript, answer, and PCM audio without exposing keys."""

    def __init__(
        self,
        service: TroubleshootingService,
        stt_settings: SarvamRealtimeSettings | None = None,
        tts_settings: SarvamTTSSettings | None = None,
    ) -> None:
        self.service = service
        self.stt_settings = stt_settings or SarvamRealtimeSettings.from_env()
        self.tts_settings = tts_settings or SarvamTTSSettings.from_env()
        self._turn_task: asyncio.Task[None] | None = None
        self._tts: ClientConnection | None = None
        self._send_lock = asyncio.Lock()

    async def serve(self, client: WebSocket, *, accepted: bool = False) -> None:
        """Serve one browser voice socket until it closes; audio is never stored."""

        self.stt_settings.require_credentials()
        self.tts_settings.require_credentials()
        if not accepted:
            await client.accept()
        context: VoiceTurnContext | None = None
        try:
            async with connect(
                _stt_url(self.stt_settings),
                additional_headers={"api-subscription-key": self.stt_settings.api_key or ""},
                open_timeout=15,
                close_timeout=5,
            ) as stt:
                await self._send(client, {"type": "voice.ready", "sample_rate": self.stt_settings.sample_rate})
                stt_reader = asyncio.create_task(self._forward_stt(stt, client, lambda: context))
                try:
                    while True:
                        message = await client.receive_json()
                        event_type = str(message.get("type", ""))
                        if event_type == "session.start":
                            context = _voice_context(message)
                            await self._send(client, {"type": "session.ready", "session_id": context.session_id})
                        elif event_type == "audio":
                            if context is not None:
                                await stt.send(
                                    json.dumps({"event": "audio_input", "audio": str(message.get("audio", ""))})
                                )
                        elif event_type == "session.stop":
                            break
                        elif event_type == "assistant.cancel":
                            await self._cancel_active_turn(client)
                finally:
                    stt_reader.cancel()
                    await asyncio.gather(stt_reader, return_exceptions=True)
        except WebSocketDisconnect:
            pass
        except Exception as error:
            logger.warning("Voice session unavailable (%s)", type(error).__name__)
            await self._send(client, {"type": "voice.error", "message": "Voice service is temporarily unavailable."})
        finally:
            await self._cancel_active_turn(client, notify=False)
            if self._tts is not None:
                await self._tts.close()

    async def _forward_stt(
        self,
        stt: ClientConnection,
        client: WebSocket,
        context: Callable[[], VoiceTurnContext | None],
    ) -> None:
        async for raw_message in stt:
            if not isinstance(raw_message, str):
                continue
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                continue
            event_name = _event_name(payload)
            transcript = _transcript(payload)
            if event_name == "session.begin":
                logger.info("Sarvam realtime STT session began request_id=%s", _request_id(payload) or "unknown")
            if event_name == "vad.speech_start":
                await self._cancel_active_turn(client)
                await self._send(client, {"type": "speech.start"})
            elif event_name == "vad.speech_end":
                await self._send(client, {"type": "speech.end"})
            elif event_name == "transcript.partial" and transcript:
                await self._send(client, {"type": "transcript.partial", "text": transcript})
            elif event_name == "transcript.final" and transcript:
                logger.info("Sarvam realtime STT final transcript received chars=%d", len(transcript))
                await self._send(client, {"type": "transcript.final", "text": transcript})
                active_context = context()
                if active_context is not None:
                    await self._cancel_active_turn(client, notify=False)
                    self._turn_task = asyncio.create_task(self._answer_turn(client, active_context, transcript))
            elif event_name == "error":
                logger.warning(
                    "Sarvam realtime STT error code=%s fatal=%s",
                    payload.get("code", "unknown"),
                    payload.get("is_fatal", False),
                )
                await self._send(client, {"type": "voice.error", "message": _error_message(payload)})

    async def _answer_turn(self, client: WebSocket, context: VoiceTurnContext, transcript: str) -> None:
        request = TroubleshootingRequest(
            query=transcript,
            observation=transcript,
            session_id=context.session_id,
            manufacturer=context.manufacturer,
            model=context.model,
        )
        try:
            async for event in self.service.stream_answer(request):
                event_type = str(event.get("type", ""))
                if event_type == "token":
                    await self._send(client, {"type": "assistant.token", "text": str(event.get("text", ""))})
                elif event_type == "retrieval":
                    await self._send(client, {"type": "retrieval", "retrieval": event.get("retrieval", {})})
                elif event_type == "complete":
                    response = event.get("response", {})
                    await self._send(client, {"type": "assistant.complete", "response": response})
                    if isinstance(response, dict) and response.get("status") == "ready":
                        await self._speak_step(client, response)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("Voice answer turn failed (%s)", type(error).__name__)
            await self._send(client, {"type": "voice.error", "message": "Friday could not complete that check."})

    async def _speak_step(self, client: WebSocket, response: dict[str, object]) -> None:
        turn = response.get("turn")
        if isinstance(turn, dict):
            answer = str(turn.get("response", "")).strip()
            action = turn.get("next_action")
            action_text = str(action.get("instruction", "")).strip() if isinstance(action, dict) else ""
            request = turn.get("observation_request")
            question = str(request.get("question", "")).strip() if isinstance(request, dict) else ""
            text = " ".join(part for part in (answer, action_text, question) if part)
        else:
            step = response.get("step")
            if not isinstance(step, dict):
                return
            instruction = str(step.get("instruction", "")).split(" [", 1)[0].strip()
            question = str(step.get("question", "")).strip()
            text = " ".join(part for part in (instruction, question) if part)
        if not text:
            return
        tts = await self._ensure_tts()
        await tts.send(json.dumps({"type": "text", "data": {"text": text}}))
        await tts.send(json.dumps({"type": "flush"}))
        async for raw_message in tts:
            if not isinstance(raw_message, str):
                continue
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                continue
            event_type = str(payload.get("type", ""))
            if event_type == "audio":
                audio = _audio(payload)
                if audio:
                    await self._send(
                        client,
                        {"type": "assistant.audio", "audio": audio, "sample_rate": self.tts_settings.sample_rate},
                    )
            elif event_type in {"event", "completion"}:
                await self._send(client, {"type": "assistant.audio_complete"})
                return
            elif event_type == "error":
                await self._send(client, {"type": "voice.error", "message": _error_message(payload)})
                return

    async def _ensure_tts(self) -> ClientConnection:
        if self._tts is None or self._tts.state.name != "OPEN":
            self._tts = await connect(
                _tts_url(self.tts_settings),
                additional_headers={"api-subscription-key": self.tts_settings.api_key or ""},
                open_timeout=15,
                close_timeout=5,
            )
            await self._tts.send(
                json.dumps(
                    {
                        "type": "config",
                        "data": {
                            "target_language_code": self.tts_settings.language,
                            "speaker": self.tts_settings.speaker,
                            "pace": self.tts_settings.pace,
                            "speech_sample_rate": self.tts_settings.sample_rate,
                            "output_audio_codec": self.tts_settings.codec,
                        },
                    }
                )
            )
        return self._tts

    async def _cancel_active_turn(self, client: WebSocket, *, notify: bool = True) -> None:
        task = self._turn_task
        self._turn_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if notify:
            await self._send(client, {"type": "assistant.cancelled"})

    async def _send(self, client: WebSocket, payload: dict[str, object]) -> None:
        async with self._send_lock:
            await client.send_json(payload)


def _stt_url(settings: SarvamRealtimeSettings) -> str:
    params = {
        "language_code": settings.language,
        "model": settings.model,
        "stream_type": settings.stream_type,
        "mode": settings.mode,
        "endpointing": settings.endpointing,
        "encoding": settings.encoding,
        "sample_rate": str(settings.sample_rate),
        "threshold": str(settings.vad_threshold),
        "silence_duration_ms": str(settings.silence_ms),
        "min_speech_duration_ms": str(settings.min_speech_ms),
        "prompt": "Wi-Fi, WLAN, Ethernet, DHCP, DNS, BIOS, UEFI, ThinkPad, router, SSID, printer, toner, paper jam, HP, Brother, Epson, Canon.",
    }
    return f"{settings.endpoint}?{urlencode(params)}"


def _tts_url(settings: SarvamTTSSettings) -> str:
    params = {"model": settings.model, "send_completion_event": str(settings.send_completion_event).lower()}
    return f"{settings.endpoint}?{urlencode(params)}"


def _voice_context(message: dict[str, object]) -> VoiceTurnContext:
    session_id = str(message.get("session_id", "")).strip()
    if not session_id:
        raise ValueError("session.start requires a session_id")
    return VoiceTurnContext(
        session_id=session_id,
        manufacturer=_optional_text(message.get("manufacturer")),
        model=_optional_text(message.get("model")),
    )


def _event_name(payload: dict[str, object]) -> str:
    return str(payload.get("event") or payload.get("type") or "")


def _transcript(payload: dict[str, object]) -> str:
    # Realtime Saaras v3 events use ``text``.  The legacy endpoint used
    # ``transcript``; accepting both keeps the bridge compatible without
    # treating an unknown provider event as a user turn.
    value = payload.get("text") or payload.get("transcript")
    data = payload.get("data")
    if value is None and isinstance(data, dict):
        value = data.get("text") or data.get("transcript")
    return str(value or "").strip()


def _request_id(payload: dict[str, object]) -> str:
    value = payload.get("request_id")
    data = payload.get("data")
    if value is None and isinstance(data, dict):
        value = data.get("request_id")
    return str(value or "").strip()


def _audio(payload: dict[str, object]) -> str:
    data = payload.get("data")
    if isinstance(data, dict):
        return str(data.get("audio") or "")
    return str(payload.get("audio") or "")


def _error_message(payload: dict[str, object]) -> str:
    return str(payload.get("message") or payload.get("error") or "Voice service reported an error.")


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
