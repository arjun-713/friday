"""Backend-only Sarvam voice bridge for a full-duplex troubleshooting turn."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from urllib.parse import urlencode
from uuid import uuid4

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


class VoiceSessionState(StrEnum):
    """Explicit full-duplex states shared with the frontend voice console."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTING = "interrupting"
    RECOVERING = "recovering"
    ERROR = "error"


def classify_interruption(text: str) -> str:
    """Deterministically classify a barge-in without an LLM call."""

    normalized = re.sub(r"[^a-z0-9\s]", " ", text.casefold())
    normalized = " ".join(normalized.split())
    if not normalized:
        return "acknowledgement"
    if any(word in normalized.split() for word in ("stop", "quiet", "halt", "cancel")):
        return "stop"
    if normalized.startswith(("actually ", "no ", "correction", "i meant", "wrong")) or "actually" in normalized:
        return "correction"
    if normalized.startswith(("repeat", "say again", "pardon", "what did")) or "repeat" in normalized:
        return "repeat"
    if (
        any(token in normalized for token in ("got it", "thanks", "thank you", "okay", "ok", "understood", "yes"))
        and len(normalized.split()) <= 4
    ):
        return "acknowledgement"
    if normalized.endswith("?") or normalized.startswith(
        ("what", "why", "how", "which", "where", "when", "can you", "could you")
    ):
        return "clarification"
    return "new_information"


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
        self._active_turn_id: str | None = None
        self._tts: ClientConnection | None = None
        self._tts_lock = asyncio.Lock()
        self._tts_warm_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._state: VoiceSessionState = VoiceSessionState.IDLE

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
                            if self._tts_warm_task is None:
                                self._tts_warm_task = asyncio.create_task(self._warm_tts())
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
            if self._tts_warm_task is not None:
                self._tts_warm_task.cancel()
                await asyncio.gather(self._tts_warm_task, return_exceptions=True)
                self._tts_warm_task = None
            await self._cancel_active_turn(client, notify=False)
            if self._tts is not None:
                await self._tts.close()

    async def _forward_stt(
        self,
        stt: ClientConnection,
        client: WebSocket,
        context: Callable[[], VoiceTurnContext | None],
    ) -> None:
        speech_active = False
        speech_confirmed = False
        speech_started_at: float | None = None
        first_partial_at: float | None = None

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
                # Saaras can emit VAD starts for keyboard clicks, fan noise, or
                # a short microphone transient. Do not interrupt Friday until
                # there is actual transcript evidence of a user utterance.
                speech_active = True
                speech_confirmed = False
                speech_started_at = perf_counter()
                first_partial_at = None
                logger.info("voice_stt_speech_start")
            elif event_name == "vad.speech_end":
                if speech_active:
                    logger.info(
                        "voice_stt_speech_end duration_ms=%.1f",
                        (perf_counter() - speech_started_at) * 1000 if speech_started_at else 0,
                    )
                    await self._send(client, {"type": "speech.end"})
            elif event_name == "transcript.partial" and _meaningful_transcript(transcript):
                if not speech_confirmed:
                    speech_confirmed = True
                    first_partial_at = perf_counter()
                    interruption = (
                        classify_interruption(transcript) if self._active_turn_id is not None else "new_information"
                    )
                    logger.info(
                        "voice_stt_first_partial chars=%d latency_ms=%.1f interruption=%s",
                        len(transcript),
                        (first_partial_at - speech_started_at) * 1000 if speech_started_at else 0,
                        interruption,
                    )
                    self._state = VoiceSessionState.INTERRUPTING
                    await self._cancel_active_turn(client)
                    self._state = VoiceSessionState.LISTENING
                    await self._send(client, {"type": "speech.start"})
                await self._send(client, {"type": "transcript.partial", "text": transcript})
            elif event_name == "transcript.final" and _meaningful_transcript(transcript):
                if not speech_confirmed:
                    logger.info("Ignoring final transcript without confirmed partial speech chars=%d", len(transcript))
                    speech_active = False
                    continue
                logger.info("Sarvam realtime STT final transcript received chars=%d", len(transcript))
                active_context = context()
                if active_context is not None:
                    speech_active = False
                    speech_confirmed = False
                    logger.info(
                        "voice_stt_final chars=%d speech_to_final_ms=%.1f partial_to_final_ms=%.1f",
                        len(transcript),
                        (perf_counter() - speech_started_at) * 1000 if speech_started_at else 0,
                        (perf_counter() - first_partial_at) * 1000 if first_partial_at else 0,
                    )
                    await self._cancel_active_turn(client, notify=False)
                    turn_id = uuid4().hex
                    self._active_turn_id = turn_id
                    await self._send(client, {"type": "transcript.final", "text": transcript, "turn_id": turn_id})
                    self._turn_task = asyncio.create_task(
                        self._answer_turn(client, active_context, transcript, turn_id)
                    )
            elif event_name == "error":
                logger.warning(
                    "Sarvam realtime STT error code=%s fatal=%s",
                    payload.get("code", "unknown"),
                    payload.get("is_fatal", False),
                )
                await self._send(client, {"type": "voice.error", "message": _error_message(payload)})

    async def _answer_turn(self, client: WebSocket, context: VoiceTurnContext, transcript: str, turn_id: str) -> None:
        started = perf_counter()
        self._state = VoiceSessionState.THINKING
        first_token = True
        speech_buffer = ""
        spoken_text = ""
        tts_queue: asyncio.Queue[str | None] | None = None
        tts_task: asyncio.Task[None] | None = None
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
                    piece = str(event.get("text", ""))
                    if first_token:
                        first_token = False
                        self._state = VoiceSessionState.SPEAKING
                        logger.info(
                            "voice_llm_first_token turn_id=%s latency_ms=%.1f",
                            turn_id,
                            (perf_counter() - started) * 1000,
                        )
                    await self._send(
                        client,
                        {"type": "assistant.token", "text": piece, "turn_id": turn_id},
                    )
                    spoken_text += piece
                    speech_buffer += piece
                    sentences, speech_buffer = _take_tts_sentences(speech_buffer)
                    for sentence in sentences:
                        if tts_queue is None:
                            tts_queue = asyncio.Queue()
                            tts_task = asyncio.create_task(self._run_tts_stream(client, turn_id, tts_queue))
                        await tts_queue.put(sentence)
                elif event_type == "retrieval":
                    retrieval = event.get("retrieval", {})
                    timings = retrieval.get("timings_ms", {}) if isinstance(retrieval, dict) else {}
                    logger.info(
                        "voice_retrieval_complete turn_id=%s latency_ms=%.1f timings=%s",
                        turn_id,
                        (perf_counter() - started) * 1000,
                        timings,
                    )
                    await self._send(
                        client,
                        {"type": "retrieval", "retrieval": retrieval, "turn_id": turn_id},
                    )
                elif event_type == "complete":
                    logger.info(
                        "voice_answer_complete turn_id=%s latency_ms=%.1f",
                        turn_id,
                        (perf_counter() - started) * 1000,
                    )
                    response = event.get("response", {})
                    await self._send(
                        client,
                        {"type": "assistant.complete", "response": response, "turn_id": turn_id},
                    )
                    if isinstance(response, dict) and response.get("status") == "ready":
                        if tts_queue is None and speech_buffer.strip():
                            tts_queue = asyncio.Queue()
                            tts_task = asyncio.create_task(self._run_tts_stream(client, turn_id, tts_queue))
                            await tts_queue.put(speech_buffer.strip())
                        elif tts_queue is not None and speech_buffer.strip():
                            await tts_queue.put(speech_buffer.strip())
                        if tts_queue is not None and tts_task is not None:
                            await tts_queue.put(None)
                            await tts_task
                        elif not spoken_text.strip():
                            await self._speak_step(client, response, turn_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("Voice answer turn failed (%s)", type(error).__name__)
            await self._send(client, {"type": "voice.error", "message": "Friday could not complete that check."})
        finally:
            if tts_task is not None and not tts_task.done():
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)

    async def _speak_step(self, client: WebSocket, response: dict[str, object], turn_id: str) -> None:
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
            if isinstance(step, dict):
                instruction = str(step.get("instruction", "")).split(" [", 1)[0].strip()
                question = str(step.get("question", "")).strip()
                text = " ".join(part for part in (instruction, question) if part)
            else:
                text = str(response.get("answer", "")).strip()
        if not text:
            return
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        task = asyncio.create_task(self._run_tts_stream(client, turn_id, queue))
        await queue.put(text)
        await queue.put(None)
        await task

    async def _run_tts_stream(
        self,
        client: WebSocket,
        turn_id: str,
        queue: asyncio.Queue[str | None],
    ) -> None:
        started = perf_counter()
        logger.info("voice_tts_start turn_id=%s", turn_id)
        connect_started = perf_counter()
        had_open_tts = self._tts is not None and self._tts.state.name == "OPEN"
        tts = await self._ensure_tts()
        logger.info(
            "voice_tts_connection_ready turn_id=%s latency_ms=%.1f reused=%s",
            turn_id,
            (perf_counter() - connect_started) * 1000,
            had_open_tts,
        )
        reader = asyncio.create_task(self._read_tts_audio(client, tts, turn_id, started))
        try:
            while True:
                text = await queue.get()
                if text is None:
                    await tts.send(json.dumps({"type": "flush"}))
                    break
                await tts.send(json.dumps({"type": "text", "data": {"text": text}}))
            await reader
        except asyncio.CancelledError:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            raise

    async def _read_tts_audio(
        self,
        client: WebSocket,
        tts: ClientConnection,
        turn_id: str,
        started: float,
    ) -> None:
        first_audio = True
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
                    if first_audio:
                        first_audio = False
                        logger.info(
                            "voice_tts_first_audio turn_id=%s latency_ms=%.1f bytes=%d",
                            turn_id,
                            (perf_counter() - started) * 1000,
                            len(audio),
                        )
                    await self._send(
                        client,
                        {
                            "type": "assistant.audio",
                            "audio": audio,
                            "sample_rate": self.tts_settings.sample_rate,
                            "turn_id": turn_id,
                        },
                    )
            elif event_type in {"event", "completion"}:
                logger.info(
                    "voice_tts_complete turn_id=%s latency_ms=%.1f",
                    turn_id,
                    (perf_counter() - started) * 1000,
                )
                await self._send(client, {"type": "assistant.audio_complete", "turn_id": turn_id})
                return
            elif event_type == "error":
                await self._send(client, {"type": "voice.error", "message": _error_message(payload)})
                return

    async def _ensure_tts(self) -> ClientConnection:
        async with self._tts_lock:
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
                            "data": _tts_config(self.tts_settings),
                        }
                    )
                )
        return self._tts

    async def _warm_tts(self) -> None:
        """Open Bulbul while the user is speaking so the first turn is not cold."""

        started = perf_counter()
        try:
            await self._ensure_tts()
            logger.info("voice_tts_warm_complete latency_ms=%.1f", (perf_counter() - started) * 1000)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # The actual turn still owns the retry path; warming is best effort
            # and must never prevent the voice session from listening.
            logger.info("voice_tts_warm_failed type=%s", type(error).__name__)

    async def _cancel_active_turn(self, client: WebSocket, *, notify: bool = True) -> None:
        task = self._turn_task
        turn_id = self._active_turn_id
        self._turn_task = None
        self._active_turn_id = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._tts is not None:
                await self._tts.close()
                self._tts = None
        if notify:
            payload: dict[str, object] = {"type": "assistant.cancelled"}
            if turn_id is not None:
                payload["turn_id"] = turn_id
            await self._send(client, payload)

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


def _tts_config(settings: SarvamTTSSettings) -> dict[str, object]:
    """Build the Bulbul v3 config using the public WebSocket field names."""

    return {
        "language_code": settings.language,
        "speaker": settings.speaker,
        "pace": settings.pace,
        "speech_sample_rate": settings.sample_rate,
        "output_audio_codec": settings.codec,
        "min_buffer_size": 30,
        "max_chunk_length": 200,
    }


def _take_tts_sentences(buffer: str, *, final: bool = False) -> tuple[list[str], str]:
    """Release complete speech units without waiting for the whole answer."""

    sentences: list[str] = []
    remaining = buffer
    while remaining:
        match = re.search(r"[.!?](?:\s+|$)|\n+", remaining)
        if match is None:
            break
        candidate = remaining[: match.end()].strip()
        if candidate:
            sentences.append(candidate)
        remaining = remaining[match.end() :].lstrip()

    if not final and len(remaining) > 140:
        split_at = remaining.rfind(" ", 0, 140)
        if split_at > 30:
            sentences.append(remaining[:split_at].strip())
            remaining = remaining[split_at + 1 :].lstrip()
    elif final and remaining.strip():
        sentences.append(remaining.strip())
        remaining = ""
    return sentences, remaining


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


def _meaningful_transcript(text: str) -> bool:
    """Require enough spoken content to reject noise-only STT events."""

    return sum(character.isalnum() for character in text) >= 3


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
