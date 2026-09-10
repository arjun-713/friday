"""Measure end-to-end voice latency against the local Friday voice bridge.

The input is intentionally sent at real-time pace because a burst upload would
not exercise speech endpointing or the streaming path. The benchmark records
only timings and event presence; it never writes audio or transcripts.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from friday.paths import index_dir
from websockets.asyncio.client import ClientConnection, connect

CHUNK_BYTES = 1600  # 50 ms of mono, 16-bit PCM at 16 kHz.
CHUNK_SECONDS = 0.05
DEFAULT_TIMEOUT_SECONDS = 90.0


@dataclass
class Trial:
    trial: int
    session_id: str
    status: str = "failed"
    error: str | None = None
    speech_end_ms: float | None = None
    transcript_final_ms: float | None = None
    retrieval_ms: float | None = None
    first_token_ms: float | None = None
    first_audio_ms: float | None = None
    audio_complete_ms: float | None = None
    transcript_chars: int | None = None

    def stage_values(self) -> dict[str, float]:
        speech_end = self.speech_end_ms
        transcript_final = self.transcript_final_ms
        retrieval = self.retrieval_ms
        first_token = self.first_token_ms
        first_audio = self.first_audio_ms
        audio_complete = self.audio_complete_ms
        if any(
            value is None
            for value in (
                speech_end,
                transcript_final,
                retrieval,
                first_token,
                first_audio,
                audio_complete,
            )
        ):
            return {}
        assert speech_end is not None
        assert transcript_final is not None
        assert retrieval is not None
        assert first_token is not None
        assert first_audio is not None
        assert audio_complete is not None
        return {
            "speech_end_to_transcript_final_ms": transcript_final - speech_end,
            "transcript_final_to_retrieval_ms": retrieval - transcript_final,
            "retrieval_to_first_token_ms": first_token - retrieval,
            "first_token_to_first_audio_ms": first_audio - first_token,
            "speech_end_to_first_audio_ms": first_audio - speech_end,
            "first_audio_to_audio_complete_ms": audio_complete - first_audio,
        }


def read_pcm(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read raw PCM or validate and unwrap a WAV fixture."""

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as source:
            metadata = {
                "format": "wav",
                "channels": source.getnchannels(),
                "sample_width": source.getsampwidth(),
                "sample_rate": source.getframerate(),
                "frames": source.getnframes(),
            }
            if (
                metadata["channels"] != 1
                or metadata["sample_width"] != 2
                or metadata["sample_rate"] != 16000
            ):
                raise ValueError("WAV must be mono, 16-bit PCM at 16 kHz")
            return source.readframes(source.getnframes()), metadata

    audio = path.read_bytes()
    if not audio:
        raise ValueError("audio file is empty")
    if len(audio) % 2:
        raise ValueError("raw PCM must contain an even number of bytes")
    return audio, {
        "format": "raw_pcm",
        "channels": 1,
        "sample_width": 2,
        "sample_rate": 16000,
        "frames": len(audio) // 2,
    }


async def send_audio(socket: ClientConnection, audio: bytes, tail_ms: int) -> None:
    """Send microphone PCM at real-time pace, then add an in-memory silence tail."""

    tail_bytes = int(16000 * 2 * tail_ms / 1000)
    payload = audio + b"\x00" * tail_bytes
    for offset in range(0, len(payload), CHUNK_BYTES):
        chunk = payload[offset : offset + CHUNK_BYTES]
        await socket.send(
            json.dumps(
                {"type": "audio", "audio": base64.b64encode(chunk).decode("ascii")}
            )
        )
        await asyncio.sleep(CHUNK_SECONDS)


def _event_time(trial_start: float) -> float:
    return (perf_counter() - trial_start) * 1000


async def run_trial(
    audio: bytes,
    *,
    url: str,
    trial_number: int,
    manufacturer: str,
    model: str,
    tail_ms: int,
    timeout_seconds: float,
) -> Trial:
    session_id = f"voice-benchmark-{uuid.uuid4().hex}"
    result = Trial(trial=trial_number, session_id=session_id)
    trial_start = perf_counter()
    events: set[str] = set()

    try:
        async with connect(url, open_timeout=15, close_timeout=5) as socket:
            await socket.send(
                json.dumps(
                    {
                        "type": "session.start",
                        "session_id": session_id,
                        "manufacturer": manufacturer,
                        "model": model,
                    }
                )
            )
            sender = asyncio.create_task(send_audio(socket, audio, tail_ms))
            deadline = asyncio.get_running_loop().time() + timeout_seconds
            try:
                while asyncio.get_running_loop().time() < deadline:
                    remaining = deadline - asyncio.get_running_loop().time()
                    raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
                    payload = json.loads(raw)
                    event_type = str(payload.get("type", ""))
                    if event_type:
                        events.add(event_type)
                    elapsed = _event_time(trial_start)
                    if event_type == "speech.end" and result.speech_end_ms is None:
                        result.speech_end_ms = elapsed
                    elif (
                        event_type == "transcript.final"
                        and result.transcript_final_ms is None
                    ):
                        result.transcript_final_ms = elapsed
                        text = payload.get("text")
                        if isinstance(text, str):
                            result.transcript_chars = len(text)
                    elif event_type == "retrieval" and result.retrieval_ms is None:
                        result.retrieval_ms = elapsed
                    elif (
                        event_type == "assistant.token"
                        and result.first_token_ms is None
                    ):
                        result.first_token_ms = elapsed
                    elif (
                        event_type == "assistant.audio"
                        and result.first_audio_ms is None
                    ):
                        result.first_audio_ms = elapsed
                    elif (
                        event_type == "assistant.audio_complete"
                        and result.audio_complete_ms is None
                    ):
                        result.audio_complete_ms = elapsed
                    if {"assistant.complete", "assistant.audio_complete"} <= events:
                        break
                    if event_type == "voice.error":
                        raise RuntimeError(
                            str(payload.get("message", "voice bridge error"))
                        )
            finally:
                if not sender.done():
                    sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            await socket.send(json.dumps({"type": "session.stop"}))

        required = (
            result.speech_end_ms,
            result.transcript_final_ms,
            result.retrieval_ms,
            result.first_token_ms,
            result.first_audio_ms,
            result.audio_complete_ms,
        )
        if any(value is None for value in required):
            expected_events = {
                "speech.end",
                "transcript.final",
                "retrieval",
                "assistant.token",
                "assistant.audio",
                "assistant.audio_complete",
            }
            raise TimeoutError(
                f"missing voice events: {sorted(expected_events - events)}"
            )
        result.status = "ok"
    except (
        OSError,
        asyncio.TimeoutError,
        TimeoutError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        result.error = f"{type(error).__name__}: {error}"
    return result


def percentile(values: list[float], percentile_value: float) -> float:
    """Return an interpolated percentile, retaining useful decimals for small samples."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile without values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(samples: list[Trial]) -> dict[str, dict[str, float] | int]:
    stage_names = tuple(samples[0].stage_values()) if samples else ()
    report: dict[str, dict[str, float] | int] = {}
    for stage in stage_names:
        values = [
            sample.stage_values()[stage]
            for sample in samples
            if stage in sample.stage_values()
        ]
        report[stage] = {
            "count": len(values),
            "p50_ms": round(percentile(values, 50), 3),
            "p70_ms": round(percentile(values, 70), 3),
            "p95_ms": round(percentile(values, 95), 3),
            "p99_ms": round(percentile(values, 99), 3),
            "p100_ms": round(percentile(values, 100), 3),
            "max_ms": round(max(values), 3),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="16 kHz mono PCM or WAV fixture")
    parser.add_argument("--url", default="ws://localhost:8000/v1/voice")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--tail-ms",
        type=int,
        default=1500,
        help="in-memory silence after audio to allow endpointing",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--manufacturer", default="TP-Link")
    parser.add_argument("--model", default="Archer C6")
    parser.add_argument("--output", type=Path, default=index_dir() / "voice_latency_eval.json")
    return parser


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    audio, audio_metadata = read_pcm(args.audio)
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    trials: list[Trial] = []
    for trial_number in range(1, args.trials + 1):
        result = await run_trial(
            audio,
            url=args.url,
            trial_number=trial_number,
            manufacturer=args.manufacturer,
            model=args.model,
            tail_ms=args.tail_ms,
            timeout_seconds=args.timeout,
        )
        trials.append(result)
        if result.status == "ok":
            first_audio = result.stage_values().get("speech_end_to_first_audio_ms")
            print(
                f"trial {trial_number}/{args.trials}: ok first_audio={first_audio:.1f} ms"
            )
        else:
            print(
                f"trial {trial_number}/{args.trials}: failed {result.error}",
                file=sys.stderr,
            )

    successful = [trial for trial in trials if trial.status == "ok"]
    report: dict[str, Any] = {
        "benchmark": "voice_latency.v1",
        "url": args.url,
        "audio": {
            "path": str(args.audio),
            **audio_metadata,
            "duration_ms": round(len(audio) / 32, 3),
        },
        "configuration": {
            "trials_requested": args.trials,
            "trials_succeeded": len(successful),
            "trials_failed": len(trials) - len(successful),
            "tail_ms": args.tail_ms,
            "chunk_ms": 50,
            "timing_clock": "client perf_counter",
        },
        "latency_ms": summarize(successful),
        "trials": [
            asdict(trial) | {"stages": trial.stage_values()} for trial in trials
        ],
    }
    return report


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = asyncio.run(benchmark(args))
    except (OSError, ValueError) as error:
        print(f"voice benchmark failed: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"wrote {args.output}")
    return 0 if report["configuration"]["trials_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
