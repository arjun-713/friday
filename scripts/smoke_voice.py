"""Smoke-test the local full-duplex voice bridge with a 16 kHz PCM file."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

from websockets.asyncio.client import connect


async def run(audio_path: Path, url: str, session_id: str) -> int:
    audio = audio_path.read_bytes()
    if not audio:
        raise ValueError("audio file is empty")
    events: list[str] = []

    async with connect(url, open_timeout=15, close_timeout=5) as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "session.start",
                    "session_id": session_id,
                    "manufacturer": "HP",
                    "model": "LaserJet Pro M404/M405",
                }
            )
        )

        async def send_audio() -> None:
            chunk_size = 1600  # 50 ms of mono 16-bit PCM at 16 kHz.
            for offset in range(0, len(audio), chunk_size):
                chunk = audio[offset : offset + chunk_size]
                await socket.send(json.dumps({"type": "audio", "audio": base64.b64encode(chunk).decode("ascii")}))
                await asyncio.sleep(0.05)

        async def receive_events() -> None:
            deadline = asyncio.get_running_loop().time() + 90
            while asyncio.get_running_loop().time() < deadline:
                remaining = deadline - asyncio.get_running_loop().time()
                raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
                payload = json.loads(raw)
                event_type = str(payload.get("type", ""))
                if event_type:
                    events.append(event_type)
                if {"assistant.complete", "assistant.audio"} <= set(events):
                    return
                if event_type == "voice.error":
                    raise RuntimeError(str(payload.get("message", "voice bridge error")))
            raise TimeoutError("voice smoke timed out waiting for an assistant response and audio")

        sender = asyncio.create_task(send_audio())
        receiver = asyncio.create_task(receive_events())
        await sender
        await receiver

    print(json.dumps({"events": events, "received_assistant_response": "assistant.complete" in events, "received_audio": "assistant.audio" in events}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="mono 16-bit little-endian PCM at 16 kHz")
    parser.add_argument("--url", default="ws://localhost:8000/v1/voice")
    parser.add_argument("--session-id", default="smoke-voice")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args.audio, args.url, args.session_id))
    except (OSError, ValueError, RuntimeError, TimeoutError, json.JSONDecodeError) as error:
        print(f"voice smoke failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
