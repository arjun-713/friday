from __future__ import annotations

import wave

import pytest
from scripts.benchmark_voice import Trial, percentile, read_pcm


def test_read_pcm_accepts_normalized_wav(tmp_path) -> None:
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 160)

    audio, metadata = read_pcm(path)

    assert len(audio) == 320
    assert metadata["channels"] == 1
    assert metadata["sample_rate"] == 16000


def test_read_pcm_rejects_non_normalized_wav(tmp_path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44100)
        output.writeframes(b"\x00\x00\x00\x00" * 16)

    with pytest.raises(ValueError, match="mono, 16-bit PCM at 16 kHz"):
        read_pcm(path)


def test_percentile_interpolates_small_samples() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def test_trial_stage_values_require_all_events() -> None:
    trial = Trial(trial=1, session_id="voice-test")
    assert trial.stage_values() == {}

    trial.speech_end_ms = 100.0
    trial.transcript_final_ms = 150.0
    trial.retrieval_ms = 180.0
    trial.first_token_ms = 300.0
    trial.first_audio_ms = 400.0
    trial.audio_complete_ms = 900.0

    assert trial.stage_values() == {
        "speech_end_to_transcript_final_ms": 50.0,
        "transcript_final_to_retrieval_ms": 30.0,
        "retrieval_to_first_token_ms": 120.0,
        "first_token_to_first_audio_ms": 100.0,
        "speech_end_to_first_audio_ms": 300.0,
        "first_audio_to_audio_complete_ms": 500.0,
    }
