# Voice latency optimization

Friday measures the user-visible path as:

```text
speech end → final transcript → retrieval → provider request → first token
→ first sentence → first actionable sentence → first TTS audio → completion
```

The important production metric is speech-end-to-first-audio. Individual model
latency is useful for diagnosis, but it is not the experience the user feels.
The bridge logs each stage with `turn_id` and a millisecond duration:

```text
voice_stt_final
voice_retrieval_complete
voice_llm_first_token
voice_first_sentence
voice_first_actionable_sentence
voice_tts_connection_ready
voice_tts_first_audio
```

`voice_first_sentence` fires when the first TTS sentence unit is queued;
`voice_first_actionable_sentence` fires for the first unit that asks the user
something (ends with `?`) or opens with an imperative troubleshooting verb
(`_is_actionable_sentence`, deterministic by design). Acknowledgements
("Your phone is connected…") are speakable but not actionable, which is why
both milestones are tracked: most replies acknowledge before instructing.

Provider-side spans come from per-call `LLMCallRecord`s attached to every
`complete` SSE event as `llm_calls`: `ttft_ms` isolates queue/scheduling delay
from generation speed (`tokens_per_sec`, computed over post-TTFT time from
provider usage). Measured on Luna: TTFT alone swings 0.8–4.1 s call to call,
so TTFT variance — not generation speed or application overhead — dominates
end-to-end variance. Streaming the planner call does not reduce TTFT, but it
removes the wait after it: first sentence follows TTFT by ~100–300 ms.

## Implemented decisions

- Keep one Saaras realtime WebSocket per voice session.
- Warm the Bulbul WebSocket after `session.start`, while the user is speaking.
- Start Bulbul when the first complete sentence arrives from the streamed LLM.
- Feed later sentence-sized text units over the same TTS connection and flush
  only after the LLM stream completes.
- Use Bulbul's documented 30-character minimum buffer rather than waiting for
  a larger first buffer.
- Cancel the TTS reader, active LLM task, and playback queue on interruption.
- Keep provider prompt caching opt-in and provider-aware. Unsupported providers
  receive no cache-specific request fields.

## Prompt-cache behavior

The conversation prompt deliberately puts the stable system instructions first
and the changing user/evidence content after them. When the selected provider
supports it, Friday adds the documented LiteLLM cache routing hint or content
cache marker. Cache usage is accepted only as real when the provider reports
`cached_tokens` (or its equivalent) in the response usage object.

The current default model is routed through Groq. LiteLLM does not document
Groq prompt caching in its supported-provider list, so Friday logs this route
as `prompt_cache=unsupported` and sends a normal request. This is intentional:
adding undocumented fields could break the live request. Switching to a
provider with documented caching enables the existing configuration without a
retrieval or voice rewrite.

## Evidence used for the design

- [Deepgram endpointing and interim results](https://developers.deepgram.com/docs/understand-endpointing-interim-results)
  separates partial transcript delivery from end-of-turn detection and
  recommends measuring the complete utterance rather than treating every
  partial result as a new request.
- [Deepgram's voice-agent latency report](https://developers.deepgram.com/docs/voice-agent-latency-report)
  breaks the path into STT, LLM time-to-first-text, TTS time-to-first-audio,
  and total latency.
- [Sarvam TTS WebSocket streaming](https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/streaming-api/web-socket)
  recommends a persistent connection, progressive audio, strategic flushes,
  and a small `min_buffer_size` for streaming.
- [Sarvam's TTS WebSocket reference](https://docs.sarvam.ai/api-reference/text-to-speech/stream)
  confirms that Bulbul v3 uses 24 kHz by default and exposes completion events.
- [LiteLLM prompt caching](https://docs.litellm.ai/docs/completion/prompt_caching)
  documents provider-specific cache controls and requires checking usage
  fields because caching can be silently skipped below provider thresholds.

## What is not claimed yet

The code change reduces avoidable application waiting, but it cannot guarantee
500 ms for every network, STT, provider, or speech length. Real provider
measurements must be collected over representative turns and reported as
P50/P95/P99/max for:

```text
speech-end → final transcript
final transcript → retrieval complete
retrieval complete → first LLM text
first LLM text → first TTS audio
speech-end → first TTS audio
```

Only after those measurements should endpointing, model, region, or audio
chunk sizes be changed. Retrieval tuning remains a separate optimization pass.
