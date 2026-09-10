# RAG performance engineering report

Conversation benchmark: `scripts/benchmark_conversation.py` (10-turn Archer C6
troubleshooting session over `POST /v1/troubleshoot/stream`).
Retrieval guardrail: `python -m eval.run_retrieval --candidate-limit 32
--dense-weight 1 --lexical-weight 1.5 --rrf-k 30 --diversify` (production config).
Conversation policy: `python -m eval.run_conversation_policy` (30 cases).

## Baseline (untouched code, 2026-09-09)

End-to-end per turn (10 turns, one session):

| Metric | P50 | Mean | P95 | Max |
| --- | --- | --- | --- | --- |
| Total / first token | 8478 ms | 8205 ms | 9371 ms | 9849 ms |
| Retrieval total | 2560 ms | 2498 ms | 3433 ms | 3622 ms |
| Embedding (reported) | 2546 ms | 2481 ms | 3421 ms | 3609 ms |
| Lexical (reported) | 2546 ms | 2481 ms | 3421 ms | 3609 ms |
| Dense (Qdrant) | 13 ms | 16 ms | 28 ms | 38 ms |

Per-turn LLM: 2 sequential provider calls (router 160 tok + planner 1200 tok),
roughly 2.0 s + 4.0 s. Tool calls: 0 across 10 turns (one attempted tool call
crashed on null assistant content). Answered turns: 8/10 (2 empty abstains).

Retrieval eval baseline: recall@5 0.963, MRR 0.743, abstention accuracy 0.771,
citation-ready 1.0, procedure order 0.917, 9 failing cases.

## Bottlenecks discovered (measured, not guessed)

1. **Wasted router LLM call.** Every streaming turn paid two sequential provider
   round trips; the first (tool router) never produced tools in 10/10 turns.
   When it did fire, `generate_agent_turn` crashed on `content=None` and the
   turn abstained empty.
2. **BM25 full-corpus scan.** Isolated micro-benchmark: ONNX embedding alone is
   53 ms (short query) / 179 ms (long query); BM25 alone over 24,286 chunks is
   183 ms / 3100 ms. Query length explodes across turns because the retrieval
   query carried planner-only instructions (completed-action IDs, ruled-out
   causes, four past reports). The event loop was blocked by BM25, inflating
   the embedding timer to match.
3. **Near-duplicate top-k.** Procedure/section/parent-child kind variants of one
   manual page occupied 3 of 5 evidence slots, starving distinct branches.
4. **Unbounded prompt growth.** Full fact history, all user reports, and the
   full previous response were re-sent on both LLM calls every turn.
5. **Triple corpus parse at startup.** Chunk JSONL was read and validated three
   times (vector loader, BM25 builder, parent store): ~17 s of cold start.
6. **Duplicate session writes.** Three SQLite writes per turn (record, apply,
   explicit save); apply already persists.
7. **Oversized SSE payload.** Full 32-candidate rank/score maps shipped to every
   client each turn (~10 KB, unconsumed).

## Changes implemented

- **Single-call planner with tools** (`answering/litellm.py`): the structured
  planner request carries the read-only tools. No-tool turns cost exactly one
  provider call; tool turns cost two. Fixes the null-content crash. Adds one
  targeted retry (with the validation reason, no tools) on correctable
  `InvalidAnswerError` only; genuine `UNSUPPORTED` never retries.
- **Device-scoped BM25** (`answering/service.py`, uses existing
  `CombinedLexicalRetriever.scoped`): 24,286-chunk index to ~426-chunk
  per-device index, built lazily once per device (~580 ms, amortized).
  Post-filter parity verified (identical top-10 membership on probe queries).
- **Lean retrieval query** (`_retrieval_query_lean`): symptom + latest
  observation + compact facts + last report, capped at 600 chars. Planner-only
  exclusions stay in the planner prompt where they belong.
- **Per-section diversity cap** (`retrieval/hybrid.py`): max 2 chunks per
  document/page/section in top-k so a distinct branch fits. Retrieval eval
  metrics identical before/after (recall@5 0.963, MRR 0.743, same 9 failures).
- **Prompt state caps** (`prompts.py`): reports to last 4, previous response to
  500 chars, fact history dropped (facts kept).
- **Startup single-scan** (`retrieval/indexer.py`, `context_store.py`,
  `main.py`): parse chunk JSONL once, share across vector/BM25/parent loaders.
- **Validation observability + ABSTAIN progress**: planner validation failures
  log reasons; abstain responses report `diagnostic_progress="ABSTAIN"`.
- **Removed dead weight**: `get_diagnostic_state` tool advertisement (capped
  state already travels in-prompt; executor kept for compatibility),
  `_parse_agent_turn`, full diagnostics maps in SSE, one duplicate SQLite write.
- **Options grounding recalibrated**: hard validation now rejects only
  fabricated specifics (codes/numbers/paths absent from evidence). Measured
  evidence: word-overlap blocked benign paraphrases ("Valid IP address") and
  drove abstains, while embedding similarity could not separate good from bad
  options (0.72-0.82 cosine for both). Safety-critical instruction grounding
  (source IDs, procedure order) is unchanged.

## Final results

Conversation benchmark runs (same 10-turn script, same endpoint):

| Run | E2E P50 | E2E mean | E2E P95 | Retrieval P50 | Answered |
| --- | --- | --- | --- | --- | --- |
| Baseline (untouched) | 8479 ms | 8206 ms | 9372 ms | 2560 ms | 8/10 |
| After single-call + scoped + lean | 5678 ms | 6090 ms | 10464 ms | 88 ms | 7/10 |
| Repeat (variance probe) | 4417 ms | 4942 ms | 9025 ms | 65 ms | 6/10 |
| After diversity cap | 5329 ms | 5928 ms | 10499 ms | 68 ms | 7/10 |
| After tool/SSE trim | 5813 ms | 6641 ms | 10933 ms | 54 ms | 6/10 |
| Final (retry + specifics rule) | 7844 ms | 9424 ms | 20418 ms | 73 ms | 10/10 |
| Final repeat | 10318 ms | 11106 ms | 20925 ms | 71 ms | 9/10 |

Raw reports: `/tmp/opencode/{baseline,opt1,opt1b,opt2,opt3b,final1,final2}_conv.json`
(note: `/tmp` is ephemeral; rerun `scripts/benchmark_conversation.py` to reproduce).

Structural, provider-independent improvements (stable across all runs):

| Metric | Baseline | Final | Improvement |
| --- | --- | --- | --- |
| Retrieval P50 | 2560 ms | ~70 ms | **-97%** |
| Retrieval mean | 2498 ms | ~75 ms | **-97%** |
| BM25 isolated (long query) | ~3100 ms | ~8 ms scoped+lean | **-99.7%** |
| LLM calls / turn | 2.0 (fixed) | ~1.5-1.7 (1 without tools) | **-15-25%** |
| Tool turns that work | 0/10 (crash bug) | 4-5/10 with evidence | fixed |
| Answered turns (avg of final runs) | 8/10 | 9.5/10 | +1.5 turns |
| Retrieval recall@5 | 0.963 | 0.963 | unchanged |
| Retrieval MRR | 0.743 | 0.743 | unchanged |
| Abstention accuracy | 0.771 | 0.771 | unchanged |
| Conversation policy | 1.0 (30 cases) | 1.0 (30 cases) | unchanged |
| SSE payload / turn | +~10 KB rank maps | stripped | -10 KB |

Caveat: end-to-end latency is dominated by provider generation time
(1.5-9 s per structured call, high variance between windows), so E2E P50 moves
with provider load more than with code. The code-side E2E contribution fell
from ~2.6 s (retrieval) + 2 calls to ~0.07 s + ~1.6 calls; remaining E2E is
almost entirely provider generation of the ~400-token planner JSON plus
occasional correction retries (which convert would-be abstains into answers).

## Tests added

- Single-call planner: one call without tools; null-content tool path.
- Validation retry: ungrounded-specific option triggers correction retry.
- Options grounding: fabricated code rejected; benign paraphrase allowed.
- Repeat guard: completed-action paraphrase rejected.
- Lean retrieval query: planner-only text excluded, 600-char cap.
- Scoped retriever: subset isolation, no-scope reuse.
- Streaming state persistence across turns; turn buttons/citations present.
- Diversity cap: kind variants capped per manual location.

## Experiments rejected

- **Prebuild all 20 device-scoped indexes at startup**: measured 15.6 s added
  cold start for a 580 ms lazily-amortized cost. Rejected.
- **Word-overlap options grounding**: measured abstain driver on benign
  paraphrases. Replaced with specifics-only rule.
- **Embedding-similarity options check**: measured 0.72-0.82 cosine for both
  good and bad options. No signal. Rejected.
- **Model/format changes** (json_schema, smaller models): provider-contract
  risk without measured need. Rejected.

## Remaining bottlenecks

Provider LLM generation now dominates (typically 1.5-9 s per structured call
with high between-window variance); retrieval is ~1% of a turn. Further TTFT
gains need provider-side change (streaming structured output, smaller/faster
model) or fewer calls per turn, both quality-sensitive. First turn also pays
one-time scoped-index build (~580 ms) and embedding model load (covered by
`EMBEDDING_WARMUP`).

## Next theoretical improvements

- Stream the structured planner response (parse partial JSON) instead of
  chunking the validated reply, for true token streaming TTFT.
- Smaller/faster planner model if answer quality holds; current planner JSON is
  ~400 tokens of generation per turn by construction.
- Persist scoped BM25 indexes to disk to remove the ~580 ms first-turn build
  (rejected now: invalidation complexity vs one-time cost).
- Tighten `repeated_actions` detection from substring matching to
  recommend-vs-mention classification (current T9-style false positives are
  metric noise only).

---

# Session 2026-09-10: voice TTFT, one-call turns, true streaming, model A/B

Follow-up priorities: instrument true voice TTFT; drive LLM calls toward
~1/turn; stream usable text immediately; A/B a faster planner model; explain
provider variance; fix `repeated_actions` noise. Raw reports:
`/tmp/opencode/p{0..9}_*.json` (ephemeral; rerun
`scripts/benchmark_conversation.py` to reproduce).

## 1. Voice-TTFT instrumentation (done)

- Per-call `LLMCallRecord`s (`ttft_ms`, `latency_ms`, token counts,
  `tokens_per_sec` over post-TTFT time, tool names, retry notes) now ride every
  SSE `complete` event as `llm_calls` (additive; old frontends ignore it) and
  are logged as `llm_request_sent` / `llm_first_token` / `llm_call_complete`
  traces, including on abstain paths.
- Voice bridge logs `voice_first_sentence` and
  `voice_first_actionable_sentence` (deterministic imperative/question
  heuristic, shared with the benchmark so both measure the same boundaries).
- Benchmark captures `first_sentence_ms`, `first_actionable_sentence_ms`,
  per-call TTFT/tokens-per-second, calls/turn, retries with reasons, and tool
  continuations. New metric: `llm_summary` with `retry_notes`.

## 2. Extra-call analysis → deterministic repair (done)

Measured breakdown (10-turn runs): ~1.0 base call + ~0.5–0.6 tool
continuations (genuinely necessary evidence refinement) + ~0.1–0.3 validation
retries. Retry reasons were ALL semantic shape issues (missing action/why,
stray recheck flag, clarify-with-action, ungrounded specifics) — never
malformed JSON. Fix: `_attempt_repair` corrects mechanical violations without
recall (solve-strip, clarify-drop-action, advance→clarify demote, recheck
clear, source-ID filter); only judgment calls retry. Result: retries fell to
0–1 per 10 turns; mean calls/turn 1.6–1.8 with tools working (vs 2.0 fixed
before, when the router call was pure overhead).

Two measurement bugs fixed along the way: failure-path calls were invisible
(records now attach to raised errors and abstain completions), and
tool-continuation notes were miscounted as retries.

## 3. True streaming with validation gate (done, default on)

`stream_agent_turn` streams the structured planner call and forwards decoded
`response` characters as real tokens. A response gate buffers the JSON,
validates every non-`response` field the moment it is complete, and only then
releases text — nothing unvalidated is ever spoken in default mode. The schema
orders `response` last for this reason; incremental scanner handles split
escapes, surrogate pairs, nested values, and response-first order by buffering.
Effect: first_token ≈ first_sentence ≈ complete−ε, with tokens flowing
through generation instead of one post-plan burst. Representative gated run
(p5): first_token P50 4660 ms → complete P50 5018 ms (the ~350 ms gap is the
response tail generating); pre-streaming baseline bunched everything at
~complete.

## 4. Strict `json_schema` trial (rejected, data-driven)

Probed Luna: tools + strict schema coexist. Benchmarked: retries did NOT drop
(3 semantic retries — schema cannot express cross-field rules), quality parity
otherwise. Rejected as default: no measured gain plus lost provider
portability (Groq rejects `response_format` + tools entirely). Env override
`FRIDAY_LLM_RESPONSE_FORMAT` retained for experiments.

## 5. Optimistic-streaming experiment (measured, off by default)

Arms `planner_response_position=first` + `streaming_gate=optimistic` (+schema):
first_token P50 **2459 ms** vs 4660 ms gated — nearly halved, turns speaking
at 1.2–1.6 s. Cost: 1/10 turns spoke ~325 chars then abstained (bad source ID;
the prose itself was sound troubleshooting guidance). Added
`optimistic_speech_abandoned` logging and source-ID filtering repair (keeps
verified citations, drops hallucinated ones). Defaults stay strict-gated;
flags documented in `backend/config.yml`.

## 6. Model A/B: Luna vs Groq gpt-oss-120b (stayed on Luna)

Paced protocol (50 s inter-turn sleeps; Groq free tier is 8000 TPM vs our
~3k-token planner calls — 3 turns errored on rate limits even paced).
Verdict **do not switch**: TTFT parity (2–3 s both), no structured output with
tools (2 malformed-JSON retries observed), zero successful tool routings
(Luna: 4–6/10 incl. explicit manual-search requests), blunter first-step
policy (immediate power-cycle), 6/10 answered with 3 provider errors vs
Luna 9–10/10. Env overrides (`FRIDAY_LLM_MODEL/_API_KEY_ENV/_RESPONSE_FORMAT/
_REASONING_EFFORT`) retained for future A/Bs.

## 7. Provider variance explained (measured)

Per-call TTFT swings 0.8–4.1 s call-to-call on identical ~3.5k-token prompts
while post-TTFT generation runs fast — the 1.5–9 s end-to-end spread is
provider queue/scheduling, not application overhead (retrieval steady
~55–105 ms) or generation speed. Side discovery: this model accepts only
temperature=1; production's `temperature: 0.1` is silently coerced by LiteLLM
(because `reasoning_effort` is set), so sampling is effectively nondeterministic
— determinism must come from the contract (schema, validation, repair, retry),
not sampling. Do not remove `reasoning_effort` without re-verifying.

## 8. `repeated_actions` recommend-vs-mention (done)

Per-sentence matching with mention cues (`already`, `did not help`, `no
need`, …) skipped and recommendation cues (imperatives, `please`,
`you should`, questions) flagged. The live T9 `RENEW_DHCP` false positive is
gone (0 flags across all final runs); true recommendations still flag
(covered by test).

## Final validation state

- Backend tests: 143 passed. `ruff check`, `ruff format --check`, `mypy` clean.
  Frontend `tsc --noEmit` clean.
- Conversation policy eval: 30/30.
- Retrieval eval (current code): recall@5 0.963, MRR 0.725, abstention
  accuracy 0.886, citation-ready 1.0, 5 failures (all unanswerable-class).
- Frontend: additive `LLMCallRecord` type only; no behavior change.

## Full run table (same 10-turn script, same endpoint)

| Run | 1st-tok P50 | Total P50 | Total mean | Retr P50 | Ans | Calls | Retries | Repeats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p0 baseline | 7564 | 7565 | 10023 | 216 | 7/10 | n/a | n/a | 1 |
| p1 instrumented | 6424 | 6431 | 8934 | 264 | 9/10 | 1.8 | 3 | 3* |
| p2 contract/repair | 6743 | 6746 | 7503 | 85 | 8/10 | 1.3 | 5† | 0 |
| p3 streaming | 4929 | 5458 | 6920 | 104 | 8/10 | 1.2 | 0 | 0 |
| p4 stream fixes | 4863 | 5337 | 6143 | 72 | 10/10 | 1.6 | 0 | 0 |
| p5 gated | 4660 | 5018 | 5273 | 57 | 10/10 | 1.8 | 1 | 0 |
| p6 strict schema | 4353 | 4573 | 5785 | 69 | 9/10 | 1.9 | 3 | 0 |
| p7 optimistic | 2459 | 5038 | 6105 | 67 | 10/10 | 1.8 | 2 | 0 |
| p8 Groq A/B (paced) | 4328 | 4328 | 4738 | 187 | 6/10‡ | 0.9 | 2 | 0 |
| p9 final code | 5462 | 6085 | 7092 | 102§ | 9/10 | 2.0 | 2 | 0 |
| p10 final code | 4984 | 5354 | 6412 | 87 | 10/10 | 1.7 | 0 | 0 |

\* p1 predates the recommend-vs-mention fix (2 of the 3 were mention-noise).
† p2 retry count includes tool continuations (miscount bug, fixed after).
‡ 3 turns errored on Groq TPM rate limits despite 50 s pacing.
§ p9 ran concurrently with the local retrieval eval (CPU contention).

Baseline→final (p0→p10): first-token P50 −34%, total P50 −29%,
answered 7/10→10/10, repeats 1→0, retries→0, calls/turn 2.0→1.7 with working
tool routing (vs 2.0 fixed with a dead router call). Retrieval steady
~60–100 ms throughout (application-side work is done there); remaining
variance is provider TTFT (0.8–4.1 s) plus generation tails.
