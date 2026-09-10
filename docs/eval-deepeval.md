# DeepEval harness for Friday

DeepEval is the LLM-judged quality layer for the Friday troubleshooting
assistant. It **supplements** the authoritative deterministic guardrails
(`eval/run_retrieval.py`: Recall@5/MRR/abstention; `eval/run_conversation_policy.py`:
30/30 policy) — it never replaces them, and no DeepEval-driven change may
regress them without an explicitly documented reason.

## Architecture

```text
eval/evals/
  config.py          judge model (gpt-4o, temperature 0), tiers, budgets
  adapter.py         in-process Friday runner (REAL pipeline, isolated sessions)
  tracing.py         @observe spans attached at eval time (prod code untouched)
  metrics.py         shared standard-metric lists
  friday_metrics.py  custom product judges (GEval / ConversationalGEval)
  test_rag.py        single-turn RAG: actual chunks + deterministic chunk check
  test_conversation.py  scripted multi-turn arcs with per-turn context
  test_agent.py      tool routing + deterministic trajectory guards
  test_simulator.py  ConversationSimulator vs the real backend (heavy tier)
  test_voice.py      speakability + latency boundaries (no native voice in 3.9.9)
  test_calibration.py  judge validation against known-good/bad responses
  datasets/
    rag_curated.json            35 goldens derived from retrieval_cases.jsonl
    conversations_curated.json  3 scripted troubleshooting arcs
    simulator_goldens.json      4 simulator personas
    calibration.json            12 hand-authored judge checks (fixed reference)
    synthetic_rag.json          offline-generated candidates (review required)
    build_rag_curated.py        deterministic dataset builder (no LLM)
    generate_synthetic.py       offline Synthesizer runner (manual, never CI)
  conftest.py        path setup shared by all suites
```

Key design points:

- The adapter builds the **same components as production** (`main._build_service`
  equivalents) but with an **in-memory session store**, so eval turns never
  pollute `data/index/diagnostic_sessions.sqlite3`.
- One dedicated event loop per adapter: the Qdrant gRPC client binds to its
  creating loop, so per-test `asyncio.run()` breaks reuse.
- Tracing is applied by **runtime monkeypatch in eval code only**. Production
  modules never import deepeval: zero prod overhead, zero image bloat
  (deepeval lives in `requirements-dev.txt`; the Docker image installs
  `requirements-runtime.txt`). Spans: `friday_turn` (agent) →
  `friday_retrieve` (retriever) → `friday_plan_llm` (llm) → `friday_tool`
  (tool) → `friday_validate` (agent).
- Every RAG metric scores the **actual chunks Friday returned**
  (`result.retrieval_context`), never an idealized context.
- Friday runs at temperature 0.1 (`backend/config.yml`, untouched). Judges run
  at temperature 0 (`eval/evals/config.py`). The two are configured separately.

## Installation

```bash
uv pip install --index-strategy unsafe-best-match --python backend/.venv/bin/python -r backend/requirements-dev.txt
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Judge model (gpt-4o) for all DeepEval metrics. Source `backend/.env` (never commit keys). Required for every suite except dataset-contract checks. |
| `EMBEDDING_BACKEND=onnx`, `EMBEDDING_MODEL=...`, `EMBEDDING_MODEL_FILE=...` | Same ONNX embedding setup as `make eval-retrieval-optimized`. |
| `FRIDAY_EVAL_TIER=fast\|full` | Case selection (default `fast`). Full-tier RAG paces cases ~60 s apart (`FRIDAY_CASE_PACE_SECONDS`) — 35 cases × (Luna + 8 judges) exceeds the 30 k TPM pool otherwise; expect ~40 min. |
| `FRIDAY_EVAL_SIM=1` | Enables the simulator suite. |
| `FRIDAY_JUDGE_MODEL` | Override the judge (default `gpt-4o`). Calibrate before trusting a cheaper judge. |

Qdrant must be reachable (`docker compose up -d qdrant`, default `http://localhost:6333`).

## Suites and CI strategy

| Make target | What runs | When |
| --- | --- | --- |
| `eval-deepeval-fast` | 8-case RAG smoke + short conversations + calibration | Every PR |
| `eval-deepeval-full` | All 35 RAG cases + all conversations + agent + calibration | Pre-release / nightly |
| `eval-deepeval-agent` | Tool routing + trajectory guards | On planner/tool changes |
| `eval-deepeval-conversation` | Full conversation arcs | On prompt/policy changes |
| `eval-deepeval-calibrate` | Judge validation (no app calls) | After rubric/threshold edits |
| `eval-deepeval-voice` | Speakability + latency boundaries | Manually / nightly |
| `eval-deepeval-sim` | 4 simulator personas vs real backend | Manually (expensive) |
| `eval-deepeval-synthetic` | Offline candidate generation | Manually, then human review |

Run suites with `deepeval test run` (not raw pytest) so traces link to tests:

```bash
set -a; source backend/.env; set +a
make eval-deepeval-fast
```

## Metrics and thresholds

Standard: Faithfulness, Answer Relevancy, Contextual Relevancy (curated);
+ Contextual Precision/Recall on synthetic goldens with expected outputs;
Turn Relevancy, Turn Faithfulness, Conversation Completeness (conversations);
Tool/Argument Correctness, Task Completion (agent).

Custom product judges (`friday_metrics.py`): Evidence Grounding, One Safe
Step, Abstention Quality, Observation Button Quality, Voice Concision,
Friday Troubleshooting Policy (conversational). Thresholds start at 0.5 and
are set from `test_calibration.py` distributions — see "Judge calibration"
below. Never lower a threshold to hide a failure.

## Datasets

- **Curated RAG** (`rag_curated.json`): derived deterministically from
  `eval/retrieval_cases.jsonl` via `build_rag_curated.py`. No expected answer
  text is invented; expected chunk IDs live in metadata for the
  recall-style check.
- **Conversations** (`conversations_curated.json`): scripted user turns
  (Archer C6 10-turn arc from the benchmark script, RT-AX3000 short arc,
  out-of-domain abstain arc).
- **Simulator** (`simulator_goldens.json`): impatient, vague,
  already-tried-it, and error-code personas.
- **Calibration** (`calibration.json`): fixed known-good/bad behaviors with
  expected verdicts. Edit only to add cases, never to match judge behavior.
- **Synthetic** (`synthetic_rag.json`, gitignored until reviewed): candidates
  from `generate_synthetic.py`. Expected outputs are unvalidated — use for
  ranking metrics only until a human reviews samples.

To add goldens: prefer curating from real manuals/sessions; for scale run
`make eval-deepeval-synthetic`, review the candidates, then promote the good
ones with `provenance: synthetic-reviewed`.

## Judge calibration

`make eval-deepeval-calibrate` scores 12 fixed behaviors (excellent step,
step dump, hallucinated value, repeated check, correct/incorrect abstention,
( un)grounded options, markdown dump, concise reply, unsupported branch,
benign paraphrase). A failure means the rubric is wrong — fix
`friday_metrics.py`, not the case. Results: TBD (baseline run).

## Traces: how OpenCode/Codex should debug

1. `make eval-deepeval-fast` (or the failing suite).
2. Open the failing test's trace: spans nest as `friday_turn` →
   `friday_retrieve` (chunk ids, timings) → `friday_plan_llm` (model,
   structured, tools attached) → `friday_tool` (name, args, evidence count)
   → `friday_validate` (valid + reason).
3. Locate the first failing span: retrieval (wrong chunks), planner (bad
   mode/content), tool (wrong call/args), validation (contract reason).
4. Form a hypothesis, make the smallest production change, run deterministic
   tests (`pytest backend/tests`, `make eval-retrieval eval-conversation-policy`),
   rerun the DeepEval suite, compare.

## Known limitations

- DeepEval 3.9.9 has **no voice simulation stack** (no VoiceConversationSimulator,
  connectors, or voice metrics). The voice tier scores live-reply transcripts
  and latency boundaries; barge-in stays deterministic in
  `backend/tests/test_voice_bridge.py`. Revisit when DeepEval ships voice.
- The simulator tier costs real Luna turns + judge calls per simulated turn;
  keep it small and manual.
- `printer-001` (Toner LED): production's exact-identifier early return serves
  identifier chunks and skips fusion, so the LED section is missed while the
  deterministic eval (vector-only exact index) passes. Known divergence, first
  candidate for the fix-and-verify loop. See baseline report.
- Judge calls add cost: roughly (TBD) per fast run. Track via run output.

## Baseline report

TBD — recorded after the harness goes green (or honestly red with documented
known failures).
