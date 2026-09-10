# DeepEval baseline (pre-optimization, current production code)

Recorded before any DeepEval-driven production change. Production code is
untouched at this point; only `eval/evals/`, docs, Makefile, and
`requirements-dev.txt` were added. Reproduce with the make targets in
`docs/eval-deepeval.md` (source `backend/.env` first).

Judge: gpt-4o, temperature 0. App: Luna, temperature 0.1 (unchanged).

## Deterministic guardrails (authoritative)

| Metric | Score | Source |
| --- | --- | --- |
| Recall@5 | 0.988 | `make eval-retrieval-optimized` (2026-09-10) |
| MRR | 0.740 | same |
| Abstention accuracy | 0.914 | same |
| Failing cases | 3, all unanswerable-class (`unanswerable-medical-device`, `unanswerable-automotive`, `unanswerable-high-voltage`: `should_abstain_but_returned_hits`) | same |
| Conversation policy | 30/30 | `make eval-conversation-policy` |
| Backend unit tests | 143 passed (pre-harness; rerun before rounds) | `pytest backend/tests` |

## Judge calibration (harness trust)

| Round | Result | Notes |
| --- | --- | --- |
| r1 | 19/20 | `good-options/Voice Concision` failed on a flawed case (WAN/Internet term swap in authored text); case fixed to unambiguous wording. No threshold touched. |
| r2 | 18/20 | Both flips trace to the One Safe Step steps-rewrite under test, not the app: `concise-speakable` (a legit 3-action setup sequence) newly failed, and the unchanged Voice Concision rubric let the 7-item `step-dump` pass on a formatting technicality (judge variance on the same rubric). Rubric fixed both ways: procedural sequences (2-4 steps, one procedure) explicitly pass; >3 distinct actions in one reply explicitly fail concision. No threshold touched, no case weakened. |
| r3 | 20/20 | Judges trusted. Baseline fast-RAG rerun below uses these rubrics. |

## Fast RAG suite (`test_rag_fast`, 8 curated cases, trusted rubrics)

Baseline (`friday-baseline` identifier): **4/8 pass**.

| Case | Verdict | Notes |
| --- | --- | --- |
| computer-001 | pass | Faithfulness 1.0, Grounding 0.92, Step 0.85 |
| computer-003 | pass | Enumeration carve-out works (Step 0.86) |
| computer-005 | **fail (variance)** | Step 0.03 vs 0.85 in r1 on a different sampled output for a why-question. No product change; watch across runs. |
| router-001 | pass | All ≥0.8 |
| router-007 | pass (thin) | Step 0.55 twice running: explains red WAN, names no check. Watch item. |
| unsupported-001 | **fail** | Bare `UNSUPPORTED` answer → Round 1 Fix A |
| unanswerable-medical-device | **fail** | Bare `UNSUPPORTED` answer → Round 1 Fix A |
| printer-001 | **fail (retrieval)** | Deterministic chunk assert: exact-identifier early return skips fusion → Round 1 Fix B |

Known r1 observations (to confirm in baseline rerun):

- `printer-001`: production exact-identifier early return serves identifier
  chunks and skips fusion, so the Toner LED section is missed; deterministic
  eval passes because its exact index is vector-chunks-only. Real
  prod-vs-eval divergence, Round 1 candidate. No threshold/data change.
- `computer-003`: "what tools" answer lists tools; v1 One Safe Step rubric
  failed it (0.37). Rubric fixed with enumeration carve-out; judge rerun
  pending. No app change.
- `unsupported-001`, `unanswerable-medical-device`: app answers bare
  `UNSUPPORTED` (planner abstain turn carries the sentinel as prose through
  validation to the user). v1 rubric scored 0.0. Round 1 app-fix candidate:
  map bare-sentinel abstains to the graceful message. No threshold change.
- `computer-001` Faithfulness 0.67 (pass): judge alleges display-before-
  grounding slip; "display" IS in retrieved chunk 3. Watch item, not a failure.
- `router-007` One Safe Step 0.52 (pass but thin): explains red WAN, no next
  check. Watch item.

## Cost/latency reference (r1 fast RAG, 7 judged tests)

- 100 s wall, $0.14 judge cost. Full-suite and conversation costs TBD.

## Round 1: bare-`UNSUPPORTED` leak (app fix + suite gating)

- Finding: planner abstain turns carried the literal sentinel as user-facing prose through validation (voice would speak "unsupported").
- Fix (`litellm.py:_parse_turn_text`, +2 unit tests): exact-match sentinel responses raise `UnsupportedAnswerError` → graceful message, never retried. 145 unit tests pass, ruff/mypy clean.
- Suite finding: One Safe Step will not credit abstentions in ANY rubric form (0.12 with a passing reason; v1-criteria also fails them). Design fix, not rubric fix: abstained turns are now scored by Faithfulness + Evidence Grounding + Abstention Quality only (`metrics.abstention_metrics`, gated in `run_case`). Verified by experiment, not by threshold tuning.
- Verification: `friday-round1b` rerun on both abstain cases (running).

Round 1 close-out (`friday-round1b`): `unsupported-001` passes 3/3
(Faithfulness 1.0, Grounding 0.94, Abstention 0.91) — sentinel gone, graceful
message in place. `unanswerable-medical-device` scores 1.0/0.26/0.48: the
triage question ("which manufacturer and model") is designed safety-gate
behavior pinned by `test_answering.py:257`, so the gate copy stays; the
judge's 0.48 asks for a coverage admission the gate was never designed to
make. Recorded as borderline/watch, not a defect. No thresholds touched.

## Round 2: exact-identifier early return (retrieval fix)

- Finding: production built its exact index over all chunks, so EXACT-only
  identifier stubs (no vector profile) triggered the exact early-return for
  any query merely mentioning a model number, skipping fusion entirely.
  The deterministic eval never saw this (its exact index is vector-only).
- Fix (`main._lexical_retriever`, +1 unit test): exact index built from
  vector-profile chunks only — same universe as the trusted eval config.
  Eval adapter uses the same helper (no drift by construction).
- Effect on printer-001: deterministic-assert failure + abstain → grounded
  answer, Faithfulness 1.0, Grounding 1.0. Remaining 0.49 on One Safe Step
  was rubric over-reach on factual questions (same family as the enumeration
  fix): added factual-answer carve-out + calibration case; r4 22/22.
- Verification: `friday-round2b` printer-001 passes 3/3 (1.0/1.0/1.0);
  deterministic guardrails identical (recall 0.988, MRR 0.740, abstention
  0.914, same 3 unanswerable-class failures; policy 30/30). No regressions.

## Round 3: full-sweep verification (harness fixes, no app changes)- RAG fast: **8/8** (baseline was 4/8). Conversation fast: **2/2**.
- Agent 1/3 → two harness bugs found, both verified by experiment:
  - `TaskCompletionMetric` without a task definition demands full resolution
    and fails correct one-step answers (0.4). Fixed by passing Friday's
    incremental task definition (`task=` in `test_agent_tool_routing`).
    Tool/argument routing itself was 1.0 throughout.
  - Voice turn 1 tripped the 5 s retrieval bound at 7.5 s: one-time ONNX
    model load inside the first measured retrieval, not a regression.
    Fixed with embedding warmup at adapter construction (mirrors `main.py`
    startup warmup).
- Re-verification running (`friday-round3-agent2`, `friday-round3-voice2`).

Verified outcomes:

- Agent (`friday-round3-agent2`): 2/3. Tool/argument routing 1.0 everywhere.
  `tool-not-needed` (Toner LED) failed Task Completion 0.3 because the
  planner ABSTAINED despite good retrieval — same query answered 3/3 in the
  RAG suite. Planner under-answer variance (abstention judgment unstable on
  thin-but-sufficient evidence), 1 sample. Watch item, no code change.
- Voice (`friday-round3-voice2`): turn 2 passes (0.82, remembers cable fact
  across turns: "Since the cable is firmly connected..."). Turn 1 fails the
  deterministic actionability assert (explanation-only red-WAN meaning,
  same pattern as router-007's thin 0.55). Documented tension: RAG factual
  carve-out vs voice actionability standard. No change on one sample.
- RAG fast 8/8, conversation fast 2/2 stand.
- Simulator (`friday-sim-scoped`, 1 persona): **pass** — Turn Relevancy 1.0,
  Troubleshooting Policy 0.81 over a 12-turn impatient-user conversation
  (short answers, restart→WAN IP→0.0.0.0→cable→MAC Clone, results remembered,
  no repeats). Confirms the earlier 0.33 was the missing-scope harness bug,
  not app behavior. Full 4-persona run in flight (`friday-sim-full`).
- Simulator full (`friday-sim-full`): **4/4 pass**. Impatient Archer 1.0/0.86,
  vague ASUS 1.0/0.9, already-tried repeat-stress 1.0/0.82, Brother
  error-code 0.5/0.8 (relevancy exactly at threshold — watch item). The
  repeat-suppression stress persona passes with no repeated completed checks.
- Dedup-vs-eval finding (printer-004): the expected chunk's text was present
  via its identical duplicate, but 4 judges failed the turn anyway — the
  shared text is an overview without the enumerated controls, and Friday
  hedged honestly ("the retrieved excerpt does not identify each individual
  control") instead of asking a discriminating clarification. Text-aware
  matching now honors received evidence; the residual failure is
  evidence-thinness for enumeration questions, not planner dishonesty.
  Known limitation, watch across runs. No code change.
- Full-tier rerun was invalidated by 429s (25/29 failures infra). Suite now
  paces full-tier cases 60 s apart. Paced rerun in flight (`friday-full-rag2`).
- Paced full-tier rerun (`friday-full-rag2`) ABORTED by operator decision:
  killed mid-run for token cost. No partial scores are quoted from it.

## Judge selection (measured)

Local-judge benchmark (22-check calibration agreement): gpt-4o 22/22;
gpt-4o-mini 20/22 (misses ungrounded-options — covered by deterministic
button tests in PR + gpt-4o nightly); best local (qwen2.5:1.5b) 13/22 with
dangerous-direction failures; qwen3:4b infeasible on 2-core runners
(all timeouts). Decision: mini for PR, gpt-4o nightly, no local judge.
Full table in docs/eval-deepeval.md.
