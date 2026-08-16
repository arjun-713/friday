# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Next.js frontend with a FastAPI backend. The frontend must support text and voice interaction equally; the backend currently exposes the text-only troubleshooting endpoint at `/v1/troubleshoot`.

## Users

The primary user is an everyday device owner who may not know what an observed symptom means, which test to perform, or how to safely fix a laptop, desktop, Wi-Fi router, or printer problem.

A secondary user is a new technical support person who uses the assistant to learn troubleshooting workflows while helping someone.

## Product Purpose

Friday is a voice-first technical troubleshooting copilot grounded in official public manufacturer manuals and support documents. It helps a user describe a problem naturally, asks for missing observations, and gives one verified diagnostic step at a time with document, page, and section citations.

Success means the user can move from an unclear symptom to a safe, evidence-backed next diagnostic action without receiving invented repair instructions.

## Positioning

Friday combines hybrid retrieval, exact technical-identifier lookup, structured diagnostic state, and source-preserving citations so every technical instruction can be traced to an official manual or the system can abstain.

## Operating Context

The user may be troubleshooting a device hands-on and may interrupt or correct the assistant. The interface must support both typed text and voice interaction, show the live conversation and citations, and preserve diagnostic state across turns.

The current text workflow retrieves from locally indexed manuals through FastAPI, Qdrant, BM25, and a local Granite embedding model. A DeepSeek answer generator is planned but is not connected yet.

## Capabilities and Constraints

- Initial device domains are laptops/desktops, Wi-Fi routers, and printers.
- Source evidence is limited to public manufacturer manuals and support documents.
- Every technical instruction needs a document, page, and section citation.
- The assistant must ask for missing observations instead of assuming them.
- The assistant gives one diagnostic step at a time and preserves warnings, prerequisites, and procedure order.
- Unsupported questions must produce abstention rather than a hallucinated answer.
- Voice transport, speech recognition, interruption handling, and text-to-speech are later frontend phases; the first frontend must still establish their shared interaction model.
- Raw user audio is not stored by default.
- Medical, automotive, aviation, high-voltage, autonomous repair, community advice, and foundation-model training are out of scope.

## Evidence on Hand

- Official manuals are organized under `data/manuals` and parsed, cleaned, chunked, and indexed locally.
- The hybrid retriever currently achieves 98.8% Recall@5 and zero failures on the current supported benchmark.
- The text-only backend contract is documented in `docs/text-answering.md`.
- The current frontend is an unstyled Next.js scaffold in `frontend/app`.
- No product logo, visual identity, testimonials, commercial claims, or user-provided imagery has been established. Future UI work must not fabricate them.

## Product Principles

1. Evidence before confidence.
2. One safe next step at a time.
3. Ask instead of assuming.
4. Voice and text are equal entry points.
5. Make the reasoning traceable without making the user learn the system.

## Accessibility & Inclusion

The interface must support keyboard and screen-reader use alongside voice input. Text transcripts and visible citations are required even when voice is active. Controls must expose clear labels, focus states, live status updates, and an accessible alternative to every voice-only action.
