---
version: 1
slug: "frontend-app-page-tsx"
primary_target: "frontend/app/page.tsx"
related_targets: ["frontend/app/layout.tsx"]
---

# Frontend troubleshooting workspace

## Scope and visitor mode

The primary surface is the authenticated-free troubleshooting workspace in `frontend/app/page.tsx`. Mode: Operate. The user is trying to understand a device problem and complete the next safe diagnostic action.

## Audience, job, action, proof, and constraints

The main audience is an everyday device owner; a new technical support person is a secondary learner. Their job is to describe a symptom, confirm the device, receive one verified next step, report the result, and continue. The primary action is submitting a text or voice observation. Proof is the visible manual citation, page, section, warning, and prerequisite attached to the step. The surface must treat text and voice equally, expose transcript and listening/speaking state, support interruption and correction later, and never imply unsupported certainty.

## Chosen direction and memorable moment

Conversation-first troubleshooting workspace: a familiar chatbot canvas is the primary surface, while a persistent diagnostic rail turns the free-form conversation into visible state. The rail should progressively reveal the device, symptom/error, error code, observations, completed tests, current procedure step, and likely causes as the user talks; it must never force the user to complete a form before speaking naturally.

The memorable moment is an unclear statement becoming legible without interrupting the user: Friday recognizes the device and current problem in the right rail, then presents one clearly cited next diagnostic step in the conversation with the option to ask why.

## States and ranges

- First-run conversation with no device or symptom known.
- Device/model discovered or corrected naturally during chat.
- Error code recognized, suggested, corrected, or unresolved.
- Listening, microphone permission request, permission denied, transcription, thinking/retrieving, cited answer ready, speaking, user interruption, recovery, abstention, backend error, and retry.
- Active diagnostic state with zero or several observations, one current step, and completed tests.
- One active session only; no session history in the first surface.

## Interaction and layout

The conversation occupies the dominant column. User and assistant turns remain readable as a live transcript, with citations attached to the specific assistant instruction that uses them. The composer supports typed messages and voice input equally; listening, recording, cancel, interruption, and retry states must be explicit. A right-side diagnostic rail stays available on desktop and becomes a compact expandable summary above or beside the conversation on smaller screens. The rail is informative first and editable through natural-language corrections, with direct controls only where they reduce ambiguity.

## Scope and boundaries

The first target is the active troubleshooting workspace in `frontend/app/page.tsx`, at production-ready screen fidelity for desktop and mobile. It includes the empty/first-run conversation, active conversation, diagnostic rail, citation presentation, and all loading, permission, interruption, abstention, and error states. It does not include session history, account settings, multi-agent views, GraphRAG visualization, or autonomous repair actions.

## Unresolved decisions

- Visual world, typography, color, material, and component grammar.
- Comp-first versus code-first build path.
- Exact DeepSeek answer schema and loading/error presentation.
- Browser voice transport and permission implementation.
