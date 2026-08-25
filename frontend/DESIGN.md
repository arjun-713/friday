# Friday interface direction

Friday is a calm technical evidence workbench for ordinary device owners and
support trainees. The interface should make the current case, the next safe
check, and its manufacturer evidence immediately legible. It is not a generic
AI chat surface or a dashboard.

## Design thesis

Friday is a diagnostic desk / casebook. The current case is the primary object:
the left rail holds the case history, the center is the live conversation, and
the right rail is an evidence ledger. The identity carrier is a restrained
cobalt signal spine beside the active diagnostic thread. It gives the product a
recognisable geometry without decorative AI motifs, gradients, or illustration.

The architecture deliberately keeps all three regions visible on a wide screen.
A chat-first drawer layout was rejected because it hides the manual evidence
that makes Friday trustworthy. A checklist/stepper layout was rejected because
it would turn natural troubleshooting into the decision-tree interaction the
product is designed to avoid.

## Composition

- A fluid three-region desktop shell: device and casebook navigation, a
  centered conversation work surface, and an evidence ledger.
- The conversation itself is capped at 960px for readable scan paths; the app
  shell remains full-width on large screens.
- One raised surface is reserved for an active diagnostic procedure. Supporting
  information uses flat sections and divider lines rather than nested cards.
- At widths below 980px, the evidence ledger becomes a drawer. Below 760px,
  navigation is omitted from the canvas and the conversation remains primary.

## Tokens

| Role | Token |
| --- | --- |
| Application background | `--bg` |
| Navigation and rail | `--surface` |
| Raised procedure / composer | `--surface-raised` |
| Primary text | `--text` |
| Secondary text | `--muted` |
| Structural rule | `--border` |
| Product action and focus | `--accent` |
| Confirmed observation | `--success` |
| Clarification | `--warning` |
| Warning / error | `--danger` |

The palette is neutral blue-gray with one cobalt accent. Cobalt signals a
selected device, a current choice, primary sending/listening, and focus. Green
is reserved for a completed observation. No decorative gradients or glow are
used.

## Typography and geometry

- UI type: Avenir Next, Segoe UI, system sans-serif fallback.
- Metadata type: system monospace, only for concise device/source labels.
- Type scale: 12, 13, 14, 16, 20, 28, 34px.
- Corner radii: 6px controls, 8px normal components, 12px elevated procedure.
- The procedure surface may use a restrained shadow; all other structure uses
  rules and tone.

## Interaction language

- The primary action is always answering the current diagnostic question.
- Voice starts a persistent, interruptible session. Its status lives directly
  beside the composer, not in the global navigation.
- User observations are quiet transcript entries; Friday's response is a
  structured assessment with one raised next-check surface.
- “Evidence ledger” is the source of truth beside the conversation. It separates confirmed facts from
  only the unknown currently relevant to the next check.

## Content discipline

- The active case heading names the user’s actual problem when available; it
  does not use generic editorial slogans during diagnosis.
- The assistant response owns its full message bubble. Citations remain a
  separate source row so evidence never leaks into the spoken/chat copy.
- The rail shows current device, confirmed observations, the next unknown, and
  the latest source. It does not announce that manuals are “loaded” or repeat
  the entire assistant response.
- Voice state belongs next to the composer. The global header describes the
  case, not the fact that an AI system is running.
