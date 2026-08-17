# Friday interface direction

Friday is a calm technical evidence workbench for ordinary device owners and
support trainees. The interface should make the current observation, the next
safe check, and its manufacturer evidence immediately legible. It is not a
generic AI chat surface or a dashboard.

## Composition

- A fluid three-region desktop shell: device and session navigation, a centered
  conversation work surface, and an evidence ledger.
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
- User messages are right-aligned inside the work surface; Friday's procedural
  response is left-aligned and carries citations at the point of instruction.
- “What we know” is an evidence ledger. It never repeats generic schema fields
  or placeholder values, and changes as observations become known.
