# Friday — graphite / jade

## Direction

Friday helps device owners describe a fault, consult manufacturer evidence,
and troubleshoot in conversation. The visual concept is a portable diagnostic
instrument: tactile hardware, legible evidence, calm controls. This replaces
the previous cobalt casebook styling.

The landing centers an original interactive Three.js router. Its exploded view
is illustrative, not a manufacturer-specific disassembly instruction. The page
moves through conversation, manual evidence, voice, supported device categories,
and practical questions. Demonstrations are labeled; no fabricated customer
logos, benchmarks, testimonials, or live activity are used.

## System

`app/globals.css` contains authoritative custom properties. Graphite `#101715`,
navigation `#0c1210`, raised surfaces `#19231e`, text `#f1f5ef`, secondary text
`#a8b6ac`, and jade `#b5e4bc` form the workspace. Landing paper is `#edf1e9`
with `#15231b` ink. Self-hosted Manrope variable supplies display and UI type;
monospace is reserved for brief technical metadata. The font's OFL license is
kept beside it in `public/fonts`.

Spacing follows a 4px foundation, with compact control groups and larger section
intervals. Controls use 8px radii; major surfaces use 16–24px. Shadows indicate
elevation rather than decorating every element. Hero typography supports the
hardware composition rather than replacing it.

## Workspace and flow

Desktop has device/session navigation, a readable transcript, and evidence.
Below 1200px evidence becomes a drawer; at 760px navigation becomes a sheet.
The app occupies the viewport and only its transcript and rails scroll.
New-session prompts fill the draft without sending it. Device selection,
session restoration, sources, and voice remain available through existing APIs.

User messages align right; assistant messages align left with a quiet brand
marker. Streaming text appears before structured completion. Selected answers
remain attached to their original response. Sources are separate from prose.
Copy works on completed historical responses; regeneration targets the latest.
Voice controls explicitly pause, resume, interrupt, and end the voice session.
Do not add nonfunctional authentication, attachments, or ratings.

## Motion and access

The router responds to pointer movement and an explicit inspection toggle.
Rendering is capped and skipped when not visible; reduced motion gives stable
transitions. Voice animation reflects an interaction, not permanent background
activity. Keyboard focus is visible; drawers support Escape and focus cycling.
Mobile gives the device composition its own space instead of shrinking desktop.

## Verification

`npm run test:ui` exercises the landing, streaming, saved sessions, copy,
regeneration, selected observations, mobile drawers, voice controls, and error
recovery with controlled fixtures. This is not a live speech-quality or backend
diagnosis evaluation. Run `npm run build` and inspect rendered layouts after
composition changes.

Development writes `.next-dev`; production writes `.next`, preventing concurrent
builds from corrupting the hot-reload chunks. Keep frontend port 3000.
