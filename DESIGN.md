# Friday UI direction

## Direction

**Evidence workbench**: a Swiss/editorial structure adapted for technical support. The interface should feel like a calm service desk and a well-indexed manual, not an AI chat demo. The conversation is the work surface; device selection, session history, and cited procedure evidence are the supporting rails.

## Product job

Help a person identify their device, describe a fault naturally, and complete one verified diagnostic observation at a time.

## Type

- UI/body: `Avenir Next`, `Segoe UI`, system sans fallback.
- Technical metadata: `ui-monospace`, `SFMono-Regular`, `Menlo`, monospace fallback.
- Use sentence case for primary content. Uppercase is reserved for short navigation labels and source metadata.
- Type scale: 11px metadata, 12px compact UI, 14px supporting UI, 16px body, 22px section title, 40–58px page title.

## Palette

The palette is intentionally restrained: neutral surfaces carry most of the screen, blue is reserved for selection and the next action, green only confirms a known state.

```css
--paper: #f7f8fa;
--paper-deep: #eef1f5;
--ink: #172033;
--ink-soft: #667085;
--line: #e3e7ee;
--line-dark: #cbd3df;
--signal: #2563eb;
--signal-dark: #1d4ed8;
--success: #37a174;
--warning: #c78a20;
--white: #ffffff;
```

## Geometry and depth

- Small controls: 6–8px radius.
- Standard panels: 10–14px radius.
- No decorative pills or oversized rounded containers.
- Borders and tonal surfaces do most of the separation work.
- Shadows are reserved for the active procedure panel and composer; never combine heavy shadow, glow, and gradient on ordinary content.
- Main shell is fluid and fills the viewport. Side rails use `clamp()` rather than fixed page widths.

## Motion

- Motion is functional only: pressed controls, listening/thinking feedback, and session selection continuity.
- Micro-interactions use 100–180ms transitions.
- No looping decorative motion, bounce, glow, or animated entrance choreography.
- `prefers-reduced-motion: reduce` removes non-essential transitions and the thinking pulse.

## Content rules

- Use the minimum copy needed to identify the current state or next action.
- Diagnostic instructions must keep their document, page, and section evidence visible.
- Never present confidence percentages or unsupported device facts as if they were verified.
- The next action is always one observation, followed by a request for the result.
