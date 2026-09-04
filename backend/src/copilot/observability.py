"""Small, provider-neutral timing traces for end-to-end troubleshooting turns."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any


def trace_event(
    logger: logging.Logger,
    event: str,
    *,
    turn_id: str | None = None,
    started: float | None = None,
    sequence: int | None = None,
    **fields: Any,
) -> None:
    """Emit one grep-friendly JSON event without logging sensitive content."""

    payload: dict[str, Any] = {
        "event": event,
        "turn_id": turn_id,
        "elapsed_ms": round((perf_counter() - started) * 1000, 2) if started is not None else None,
        **fields,
    }
    if sequence is not None:
        payload["sequence"] = sequence
    logger.info("friday_trace %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
