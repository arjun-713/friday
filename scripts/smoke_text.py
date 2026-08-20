"""Smoke-test the running text troubleshooting stack without printing secrets."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--query", default="My HP LaserJet Pro M404 is powered on but it will not print.")
    parser.add_argument("--manufacturer", default="HP")
    parser.add_argument("--model", default="LaserJet Pro M404/M405")
    args = parser.parse_args()

    payload = json.dumps(
        {
            "query": args.query,
            "manufacturer": args.manufacturer,
            "model": args.model,
            "session_id": "smoke-text",
        }
    ).encode("utf-8")
    request = Request(
        f"{args.base_url.rstrip('/')}/v1/troubleshoot",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        detail = error.read().decode("utf-8", errors="replace") if isinstance(error, HTTPError) else str(error)
        print(f"text smoke failed: {detail}", file=sys.stderr)
        return 1

    turn = result.get("turn") if isinstance(result, dict) else None
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "turn_mode": turn.get("mode") if isinstance(turn, dict) else None,
                "has_response": bool(turn and turn.get("response")) if isinstance(turn, dict) else False,
                "citation_count": len(result.get("citations", [])) if isinstance(result, dict) else 0,
                "retrieval": result.get("retrieval") if isinstance(result, dict) else None,
            },
            indent=2,
        )
    )
    if not isinstance(turn, dict) or not turn.get("response"):
        print("text smoke failed: response did not contain a structured diagnostic turn", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
