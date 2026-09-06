"""Benchmark Granite query embedding latency and output parity."""

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from .granite import GraniteEmbeddingProvider, GraniteEmbeddingSettings
from .metrics import latency_summary

QUERIES = (
    "My router keeps disconnecting from Wi-Fi every few minutes.",
    "What does printer error E42 mean?",
    "The laptop powers on but the screen stays black.",
    "How do I restore the router factory settings?",
    "The paper light blinks twice and printing stops.",
    "Where is the wireless diagnostic test in this manual?",
    "The desktop emits three short beeps during startup.",
    "How do I reconnect the printer after changing the Wi-Fi password?",
)


async def benchmark(settings: GraniteEmbeddingSettings, warmup: int, iterations: int) -> dict[str, object]:
    provider = GraniteEmbeddingProvider(settings)
    load_started = perf_counter()
    await provider.embed_query(QUERIES[0])
    load_ms = (perf_counter() - load_started) * 1000
    for index in range(warmup):
        await provider.embed_query(QUERIES[index % len(QUERIES)])
    samples: list[float] = []
    for index in range(iterations):
        started = perf_counter()
        await provider.embed_query(QUERIES[index % len(QUERIES)])
        samples.append((perf_counter() - started) * 1000)
    return {
        "backend": settings.backend,
        "model_file": settings.model_file,
        "cpu_threads": settings.cpu_threads,
        "cold_start_ms": round(load_ms, 3),
        "iterations": iterations,
        "mean_ms": round(mean(samples), 3),
        "latency_ms": latency_summary(samples),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "onnx", "openvino"), default="torch")
    parser.add_argument("--model", default="ibm-granite/granite-embedding-small-english-r2")
    parser.add_argument("--model-file")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = GraniteEmbeddingSettings(
        model_name=args.model,
        backend=args.backend,
        model_file=args.model_file,
        cpu_threads=args.threads,
    )
    report = asyncio.run(benchmark(settings, args.warmup, args.iterations))
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
