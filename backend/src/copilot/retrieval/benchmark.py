"""Repeatable retrieval benchmark helpers."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from .metrics import latency_summary


@dataclass(frozen=True)
class BenchmarkQuery:
    query: str
    expected_chunk_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class BenchmarkResult:
    latencies_ms: list[float]
    recall_at_5: float | None
    mrr: float | None

    @property
    def latency(self) -> dict[str, float]:
        return latency_summary(self.latencies_ms)


async def run_benchmark(
    queries: Sequence[BenchmarkQuery],
    retrieve: Callable[[str], Awaitable[Sequence[str]]],
) -> BenchmarkResult:
    """Measure end-to-end retrieval and optional relevance metrics.

    The callback should include query embedding, filtered vector/lexical search,
    fusion, and parent fetch when those are part of the production path.
    """

    latencies: list[float] = []
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    has_relevance_labels = False
    for benchmark_query in queries:
        started = perf_counter()
        retrieved_ids = list(await retrieve(benchmark_query.query))
        latencies.append((perf_counter() - started) * 1000)
        if benchmark_query.expected_chunk_ids:
            has_relevance_labels = True
            relevant = benchmark_query.expected_chunk_ids
            recall_values.append(len(set(retrieved_ids[:5]) & relevant) / len(relevant))
            reciprocal_ranks.append(_reciprocal_rank(retrieved_ids, relevant))
    return BenchmarkResult(
        latencies_ms=latencies,
        recall_at_5=sum(recall_values) / len(recall_values) if has_relevance_labels else None,
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if has_relevance_labels else None,
    )


def _reciprocal_rank(retrieved_ids: Sequence[str], expected_ids: frozenset[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected_ids:
            return 1 / rank
    return 0.0
