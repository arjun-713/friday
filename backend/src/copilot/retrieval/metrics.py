"""Latency measurements for retrieval benchmarks."""

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class RetrievalTrace:
    started_at: float = field(default_factory=perf_counter)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        self.marks[name] = perf_counter()

    def duration_ms(self, start: str, end: str) -> float:
        return (self.marks[end] - self.marks[start]) * 1000


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile over no values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * percentile_value / 100))
    return ordered[index]


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": percentile(values, 50),
        "p70_ms": percentile(values, 70),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }
