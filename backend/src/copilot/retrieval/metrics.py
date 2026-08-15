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
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between zero and one hundred")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": percentile(values, 50),
        "p70_ms": percentile(values, 70),
        "p99_ms": percentile(values, 99),
        "p100_ms": percentile(values, 100),
        "max_ms": max(values),
    }
