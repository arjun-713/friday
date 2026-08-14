"""Run a live hybrid retrieval smoke test and latency benchmark."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from ..ingestion.models import DocumentChunk
from .benchmark import BenchmarkQuery, run_benchmark
from .bm25 import CombinedLexicalRetriever, InMemoryBM25Retriever, InMemoryExactIdentifierRetriever
from .cache import RetrievalSessionCache
from .context_store import JsonlParentChunkStore
from .contracts import MetadataFilter
from .granite import GraniteEmbeddingProvider
from .hybrid import retrieve
from .indexer import load_vector_chunks
from .qdrant import QdrantSettings, QdrantVectorIndex


def build_proxy_queries(chunks: list[DocumentChunk], limit: int) -> list[BenchmarkQuery]:
    """Create deterministic smoke labels from real indexed chunks.

    These labels verify end-to-end plumbing and ranking stability. They are not
    a substitute for a manually verified relevance benchmark.
    """

    queries: list[BenchmarkQuery] = []
    seen: set[str] = set()
    for chunk in chunks:
        words = chunk.content.split()
        query = " ".join(words[:14]).strip(" .,:;()[]")
        if len(query) < 24 or query in seen:
            continue
        seen.add(query)
        queries.append(BenchmarkQuery(query=query, expected_chunk_ids=frozenset({chunk.chunk_id})))
        if len(queries) >= limit:
            break
    if not queries:
        raise ValueError("could not build benchmark queries from vector chunks")
    return queries


async def run_live_benchmark(
    chunks_root: Path,
    query_count: int,
    warmup_count: int,
) -> dict[str, object]:
    chunks = load_vector_chunks(chunks_root)
    provider = GraniteEmbeddingProvider()
    vector_index = QdrantVectorIndex(QdrantSettings())
    bm25 = InMemoryBM25Retriever.from_directory(chunks_root)
    lexical = CombinedLexicalRetriever(bm25, InMemoryExactIdentifierRetriever(chunks))
    parent_store = JsonlParentChunkStore.from_directory(chunks_root)
    queries = build_proxy_queries(chunks, query_count)

    async def run_query(query: str) -> list[str]:
        result = await retrieve(
            query,
            provider,
            vector_index,
            lexical_retriever=lexical,
            parent_store=parent_store,
            limit=5,
        )
        return [hit.id for hit in result.hits]

    try:
        await vector_index.ensure_collection(provider.dimension)
        for query in queries[:warmup_count]:
            await run_query(query.query)
        timings: list[dict[str, float]] = []
        result_rows: list[dict[str, object]] = []

        async def measured_query(query: str) -> list[str]:
            started = perf_counter()
            result = await retrieve(
                query,
                provider,
                vector_index,
                lexical_retriever=lexical,
                parent_store=parent_store,
                limit=5,
            )
            timings.append({**result.timings_ms, "wall_clock_ms": (perf_counter() - started) * 1000})
            result_rows.append(
                {
                    "query": query,
                    "hits": [
                        {
                            "chunk_id": hit.id,
                            "score": hit.score,
                            "kind": hit.payload.get("kind"),
                            "section": hit.payload.get("section"),
                            "page": hit.payload.get("page"),
                            "preview": str(hit.payload.get("text", ""))[:180],
                        }
                        for hit in result.hits
                    ],
                }
            )
            return [hit.id for hit in result.hits]

        benchmark = await run_benchmark(queries, measured_query)
        cache = RetrievalSessionCache()
        first_chunk = chunks[0]
        cache.set_scope(
            MetadataFilter(manufacturer=first_chunk.document.manufacturer, model=first_chunk.document.model)
        )
        cache_cold = await cache.retrieve(
            queries[0].query,
            provider,
            vector_index,
            lexical_retriever=lexical,
            parent_store=parent_store,
            limit=5,
        )
        cache_warm = await cache.retrieve(
            queries[0].query,
            provider,
            vector_index,
            lexical_retriever=lexical,
            parent_store=parent_store,
            limit=5,
        )
        component_latency: dict[str, dict[str, float]] = {}
        for key in sorted({key for timing in timings for key in timing}):
            values = [timing[key] for timing in timings]
            from .metrics import latency_summary

            component_latency[key] = latency_summary(values)
        return {
            "query_count": len(queries),
            "warmup_count": min(warmup_count, len(queries)),
            "label_type": "synthetic_chunk_prefix_proxy",
            "warning": "Recall@5 and MRR are plumbing proxies until manually verified query labels are added.",
            "latency_ms": benchmark.latency,
            "component_latency_ms": component_latency,
            "recall_at_5": benchmark.recall_at_5,
            "mrr": benchmark.mrr,
            "cache_comparison_ms": {
                "cold_total_ms": cache_cold.timings_ms.get("total_ms", 0.0),
                "warm_total_ms": cache_warm.timings_ms.get("total_ms", 0.0),
                "warm_cache_hit": cache_warm.timings_ms.get("cache_hit", 0.0),
            },
            "results": result_rows,
        }
    finally:
        await vector_index.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-root", type=Path, default=Path("data/chunks"))
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("data/index/retrieval_benchmark.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(run_live_benchmark(args.chunks_root, args.queries, args.warmup))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
