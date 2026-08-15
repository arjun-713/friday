"""Run the manually verified retrieval benchmark against local Qdrant."""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from copilot.retrieval.bm25 import (
    CombinedLexicalRetriever,
    InMemoryBM25Retriever,
    InMemoryExactIdentifierRetriever,
)
from copilot.retrieval.cache import RetrievalSessionCache
from copilot.retrieval.context_store import JsonlParentChunkStore
from copilot.retrieval.contracts import MetadataFilter
from copilot.retrieval.granite import GraniteEmbeddingProvider
from copilot.retrieval.hybrid import retrieve
from copilot.retrieval.indexer import load_vector_chunks
from copilot.retrieval.metrics import latency_summary
from copilot.retrieval.qdrant import QdrantSettings, QdrantVectorIndex

from .retrieval_schema import RetrievalCase


def load_cases(path: Path) -> list[RetrievalCase]:
    cases = [
        RetrievalCase.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"no evaluation cases found in {path}")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")
    return cases


async def run(
    cases: list[RetrievalCase],
    chunks_root: Path,
    candidate_limit: int = 32,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.5,
    rrf_k: int = 30,
    diversify: bool = False,
    scoped: bool = True,
) -> dict[str, Any]:
    chunks = load_vector_chunks(chunks_root)
    provider = GraniteEmbeddingProvider()
    index = QdrantVectorIndex(QdrantSettings())
    bm25 = InMemoryBM25Retriever.from_directory(chunks_root)
    lexical = CombinedLexicalRetriever(bm25, InMemoryExactIdentifierRetriever(chunks))
    parent_store = JsonlParentChunkStore.from_directory(chunks_root)
    session_caches: dict[tuple[str | None, str | None], RetrievalSessionCache] = {}
    rows: list[dict[str, Any]] = []
    try:
        await index.ensure_collection(provider.dimension)
        warmup_case = cases[0]
        warmup_filter = (
            MetadataFilter(
                manufacturer=warmup_case.manufacturer, model=warmup_case.model
            )
            if warmup_case.manufacturer or warmup_case.model
            else None
        )
        await _retrieve_case(
            warmup_case.query,
            warmup_filter,
            provider,
            index,
            lexical,
            parent_store,
            session_caches,
            candidate_limit,
            dense_weight,
            lexical_weight,
            rrf_k,
            diversify,
            scoped,
        )
        for case in cases:
            started = perf_counter()
            metadata_filter = None
            if case.manufacturer or case.model:
                metadata_filter = MetadataFilter(
                    manufacturer=case.manufacturer, model=case.model
                )
            result = await _retrieve_case(
                case.query,
                metadata_filter,
                provider,
                index,
                lexical,
                parent_store,
                session_caches,
                candidate_limit,
                dense_weight,
                lexical_weight,
                rrf_k,
                diversify,
                scoped,
                include_diagnostics=True,
            )
            wall_clock_ms = (perf_counter() - started) * 1000
            retrieved_ids = [hit.id for hit in result.hits]
            relevant_hits = set(retrieved_ids) & set(case.expected_chunk_ids)
            first_relevant_rank = next(
                (
                    rank
                    for rank, chunk_id in enumerate(retrieved_ids, start=1)
                    if chunk_id in case.expected_chunk_ids
                ),
                None,
            )
            citation_ready = [
                hit
                for hit in result.hits
                if hit.payload.get("document_id")
                and hit.payload.get("page")
                and hit.payload.get("section")
            ]
            rows.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "category": case.category,
                    "question_type": case.question_type,
                    "expected_chunk_ids": sorted(case.expected_chunk_ids),
                    "expected_pages": sorted(case.expected_pages),
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieved_pages": [hit.payload.get("page") for hit in result.hits],
                    "relevant_hit_count": len(relevant_hits),
                    "recall_at_5": len(relevant_hits) / len(case.expected_chunk_ids)
                    if case.expected_chunk_ids
                    else None,
                    "reciprocal_rank": 1 / first_relevant_rank
                    if first_relevant_rank
                    else 0.0,
                    "abstained": result.abstained,
                    "abstention_correct": result.abstained == case.should_abstain,
                    "citation_ready_hit_rate": len(citation_ready) / len(result.hits)
                    if result.hits
                    else 0.0,
                    "timings_ms": {**result.timings_ms, "wall_clock_ms": wall_clock_ms},
                    "diagnostics": result.diagnostics,
                    "failure_reason": _failure_reason(
                        case, result.abstained, retrieved_ids
                    ),
                }
            )
    finally:
        await index.close()
    return _report(
        cases,
        rows,
        candidate_limit=candidate_limit,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
        rrf_k=rrf_k,
        diversify=diversify,
        scoped=scoped,
    )


async def _retrieve_case(
    query: str,
    metadata_filter: MetadataFilter | None,
    provider: GraniteEmbeddingProvider,
    index: QdrantVectorIndex,
    lexical: CombinedLexicalRetriever,
    parent_store: JsonlParentChunkStore,
    session_caches: dict[tuple[str | None, str | None], RetrievalSessionCache],
    candidate_limit: int,
    dense_weight: float,
    lexical_weight: float,
    rrf_k: int,
    diversify: bool,
    scoped: bool,
    include_diagnostics: bool = False,
):
    options = {
        "lexical_retriever": lexical,
        "parent_store": parent_store,
        "metadata_filter": metadata_filter,
        "limit": 5,
        "candidate_limit": candidate_limit,
        "dense_weight": dense_weight,
        "lexical_weight": lexical_weight,
        "rrf_k": rrf_k,
        "diversify": diversify,
        "include_diagnostics": include_diagnostics,
    }
    if not scoped or metadata_filter is None:
        return await retrieve(query, provider, index, **options)
    key = (metadata_filter.manufacturer, metadata_filter.model)
    cache = session_caches.setdefault(key, RetrievalSessionCache())
    cache.set_scope(metadata_filter, lexical_retriever=lexical)
    return await cache.retrieve(query, provider, index, **options)


def _failure_reason(
    case: RetrievalCase, abstained: bool, retrieved_ids: list[str]
) -> str | None:
    if case.should_abstain and not abstained:
        return "should_abstain_but_returned_hits"
    if not case.should_abstain and not set(retrieved_ids) & set(
        case.expected_chunk_ids
    ):
        return "expected_evidence_not_in_top_5"
    return None


def _report(
    cases: list[RetrievalCase],
    rows: list[dict[str, Any]],
    *,
    candidate_limit: int,
    dense_weight: float,
    lexical_weight: float,
    rrf_k: int,
    diversify: bool,
    scoped: bool,
) -> dict[str, Any]:
    labeled = [row for row in rows if row["expected_chunk_ids"]]
    timings = [row["timings_ms"]["wall_clock_ms"] for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)

    def group_metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        labeled_group = [row for row in group if row["recall_at_5"] is not None]
        return {
            "cases": len(group),
            "recall_at_5": _mean(row["recall_at_5"] for row in labeled_group),
            "mrr": _mean(row["reciprocal_rank"] for row in labeled_group),
            "abstention_accuracy": _mean(
                float(row["abstention_correct"]) for row in group
            ),
            "latency_ms": latency_summary(
                [row["timings_ms"]["wall_clock_ms"] for row in group]
            ),
        }

    return {
        "case_count": len(cases),
        "labeled_case_count": len(labeled),
        "unanswerable_case_count": len(cases) - len(labeled),
        "warmup": "one unscored retrieval using the first case before measurement",
        "retrieval_config": {
            "candidate_limit": candidate_limit,
            "dense_weight": dense_weight,
            "lexical_weight": lexical_weight,
            "rrf_k": rrf_k,
            "diversify": diversify,
            "scoped": scoped,
        },
        "metrics": {
            "recall_at_5": _mean(row["recall_at_5"] for row in labeled),
            "mrr": _mean(row["reciprocal_rank"] for row in labeled),
            "abstention_accuracy": _mean(
                float(row["abstention_correct"]) for row in rows
            ),
            "citation_ready_hit_rate": _mean(
                row["citation_ready_hit_rate"] for row in rows
            ),
            "latency_ms": latency_summary(timings),
        },
        "by_question_type": {
            key: group_metrics(value) for key, value in sorted(groups.items())
        },
        "failures": [row for row in rows if row["failure_reason"]],
        "results": rows,
    }


def _mean(values: Any) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("eval/retrieval_cases.jsonl")
    )
    parser.add_argument("--chunks-root", type=Path, default=Path("data/chunks"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/index/retrieval_eval.json")
    )
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--lexical-weight", type=float, default=1.5)
    parser.add_argument("--rrf-k", type=int, default=30)
    parser.add_argument("--diversify", action="store_true")
    parser.add_argument(
        "--global",
        action="store_false",
        dest="scoped",
        help="measure filtered global BM25 instead of the confirmed-device scoped path",
    )
    args = parser.parse_args()
    report = asyncio.run(
        run(
            load_cases(args.cases),
            args.chunks_root,
            candidate_limit=args.candidate_limit,
            dense_weight=args.dense_weight,
            lexical_weight=args.lexical_weight,
            rrf_k=args.rrf_k,
            diversify=args.diversify,
            scoped=args.scoped,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"metrics": report["metrics"], "failures": len(report["failures"])},
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
