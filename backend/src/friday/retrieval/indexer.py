"""CLI and file-loading helpers for persistent vector ingestion."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from ..ingestion.models import DocumentChunk, RetrievalProfile
from .contracts import EmbeddingProvider, MetadataFilter, VectorIndex
from .granite import GraniteEmbeddingProvider
from .ingest import index_chunks
from .qdrant import QdrantSettings, QdrantVectorIndex


@dataclass(frozen=True)
class IndexingReport:
    selected_chunks: int
    indexed_chunks: int
    model_name: str
    dimension: int
    chunks_root: str
    generated_at: str
    elapsed_ms: float
    chunks_per_second: float


def load_vector_chunks(
    chunks_root: Path = Path("data/chunks"),
    category: str | None = None,
    document_id: str | None = None,
    limit: int | None = None,
) -> list[DocumentChunk]:
    """Load vector-profile chunks in stable path and line order."""

    if not chunks_root.exists():
        raise FileNotFoundError(f"chunk directory does not exist: {chunks_root}")
    paths = sorted((chunks_root / category).glob("*.jsonl") if category else chunks_root.glob("*/*.jsonl"))
    chunks: list[DocumentChunk] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                chunk = DocumentChunk.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid chunk at {path}:{line_number}") from error
            if RetrievalProfile.VECTOR in chunk.retrieval_profiles:
                if document_id is not None and chunk.document.document_id != document_id:
                    continue
                chunks.append(chunk)
                if limit is not None and len(chunks) >= limit:
                    return chunks
    return chunks


async def index_from_chunks(
    chunks: Sequence[DocumentChunk],
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndex,
    batch_size: int,
) -> IndexingReport:
    """Embed and upsert selected chunks, returning a reproducible run summary."""

    started = perf_counter()
    indexed = await index_chunks(chunks, embedding_provider, vector_index, batch_size=batch_size)
    elapsed_ms = (perf_counter() - started) * 1000
    return IndexingReport(
        selected_chunks=len(chunks),
        indexed_chunks=indexed,
        model_name=embedding_provider.model_name,
        dimension=embedding_provider.dimension,
        chunks_root="",
        generated_at=datetime.now(UTC).isoformat(),
        elapsed_ms=elapsed_ms,
        chunks_per_second=indexed / (elapsed_ms / 1000) if elapsed_ms else 0.0,
    )


async def run_indexing(args: argparse.Namespace) -> IndexingReport:
    chunks = load_vector_chunks(Path(args.chunks_root), args.category, args.document_id, args.limit)
    if not chunks:
        raise ValueError("no vector-profile chunks found")
    provider = GraniteEmbeddingProvider()
    index = QdrantVectorIndex(QdrantSettings(batch_size=args.qdrant_batch_size))
    try:
        if args.document_id is not None:
            await index.ensure_collection(provider.dimension)
            await index.delete(MetadataFilter(document_id=args.document_id))
        report = await index_from_chunks(chunks, provider, index, args.embedding_batch_size)
    finally:
        await index.close()
    report = IndexingReport(**{**asdict(report), "chunks_root": args.chunks_root})
    output_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(
        f"indexed {report.indexed_chunks}/{report.selected_chunks} vector chunks "
        f"with {report.model_name} ({report.dimension} dimensions) in {report.elapsed_ms:.1f} ms "
        f"({report.chunks_per_second:.1f} chunks/s)"
    )
    print(f"wrote {output_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-root", default="data/chunks")
    parser.add_argument("--category", choices=("computers", "routers", "printers"))
    parser.add_argument("--document-id", help="replace and index one generated document scope")
    parser.add_argument("--limit", type=int, help="index only the first N vector chunks")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--qdrant-batch-size", type=int, default=128)
    parser.add_argument("--manifest", default="data/index/embedding_manifest.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run_indexing(args))


if __name__ == "__main__":
    main()
