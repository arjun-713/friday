import asyncio
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from .answering import TroubleshootingRequest, TroubleshootingResponse, TroubleshootingService
from .retrieval.bm25 import CombinedLexicalRetriever, InMemoryBM25Retriever, InMemoryExactIdentifierRetriever
from .retrieval.context_store import JsonlParentChunkStore
from .retrieval.granite import GraniteEmbeddingProvider
from .retrieval.indexer import load_vector_chunks
from .retrieval.qdrant import QdrantSettings, QdrantVectorIndex

app = FastAPI(title="Troubleshooting Copilot", version="0.1.0")
_service: TroubleshootingService | None = None
_service_lock = asyncio.Lock()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "text-only-answering"}


async def get_troubleshooting_service() -> TroubleshootingService:
    global _service
    if _service is None:
        async with _service_lock:
            if _service is None:
                _service = _build_service()
    return _service


_service_dependency = Depends(get_troubleshooting_service)


@app.post("/v1/troubleshoot", response_model=TroubleshootingResponse)
async def troubleshoot(
    request: TroubleshootingRequest,
    service: TroubleshootingService = _service_dependency,
) -> TroubleshootingResponse:
    try:
        return await service.answer(request)
    except (ConnectionError, TimeoutError) as error:
        raise HTTPException(status_code=503, detail="retrieval service is unavailable") from error


def _build_service() -> TroubleshootingService:
    chunks_root = Path(os.getenv("CHUNKS_ROOT", "data/chunks"))
    chunks = load_vector_chunks(chunks_root)
    lexical = CombinedLexicalRetriever(
        InMemoryBM25Retriever.from_directory(chunks_root),
        InMemoryExactIdentifierRetriever(chunks),
    )
    return TroubleshootingService(
        embedding_provider=GraniteEmbeddingProvider(),
        vector_index=QdrantVectorIndex(QdrantSettings()),
        lexical_retriever=lexical,
        parent_store=JsonlParentChunkStore.from_directory(chunks_root),
    )
