import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .answering import (
    EvidenceOnlyAnswerGenerator,
    LiteLLMAnswerGenerator,
    LiteLLMSettings,
    TroubleshootingRequest,
    TroubleshootingResponse,
    TroubleshootingService,
)
from .retrieval.bm25 import CombinedLexicalRetriever, InMemoryBM25Retriever, InMemoryExactIdentifierRetriever
from .retrieval.context_store import JsonlParentChunkStore
from .retrieval.granite import GraniteEmbeddingProvider
from .retrieval.indexer import load_vector_chunks
from .retrieval.qdrant import QdrantSettings, QdrantVectorIndex

app = FastAPI(title="Troubleshooting Copilot", version="0.1.0")
_service: TroubleshootingService | None = None
_service_lock = asyncio.Lock()
_image_manifest: dict[str, object] = {"assets": {}}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "text-only-answering"}


@app.get("/v1/assets/images/{asset_id}")
def get_image(asset_id: str) -> FileResponse:
    global _image_manifest
    if not _image_manifest.get("assets"):
        _image_manifest = _load_image_manifest()
    asset = _image_manifest.get("assets", {})
    if not isinstance(asset, dict) or asset_id not in asset:
        raise HTTPException(status_code=404, detail="image asset not found")
    metadata = asset[asset_id]
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=404, detail="image asset not found")
    relative_path = metadata.get("path")
    if not isinstance(relative_path, str):
        raise HTTPException(status_code=404, detail="image asset not found")
    path = (Path("data") / relative_path).resolve()
    image_root = (Path("data") / "assets" / "images").resolve()
    if image_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="image asset not found")
    return FileResponse(path, media_type=str(metadata.get("mime_type", "application/octet-stream")))


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


@app.post("/v1/troubleshoot/stream")
async def troubleshoot_stream(
    request: TroubleshootingRequest,
    service: TroubleshootingService = _service_dependency,
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        try:
            async for event in service.stream_answer(request):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ConnectionError, TimeoutError) as error:
            yield f"data: {json.dumps({'type': 'error', 'message': 'retrieval service is unavailable'})}\n\n"
            del error

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _build_service() -> TroubleshootingService:
    global _image_manifest
    chunks_root = Path(os.getenv("CHUNKS_ROOT", "data/chunks"))
    _image_manifest = _load_image_manifest()
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
        answer_generator=_answer_generator(),
        image_manifest=_image_manifest,
    )


def _answer_generator() -> EvidenceOnlyAnswerGenerator | LiteLLMAnswerGenerator:
    settings = LiteLLMSettings.from_env()
    if settings.enabled:
        return LiteLLMAnswerGenerator(settings)
    return EvidenceOnlyAnswerGenerator()


def _load_image_manifest() -> dict[str, object]:
    path = Path(os.getenv("IMAGE_MANIFEST", "data/assets/image_manifest.json"))
    if not path.is_file():
        return {"assets": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"assets": {}}
    return payload if isinstance(payload, dict) else {"assets": {}}
