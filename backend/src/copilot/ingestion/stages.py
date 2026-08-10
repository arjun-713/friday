"""Small deterministic stage contracts; integrations are intentionally deferred."""

from dataclasses import dataclass
from typing import Protocol

from .models import DocumentChunk, PageRecord, PdfKind, SourceDocument


class PdfClassifier(Protocol):
    def classify(self, path: str) -> PdfKind: ...


class PageExtractor(Protocol):
    def extract(self, path: str, kind: PdfKind) -> list[PageRecord]: ...


class ChunkBuilder(Protocol):
    def build(self, document: SourceDocument, pages: list[PageRecord]) -> list[DocumentChunk]: ...


class IndexWriter(Protocol):
    def write(self, chunks: list[DocumentChunk]) -> None: ...


@dataclass(frozen=True)
class IngestionResult:
    document: SourceDocument
    kind: PdfKind
    pages: int
    chunks: int


class IngestionPipeline:
    def __init__(
        self, classifier: PdfClassifier, extractor: PageExtractor, chunker: ChunkBuilder, index: IndexWriter
    ) -> None:
        self.classifier = classifier
        self.extractor = extractor
        self.chunker = chunker
        self.index = index

    def run(self, path: str, document: SourceDocument) -> IngestionResult:
        kind = self.classifier.classify(path)
        pages = self.extractor.extract(path, kind)
        chunks = self.chunker.build(document, pages)
        self.index.write(chunks)
        return IngestionResult(document=document, kind=kind, pages=len(pages), chunks=len(chunks))
