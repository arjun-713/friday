from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class PdfKind(StrEnum):
    TEXT = "text"
    SCANNED = "scanned"
    IMAGE = "image"
    MIXED = "mixed"


class SourceDocument(BaseModel):
    document_id: str
    title: str
    manufacturer: str
    model: str | None = None
    version: str | None = None
    source_url: str
    retrieved_at: str
    sha256: str | None = None


class TextSpan(BaseModel):
    text: str
    page: int = Field(gt=0)
    x: float
    y: float
    width: float
    height: float
    font: str
    font_size: float
    is_bold: bool = False
    is_italic: bool = False
    item_type: str


class OcrTextItem(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    page: int = Field(gt=0)
    polygon: list[tuple[float, float]] = Field(min_length=4)


class OcrPage(BaseModel):
    page: int = Field(gt=0)
    model: str
    image_path: str
    text: str
    confidence: float = Field(ge=0, le=1)
    items: list[OcrTextItem] = Field(default_factory=list)


class PageRecord(BaseModel):
    page_number: int = Field(gt=0)
    text: str
    parser: str
    confidence: float = Field(ge=0, le=1)
    requires_ocr: bool = False
    spans: list[TextSpan] = Field(default_factory=list)


class ChunkKind(StrEnum):
    SECTION = "section"
    PARENT = "parent"
    CHILD = "child"
    PROCEDURE = "procedure"
    TABLE_ROW = "table_row"
    EXACT_MATCH = "exact_match"


class ChunkStrategy(StrEnum):
    SECTION = "section"
    PARENT_CHILD = "parent_child"
    PROCEDURE = "procedure"
    TABLE_ROW = "table_row"
    EXACT_MATCH = "exact_match"


class Evidence(BaseModel):
    source_file: str = Field(min_length=1)
    page: int = Field(gt=0)
    section: str = Field(min_length=1)
    content: str = Field(min_length=1)
    coordinates: tuple[float, float, float, float] | None = None


class DocumentChunk(BaseModel):
    chunk_id: str
    document: SourceDocument
    page: int = Field(gt=0)
    pages: list[int] = Field(min_length=1)
    section: str = Field(min_length=1)
    content: str = Field(min_length=1)
    kind: ChunkKind
    parser: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    strategy: ChunkStrategy = ChunkStrategy.SECTION
    ordinal: int = Field(default=0, ge=0)
    parent_chunk_id: str | None = None
    overlap_before: int = Field(default=0, ge=0)
    overlap_after: int = Field(default=0, ge=0)
    metadata: dict[str, str | int | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provenance(self) -> "DocumentChunk":
        if self.document.model is None:
            raise ValueError("chunk document.model must be resolved before indexing")
        if self.document.version is None:
            raise ValueError("chunk document.version must be resolved before indexing")
        if self.pages != sorted(set(self.pages)):
            raise ValueError("chunk pages must be sorted and unique")
        if self.page != self.pages[0]:
            raise ValueError("chunk page must be the first page in chunk pages")
        if any(e.page not in self.pages for e in self.evidence):
            raise ValueError("every evidence page must belong to chunk pages")
        if any(e.source_file != self.evidence[0].source_file for e in self.evidence):
            raise ValueError("all evidence must reference the same raw source file")
        return self
