from enum import StrEnum

from pydantic import BaseModel, Field


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


class Evidence(BaseModel):
    page: int = Field(gt=0)
    section: str
    content: str
    coordinates: tuple[float, float, float, float] | None = None


class DocumentChunk(BaseModel):
    chunk_id: str
    document: SourceDocument
    page: int = Field(gt=0)
    section: str
    content: str
    kind: ChunkKind
    parser: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(min_length=1)
