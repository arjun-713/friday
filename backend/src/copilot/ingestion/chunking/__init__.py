"""Structure detection and future chunk construction."""

from .exact_matches import extract_exact_matches
from .generator import ChunkingConfig, generate_chunks
from .procedures import extract_procedures
from .structure import structure_document
from .tables import extract_table_rows

__all__ = [
    "ChunkingConfig",
    "extract_exact_matches",
    "extract_procedures",
    "extract_table_rows",
    "generate_chunks",
    "structure_document",
]
