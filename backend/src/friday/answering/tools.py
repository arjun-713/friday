"""Narrow, local tools exposed to Friday's single diagnostic agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .models import EvidenceContext


@dataclass(frozen=True)
class AgentToolResult:
    """Tool output supplied back to the model, plus its validated evidence."""

    content: str
    evidence: list[EvidenceContext] = field(default_factory=list)


AgentToolExecutor = Callable[[str, dict[str, Any]], Awaitable[AgentToolResult]]


AGENT_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": "Search the selected manufacturer's manuals for evidence about a symptom, component, or procedure.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_error_code",
            "description": "Look up an exact reported error code or status code in the selected manuals.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_manual_page",
            "description": "Inspect already retrieved evidence from one manufacturer manual page.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "document_id": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                },
                "required": ["document_id", "page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diagnostic_state",
            "description": "Read known user observations, attempted actions, and unresolved observation request for this session.",
            "parameters": {"type": "object", "additionalProperties": False, "properties": {}},
        },
    },
]
