"""Narrow, local tools exposed to Friday's single diagnostic agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .. import prompts
from .models import EvidenceContext


@dataclass(frozen=True)
class AgentToolResult:
    """Tool output supplied back to the model, plus its validated evidence."""

    content: str
    evidence: list[EvidenceContext] = field(default_factory=list)


AgentToolExecutor = Callable[[str, dict[str, Any]], Awaitable[AgentToolResult]]
AGENT_TOOLS = prompts.AGENT_TOOLS
