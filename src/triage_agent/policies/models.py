"""Per-repository policy models (the FDE-style customer configuration)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class DuplicateClosing(StrEnum):
    HUMAN_CONFIRMATION = "human_confirmation"
    DISABLED = "disabled"


class RepositoryPolicy(BaseModel):
    """What this repository's maintainers allow the agent to do on its own."""

    automatic_labels: bool = True
    automatic_assignment: bool = False
    automatic_comments: bool = False
    duplicate_closing: DuplicateClosing = DuplicateClosing.HUMAN_CONFIRMATION
    priority_labels: dict[str, str] = Field(
        default_factory=lambda: {"critical": "P0", "high": "P1", "medium": "P2", "low": "P3"}
    )
