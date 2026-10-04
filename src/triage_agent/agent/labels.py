"""Label grounding: the model may only suggest labels that actually exist."""

from __future__ import annotations

from triage_agent.github.models import RepositoryContext


def filter_labels(suggested: list[str], ctx: RepositoryContext | None) -> list[str]:
    """Keep only labels that exist in the repository, preserving the
    repository's own casing and dropping duplicates."""
    if ctx is None:
        return []
    canonical = {label.name.lower(): label.name for label in ctx.available_labels}
    seen: set[str] = set()
    result: list[str] = []
    for name in suggested:
        canonical_name = canonical.get(name.lower())
        if canonical_name and canonical_name not in seen:
            seen.add(canonical_name)
            result.append(canonical_name)
    return result
