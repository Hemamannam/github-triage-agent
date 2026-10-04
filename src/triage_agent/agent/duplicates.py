"""Hybrid duplicate detection — retrieval and scoring half.

Step 1: keyword search against GitHub (done by the node).
Step 2: lexical scoring here ranks the returned candidates.
Step 3: the LLM compares only the strongest candidates (node + prompts).
Step 4: the verdict is validated against the candidate set — the model can
        only pick from issues that actually exist.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from triage_agent.agent.state import DuplicateCandidate
from triage_agent.github.models import IssueData, SearchIssueItem

_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "i", "in", "is", "it", "its", "not", "of", "on", "or", "that", "the", "this", "to", "was", "were", "when", "will", "with", "does", "doesn't", "don't", "can", "cannot"]
)
_TOKEN = re.compile(r"[a-z0-9_.]+")


def tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def extract_keywords(title: str, limit: int = 5) -> list[str]:
    """Fallback keyword extraction when the LLM didn't provide any."""
    seen: list[str] = []
    for token in _TOKEN.findall(title.lower()):
        if token in _STOPWORDS or len(token) <= 2 or token in seen:
            continue
        seen.append(token)
    return seen[:limit]


def lexical_similarity(issue: IssueData, candidate: SearchIssueItem) -> float:
    """Blend of title sequence similarity and title+body token overlap."""
    title_ratio = SequenceMatcher(
        None, issue.title.lower(), candidate.title.lower()
    ).ratio()

    a = tokenize(f"{issue.title} {issue.body or ''}")
    b = tokenize(f"{candidate.title} {candidate.body or ''}")
    jaccard = len(a & b) / len(a | b) if a | b else 0.0

    return round(0.6 * title_ratio + 0.4 * jaccard, 4)


def score_candidates(
    issue: IssueData,
    items: list[SearchIssueItem],
    *,
    max_candidates: int,
    min_similarity: float = 0.15,
) -> list[DuplicateCandidate]:
    """Rank search hits by lexical similarity, dropping self and weak hits."""
    scored = [
        DuplicateCandidate(
            issue_number=item.number,
            title=item.title,
            similarity=lexical_similarity(issue, item),
            reason=(item.body or "")[:400],
        )
        for item in items
        if item.number != issue.number and not item.is_pull_request
    ]
    scored = [c for c in scored if c.similarity >= min_similarity]
    scored.sort(key=lambda c: -c.similarity)
    return scored[:max_candidates]
