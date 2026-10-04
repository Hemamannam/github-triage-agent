"""Strongly-typed agent state and decision schemas.

Every critical decision the LLM makes is validated through these Pydantic
models — decisions are never parsed out of unstructured model text.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from triage_agent.github.models import (
    IssueData,
    PullRequestData,
    RepositoryContext,
    SearchIssueItem,
)


class ObjectType(StrEnum):
    ISSUE = "issue"
    PULL_REQUEST = "pull_request"


class Classification(StrEnum):
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    DOCUMENTATION = "documentation"
    QUESTION = "question"
    SECURITY = "security"
    INVALID = "invalid"
    OTHER = "other"


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Priority(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionRisk(StrEnum):
    SAFE = "safe"            # e.g. add labels
    LOW = "low"              # e.g. post an informational comment
    HIGH = "high"            # e.g. assign a person, request review
    DESTRUCTIVE = "destructive"  # e.g. close / reopen an issue


class ActionKind(StrEnum):
    ADD_LABELS = "add_labels"
    POST_COMMENT = "post_comment"
    ASSIGN_ISSUE = "assign_issue"
    REQUEST_REVIEW = "request_review"
    CLOSE_ISSUE = "close_issue"
    REOPEN_ISSUE = "reopen_issue"


# Baseline risk of each action kind; the safety layer may escalate
# (e.g. a security-related comment is treated as high risk).
ACTION_BASE_RISK: dict[ActionKind, ActionRisk] = {
    ActionKind.ADD_LABELS: ActionRisk.SAFE,
    ActionKind.POST_COMMENT: ActionRisk.LOW,
    ActionKind.ASSIGN_ISSUE: ActionRisk.HIGH,
    ActionKind.REQUEST_REVIEW: ActionRisk.HIGH,
    ActionKind.CLOSE_ISSUE: ActionRisk.DESTRUCTIVE,
    ActionKind.REOPEN_ISSUE: ActionRisk.DESTRUCTIVE,
}


# --------------------------------------------------------------------- LLM I/O


class IssueAssessment(BaseModel):
    """Structured output of the classification step for issues."""

    model_config = ConfigDict(extra="forbid")

    classification: Classification
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity
    priority: Priority
    suggested_labels: list[str] = Field(default_factory=list, max_length=8)
    missing_information: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Concrete items a bug report lacks (e.g. 'package version', 'traceback')",
    )
    is_security_sensitive: bool = False
    search_keywords: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Keywords for finding related/duplicate issues",
    )
    reasoning_summary: str = Field(max_length=1500)


class PRAssessment(BaseModel):
    """Structured output of the classification step for pull requests."""

    model_config = ConfigDict(extra="forbid")

    classification: Classification
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity = Severity.NONE
    priority: Priority = Priority.MEDIUM
    suggested_labels: list[str] = Field(default_factory=list, max_length=8)
    missing_tests: bool = False
    missing_documentation: bool = False
    unusually_large: bool = False
    is_security_sensitive: bool = False
    ci_failing: bool = False
    summary: str = Field(max_length=1500)
    reasoning_summary: str = Field(max_length=1500)


class DuplicateCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_number: int
    title: str
    similarity: float = Field(ge=0.0, le=1.0, description="Lexical similarity score")
    llm_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_duplicate: bool = False
    reason: str = ""


class DuplicateVerdict(BaseModel):
    """Structured output of the LLM duplicate comparison."""

    model_config = ConfigDict(extra="forbid")

    duplicate_of: int | None = Field(
        default=None, description="Issue number this duplicates, or null"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=1000)


class CommentDraft(BaseModel):
    """Structured output when the LLM drafts a comment."""

    model_config = ConfigDict(extra="forbid")

    body: str = Field(min_length=1, max_length=10_000)


# --------------------------------------------------------------------- actions


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ActionKind
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    risk: ActionRisk = ActionRisk.SAFE
    requires_approval: bool = True  # safety layer must explicitly relax this
    evidence: list[str] = Field(default_factory=list)

    def describe(self) -> str:
        if self.kind is ActionKind.ADD_LABELS:
            return f"Add labels: {', '.join(self.params.get('labels', []))}"
        if self.kind is ActionKind.POST_COMMENT:
            body = str(self.params.get("comment", ""))
            return f"Post comment ({len(body)} chars)"
        if self.kind is ActionKind.ASSIGN_ISSUE:
            return f"Assign: {', '.join(self.params.get('assignees', []))}"
        if self.kind is ActionKind.REQUEST_REVIEW:
            return f"Request review from: {', '.join(self.params.get('reviewers', []))}"
        if self.kind is ActionKind.CLOSE_ISSUE:
            return f"Close as {self.params.get('state_reason', 'not_planned')}"
        return "Reopen issue"


class ExecutionStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"          # idempotency guard (already applied)
    FAILED = "failed"
    BLOCKED = "blocked"          # safety refused
    DRY_RUN = "dry_run"


class ExecutedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ProposedAction
    status: ExecutionStatus
    detail: str = ""
    latency_ms: float = 0.0
    retries: int = 0


class ApprovalDecision(BaseModel):
    """A human decision about one proposed action."""

    model_config = ConfigDict(extra="forbid")

    action_index: int
    approved: bool
    note: str = ""


# ----------------------------------------------------------------------- state


class TriageState(BaseModel):
    """The full state carried through the LangGraph triage run."""

    model_config = ConfigDict(extra="forbid")

    # identity
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    owner: str
    repo: str
    issue_number: int
    object_type: ObjectType = ObjectType.ISSUE

    # gathered context
    issue_data: IssueData | None = None
    pr_data: PullRequestData | None = None
    repository_context: RepositoryContext | None = None

    # analysis
    recent_activity: list[SearchIssueItem] = Field(default_factory=list)
    issue_assessment: IssueAssessment | None = None
    pr_assessment: PRAssessment | None = None
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)
    duplicate_verdict: DuplicateVerdict | None = None
    codeowner_suggestions: list[str] = Field(default_factory=list)

    # planning & execution
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    pending_approval: list[ProposedAction] = Field(default_factory=list)
    approved_actions: list[ProposedAction] = Field(default_factory=list)
    rejected_actions: list[ProposedAction] = Field(default_factory=list)
    executed_actions: list[ExecutedAction] = Field(default_factory=list)

    # bookkeeping
    tool_errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    escalated_to_human: bool = False
    escalation_reason: str = ""
    final_summary: str = ""
    error: str | None = None

    @property
    def repository(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def classification(self) -> Classification | None:
        if self.object_type is ObjectType.PULL_REQUEST:
            return self.pr_assessment.classification if self.pr_assessment else None
        return self.issue_assessment.classification if self.issue_assessment else None

    @property
    def confidence(self) -> float:
        if self.object_type is ObjectType.PULL_REQUEST:
            return self.pr_assessment.confidence if self.pr_assessment else 0.0
        return self.issue_assessment.confidence if self.issue_assessment else 0.0
