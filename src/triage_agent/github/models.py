"""Pydantic domain models for GitHub objects.

These are deliberately thinner than the raw REST payloads: the agent only
sees fields that matter for triage decisions, which keeps LLM context small
and makes fakes trivial to build in tests.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GitHubModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class User(GitHubModel):
    login: str
    type: str = "User"
    site_admin: bool = False


class Label(GitHubModel):
    name: str
    color: str = ""
    description: str | None = None


class Comment(GitHubModel):
    id: int
    user: User
    body: str = ""
    created_at: datetime | None = None
    author_association: str = "NONE"


class IssueState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class IssueData(GitHubModel):
    number: int
    title: str
    body: str | None = None
    state: IssueState = IssueState.OPEN
    user: User
    labels: list[Label] = Field(default_factory=list)
    assignees: list[User] = Field(default_factory=list)
    comments_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    author_association: str = "NONE"
    html_url: str = ""
    is_pull_request: bool = False
    comments: list[Comment] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _remap_raw_payload(cls, data: object) -> object:
        # GitHub's REST payload uses "comments" for the *count*; our
        # ``comments`` field holds the actual comment objects.
        if isinstance(data, dict) and isinstance(data.get("comments"), int):
            data = dict(data)
            data["comments_count"] = data.pop("comments")
        return data

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]


class SearchIssueItem(GitHubModel):
    number: int
    title: str
    body: str | None = None
    state: IssueState = IssueState.OPEN
    labels: list[Label] = Field(default_factory=list)
    html_url: str = ""
    is_pull_request: bool = False


class SearchResult(GitHubModel):
    total_count: int = 0
    items: list[SearchIssueItem] = Field(default_factory=list)


class RepoInfo(GitHubModel):
    full_name: str
    description: str | None = None
    topics: list[str] = Field(default_factory=list)
    default_branch: str = "main"
    open_issues_count: int = 0
    language: str | None = None


class RepositoryContext(GitHubModel):
    """Everything the agent needs to reason about a repository."""

    info: RepoInfo
    readme: str | None = None
    contributing: str | None = None
    codeowners: str | None = None
    available_labels: list[Label] = Field(default_factory=list)

    @property
    def available_label_names(self) -> list[str]:
        return [label.name for label in self.available_labels]


class FileChange(GitHubModel):
    filename: str
    status: str = "modified"  # added | removed | modified | renamed
    additions: int = 0
    deletions: int = 0
    patch: str | None = None


class CommitInfo(GitHubModel):
    sha: str
    message: str = ""


class Review(GitHubModel):
    user: User
    state: str = "COMMENTED"  # APPROVED | CHANGES_REQUESTED | COMMENTED | DISMISSED


class CheckRun(GitHubModel):
    name: str
    status: str = "completed"  # queued | in_progress | completed
    conclusion: str | None = None  # success | failure | neutral | cancelled | ...


class PullRequestData(GitHubModel):
    number: int
    title: str
    body: str | None = None
    state: IssueState = IssueState.OPEN
    user: User
    labels: list[Label] = Field(default_factory=list)
    assignees: list[User] = Field(default_factory=list)
    draft: bool = False
    merged: bool = False
    base_ref: str = "main"
    head_ref: str = ""
    additions: int = 0
    deletions: int = 0
    changed_files_count: int = 0
    commits_count: int = 0
    html_url: str = ""
    changed_files: list[FileChange] = Field(default_factory=list)
    commits: list[CommitInfo] = Field(default_factory=list)
    reviews: list[Review] = Field(default_factory=list)
    check_runs: list[CheckRun] = Field(default_factory=list)
    linked_issues: list[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _remap_raw_payload(cls, data: object) -> object:
        # GitHub's raw REST payload uses "changed_files" and "commits" for
        # *counts*; our fields of those names hold the actual objects.
        if isinstance(data, dict):
            remapped = dict(data)
            if isinstance(remapped.get("changed_files"), int):
                remapped["changed_files_count"] = remapped.pop("changed_files")
            if isinstance(remapped.get("commits"), int):
                remapped["commits_count"] = remapped.pop("commits")
            return remapped
        return data

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]

    @property
    def total_diff_size(self) -> int:
        return self.additions + self.deletions

    @property
    def ci_failing(self) -> bool:
        return any(
            run.status == "completed" and run.conclusion in {"failure", "timed_out"}
            for run in self.check_runs
        )
