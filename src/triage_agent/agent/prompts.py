"""Prompt construction for the triage LLM calls.

Prompts state goals and constraints; the deterministic pipeline (policy
engine, safety layer, verified assignees) enforces the hard rules. Repository
context is truncated defensively so a giant README cannot blow the context
window.
"""

from __future__ import annotations

from triage_agent.agent.state import TriageState
from triage_agent.github.models import IssueData, PullRequestData, RepositoryContext

_MAX_BODY = 6_000
_MAX_DOC = 2_500
_MAX_COMMENTS = 8
_MAX_FILES = 40


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return "(empty)"
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def _repo_block(ctx: RepositoryContext | None) -> str:
    if ctx is None:
        return "(no repository context available)"
    lines = [
        f"Repository: {ctx.info.full_name}",
        f"Description: {ctx.info.description or '(none)'}",
        f"Topics: {', '.join(ctx.info.topics) or '(none)'}",
        f"Primary language: {ctx.info.language or '(unknown)'}",
        f"Available labels: {', '.join(ctx.available_label_names) or '(none)'}",
    ]
    if ctx.contributing:
        lines.append(f"CONTRIBUTING.md excerpt:\n{_clip(ctx.contributing, _MAX_DOC)}")
    if ctx.readme:
        lines.append(f"README excerpt:\n{_clip(ctx.readme, _MAX_DOC)}")
    return "\n".join(lines)


def _issue_block(issue: IssueData) -> str:
    lines = [
        f"Issue #{issue.number}: {issue.title}",
        f"Author: @{issue.user.login} (association: {issue.author_association})",
        f"Existing labels: {', '.join(issue.label_names) or '(none)'}",
        f"State: {issue.state.value}",
        f"Body:\n{_clip(issue.body, _MAX_BODY)}",
    ]
    if issue.comments:
        lines.append("Comments:")
        for comment in issue.comments[:_MAX_COMMENTS]:
            lines.append(f"- @{comment.user.login}: {_clip(comment.body, 600)}")
    return "\n".join(lines)


ISSUE_CLASSIFY_SYSTEM = """\
You are a senior open-source maintainer triaging GitHub issues. You produce a
single structured assessment used by an automated pipeline; hard rules (which
labels exist, who may be assigned, what runs automatically) are enforced
downstream, so your job is accurate judgment, not caution theater.

Classification meanings:
- bug: a defect in the software with observable incorrect behaviour
- feature_request: asks for new functionality or an enhancement
- documentation: docs are missing, wrong, or unclear
- question: support request / usage question, not a defect
- security: reports or strongly suggests a security vulnerability
- invalid: spam, off-topic, or clearly not actionable in this repository
- other: none of the above fits

Severity is about impact if true (critical = data loss / security / broad
outage, high = major feature broken, medium = feature impaired with
workaround, low = cosmetic). Priority is about what maintainers should do
first and may differ from severity. Confidence is your honest probability
that the classification is correct — do not inflate it; anything that is
ambiguous belongs below 0.7 so a human reviews it.

For bug reports, list concretely what debugging information is missing
(e.g. "package version", "complete traceback", "minimal reproduction steps",
"expected vs actual behaviour", "environment/OS"). An empty list means the
report is complete enough to act on. Only suggest labels from the available
list. Provide 2-5 search keywords that would find duplicates of this issue."""


def build_issue_classify_prompt(state: TriageState) -> str:
    assert state.issue_data is not None
    return (
        f"{_repo_block(state.repository_context)}\n\n"
        f"{_issue_block(state.issue_data)}\n\n"
        "Produce your structured triage assessment of this issue."
    )


PR_CLASSIFY_SYSTEM = """\
You are a senior open-source maintainer triaging pull requests. You produce a
single structured assessment used by an automated pipeline. You never approve
or merge; you assess.

Guidance:
- classification describes what the PR does (bug fix -> bug, new feature ->
  feature_request, docs-only -> documentation, security fix -> security)
- missing_tests: code behaviour changed but no test files were added/updated
  (trust the deterministic signals provided; override only with clear reason)
- missing_documentation: user-facing behaviour changed but no docs updated
- unusually_large / ci_failing: deterministic signals are provided — echo them
- is_security_sensitive: touches auth, crypto, secrets handling, input
  sanitisation, dependency pins, CI permissions, or similar
- summary: 2-4 sentences a reviewer would want before opening the diff
- confidence: honest probability your classification is correct"""


def _pr_block(pr: PullRequestData) -> str:
    files = pr.changed_files[:_MAX_FILES]
    file_lines = [
        f"- {f.filename} ({f.status}, +{f.additions}/-{f.deletions})" for f in files
    ]
    if len(pr.changed_files) > _MAX_FILES:
        file_lines.append(f"... and {len(pr.changed_files) - _MAX_FILES} more files")
    reviews = ", ".join(f"@{r.user.login}:{r.state}" for r in pr.reviews) or "(none)"
    checks = ", ".join(
        f"{c.name}={c.conclusion or c.status}" for c in pr.check_runs
    ) or "(none)"
    return "\n".join(
        [
            f"PR #{pr.number}: {pr.title}",
            f"Author: @{pr.user.login}",
            f"Draft: {pr.draft} | Base: {pr.base_ref} | +{pr.additions}/-{pr.deletions}"
            f" across {len(pr.changed_files)} files",
            f"Linked issues: {pr.linked_issues or '(none)'}",
            f"Reviews: {reviews}",
            f"CI checks: {checks}",
            f"Body:\n{_clip(pr.body, _MAX_BODY)}",
            "Changed files:",
            *file_lines,
        ]
    )


def build_pr_classify_prompt(state: TriageState, deterministic_signals: dict[str, bool]) -> str:
    assert state.pr_data is not None
    signals = "\n".join(f"- {k}: {v}" for k, v in deterministic_signals.items())
    return (
        f"{_repo_block(state.repository_context)}\n\n"
        f"{_pr_block(state.pr_data)}\n\n"
        f"Deterministic signals (computed from the diff and CI, echo these):\n{signals}\n\n"
        "Produce your structured triage assessment of this pull request."
    )


DUPLICATE_SYSTEM = """\
You compare a new GitHub issue against candidate existing issues and decide
whether the new issue is a duplicate of one of them. A duplicate reports the
same underlying problem or request — not merely the same component or a
similar-sounding title. Matching evidence includes: same error/exception,
same reproduction path, same requested capability, same root cause.

If none is a true duplicate, return duplicate_of = null. Confidence is your
honest probability; closing an issue wrongly is costly, so only exceed 0.9
when the evidence is strong (same symptom AND same cause/path)."""


def build_duplicate_prompt(state: TriageState) -> str:
    assert state.issue_data is not None
    candidates = "\n\n".join(
        f"Candidate #{c.issue_number}: {c.title}\n"
        f"(lexical similarity {c.similarity:.2f})\n{c.reason}"
        for c in state.duplicate_candidates
    )
    return (
        f"New issue:\n{_issue_block(state.issue_data)}\n\n"
        f"Candidate existing issues:\n{candidates}\n\n"
        "Decide whether the new issue duplicates one of the candidates."
    )


COMMENT_SYSTEM = """\
You draft a single GitHub comment on behalf of an automated triage assistant.
Requirements: courteous, concise, specific to this issue — never generic
boilerplate. Ask only for the concrete items listed as missing. Do not
promise timelines, do not speculate about causes, do not @-mention anyone.
Start with a one-line thank-you, then a short bulleted list of what is
needed. Sign off as the triage automation (one short line)."""


def build_missing_info_prompt(state: TriageState) -> str:
    assert state.issue_data is not None
    assert state.issue_assessment is not None
    missing = "\n".join(f"- {m}" for m in state.issue_assessment.missing_information)
    return (
        f"Issue #{state.issue_data.number}: {state.issue_data.title}\n"
        f"Body:\n{_clip(state.issue_data.body, 2000)}\n\n"
        f"Missing information to request:\n{missing}\n\n"
        "Draft the comment."
    )
