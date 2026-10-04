"""Human approval flow: rendering proposed actions and applying decisions."""

from __future__ import annotations

from typing import Any

from triage_agent.agent.state import ApprovalDecision, ProposedAction, TriageState


def render_approval_request(state: TriageState, pending: list[ProposedAction]) -> str:
    """Human-readable approval prompt for the pending actions."""
    lines: list[str] = [
        "PROPOSED ACTION" + ("S" if len(pending) > 1 else ""),
        f"Repository: {state.repository}  •  #{state.issue_number}",
        "",
    ]
    for i, action in enumerate(pending):
        lines.append(f"[{i}] {action.describe()}")
        lines.append(f"    Risk: {action.risk.value}")
        lines.append(f"    Reason: {action.reason}")
        if action.evidence:
            lines.append("    Evidence:")
            lines.extend(f"      - {item}" for item in action.evidence)
        lines.append("")
    lines.append("Approve? [yes/no]")
    return "\n".join(lines)


def parse_resume_payload(
    payload: Any, pending: list[ProposedAction]
) -> list[ApprovalDecision]:
    """Normalise the resume value of an interrupt into per-action decisions.

    Accepted shapes:
      * ``True`` / ``False`` / ``"yes"`` / ``"no"`` — decide all pending at once
      * list of dicts / ``ApprovalDecision`` — per-action decisions
    Any pending action without an explicit decision is treated as NOT approved.
    """
    if isinstance(payload, bool):
        return [
            ApprovalDecision(action_index=i, approved=payload)
            for i in range(len(pending))
        ]
    if isinstance(payload, str):
        approved = payload.strip().lower() in {"y", "yes", "approve", "approved"}
        return [
            ApprovalDecision(action_index=i, approved=approved)
            for i in range(len(pending))
        ]
    if isinstance(payload, list):
        decisions = []
        for item in payload:
            if isinstance(item, ApprovalDecision):
                decisions.append(item)
            else:
                decisions.append(ApprovalDecision.model_validate(item))
        return decisions
    raise ValueError(f"unsupported approval payload: {type(payload).__name__}")


def apply_decisions(
    pending: list[ProposedAction], decisions: list[ApprovalDecision]
) -> tuple[list[ProposedAction], list[ProposedAction]]:
    """Split pending actions into (approved, rejected) per the decisions.
    Undecided actions default to rejected — silence never approves."""
    by_index = {d.action_index: d for d in decisions}
    approved: list[ProposedAction] = []
    rejected: list[ProposedAction] = []
    for i, action in enumerate(pending):
        decision = by_index.get(i)
        if decision is not None and decision.approved:
            approved.append(action.model_copy(update={"requires_approval": False}))
        else:
            rejected.append(action)
    return approved, rejected
