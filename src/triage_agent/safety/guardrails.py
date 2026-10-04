"""Safety guardrails: decide what may run automatically.

Invariants (tested, not aspirational):
    * destructive actions (close/reopen) NEVER auto-run while
      ``destructive_actions_require_confirmation`` is set — no confidence
      score overrides this
    * security-sensitive contexts force comments and assignments to review
    * below ``review_threshold`` nothing runs automatically at all
    * per-repository policy can veto automation for any action kind
    * bulk batches are held for a human even if each action alone is safe

The guardrails only ever *approve* or *hold* — actual rejection is a human
decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from triage_agent.agent.state import (
    ActionKind,
    ActionRisk,
    Classification,
    ProposedAction,
    TriageState,
)
from triage_agent.config import AppConfig
from triage_agent.policies.engine import PolicyEngine
from triage_agent.policies.models import RepositoryPolicy

logger = structlog.get_logger("safety.guardrails")

# More simultaneous mutations than this counts as a "bulk change" and is
# always held for human review.
MAX_AUTO_ACTIONS = 5


@dataclass
class SafetyResult:
    auto_approved: list[ProposedAction] = field(default_factory=list)
    pending_approval: list[ProposedAction] = field(default_factory=list)
    escalation_reasons: list[str] = field(default_factory=list)


def _is_security_context(state: TriageState) -> bool:
    if state.issue_assessment and (
        state.issue_assessment.is_security_sensitive
        or state.issue_assessment.classification is Classification.SECURITY
    ):
        return True
    return bool(
        state.pr_assessment
        and (
            state.pr_assessment.is_security_sensitive
            or state.pr_assessment.classification is Classification.SECURITY
        )
    )


class SafetyGuardrails:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def evaluate(
        self, state: TriageState, policy: RepositoryPolicy
    ) -> SafetyResult:
        result = SafetyResult()
        confidence = state.confidence
        security_context = _is_security_context(state)
        bulk = len(state.proposed_actions) > MAX_AUTO_ACTIONS

        if bulk:
            result.escalation_reasons.append(
                f"bulk change ({len(state.proposed_actions)} actions) requires review"
            )
        if security_context:
            result.escalation_reasons.append(
                "security-sensitive context: comments/assignments require review"
            )

        for action in state.proposed_actions:
            holds = self._holds_for(
                action,
                policy=policy,
                confidence=confidence,
                security_context=security_context,
                bulk=bulk,
            )
            if holds:
                result.pending_approval.append(action)
                result.escalation_reasons.extend(holds)
                logger.info(
                    "safety.hold",
                    kind=action.kind.value,
                    reasons=holds,
                    confidence=confidence,
                )
            else:
                result.auto_approved.append(
                    action.model_copy(update={"requires_approval": False})
                )
                logger.info(
                    "safety.auto_approved", kind=action.kind.value, confidence=confidence
                )
        return result

    def _holds_for(
        self,
        action: ProposedAction,
        *,
        policy: RepositoryPolicy,
        confidence: float,
        security_context: bool,
        bulk: bool,
    ) -> list[str]:
        """Reasons this action must be held for a human. Empty = may auto-run."""
        automation = self._config.automation
        holds: list[str] = []

        # 1. Destructive actions: unconditional confirmation requirement.
        if action.risk is ActionRisk.DESTRUCTIVE:
            if automation.destructive_actions_require_confirmation:
                holds.append(f"{action.kind.value} is destructive and requires confirmation")
            return holds or [f"{action.kind.value} is destructive"]

        # 2. Bulk batches are reviewed as a whole.
        if bulk:
            holds.append("part of a bulk change")

        # 3. Security-sensitive context escalates comments and people actions.
        if security_context and action.kind in (
            ActionKind.POST_COMMENT,
            ActionKind.ASSIGN_ISSUE,
            ActionKind.REQUEST_REVIEW,
        ):
            holds.append("security-sensitive context")

        # 4. Repository policy may veto automation for this kind entirely.
        if not PolicyEngine.automation_allowed(policy, action.kind):
            holds.append(f"repository policy does not allow automatic {action.kind.value}")

        # 5. Confidence gates.
        if confidence < automation.review_threshold:
            holds.append(
                f"confidence {confidence:.2f} below review threshold "
                f"{automation.review_threshold:.2f} — recommendations only"
            )
        elif confidence < automation.safe_action_threshold and action.risk in (
            ActionRisk.LOW,
            ActionRisk.HIGH,
        ):
            holds.append(
                f"confidence {confidence:.2f} below safe-action threshold "
                f"{automation.safe_action_threshold:.2f} for {action.risk.value}-risk action"
            )
        return holds
