"""Policy engine: turns repository policy + confidence into concrete gates.

The engine answers two questions the planner and safety layer ask:
  1. May this action kind ever run automatically in this repository?
  2. Which label does a priority level map to here?

Confidence-based gating (thresholds) lives in the safety layer; the policy
engine is purely about what the repository's maintainers have allowed.
"""

from __future__ import annotations

from triage_agent.agent.state import ActionKind, Priority
from triage_agent.config import AppConfig
from triage_agent.policies.models import DuplicateClosing, RepositoryPolicy


class PolicyEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def policy_for(self, owner: str, repo: str) -> RepositoryPolicy:
        return self._config.policy_for(owner, repo)

    @staticmethod
    def automation_allowed(policy: RepositoryPolicy, kind: ActionKind) -> bool:
        """May this action kind run without a human, as far as the repository
        policy is concerned? (Confidence gates may still block it.)"""
        if kind is ActionKind.ADD_LABELS:
            return policy.automatic_labels
        if kind is ActionKind.POST_COMMENT:
            return policy.automatic_comments
        if kind in (ActionKind.ASSIGN_ISSUE, ActionKind.REQUEST_REVIEW):
            return policy.automatic_assignment
        # close / reopen are never automatic at the policy level
        return False

    @staticmethod
    def duplicate_closing_enabled(policy: RepositoryPolicy) -> bool:
        return policy.duplicate_closing is not DuplicateClosing.DISABLED

    @staticmethod
    def priority_label(policy: RepositoryPolicy, priority: Priority) -> str | None:
        return policy.priority_labels.get(priority.value)
