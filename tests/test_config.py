from __future__ import annotations

from pathlib import Path

import pytest

from triage_agent.config import (
    AppConfig,
    AutomationConfig,
    DuplicateClosing,
    SecurityConfig,
    load_app_config,
)


def test_defaults_are_safe() -> None:
    cfg = AppConfig()
    assert cfg.automation.safe_action_threshold == 0.90
    assert cfg.automation.review_threshold == 0.70
    assert cfg.automation.destructive_actions_require_confirmation is True
    assert cfg.duplicates.close_threshold == 0.90


def test_threshold_ordering_enforced() -> None:
    with pytest.raises(ValueError):
        AutomationConfig(safe_action_threshold=0.5, review_threshold=0.9)


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load_app_config(tmp_path / "nope.yaml")
    assert cfg.automation.safe_action_threshold == 0.90


def test_load_yaml_and_policy_resolution(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
automation:
  safe_action_threshold: 0.95
  review_threshold: 0.60
repositories:
  default:
    automatic_labels: true
  acme/backend:
    automatic_labels: true
    automatic_assignment: true
    duplicate_closing: human_confirmation
    priority_labels:
      critical: P0
      high: P1
"""
    )
    cfg = load_app_config(config_file)
    assert cfg.automation.safe_action_threshold == 0.95

    backend = cfg.policy_for("acme", "backend")
    assert backend.automatic_assignment is True
    assert backend.priority_labels["critical"] == "P0"
    assert backend.duplicate_closing is DuplicateClosing.HUMAN_CONFIRMATION

    # Unknown repo falls back to default
    other = cfg.policy_for("acme", "frontend")
    assert other.automatic_assignment is False


def test_repo_allowlist() -> None:
    open_security = SecurityConfig()
    assert open_security.repo_allowed("any", "repo")

    restricted = SecurityConfig(allowed_repositories=["Acme/Backend"])
    assert restricted.repo_allowed("acme", "backend")  # case-insensitive
    assert not restricted.repo_allowed("acme", "frontend")
