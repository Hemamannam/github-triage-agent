"""Application configuration.

Two layers, kept deliberately separate:

* ``Settings``  — secrets and runtime switches, loaded from the environment
  (or a ``.env`` file). Never persisted, never logged.
* ``AppConfig`` — behavioural configuration (LLM, automation thresholds,
  per-repository policies), loaded from a YAML file so different deployments
  can be tuned without code changes.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from triage_agent.policies.models import DuplicateClosing, RepositoryPolicy

__all__ = [
    "AppConfig",
    "AutomationConfig",
    "DuplicateClosing",
    "DuplicateConfig",
    "LLMConfig",
    "LLMProvider",
    "RepositoryPolicy",
    "SecurityConfig",
    "Settings",
    "load_app_config",
]


class Settings(BaseSettings):
    """Environment-derived settings. Secrets live here and only here."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    github_token: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")

    dry_run: bool = True
    triage_config: Path = Path("config.yaml")
    triage_db_path: Path = Path("triage_agent.db")
    log_format: str = "json"  # json | console
    log_level: str = "INFO"


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-opus-5"
    max_tokens: int = Field(default=4096, ge=256)
    timeout_seconds: float = Field(default=120.0, gt=0)


class AutomationConfig(BaseModel):
    safe_action_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    destructive_actions_require_confirmation: bool = True

    def model_post_init(self, __context: Any) -> None:
        if self.review_threshold > self.safe_action_threshold:
            raise ValueError("review_threshold must be <= safe_action_threshold")


class DuplicateConfig(BaseModel):
    max_candidates: int = Field(default=5, ge=1, le=20)
    close_threshold: float = Field(default=0.90, ge=0.0, le=1.0)


class SecurityConfig(BaseModel):
    allowed_repositories: list[str] = Field(default_factory=list)

    def repo_allowed(self, owner: str, repo: str) -> bool:
        if not self.allowed_repositories:
            return True
        return f"{owner}/{repo}".lower() in {r.lower() for r in self.allowed_repositories}


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    duplicates: DuplicateConfig = Field(default_factory=DuplicateConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    repositories: dict[str, RepositoryPolicy] = Field(
        default_factory=lambda: {"default": RepositoryPolicy()}
    )

    def policy_for(self, owner: str, repo: str) -> RepositoryPolicy:
        """Resolve the policy for a repository, falling back to ``default``."""
        full = f"{owner}/{repo}"
        for key, policy in self.repositories.items():
            if key.lower() == full.lower():
                return policy
        return self.repositories.get("default", RepositoryPolicy())


def load_app_config(path: Path | str | None = None) -> AppConfig:
    """Load YAML config; a missing file yields safe defaults."""
    if path is None:
        return AppConfig()
    p = Path(path)
    if not p.exists():
        return AppConfig()
    raw = yaml.safe_load(p.read_text()) or {}
    return AppConfig.model_validate(raw)
