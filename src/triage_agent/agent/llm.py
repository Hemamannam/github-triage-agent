"""LLM backends producing schema-validated structured output.

Critical decisions are never parsed from free text: every call goes through
``messages.parse`` (Anthropic) / ``responses.parse`` (OpenAI), which validate
against the Pydantic schema. Malformed output is retried once with the
validation error included, then surfaces as ``LLMError`` for the caller's
graceful-degradation path.
"""

from __future__ import annotations

from typing import TypeVar

import structlog
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from triage_agent.config import LLMConfig, LLMProvider, Settings

logger = structlog.get_logger("agent.llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The model backend failed to produce valid structured output."""


class MalformedOutputError(LLMError):
    """The model returned output that failed schema validation."""


class AnthropicBackend:
    """Anthropic Messages API with native structured outputs."""

    def __init__(self, api_key: str, config: LLMConfig) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=config.timeout_seconds
        )
        self._config = config

    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        import anthropic

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_random_exponential(multiplier=1, max=20),
            retry=retry_if_exception_type(
                (anthropic.APIStatusError, anthropic.APIConnectionError, MalformedOutputError)
            ),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "llm.retry",
                        attempt=attempt.retry_state.attempt_number,
                        schema=schema.__name__,
                    )
                try:
                    response = await self._client.messages.parse(
                        model=self._config.model,
                        max_tokens=self._config.max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": prompt}],
                        output_format=schema,
                    )
                except anthropic.BadRequestError as exc:
                    raise LLMError(f"bad request to Anthropic API: {exc}") from exc
                if response.parsed_output is None:
                    raise MalformedOutputError(
                        f"model returned no parseable {schema.__name__}"
                    )
                return response.parsed_output
        raise LLMError("retry loop exhausted")  # pragma: no cover


class OpenAIBackend:
    """OpenAI Responses API with structured outputs (optional extra)."""

    def __init__(self, api_key: str, config: LLMConfig) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "openai extra not installed: pip install 'github-triage-agent[openai]'"
            ) from exc
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=config.timeout_seconds)
        self._config = config

    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        import openai

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_random_exponential(multiplier=1, max=20),
            retry=retry_if_exception_type(
                (openai.APIStatusError, openai.APIConnectionError, MalformedOutputError)
            ),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await self._client.responses.parse(
                        model=self._config.model,
                        instructions=system,
                        input=prompt,
                        text_format=schema,
                    )
                except (openai.BadRequestError, ValidationError) as exc:
                    raise MalformedOutputError(str(exc)) from exc
                parsed = response.output_parsed
                if not isinstance(parsed, schema):
                    raise MalformedOutputError(
                        f"model returned no parseable {schema.__name__}"
                    )
                return parsed
        raise LLMError("retry loop exhausted")  # pragma: no cover


def build_llm(settings: Settings, config: LLMConfig) -> AnthropicBackend | OpenAIBackend | None:
    """Construct the configured backend, or None if no API key is present."""
    if config.provider is LLMProvider.ANTHROPIC and settings.anthropic_api_key.get_secret_value():
        return AnthropicBackend(settings.anthropic_api_key.get_secret_value(), config)
    if config.provider is LLMProvider.OPENAI and settings.openai_api_key.get_secret_value():
        return OpenAIBackend(settings.openai_api_key.get_secret_value(), config)
    logger.warning("llm.no_backend", provider=config.provider.value)
    return None
