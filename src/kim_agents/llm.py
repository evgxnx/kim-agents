"""LLM access. Agents talk to an `LLMClient`; OpenAI in real runs, a scripted fake in tests and offline demos."""

from dataclasses import dataclass, field
from typing import Protocol

import openai


class TransientLLMError(Exception):
    """A failure worth retrying: timeout, rate limit, 5xx, dropped connection."""


class FatalLLMError(Exception):
    """A failure retries will not fix: bad key, unknown model, rejected request."""


@dataclass
class LLMResponse:
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass
class LLMRequest:
    agent: str
    task: str
    model: str
    system: str
    user: str
    payload: dict = field(default_factory=dict)  # the structured input behind `user`, for fakes


class LLMClient(Protocol):
    async def complete(self, request: LLMRequest, timeout: float) -> LLMResponse: ...


class OpenAIClient:
    def __init__(self, api_key: str):
        # Retries are done by the agent so that every attempt is logged.
        self._client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)

    async def complete(self, request: LLMRequest, timeout: float) -> LLMResponse:
        try:
            resp = await self._client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
                response_format={"type": "json_object"},
                timeout=timeout,
            )
        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise TransientLLMError(f"{type(e).__name__}: {e}") from e
        except openai.OpenAIError as e:
            raise FatalLLMError(f"{type(e).__name__}: {e}") from e
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else None,
            tokens_out=usage.completion_tokens if usage else None,
        )
