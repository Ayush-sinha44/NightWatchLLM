"""
Thin async wrapper around the Groq Python SDK.

Groq exposes an OpenAI-compatible API, so the SDK usage is nearly identical.
Install with: pip install groq
"""

from groq import AsyncGroq

from app.config import settings

_DEFAULT_SYSTEM = (
    "You are a helpful, accurate, and concise assistant. "
    "Answer the user's questions clearly and directly."
)


class LLMClient:
    """Async client for Groq completions."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def complete(
        self,
        message: str,
        history: list,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt or _DEFAULT_SYSTEM}]

        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": message})

        response = await self._client.chat.completions.create(
            model=settings.model,
            max_tokens=max_tokens or settings.default_max_tokens,
            messages=messages,
        )

        return response.choices[0].message.content or ""