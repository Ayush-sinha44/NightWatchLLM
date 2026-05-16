"""
Request and response schemas for the gateway API.
"""

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    """A single turn in a conversation (mirrors the Anthropic messages format)."""

    role: str = Field(..., pattern="^(user|assistant)$", examples=["user", "assistant"])
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Payload sent by the caller to the gateway."""

    message: str = Field(..., min_length=1, description="The latest user message.")
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="Previous turns in the conversation (oldest first).",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt to prepend. Overrides the gateway default.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=8096,
        description="Max tokens in the response. Defaults to the gateway setting.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is the capital of France?",
                    "history": [],
                },
                {
                    "message": "And what is it famous for?",
                    "history": [
                        {"role": "user", "content": "What is the capital of France?"},
                        {"role": "assistant", "content": "The capital of France is Paris."},
                    ],
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """Payload returned by the gateway after a successful LLM call."""

    reply: str = Field(..., description="The LLM's response text.")
    model: str = Field(..., description="The model that produced the response.")
    elapsed_ms: int = Field(..., description="Round-trip time to the LLM in milliseconds.")
    # Phase 2: output guardrail fields
    blocked: bool = Field(default=False, description="True if the response was blocked by an output guardrail.")
    reason: str = Field(default="", description="Reason the response was blocked (empty when not blocked).")


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable description.")