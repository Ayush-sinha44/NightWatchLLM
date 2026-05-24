"""
LLM Guardrails Gateway — Phase 4
A FastAPI server that forwards user messages to Groq with API-key
authentication, input guardrails (PII detection, prompt-injection /
jailbreak blocking), and output guardrails (toxic-content filtering).
"""

import time
import logging
from contextlib import asynccontextmanager

from groq import APIConnectionError, APIStatusError
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import ChatRequest, ChatResponse, ErrorResponse
from app.llm_client import LLMClient
from app.guardrails.output_guard import OutputGuard
from app.guardrails.input_guard import InputGuard
from app.auth import verify_api_key

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("gateway")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 LLM Guardrails Gateway starting up (Phase 4 — auth + input + output guardrails)")
    app.state.llm_client = LLMClient()
    yield
    logger.info("🛑 Gateway shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Guardrails Gateway",
    description=(
        "Middleware service that sits between a user and an LLM, "
        "enforcing safety and compliance rules. "
        "Phase 4: API key authentication + input/output guardrails."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(APIStatusError)
async def groq_status_handler(_: Request, exc: APIStatusError):
    logger.error("Groq API error: status=%s body=%s", exc.status_code, exc.body)
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(
            error="upstream_error",
            message=f"The LLM returned an error: {exc}",
        ).model_dump(),
    )


@app.exception_handler(APIConnectionError)
async def groq_connection_handler(_: Request, exc: APIConnectionError):
    logger.error("Groq connection error: %s", exc)
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="upstream_unavailable",
            message="Could not reach the LLM. Check your network or API key.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe."""
    return {"status": "ok", "phase": 4}


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["gateway"],
    summary="Send a message through the gateway to the LLM",
)
async def chat(
    request: ChatRequest,
    http_request: Request,
    _: None = Depends(verify_api_key),
):
    """
    Gateway endpoint with authentication + input + output guardrails.

    0. Phase 4 API-key auth is enforced via the verify_api_key dependency.
    1. Runs Phase 3 input guardrails (PII, injection, jailbreak, toxic, LLM judge).
    2. Forwards the message to Groq if it passes.
    3. Runs Phase 2 output guardrails on the LLM response.

    Phase 5 hook: add rate-limiting, analytics, or policy-engine hooks.
    """
    client: LLMClient = http_request.app.state.llm_client

    logger.info(
        "chat request | model=%s | history_turns=%d | user_msg_len=%d",
        settings.model,
        len(request.history),
        len(request.message),
    )

    # ----- Phase 3: Input Guardrail ------------------------------------
    input_guard = InputGuard()
    input_result = await input_guard.check(request.message)

    if input_result.blocked:
        logger.warning(
            "input BLOCKED | reason=%s | user_msg_len=%d",
            input_result.reason,
            len(request.message),
        )
        return ChatResponse(
            reply="I can't process that request.",
            model=settings.model,
            elapsed_ms=0,
            blocked=True,
            reason=input_result.reason,
        )
    # -------------------------------------------------------------------

    # Phase 4 hook: add pre-LLM processing (e.g. PII redaction, rate-limiting) here.

    t0 = time.perf_counter()
    response_text = await client.complete(
        message=request.message,
        history=request.history,
        system_prompt=request.system_prompt,
        max_tokens=request.max_tokens,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    logger.info(
        "chat response | elapsed_ms=%d | response_len=%d",
        elapsed_ms,
        len(response_text),
    )

    # ----- Phase 2: Output Guardrail -----------------------------------
    output_guard = OutputGuard()
    guard_result = await output_guard.check(response_text)

    if guard_result.blocked:
        logger.warning(
            "output BLOCKED | reason=%s | original_len=%d",
            guard_result.reason,
            len(response_text),
        )
        return ChatResponse(
            reply="I can't help with that.",
            model=settings.model,
            elapsed_ms=elapsed_ms,
            blocked=True,
            reason=guard_result.reason,
        )
    # -------------------------------------------------------------------

    return ChatResponse(
        reply=response_text,
        model=settings.model,
        elapsed_ms=elapsed_ms,
        blocked=False,
    )