"""
Phase 4 — API Key Authentication

Validates incoming requests against a static API key stored in the
gateway configuration.  Designed as a FastAPI dependency so it can be
injected into individual routes without affecting public endpoints.

Usage:
    from app.auth import verify_api_key

    @app.post("/v1/chat", dependencies=[Depends(verify_api_key)])
    async def chat(...):
        ...
"""

import logging

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger("gateway.auth")

# Read the key from the X-API-Key header.
# auto_error=False so we can return a clean 401 ourselves instead of
# FastAPI's default 403 when the header is missing.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Depends(_api_key_header),
) -> None:
    """FastAPI dependency that enforces API-key authentication.

    Raises:
        HTTPException 401 — header is missing.
        HTTPException 403 — header is present but the key is invalid.

    Returns:
        None on success (silently allows the request through).
    """
    # Phase 5: replace this with a database lookup for per-client keys

    if api_key is None:
        logger.warning("Auth failed: missing X-API-Key header")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Pass it as X-API-Key header.",
        )

    if api_key != settings.nightwatch_api_key:
        logger.warning("Auth failed: invalid API key")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )
