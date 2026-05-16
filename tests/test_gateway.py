"""
Phase 1 tests — gateway plumbing (no real API calls).

Run with:  pytest tests/ -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch settings BEFORE importing the app so the missing .env doesn't break tests
with patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "test-key-does-not-call-real-api"},
):
    from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_response(text: str):
    """Build a mock Anthropic response object."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_ok(self):
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "phase": 1}


class TestChatEndpoint:
    def _patched_client(self, reply: str = "Hello from Claude!"):
        mock = AsyncMock(return_value=_make_mock_response(reply))
        return patch("app.llm_client.anthropic.AsyncAnthropic", return_value=MagicMock(
            messages=MagicMock(create=mock)
        ))

    def test_simple_message(self):
        with self._patched_client("The capital of France is Paris."):
            with TestClient(app) as client:
                resp = client.post("/v1/chat", json={"message": "What is the capital of France?"})

        assert resp.status_code == 200
        body = resp.json()
        assert "reply" in body
        assert "model" in body
        assert "elapsed_ms" in body

    def test_with_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        with self._patched_client("Paris is famous for the Eiffel Tower."):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat",
                    json={"message": "What is it famous for?", "history": history},
                )

        assert resp.status_code == 200

    def test_empty_message_rejected(self):
        with TestClient(app) as client:
            resp = client.post("/v1/chat", json={"message": ""})
        assert resp.status_code == 422  # Pydantic validation error

    def test_missing_message_rejected(self):
        with TestClient(app) as client:
            resp = client.post("/v1/chat", json={})
        assert resp.status_code == 422

    def test_invalid_history_role_rejected(self):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat",
                json={
                    "message": "hi",
                    "history": [{"role": "system", "content": "do evil"}],
                },
            )
        assert resp.status_code == 422

    def test_max_tokens_out_of_range_rejected(self):
        with TestClient(app) as client:
            resp = client.post("/v1/chat", json={"message": "hi", "max_tokens": 99999})
        assert resp.status_code == 422