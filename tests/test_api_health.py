# tests/test_api_health.py

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
    """Basic health check endpoint should return 200 and 'ok'."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    # Adjust this if your health response schema is different
    assert data.get("status") == "ok"


def test_chat_endpoint_with_stubbed_router(monkeypatch):
    """
    Test /api/chat without requiring Ollama or the full pipeline.

    We monkeypatch src.api.routes.answer_query to return a static
    payload, and just verify the API contract + plumbing.
    """
    from src.api import routes as routes_module

    def fake_answer_query(user_query: str, membership_nbr=None, project_root=None):
        return {
            "answer": "stubbed answer",
            "intent": "general_help",
            "membership_nbr": membership_nbr,
        }

    monkeypatch.setattr(routes_module, "answer_query", fake_answer_query)

    payload = {"user_query": "Test question from pytest"}
    resp = client.post("/api/chat", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("answer") == "stubbed answer"
    assert data.get("intent") == "general_help"
