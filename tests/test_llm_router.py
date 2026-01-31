# tests/test_llm_router.py

from unittest.mock import MagicMock
from pathlib import Path

import src.llm.router as router_module


def test_route_query_smoke(monkeypatch):
    # Mock LLM client so no real Ollama call is made
    fake_client = MagicMock()
    fake_client.chat.return_value = "stubbed answer"
    monkeypatch.setattr(router_module, "OllamaClient", MagicMock())
    router_module.OllamaClient.from_llm_config.return_value = fake_client

    project_root = Path(__file__).resolve().parents[1]
    result = router_module.answer_query(
        user_query="What are the best actions for persona 3?",
        membership_nbr=123456,
        project_root=project_root,
    )

    assert "answer" in result
    assert result["answer"]
    assert result["intent"] in {
        "member_nba",
        "segment_analysis",
        "why_explanation",
        "kg_explore",
        "general_help",
    }
