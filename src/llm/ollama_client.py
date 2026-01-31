# src/llm/ollama_client.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import requests

from src.common.utils import load_yaml
from src.common.logging import setup_logger


class OllamaClient:
    """
    Thin wrapper around the Ollama HTTP API.

    - Reads base URL + model from configs/env.
    - On any error, returns a readable string instead of raising,
      so FastAPI never crashes because of the LLM call.
    """

    def __init__(self, base_url: str, model: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.logger = setup_logger("ollama_client")

    @classmethod
    def from_llm_config(cls, project_root: Path) -> "OllamaClient":
        cfg = load_yaml(project_root / "configs" / "llm.yaml")

        provider = cfg.get("provider", "ollama")
        if provider != "ollama":
            raise ValueError(f"Unsupported LLM provider: {provider}")

        ollama_cfg = cfg.get("ollama", {})

        base_url_env = ollama_cfg.get("base_url_env", "OLLAMA_BASE_URL")
        model_env = ollama_cfg.get("model_env", "OLLAMA_MODEL")
        timeout = int(ollama_cfg.get("timeout_seconds", 60))

        # When running inside Docker, this should point to the host Ollama.
        base_url = os.getenv(base_url_env, "http://host.docker.internal:11434")
        # Fallback model name if env var is not set
        model = os.getenv(model_env, cfg.get("default_model", "llama3"))

        return cls(base_url=base_url, model=model, timeout=timeout)

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """
        Call Ollama's /api/chat endpoint.

        Returns a plain string. On any error, returns an error message instead
        of raising, so that the API layer can still respond.
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Typical Ollama chat response:
            # {"message": {"role": "assistant", "content": "..."}, ...}
            if isinstance(data, dict):
                msg = data.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    return str(content).strip()

                # Very defensive fallback, in case structure changes
                if "choices" in data:
                    choice0 = data["choices"][0]
                    content = choice0.get("message", {}).get("content", "")
                    return str(content).strip()

            # Last-resort: just string-ify whatever we got
            return str(data)

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error calling Ollama at %s: %s", url, e)
            return f"Error calling Ollama model: {e}"
