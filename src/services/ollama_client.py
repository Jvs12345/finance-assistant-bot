"""Small HTTP client for local Ollama."""

from __future__ import annotations

from typing import Dict, Any, List
import os

import requests
from requests import RequestException

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised for Ollama API errors."""


def _candidate_base_urls() -> List[str]:
    """Return candidate Ollama URLs in fallback order."""
    configured = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    candidates: List[str] = []
    seen = set()

    for part in str(configured).split(","):
        url = part.strip()
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        candidates.append(url)

    for url in [
        "http://host.docker.internal:11434",
        "http://gateway.docker.internal:11434",
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]:
        if url in seen:
            continue
        seen.add(url)
        candidates.append(url)

    return candidates


def chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.7, num_predict: int = 500) -> str:
    """Call Ollama /api/chat and return the response text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict
        }
    }

    last_exc: Exception | None = None
    tried_endpoints: List[str] = []

    for base_url in _candidate_base_urls():
        endpoint = f"{base_url}/api/chat"
        tried_endpoints.append(endpoint)
        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as exc:
                raise OllamaError(f"Failed to parse JSON response from Ollama at {endpoint}: {exc}") from exc

            message = data.get("message") or {}
            content = message.get("content")
            if not content:
                raise OllamaError(f"Ollama chat response at {endpoint} did not contain message content")

            return content.strip()
        except Exception as exc:
            last_exc = exc
            continue

    raise OllamaError(
        "Failed to reach Ollama via all endpoints: "
        + ", ".join(tried_endpoints)
        + f". Last error: {last_exc}"
    )


def list_models() -> List[str]:
    """Return local Ollama model names from /api/tags."""
    last_exc: Exception | None = None
    tried_endpoints: List[str] = []

    for base_url in _candidate_base_urls():
        endpoint = f"{base_url}/api/tags"
        tried_endpoints.append(endpoint)
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            names = []
            for model in models:
                name = model.get("name")
                if name:
                    names.append(name)
            return names
        except Exception as exc:
            last_exc = exc
            continue

    raise OllamaError(
        "Failed to list Ollama models via endpoints: "
        + ", ".join(tried_endpoints)
        + f". Last error: {last_exc}"
    )
