from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import requests


class OllamaEmbeddingError(RuntimeError):
    """Raised when an Ollama embedding request fails."""


@dataclass
class OllamaEmbedder:
    """Lightweight client for fetching embeddings from a local Ollama instance."""

    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0

    def embed(self, text: str) -> List[float]:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            payload = {"model": self.model, "prompt": text}
            try:
                response = requests.post(
                    f"{self.base_url.rstrip('/')}/api/embeddings",
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise OllamaEmbeddingError(
                    f"Ollama embedding request failed: {exc}"
                ) from exc

            if response.status_code != 200:
                raise OllamaEmbeddingError(
                    f"Ollama embedding request returned {response.status_code}: {response.text}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise OllamaEmbeddingError("Invalid JSON from Ollama embeddings API") from exc

            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise OllamaEmbeddingError("Embedding payload missing 'embedding' list")

            embeddings.append(embedding)
        return embeddings

    def embed_iter(self, texts: Iterable[str]) -> Iterable[List[float]]:
        for text in texts:
            yield self.embed(text)

