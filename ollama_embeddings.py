from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

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
        endpoint = f"{self.base_url.rstrip('/')}/api/embeddings"
        for text in texts:
            payload = {"model": self.model, "prompt": text}
            try:
                with requests.post(
                    endpoint,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                    stream=True,
                ) as response:
                    if response.status_code != 200:
                        body = response.text
                        raise OllamaEmbeddingError(
                            f"Ollama embedding request returned {response.status_code}: {body}"
                        )

                    embedding: Optional[List[float]] = None
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                        except ValueError:
                            continue

                        if chunk.get("error"):
                            raise OllamaEmbeddingError(
                                f"Ollama embedding request error: {chunk['error']}"
                            )

                        current = chunk.get("embedding")
                        if isinstance(current, list):
                            embedding = current

                        if chunk.get("done", True) and embedding is not None:
                            break

                    if embedding is None:
                        raise OllamaEmbeddingError("Embedding payload missing 'embedding' list")

            except requests.RequestException as exc:
                raise OllamaEmbeddingError(
                    f"Ollama embedding request failed: {exc}"
                ) from exc

            embeddings.append(embedding)
        return embeddings

    def embed_iter(self, texts: Iterable[str]) -> Iterable[List[float]]:
        for text in texts:
            yield self.embed(text)
