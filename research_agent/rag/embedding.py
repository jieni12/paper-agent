"""Embedding backends for RAG.

The default local-hash embedder is deterministic and dependency-light for
development. Production deployments should use OpenAI embeddings or a local
BGE model exposed through a dedicated service.
"""

from __future__ import annotations

from hashlib import blake2b
import math
from typing import Protocol

from research_agent.config import settings


class Embedder(Protocol):
    """Embedding interface shared by index and retriever."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


class LocalHashEmbedder:
    """Deterministic lexical embedder for local development.

    This is not semantically comparable to bge-large/OpenAI embeddings, but it
    keeps the system runnable without network access or model downloads.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = text.lower().split()
        for token in tokens:
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 1) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbedder:
    """OpenAI-compatible embedding adapter."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.embedding_model
        self.dim = settings.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")

        from openai import OpenAI  # type: ignore

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        response = client.embeddings.create(model=self.model, input=texts)
        vectors = [item.embedding for item in response.data]
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def build_embedder() -> Embedder:
    """Create the configured embedding backend."""

    provider = settings.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder()
    if provider in {"local-hash", "bge-large"}:
        # TODO: Implement native bge-large loading via sentence-transformers when
        # model download/runtime constraints are known for the deployment target.
        return LocalHashEmbedder()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")
