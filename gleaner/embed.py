"""Embedder protocol, FakeEmbedder, OllamaEmbedder, and vector utility functions."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, runtime_checkable

lazy import httpx

DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors.

    Attributes:
        model_id: The embedding model's name.
        dim: The vector dimension.
    """

    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute normalized vector embeddings for a list of texts.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text, in order.
        """
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        The cosine similarity; 0.0 when either vector is (near) zero.

    Raises:
        ValueError: The vectors have different dimensions.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match ({len(a)} != {len(b)})")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder for offline tests.

    Produces normalized vectors where texts sharing words have high cosine similarity.

    Attributes:
        dim: The vector dimension.
        model_id: The name reported for the model.
    """

    def __init__(self, dim: int = 64, model_id: str = "fake-bow") -> None:
        """Initialize the embedder.

        Args:
            dim: The vector dimension.
            model_id: The name reported for the model.
        """
        self.dim = dim
        self.model_id = model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text as its hashed, signed bag of words.

        A ``search_document:`` or ``search_query:`` prefix is ignored.

        Args:
            texts: The texts to embed.

        Returns:
            One L2-normalized vector per text; all zeros for a text with no words.
        """
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            # Simple word extraction
            cleaned = text.lower()
            # Strip prefixes if present
            cleaned = re.sub(r"^(search_document|search_query):\s*", "", cleaned)
            words = re.findall(r"\b\w+\b", cleaned)
            if not words:
                results.append(vec)
                continue

            for word in words:
                h = hashlib.sha256(word.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if (h[4] % 2 == 0) else -1.0
                vec[idx] += sign

            # L2 normalize
            norm_sq = sum(x * x for x in vec)
            if norm_sq > 1e-12:
                norm = math.sqrt(norm_sq)
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


class OllamaEmbedder:
    """Embeddings from an Ollama server's ``/api/embed`` endpoint.

    The endpoint returns L2-normalized vectors.

    Attributes:
        model_id: The embedding model's name.
        base_url: The server's root URL.
        dim: The expected vector dimension.
        batch_size: How many texts go in one request.
    """

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        dim: int = 768,
        batch_size: int = 32,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the embedder.

        Args:
            model_id: The model; else ``OLLAMA_EMBED_MODEL``, else ``nomic-embed-text``
                (a non-Meta helper model: Llama has no embedding model).
            base_url: The server; defaults to ``OLLAMA_BASE_URL`` (an OpenAI-style ``/v1``
                suffix is dropped).
            dim: Must match the model and the database's ``vec_dim`` (768 for nomic-embed-text).
            batch_size: How many texts go in one request.
            client: The HTTP client to use; one is created when None.
            timeout: The request timeout in seconds for a created client.
        """
        self.model_id = model_id or os.environ.get("OLLAMA_EMBED_MODEL") or DEFAULT_OLLAMA_EMBED_MODEL
        base = base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
        self.base_url = base.rstrip("/").removesuffix("/v1")
        self.dim = dim
        self.batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed the texts in batches.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text, in order.

        Raises:
            httpx.HTTPError: The server fails.
            ValueError: The server returns the wrong number of vectors or the wrong dimension.
        """
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self._client.post(f"{self.base_url}/api/embed", json={"model": self.model_id, "input": batch})
            resp.raise_for_status()
            vectors = resp.json().get("embeddings") or []
            if len(vectors) != len(batch):
                raise ValueError(f"{self.model_id} returned {len(vectors)} embeddings for {len(batch)} texts")
            for vec in vectors:
                if len(vec) != self.dim:
                    raise ValueError(f"{self.model_id} returned dimension {len(vec)}, expected {self.dim}")
            results.extend([float(x) for x in vec] for vec in vectors)
        return results
