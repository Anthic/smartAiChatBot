from __future__ import annotations

from typing import Iterable

from mistralai.client import Mistral

from app.config import settings


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class MistralEmbeddingClient:
    """Wrapper around the Mistral embedding API."""

    def __init__(self, batch_size: int = 32) -> None:
        self._client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self._model = settings.MISTRAL_EMBED_MODEL
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []

        for batch in self._batched(texts, self._batch_size):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    inputs=batch,
                )
            except Exception as exc:
                raise EmbeddingError("Failed to generate embeddings with Mistral.") from exc

            batch_vectors = [item.embedding for item in response.data]
            self._validate_vectors(batch_vectors)
            vectors.extend(batch_vectors)

        return vectors

    def embed_query(self, query: str) -> list[float]:
        query = query.strip()
        if not query:
            raise EmbeddingError("Query text is empty.")

        return self.embed_texts([query])[0]

    @staticmethod
    def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    @staticmethod
    def _validate_vectors(vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != settings.EMBEDDING_DIM:
                raise EmbeddingError(
                    f"Expected embedding dimension {settings.EMBEDDING_DIM}, "
                    f"got {len(vector)}."
                )


embedding_client = MistralEmbeddingClient()