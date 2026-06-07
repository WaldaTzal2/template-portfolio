"""Cache em 2 niveis: exact-match (SHA256) + semantic (cosine similarity).

Reaproveita o notebook 05. Voce vai preencher 1 TODO aqui.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from openai import OpenAI


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura parafrases (~20% adicional)."""

    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []

        # Inicializa cliente para embeddings (mesmo provider do RAG)
        if "GEMINI_API_KEY" in os.environ:
            self._client = OpenAI(
                api_key=os.environ["AQ.Ab8RN6IojPcLq5F6_BgzMaOR_NbZJVLFozc6WkvNA6CMGNHyng"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._embed_model = os.environ.get(
                "EMBED_MODEL", "gemini-embedding-001")
        else:
            self._client = OpenAI()
            self._embed_model = "text-embedding-3-small"

    def _embed(self, text: str) -> np.ndarray:
        r = self._client.embeddings.create(model=self._embed_model, input=text)
        return np.array(r.data[0].embedding)

    # ------------------------------------------------------------------ TODO 5
    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar a query alguma anterior, OU None."""
        if not self._queries:
            return None

        # 1. Obter o embedding da query atual
        query_vector = self._embed(query)

        # 2. Calcular a similaridade de cosseno com cada embedding na memória
        for idx, cached_vector in enumerate(self._embeddings):
            dot_product = np.dot(query_vector, cached_vector)
            norm_q = np.linalg.norm(query_vector)
            norm_c = np.linalg.norm(cached_vector)

            if norm_q == 0 or norm_c == 0:
                continue

            cosine_similarity = dot_product / (norm_q * norm_c)

            # 3. Se passar do limite configurado (threshold), retorna a resposta
            if cosine_similarity >= self.threshold:
                return self._answers[idx]

        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        try:
            self._embeddings.append(self._embed(query))
            self._answers.append(answer)
        except Exception:
            # Fallback se falhar a rede do embedding
            self._queries.pop()

    def stats(self) -> dict[str, int]:
        return {"size": len(self._queries)}
