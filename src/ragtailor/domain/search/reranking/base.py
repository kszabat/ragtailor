from __future__ import annotations

from typing import Protocol

from ragtailor.domain.models import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]: ...