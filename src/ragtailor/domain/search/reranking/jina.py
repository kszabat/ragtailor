from __future__ import annotations

from ragtailor.domain.models import SearchResult, SourceType

from ragtailor.domain.search.reranking.base import Reranker


class JinaRerankerM0(Reranker):
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-m0",
        max_length: int = 2048,
        device: str | None = None,
    ) -> None:
        from transformers import AutoModel

        self._model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            .to(device)
            .eval()
        )
        self._max_length = max_length

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []

        source_types = {result.source_type for result in results}

        assert len(source_types) == 1, "All results must have the same source type"

        doc_type = "image" if results[0].source_type == SourceType.VISUAL else "text"
        query_type = "text"

        pairs = [(query, _document_for(result)) for result in results]

        scores = self._model.compute_score(
            pairs, query_type=query_type, doc_type=doc_type, max_length=self._max_length
        )

        reranked_results = [
            result.model_copy(update={"rerank_score": float(score)})
            for result, score in zip(results, scores)
        ]

        reranked_results.sort(key=lambda result: result.reranker_score, reverse=True)

        return reranked_results


def _document_for(result: SearchResult) -> str:
    if result.source_type == SourceType.VISUAL:
        assert result.image_path is not None, "Visual result must have an image path"

        return result.image_path

    assert result.text is not None, "Text result must have text"

    return result.text
