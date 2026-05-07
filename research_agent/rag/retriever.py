"""High-level RAG retrieval service."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.rag.index import FaissPaperIndex


@dataclass
class RAGRetriever:
    """Query interface over the paper vector index."""

    index: FaissPaperIndex | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = FaissPaperIndex()

    def retrieve(self, query: str, top_k: int = 5, rerank: bool = False) -> list[dict[str, object]]:
        """Retrieve top-k chunks, with an optional lexical rerank hook."""

        assert self.index is not None
        results = self.index.search(query=query, top_k=top_k)
        if rerank:
            return self._lexical_rerank(query, results)
        return results

    def _lexical_rerank(
        self,
        query: str,
        results: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        query_terms = set(query.lower().split())

        def score(item: dict[str, object]) -> tuple[int, float]:
            text_terms = set(str(item.get("text", "")).lower().split())
            overlap = len(query_terms & text_terms)
            return overlap, float(item.get("score", 0.0))

        return sorted(results, key=score, reverse=True)

