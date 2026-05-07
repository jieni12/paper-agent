"""Tool wrapper around the RAG retriever."""

from __future__ import annotations

from dataclasses import dataclass
import json

from research_agent.rag.retriever import RAGRetriever
from research_agent.config import settings
from research_agent.tools.base import BaseTool


@dataclass
class RAGRetrieveTool(BaseTool):
    """Retrieve relevant paper chunks from the vector index."""

    retriever: RAGRetriever | None = None
    name: str = "rag_retrieve"
    description: str = "Retrieve top-k paper chunks. Input JSON: query, top_k."

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = RAGRetriever()

    def run(self, input: str) -> str:
        payload = self._parse_input(input)
        query = str(payload.get("query", input))
        top_k = int(payload.get("top_k", settings.retrieval_top_k))
        assert self.retriever is not None
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            rerank=settings.rerank_enabled,
        )
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)

    def _parse_input(self, input: str) -> dict[str, object]:
        try:
            parsed = json.loads(input)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
