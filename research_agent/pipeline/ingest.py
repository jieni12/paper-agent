"""Paper ingestion pipeline.

Flow: arXiv API -> PDF download -> PDF parsing -> structured extraction ->
FAISS indexing. Each stage is composed from independently testable tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import urllib.request

from research_agent.config import settings
from research_agent.memory.memory import ResearchMemory
from research_agent.rag.index import FaissPaperIndex, chunks_from_structured_paper
from research_agent.tools.paper_parse import PaperParseTool
from research_agent.tools.paper_search import PaperSearchTool
from research_agent.tools.summarizer import SummarizerTool


@dataclass
class IngestionResult:
    """Summary returned by the ingestion pipeline."""

    query: str
    fetched: int
    ingested: int
    skipped_duplicates: int
    papers: list[dict[str, str]]


class PaperIngestionPipeline:
    """End-to-end paper ingestion service."""

    def __init__(
        self,
        search_tool: PaperSearchTool | None = None,
        parse_tool: PaperParseTool | None = None,
        summarizer: SummarizerTool | None = None,
        index: FaissPaperIndex | None = None,
        memory: ResearchMemory | None = None,
    ) -> None:
        self.search_tool = search_tool or PaperSearchTool()
        self.parse_tool = parse_tool or PaperParseTool()
        self.summarizer = summarizer or SummarizerTool()
        self.index = index or FaissPaperIndex()
        self.memory = memory or ResearchMemory().load()

    def run(self, query: str, max_results: int | None = None) -> IngestionResult:
        """Fetch latest papers matching the query and index parsed content."""

        max_results = max_results or settings.arxiv_max_results
        search_payload = json.dumps({"query": query, "max_results": max_results})
        search_result = json.loads(self.search_tool.run(search_payload))
        papers = search_result.get("papers", [])
        if not isinstance(papers, list):
            papers = []

        unread = self.memory.filter_unread(papers)
        ingested = 0
        stored_papers: list[dict[str, str]] = []

        settings.paper_dir.mkdir(parents=True, exist_ok=True)
        settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        for paper in unread:
            if not isinstance(paper, dict):
                continue
            pdf_url = str(paper.get("pdf_url", ""))
            if not pdf_url:
                continue

            pdf_path = self._download_pdf(pdf_url, str(paper.get("id", "")))
            parsed = json.loads(self.parse_tool.run(json.dumps({"pdf_path": str(pdf_path)})))
            summary = json.loads(
                self.summarizer.run(
                    json.dumps(
                        {
                            "title": paper.get("title", parsed.get("title", "")),
                            "text": parsed.get("text", ""),
                            "sections": parsed.get("sections", {}),
                        },
                        ensure_ascii=False,
                    )
                )
            )
            structured = {
                **paper,
                **summary,
                "paper_id": str(paper.get("id", "")),
                "source_url": pdf_url,
                "sections": parsed.get("sections", {}),
            }
            chunks = chunks_from_structured_paper(structured)
            self.index.add_chunks(chunks)
            self._append_metadata(structured)
            self.memory.mark_read(str(paper.get("id", "")), {"title": paper.get("title", ""), "pdf_url": pdf_url})
            stored_papers.append(
                {
                    "id": str(paper.get("id", "")),
                    "title": str(paper.get("title", "")),
                    "pdf_path": str(pdf_path),
                }
            )
            ingested += 1

        return IngestionResult(
            query=query,
            fetched=len(papers),
            ingested=ingested,
            skipped_duplicates=len(papers) - len(unread),
            papers=stored_papers,
        )

    def _download_pdf(self, pdf_url: str, paper_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", paper_id or pdf_url.rsplit("/", 1)[-1])
        if not safe_id.endswith(".pdf"):
            safe_id += ".pdf"
        path = settings.paper_dir / safe_id
        if path.exists():
            return path

        request = urllib.request.Request(pdf_url, headers={"User-Agent": settings.user_agent})
        with urllib.request.urlopen(request, timeout=settings.request_timeout_seconds) as response:
            content = response.read(settings.max_pdf_mb * 1024 * 1024 + 1)
            if len(content) > settings.max_pdf_mb * 1024 * 1024:
                raise RuntimeError(f"PDF exceeds MAX_PDF_MB={settings.max_pdf_mb}: {pdf_url}")
            path.write_bytes(content)
        return path

    def _append_metadata(self, paper: dict[str, object]) -> None:
        with settings.metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
